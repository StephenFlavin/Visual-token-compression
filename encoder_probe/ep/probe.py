"""Probe: frozen 14x14 token grid -> per-line CTC reader that works at token resolution.

The same probe is trained for every (encoding, encoder, condition). It can only lose information,
never add it, so the CER it reaches is an upper bound on what the frozen tokens destroy; the
identity encoder (raw pixel patches) gives each encoding's ceiling under the same probe.

For text line i (rows [i*p, (i+1)*p)), let r0 = top row // 16 (patch row), dy = top row % 16:
  x_t = [token(r0, t), token(r0+1, t)]          for the 14 patch columns t (r0+1 zeroed past the edge)
   -> Linear_dy (2*768 -> hid)                   one input projection per in-patch offset dy
   -> BN, GELU, dilated residual Conv1d stack over the 14 positions
   -> Conv1x1 (hid -> 16*E*(n_classes+1))       per-token head: 16 columns x E frames each
   -> 224*E frames, CTC over the line's labels.

(v1 unpatchified tokens to a per-pixel feature map and read pixel strips; it could not learn even on
raw pixels -- the random token->pixel mixing gives CTC no locality to bootstrap from. See PLAN.md.)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein

GRID, PATCH, W = 14, 16, 224


def unpatchify(f: torch.Tensor, c: int) -> torch.Tensor:
    """(B, c*16*16, 14, 14) with channel order (c, dy, dx) -> (B, c, 224, 224)."""
    B = f.shape[0]
    f = f.reshape(B, c, PATCH, PATCH, GRID, GRID)
    return f.permute(0, 1, 4, 2, 5, 3).reshape(B, c, GRID * PATCH, GRID * PATCH)


class ResBlock(nn.Module):
    def __init__(self, w: int, dil: int):
        super().__init__()
        self.c1 = nn.Conv1d(w, w, 3, padding=dil, dilation=dil)
        self.b1 = nn.BatchNorm1d(w)
        self.c2 = nn.Conv1d(w, w, 3, padding=dil, dilation=dil)
        self.b2 = nn.BatchNorm1d(w)

    def forward(self, x):
        return x + self.b2(self.c2(F.gelu(self.b1(self.c1(x)))))


class Probe(nn.Module):
    def __init__(self, pitch: int, n_classes: int, d: int = 768, hid: int = 512, expand: int = 3,
                 dils=(1, 2, 4, 1)):
        super().__init__()
        self.pitch, self.d, self.expand, self.n_out = pitch, d, expand, n_classes + 1
        self.w_in = nn.Parameter(torch.randn(PATCH, 2 * d, hid) / math.sqrt(2 * d))
        self.b_in = nn.Parameter(torch.zeros(PATCH, hid))
        self.bn_in = nn.BatchNorm1d(hid)
        self.blocks = nn.Sequential(*[ResBlock(hid, k) for k in dils])
        self.out = nn.Conv1d(hid, PATCH * expand * self.n_out, 1)

    def features(self, tok: torch.Tensor) -> torch.Tensor:
        """(B, 196, d) -> (B, 14, 14, d) token grid."""
        return tok.reshape(tok.shape[0], GRID, GRID, -1)

    def read(self, feat: torch.Tensor, page_idx: torch.Tensor, line_idx: torch.Tensor) -> torch.Tensor:
        """feat (B,14,14,d); (page, line) index pairs -> log-probs (T=224*E, N, n_out) for CTC."""
        top = line_idx * self.pitch
        r0, dy = top // PATCH, top % PATCH
        r1 = (r0 + 1).clamp(max=GRID - 1)
        nxt = feat[page_idx, r1] * (r0 + 1 < GRID).to(feat.dtype)[:, None, None]
        x = torch.cat([feat[page_idx, r0], nxt], -1)                               # (N, 14, 2d)
        h = x.new_zeros(len(line_idx), GRID, self.w_in.shape[2])
        for v in torch.unique(dy).tolist():
            m = dy == v
            h[m] = x[m] @ self.w_in[v].to(x.dtype) + self.b_in[v].to(x.dtype)
        h = F.gelu(self.bn_in(h.transpose(1, 2)))
        y = self.out(self.blocks(h))                                               # (N, 16*E*n_out, 14)
        N = y.shape[0]
        y = y.reshape(N, PATCH, self.expand, self.n_out, GRID).permute(4, 1, 2, 0, 3)
        return F.log_softmax(y.reshape(GRID * PATCH * self.expand, N, self.n_out), -1)


def greedy(logp: torch.Tensor) -> list[list[int]]:
    best = logp.argmax(-1).T.tolist()                                              # (N, T)
    out = []
    for seq in best:
        res, prev = [], 0
        for s in seq:
            if s != prev and s != 0:
                res.append(s)
            prev = s
        out.append(res)
    return out


def edit_distance(a: str, b: str) -> int:
    return Levenshtein.distance(a, b)
