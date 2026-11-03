"""Probe: frozen token grid -> per-pixel feature map -> per-line dilated-conv CTC reader.

The same probe is trained for every (encoding, encoder, condition). It can only lose information,
never add it, so the CER it reaches is an upper bound on what the frozen tokens destroy; the
identity encoder (raw pixel patches) gives each encoding's ceiling under the same probe.

  tokens (B,196,768), standardised per dim
   -> [Conv3x3 (768->1024) -> GELU -> Conv1x1 (1024 -> 16*16*c)] + Conv1x1 skip   [unpatchify, c=3]
   -> feature map (B, 3, 224, 224) = a reconstructed page: auxiliary MSE loss (x10) to the clean page
      (training pages only). Optional extra "free" channels (c > 3) enter through a gate initialised at 0;
      the grid uses none (with 5 free channels, identity/latin_p10 memorised its training pages: 77% test CER).
   -> line strip i = rows [i*p, (i+1)*p) -> (c*p, 224) -> dilated residual Conv1d stack (shared over columns)
   -> E frames per column -> CTC over the line's labels.

The auxiliary pixel loss is what makes this trainable: without it CTC alone never bootstraps the
token->pixel map (pilot: unigram-level CER after 1500 steps even on identity tokens), and a probe that
reads tokens directly with per-position heads learns but memorises the 200 training pages (latin_p8
identity: train loss 0.006, test CER 13.7%). See PLAN.md. The loss only uses training pages, so the
probe still cannot put information into test tokens that the encoder removed.
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
    def __init__(self, pitch: int, n_classes: int, d: int = 768, c: int = 3, hid: int = 1024,
                 w: int = 128, expand: int = 3, dils=(1, 2, 4, 8)):
        super().__init__()
        self.pitch, self.c, self.expand, self.n_out = pitch, c, expand, n_classes + 1
        self.grid = nn.Sequential(nn.Conv2d(d, hid, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(hid, PATCH * PATCH * c, 1))
        self.skip = nn.Conv2d(d, PATCH * PATCH * c, 1)
        self.gate = nn.Parameter(torch.zeros(c - 3))   # free channels 3.. start closed (see features)
        self.inp = nn.Sequential(nn.Conv1d(c * pitch, w, 3, padding=1), nn.BatchNorm1d(w), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(w, k) for k in dils])
        self.out = nn.Conv1d(w, expand * self.n_out, 1)

    def features(self, tok: torch.Tensor) -> torch.Tensor:
        """(B, 196, d) tokens -> (B, c, 224, 224) feature map; channels 0-2 are trained to be the page."""
        B = tok.shape[0]
        g = tok.transpose(1, 2).reshape(B, -1, GRID, GRID)
        f = unpatchify(self.grid(g) + self.skip(g), self.c)
        # channels 0-2 are the pixel reconstruction; the free channels enter through a learned gate that
        # starts at 0 (open, their untrained noise swamped the reconstruction and stalled the reader)
        return torch.cat([f[:, :3], f[:, 3:] * self.gate.to(f.dtype)[None, :, None, None]], 1)

    @staticmethod
    def recon_loss(feat: torch.Tensor, page: torch.Tensor) -> torch.Tensor:
        """MSE between feature channels 0-2 and the clean page (B, 3, 224, 224) scaled to [-1, 1]."""
        return F.mse_loss(feat[:, :3].float(), page)

    def read(self, feat: torch.Tensor, page_idx: torch.Tensor, line_idx: torch.Tensor) -> torch.Tensor:
        """feat (B,c,224,224); (page, line) index pairs -> log-probs (T=224*E, N, n_out) for CTC."""
        p = self.pitch
        rows = line_idx[:, None] * p + torch.arange(p)[None]                      # (N, p)
        strips = feat[page_idx[:, None], :, rows]                                 # (N, p, c, 224)
        x = strips.permute(0, 2, 1, 3).reshape(len(line_idx), self.c * p, W)
        y = self.out(self.blocks(self.inp(x)))                                    # (N, E*n_out, 224)
        N = y.shape[0]
        y = y.reshape(N, self.expand, self.n_out, W).permute(3, 1, 0, 2).reshape(W * self.expand, N, self.n_out)
        return F.log_softmax(y, -1)


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
