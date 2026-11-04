"""Round-2 probe: the final round-1 probe (ep/probe.py, 3 channels, no free channels) on a G x G grid of P-px tokens.

  tokens (B, G*G, d), standardised per dim
   -> [Conv3x3 (d -> 1024) -> GELU -> Conv1x1 (1024 -> P*P*3)] + Conv1x1 skip (d -> P*P*3)   [unpatchify]
   -> (B, 3, S, S) reconstructed page, S = G*P (aux MSE x10 to the clean page, training pages only)
   -> line strip i = rows [i*p, (i+1)*p) -> (3p, S) -> the round-1 dilated residual Conv1d reader
   -> 3 frames per column (T = 3S) -> CTC.
With G = 14, P = 16 this is exactly round 1's Probe (same modules and parameter shapes; tested).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder_probe.ep.probe import ResBlock


def unpatchify_g(f: torch.Tensor, c: int, P: int, G: int) -> torch.Tensor:
    """(B, c*P*P, G, G) with channel order (c, dy, dx) -> (B, c, G*P, G*P)."""
    B = f.shape[0]
    return f.reshape(B, c, P, P, G, G).permute(0, 1, 4, 2, 5, 3).reshape(B, c, G * P, G * P)


class GridProbe(nn.Module):
    def __init__(self, pitch: int, n_classes: int, d: int, G: int, P: int, hid: int = 1024, w: int = 128,
                 expand: int = 3, dils=(1, 2, 4, 8)):
        super().__init__()
        self.pitch, self.c, self.expand, self.n_out = pitch, 3, expand, n_classes + 1
        self.G, self.P, self.S = G, P, G * P
        self.grid = nn.Sequential(nn.Conv2d(d, hid, 3, padding=1), nn.GELU(), nn.Conv2d(hid, P * P * 3, 1))
        self.skip = nn.Conv2d(d, P * P * 3, 1)
        self.inp = nn.Sequential(nn.Conv1d(3 * pitch, w, 3, padding=1), nn.BatchNorm1d(w), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(w, k) for k in dils])
        self.out = nn.Conv1d(w, expand * self.n_out, 1)

    def features(self, tok: torch.Tensor) -> torch.Tensor:
        """(B, G*G, d) -> (B, 3, S, S): the reconstructed page."""
        B = tok.shape[0]
        g = tok.transpose(1, 2).reshape(B, -1, self.G, self.G)
        return unpatchify_g(self.grid(g) + self.skip(g), 3, self.P, self.G)

    @staticmethod
    def recon_loss(feat: torch.Tensor, page: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(feat.float(), page)

    def read(self, feat: torch.Tensor, page_idx: torch.Tensor, line_idx: torch.Tensor) -> torch.Tensor:
        """feat (B,3,S,S); (page, line) pairs -> log-probs (T = S*E, N, n_out)."""
        p, S = self.pitch, self.S
        rows = line_idx[:, None] * p + torch.arange(p, device=line_idx.device)[None]
        strips = feat[page_idx[:, None], :, rows]                                 # (N, p, 3, S)
        x = strips.permute(0, 2, 1, 3).reshape(len(line_idx), 3 * p, S)
        y = self.out(self.blocks(self.inp(x)))
        N = y.shape[0]
        y = y.reshape(N, self.expand, self.n_out, S).permute(3, 1, 0, 2).reshape(S * self.expand, N, self.n_out)
        return F.log_softmax(y, -1)
