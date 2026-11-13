"""Small CRNN line reader: height-agnostic CNN -> BiLSTM -> frame expansion -> linear (CTC).

Input-resolution policy
-----------------------
Images are fed at their *native* rendered resolution (H = round(1.2*em) + 4 px, never resized).
The CNN is height-agnostic: a full-resolution 3x3 conv (16 channels, so the following 2x2 max-pool
can keep every pixel's information in its channels), 2x2 max-pool, a 3x3 conv, an adaptive
max-pool capping the height at 6 rows (no-op for em <= 7; bounds CPU cost at large ems), a width
max-pool if col_stride = 4 (unused by default), a 3x3 conv, then
``AdaptiveAvgPool2d((4, None))`` + a learned projection collapse the height to one feature vector
per column (keeping coarse vertical position, which matters for packed quadrant cells).
We do NOT resize to a fixed model height: downscaling would destroy the resolution effect we
measure, and upscaling small ems to a common height multiplies CPU cost without adding
information. Architecture and parameter count are identical for every scheme and em (except the
vocabulary-sized output layer and the frame-expansion layer, set by the rule below).

CTC time axis
-------------
The BiLSTM runs over W // s columns, s = ``data.col_stride_for_em(em)`` = 2 px for every scheme
and em (s = 4 is supported - an extra width pool - but learned more slowly in the smoke test).
The LSTM is the sequential, CPU-expensive part, so instead of making it longer we expand each
LSTM output into E CTC frames with a learned linear map *after* the LSTM (T = (W // s) * E).
E (``data.choose_expand``; one rule for every scheme) is the smallest integer giving T >= 2L+1
for every sample, so dense multi-label cells at tiny ems get enough CTC slots.
"""
from __future__ import annotations

import torch
from torch import nn


def _conv(cin: int, cout: int, norm: str = "bn") -> nn.Sequential:
    nl = nn.BatchNorm2d(cout) if norm == "bn" else nn.GroupNorm(8, cout)
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nl, nn.ReLU(inplace=True))


class _HeightCap(nn.Module):
    """Adaptive max-pool of the height down to ``cap`` rows (no-op if already <= cap). Bounds the
    CNN cost at large ems; small ems (where CER thresholds are crossed) are never pooled here."""

    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] <= self.cap:
            return x
        return nn.functional.adaptive_max_pool2d(x, (self.cap, x.shape[3]))


class CRNN(nn.Module):
    def __init__(self, n_classes: int, col_stride: int = 2, expand: int = 1, channels=(16, 64, 96),
                 mid_height: int = 6, pooled_height: int = 4, d_model: int = 192, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.1, norm: str = "bn"):
        super().__init__()
        assert col_stride in (2, 4), col_stride
        c1, c2, c3 = channels
        self.col_stride, self.expand = col_stride, int(expand)
        self.pw2 = col_stride // 2
        self.cnn = nn.Sequential(
            _conv(1, c1, norm), nn.MaxPool2d(2, 2), _conv(c1, c2, norm),
            _HeightCap(mid_height),
            nn.MaxPool2d((1, 2), (1, 2)) if self.pw2 > 1 else nn.Identity(),
            _conv(c2, c3, norm),
            nn.AdaptiveAvgPool2d((pooled_height, None)),
        )
        self.proj = nn.Sequential(nn.Linear(c3 * pooled_height, d_model), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))
        self.rnn = nn.LSTM(d_model, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        h2 = 2 * hidden
        # always present (E=1 too) so every run has the same depth
        self.expander = nn.Sequential(nn.Linear(h2, h2 * self.expand), nn.ReLU(inplace=True))
        self.head = nn.Linear(h2, n_classes)

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        return (widths // self.col_stride) * self.expand

    def forward(self, x: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B,1,H,W) -> (log_probs (T,B,C), out_lens (B,)).

        Sequences are not packed (packed LSTMs are ~7x slower on CPU); batches are bucketed by
        width so padding is small, and padded columns are zeroed before the LSTM.
        """
        f = self.cnn(x)                                   # B, C, h, T
        B, C, h, T = f.shape
        f = self.proj(f.permute(0, 3, 1, 2).reshape(B, T, C * h))
        cols = (widths // self.col_stride).clamp(min=1, max=T)
        mask = torch.arange(T)[None, :] < cols[:, None]
        f = f * mask[:, :, None]
        out, _ = self.rnn(f)
        out = self.expander(out).reshape(B, T * self.expand, -1)
        lens = cols * self.expand
        return self.head(out).log_softmax(-1).transpose(0, 1), lens


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def greedy_decode(log_probs: torch.Tensor, lens: torch.Tensor, blank: int = 0) -> list[list[int]]:
    """Best-path CTC decoding: argmax, collapse repeats, drop blanks."""
    best = log_probs.argmax(-1).transpose(0, 1).cpu().numpy()   # B, T
    out = []
    for row, n in zip(best, lens.tolist()):
        seq, prev = [], -1
        for t in row[:n]:
            if t != prev and t != blank:
                seq.append(int(t))
            prev = t
        out.append(seq)
    return out
