"""Small CRNN line reader: height-agnostic CNN -> BiLSTM -> frame expansion -> linear (CTC).

Input-resolution policy
-----------------------
Images are fed at their *native* rendered resolution (H = round(1.2*em) + 4 px, never resized).
The CNN is height-agnostic: a full-resolution 3x3 conv (32 channels, so the following 2x2 max-pool
can keep every pixel's information in its channels), 2x2 max-pool, two more 3x3 convs, then
``AdaptiveAvgPool2d((4, None))`` + a learned projection collapse the height to one feature vector
per column (keeping coarse vertical position, which matters for packed quadrant cells).
We do NOT resize to a fixed model height: downscaling would destroy the resolution effect we
measure, and upscaling small ems to a common height multiplies CPU cost without adding
information. Architecture and parameter count are identical for every scheme and em (except the
vocabulary-sized output layer and the frame-expansion layer, set by the rule below).

CTC time axis
-------------
The BiLSTM runs over W/2 columns (W/4 when f = 1/4). The per-run *frame factor* f
(``data.choose_frame_factor``; one rule for every scheme: smallest f in {1/4,1/2,1,2,3,4} with
T >= 2L+1 for every sample) sets the number of CTC frames T ~= f*W: for f >= 1 each LSTM output
is expanded into E = 2f frames by a learned linear map *after* the LSTM, so dense multi-label
cells at tiny ems get enough CTC slots without making the (sequential, CPU-expensive) LSTM longer.
"""
from __future__ import annotations

import torch
from torch import nn


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, n_classes: int, frame_factor: float = 1, channels=(32, 64, 96),
                 pooled_height: int = 4, d_model: int = 192, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.frame_factor = frame_factor
        c1, c2, c3 = channels
        self.pw2 = 2 if frame_factor <= 0.25 else 1
        self.expand = int(round(2 * frame_factor)) if frame_factor >= 1 else 1
        self.cnn = nn.Sequential(
            _conv(1, c1), nn.MaxPool2d(2, 2),
            _conv(c1, c2), _conv(c2, c3),
            nn.MaxPool2d((1, 2), (1, 2)) if self.pw2 > 1 else nn.Identity(),
            nn.AdaptiveAvgPool2d((pooled_height, None)),
        )
        self.proj = nn.Sequential(nn.Linear(c3 * pooled_height, d_model), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))
        self.rnn = nn.LSTM(d_model, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        h2 = 2 * hidden
        self.expander = (nn.Sequential(nn.Linear(h2, h2 * self.expand), nn.ReLU(inplace=True))
                         if self.expand > 1 else None)
        self.head = nn.Linear(h2, n_classes)

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        return (widths // 2 // self.pw2) * self.expand

    def forward(self, x: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B,1,H,W) -> (log_probs (T,B,C), out_lens (B,)).

        Sequences are not packed (packed LSTMs are ~7x slower on CPU); batches are bucketed by
        width so padding is small, and padded columns are zeroed before the LSTM.
        """
        f = self.cnn(x)                                   # B, C, h, T
        B, C, h, T = f.shape
        f = self.proj(f.permute(0, 3, 1, 2).reshape(B, T, C * h))
        cols = (widths // 2 // self.pw2).clamp(min=1, max=T)
        mask = torch.arange(T)[None, :] < cols[:, None]
        f = f * mask[:, :, None]
        out, _ = self.rnn(f)
        if self.expander is not None:
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
