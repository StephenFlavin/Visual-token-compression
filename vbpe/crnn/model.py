"""Small CRNN line reader: height-agnostic CNN -> BiLSTM -> linear, trained with CTC.

Input-resolution policy
-----------------------
Images are fed at their *native* rendered resolution (H = round(1.2*em) + 4 px, never resized).
The CNN is height-agnostic: 3x3 convs, one height-halving max-pool, then
``AdaptiveAvgPool2d((4, None))`` + a learned projection collapse the height to a single feature
vector per column (keeping coarse vertical position, which matters for packed quadrant cells).
We do NOT resize to a fixed model height: downscaling would destroy the resolution effect we
measure, and upscaling small ems to a common height multiplies CPU cost for no information gain
(the convs see every native pixel either way). Same architecture (and parameter count, up to the
output layer whose size is the vocabulary) for every scheme and em.

CTC time axis
-------------
No width pooling by default; the per-run *frame factor* ``f`` (``data.choose_frame_factor``,
same rule for every scheme: smallest f with T >= 2L+1 for every sample) sets frames per input
column: f = 1/2, 1/4 -> width max-pooling (cheap, for large ems); f = 2, 3, 4 -> each CNN column
is expanded into f frames by a learned linear map (dense multi-label cells at tiny ems).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, n_classes: int, frame_factor: float = 1, channels=(32, 64, 96, 96),
                 pooled_height: int = 4, d_model: int = 192, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.frame_factor = frame_factor
        c1, c2, c3, c4 = channels
        self.pw1 = 2 if frame_factor <= 0.5 else 1
        self.pw2 = 2 if frame_factor <= 0.25 else 1
        self.expand = int(frame_factor) if frame_factor > 1 else 1
        self.cnn = nn.Sequential(
            _conv(1, c1), _conv(c1, c2), nn.MaxPool2d((2, self.pw1), (2, self.pw1)),
            _conv(c2, c3), _conv(c3, c4),
            nn.MaxPool2d((1, self.pw2), (1, self.pw2)) if self.pw2 > 1 else nn.Identity(),
            nn.AdaptiveAvgPool2d((pooled_height, None)),
        )
        self.proj = nn.Sequential(nn.Linear(c4 * pooled_height, d_model), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))
        self.expander = nn.Linear(d_model, d_model * self.expand) if self.expand > 1 else None
        self.rnn = nn.LSTM(d_model, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, n_classes)

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        w = widths // self.pw1 // self.pw2
        return w * self.expand

    def forward(self, x: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B,1,H,W) -> (log_probs (T,B,C), out_lens (B,))."""
        f = self.cnn(x)                                   # B, C, h, T
        B, C, h, T = f.shape
        f = self.proj(f.permute(0, 3, 1, 2).reshape(B, T, C * h))
        if self.expander is not None:
            f = self.expander(f).reshape(B, T * self.expand, -1)
        lens = self.output_lengths(widths).clamp(min=1, max=f.shape[1])
        packed = pack_padded_sequence(f, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=f.shape[1])
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
