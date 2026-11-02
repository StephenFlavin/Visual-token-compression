"""Compose glyph cells into line canvases at supersampled resolution (WP-LG1).

Cells are placed left to right with no overlap: symbol j of a line starts at
``margin_ss + sum(widths_ss[ids[:j]])``. Fully vectorised: one ``repeat_interleave`` builds the
(source column -> canvas column) map for every column of every placed cell, one advanced-index
gather reads those ink columns and one ``index_copy`` writes them (differentiable w.r.t. ``ink``).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def line_widths_ss(ids_batch: Sequence[Sequence[int]], widths_ss: torch.Tensor) -> torch.Tensor:
    """Content width (ss px) of each line = sum of its cells' widths."""
    widths_ss = torch.as_tensor(widths_ss, dtype=torch.long)
    return torch.tensor([int(widths_ss[list(ids)].sum()) if len(ids) else 0 for ids in ids_batch],
                        dtype=torch.long)


def compose(ink: torch.Tensor, ids_batch: Sequence[Sequence[int]], widths_ss: torch.Tensor,
            margin_ss: int = 0, min_width_ss: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """ink (n, Hs, Wmax) in [0,1] -> (canvas (B,1,Hs + 2*margin_ss, Wtot) , line widths_ss (B,)).

    ``widths_ss`` = per-symbol cell widths (n,). Returned widths include ``2*margin_ss`` (left +
    right); ``Wtot`` = max returned width (>= ``min_width_ss``). With the Channel's own margin
    (default 2 px) use ``margin_ss=0`` here (the default) - the Channel adds the Phase-2 margin.
    """
    device = ink.device
    n, Hs, Wm = ink.shape
    widths_ss = torch.as_tensor(widths_ss, dtype=torch.long, device=device)
    B = len(ids_batch)
    lens = torch.tensor([len(x) for x in ids_batch], dtype=torch.long, device=device)
    flat = torch.tensor([i for x in ids_batch for i in x], dtype=torch.long, device=device)
    line_of = torch.repeat_interleave(torch.arange(B, device=device), lens)
    w = widths_ss[flat]                                            # (P,)
    # x offset of each placed cell within its line: exclusive cumsum restarted per line
    col_start = torch.cumsum(w, 0) - w                             # exclusive cumsum over all cells
    first = torch.cumsum(lens, 0) - lens                           # index of each line's 1st cell
    line_base = torch.cat([col_start, col_start.new_zeros(1)])[first]   # (any value if empty line)
    off = col_start - line_base[line_of]
    content = torch.zeros(B, dtype=torch.long, device=device).index_add(0, line_of, w)
    out_w = content + 2 * margin_ss
    Wtot = max(int(out_w.max()) if B else 0, int(min_width_ss))
    canvas = ink.new_zeros(B * Wtot, Hs)
    if flat.numel():
        pair = torch.repeat_interleave(torch.arange(flat.numel(), device=device), w)  # (N,)
        c = torch.arange(pair.numel(), device=device) - col_start[pair]               # col in cell
        tgt = line_of[pair] * Wtot + margin_ss + off[pair] + c
        vals = ink.permute(0, 2, 1)[flat[pair], c]                                   # (N, Hs)
        canvas = canvas.index_copy(0, tgt, vals)
    canvas = canvas.reshape(B, Wtot, Hs).permute(0, 2, 1).unsqueeze(1)
    if margin_ss:
        canvas = F.pad(canvas, (0, 0, margin_ss, margin_ss))
    return canvas, out_w
