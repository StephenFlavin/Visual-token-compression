"""Render a flow (see minify/base.py) as pages with the Phase 1 / Claude QA Latin renderer.

Pages are W px wide (1932: the largest side Opus/Sonnet/Haiku 5.5 read without downscaling) and at most W tall; the
last page is cut to the height its text needs. ``budget`` is px² per character: 117.6 (2.2x fewer tokens than text
for prose; lossless for Opus and Sonnet at high effort), 88.2 (2.9x; lossless for Opus at high effort, the default
here), 58.8 (4.2x; loses multi-step accuracy).
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_BUDGET = float(os.environ.get("VBHOOK_BUDGET", "88.2"))
W = 1932
PATCH = 28


def vision_tokens(w: int, h: int) -> int:
    return math.ceil(w / PATCH) * math.ceil(h / PATCH)


def chunk(flow: str, max_chars: int) -> list[str]:
    out, i = [], 0
    while i < len(flow):
        j = min(len(flow), i + max_chars)
        if j < len(flow):
            k = flow.rfind(" ", i, j)
            j = k + 1 if k > i else j
        out.append(flow[i:j].strip())
        i = j
    return [c for c in out if c]


@dataclass
class Page:
    image: object
    first_line: int | None      # first §-tagged line number on the page
    last_line: int | None
    chars: int

    @property
    def tokens(self) -> int:
        return vision_tokens(*self.image.size)


def _height(n_chars: int, budget: float) -> int:
    return min(W, max(2 * PATCH, math.ceil(n_chars * budget / W / PATCH) * PATCH + PATCH))


def render(flow: str, budget: float = DEFAULT_BUDGET) -> list[Page]:
    from vbpe import metrics
    from vbpe.layout import render_page
    from vbpe.schemes import get_scheme
    sch = get_scheme("latin")
    max_chars = int(W * W / budget * 0.97)
    while True:
        pages, ok = [], True
        for c in chunk(flow, max_chars):
            n = len(metrics.normalize(c))
            img, inf = render_page(sch.encode(c), W, _height(n, budget))
            if not inf["fit"]:
                img, inf = render_page(sch.encode(c), W, min(W, _height(n, budget) + 2 * PATCH))
            if not inf["fit"]:
                ok = False
                break
            tags = [int(x) for x in re.findall(r"§(\d+) ", c)]
            pages.append(Page(img, tags[0] if tags else None, tags[-1] if tags else None, len(c)))
        if ok:
            return pages
        max_chars = int(max_chars * 0.95)


def page_tokens(flow: str, budget: float = DEFAULT_BUDGET) -> int:
    """Image tokens for a flow without rendering it (estimate from the page arithmetic; within a few %)."""
    per_page = int(W * W / budget * 0.97)
    full, rem = divmod(len(flow), per_page)
    return full * vision_tokens(W, W) + (vision_tokens(W, _height(rem, budget)) if rem else 0)


def page_chars(budget: float = DEFAULT_BUDGET) -> int:
    """Flow characters that fit on one full page (render() may still need slightly fewer)."""
    return int(W * W / budget * 0.97)
