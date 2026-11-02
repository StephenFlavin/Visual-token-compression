"""Shared layout engine used by every rendering scheme.

Contract
--------
A *scheme* turns source text into a sequence of ``Unit`` objects (see ``vbpe/schemes/base.py``).
This module wraps units into lines, finds the largest em size that fits a W x H canvas, and
rasterises. All schemes go through the same code so that comparisons differ only in glyph design.

Geometry conventions (all in pixels, ``em`` = font size passed to FreeType):
  * Every unit is drawn relative to a *baseline*. Its ink must stay inside the em box
    ``[baseline - 0.85*em, baseline + 0.25*em]``.
  * Line pitch = ``line_spacing * em``. Baseline of line i =
    ``margin + i*pitch + (line_spacing - 1)/2*em + 0.8*em``.
  * Line breaks are allowed only *after* a unit whose ``source`` ends with whitespace.
    Units with ``is_space=True`` that land at the end of a line are not drawn and take no width.
  * A unit's ``advance(em)`` is its horizontal footprint in pixels at that em.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=512)
def get_font(path: str | Path, size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), round(size, 3))


@lru_cache(maxsize=200_000)
def glyph_advance(path: str, size: float, text: str) -> float:
    return get_font(path, size).getlength(text)


class Unit:
    """One drawable item. Subclass and implement ``advance`` and ``draw``.

    Attributes
      source  : exact source text this unit encodes (may contain a trailing space).
      labels  : transcription symbols for this unit. ``"".join(labels)`` over all units is the
                exact *transcription string* a perfect reader would output; ``scheme.decode`` maps
                that string back to source text. CRNN (Phase 2) uses labels as CTC targets.
      is_space: pure whitespace unit (droppable at line ends).
    """

    source: str
    labels: tuple[str, ...]
    is_space: bool = False

    def advance(self, em: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def draw(self, d: ImageDraw.ImageDraw, x: float, baseline: float, em: float) -> None:  # pragma: no cover
        raise NotImplementedError


class GlyphUnit(Unit):
    """A single character (or short string) drawn with one font on the shared baseline."""

    __slots__ = ("char", "font_path", "source", "labels", "is_space")

    def __init__(self, char: str, font_path: str | Path, source: str | None = None,
                 labels: tuple[str, ...] | None = None):
        self.char = char
        self.font_path = str(font_path)
        self.source = char if source is None else source
        self.labels = (char,) if labels is None else labels
        self.is_space = char.isspace()

    def advance(self, em: float) -> float:
        return glyph_advance(self.font_path, round(em, 3), self.char)

    def draw(self, d: ImageDraw.ImageDraw, x: float, baseline: float, em: float) -> None:
        if not self.is_space:
            d.text((x, baseline), self.char, font=get_font(self.font_path, em), fill=0, anchor="ls")


@dataclass
class LayoutParams:
    margin: int = 4
    line_spacing: float = 1.2
    supersample: int = 1      # >1: render at k x size then BOX-downsample (applied to all schemes)

    def pitch(self, em: float) -> float:
        return self.line_spacing * em

    def first_baseline(self, em: float) -> float:
        return self.margin + (self.line_spacing - 1) / 2 * em + 0.8 * em


def _words(units: list[Unit]) -> list[list[Unit]]:
    words, cur = [], []
    for u in units:
        cur.append(u)
        if u.source[-1:].isspace():
            words.append(cur)
            cur = []
    if cur:
        words.append(cur)
    return words


def wrap(units: list[Unit], em: float, max_width: float) -> list[list[Unit]] | None:
    """Greedy word wrap. Returns lines, or None if a single unit is wider than max_width."""
    lines: list[list[Unit]] = []
    line: list[Unit] = []
    width = 0.0
    for word in _words(units):
        advs = [u.advance(em) for u in word]
        trail = advs[-1] if word[-1].is_space else 0.0
        ink_w = sum(advs) - trail
        if line and width + ink_w > max_width:
            lines.append(line)
            line, width = [], 0.0
        if not line and ink_w > max_width:  # hard-split an over-long word
            for u, a in zip(word, advs):
                if a > max_width:
                    return None
                if line and width + a > max_width and not u.is_space:
                    lines.append(line)
                    line, width = [], 0.0
                line.append(u)
                width += a
            continue
        line.extend(word)
        width += sum(advs)
    if line:
        lines.append(line)
    return lines


def fits(units: list[Unit], em: float, W: int, H: int, p: LayoutParams) -> list[list[Unit]] | None:
    lines = wrap(units, em, W - 2 * p.margin)
    if lines is None:
        return None
    # height used: pitch per line, last line only needs its em box below the baseline
    used = p.first_baseline(em) + (len(lines) - 1) * p.pitch(em) + 0.25 * em + p.margin
    return lines if used <= H else None


def fit_em(units: list[Unit], W: int, H: int, p: LayoutParams | None = None,
           lo: float = 1.0, hi: float | None = None, iters: int = 22) -> float:
    """Largest em (to ~0.01 px) at which ``units`` fit in W x H. Returns 0.0 if even ``lo`` fails."""
    p = p or LayoutParams()
    hi = hi or float(H)
    if not fits(units, lo, W, H, p):
        return 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if fits(units, mid, W, H, p):
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.01:
            break
    return math.floor(lo * 100) / 100  # floor, never round up past the fitting size


def _draw_lines(lines: list[list[Unit]], em: float, size: tuple[int, int], p: LayoutParams,
                scale: int) -> Image.Image:
    W, H = size
    img = Image.new("L", (W * scale, H * scale), 255)
    d = ImageDraw.Draw(img)
    sem = em * scale
    base = p.first_baseline(em) * scale
    for i, line in enumerate(lines):
        x = p.margin * scale
        y = base + i * p.pitch(em) * scale
        while line and line[-1].is_space:
            line = line[:-1]
        for u in line:
            u.draw(d, x, y, sem)
            x += u.advance(sem)
    if scale > 1:
        img = img.resize((W, H), Image.Resampling.BOX)
    return img


def render_page(units: list[Unit], W: int, H: int, p: LayoutParams | None = None,
                em: float | None = None) -> tuple[Image.Image, dict]:
    """Render units on a W x H canvas at the largest fitting em (or a given ``em``).

    Returns (image, info) where info has em, n_lines, fit (bool). If nothing fits the image
    is blank and fit=False.
    """
    p = p or LayoutParams()
    em = fit_em(units, W, H, p) if em is None else em
    lines = fits(units, em, W, H, p) if em > 0 else None
    if lines is None:
        return Image.new("L", (W, H), 255), {"em": em, "n_lines": 0, "fit": False}
    s = p.supersample
    # wrap is recomputed at the supersampled em so widths stay consistent with drawing
    if s > 1:
        lines = wrap(units, em * s, (W - 2 * p.margin) * s) or lines
    img = _draw_lines(lines, em, (W, H), p, s)
    return img, {"em": em, "n_lines": len(lines), "fit": True}


def render_line(units: list[Unit], em: float, p: LayoutParams | None = None,
                pad_right: int = 0) -> Image.Image:
    """Render all units on ONE line at a fixed em (for Phase 2 line-level OCR training).

    Height = round(line_spacing*em) + 2*margin; width = content width + 2*margin + pad_right.
    """
    p = p or LayoutParams()
    width = sum(u.advance(em) for u in units)
    W = int(round(width)) + 2 * p.margin + pad_right
    H = int(round(p.pitch(em))) + 2 * p.margin
    return _draw_lines([list(units)], em, (W, H), p, p.supersample)
