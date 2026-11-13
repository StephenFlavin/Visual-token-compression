"""Quadrant ("Hangul-style") packing: k consecutive letters of a word share one character cell.

Layouts (reading order in brackets), all inside the em box ``[baseline-0.84em, baseline+0.24em]``
(1.08 em tall, a hair inside the layout contract's 0.85/0.25 box)::

    m=1  [1]          one full-height glyph               width 2s+gi
    m=2  [1 / 2]      top half, bottom half               width s
    m=3  [1 2 / 3]    top-left, top-right, bottom (wide)   width 2s+gi
    m=4  [1 2 / 3 4]  2x2 grid: TL, TR, BL, BR             width 2s+gi

``s`` is the slot width and ``gi`` the intra-cell column gap; every cell is followed by an inter-cell
gap ``ge`` (~5x gi) so that cell boundaries stay visible. Default cells are *proportional*: each
slot column is as wide as its widest glyph (capped at 1.25 x 'o'); ``cell_width_em=...`` gives the
fixed-width cells sketched above (~5 % smaller fitted em). Each maximal run of non-space characters
is split left-to-right into chunks of k; a final shorter chunk of m < k characters uses the m-layout
(m=1: a full-height glyph at its natural width, ``single="top"`` puts it half-size in the top slot).
Spaces are ``SpaceUnit``s of 0.25 em (+ the preceding cell gap 0.18 em).

Glyph design (aimed at maximal legibility):
  * Every slot is a miniature line: the font's *letter* ink range (top of ``b``/``f`` to bottom of
    ``g``) is mapped onto the slot height, so case, x-height and descenders are preserved and all
    letters in a row share a baseline. Rare glyphs that poke outside that range (brackets, ``_``,
    `````) are shifted (or, if taller than the slot, squashed) back inside.
  * Horizontally a glyph keeps its natural width (times a per-slot stretch, 1.0 normally, >1 for the
    wide bottom slot of the 3-layout) and is centred; glyphs wider than the slot (m, w, M, W, ...)
    are squashed to the slot width.
  * Glyphs are rasterised at 4x, transformed, placed at a quarter-pixel offset and BOX-downsampled;
    the resulting alpha masks are cached (``_glyph_mask``) and blitted with ``ImageDraw.bitmap``.

Default font: Archivo Narrow (``config.CONDENSED_FONT``) -- see ``DEFAULT_FONT`` comment.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vtc import config
from vtc.layout import GlyphUnit, Unit, get_font
from vtc.schemes.base import Scheme

# Vertical extent of a cell, in layout em, relative to the baseline (contract box is 0.85 / 0.25).
CELL_TOP = 0.84
CELL_BOT = 0.24
CELL_H = CELL_TOP + CELL_BOT

# Archivo Narrow gave the most legible packed letters in side-by-side samples (DejaVu Sans needs
# ~25 % wider slots for the same letter height, so at equal canvas area its letters end up smaller;
# DejaVu Sans Mono squashes badly because its i/l/r are drawn as wide as m/w).
DEFAULT_FONT = config.CONDENSED_FONT

_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_OVERSAMPLE = 4
_Q = 4  # sub-pixel quantisation of positions / sizes (1/4 px)


@lru_cache(maxsize=16)
def font_metrics(font_path: str) -> dict:
    """Ink metrics in em units: letter/digit top & bottom vs baseline, ink width of each ASCII char."""
    size = 400
    f = ImageFont.truetype(font_path, size)
    top = bot = 0.0
    widths = {}
    for c in map(chr, range(33, 127)):
        im = Image.new("L", (3 * size, 3 * size), 0)
        ImageDraw.Draw(im).text((size, 2 * size), c, font=f, fill=255, anchor="ls")
        b = im.getbbox()
        if b is None:
            continue
        widths[c] = (b[2] - b[0]) / size
        if c in _LETTERS:
            top = max(top, (2 * size - b[1]) / size)
            bot = max(bot, (b[3] - 2 * size) / size)
    return {"top": top, "bot": bot, "range": top + bot, "ink_w": widths}


@lru_cache(maxsize=200_000)
def _glyph_mask(font_path: str, char: str, sw: float, sh: float, stretch: float,
                fx: float, fy: float) -> Image.Image:
    """Alpha mask (L, 255 = ink) of ``char`` fitted into a sw x sh px slot at offset (fx, fy) px.

    The mask is meant to be pasted at the integer pixel that contains the slot's top-left corner;
    ``fx, fy`` in [0, 1) are the slot corner's fractional position inside that pixel.
    """
    Wl = int(math.ceil(fx + sw)) + 1
    Hl = int(math.ceil(fy + sh)) + 1
    out = Image.new("L", (Wl, Hl), 0)
    if sw <= 0 or sh <= 0:
        return out
    O = _OVERSAMPLE if sh * _OVERSAMPLE <= 320 else max(1, int(320 // sh))
    m = font_metrics(font_path)
    Hs, As = sh * O, sw * O
    F = Hs / m["range"]                       # font size whose letter ink range == slot height
    font = get_font(font_path, F)
    pad = int(F) + 2
    base = pad + m["top"] * F
    can = Image.new("L", (int(font.getlength(char) + 2 * pad + 1), int(Hs + 2 * pad + 1)), 0)
    ImageDraw.Draw(can).text((pad, base), char, font=font, fill=255, anchor="ls")
    bb = can.getbbox()
    if bb is None:
        return out
    glyph = can.crop(bb)
    gw, gh = glyph.size
    y0 = bb[1] - pad                          # glyph top relative to slot top (hi-res px)
    th = gh
    if gh > Hs:                               # taller than the slot: squash vertically
        th, y0 = max(1, int(Hs)), 0.0
    elif y0 < 0 or y0 + gh > Hs:              # sticks out (e.g. '(' '_' '`'): shift inside
        y0 = min(max(y0, 0.0), Hs - gh)
    tw = max(1, int(round(min(As, gw * stretch))))
    if (tw, th) != (gw, gh):
        glyph = glyph.resize((tw, th), Image.Resampling.LANCZOS)
    hi = Image.new("L", (Wl * O, Hl * O), 0)
    hi.paste(glyph, (int(round(fx * O + (As - tw) / 2)), int(round(fy * O + y0))))
    return hi.resize((Wl, Hl), Image.Resampling.BOX) if O > 1 else hi


def _q(v: float) -> float:
    return round(v * _Q) / _Q


class QuadCell(Unit):
    """One packed cell holding a chunk of 1..4 consecutive characters of a word."""

    __slots__ = ("source", "labels", "is_space", "font_path", "slots", "adv_em")

    def __init__(self, chunk: str, scheme: "QuadrantScheme"):
        assert 1 <= len(chunk) <= 4 and not any(c.isspace() for c in chunk)
        self.source = chunk
        self.labels = tuple(chunk)
        self.is_space = False
        self.font_path = scheme.font_path
        self.slots, self.adv_em = scheme.cell_geometry(chunk)

    def advance(self, em: float) -> float:
        return self.adv_em * em

    def draw(self, d: ImageDraw.ImageDraw, x: float, baseline: float, em: float) -> None:
        for ch, (sx, sy, sw, sh, stretch) in zip(self.source, self.slots):
            px, py = x + sx * em, baseline + sy * em
            ix, iy = math.floor(px), math.floor(py)
            fx, fy = _q(px - ix), _q(py - iy)
            if fx >= 1:
                ix, fx = ix + 1, 0.0
            if fy >= 1:
                iy, fy = iy + 1, 0.0
            mask = _glyph_mask(self.font_path, ch, _q(sw * em), _q(sh * em), stretch, fx, fy)
            d.bitmap((ix, iy), mask, fill=0)


class SpaceUnit(GlyphUnit):
    """A space with a fixed advance (em units) independent of the font's space width."""

    __slots__ = ("width_em",)

    def __init__(self, char: str, font_path: str, width_em: float):
        super().__init__(char, font_path)
        self.is_space = True
        self.width_em = width_em

    def advance(self, em: float) -> float:
        return self.width_em * em

    def draw(self, d, x, baseline, em) -> None:
        pass


_LAYOUT_TEXT = {
    1: "a cell with ONE full-height letter is just that letter",
    2: "a cell with TWO letters stacked vertically reads top, then bottom",
    3: ("a cell with THREE letters (two small letters side by side on top, one letter stretched "
        "across the whole bottom half) reads top-left, top-right, then bottom"),
    4: ("a cell with FOUR letters in a 2x2 grid reads top-left, top-right, bottom-left, "
        "bottom-right (like a book: left to right, then the next row)"),
}

# cover every chunk length (1..k) for each k, plus capitals, digits and punctuation
_LEGEND_WORDS = {
    2: ["a", "of", "the", "word", "quick", "brown,", "jumps", "1984.", "Over", "(lazy)",
        "letters;", "language"],
    3: ["a", "of", "the", "word", "quick", "brown,", "jumps", "1984.", "Over", "(lazy)",
        "letters;", "language"],
    4: ["a", "of", "the", "word", "quick", "brown,", "jumps", "1984.", "Over", "(lazy)",
        "letters;", "language"],
}


class QuadrantScheme(Scheme):
    """Hangul-style packing of k = 2, 3 or 4 characters per cell.

    ``cell_width_em=None`` (default) uses *proportional* cells: each slot column is as wide as the
    widest glyph in it (capped at ``max_w`` x the width of 'o'; wider glyphs are squashed), like a
    proportional font. A number gives fixed-width cells of that width (k-layout, in layout em);
    then every slot is ``s`` wide (``s = cell_width_em`` for k=2, ``(cell_width_em - gap)/2`` else).
    """

    def __init__(self, k: int = 3, font_path: str | Path = DEFAULT_FONT,
                 cell_width_em: float | None = None, row_gap_em: float = 0.05,
                 col_gap_em: float = 0.035, cell_gap_em: float = 0.18, space_em: float = 0.25,
                 wide_stretch: float = 1.5, max_w: float = 1.25, single: str = "full",
                 name: str | None = None):
        if k not in (2, 3, 4):
            raise ValueError("k must be 2, 3 or 4")
        self.k = k
        self.font_path = str(font_path)
        self.name = name or f"quad{k}"
        self.gv, self.gi, self.ge = row_gap_em, col_gap_em, cell_gap_em
        self.space_em = space_em
        self.wide_stretch = wide_stretch
        self.max_w = max_w
        if single not in ("full", "top"):
            raise ValueError("single must be 'full' or 'top'")
        self.single = single
        self.row_h = (CELL_H - self.gv) / 2
        self.cell_width_em = cell_width_em
        fm = font_metrics(self.font_path)
        self._ink_w = fm["ink_w"]
        self._o = fm["ink_w"]["o"]
        if cell_width_em is not None:
            self.s = cell_width_em if k == 2 else (cell_width_em - self.gi) / 2
        else:
            self.s = None
        self._geom: dict[str, tuple] = {}

    # ---- geometry ----
    def letter_font_em(self, m: int = 2) -> float:
        """Font size (in layout em) of letters in an m-layout cell (letter ink range == slot h)."""
        h = CELL_H if m == 1 else self.row_h
        return h / font_metrics(self.font_path)["range"]

    def _w(self, c: str, f: float) -> float:
        """Slot width (em) for glyph ``c`` at font size ``f`` (em) in proportional mode."""
        w = self._ink_w.get(c, self._o)
        return min(max(w, 0.06), self.max_w * self._o) * f

    def cell_geometry(self, chunk: str) -> tuple[tuple, float]:
        """Slots ``((x, y_top, w, h, stretch), ...)`` in layout em relative to (pen x, baseline),
        one per character in reading order, and the cell advance (incl. the inter-cell gap)."""
        g = self._geom.get(chunk)
        if g is not None:
            return g
        m, gi, rh = len(chunk), self.gi, self.row_h
        top, bot = -CELL_TOP, CELL_BOT - rh
        if self.s is not None:                      # fixed-width cells
            s = self.s
            ws = [s] * m
            w1 = 2 * s + gi
        else:
            f = self.letter_font_em(2)
            ws = [self._w(c, f) for c in chunk]
            w1 = self._w(chunk[0], self.letter_font_em(1))
        if m == 1 and self.single == "top":         # half-size letter in the top slot
            slots = ((0.0, top, ws[0], rh, 1.0),)
        elif m == 1:                                # one full-height letter
            slots = ((0.0, top, w1, CELL_H, 1.0),)
        elif m == 2:
            c = max(ws)
            slots = ((0.0, top, c, rh, 1.0), (0.0, bot, c, rh, 1.0))
        elif m == 3:
            topw = ws[0] + gi + ws[1]
            width = topw if self.s is not None else max(topw, ws[2])
            x0 = (width - topw) / 2
            slots = ((x0, top, ws[0], rh, 1.0), (x0 + ws[0] + gi, top, ws[1], rh, 1.0),
                     (0.0, bot, width, rh, self.wide_stretch))
        else:
            c1, c2 = max(ws[0], ws[2]), max(ws[1], ws[3])
            slots = ((0.0, top, c1, rh, 1.0), (c1 + gi, top, c2, rh, 1.0),
                     (0.0, bot, c1, rh, 1.0), (c1 + gi, bot, c2, rh, 1.0))
        adv = max(sl[0] + sl[2] for sl in slots) + self.ge
        if len(self._geom) > 500_000:
            self._geom.clear()
        g = self._geom[chunk] = (slots, adv)
        return g

    # ---- encoding ----
    def chunks(self, text: str) -> list[str]:
        out = []
        for tok in re.findall(r"\s|\S+", text):
            if tok.isspace():
                out.append(tok)
            else:
                out.extend(tok[i:i + self.k] for i in range(0, len(tok), self.k))
        return out

    def encode(self, text: str) -> list[Unit]:
        return [SpaceUnit(c, self.font_path, self.space_em) if c.isspace() else QuadCell(c, self)
                for c in self.chunks(text)]

    # ---- VLM ----
    def vlm_system_prompt(self) -> str:
        k = self.k
        rules = [_LAYOUT_TEXT[m] for m in range(k, 0, -1)]
        return (
            "You are a precise OCR engine for an unusual but simple way of writing English. "
            f"Words are written with ordinary Latin letters, digits and punctuation, but up to {k} "
            "consecutive characters of a word are packed into one square-ish cell, drawn at reduced "
            "size. Each word is split left to right into cells of "
            f"{k} characters; only the last cell of a word may hold fewer. Punctuation and digits "
            "are packed like letters. Reading rule inside a cell: "
            + "; ".join(rules) + ". "
            "Cells of the same word sit close together with a narrow gap; a wider gap between cells "
            "is a real space between words. Letter case is preserved (small letters are drawn "
            "smaller than capitals). Read cells left to right, lines top to bottom. "
            "If more than one image is given, the earlier images are a legend with example words "
            "and their plain spelling printed beneath; transcribe ONLY the last image. "
            "Output only the plain English text, unpacked into normal spelling, with lines joined "
            "by single spaces. Do not correct, complete, summarise or comment."
        )

    def vlm_prompt(self) -> str:
        return ("Transcribe all text in the last image as plain English text (unpack each cell "
                "in its reading order). Output only the text.")

    def vlm_key_images(self) -> list[Image.Image]:
        return [self.legend_image()]

    def legend_image(self, em: float = 48.0, width: int = 800) -> Image.Image:
        """Legend: reading-order diagrams (digits 1..m packed as cells) + example words, each with
        its plain spelling beneath in normal type."""
        lab_font = get_font(config.LATIN_FONT, 17)
        words = _LEGEND_WORDS[self.k]
        cols, pad, row_h = 3, 20, int(em * 1.1 + 34)
        top_h = int(em * 1.1 + 74)
        H = top_h + math.ceil(len(words) / cols) * row_h + 6
        img = Image.new("L", (width, H), 255)
        d = ImageDraw.Draw(img)
        d.text((pad, 8), f"{self.name}: up to {self.k} characters per cell. Reading order "
               "inside a cell (digits show the order):", font=lab_font, fill=0)

        def draw_word(word: str, x: float, base: float) -> None:
            for u in self.encode(word):
                u.draw(d, x, base, em)
                x += u.advance(em)

        x = pad
        for m in range(self.k, 0, -1):
            draw_word("1234"[:m], x, 36 + 0.86 * em)
            d.text((x, 36 + 1.14 * em + 2), f"{m} char" + ("s" if m > 1 else ""),
                   font=lab_font, fill=90)
            x += 150
        d.line((pad, top_h - 8, width - pad, top_h - 8), fill=170)
        colw = (width - 2 * pad) // cols
        for i, w in enumerate(words):
            r, c = divmod(i, cols)
            x0, y0 = pad + c * colw, top_h + r * row_h
            draw_word(w, x0, y0 + 0.86 * em)
            d.text((x0, y0 + 1.14 * em + 2), f'"{w}"', font=lab_font, fill=90)
        return img


def make_quadrant_scheme(name: str, **kw) -> QuadrantScheme:
    """``quad2`` / ``quad3`` / ``quad4``, optionally with a font suffix: ``_dejavu`` | ``_mono``
    | ``_narrow`` (default)."""
    mt = re.fullmatch(r"quad([234])(?:_(dejavu|mono|narrow))?", name)
    if not mt:
        raise KeyError(f"unknown quadrant scheme {name!r}")
    fonts = {"dejavu": config.LATIN_FONT, "mono": config.LATIN_MONO_FONT,
             "narrow": config.CONDENSED_FONT, None: DEFAULT_FONT}
    kw.setdefault("font_path", fonts[mt.group(2)])
    return QuadrantScheme(k=int(mt.group(1)), name=name, **kw)
