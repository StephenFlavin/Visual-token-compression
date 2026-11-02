"""Synthetic schemes for exercising Phase-2 code paths before WP-C/WP-D land. NOT experiments.

* ``stub_cjk``  : every pair of consecutive ASCII letters -> one CJK ideograph (52*52 = 2704
                  symbols, + ~90 Latin leftovers ~= a 3000-symbol alphabet), label = the ideograph.
* ``stub_quad`` : runs of up to 4 non-space chars packed into one 2x2 cell drawn at em/2;
                  ONE unit emits 1-4 labels (tests the CTC T >= 2L+1 frame rule).
"""
from __future__ import annotations

import string

from vbpe import config
from vbpe.layout import GlyphUnit, Unit, get_font, glyph_advance
from vbpe.schemes.base import Scheme

LETTERS = string.ascii_letters


class _CellUnit(Unit):
    __slots__ = ("chars", "font_path", "source", "labels", "is_space")

    def __init__(self, chars: str, font_path: str):
        self.chars, self.font_path = chars, str(font_path)
        self.source, self.labels, self.is_space = chars, tuple(chars), False

    def _rows(self):
        return self.chars[:2], self.chars[2:4]

    def advance(self, em: float) -> float:
        s = round(em / 2, 3)
        return max(glyph_advance(self.font_path, s, r) for r in self._rows() if r) + 0.05 * em

    def draw(self, d, x, baseline, em) -> None:
        f = get_font(self.font_path, em / 2)
        top, bot = self._rows()
        d.text((x, baseline - 0.45 * em), top, font=f, fill=0, anchor="ls")
        if bot:
            d.text((x, baseline + 0.10 * em), bot, font=f, fill=0, anchor="ls")


class StubCJK(Scheme):
    def __init__(self):
        self.name = "stub_cjk"
        self.pairs = {a + b: chr(0x4E00 + i) for i, (a, b) in
                      enumerate((a, b) for a in LETTERS for b in LETTERS)}
        self.inv = {v: k for k, v in self.pairs.items()}

    def encode(self, text: str) -> list[Unit]:
        out, i = [], 0
        while i < len(text):
            sym = self.pairs.get(text[i:i + 2])
            if sym:
                out.append(GlyphUnit(sym, config.CJK_FONT, source=text[i:i + 2]))
                i += 2
            else:
                out.append(GlyphUnit(text[i], config.CONDENSED_FONT))
                i += 1
        return out

    def decode(self, transcription: str) -> str:
        return "".join(self.inv.get(c, c) for c in transcription)


class StubQuad(Scheme):
    def __init__(self):
        self.name = "stub_quad"

    def encode(self, text: str) -> list[Unit]:
        out = []
        for word in text.split(" "):
            for i in range(0, len(word), 4):
                out.append(_CellUnit(word[i:i + 4], config.LATIN_FONT))
            out.append(GlyphUnit(" ", config.LATIN_FONT))
        return out[:-1] if out else out


def make_stub_scheme(name: str) -> Scheme:
    return {"stub_cjk": StubCJK, "stub_quad": StubQuad}[name]()
