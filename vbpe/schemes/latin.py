"""Baselines: plain Latin typography in a regular and a condensed font."""
from __future__ import annotations

from pathlib import Path

from vbpe import config
from vbpe.layout import GlyphUnit, Unit
from vbpe.schemes.base import Scheme


class LatinScheme(Scheme):
    def __init__(self, font_path: str | Path = config.LATIN_FONT, name: str = "latin"):
        self.font_path = str(font_path)
        self.name = name

    def encode(self, text: str) -> list[Unit]:
        return [GlyphUnit(c, self.font_path) for c in text]
