"""Substitution scheme ("Visual BPE" zero-shot proxy).

Frequent multi-character chains from the dictionary (``vbpe.dictionary.Dictionary``, WP-A) are
each drawn as ONE common CJK ideograph (``assets/cjk_symbols.txt``, frequency order); everything
else is drawn in the condensed Latin font so the comparison with ``latin_condensed`` is fair.

Mapping: ``dictionary.tokens[i] -> symbols[i]`` (both in priority / frequency order).

Vertical metrics: Noto Sans CJK ideographs have 1.0 em advance and, over the symbol pool, ink in
about ``[baseline-0.861em, baseline+0.101em]``. The layout em box is ``[-0.85, +0.25]``, so the
glyphs are kept at full size (``cjk_scale=1``) and shifted down by the minimum amount (plus a
0.01 em safety pad) needed to keep the ink inside the box (~0.02 em for the default font). The
shift is computed over the whole symbol pool (identical for every substN), so other fonts /
scales also stay inside the box.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from PIL import ImageDraw

from vbpe import config
from vbpe.layout import GlyphUnit, Unit, get_font
from vbpe.schemes.base import Scheme

SYMBOLS_PATH = config.ROOT / "assets" / "cjk_symbols.txt"
EM_TOP, EM_BOTTOM = -0.85, 0.25     # layout em-box convention (vbpe/layout.py)
_PAD = 0.01                         # em safety margin inside the box
VISIBLE_SPACE = "␣"            # ␣ — how spaces inside tokens are shown in the legend

# CJK / full-width punctuation the VLM may emit instead of ASCII.
_PUNCT_MAP = {
    "　": " ", "，": ",", "。": ".", "：": ":", "；": ";", "“": '"', "”": '"', "‘": "'",
    "’": "'", "（": "(", "）": ")", "！": "!", "？": "?", "、": ",", "「": '"', "」": '"',
    "『": '"', "』": '"', "【": "[", "】": "]", "《": '"', "》": '"', "…": "...",
    "␣": " ",                  # a visible-space mark copied from the legend
}


@lru_cache(maxsize=4)
def load_symbols(path: str | Path = SYMBOLS_PATH) -> tuple[str, ...]:
    """Symbol pool, most frequent first (comment lines start with '#')."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return tuple(out)


@lru_cache(maxsize=32)
def ink_extent(font_path: str, symbols: tuple[str, ...], size: int = 400) -> tuple[float, float]:
    """(top, bottom) of the union of the symbols' ink relative to the baseline, in em (top < 0)."""
    f = get_font(font_path, size)
    top, bot = 0.0, 0.0
    for c in symbols:
        mask, (_, oy) = f.getmask2(c, anchor="ls")
        if mask.size[1]:
            top, bot = min(top, oy), max(bot, oy + mask.size[1])
    return top / size, bot / size


def vertical_shift(top: float, bottom: float, scale: float) -> float:
    """Smallest downward shift (em, + = down) that keeps scaled ink inside the em box."""
    t, b = top * scale, bottom * scale
    if b - t > (EM_BOTTOM - EM_TOP) - 2 * _PAD:
        raise ValueError(f"cjk_scale={scale} too large: ink height {b - t:.3f} em exceeds the em box")
    dy = 0.0
    if t < EM_TOP + _PAD:
        dy = EM_TOP + _PAD - t
    if b + dy > EM_BOTTOM - _PAD:
        dy = EM_BOTTOM - _PAD - b
    return dy


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x3134F or 0x2E80 <= o <= 0x2FDF or 0x3040 <= o <= 0x30FF
            or 0x3100 <= o <= 0x312F or 0xAC00 <= o <= 0xD7AF)


def normalize_fullwidth(s: str) -> str:
    """Full-width ASCII (U+FF01-FF5E), ideographic space and CJK punctuation -> ASCII."""
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            out.append(chr(o - 0xFEE0))
        else:
            out.append(_PUNCT_MAP.get(ch, ch))
    return "".join(out)


class CJKUnit(Unit):
    """One ideograph standing for a dictionary token. Advance is exactly ``scale * em``."""

    __slots__ = ("char", "font_path", "source", "labels", "is_space", "scale", "dy")

    def __init__(self, char: str, font_path: str | Path, source: str, scale: float = 1.0,
                 dy: float = 0.0):
        self.char = char
        self.font_path = str(font_path)
        self.source = source
        self.labels = (char,)
        self.is_space = False       # a token is always drawn, even if it ends in a space
        self.scale = scale
        self.dy = dy                # vertical shift in em (+ = down)

    def advance(self, em: float) -> float:
        return self.scale * em

    def draw(self, d: ImageDraw.ImageDraw, x: float, baseline: float, em: float) -> None:
        d.text((x, baseline + self.dy * em), self.char, font=get_font(self.font_path, em * self.scale),
               fill=0, anchor="ls")


def _show(token: str) -> str:
    return token.replace(" ", VISIBLE_SPACE)


class SubstitutionScheme(Scheme):
    def __init__(self, n_tokens: int = 300, dictionary=None, legend: bool = False,
                 leftover_font: str | Path = config.CONDENSED_FONT,
                 cjk_font: str | Path = config.CJK_FONT, cjk_scale: float = 1.0,
                 name: str | None = None, symbols_path: str | Path = SYMBOLS_PATH):
        self.n_tokens = int(n_tokens)
        self.legend = bool(legend)
        self.leftover_font = str(leftover_font)
        self.cjk_font = str(cjk_font)
        self.cjk_scale = float(cjk_scale)
        self.name = name or f"subst{self.n_tokens}" + ("_legend" if legend else "")
        self._symbols_path = str(symbols_path)
        self._dictionary = dictionary
        self._ready = False

    # ---- lazy setup (so the scheme can be constructed before data/dict_*.json exists) ----
    @property
    def dictionary(self):
        if self._dictionary is None:
            from vbpe.dictionary import Dictionary
            self._dictionary = Dictionary.load(self.n_tokens)
        return self._dictionary

    def _setup(self) -> None:
        if self._ready:
            return
        tokens = list(self.dictionary.tokens)[: self.n_tokens]
        pool = load_symbols(self._symbols_path)
        if len(tokens) > len(pool):
            raise ValueError(f"{len(tokens)} tokens but only {len(pool)} symbols in {self._symbols_path}")
        if len(set(tokens)) != len(tokens):
            raise ValueError("dictionary tokens are not unique")
        syms = pool[: len(tokens)]
        self.tokens = tokens
        self.symbols = list(syms)
        self.tok2sym = dict(zip(tokens, syms))
        self.sym2tok = dict(zip(syms, tokens))
        # shift computed over the WHOLE pool so every substN places glyphs identically
        top, bot = ink_extent(self.cjk_font, tuple(pool))
        self.cjk_dy = vertical_shift(top, bot, self.cjk_scale)
        self._ready = True

    _LAZY = frozenset({"tokens", "symbols", "tok2sym", "sym2tok", "cjk_dy"})

    def __getattr__(self, attr):     # only called for missing attributes
        if attr in SubstitutionScheme._LAZY and not self.__dict__.get("_ready", True):
            self._setup()
            return self.__dict__[attr]
        raise AttributeError(attr)

    # ---- scheme interface ----
    def encode(self, text: str) -> list[Unit]:
        self._setup()
        units: list[Unit] = []
        for seg in self.dictionary.segment(text):
            sym = self.tok2sym.get(seg)
            if sym is not None:
                units.append(CJKUnit(sym, self.cjk_font, seg, self.cjk_scale, self.cjk_dy))
            else:   # single leftover char (or a token beyond n_tokens: spell it out)
                units.extend(GlyphUnit(c, self.leftover_font) for c in seg)
        return units

    def decode(self, transcription: str) -> str:
        self._setup()
        out = []
        for ch in normalize_fullwidth(transcription):
            tok = self.sym2tok.get(ch)
            if tok is not None:
                out.append(tok)
            elif is_cjk(ch):
                out.append("?")     # unknown ideograph: counts as an error
            else:
                out.append(ch)
        return "".join(out)

    # ---- VLM prompts ----
    def legend_text(self) -> str:
        self._setup()
        return "\n".join(f"{s}\t{_show(t)}" for t, s in zip(self.tokens, self.symbols))

    def vlm_system_prompt(self) -> str:
        if not self.legend:
            return (
                "You are a precise OCR engine. The image shows English text in which many frequent "
                "character sequences (words, word pieces and the spaces next to them) have been "
                "replaced by single Chinese characters. All other characters are ordinary Latin "
                "letters, digits, spaces and punctuation.\n"
                "Transcribe EXACTLY what is written, character by character, in reading order (left "
                "to right, top to bottom): write every Chinese character as that same Chinese "
                "character, every Latin letter, digit and punctuation mark as itself (ASCII), and "
                "keep spaces where there is a visible gap. Do not add spaces around Chinese "
                "characters unless a gap is visible. Join lines with single spaces.\n"
                "Do NOT translate, decode, interpret, correct or complete anything. Do not explain. "
                "Output only the transcription."
            )
        return (
            "You are a precise OCR engine. The image shows English text in which frequent character "
            "sequences have been replaced by single Chinese characters according to the legend "
            "below. All other characters are ordinary Latin letters, digits, spaces and punctuation.\n"
            "Read the image in order (left to right, top to bottom) and output the ORIGINAL plain "
            "English text: replace every Chinese character by the sequence it stands for and copy "
            "everything else exactly. Join lines with single spaces. Do not correct, complete, "
            "summarise or comment. Output only the decoded text.\n\n"
            f"Legend format: one entry per line: the Chinese character, a TAB, then the character "
            f"sequence it stands for. Inside sequences a space is shown as {VISIBLE_SPACE} (write it "
            f"as a normal space); all other characters are literal.\n\n"
            "LEGEND\n" + self.legend_text()
        )

    def vlm_prompt(self) -> str:
        if not self.legend:
            return ("Transcribe the text in the image exactly as written, character by character "
                    "(Chinese characters as Chinese characters, Latin as Latin). Output only the "
                    "transcription.")
        return ("Decode the text in the image into plain English using the legend. Output only the "
                "decoded text.")


_NAME_RE = re.compile(r"^subst(\d+)(_legend)?$")


def make_substitution_scheme(name: str, **kw) -> SubstitutionScheme:
    """``subst300``, ``subst1000``, ``subst3000``, ``subst300_legend`` ... (+ constructor kwargs)."""
    m = _NAME_RE.match(name)
    if not m:
        raise KeyError(f"unknown substitution scheme {name!r} (expected substN or substN_legend)")
    kw.setdefault("name", name)
    return SubstitutionScheme(n_tokens=int(m.group(1)), legend=bool(m.group(2)), **kw)
