"""224x224 RGB pages for the frozen-encoder probe.

Every encoding lays text out in horizontal lines of integer pitch ``p`` px, stacked from the top
(line i occupies rows [i*p, (i+1)*p)), full page width. The probe reads each line strip with CTC.

Families
  latin_p<p>        DejaVu Sans at em = p/1.2, 4x supersampled (same renderer as Phase 2)
  lg_<tag>          a learned glyph bank (learned_glyphs/results/...) at its native pitch
  col_b<b>          one solid b x b colour block per character, 95-colour palette (5 levels/channel)
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vtc import config
from vtc.layout import LayoutParams, _draw_lines, wrap
from vtc.metrics import normalize

ROOT = Path(__file__).resolve().parents[2]
W = H = 224
PATCH = 16
PRINTABLE = [chr(i) for i in range(32, 127)]
LP = LayoutParams(margin=0, line_spacing=1.2, supersample=4)

LG_BANKS = {  # tag -> bank path (converged e1long banks, N=300 tokens)
    "lg_free_h4": "learned_glyphs/results/e1long/banks/lg_learned_n300_h4.pt",
    "lg_free_w23_h3": "learned_glyphs/results/e1long/banks/lg_learned_n300_w23_h3.pt",
    "lg_free_w23_h4": "learned_glyphs/results/e1long/banks/lg_learned_n300_w23_h4.pt",
    "lg_pb_h4": "learned_glyphs/results/e1long/banks/lg_learned_n300_pb_h4.pt",
    "lg_pb_h5": "learned_glyphs/results/e1long/banks/lg_learned_n300_pb_h5.pt",
    "lg_n0_pb_h5": "learned_glyphs/results/e1long/banks/lg_learned_n0_pb_h5.pt",   # characters only (95 symbols)
    # grid-aligned whole-pixel banks (7-px rows; PLAN_3090 amendment 1, fix 3): cells w x h
    "lg_n0_pb_c2_h7": "learned_glyphs/results/e1align/banks/lg_learned_n0_pb_c2_h7.pt",     # chars 2x7: 56 chars/token
    "lg_n0_pb_c4_h7": "learned_glyphs/results/e1align/banks/lg_learned_n0_pb_c4_h7.pt",     # chars 4x7: 28
    "lg_n0_pb_c7_h7": "learned_glyphs/results/e1align/banks/lg_learned_n0_pb_c7_h7.pt",     # chars 7x7: 16
    "lg_pb_c2t4_h7": "learned_glyphs/results/e1align/banks/lg_learned_n300_pb_c2t4_h7.pt",  # BPE 2x7 + 4x7: ~76
    # the same two dense banks retrained with a minimum glyph distance (same-width glyphs >= 2 px apart;
    # PLAN_3090 amendment 4, stage C; train_joint.py --min-dist-px 2)
    "lg_n0_pb_c2_h7md": "learned_glyphs/results/e1align_md/banks/lg_learned_n0_pb_c2_h7md.pt",
    "lg_pb_c2t4_h7md": "learned_glyphs/results/e1align_md/banks/lg_learned_n300_pb_c2t4_h7md.pt",
}


@dataclass
class Page:
    img: np.ndarray                       # (H, W, 3) uint8
    labels: list[tuple[str, ...]]         # CTC label tuple per line
    refs: list[str]                       # normalised source text per line

    @property
    def n_chars(self) -> int:
        return sum(len(r) for r in self.refs)


def clean_text(s: str) -> str:
    s = "".join(c if 32 <= ord(c) < 127 else " " for c in s)
    return normalize(s)


def text_stream(split: str, max_chars: int) -> str:
    with open(ROOT / "data" / f"{split}.txt", encoding="utf-8") as f:
        return clean_text(f.read(int(max_chars * 1.05) + 1000))


@dataclass
class Encoding:
    name: str
    family: str
    pitch: int
    classes: list[str] = field(default_factory=list)   # CTC classes (blank is extra, index 0)

    def decode(self, labels: str) -> str:
        return labels

    def clean_hyp(self, s: str) -> str:
        """Hypothesis post-processing matching how ``refs`` were made (normalised source text)."""
        return normalize(s)

    def render(self, text: str, pos: int) -> tuple[Page, int]:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def n_lines(self) -> int:
        return H // self.pitch


class SchemeEncoding(Encoding):
    """Any vtc Scheme (Latin font or learned-glyph bank) rendered with the shared layout engine."""

    def __init__(self, name, family, pitch, scheme, classes):
        super().__init__(name, family, pitch, classes)
        self.scheme = scheme
        self.em = pitch / LP.line_spacing

    def decode(self, labels: str) -> str:
        return self.scheme.decode(labels)

    def render(self, text: str, pos: int) -> tuple[Page, int]:
        s = LP.supersample
        est = self.n_lines * W + 200
        units = self.scheme.encode(text[pos:pos + est])
        lines = (wrap(units, self.em * s, W * s) or [])[: self.n_lines]
        consumed = sum(len(u.source) for ln in lines for u in ln)
        img = _draw_lines([list(ln) for ln in lines], self.em, (W, H), LP, s)
        g = np.asarray(img, dtype=np.uint8)
        labs, refs = [], []
        for ln in lines:
            while ln and ln[-1].is_space:
                ln = ln[:-1]
            labs.append(tuple(lab for u in ln for lab in u.labels))
            refs.append(normalize("".join(u.source for u in ln)))
        return Page(np.repeat(g[:, :, None], 3, 2), labs, refs), consumed


def palette(n: int = len(PRINTABLE), levels=(0, 64, 128, 191, 255), seed: int = 0) -> np.ndarray:
    cols = list(itertools.product(levels, repeat=3))
    random.Random(seed).shuffle(cols)
    return np.array(cols[:n], dtype=np.uint8)


class ColourEncoding(Encoding):
    """One solid b x b block per character; chars fill lines left to right (no word wrap)."""

    def __init__(self, b: int):
        super().__init__(f"col_b{b}", "colour", b, list(PRINTABLE))
        self.b = b
        self.pal = palette()
        self.idx = {c: i for i, c in enumerate(PRINTABLE)}

    def clean_hyp(self, s: str) -> str:
        # refs are raw fixed-length chunks (they can start/end with a space): do not strip the hypothesis
        return s

    def render(self, text: str, pos: int) -> tuple[Page, int]:
        b, per = self.b, W // self.b
        img = np.full((H, W, 3), 255, np.uint8)
        labs, refs, p = [], [], pos
        for i in range(self.n_lines):
            chunk = text[p:p + per]
            p += len(chunk)
            ids = np.array([self.idx[c] for c in chunk], dtype=np.int64)
            row = np.repeat(np.repeat(self.pal[ids][None], b, 0), b, 1)       # (b, len*b, 3)
            img[i * b:(i + 1) * b, : row.shape[1]] = row
            labs.append(tuple(chunk))
            refs.append(chunk)
        return Page(img, labs, refs), p - pos


def get_encoding(name: str) -> Encoding:
    if name.startswith(("mono_p", "rgb3_p", "rgb3a_p", "rgbw_p", "rgb3_lg_")):  # RGB overlay families (rgb_overlay.py)
        from encoder_probe.ep.rgb_overlay import get_rgb_encoding
        return get_rgb_encoding(name)
    if name.startswith("latin_p"):
        from vtc.schemes.latin import LatinScheme
        p = int(name[len("latin_p"):])
        return SchemeEncoding(name, "latin", p, LatinScheme(config.LATIN_FONT), list(PRINTABLE))
    if name.startswith("col_b"):
        return ColourEncoding(int(name[len("col_b"):]))
    if name in LG_BANKS:
        from learned_glyphs.lg.scheme import LearnedGlyphScheme, label_of
        sc = LearnedGlyphScheme(ROOT / LG_BANKS[name], name=name)
        p = int(round(sc.height_px))
        fam = "lg_pb" if "_pb_" in name else "lg_free"
        return SchemeEncoding(name, fam, p, sc, [label_of(i) for i in range(len(sc.symbols))])
    raise KeyError(name)


def make_pages(enc: Encoding, split: str, n: int) -> list[Page]:
    """``n`` consecutive pages of ``split`` text (deterministic)."""
    need = n * enc.n_lines * (W + 8)
    text = text_stream(split, need)
    pages, pos = [], 0
    for _ in range(n):
        if pos > len(text) - enc.n_lines * W:
            pos = 0
        pg, used = enc.render(text, pos)
        pages.append(pg)
        pos += used
    return pages
