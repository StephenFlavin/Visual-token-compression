"""Bridge (E3): a learned (or fixed) GlyphBank as a vbpe ``Scheme`` -- a bitmap font.

``LearnedGlyphScheme(bank, alphabet)`` renders text through the unchanged vbpe layout engine, so a
bank can be scored by the Phase-2 pipeline (``experiments/phase2_train.py``) exactly like
latin / subst, at any em.

Units and labels
  ``encode(text)`` -> one ``BankGlyphUnit`` per symbol id of ``Alphabet.encode(text)``. Each unit's
  ``source`` is the symbol string (a char or a dictionary token) and its single label is a
  private-use code point ``U+E000 + id``, so Phase-2 CTC treats every glyph as ONE class (like
  subst). ``decode`` maps those code points back to symbol strings (other chars pass through).
  The pure-space symbol ``" "`` is ``is_space`` (droppable at line ends) but its learned ink, if
  any, is drawn like every other glyph so that training pixels are reproduced.

Geometry (bank = cells of height ``height_ss`` at supersampling ``ss``; its line pitch is
``height_px = height_ss / ss``)
  * LG1 convention (lg/glyphs.py): a bank is drawn for a reference em ``em_ref = bank.em``
    (default ``height_px / 1.2``); the cell is the line box (pitch ``height_px``) and the baseline
    sits ``0.9 * em_ref * ss`` ss-rows below the cell top (``glyphs.baseline_ss`` =
    ``LayoutParams.first_baseline`` minus the margin).
  * At layout em ``E`` the bank is scaled uniformly by ``s = E / (em_ref * ss)`` drawing px per
    ss px: cell height = ``height_px / em_ref * E`` (= 1.2 E for learned banks), advance =
    ``width_ss * s``, cell top = ``baseline - 0.9 E``. For learned banks the cell therefore
    occupies exactly one line pitch ``[baseline - 0.9 E, baseline + 0.3 E]`` -- 0.05 E beyond the
    layout contract's em box on each side, but never outside the line (render_line's canvas /
    page pitch), so lines can't overlap. A font bank (from_font) reproduces vbpe's glyph advances
    and baseline at every em.
  * Native em = ``em_ref``: at the Phase-2 supersampled resolution (ss=4 -> drawing em =
    4 * em_ref) the scale is exactly 1, all cell origins are integers, and the drawn canvas equals
    ``lg.compose`` of the bank's glyphs pixel for pixel (see tests/test_transfer.py).
  * Other ems: exact area resampling of the binary bitmap onto the (fractional) target
    rectangle; x/y origins are quantised to 1/4 drawing px (1/16 px after Phase-2 BOX-downsampling).
    Masks are cached per (symbol, em, fx, fy).

Registry name (for the parent Phase-2 pipeline): ``lg:<key>[@<sha8>]`` where key is the bank
path relative to ``learned_glyphs/results`` with ``/`` written as ``+`` and a trailing
``bank.pt`` optional (``lg:e1_learned_n300_h5`` -> learned_glyphs/results/e1_learned_n300_h5/bank.pt).
Names must not contain ``/`` because Phase 2 uses the scheme name in cache / log / checkpoint
file names. ``@sha8`` (first 8 hex chars of the bank file's sha1) is verified on load so cached
Phase-2 renders are content-addressed. ``make_lg_scheme(name)`` resolves such names; the parent
hook is a 3-line addition to ``vbpe.crnn.data.resolve_scheme`` (``BRIDGE_PATCH`` below);
``install_phase2_hook()`` applies the same hook at runtime, so ``lg.transfer.bridge_phase2`` needs
no parent edit.
"""
from __future__ import annotations

import hashlib
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve()
LG_ROOT = _HERE.parent.parent            # learned_glyphs/
REPO_ROOT = LG_ROOT.parent               # visual-bpe/
RESULTS = LG_ROOT / "results"
if str(REPO_ROOT) not in sys.path:     # vbpe (normally already importable)
    sys.path.append(str(REPO_ROOT))

from vbpe.layout import Unit  # noqa: E402
from vbpe.schemes.base import Scheme  # noqa: E402

PUA_BASE = 0xE000
PUA_MAX = 0xF8FF
LINE_SPACING = 1.2
TOP_EM = 0.9                             # vbpe.layout / lg.glyphs: baseline 0.9 em below the line-box top
_Q = 4                                   # sub-pixel quantisation of origins (1/4 drawing px)


def label_of(i: int) -> str:
    c = PUA_BASE + int(i)
    if c > PUA_MAX:
        raise ValueError(f"symbol id {i} exceeds the BMP private-use area")
    return chr(c)


def bank_sha8(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:8]


# ----------------------------------------------------------------------------- resampling
def _area_matrix(n_src: int, scale: float, frac: float, n_out: int) -> np.ndarray:
    """(n_out, n_src): overlap (in output px) of source cell k = [frac + k*s, frac + (k+1)*s) with
    output pixel i = [i, i+1). ``out = Ry @ B @ Rx.T`` = ink coverage fraction per output pixel."""
    e = frac + np.arange(n_src + 1) * scale
    i = np.arange(n_out)[:, None]
    lo = np.maximum(e[None, :-1], i)
    hi = np.minimum(e[None, 1:], i + 1)
    return np.clip(hi - lo, 0.0, None)


def resample_cell(bitmap: np.ndarray, scale: float, fx: float = 0.0, fy: float = 0.0) -> np.ndarray:
    """Area-resample ``bitmap`` (h, w) in [0, 1] by ``scale`` with its top-left corner at (fx, fy)
    inside the first output pixel -> coverage array (ceil(fy + h*s), ceil(fx + w*s)) in [0, 1]."""
    h, w = bitmap.shape
    H = max(1, int(math.ceil(fy + h * scale - 1e-9)))
    W = max(1, int(math.ceil(fx + w * scale - 1e-9)))
    if w == 0 or h == 0:
        return np.zeros((H, W))
    if scale == 1.0 and fx == 0.0 and fy == 0.0:
        return bitmap.astype(np.float64)
    ry = _area_matrix(h, scale, fy, H)
    rx = _area_matrix(w, scale, fx, W)
    return np.clip(ry @ bitmap @ rx.T, 0.0, 1.0)


def _qfrac(v: float) -> tuple[int, float]:
    i = math.floor(v)
    f = round((v - i) * _Q) / _Q
    if f >= 1.0:
        i, f = i + 1, 0.0
    return i, f


# ----------------------------------------------------------------------------- bank access
def _load_bank(bank):
    from .glyphs import GlyphBank  # WP-LG1
    if isinstance(bank, (str, Path)):
        return GlyphBank.load(str(bank))
    return bank


def bank_bitmaps(bank, hard: bool = True) -> tuple[list[np.ndarray], int, int, dict]:
    """GlyphBank -> (per-symbol float bitmaps (height_ss, width_ss_i) in [0,1], height_ss, ss, meta)."""
    import torch

    with torch.no_grad():
        was = bank.training
        bank.eval()
        ink = bank(tau=1.0, hard=hard).detach().float().cpu().numpy()
        bank.train(was)
    widths = [int(w) for w in bank.widths_ss.tolist()]
    hs = int(bank.height_ss)
    maps = [np.ascontiguousarray(ink[i, :hs, :w]) for i, w in enumerate(widths)]
    meta = {}
    for k in ("em", "symbols", "n_tokens", "height_px"):
        v = getattr(bank, k, None)
        if v is None and isinstance(getattr(bank, "meta", None), dict):
            v = bank.meta.get(k)
        if v is not None:
            meta[k] = v
    return maps, hs, int(bank.ss), meta


# ----------------------------------------------------------------------------- units
class BankGlyphUnit(Unit):
    __slots__ = ("sid", "scheme", "source", "labels", "is_space")

    def __init__(self, sid: int, scheme: "LearnedGlyphScheme"):
        self.sid = int(sid)
        self.scheme = scheme
        self.source = scheme.symbols[self.sid]
        self.labels = (label_of(self.sid),)
        self.is_space = self.source.isspace()

    def advance(self, em: float) -> float:
        return self.scheme.adv_em[self.sid] * em

    def draw(self, d: ImageDraw.ImageDraw, x: float, baseline: float, em: float) -> None:
        sc = self.scheme
        if sc.empty[self.sid]:
            return
        top = baseline - sc.top_em * em
        ix, fx = _qfrac(x)
        iy, fy = _qfrac(top)
        d.bitmap((ix, iy), sc.mask(self.sid, em, fx, fy), fill=0)


class LearnedGlyphScheme(Scheme):
    """Bitmap-font Scheme from a GlyphBank (path / instance) or raw bitmaps (``from_bitmaps``).

    ``alphabet``: an ``Alphabet`` (lg/alphabet.py), an int (n_tokens), or None (the bank's saved
    ``symbols``; else n_symbols - 95 tokens). ``hard``: binarise learned glyphs (straight-through
    forward value); fixed font banks are returned as stored coverage by GlyphBank anyway.
    ``em_ref``: em the bank is drawn for (default ``bank.em``, else ``height_px / 1.2``).
    """

    def __init__(self, bank=None, alphabet=None, *, hard: bool = True, name: str | None = None,
                 em_ref: float | None = None, top_em: float = TOP_EM, _bitmaps=None):
        if _bitmaps is not None:
            maps, hs, ss, meta = _bitmaps
            self.bank_path = None
        else:
            self.bank_path = str(bank) if isinstance(bank, (str, Path)) else None
            maps, hs, ss, meta = bank_bitmaps(_load_bank(bank), hard=hard)
        self.alphabet = self._resolve_alphabet(alphabet, len(maps), meta)
        self.symbols = list(self.alphabet.symbols)
        if len(self.symbols) != len(maps):
            raise ValueError(f"alphabet has {len(self.symbols)} symbols, bank has {len(maps)}")
        self.bitmaps = [np.clip(np.asarray(m, dtype=np.float64), 0.0, 1.0) for m in maps]
        self.height_ss, self.ss = int(hs), int(ss)
        self.height_px = self.height_ss / self.ss
        self.em_ref = float(em_ref or meta.get("em") or self.height_px / LINE_SPACING)
        self.top_em = float(top_em)
        self.adv_em = [m.shape[1] / (self.ss * self.em_ref) for m in self.bitmaps]
        self.empty = [not bool(m.any()) for m in self.bitmaps]
        self.hard = hard
        self.name = name or (f"lg:{Path(self.bank_path).parent.name}" if self.bank_path else "lg")
        self._lab = {label_of(i): s for i, s in enumerate(self.symbols)}
        self._mask = lru_cache(maxsize=200_000)(self._mask_uncached)

    @classmethod
    def from_bitmaps(cls, bitmaps, height_ss: int, ss: int = 4, alphabet=None, **kw):
        """Raw per-symbol bitmaps (list of (height_ss, w_i) arrays in [0,1]) -> scheme."""
        return cls(alphabet=alphabet, _bitmaps=(list(bitmaps), height_ss, ss, {}), **kw)

    @staticmethod
    def _resolve_alphabet(alphabet, n_symbols: int, meta: dict):
        from .alphabet import N_CHARS, Alphabet  # WP-LG2
        if alphabet is None:
            syms = meta.get("symbols")
            if syms is not None:
                syms = list(syms)
                a = Alphabet.build(len(syms) - N_CHARS)
                return a if a.symbols == syms else Alphabet(syms[N_CHARS:])
            alphabet = int(meta.get("n_tokens", n_symbols - N_CHARS))
        if isinstance(alphabet, int):
            return Alphabet.build(alphabet)
        return alphabet

    # ---- geometry
    @property
    def native_em(self) -> float:
        """em at which one bank ss-pixel == one drawing pixel at supersampling ``ss``."""
        return self.em_ref

    def scale(self, em: float) -> float:
        """Drawing px per bank (ss) px at drawing em ``em``."""
        return em / (self.em_ref * self.ss)

    def ink_box_em(self) -> tuple[float, float]:
        """(top, bottom) of the cell relative to the baseline, in em (+ = below)."""
        return -self.top_em, self.height_px / self.em_ref - self.top_em

    def _mask_uncached(self, sid: int, em: float, fx: float, fy: float) -> Image.Image:
        cov = resample_cell(self.bitmaps[sid], self.scale(em), fx, fy)
        return Image.fromarray(np.round(cov * 255).astype(np.uint8), mode="L")

    def mask(self, sid: int, em: float, fx: float = 0.0, fy: float = 0.0) -> Image.Image:
        return self._mask(int(sid), round(float(em), 4), float(fx), float(fy))

    # ---- Scheme API
    def encode(self, text: str) -> list[Unit]:
        return [BankGlyphUnit(i, self) for i in self.alphabet.encode(text)]

    def decode(self, transcription: str) -> str:
        lab = self._lab
        return "".join(lab.get(c, c) for c in transcription)

    def encode_ids(self, text: str) -> list[int]:
        return list(self.alphabet.encode(text))


# ----------------------------------------------------------------------------- registry names
def bank_key(path: str | Path) -> str:
    """Bank path -> registry key (relative to learned_glyphs/results, '/' -> '+')."""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(RESULTS.resolve())
    except ValueError as e:
        raise ValueError(f"bridge banks must live under {RESULTS} (got {p})") from e
    parts = list(rel.parts)
    if parts[-1] == "bank.pt":
        parts = parts[:-1]
    key = "+".join(parts)
    if any(c in key for c in "/\\ ") or not key:
        raise ValueError(f"cannot make a registry key from {rel}")
    return key


def scheme_name_for(path: str | Path, with_hash: bool = True) -> str:
    name = f"lg:{bank_key(path)}"
    return f"{name}@{bank_sha8(path)}" if with_hash else name


def resolve_bank_name(name: str) -> tuple[Path, str | None]:
    if not name.startswith("lg:"):
        raise KeyError(f"not a learned-glyph scheme name: {name!r}")
    key, _, sha = name[3:].partition("@")
    p = RESULTS.joinpath(*key.split("+"))
    if p.is_dir() or p.suffix != ".pt":
        p = p / "bank.pt"
    if not p.exists():
        raise FileNotFoundError(f"{name!r}: no bank at {p}")
    return p, (sha or None)


@lru_cache(maxsize=8)
def make_lg_scheme(name: str) -> LearnedGlyphScheme:
    """``lg:<key>[@<sha8>]`` -> LearnedGlyphScheme (alphabet from bank metadata / its size)."""
    p, sha = resolve_bank_name(name)
    if sha is not None and bank_sha8(p) != sha:
        raise ValueError(f"{name!r}: bank {p} changed (sha {bank_sha8(p)} != {sha})")
    return LearnedGlyphScheme(p, name=name)


def phase2_hook_installed() -> bool:
    """True if ``vbpe.crnn.data.resolve_scheme`` resolves ``lg:`` names (parent patch or runtime hook)."""
    import inspect

    from vbpe.crnn import data as D
    if getattr(D.resolve_scheme, "_lg_hook", False):
        return True
    try:
        return '"lg:"' in inspect.getsource(D.resolve_scheme)
    except (OSError, TypeError):
        return False


def install_phase2_hook() -> None:
    """Runtime equivalent of the parent patch (BRIDGE_PATCH): make the unchanged Phase-2 data /
    training code resolve ``lg:`` scheme names in THIS process (and forked render workers).
    Idempotent; a no-op when the parent already has the hook."""
    from vbpe.crnn import data as D
    if phase2_hook_installed():
        return
    orig = D.resolve_scheme

    def resolve_scheme(name: str):
        if name.startswith("lg:"):
            return make_lg_scheme(name)
        return orig(name)

    resolve_scheme._lg_hook = True
    resolve_scheme.__doc__ = orig.__doc__
    D.resolve_scheme = resolve_scheme


BRIDGE_PATCH = '''\
--- a/vbpe/crnn/data.py
+++ b/vbpe/crnn/data.py
@@ def resolve_scheme(name: str):
     """``get_scheme`` plus the synthetic stub schemes used for tests (``stub_*``)."""
     if name.startswith("stub_"):
         from vbpe.crnn.stubs import make_stub_scheme
         return make_stub_scheme(name)
+    if name.startswith("lg:"):         # learned glyph bank: lg:<key>[@sha8] (learned_glyphs/lg/scheme.py)
+        from learned_glyphs.lg.scheme import make_lg_scheme
+        return make_lg_scheme(name)
     from vbpe.schemes import get_scheme
     return get_scheme(name)
'''
