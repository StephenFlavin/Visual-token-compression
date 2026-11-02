"""Visual-BPE dictionary: the N multi-character chains that each become ONE 1.0-em glyph.

Width-aware selection (PLAN R2)
-------------------------------
Left-over text is set in ``config.CONDENSED_FONT`` (Archivo Narrow, ~0.37 em/char). Char widths are
the real per-glyph advances measured with PIL at 1000 px / 1000 (layout draws one GlyphUnit per char,
so no kerning). A token ``t`` replacing an occurrence saves ``width_em(t) - symbol_em`` em, so only
chains wider than 1.0 em are ever useful (~3+ chars).

Builder (``build``): greedy batch selection with re-segmentation on a ~6M-char random-paragraph
sample of the cleaned WikiText-103 train text.
  1. Candidates: every substring of length 2..``MAX_LEN`` (spaces allowed, never crossing a
     paragraph) with width > 1 em, count >= ``MIN_COUNT`` and potential ``count*(w-1)`` >=
     ``MIN_POTENTIAL`` em. Counted with 64-bit rolling hashes in numpy.
  2. Loop: segment the whole sample with the current dictionary by min-width DP; for every
     candidate, estimated marginal gain = sum over its occurrences whose both ends lie on segment
     boundaries of ``(current width of that span) - 1.0`` (clipped at 0). This handles overlap:
     chars already inside tokens are not counted again, and a longer chain gets credit only for the
     width it saves beyond the tokens already covering it.
  3. Take the top-k candidates by gain (k = 8 until 100 tokens, then 8 % of the current size),
     skipping any that are sub/super-strings of, or overlap by >= 2 chars with, one already picked
     in this batch (they compete for the same text; the next re-segmentation re-scores them).
  4. Repeat until N tokens. Selection order = priority order, so ``tokens[:n]`` for n < N is
     exactly the dictionary the greedy had at size n (nested experiments).

Segmentation (``Dictionary.segment``): exact DP that minimises total rendered width
(leftover chars at their condensed advance, tokens at ``symbol_em``); a token is only used when it
strictly saves width, so the result is never wider than plain condensed Latin.
"""
from __future__ import annotations

import json
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import ImageFont

from vbpe import config

MASTER_N = 3000
MAX_LEN = 20
MIN_COUNT = 5
MIN_POTENTIAL = 10.0     # em over the sample
SAMPLE_CHARS = 6_000_000
SAMPLE_SEED = 0
MEASURE_PX = 1000


@lru_cache(maxsize=8)
def _font(path: str):
    return ImageFont.truetype(path, MEASURE_PX)


@lru_cache(maxsize=100_000)
def char_width_em(ch: str, font_path: str = str(config.CONDENSED_FONT)) -> float:
    """Advance of one char in em (measured at 1000 px)."""
    return _font(str(font_path)).getlength(ch) / MEASURE_PX


def text_width_em(text: str, font_path: str | Path = config.CONDENSED_FONT) -> float:
    """Width of plain text set one GlyphUnit per char (as vbpe.layout does), in em."""
    fp = str(font_path)
    return sum(char_width_em(c, fp) for c in text)


def dict_path(n: int) -> Path:
    return config.DATA / f"dict_{n}.json"


class Dictionary:
    def __init__(self, tokens: list[str], font_path: str | Path = config.CONDENSED_FONT,
                 symbol_em: float = 1.0, meta: dict | None = None):
        if len(set(tokens)) != len(tokens):
            raise ValueError("duplicate tokens")
        if any(len(t) < 2 for t in tokens):
            raise ValueError("tokens must have length >= 2")
        self.tokens = list(tokens)
        self.font_path = str(font_path)
        self.symbol_em = float(symbol_em)
        self.meta = meta or {}
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self._lengths = sorted({len(t) for t in self.tokens})

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, s: str) -> bool:
        return s in self.index

    def prefix(self, n: int) -> "Dictionary":
        meta = dict(self.meta, n=n, prefix_of=len(self.tokens))
        if "est_gain_em" in meta:
            meta["est_gain_em"] = meta["est_gain_em"][:n]
        return Dictionary(self.tokens[:n], self.font_path, self.symbol_em, meta)

    # ------------------------------------------------------------------ widths
    def char_width(self, ch: str) -> float:
        return char_width_em(ch, self.font_path)

    def segment_width(self, seg: str) -> float:
        """Rendered width (em) of one segment: symbol_em for a token, glyph advance for a char."""
        return self.symbol_em if seg in self.index else self.char_width(seg)

    def width(self, segments: list[str]) -> float:
        return sum(self.segment_width(s) for s in segments)

    def latin_width(self, text: str) -> float:
        return text_width_em(text, self.font_path)

    def token_saving(self, token: str) -> float:
        """em saved per occurrence when ``token`` is drawn as one symbol."""
        return self.latin_width(token) - self.symbol_em

    # ------------------------------------------------------------------ segmentation
    def segment(self, text: str) -> list[str]:
        """Min-width segmentation into dictionary tokens and single chars. ''.join(...) == text."""
        n = len(text)
        if n == 0:
            return []
        cw = [self.char_width(c) for c in text]
        idx, lengths, sym = self.index, self._lengths, self.symbol_em
        best = [0.0] * (n + 1)          # best[i] = min width of text[i:]
        choice = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            b = best[i + 1] + cw[i]
            c = 1
            for L in lengths:
                if i + L > n:
                    break
                if text[i:i + L] in idx:
                    v = best[i + L] + sym
                    if v < b - 1e-9:
                        b, c = v, L
            best[i] = b
            choice[i] = c
        out, i = [], 0
        while i < n:
            out.append(text[i:i + choice[i]])
            i += choice[i]
        return out

    def encode_ids(self, text: str) -> list[int | str]:
        """Segments with tokens replaced by their priority index (chars left as str)."""
        return [self.index.get(s, s) if len(s) > 1 else s for s in self.segment(text)]

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"tokens": self.tokens, "font_path": Path(self.font_path).name,
               "symbol_em": self.symbol_em, "meta": self.meta}
        path.write_text(json.dumps(rec, indent=0))

    @classmethod
    def from_file(cls, path: str | Path) -> "Dictionary":
        rec = json.loads(Path(path).read_text())
        fp = config.FONTS / rec.get("font_path", config.CONDENSED_FONT.name)
        return cls(rec["tokens"], font_path=fp if fp.exists() else config.CONDENSED_FONT,
                   symbol_em=rec.get("symbol_em", 1.0), meta=rec.get("meta", {}))

    @classmethod
    def load(cls, n: int) -> "Dictionary":
        """data/dict_{n}.json; built on demand (as a prefix of the cached master dictionary)."""
        p = dict_path(n)
        if p.exists():
            return cls.from_file(p)
        master_n = max(n, MASTER_N)
        mp = dict_path(master_n)
        master = cls.from_file(mp) if mp.exists() else build(master_n)
        if len(master) < n:
            raise RuntimeError(f"master dictionary has only {len(master)} tokens < {n}")
        d = master.prefix(n)
        d.save(p)
        return d


# ====================================================================== builder

def train_sample(chars: int = SAMPLE_CHARS, seed: int = SAMPLE_SEED) -> str:
    """Random train paragraphs joined by '\\n' until >= ``chars`` characters."""
    from vbpe.corpus import read_paragraphs

    pars = list(read_paragraphs("train"))
    random.Random(seed).shuffle(pars)
    out, total = [], 0
    for p in pars:
        out.append(p)
        total += len(p) + 1
        if total >= chars:
            break
    return "\n".join(out)


class _Builder:
    def __init__(self, text: str, font_path: str, symbol_em: float, log=print):
        self.log = log
        self.text = text
        self.sym = symbol_em
        a = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        self.n = n = len(a)
        table = np.array([char_width_em(chr(c), font_path) if 32 <= c < 127 else 0.0
                          for c in range(128)])
        self.cw = table[a]
        self.cum = np.concatenate([[0.0], np.cumsum(self.cw)])
        nl = np.concatenate([[0], np.cumsum(a == 10)])
        P = np.uint64(1_000_003)
        h = a.astype(np.uint64) + np.uint64(1)
        self.ids: dict[int, np.ndarray] = {}
        self.cand: list[str] = []
        t0 = time.time()
        with np.errstate(over="ignore"):
            for L in range(2, MAX_LEN + 1):
                h = h[:n - L + 1] * P + a[L - 1:].astype(np.uint64) + np.uint64(1)
                w = self.cum[L:] - self.cum[:n - L + 1]
                ok = (nl[L:] - nl[:n - L + 1] == 0) & (w > symbol_em + 1e-9)
                pos = np.nonzero(ok)[0]
                if len(pos) == 0:
                    continue
                uniq, first, inv, cnt = np.unique(h[pos], return_index=True, return_inverse=True,
                                                  return_counts=True)
                wu = w[pos[first]]
                keep = (cnt >= MIN_COUNT) & (cnt * (wu - symbol_em) >= MIN_POTENTIAL)
                newid = np.full(len(uniq), -1, dtype=np.int32)
                k = int(keep.sum())
                newid[keep] = np.arange(len(self.cand), len(self.cand) + k, dtype=np.int32)
                for f in pos[first[keep]]:
                    self.cand.append(text[f:f + L])
                arr = np.full(n - L + 1, -1, dtype=np.int32)
                arr[pos] = newid[inv]
                self.ids[L] = arr
        self.ncand = len(self.cand)
        self.log(f"[dict] {self.ncand} candidates (len 2..{MAX_LEN}) from {n} chars "
                 f"in {time.time() - t0:.1f}s")

    def segment(self, in_dict: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Min-width DP over the sample. Returns (boundary mask [n+1], cum cost best[n+1])."""
        n = self.n
        ext = np.append(in_dict, False)
        endmask = np.zeros(n + 1, dtype=np.int64)
        for L, arr in self.ids.items():
            st = np.nonzero(ext[arr])[0]
            if len(st):
                endmask[st + L] |= (1 << L)
        cw = self.cw.tolist()
        em = endmask.tolist()
        best = [0.0] * (n + 1)
        choice = [1] * (n + 1)
        sym = self.sym
        bits_cache: dict[int, list[int]] = {}
        for j in range(1, n + 1):
            b = best[j - 1] + cw[j - 1]
            m = em[j]
            if m:
                bl = bits_cache.get(m)
                if bl is None:
                    bl = bits_cache[m] = [L for L in range(2, MAX_LEN + 1) if m >> L & 1]
                c = 1
                for L in bl:
                    v = best[j - L] + sym
                    if v < b - 1e-9:
                        b, c = v, L
                choice[j] = c
            best[j] = b
        B = np.zeros(n + 1, dtype=bool)
        j = n
        while j > 0:
            B[j] = True
            j -= choice[j]
        B[0] = True
        return B, np.asarray(best)

    def gains(self, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        g = np.zeros(self.ncand)
        n = self.n
        for L, arr in self.ids.items():
            m = (arr >= 0) & B[:n - L + 1] & B[L:]
            st = np.nonzero(m)[0]
            if len(st) == 0:
                continue
            v = C[st + L] - C[st] - self.sym
            pos = v > 0
            g += np.bincount(arr[st[pos]], weights=v[pos], minlength=self.ncand)
        return g


def _conflict(a: str, b: str) -> bool:
    if a in b or b in a:
        return True
    for k in range(2, min(len(a), len(b))):
        if a.endswith(b[:k]) or b.endswith(a[:k]):
            return True
    return False


def build(n: int = MASTER_N, text: str | None = None, font_path: str | Path = config.CONDENSED_FONT,
          symbol_em: float = 1.0, save: bool = True, log=print) -> Dictionary:
    """Greedy width-aware batch selection (see module docstring). Saves data/dict_{n}.json."""
    t0 = time.time()
    text = train_sample() if text is None else text
    bld = _Builder(text, str(font_path), symbol_em, log=log)
    in_dict = np.zeros(bld.ncand, dtype=bool)
    tokens: list[str] = []
    est_gain: list[float] = []
    base = float(bld.cum[-1])
    it = 0
    while len(tokens) < n:
        B, C = bld.segment(in_dict)
        g = bld.gains(B, C)
        g[in_dict] = 0
        k = min(n - len(tokens), 8 if len(tokens) < 100 else max(8, int(0.08 * len(tokens))))
        order = np.argsort(-g, kind="stable")
        batch: list[int] = []
        for ci in order[: 50 * k + 1000]:
            if g[ci] <= 0 or len(batch) >= k:
                break
            s = bld.cand[ci]
            if any(_conflict(s, bld.cand[b]) for b in batch):
                continue
            batch.append(int(ci))
        if not batch:
            log("[dict] no more positive-gain candidates")
            break
        for ci in batch:
            in_dict[ci] = True
            tokens.append(bld.cand[ci])
            est_gain.append(float(g[ci]))
        it += 1
        log(f"[dict] it {it:3d}: {len(tokens):5d} tokens, width ratio before this batch "
            f"{C[-1] / base:.4f}  (last: {bld.cand[batch[-1]]!r} {g[batch[-1]]:.0f} em) "
            f"{time.time() - t0:.0f}s")
    B, C = bld.segment(in_dict)
    meta = {"method": "greedy width-aware batch selection with re-segmentation",
            "sample_chars": bld.n, "sample_seed": SAMPLE_SEED, "max_len": MAX_LEN,
            "min_count": MIN_COUNT, "n_candidates": bld.ncand,
            "sample_width_ratio": float(C[-1] / base), "est_gain_em": est_gain,
            "build_seconds": round(time.time() - t0, 1), "n": len(tokens)}
    d = Dictionary(tokens, font_path=font_path, symbol_em=symbol_em, meta=meta)
    if save:
        d.save(dict_path(len(tokens)))
    log(f"[dict] done: {len(tokens)} tokens, sample width ratio {meta['sample_width_ratio']:.4f}, "
        f"{meta['build_seconds']}s")
    return d


# ====================================================================== K1 analytic check

def k1_stats(d: Dictionary, paragraphs: list[str] | tuple[str, ...],
             latin_font: str | Path = config.LATIN_FONT) -> dict:
    """Equal-em width statistics of the substitution encoding on held-out text.

    Returns chars, segments, seg_ratio (segments/chars), width ratios vs condensed and regular
    Latin, token coverage (fraction of source chars inside tokens), per-token usage counts.
    """
    from collections import Counter

    chars = segs = covered = 0
    w_sub = w_cond = w_lat = 0.0
    use: Counter = Counter()
    for p in paragraphs:
        sg = d.segment(p)
        chars += len(p)
        segs += len(sg)
        for s in sg:
            if len(s) > 1:
                covered += len(s)
                use[s] += 1
        w_sub += d.width(sg)
        w_cond += d.latin_width(p)
        w_lat += text_width_em(p, latin_font)
    return {"n": len(d), "chars": chars, "segments": segs, "seg_ratio": segs / chars,
            "width_sub": w_sub, "width_condensed": w_cond, "width_latin": w_lat,
            "ratio_vs_condensed": w_sub / w_cond, "ratio_vs_latin": w_sub / w_lat,
            "coverage": covered / chars, "token_use": use,
            "unused_tokens": sum(1 for t in d.tokens if use[t] == 0),
            "em_gain_vs_condensed": (w_cond / w_sub) ** 0.5,
            "em_gain_vs_latin": (w_lat / w_sub) ** 0.5}
