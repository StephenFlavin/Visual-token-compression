"""Phase-2 data: source lines -> rendered single-line images + CTC label sequences.

Fairness rules (identical for every scheme)
-------------------------------------------
* Source text: the same lines for every scheme (``vbpe.corpus.iter_lines``; train lines from the
  'train' split, test = first ``n_test`` lines of the 'val' split, val (checkpoint selection) =
  the next ``n_val`` lines).
* Rendering: ``vbpe.layout.render_line(units, em', LayoutParams(margin=2, supersample=4))``.
  The per-line random draws (em jitter, sub-pixel phase) are seeded by (seed, split, line index)
  only, so line *i* gets the same jitter/phase under every scheme.
* Train augmentation:  em' = em * (1 + U(-0.02, 0.02));  horizontal sub-pixel phase
  phi ~ U(0, 1) px, implemented as an invisible leading ``PhaseOffset`` unit of advance phi.
  Pillow snaps every glyph origin to an integer pixel, so the phase is only *sub*-pixel because
  we render with ``supersample=4`` (glyphs drawn at 4x then BOX-downsampled -> phase resolution
  0.25 px; it also gives area-correct antialiasing that does not depend on each font's hinting
  quality at tiny sizes, which would otherwise confound scheme comparisons).
  Photometric augmentation (contrast jitter, gaussian noise) is applied per batch in
  ``augment_batch`` (cheap, fresh every epoch).
* Mixed-em mode (one model per scheme for all ems): each train line's nominal em is drawn
  uniformly from the em list with the same per-line rng (identical draws for every scheme); val
  and test sets are rendered separately at each em (the same caches as single-em mode).
* Eval (val/test) lines: exact nominal em, seeded random phase, no photometric noise.
* px_per_char of a line = line_spacing * em' * sum(unit advances) / len(normalize(line))
  (content area, margins excluded - consistent with page-level px^2/char).
* CTC frames: the reader's BiLSTM runs over W // s columns, s = 2 px for every scheme and em
  (``col_stride_for_em``; a 4-px stride at em >= 8 was tried and learned more slowly - it packs
  several narrow glyphs into one column, which would also penalise dense schemes), and each
  column is expanded into E
  CTC frames, T = (W // s) * E. E is chosen per (scheme, em) by ONE rule for all schemes: the
  smallest integer with T >= 2L+1 for EVERY rendered sample (train, val and test).
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from vbpe import config
from vbpe.layout import LayoutParams, Unit, render_line
from vbpe.metrics import normalize

CODE_VERSION = 1                     # bump to invalidate caches when rendering code changes
CACHE_DIR = config.RESULTS / "phase2" / "cache"
MAX_EXPAND = 16
SPLIT_IDS = {"train": 0, "val": 1, "test": 2}
FALLBACK_CORPUS = "FALLBACK:pydoc_topics"


# ----------------------------------------------------------------------------- source text
def _chop(text: str, rng: random.Random, min_chars: int, max_chars: int) -> list[str]:
    words, out, cur = normalize(text).split(" "), [], ""
    target = rng.randint(min_chars, max_chars)
    for w in words:
        if not w:
            continue
        if len(w) > max_chars:
            continue
        cand = f"{cur} {w}" if cur else w
        if len(cand) > target and len(cur) >= min_chars:
            out.append(cur)
            cur, target = w, rng.randint(min_chars, max_chars)
        elif len(cand) > max_chars:
            cur = w
        else:
            cur = cand
    if len(cur) >= min_chars:
        out.append(cur)
    return out


def fallback_lines(split: str, min_chars: int = 30, max_chars: int = 70, seed: int = 0) -> list[str]:
    """SMOKE-TEST ONLY fallback when ``vbpe.corpus`` / WikiText is unavailable.

    Built-in sample text = the Python docs' help topics (``pydoc_data.topics``, ~560k chars,
    ASCII-filtered). ~1/8 of topics go to 'val'. Not the Phase-2 corpus: rows produced with it are
    tagged ``corpus=FALLBACK:pydoc_topics``.
    """
    from pydoc_data.topics import topics

    rng = random.Random(seed)
    lines: list[str] = []
    for key in sorted(topics):
        is_val = int(hashlib.md5(key.encode()).hexdigest(), 16) % 8 == 0
        if is_val != (split == "val"):
            continue
        txt = "".join(c for c in topics[key] if 32 <= ord(c) < 127 or c in "\n\t")
        lines.extend(_chop(txt, rng, min_chars, max_chars))
    rng.shuffle(lines)
    return lines


def source_lines(split: str, n: int, skip: int = 0, seed: int = 0,
                 allow_fallback: bool = False) -> tuple[list[str], str]:
    """First ``n`` lines (after ``skip``) of ``iter_lines(split)``. Returns (lines, corpus_name)."""
    try:
        from vbpe.corpus import iter_lines  # WP-A
        it = iter_lines(split, min_chars=30, max_chars=70, seed=seed)
        out = []
        for i, line in enumerate(it):
            if i < skip:
                continue
            if len(out) >= n:
                break
            out.append(normalize(line))
        if len(out) < n:
            print(f"[crnn.data] warning: corpus split {split!r} gave only {len(out)} < {n} lines",
                  file=sys.stderr)
        return out, "wikitext103"
    except (ImportError, FileNotFoundError, OSError) as e:
        if not allow_fallback:
            raise RuntimeError(f"vbpe.corpus.iter_lines unavailable ({e!r}); Phase-2 needs the "
                               "WikiText corpus (use --smoke to run on the built-in fallback)") from e
        print(f"[crnn.data] *** using {FALLBACK_CORPUS} fallback text ({e!r}) - smoke only ***",
              file=sys.stderr)
        lines = fallback_lines(split, seed=seed)
        return lines[skip:skip + n], FALLBACK_CORPUS


# ----------------------------------------------------------------------------- schemes
def resolve_scheme(name: str):
    """``get_scheme`` plus the synthetic stub schemes used for tests (``stub_*``)."""
    if name.startswith("stub_"):
        from vbpe.crnn.stubs import make_stub_scheme
        return make_stub_scheme(name)
    if name.startswith("lg:"):  # learned glyph bank as a bitmap-font scheme (learned_glyphs/lg/scheme.py)
        from learned_glyphs.lg.scheme import make_lg_scheme
        return make_lg_scheme(name)
    from vbpe.schemes import get_scheme
    return get_scheme(name)


# ----------------------------------------------------------------------------- rendering
class PhaseOffset(Unit):
    """Invisible leading unit that shifts everything right by ``phase`` px (at render em)."""

    __slots__ = ("phase", "em", "source", "labels", "is_space")

    def __init__(self, phase: float, em: float):
        self.phase, self.em = float(phase), float(em)
        self.source, self.labels, self.is_space = "", (), False

    def advance(self, em: float) -> float:
        return self.phase * em / self.em      # scales with supersampling

    def draw(self, d, x, baseline, em) -> None:
        pass


@dataclass(frozen=True)
class RenderParams:
    margin: int = 2
    supersample: int = 4
    line_spacing: float = 1.2
    em_jitter: float = 0.02      # train only
    phase: bool = True           # random sub-pixel x phase (train + eval, seeded)

    def layout(self) -> LayoutParams:
        return LayoutParams(margin=self.margin, line_spacing=self.line_spacing,
                            supersample=self.supersample)

    def max_height(self, em: float) -> int:
        return int(round(self.line_spacing * em * (1 + self.em_jitter))) + 2 * self.margin + 1


def line_rng(seed: int, split: str, idx: int) -> np.random.Generator:
    return np.random.default_rng([seed, SPLIT_IDS[split], idx])


def render_sample(scheme, line: str, em, rp: RenderParams, rng: np.random.Generator,
                  train: bool) -> tuple[np.ndarray, tuple[str, ...], float, float]:
    """-> (uint8 image HxW, white=255; label tuple; px_per_char; em used).

    ``em`` is a number, or a tuple of ems for mixed-em training: the nominal em is then drawn
    uniformly from it with the same per-line rng (after jitter and phase, so line i gets the same
    jitter/phase as in single-em mode and the same em draw under every scheme).
    """
    units = scheme.encode(line)
    j = rng.uniform(-rp.em_jitter, rp.em_jitter)        # always drawn: keeps streams aligned
    phase = rng.uniform(0.0, 1.0)
    if isinstance(em, (tuple, list)):
        em = em[int(rng.integers(len(em)))]
    e = em * (1 + j) if train else float(em)
    pre = [PhaseOffset(phase, e)] if rp.phase else []
    img = np.asarray(render_line(pre + units, e, rp.layout()), dtype=np.uint8)
    labels = tuple(lab for u in units for lab in u.labels)
    content = sum(u.advance(e) for u in units)
    ppc = rp.line_spacing * e * content / max(1, len(normalize(line)))
    return img, labels, ppc, e


_W_SCHEME = None


def _worker_init(scheme_name: str) -> None:
    global _W_SCHEME
    _W_SCHEME = resolve_scheme(scheme_name)


def _render_chunk(args):
    start, lines, em, rp, seed, split, train = args
    out = []
    for k, line in enumerate(lines):
        out.append(render_sample(_W_SCHEME, line, em, rp, line_rng(seed, split, start + k), train))
    return out


# ----------------------------------------------------------------------------- dataset
class LineSet:
    """Pre-rendered lines. Images are stored concatenated along width: ``images[:, off[i]:off[i+1]]``
    (uint8, white background, top-aligned and padded at the bottom to a common height H; the
    rendered height of each line is in ``heights`` so batches can be cropped)."""

    def __init__(self, images: np.ndarray, offsets: np.ndarray, labels: list[tuple[str, ...]],
                 texts: list[str], ppc: np.ndarray, ems: np.ndarray, info: dict,
                 heights: np.ndarray | None = None):
        self.images, self.offsets = images, offsets
        self.labels, self.texts, self.ppc, self.ems, self.info = labels, texts, ppc, ems, info
        self.widths = np.diff(offsets).astype(np.int64)
        self.heights = (np.full(len(texts), images.shape[0], dtype=np.int64) if heights is None
                        else np.asarray(heights, dtype=np.int64))
        self.label_lens = np.array([len(x) for x in labels], dtype=np.int64)

    @property
    def height(self) -> int:
        return int(self.images.shape[0])

    def __len__(self) -> int:
        return len(self.texts)

    def image(self, i: int) -> np.ndarray:
        return self.images[:, self.offsets[i]:self.offsets[i + 1]]

    # -- persistence
    def save(self, d: Path) -> None:
        """Write to a private temp dir, then rename it into place (safe with concurrent writers;
        the first complete copy wins)."""
        tmpd = d.with_name(f"{d.name}.tmp{os.getpid()}")
        tmpd.mkdir(parents=True, exist_ok=True)
        np.save(tmpd / "images.npy", self.images)
        np.save(tmpd / "offsets.npy", self.offsets)
        np.save(tmpd / "ppc.npy", self.ppc)
        np.save(tmpd / "ems.npy", self.ems)
        np.save(tmpd / "heights.npy", self.heights)
        meta = {"info": self.info, "texts": self.texts, "labels": ["\x1f".join(x) for x in self.labels]}
        (tmpd / "meta.json").write_text(json.dumps(meta))
        try:
            tmpd.rename(d)
        except OSError:                       # another process finished first
            shutil.rmtree(tmpd, ignore_errors=True)

    @classmethod
    def load(cls, d: Path, mmap: bool = False) -> "LineSet":
        meta = json.loads((d / "meta.json").read_text())
        labels = [tuple(s.split("\x1f")) if s else () for s in meta["labels"]]
        hp = d / "heights.npy"
        return cls(np.load(d / "images.npy", mmap_mode="r" if mmap else None), np.load(d / "offsets.npy"),
                   labels, meta["texts"], np.load(d / "ppc.npy"), np.load(d / "ems.npy"), meta["info"],
                   np.load(hp) if hp.exists() else None)


def _em_name(em) -> str:
    return "mixed" + "-".join(f"{e:g}" for e in em) if isinstance(em, (tuple, list)) else f"em{em:g}"


def _cache_key(scheme: str, em, split: str, n: int, skip: int, seed: int, train: bool,
               rp: RenderParams, corpus_tag: str) -> str:
    em = [float(e) for e in em] if isinstance(em, (tuple, list)) else float(em)
    blob = json.dumps([CODE_VERSION, scheme, em, split, n, skip, seed, train, asdict(rp),
                       corpus_tag], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def build_lineset(scheme: str, em, split: str, n: int, *, skip: int = 0, seed: int = 0,
                  train: bool = False, rp: RenderParams = RenderParams(), workers: int = 2,
                  cache_dir: Path | None = CACHE_DIR, allow_fallback: bool = False,
                  tag: str | None = None, verbose: bool = True) -> LineSet:
    """Render (or load from cache) ``n`` lines of ``split`` for ``scheme`` at ``em``
    (a number, or a tuple of ems: each line's em drawn uniformly from it - mixed-em training).

    ``split`` is the corpus split ('train'/'val'); ``tag`` names the subset for seeding/caching
    ('train', 'val', 'test'); defaults to ``split``.
    """
    tag = tag or split
    if isinstance(em, list):
        em = tuple(em)
    lines, corpus = source_lines(split, n, skip=skip, seed=seed, allow_fallback=allow_fallback)
    text_hash = hashlib.sha1("\n".join(lines).encode()).hexdigest()[:12]
    key = _cache_key(scheme, em, tag, n, skip, seed, train, rp, f"{corpus}:{text_hash}")
    d = cache_dir / f"{scheme}_{_em_name(em)}_{tag}_{n}_{key}" if cache_dir else None
    # caches written before per-line heights were stored load at the padded max height, which
    # differs from fresh (cropped) train batches by a row and silently wrecked latin em4 -> re-render
    if d is not None and (d / "meta.json").exists() and (d / "heights.npy").exists():
        return LineSet.load(d)
    t0 = time.time()
    chunk = max(16, math.ceil(len(lines) / (workers * 8)))
    jobs = [(i, lines[i:i + chunk], em, rp, seed, tag, train) for i in range(0, len(lines), chunk)]
    if workers > 1 and len(jobs) > 1:
        ctx = mp.get_context("fork" if sys.platform == "linux" else "spawn")
        with ctx.Pool(workers, initializer=_worker_init, initargs=(scheme,)) as pool:
            parts = pool.map(_render_chunk, jobs)
    else:
        _worker_init(scheme)
        parts = [_render_chunk(j) for j in jobs]
    samples = [s for p in parts for s in p]
    H = max(rp.max_height(max(em) if isinstance(em, tuple) else em),
            max(s[0].shape[0] for s in samples))
    widths = np.array([s[0].shape[1] for s in samples], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    images = np.full((H, int(offsets[-1])), 255, dtype=np.uint8)
    for (img, *_), o in zip(samples, offsets[:-1]):
        images[: img.shape[0], o:o + img.shape[1]] = img
    info = {"scheme": scheme, "em": list(em) if isinstance(em, tuple) else float(em), "split": split, "tag": tag, "n": len(samples),
            "seed": seed, "train": train, "render": asdict(rp), "corpus": corpus,
            "render_time_s": round(time.time() - t0, 2), "workers": workers}
    ls = LineSet(images, offsets, [s[1] for s in samples], lines,
                 np.array([s[2] for s in samples], dtype=np.float32),
                 np.array([s[3] for s in samples], dtype=np.float32), info,
                 np.array([s[0].shape[0] for s in samples], dtype=np.int64))
    del samples, parts
    if d is not None:
        ls.save(d)
    if verbose:
        print(f"[crnn.data] rendered {scheme} {_em_name(em)} {tag}: {len(ls)} lines, H={H}, "
              f"mean W={widths.mean():.0f}, {info['render_time_s']}s ({workers} workers)",
              file=sys.stderr)
    return ls


# ----------------------------------------------------------------------------- vocab
class Vocab:
    BLANK, UNK = 0, 1

    def __init__(self, symbols: Sequence[str]):
        self.itos = ["<blank>", "<unk>"] + list(symbols)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def from_labels(cls, label_seqs) -> "Vocab":
        counts: dict[str, int] = {}
        for seq in label_seqs:
            for lab in seq:
                counts[lab] = counts.get(lab, 0) + 1
        return cls(sorted(counts, key=lambda s: (-counts[s], s)))

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, labels: Sequence[str]) -> list[int]:
        return [self.stoi.get(lab, self.UNK) for lab in labels]

    def decode(self, ids) -> list[str]:
        return [self.itos[i] for i in ids if i > self.UNK]

    def to_json(self) -> list[str]:
        return self.itos[2:]


# ----------------------------------------------------------------------------- CTC frame rule
def col_stride_for_em(em: float) -> int:
    """Pixel stride of the BiLSTM columns. Fixed at 2 px (uniform per-pixel processing for every
    scheme and em). The model also supports 4 (cheaper; slower to learn in the smoke test)."""
    return 2


def n_frames(width, col_stride: int, expand: int):
    """CTC frames for an input of ``width`` px: W // col_stride LSTM columns, each expanded into
    ``expand`` output frames (must match ``CRNN.output_lengths``)."""
    return (np.asarray(width) // col_stride) * expand


def choose_expand(widths, label_lens, col_stride: int, max_expand: int = MAX_EXPAND) -> int:
    """Smallest expansion E with T = (W // s) * E >= 2L+1 for EVERY sample (rule for all schemes)."""
    w, L = np.asarray(widths), np.asarray(label_lens)
    for e in range(1, max_expand + 1):
        if np.all(n_frames(w, col_stride, e) >= 2 * L + 1):
            return e
    raise ValueError(f"no expansion <= {max_expand} satisfies T >= 2L+1 "
                     f"(worst W/L = {np.min(w / np.maximum(L, 1)):.2f})")


# ----------------------------------------------------------------------------- batching
def bucket_batches(widths: np.ndarray, batch: int, rng: np.random.Generator,
                   bucket_mult: int = 50) -> Iterator[np.ndarray]:
    """Infinite stream of index batches; within chunks of batch*bucket_mult, sort by width."""
    n = len(widths)
    while True:
        perm = rng.permutation(n)
        batches = []
        for s in range(0, n, batch * bucket_mult):
            c = perm[s:s + batch * bucket_mult]
            c = c[np.argsort(widths[c], kind="stable")]
            batches += [c[i:i + batch] for i in range(0, len(c), batch) if len(c[i:i + batch]) == batch
                        or n < batch]
        for i in rng.permutation(len(batches)):
            yield batches[i]


def eval_batches(widths: np.ndarray, batch: int) -> list[np.ndarray]:
    order = np.argsort(widths, kind="stable")
    return [order[i:i + batch] for i in range(0, len(order), batch)]


def collate(ls: LineSet, idx: np.ndarray, vocab: Vocab | None = None):
    """-> x (B,1,H,Wmax) float32 ink=1 bg=0, widths (B,), [targets (sum L,), target_lens (B,)]."""
    import torch

    idx = np.asarray(idx)
    widths = ls.widths[idx]
    Wm, Hb = int(widths.max()), int(ls.heights[idx].max())      # crop height to the batch max
    x = np.zeros((len(idx), 1, Hb, Wm), dtype=np.float32)
    for b, i in enumerate(idx):
        x[b, 0, :, : widths[b]] = (255 - ls.image(int(i))[:Hb]) / 255.0
    out = [torch.from_numpy(x), torch.from_numpy(widths)]
    if vocab is not None:
        tg = [vocab.encode(ls.labels[int(i)]) for i in idx]
        out.append(torch.tensor([t for s in tg for t in s], dtype=torch.long))
        out.append(torch.tensor([len(s) for s in tg], dtype=torch.long))
    return out


def augment_batch(x, gen, contrast=(0.7, 1.0), noise_max=0.06):
    """Photometric augmentation (train only, same for all schemes): per-sample ink contrast
    c ~ U(contrast), background lift b ~ U(0, 0.05), gaussian noise sigma ~ U(0, noise_max)."""
    import torch

    B = x.shape[0]
    c = torch.empty(B, 1, 1, 1).uniform_(*contrast, generator=gen)
    b = torch.empty(B, 1, 1, 1).uniform_(0.0, 0.05, generator=gen)
    s = torch.empty(B, 1, 1, 1).uniform_(0.0, noise_max, generator=gen)
    x = b + x * (c - b) + s * torch.randn(x.shape, generator=gen)
    return x.clamp_(0.0, 1.0)
