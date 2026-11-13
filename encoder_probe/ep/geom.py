"""Round 2: S x S pages (any page size) for the frozen-encoder probe, with a text budget that never repeats pages.

Round 1 (pages.py, rgb_overlay.py) hard-codes 224 x 224. This module wraps the same renderers with a page size S
(the files of round 1 stay unchanged while its grid runs). Layout is identical: line i occupies rows
[i*p, (i+1)*p), full page width, no margin; Latin / mono / learned glyphs use the Phase-2 renderer
(em = p/1.2, 4x supersampled, BOX-downsampled); colour blocks are b x b solid blocks.

Text supply (the round-1 bug: ``make_pages`` read ~n_lines*(224+8)*1.05 chars per page, so dense encodings ran
out and silently restarted at the first page):
  * every encoding states ``page_chars_max``, an upper bound on the source chars one page can consume;
  * a page is rendered only if at least that many unread chars remain, so every page is full and fresh;
  * pages never wrap back to the start: ``make_pages_s`` asserts all pages are distinct;
  * train text = data/train.txt (first 40M cleaned chars of WikiText-103 train); if an encoding needs more,
    data/train_r2.txt (same cleaning, larger cap, same prefix) is built and used;
  * test text = data/test.txt followed by data/val.txt (held-out WikiText splits, 2.06M chars). An encoding
    whose test pages need more (only col_b1) gets as many full test pages as fit (``allow_fewer``).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from encoder_probe.ep import flock
from encoder_probe.ep import pages as P1
from encoder_probe.ep import rgb_overlay as R1
from encoder_probe.ep.pages import LG_BANKS, LP, PRINTABLE, ROOT, Encoding, Page, clean_text
from vbpe import config
from vbpe.layout import _draw_lines, glyph_advance, wrap
from vbpe.metrics import normalize

# EP_TRAIN_R2_CAP raises the cap for a fresh build (same prefix, so existing pages are unchanged); needed for
# > ~1000 train pages of the densest encodings (see docs/gpu/LOCAL_GPU.md)
TRAIN_R2_CAP = int(os.environ.get("EP_TRAIN_R2_CAP", 64_000_000))
MIN_TEST_PAGES = 4


class Sized:
    """Mixin: page side S instead of the module-level 224."""
    S: int

    @property
    def n_lines(self) -> int:
        return self.S // self.pitch

    @property
    def page_chars_max(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class SizedScheme(Sized, P1.SchemeEncoding):
    """latin_p<p> / lg_* on an S x S page (P1.SchemeEncoding.render with W = H = S)."""

    def __init__(self, name, family, pitch, scheme, classes, S):
        super().__init__(name, family, pitch, scheme, classes)
        self.S = S

    @property
    def page_chars_max(self) -> int:
        # >= 1 px of line width per source char for every scheme here (densest: lg_free_w23_h3, 1.37 px/char)
        return int(self.n_lines * self.S * 1.5) + 200

    def render(self, text: str, pos: int) -> tuple[Page, int]:
        s, S = LP.supersample, self.S
        units = self.scheme.encode(text[pos:pos + self.page_chars_max])
        lines = (wrap(units, self.em * s, S * s) or [])[: self.n_lines]
        consumed = sum(len(u.source) for ln in lines for u in ln)
        assert consumed < self.page_chars_max - 100, "render window too small"
        img = _draw_lines([list(ln) for ln in lines], self.em, (S, S), LP, s)
        g = np.asarray(img, dtype=np.uint8)
        labs, refs = [], []
        for ln in lines:
            while ln and ln[-1].is_space:
                ln = ln[:-1]
            labs.append(tuple(lab for u in ln for lab in u.labels))
            refs.append(normalize("".join(u.source for u in ln)))
        return Page(np.repeat(g[:, :, None], 3, 2), labs, refs), consumed


class SizedColour(Sized, P1.ColourEncoding):
    def __init__(self, b: int, S: int):
        super().__init__(b)
        self.S = S

    @property
    def page_chars_max(self) -> int:
        return self.n_lines * (self.S // self.b)

    def render(self, text: str, pos: int) -> tuple[Page, int]:
        b, S = self.b, self.S
        per = S // b
        img = np.full((S, S, 3), 255, np.uint8)
        labs, refs, p = [], [], pos
        for i in range(self.n_lines):
            chunk = text[p:p + per]
            p += len(chunk)
            ids = np.array([self.idx[c] for c in chunk], dtype=np.int64)
            row = np.repeat(np.repeat(self.pal[ids][None], b, 0), b, 1)
            img[i * b:(i + 1) * b, : row.shape[1]] = row
            labs.append(tuple(chunk))
            refs.append(chunk)
        return Page(img, labs, refs), p - pos


def mono_cells_s(pitch: int, S: int) -> int:
    em = pitch / LP.line_spacing
    s = LP.supersample
    adv = glyph_advance(str(config.LATIN_MONO_FONT), round(em * s, 3), "M")
    return int(S * s // adv)


def mono_ink_s(lines: list[str], pitch: int, S: int) -> np.ndarray:
    units = [R1._MONO.encode(ln) for ln in lines]
    return np.asarray(_draw_lines(units, pitch / LP.line_spacing, (S, S), LP, LP.supersample), dtype=np.uint8)


class SizedMono(Sized, R1.MonoEncoding):
    def __init__(self, pitch: int, S: int):
        Encoding.__init__(self, f"mono_p{pitch}", "mono", pitch, list(PRINTABLE))
        self.S = S
        self.n_cells = mono_cells_s(pitch, S)

    @property
    def page_chars_max(self) -> int:
        return self.n_lines * self.n_cells

    def render(self, text: str, pos: int) -> tuple[Page, int]:
        n, p, chunks = self.n_cells, pos, []
        for _ in range(self.n_lines):
            chunks.append(text[p:p + n])
            p += len(chunks[-1])
        g = mono_ink_s(chunks, self.pitch, self.S)
        return Page(np.repeat(g[:, :, None], 3, 2), [tuple(c) for c in chunks], chunks), p - pos


class SizedRGB3(Sized, R1.RGB3Encoding):
    """rgb3_p<p> (channel-tagged) on an S x S page; the probe emits S*3 CTC frames per line."""

    def __init__(self, pitch: int, S: int, tagged: bool = True):
        prefix = "rgb3" if tagged else "rgb3a"
        classes = [t + c for t in R1.TAGS for c in PRINTABLE] if tagged else list(PRINTABLE)
        Encoding.__init__(self, f"{prefix}_p{pitch}", prefix, pitch, classes)
        self.S, self.tagged = S, tagged
        self.n_cells = mono_cells_s(pitch, S)
        if 2 * 3 * self.n_cells + 1 > S * R1.EXPAND:
            raise ValueError(f"{self.name}@{S}: L = {3 * self.n_cells} labels/line exceeds the CTC budget")

    @property
    def page_chars_max(self) -> int:
        return self.n_lines * 3 * self.n_cells

    def render(self, text: str, pos: int) -> tuple[Page, int]:
        n, p = self.n_cells, pos
        per_ch: list[list[str]] = [[], [], []]
        labs, refs = [], []
        for _ in range(self.n_lines):
            chunk = text[p:p + 3 * n]
            p += len(chunk)
            st = R1.split_streams(chunk, n)
            for c in range(3):
                per_ch[c].append(st[c])
            labs.append(tuple((R1.TAGS[c] + ch) if self.tagged else ch for c, ch in R1.interleave(st)))
            refs.append(chunk)
        img = np.stack([mono_ink_s(per_ch[c], self.pitch, self.S) for c in range(3)], 2)
        return Page(img, labs, refs), p - pos


from dataclasses import dataclass, field  # noqa: E402

# colour-gist phase A (docs/gpu/COLOUR_GIST.md): stream c darkens only its own channel(s) on a white page
STREAM_INKS = {1: [("black", (0, 1, 2))],
               2: [("cyan", (0,)), ("magenta", (1,))],
               3: [("cyan", (0,)), ("magenta", (1,)), ("yellow", (2,))],
               # brightness-balanced 2 streams (rgb2ls): Rec. 601 luma drop G 0.587 vs R+B 0.413 (cyan/magenta:
               # 0.299 vs 0.587), for a pretrained ViT that reads mostly brightness
               "2l": [("magenta", (1,)), ("green", (0, 2))]}


@dataclass
class StreamPage(Page):
    """A page of k text streams. ``refs`` = every stream's lines, stream by stream (so n_chars, flat text and
    digests cover the whole page); ``streams[s]`` = stream s's lines; ``after`` = the source text that follows
    the page (a topic-matched phrase source that is NOT on the page)."""
    streams: list = field(default_factory=list)
    colours: list = field(default_factory=list)
    after: str = ""


class SizedStreams(Sized, Encoding):
    """rgb<k>s_p<p>: k contiguous spans of text, each an ordinary latin_p<p> page (DejaVu Sans, word wrap) in its
    own colour channel(s): stream 1 = the first page-worth of text, stream 2 the next, ... rgb1s_p<p> is the
    single-stream reference (black ink, pixel-identical to latin_p<p>)."""

    def __init__(self, k: int, pitch: int, S: int, variant: str = ""):
        if (k if not variant else f"{k}{variant}") not in STREAM_INKS:
            raise KeyError(f"rgb{k}{variant}s: unknown stream layout")
        Encoding.__init__(self, f"rgb{k}{variant}s_p{pitch}", "streams", pitch, list(PRINTABLE))
        self.S, self.k, self.inks = S, k, STREAM_INKS[f"{k}{variant}" if variant else k]
        self.base = get_encoding_s(f"latin_p{pitch}", S)

    @property
    def page_chars_max(self) -> int:
        return self.k * self.base.page_chars_max

    def clean_hyp(self, s: str) -> str:
        return self.base.clean_hyp(s)

    def render(self, text: str, pos: int) -> tuple[Page, int]:
        p, streams, labels = pos, [], []
        img = np.full((self.S, self.S, 3), 255, np.uint8)
        for c, (_, chans) in enumerate(self.inks):
            pg, used = self.base.render(text, p)
            for ch in chans:
                img[:, :, ch] = pg.img[:, :, 0]
            streams.append(list(pg.refs))
            labels += pg.labels
            p += used
        return StreamPage(img, labels, [ln for st in streams for ln in st], streams,
                          [n for n, _ in self.inks], text[p:p + 600]), p - pos


def get_encoding_s(name: str, S: int) -> Encoding:
    if name.startswith("latin_p"):
        from vbpe.schemes.latin import LatinScheme
        return SizedScheme(name, "latin", int(name[len("latin_p"):]), LatinScheme(config.LATIN_FONT),
                           list(PRINTABLE), S)
    if name.startswith("col_b"):
        return SizedColour(int(name[len("col_b"):]), S)
    if name.startswith("mono_p"):
        return SizedMono(int(name[len("mono_p"):]), S)
    if name.startswith("rgb3_p"):
        return SizedRGB3(int(name[len("rgb3_p"):]), S)
    if name.startswith("rgb3a_p"):
        return SizedRGB3(int(name[len("rgb3a_p"):]), S, tagged=False)
    import re as _re
    m = _re.fullmatch(r"rgb([123])(l?)s_p(\d+)", name)
    if m:   # colour-gist phase A: k Latin streams, one per colour channel (l = brightness-balanced inks)
        return SizedStreams(int(m.group(1)), int(m.group(3)), S, m.group(2))
    if name.startswith("rgb3_lg_"):   # round-3 moonshot: RGB overlay of learned-glyph streams (any S)
        return R1.RGB3BankEncoding(name[len("rgb3_"):], S)
    if name in LG_BANKS:
        from learned_glyphs.lg.scheme import LearnedGlyphScheme, label_of
        sc = LearnedGlyphScheme(ROOT / LG_BANKS[name], name=name)
        fam = "lg_pb" if "_pb_" in name else "lg_free"
        return SizedScheme(name, fam, int(round(sc.height_px)), sc, [label_of(i) for i in range(len(sc.symbols))], S)
    raise KeyError(name)


# ----------------------------------------------------------------------------------------------- text supply

def _locked_build_train_r2(path: Path, cap: int) -> None:
    """data/train_r2.txt: WikiText-103 train cleaned exactly like data/train.txt, with a larger cap."""
    lock = open(path.with_suffix(".lock"), "w")
    flock.lock(lock)
    try:
        if path.exists():
            return
        from vbpe.corpus import _raw_split, clean_sentences
        tmp, total = path.with_suffix(".tmp"), 0
        with tmp.open("w") as f:
            for raw in _raw_split("train"):
                sents = clean_sentences(raw)
                if not sents:
                    continue
                par = " ".join(sents)
                f.write(par + "\n")
                total += len(par) + 1
                if total >= cap:
                    break
        os.replace(tmp, path)
    finally:
        flock.unlock(lock)
        lock.close()


def _read(path: Path, max_chars: int) -> str:
    with open(path, encoding="utf-8") as f:
        return clean_text(f.read(int(max_chars * 1.05) + 1000))


def text_stream_s(split: str, need: int) -> str:
    """>= ``need`` cleaned chars of the split if available (train: extended corpus on demand)."""
    data = ROOT / "data"
    if split == "train":
        base = data / "train.txt"
        if base.stat().st_size >= need * 1.05 + 1000:
            return _read(base, need)
        ext = data / "train_r2.txt"
        if not ext.exists():
            _locked_build_train_r2(ext, TRAIN_R2_CAP)
        return _read(ext, need)
    if split == "test":
        a = _read(data / "test.txt", need)
        if len(a) >= need:
            return a
        return normalize(a + " " + _read(data / "val.txt", need))
    raise KeyError(split)


def make_pages_s(enc: Encoding, split: str, n: int, allow_fewer: bool = False) -> list[Page]:
    """``n`` consecutive, full, distinct pages of ``split`` text (deterministic).

    With ``allow_fewer`` (test split only), returns as many full pages as the text supplies (>= 4)."""
    cap = enc.page_chars_max
    text = text_stream_s(split, n * cap + cap)
    pages, pos = [], 0
    for _ in range(n):
        if len(text) - pos < cap:
            break
        pg, used = enc.render(text, pos)
        assert used > 0
        pages.append(pg)
        pos += used
    if len(pages) < n and not (allow_fewer and len(pages) >= MIN_TEST_PAGES):
        raise ValueError(f"{enc.name}@{enc.S}: only {len(pages)} of {n} full {split} pages fit the text")
    keys = {tuple(p.refs) for p in pages}
    assert len(keys) == len(pages), f"{enc.name}: repeated pages"
    return pages


def pages_digest(pages: list[Page]) -> str:
    import hashlib
    h = hashlib.sha1()
    for p in pages:
        h.update(p.img.tobytes())
    return h.hexdigest()[:16]
