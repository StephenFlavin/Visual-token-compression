"""RGB overlay encodings: up to three text streams per line, one per colour channel (docs/IDEAS.md).

All pages are 224x224 RGB uint8 with lines of integer pitch p (line i = rows [i*p, (i+1)*p)), rendered like
``latin_p``: em = p/1.2, line spacing 1.2, no margin, 4x supersampled then BOX-downsampled (vbpe.layout).

Families
  mono_p<p>      DejaVu Sans Mono black on white, fixed cells (0.60 em), fixed-length chunks of n_cells chars per
                 line (no word wrap, like col_b), grey replicated to 3 channels. Baseline for rgb3: it isolates
                 the overlay effect from the font/layout change. Channel c of an rgb3 page is exactly a mono page
                 of stream c.
  rgb3_p<p>      MAIN variant. Each line holds 3*n_cells chars of source text: R gets chars [0:n), G [n:2n),
                 B [2n:3n), all in the same mono cell grid. Per-channel compositing on white: stream c darkens
                 only channel c (value = 255 - ink), so overlapping glyphs never hide each other.
                 CTC labels are interleaved per cell R[k], G[k], B[k] (left to right), with channel-tagged
                 classes ("R"+ch, "G"+ch, "B"+ch: 3 x 95 classes) so a dropped or inserted label only costs that
                 one char after decoding (decode routes every label to its stream by tag). ref = the chunk in
                 source order (R+G+B), so CER is on real text.
  rgb3a_p<p>     Same pages, plain printable-ASCII classes; decode de-interleaves by position. One indel shifts
                 every later char to the wrong stream, so its CER mixes reading errors with alignment errors.
                 Kept as a spec-literal check, not in the grid.
  rgbw_p<p>[_chan|_paint][_s<k>]
                 The user's literal version: words go round-robin to R, G, B; the three words of one slot share
                 its x position, the word in channel c is shifted right by c*k px (k = 1 by default, output px).
                 Slot width = max_c(width_c + c*k) + one space advance; slots flow left to right and wrap.
                 DejaVu Sans (the latin_p font). ``_chan`` (default): per-channel compositing as in rgb3.
                 ``_paint``: solid (255,0,0)/(0,255,0)/(0,0,255) ink on white, B painted over G over R (at 4x,
                 alpha = glyph coverage), so overlaps hide earlier words. Labels/ref = the line's words in source
                 order joined by spaces, so CTC must emit 2-3 spatially overlapping words in sequence.

Frame budget: the probe emits T = 224*3 frames per line. CTC needs at most T >= 2L+1 (worst case: every label
repeated); rgb3 has L = 3*n_cells, which rules out p <= 3 (p3: 2*444+1 = 889 > 672; p4: 667 passes).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from encoder_probe.ep.pages import H, LP, PRINTABLE, W, Encoding, Page
from vbpe import config
from vbpe.layout import _draw_lines, get_font, glyph_advance
from vbpe.metrics import normalize
from vbpe.schemes.latin import LatinScheme

EXPAND = 3                    # probe frames per pixel column (Probe.expand)
T_FRAMES = W * EXPAND         # CTC frames per line
TAGS = "RGB"


def ctc_ok(n_labels: int) -> bool:
    """Conservative CTC feasibility: T >= 2L+1."""
    return 2 * n_labels + 1 <= T_FRAMES


@lru_cache(maxsize=None)
def mono_cells(pitch: int) -> int:
    """Number of monospace cells that fit across the page at this pitch."""
    em = pitch / LP.line_spacing
    s = LP.supersample
    adv = glyph_advance(str(config.LATIN_MONO_FONT), round(em * s, 3), "M")
    return int(W * s // adv)


_MONO = LatinScheme(config.LATIN_MONO_FONT, name="mono")


def mono_ink(lines: list[str], pitch: int) -> np.ndarray:
    """Grey (H, W) uint8 page (255 = white) with ``lines[i]`` drawn in the mono cell grid of line i."""
    units = [_MONO.encode(ln) for ln in lines]
    img = _draw_lines(units, pitch / LP.line_spacing, (W, H), LP, LP.supersample)
    return np.asarray(img, dtype=np.uint8)


class MonoEncoding(Encoding):
    """Baseline: one mono stream, fixed-length chunks of ``n_cells`` chars per line."""

    def __init__(self, pitch: int):
        super().__init__(f"mono_p{pitch}", "mono", pitch, list(PRINTABLE))
        self.n_cells = mono_cells(pitch)

    def clean_hyp(self, s: str) -> str:
        return s  # refs are raw fixed-length chunks (may start/end with a space), like ColourEncoding

    def render(self, text: str, pos: int) -> tuple[Page, int]:
        n, p, chunks = self.n_cells, pos, []
        for _ in range(self.n_lines):
            chunks.append(text[p:p + n])
            p += len(chunks[-1])
        g = mono_ink(chunks, self.pitch)
        return Page(np.repeat(g[:, :, None], 3, 2), [tuple(c) for c in chunks], chunks), p - pos


def split_streams(chunk: str, n: int) -> list[str]:
    return [chunk[:n], chunk[n:2 * n], chunk[2 * n:3 * n]]


def stream_lengths(L: int, n: int) -> list[int]:
    return [min(L, n), min(max(L - n, 0), n), max(L - 2 * n, 0)]


def interleave(streams: list[str]) -> list[tuple[int, str]]:
    """[(channel, char)] in cell order R[k], G[k], B[k], k = 0..n-1 (shorter streams just end)."""
    out = []
    for k in range(max(len(s) for s in streams)):
        for c, s in enumerate(streams):
            if k < len(s):
                out.append((c, s[k]))
    return out


class RGB3Encoding(Encoding):
    """Three independent mono streams per line, one per channel, same cell grid; interleaved CTC labels."""

    def __init__(self, pitch: int, tagged: bool = True):
        prefix = "rgb3" if tagged else "rgb3a"
        classes = [t + c for t in TAGS for c in PRINTABLE] if tagged else list(PRINTABLE)
        super().__init__(f"{prefix}_p{pitch}", prefix, pitch, classes)
        self.tagged = tagged
        self.n_cells = mono_cells(pitch)
        if not ctc_ok(3 * self.n_cells):
            raise ValueError(f"{self.name}: L = {3 * self.n_cells} labels/line exceeds the CTC budget "
                             f"(2L+1 > {T_FRAMES} frames)")

    def clean_hyp(self, s: str) -> str:
        return s  # raw fixed-length chunks: do not strip

    def decode(self, labels: str) -> str:
        if self.tagged:  # every class is 2 chars: tag + char
            streams = ["", "", ""]
            for i in range(0, len(labels) - 1, 2):
                streams[TAGS.index(labels[i])] += labels[i + 1]
            return "".join(streams)
        lens = stream_lengths(len(labels), self.n_cells)
        streams = [[], [], []]
        for (c, _), ch in zip(interleave(["x" * k for k in lens]), labels):
            streams[c].append(ch)
        return "".join("".join(s) for s in streams)

    def render(self, text: str, pos: int) -> tuple[Page, int]:
        n, p = self.n_cells, pos
        per_ch = [[], [], []]
        labs, refs = [], []
        for _ in range(self.n_lines):
            chunk = text[p:p + 3 * n]
            p += len(chunk)
            st = split_streams(chunk, n)
            for c in range(3):
                per_ch[c].append(st[c])
            il = interleave(st)
            labs.append(tuple((TAGS[c] + ch) if self.tagged else ch for c, ch in il))
            refs.append(chunk)
        img = np.stack([mono_ink(per_ch[c], self.pitch) for c in range(3)], 2)
        return Page(img, labs, refs), p - pos


PAINT = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float32)


class RGBWordEncoding(Encoding):
    """Words round-robin to R, G, B in shared slots; later channels shifted right by ``shift`` px."""

    def __init__(self, pitch: int, mode: str = "chan", shift: float = 1.0):
        assert mode in ("chan", "paint")
        sfx = "" if shift == 1.0 else f"_s{shift:g}"
        super().__init__(f"rgbw_p{pitch}_{mode}{sfx}", f"rgbw_{mode}", pitch, list(PRINTABLE))
        self.mode, self.shift = mode, shift
        self.font = str(config.LATIN_FONT)
        self.em = pitch / LP.line_spacing

    def _tokens(self, text: str, pos: int, est: int, max_w: float, size: float):
        """[(piece, start, end)] source spans; over-long words are hard-split into pieces that fit."""
        window = text[pos:pos + est]
        out = []
        for m in re.finditer(r"\S+", window):
            if m.end() == len(window) and pos + est < len(text):
                break  # word may be cut by the window
            w, st = m.group(), pos + m.start()
            if glyph_advance(self.font, size, w) <= max_w:
                out.append((w, st, st + len(w)))
                continue
            a = 0
            while a < len(w):
                b = a + 1
                while b < len(w) and glyph_advance(self.font, size, w[a:b + 1]) <= max_w:
                    b += 1
                out.append((w[a:b], st + a, st + b))
                a = b
        return out

    def layout(self, text: str, pos: int):
        """-> (placed [(line, x_ss, channel, piece)], refs per line, consumed)."""
        s = LP.supersample
        size = round(self.em * s, 3)
        sh = self.shift * s
        gap = glyph_advance(self.font, size, " ")
        Wss = W * s
        toks = self._tokens(text, pos, self.n_lines * W * 3 + 200, Wss - 2 * sh, size)
        placed, line_spans = [], [[] for _ in range(self.n_lines)]
        line, x, end = 0, 0.0, pos
        for j in range(0, len(toks), 3):
            slot = toks[j:j + 3]
            sw = max(glyph_advance(self.font, size, t[0]) + c * sh for c, t in enumerate(slot))
            if x > 0 and x + sw > Wss:
                line, x = line + 1, 0.0
            if line >= self.n_lines:
                break
            for c, t in enumerate(slot):
                placed.append((line, x + c * sh, c, t[0]))
                line_spans[line].append(t)
            x += sw + gap
            end = slot[-1][2]
        refs = [normalize(text[sp[0][1]:sp[-1][2]]) if sp else "" for sp in line_spans]
        consumed = end - pos
        if end < len(text) and text[end] == " ":
            consumed += 1
        return placed, refs, consumed

    def render(self, text: str, pos: int) -> tuple[Page, int]:
        s = LP.supersample
        placed, refs, consumed = self.layout(text, pos)
        font = get_font(self.font, self.em * s)
        ch = [Image.new("L", (W * s, H * s), 255) for _ in range(3)]
        dr = [ImageDraw.Draw(im) for im in ch]
        base = LP.first_baseline(self.em) * s
        for line, x, c, piece in placed:
            dr[c].text((x, base + line * self.pitch * s), piece, font=font, fill=0, anchor="ls")
        if self.mode == "chan":
            img = np.stack([np.asarray(im.resize((W, H), Image.Resampling.BOX), np.uint8) for im in ch], 2)
        else:
            page = np.full((H * s, W * s, 3), 255.0, np.float32)
            for c in range(3):  # R first, then G over it, then B on top
                a = (255.0 - np.asarray(ch[c], np.float32))[..., None] / 255.0
                page = page * (1 - a) + a * PAINT[c]
            page = page.reshape(H, s, W, s, 3).mean((1, 3))
            img = np.clip(np.rint(page), 0, 255).astype(np.uint8)
        return Page(img, [tuple(r) for r in refs], refs), consumed


_RGBW = re.compile(r"rgbw_p(\d+)(?:_(chan|paint))?(?:_s([\d.]+))?$")


def get_rgb_encoding(name: str) -> Encoding:
    if name.startswith("mono_p"):
        return MonoEncoding(int(name[len("mono_p"):]))
    if name.startswith("rgb3_p"):
        return RGB3Encoding(int(name[len("rgb3_p"):]), tagged=True)
    if name.startswith("rgb3a_p"):
        return RGB3Encoding(int(name[len("rgb3a_p"):]), tagged=False)
    m = _RGBW.match(name)
    if m:
        return RGBWordEncoding(int(m.group(1)), m.group(2) or "chan", float(m.group(3) or 1))
    raise KeyError(name)
