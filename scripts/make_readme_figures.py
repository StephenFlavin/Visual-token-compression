"""Draw the three README demo figures (docs/img/*.png) for one short haiku.

1. glyphs.png   plain text vs the learned whole-pixel glyph alphabet (N=300 chunks)
2. colour.png   plain text vs one coloured 2x2-pixel block per character (encoder_probe col_b2)
3. rgb.png      three lines of text vs the same three lines overlaid in the R, G and B channels (rgb3)

Every picture is drawn at one zoom factor per figure, so widths and areas compare honestly; each figure
also shows the real 1:1 pixels ("actual size"). Needs torch (to read the bank) and the fonts from
scripts/fetch_fonts.sh (Inter is optional: labels fall back to DejaVu Sans).

    python3 scripts/make_readme_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "learned_glyphs"))

from encoder_probe.ep.pages import PRINTABLE, palette  # noqa: E402
from lg.alphabet import Alphabet  # noqa: E402

OUT = ROOT / "docs" / "img"
BANK = ROOT / "learned_glyphs/results/e1long/banks/lg_learned_n300_pb_h5.pt"
HAIKU = ["the light of the moon", "is falling on the water", "and the night is still"]

DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEJAVU_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
INTER = ROOT / "assets/fonts/Inter.ttf"

PITCH = 5                     # native line pitch in px (latin_p5 / lg h5 / mono_p5)
EM = PITCH / 1.2              # layout convention: em = pitch / 1.2, baseline 0.9 em below the line top
W = 2000                      # figure width (images are drawn at 2x for sharp display)

INK = (17, 24, 39)
MUTED = (107, 114, 128)
FAINT = (229, 231, 235)
CARD = (255, 255, 255)
CELL_A, CELL_B = (243, 244, 246), (232, 234, 238)
BEFORE, AFTER = (148, 163, 184), (79, 70, 229)
# soft tints for the chunks that became one glyph (same tint on both sides)
TINTS = [(254, 215, 170), (187, 247, 208), (191, 219, 254), (233, 213, 255), (254, 202, 202),
         (253, 230, 138), (165, 243, 252), (251, 207, 232), (217, 249, 157), (199, 210, 254)]


def ui(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    if INTER.exists():
        f = ImageFont.truetype(str(INTER), size)
        f.set_variation_by_name(weight)
        return f
    return ImageFont.truetype(DEJAVU.replace("Sans.ttf", "Sans-Bold.ttf") if weight != "Regular" else DEJAVU, size)


def new_card(h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (W, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, h - 1], radius=36, fill=CARD, outline=FAINT, width=3)
    return img, d


def panel_title(d, x, y, tag, text, colour):
    f = ui(30, "Bold")
    tw = d.textlength(tag, font=f)
    d.rounded_rectangle([x, y, x + tw + 32, y + 50], radius=25, fill=colour)
    d.text((x + 16, y + 25), tag, font=f, fill=(255, 255, 255), anchor="lm")
    d.text((x + tw + 52, y + 25), text, font=ui(34, "SemiBold"), fill=INK, anchor="lm")


def arrow(d, x, y):
    d.line([x - 34, y, x + 26, y], fill=MUTED, width=6)
    d.polygon([(x + 40, y), (x + 18, y - 16), (x + 18, y + 16)], fill=MUTED)


def bars(d, x, y, width, rows, unit):
    """Horizontal comparison bars. rows = [(label, value, colour, right_note)]."""
    vmax = max(r[1] for r in rows)
    lab_w = 300
    for i, (label, v, col, note) in enumerate(rows):
        yy = y + i * 64
        d.text((x, yy + 20), label, font=ui(28, "Medium"), fill=INK, anchor="lm")
        bw = (width - lab_w - 330) * v / vmax
        d.rounded_rectangle([x + lab_w, yy, x + lab_w + bw, yy + 40], radius=10, fill=col)
        d.text((x + lab_w + bw + 18, yy + 20), f"{v:.0f} {unit}", font=ui(28, "SemiBold"), fill=INK, anchor="lm")
        if note:
            vw = d.textlength(f"{v:.0f} {unit}", font=ui(28, "SemiBold"))
            d.text((x + lab_w + bw + 30 + vw, yy + 20), note, font=ui(28, "Bold"), fill=col, anchor="lm")
    return y + len(rows) * 64


def latin_real(lines, font_path=DEJAVU, pitch=PITCH, ss=4) -> np.ndarray:
    """Real-size rendering as in the experiments: draw at 4x, then box-downsample (grey, 0 = ink)."""
    em = pitch / 1.2
    f = ImageFont.truetype(font_path, round(em * ss))
    wpx = int(max(f.getlength(t) for t in lines) / ss) + 2
    big = Image.new("L", (wpx * ss, pitch * len(lines) * ss), 255)
    g = ImageDraw.Draw(big)
    for i, t in enumerate(lines):
        g.text((0, (i * pitch + 0.9 * em) * ss), t, font=f, fill=0, anchor="ls")
    return np.asarray(big.resize((wpx, pitch * len(lines)), Image.BOX))


def actual_size(img, d, x, y, items):
    """Real-size renders, pasted at 2x (= true size on a 2x display), side by side with a label under each."""
    d.text((x, y), "Actual size (what the reader gets):", font=ui(24, "Medium"), fill=MUTED, anchor="lt")
    hmax = 2 * max(arr.shape[0] for arr, _ in items)
    for arr, label in items:
        a = Image.fromarray(arr).convert("RGB")
        a = a.resize((a.width * 2, a.height * 2), Image.NEAREST)
        d.rounded_rectangle([x - 10, y + 42, x + a.width + 10, y + 62 + a.height], radius=8, outline=FAINT, width=2)
        img.paste(a, (x, y + 52))
        d.text((x + a.width / 2, y + 76 + hmax), label, font=ui(22), fill=MUTED, anchor="mt")
        x += a.width + 70


def latin_panel(d, x, y, z, highlights=None, font_path=DEJAVU):
    """Haiku at zoom z (vector, same geometry as the 5-px render). highlights: per line [(start, end, tint)]."""
    f = ImageFont.truetype(font_path, round(EM * z))
    pitch = PITCH * z
    for i, t in enumerate(HAIKU):
        top = y + i * pitch
        for s, e, tint in (highlights or [[]] * 3)[i]:
            x0, x1 = x + f.getlength(t[:s]), x + f.getlength(t[:e])
            d.rounded_rectangle([x0 + 1, top + 8, x1 - 1, top + pitch - 8], radius=14, fill=tint)
        d.text((x, top + 0.9 * EM * z), t, font=f, fill=INK, anchor="ls")
    return max(f.getlength(t) for t in HAIKU)


GOOD = (22, 163, 74)


def reads_as_label(d, x, y, sub=""):
    d.ellipse([x, y, x + 34, y + 34], fill=GOOD)
    d.line([(x + 9, y + 18), (x + 15, y + 24), (x + 26, y + 11)], fill=(255, 255, 255), width=4)
    f = ui(28, "SemiBold")
    d.text((x + 48, y + 17), "Reads back as", font=f, fill=INK, anchor="lm")
    if sub:
        d.text((x + 58 + d.textlength("Reads back as", font=f), y + 17), sub, font=ui(26), fill=MUTED, anchor="lm")


def reads_as(d, x, y, highlights=None):
    """The decoded haiku under an AFTER panel (optionally with the chunk tints). Returns the bottom y."""
    reads_as_label(d, x, y)
    f = ImageFont.truetype(DEJAVU, 32)
    lh = 46
    for i, t in enumerate(HAIKU):
        top = y + 50 + i * lh
        for s0, e, tint in (highlights or [[]] * 3)[i]:
            x0, x1 = x + f.getlength(t[:s0]), x + f.getlength(t[:e])
            d.rounded_rectangle([x0 + 1, top + 2, x1 - 1, top + lh - 4], radius=8, fill=tint)
        d.text((x, top + 34), t, font=f, fill=INK, anchor="ls")
    return y + 50 + 3 * lh


# --------------------------------------------------------------------------- figure 1: learned glyphs
def fig_glyphs():
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    syms, ink, wss, ss = bank["symbols"], bank["ink"].numpy(), bank["widths_ss"].numpy(), bank["ss"]
    alpha = Alphabet(syms[95:])
    z = 20
    # segment, assign one tint per distinct chunk
    segs, tint_of = [], {}
    for t in HAIKU:
        ids = alpha.encode(t)
        segs.append(ids)
        for i in ids:
            if alpha.is_token(i) and i not in tint_of:
                tint_of[i] = TINTS[len(tint_of) % len(TINTS)]
    hl = []
    for t, ids in zip(HAIKU, segs):
        pos, row = 0, []
        for i in ids:
            n = len(syms[i])
            if i in tint_of:
                row.append((pos, pos + n, tint_of[i]))
            pos += n
        hl.append(row)

    def cell_px(i):          # whole-pixel bitmap (5 x w) of symbol i, 1 = ink
        w = int(wss[i]) // ss
        a = ink[i][:, : int(wss[i])]
        return a.reshape(PITCH, ss, w, ss).mean((1, 3)) > 0.5

    img, d = new_card(1160)
    lx, rx, ty = 70, 1230, 70
    panel_title(d, lx, ty, "BEFORE", "Plain text", BEFORE)
    panel_title(d, rx, ty, "AFTER", "Learned glyphs", AFTER)
    y0 = ty + 110
    lw = latin_panel(d, lx, y0, z, hl)
    arrow(d, (lx + lw + rx) / 2 - 10, y0 + 1.5 * PITCH * z)

    pitch = PITCH * z
    real = np.full((PITCH * 3, 40), 255, np.uint8)
    glyph_w = []
    for li, ids in enumerate(segs):
        x, top, k = rx, y0 + li * pitch, 0
        row_real = []
        for i in ids:
            bm = cell_px(i)
            w = bm.shape[1]
            bg = tint_of.get(i) or (CELL_A if k % 2 == 0 else CELL_B)
            k += i not in tint_of
            d.rounded_rectangle([x + 2, top + 2, x + w * z - 3, top + pitch - 3], radius=6, fill=bg)
            for r, c in zip(*np.nonzero(bm)):
                px, py = x + c * z, top + r * z
                d.rounded_rectangle([px + 2, py + 2, px + z - 2, py + z - 2], radius=3, fill=INK)
            x += w * z
            row_real.append(np.where(bm, 0, 255).astype(np.uint8))
        glyph_w.append((x - rx) / z)
        line = np.concatenate(row_real, 1)
        real[li * PITCH:(li + 1) * PITCH, : line.shape[1]] = line

    n_chars = sum(len(t) for t in HAIKU)
    n_syms = sum(len(s) for s in segs)
    f = ImageFont.truetype(DEJAVU, 400)
    latin_w = sum(f.getlength(t) / 400 * EM for t in HAIKU)
    gw = sum(glyph_w)
    # caption under the glyph panel
    yc = y0 + 3 * pitch + 24
    d.text((rx, yc), "each coloured cell is ONE glyph for a whole chunk",
           font=ui(24), fill=MUTED, anchor="lt")
    d.text((lx, yc), "coloured = chunks from the 300-entry dictionary", font=ui(24), fill=MUTED, anchor="lt")
    ye = reads_as(d, rx, yc + 60, hl)

    yb = ye + 70
    d.line([lx, yb - 24, W - 70, yb - 24], fill=FAINT, width=2)
    y = bars(d, lx, yb, 1120, [
        (f"Plain: {n_chars} chars", latin_w, BEFORE, ""),
        (f"Glyphs: {n_syms} symbols", gw, AFTER, f"  {gw / latin_w - 1:+.0%}"),
    ], "px of line")
    actual_size(img, d, 1300, yb - 6, [(latin_real(HAIKU), "plain"), (real, "glyphs")])
    img = img.crop((0, 0, W, y + 70))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, img.height - 1], radius=36, outline=FAINT, width=3)
    img.save(OUT / "glyphs.png", optimize=True)
    print(f"glyphs: {n_chars} chars -> {n_syms} symbols; width {latin_w:.1f} -> {gw:.1f} px ({gw / latin_w:.2f}x)")


# --------------------------------------------------------------------------- figure 2: colour blocks
def fig_colour():
    pal, idx = palette(), {c: i for i, c in enumerate(PRINTABLE)}
    z, b = 18, 2
    img, d = new_card(1200)
    lx, rx, ty = 70, 1060, 70
    panel_title(d, lx, ty, "BEFORE", "Plain text", BEFORE)
    panel_title(d, rx, ty, "AFTER", "One colour block per character", AFTER)
    y0 = ty + 110
    lw = latin_panel(d, lx, y0, z)
    arrow(d, (lx + lw + rx) / 2 - 10, y0 + 1.5 * PITCH * z)

    s = b * z                        # block size on the figure
    row_h = s + 46
    lab = ui(24, "Medium")
    real = np.full((b * 3, b * max(map(len, HAIKU)), 3), 255, np.uint8)
    for li, t in enumerate(HAIKU):
        top = y0 + 6 + li * row_h
        for k, ch in enumerate(t):
            col = tuple(int(v) for v in pal[idx[ch]])
            x = rx + k * s
            d.rounded_rectangle([x + 2, top, x + s - 2, top + s - 4], radius=6, fill=col)
            d.text((x + s / 2, top + s + 6), "␣" if ch == " " else ch, font=lab, fill=MUTED, anchor="mt")
            real[li * b:(li + 1) * b, k * b:(k + 1) * b] = col
    yc = y0 + 3 * row_h + 10
    d.text((rx, yc), "95 colours (5 levels per channel), letters shown only as a key",
           font=ui(24), fill=MUTED, anchor="lt")
    ye = reads_as(d, rx, yc + 60)

    n_chars = sum(len(t) for t in HAIKU)
    f = ImageFont.truetype(DEJAVU, 400)
    latin_area = sum(f.getlength(t) / 400 * EM for t in HAIKU) * PITCH
    yb = ye + 70
    d.line([lx, yb - 24, W - 70, yb - 24], fill=FAINT, width=2)
    y = bars(d, lx, yb, 1120, [
        ("Plain text", latin_area, BEFORE, ""),
        ("2×2 colour blocks", n_chars * b * b, AFTER, f"  {n_chars * b * b / latin_area - 1:+.0%}"),
    ], "px²")
    actual_size(img, d, 1300, yb - 6, [(latin_real(HAIKU), "plain"), (real, "colour")])
    img = img.crop((0, 0, W, y + 70))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, img.height - 1], radius=36, outline=FAINT, width=3)
    img.save(OUT / "colour.png", optimize=True)
    print(f"colour: {latin_area / n_chars:.1f} -> {b * b} px^2/char ({b * b * n_chars / latin_area:.2f}x)")


# --------------------------------------------------------------------------- figure 3: RGB overlay
def mono_masks(z: float, ss: int = 1) -> list[np.ndarray]:
    """One ink mask per haiku line in DejaVu Sans Mono (0..255, 255 = ink), all the same size."""
    f = ImageFont.truetype(DEJAVU_MONO, round(EM * z * ss))
    wpx = int(max(f.getlength(t) for t in HAIKU)) + 4 * ss
    out = []
    for t in HAIKU:
        m = Image.new("L", (wpx, round(PITCH * z * ss)), 0)
        ImageDraw.Draw(m).text((0, 0.9 * EM * z * ss), t, font=f, fill=255, anchor="ls")
        out.append(m)
    if ss > 1:
        out = [m.resize((m.width // ss, m.height // ss), Image.BOX) for m in out]
    return [np.asarray(m) for m in out]


def overlay(masks):
    """rgb3 compositing: stream c darkens only channel c."""
    return np.stack([255 - m for m in masks], 2).astype(np.uint8)


def fig_rgb():
    z = 13
    img, d = new_card(1300)
    lx, rx, ty = 70, 1060, 70
    panel_title(d, lx, ty, "BEFORE", "Three lines of text", BEFORE)
    panel_title(d, rx, ty, "AFTER", "One line, one text per channel", AFTER)
    y0 = ty + 110
    pitch = PITCH * z
    f = ImageFont.truetype(DEJAVU_MONO, round(EM * z))
    chips = [("R", (220, 38, 38)), ("G", (22, 163, 74)), ("B", (37, 99, 235))]
    for i, t in enumerate(HAIKU):
        top = y0 + i * (pitch + 14)
        tag, col = chips[i]
        d.rounded_rectangle([lx, top + 18, lx + 44, top + pitch - 18], radius=10, fill=col)
        d.text((lx + 22, top + pitch / 2), tag, font=ui(26, "Bold"), fill=(255, 255, 255), anchor="mm")
        d.text((lx + 66, top + 0.9 * EM * z), t, font=f, fill=INK, anchor="ls")
    lw = 66 + max(f.getlength(t) for t in HAIKU)
    arrow(d, (lx + lw + rx) / 2 - 10, y0 + 1.5 * pitch)

    masks = mono_masks(z, ss=2)
    comp = Image.fromarray(overlay(masks))
    ytop = y0 + pitch + 14
    d.rectangle([rx - 1, ytop - 1, rx + comp.width, ytop + comp.height], outline=FAINT, width=2)
    img.paste(comp, (rx, ytop))
    d.text((rx, ytop + comp.height + 18), "red-channel text shows as cyan ink, green as magenta, blue as yellow;",
           font=ui(24), fill=MUTED, anchor="lt")
    d.text((rx, ytop + comp.height + 50), "where all three overlap the pixel is black",
           font=ui(24), fill=MUTED, anchor="lt")

    # channel split: each channel alone gives one clean line back
    ys = y0 + 3 * (pitch + 14) + 40
    d.line([lx, ys - 20, W - 70, ys - 20], fill=FAINT, width=2)
    reads_as_label(d, lx, ys + 4)
    d.text((lx + 48, ys + 52), "split the colour channels and", font=ui(26), fill=MUTED, anchor="lt")
    d.text((lx + 48, ys + 88), "each line comes back on its own", font=ui(26), fill=MUTED, anchor="lt")
    arr = np.asarray(comp)
    sx, sw = 620, int(comp.width * 0.6)
    for c, (tag, col) in enumerate(chips):
        ch = Image.fromarray(arr[:, :, c]).convert("RGB")
        ch = ch.resize((sw, int(ch.height * 0.6)), Image.LANCZOS)
        yy = ys + c * (ch.height + 10)
        d.rounded_rectangle([sx, yy + 6, sx + 36, yy + ch.height - 6], radius=8, fill=col)
        d.text((sx + 18, yy + ch.height / 2), tag, font=ui(22, "Bold"), fill=(255, 255, 255), anchor="mm")
        img.paste(ch, (sx + 50, yy))
    yb = ys + 3 * (ch.height + 10) + 50
    d.line([lx, yb - 24, W - 70, yb - 24], fill=FAINT, width=2)

    real3 = overlay(mono_masks(1, ss=4))
    real_mono = np.concatenate([255 - m for m in mono_masks(1, ss=4)], 0)
    n_chars = sum(len(t) for t in HAIKU)
    mono_area = real_mono.size
    y = bars(d, lx, yb, 1120, [
        ("Three lines", real_mono.shape[0] * real_mono.shape[1], BEFORE, ""),
        ("RGB overlay", real3.shape[0] * real3.shape[1], AFTER, "  −" + f"{1 - 1 / 3:.0%}"),
    ], "px²")
    actual_size(img, d, 1300, yb - 6, [(real_mono, "three lines"), (real3, "RGB overlay")])
    img = img.crop((0, 0, W, y + 70))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, img.height - 1], radius=36, outline=FAINT, width=3)
    img.save(OUT / "rgb.png", optimize=True)
    print(f"rgb: {n_chars} chars, {mono_area / n_chars:.1f} -> {real3.shape[0] * real3.shape[1] / n_chars:.1f} px^2/char")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_glyphs()
    fig_colour()
    fig_rgb()
