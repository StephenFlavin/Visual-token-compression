"""Draw the three README demo figures (docs/img/*.png) for one short haiku.

1. glyphs.png   plain text vs Visual BPE substitution: frequent chunks become one glyph (subst300)
2. colour.png   plain text vs one coloured 2x2-pixel block per character (encoder_probe col_b2)
3. rgb.png      three lines of text vs the same three lines overlaid in the R, G and B channels (rgb3)

Every picture is drawn at one zoom factor per figure, so widths and areas compare honestly; each figure
also shows the real 1:1 pixels ("actual size"). Needs the fonts from
scripts/fetch_fonts.sh (Inter is optional: labels fall back to DejaVu Sans).

    python3 scripts/make_readme_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from encoder_probe.ep.pages import PRINTABLE, palette  # noqa: E402

OUT = ROOT / "docs" / "img"
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



# --------------------------------------------------------------------------- figure 1: chunk glyphs
def units_mask(units, em: float, height: int, top_baseline: float) -> tuple[Image.Image, float]:
    """Draw one line of scheme units (black on white, 'L') and return it with its width."""
    wpx = sum(u.advance(em) for u in units)
    m = Image.new("L", (int(wpx) + 4, height), 255)
    g, x = ImageDraw.Draw(m), 0.0
    for u in units:
        u.draw(g, x, top_baseline, em)
        x += u.advance(em)
    return m, wpx


def fig_glyphs():
    """Visual BPE as in vbpe.schemes.substitution: frequent chunks -> one glyph, other letters unchanged.
    Same font (DejaVu Sans) for leftover letters on both sides, so only the substitution differs."""
    from vbpe.schemes.substitution import CJKUnit, SubstitutionScheme
    scheme = SubstitutionScheme(300, leftover_font=DEJAVU)
    z = 20
    em, pitch = EM * z, PITCH * z
    enc = [scheme.encode(t) for t in HAIKU]
    tint_of, hl = {}, []
    for t, units in zip(HAIKU, enc):
        pos, row = 0, []
        for u in units:
            n = len(u.source)
            if isinstance(u, CJKUnit):
                tint_of.setdefault(u.char, TINTS[len(tint_of) % len(TINTS)])
                row.append((pos, pos + n, tint_of[u.char]))
            pos += n
        hl.append(row)

    img, d = new_card(1160)
    lx, rx, ty = 70, 1230, 70
    panel_title(d, lx, ty, "BEFORE", "Plain text", BEFORE)
    panel_title(d, rx, ty, "AFTER", "Chunks become one glyph", AFTER)
    y0 = ty + 110
    lw = latin_panel(d, lx, y0, z, hl)
    arrow(d, (lx + lw + rx) / 2 - 10, y0 + 1.5 * pitch)

    glyph_w = []
    for li, units in enumerate(enc):
        top = y0 + li * pitch
        x = rx
        for u in units:                       # tint behind each glyph, matching the BEFORE highlight
            a = u.advance(em)
            if isinstance(u, CJKUnit):
                d.rounded_rectangle([x + 1, top + 8, x + a - 1, top + pitch - 8], radius=14, fill=tint_of[u.char])
            x += a
        m, wpx = units_mask(units, em, pitch, 0.9 * em)
        img.paste(INK, (rx, top), Image.eval(m, lambda v: 255 - v))
        glyph_w.append(wpx / z)
    yc = y0 + 3 * pitch + 24
    d.text((lx, yc), "coloured = frequent chunks (from a 300-entry dictionary)", font=ui(24), fill=MUTED, anchor="lt")
    d.text((rx, yc), "each chunk is drawn as ONE symbol; other letters stay", font=ui(24), fill=MUTED, anchor="lt")

    # real 5-px renders (4x supersampled, box-downsampled, as in the experiments)
    ss = 4
    reals = []
    for units in enc:
        m, _ = units_mask(units, EM * ss, PITCH * ss, 0.9 * EM * ss)
        reals.append(np.asarray(m.resize((m.width // ss, PITCH), Image.BOX)))
    real = np.full((PITCH * 3, max(r.shape[1] for r in reals)), 255, np.uint8)
    for i, r in enumerate(reals):
        real[i * PITCH:(i + 1) * PITCH, : r.shape[1]] = r

    n_chars = sum(len(t) for t in HAIKU)
    n_syms = sum(len(u) for u in enc)
    f = ImageFont.truetype(DEJAVU, 400)
    latin_w = sum(f.getlength(t) / 400 * EM for t in HAIKU)
    gw = sum(glyph_w)
    yb = yc + 80
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
    # compressed output: the whole haiku as one block of 2x2-px cells
    text = " ".join(HAIKU)
    cols = 17                        # 68 = 17 x 4, a full rectangle
    rows = -(-len(text) // cols)
    block = np.full((rows * b, cols * b, 3), 255, np.uint8)
    for k, ch in enumerate(text):
        r, c = divmod(k, cols)
        block[r * b:(r + 1) * b, c * b:(c + 1) * b] = pal[idx[ch]]
    yo = yc + 60
    d.text((rx, yo), "Compressed output: the whole haiku in one block", font=ui(28, "SemiBold"), fill=INK, anchor="lt")
    zoom = 12
    big = Image.fromarray(block).resize((block.shape[1] * zoom, block.shape[0] * zoom), Image.NEAREST)
    img.paste(big, (rx, yo + 50))
    d.rectangle([rx - 1, yo + 49, rx + big.width, yo + 50 + big.height], outline=FAINT, width=2)
    tx = rx + big.width + 50
    d.text((tx, yo + 60), f"{block.shape[1]} \u00d7 {block.shape[0]} pixels", font=ui(40, "Bold"), fill=AFTER, anchor="lt")
    d.text((tx, yo + 116), f"{len(text)} characters incl. spaces (shown {zoom // 2}\u00d7)", font=ui(26), fill=MUTED, anchor="lt")
    d.text((tx, yo + 166), "actual size:", font=ui(24), fill=MUTED, anchor="lt")
    act = Image.fromarray(block).resize((block.shape[1] * 2, block.shape[0] * 2), Image.NEAREST)
    img.paste(act, (tx + 150, yo + 166))
    ye = yo + 50 + max(big.height, 170)

    n_chars = sum(len(t) for t in HAIKU)
    f = ImageFont.truetype(DEJAVU, 400)
    latin_area = sum(f.getlength(t) / 400 * EM for t in HAIKU) * PITCH
    yb = ye + 70
    d.line([lx, yb - 24, W - 70, yb - 24], fill=FAINT, width=2)
    y = bars(d, lx, yb, 1120, [
        ("Plain text", latin_area, BEFORE, ""),
        ("2×2 colour blocks", n_chars * b * b, AFTER, f"  {n_chars * b * b / latin_area - 1:+.0%}"),
    ], "px²")
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

    yb = ytop + comp.height + 140
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
