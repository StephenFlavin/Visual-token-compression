"""Draw the README figures (docs/img/*.png).

Demo figures, for one short haiku (need torch to read the glyph bank, and the fonts from
scripts/fetch_fonts.sh; Inter is optional, labels fall back to DejaVu Sans):
1. glyphs.png   plain text vs the learned whole-pixel glyph alphabet (N=300 chunks)
2. colour.png   plain text vs one coloured 2x2-pixel block per character (encoder_probe col_b2)
3. rgb.png      three lines of text vs the same three lines overlaid in the R, G and B channels (rgb3)
4. moonshot.png codes 1 + 3: each line in learned glyphs, then the three glyph lines overlaid (rgb3_lg_pb_h5)
Every picture is drawn at one zoom factor per figure, so widths and areas compare honestly; each figure
also shows the real 1:1 pixels ("actual size").

Charts (need only matplotlib; every number is read from encoder_probe/results/*/runs.jsonl):
5. readme_density.png  chars per vision token vs whether the code is read, per encoder and stage
6. readme_r3_1.png     R3-1 short budget: CER and MC accuracy per arm next to its blind control
7. claude_qa_saving.png  Claude QA v2: accuracy from page images vs from text, by token saving and effort
8. claude_qa_kinds.png   Claude QA v2 at ~4x fewer tokens: accuracy per question type
9. claude_qa_cost.png    Claude QA v2: list-price cost per call vs accuracy, by effort
   (7-9 read results/claude_qa/analysis_v2/summary.json, written by experiments/analyse_claude_qa.py)

    python3 scripts/make_readme_figures.py                 # all
    python3 scripts/make_readme_figures.py --only charts   # just the charts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "learned_glyphs"))

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



# --------------------------------------------------------------------------- figure 1: learned glyphs
def fig_glyphs():
    """Learned whole-pixel alphabet (N=300 chunks): every symbol is a learned 5-px bitmap; frequent chunks
    get ONE symbol, all other text is spelled with single-letter symbols. A key under each cell says what it
    stands for."""
    import torch
    from lg.alphabet import Alphabet

    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    syms, ink, wss, ss = bank["symbols"], bank["ink"].numpy(), bank["widths_ss"].numpy(), bank["ss"]
    alpha = Alphabet(syms[95:])
    z = 22
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

    img, d = new_card(1900)
    lx, rx, ty = 70, 1240, 70
    panel_title(d, lx, ty, "BEFORE", "Plain text", BEFORE)
    panel_title(d, rx, ty, "AFTER", "Learned glyphs", AFTER)
    y0 = ty + 110
    pitch = PITCH * z
    key_h = 40
    row = pitch + key_h + 10
    # BEFORE lines are spaced like the AFTER rows so each line sits opposite its glyphs
    f = ImageFont.truetype(DEJAVU, round(EM * z))
    for li, t in enumerate(HAIKU):
        top = y0 + li * row
        for s0, e, tint in hl[li]:
            x0, x1 = lx + f.getlength(t[:s0]), lx + f.getlength(t[:e])
            d.rounded_rectangle([x0 + 1, top + 8, x1 - 1, top + pitch - 8], radius=14, fill=tint)
        d.text((lx, top + 0.9 * EM * z), t, font=f, fill=INK, anchor="ls")
    lw = max(f.getlength(t) for t in HAIKU)
    arrow(d, (lx + lw + rx) / 2 - 10, y0 + row + pitch / 2)

    keyf, keyf_small = ui(24, "Medium"), ui(17, "SemiBold")
    real = np.full((PITCH * 3, 40), 255, np.uint8)
    glyph_w = []
    for li, ids in enumerate(segs):
        x, top, k = rx, y0 + li * row, 0
        row_real = []
        for i in ids:
            bm = cell_px(i)
            w = bm.shape[1]
            tok = i in tint_of
            bg = tint_of[i] if tok else (CELL_A if k % 2 == 0 else CELL_B)
            k += not tok
            d.rounded_rectangle([x + 2, top + 2, x + w * z - 3, top + pitch - 3], radius=6, fill=bg)
            for r, c in zip(*np.nonzero(bm)):
                px, py = x + c * z, top + r * z
                d.rounded_rectangle([px + 2, py + 2, px + z - 2, py + z - 2], radius=3, fill=INK)
            label = syms[i].replace(" ", "\u2423")
            fk = keyf_small if len(label) > 3 else keyf
            if tok:
                d.rounded_rectangle([x + 2, top + pitch + 4, x + w * z - 3, top + pitch + key_h], radius=8, fill=bg)
            d.text((x + w * z / 2, top + pitch + 4 + key_h / 2), label, font=fk, fill=INK if tok else MUTED, anchor="mm")
            x += w * z
            row_real.append(np.where(bm, 0, 255).astype(np.uint8))
        glyph_w.append((x - rx) / z)
        line = np.concatenate(row_real, 1)
        real[li * PITCH:(li + 1) * PITCH, : line.shape[1]] = line

    n_chars = sum(len(t) for t in HAIKU)
    n_syms = sum(len(s) for s in segs)
    fl = ImageFont.truetype(DEJAVU, 400)
    latin_w = sum(fl.getlength(t) / 400 * EM for t in HAIKU)
    gw = sum(glyph_w)
    yc = y0 + 3 * row + 4
    d.text((lx, yc), "coloured = frequent chunks (from a 300-entry dictionary)", font=ui(24), fill=MUTED, anchor="lt")
    d.text((rx, yc), "coloured = ONE glyph for a whole chunk;", font=ui(24), fill=MUTED, anchor="lt")
    d.text((rx, yc + 32), "grey = one glyph per letter (also learned)", font=ui(24), fill=MUTED, anchor="lt")

    # the same glyph text with no key, tints or gaps, at the same scale as the plain text
    yk = yc + 100
    d.text((rx, yk), "Without the key (same scale as the plain text):", font=ui(26, "SemiBold"), fill=INK, anchor="lt")
    d.text((lx, yk), "Plain text at the same scale:", font=ui(26, "SemiBold"), fill=INK, anchor="lt")
    yk += 50
    for li, t in enumerate(HAIKU):
        d.text((lx, yk + li * pitch + 0.9 * EM * z), t, font=f, fill=INK, anchor="ls")
    for r, c in zip(*np.nonzero(real < 128)):
        px, py = rx + c * z, yk + r * z
        d.rounded_rectangle([px + 1, py + 1, px + z - 1, py + z - 1], radius=3, fill=INK)
    yb = yk + PITCH * 3 * z + 70
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
    from encoder_probe.ep.pages import PRINTABLE, palette

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
    d.text((tx, yo + 116), f"{len(text)} chars incl. spaces, shown {zoom // 2}\u00d7", font=ui(26), fill=MUTED, anchor="lt")
    ye = yo + 50 + max(big.height, 150)

    n_chars = sum(len(t) for t in HAIKU)
    f = ImageFont.truetype(DEJAVU, 400)
    latin_area = sum(f.getlength(t) / 400 * EM for t in HAIKU) * PITCH
    yb = ye + 70
    d.line([lx, yb - 24, W - 70, yb - 24], fill=FAINT, width=2)
    y = bars(d, lx, yb, 1120, [
        ("Plain text", latin_area, BEFORE, ""),
        ("2×2 colour blocks", n_chars * b * b, AFTER, f"  {n_chars * b * b / latin_area - 1:+.0%}"),
    ], "px²")
    actual_size(img, d, 1300, yb - 6, [(latin_real(HAIKU), "plain"), (block, "colour")])
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
    """rgb3 compositing: stream c darkens only channel c (white page; R text shows as cyan ink)."""
    return np.stack([255 - m for m in masks], 2).astype(np.uint8)


def overlay_light(masks):
    """The same page inverted for display: stream c lights only channel c on black, so the R line is red.
    Same information as ``overlay`` (255 - it)."""
    return 255 - overlay(masks)


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
    comp = Image.fromarray(overlay_light(masks))
    ytop = y0 + pitch + 14
    d.rectangle([rx - 1, ytop - 1, rx + comp.width, ytop + comp.height], outline=FAINT, width=2)
    img.paste(comp, (rx, ytop))
    for k, line in enumerate(["each line lights up only its own channel; overlaps mix like light",
                              "(red + green = yellow, all three = white). Shown on black: the experiment",
                              "uses the inverted, white-page version (identical information)"]):
        d.text((rx, ytop + comp.height + 18 + 32 * k), line, font=ui(24), fill=MUTED, anchor="lt")

    yb = ytop + comp.height + 170
    d.line([lx, yb - 24, W - 70, yb - 24], fill=FAINT, width=2)

    real3 = overlay_light(mono_masks(1, ss=4))
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


# --------------------------------------------------------------------------- figure 4: the moonshot (codes 1 + 3)
CHAN = [("R", (220, 38, 38)), ("G", (22, 163, 74)), ("B", (37, 99, 235))]


def glyph_line_bitmaps():
    """Each haiku line in the learned whole-pixel alphabet (same bank and encoding as glyphs.png):
    -> list of 5 x w bool arrays (True = ink), one per line."""
    import torch
    from lg.alphabet import Alphabet

    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    syms, ink, wss, ss = bank["symbols"], bank["ink"].numpy(), bank["widths_ss"].numpy(), bank["ss"]
    alpha = Alphabet(syms[95:])

    def cell_px(i):
        w = int(wss[i]) // ss
        return ink[i][:, : int(wss[i])].reshape(PITCH, ss, w, ss).mean((1, 3)) > 0.5

    return [np.concatenate([cell_px(i) for i in alpha.encode(t)], 1) for t in HAIKU]


def fig_moonshot():
    """rgb3_lg_pb_h5: each line becomes learned glyphs (code 1), then the three glyph lines share one strip,
    one per colour channel (code 3). Shown as light on black like rgb.png; the experiment uses the white page."""
    z = 18
    lines = glyph_line_bitmaps()
    wmax = max(b.shape[1] for b in lines)
    masks = [np.pad(b, ((0, 0), (0, wmax - b.shape[1]))) for b in lines]
    light = np.stack(masks, 2).astype(np.uint8) * 255            # stream c lights channel c

    img, d = new_card(1600)
    lx, rx, ty = 70, 1100, 70
    panel_title(d, lx, ty, "BEFORE", "Three lines of text", BEFORE)
    panel_title(d, rx, ty, "AFTER", "Glyphs, one line per channel", AFTER)
    y0 = ty + 110
    pitch = PITCH * z
    gap = 22
    f = ImageFont.truetype(DEJAVU, round(EM * z))
    sub = ui(26, "SemiBold")

    # BEFORE: the haiku, one channel chip per line
    for i, t in enumerate(HAIKU):
        top = y0 + 50 + i * (pitch + gap)
        tag, col = CHAN[i]
        d.rounded_rectangle([lx, top + 26, lx + 44, top + pitch - 26], radius=10, fill=col)
        d.text((lx + 22, top + pitch / 2), tag, font=ui(26, "Bold"), fill=(255, 255, 255), anchor="mm")
        d.text((lx + 66, top + 0.9 * EM * z), t, font=f, fill=INK, anchor="ls")
    lw = 66 + max(f.getlength(t) for t in HAIKU)

    # AFTER, step 1: each line in learned glyphs (code 1), ink in its channel's colour
    d.text((rx, y0), "1. Each line in learned glyphs", font=sub, fill=INK, anchor="lt")
    for i, bm in enumerate(lines):
        top = y0 + 50 + i * (pitch + gap)
        col = CHAN[i][1]
        d.rounded_rectangle([rx, top, rx + bm.shape[1] * z, top + pitch], radius=8, fill=CELL_A)
        for r, c in zip(*np.nonzero(bm)):
            px, py = rx + c * z, top + r * z
            d.rounded_rectangle([px + 2, py + 2, px + z - 2, py + z - 2], radius=3, fill=col)
    arrow(d, (lx + lw + rx) / 2 - 10, y0 + 50 + 1.5 * pitch + gap)

    # AFTER, step 2: the three glyph lines overlaid, one per channel (code 3)
    yo = y0 + 50 + 3 * (pitch + gap) + 30
    d.text((rx, yo), "3. Overlay them, one per colour channel", font=sub, fill=INK, anchor="lt")
    ys = yo + 50
    d.rectangle([rx, ys, rx + wmax * z, ys + pitch], fill=(0, 0, 0))
    for r in range(PITCH):
        for c in range(wmax):
            rgb = tuple(int(v) for v in light[r, c])
            if any(rgb):
                px, py = rx + c * z, ys + r * z
                d.rounded_rectangle([px + 2, py + 2, px + z - 2, py + z - 2], radius=3, fill=rgb)
    notes = ["each glyph line lights up only its own channel; overlaps mix like light",
             "(red + green = yellow, all three = white). Shown on black: the experiment",
             "uses the inverted, white-page version (identical information)"]
    for k, line in enumerate(notes):
        d.text((rx, ys + pitch + 18 + 32 * k), line, font=ui(24), fill=MUTED, anchor="lt")
    ye = ys + pitch + 18 + 32 * len(notes)

    # bars: plain text area -> glyphs (code 1) -> moonshot strip (codes 1 + 3), all at the 5-px pitch
    fl = ImageFont.truetype(DEJAVU, 400)
    plain = sum(fl.getlength(t) / 400 * EM for t in HAIKU) * PITCH
    glyphs = sum(b.shape[1] for b in lines) * PITCH
    moon = wmax * PITCH
    yb = ye + 70
    d.line([lx, yb - 24, W - 70, yb - 24], fill=FAINT, width=2)
    y = bars(d, lx, yb, 1120, [
        ("Plain text", plain, BEFORE, ""),
        ("Glyphs (code 1)", glyphs, (129, 140, 248), f"  {glyphs / plain - 1:+.0%}"),
        ("Moonshot (1 + 3)", moon, AFTER, f"  {moon / plain - 1:+.0%}"),
    ], "px²")
    real_light = light.copy()
    actual_size(img, d, 1300, yb - 6, [(latin_real(HAIKU), "plain"),
                                       (np.where(np.concatenate(masks, 0), 0, 255).astype(np.uint8), "glyphs"),
                                       (real_light, "moonshot")])
    img = img.crop((0, 0, W, y + 70))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, img.height - 1], radius=36, outline=FAINT, width=3)
    img.save(OUT / "moonshot.png", optimize=True)
    n = sum(len(t) for t in HAIKU)
    print(f"moonshot: {n} chars, {plain / n:.1f} -> {glyphs / n:.1f} (glyphs) -> {moon / n:.1f} px^2/char")


# --------------------------------------------------------------------------- charts: shared data and style
RES = ROOT / "encoder_probe" / "results"
TEXT_CPT = 4.33        # chars per text token, Qwen tokenizer on held-out WikiText (docs/CONTEXT_BUDGET.md)
# DeepSeek-OCR (arXiv 2510.18234): ~97% OCR precision at <= 10x compression (text tokens per vision token, Fox
# benchmark). Converted to chars per vision token with TEXT_CPT (Qwen's tokenizer on WikiText; DeepSeek's own
# chars per token on Fox are not published), so the x position is approximate.
DEEPSEEK_OCR_RATIO = 10
DEEPSEEK_OCR_CPT = DEEPSEEK_OCR_RATIO * TEXT_CPT
PROBE_FLOOR = 0.73     # unigram-level CER of this text, "about 72-75%" (encoder_probe/RESULTS.md)
READ_CER = 0.05        # "read": CER below 5%
PARTLY_GAP = 0.25      # "partly read": CER at least 25 points below the floor (probe floor or the arm's blind run)

# validated categorical slots 1-5 (dataviz reference palette, light surface); shapes carry family too
FAMILY = {  # family -> (label, colour, marker)
    "latin": ("Latin text", "#2a78d6", "o"),
    "colour": ("colour blocks", "#eb6834", "s"),
    "rgb3": ("RGB overlay (Latin)", "#1baf7a", "^"),
    "lg": ("learned glyphs", "#eda100", "D"),
    "moon": ("moonshot (RGB overlay of glyphs)", "#e87ba4", "*"),
}
FAM_OF = {"latin": "latin", "mono": "latin", "colour": "colour", "rgb3": "rgb3", "lg_free": "lg", "lg_pb": "lg",
          "rgb3_lg_pb": "moon"}
FAM_DY = {"latin": -0.12, "colour": -0.04, "rgb3": 0.04, "lg": 0.12, "moon": 0.0}  # fixed nudge so ties don't hide
SURFACE, INK_1, INK_2, MUTED_C, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLIND_C = "#b5b3ab"


def rows(*runs: str) -> list[dict]:
    out = []
    for run in runs:
        with open(RES / run / "runs.jsonl") as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


def pick(rs: list[dict], **match) -> dict[str, dict]:
    """One row per encoding among rows matching `match`: prefer rescored rows (score_v 2), then the most steps,
    then the latest row. This is how the docs quote them."""
    best: dict[str, dict] = {}
    for r in rs:
        if any(r.get(k) != v for k, v in match.items()):
            continue
        key = (r.get("score_v") == 2, r.get("steps", 0))
        e = r["encoding"]
        if e not in best or key >= (best[e].get("score_v") == 2, best[e].get("steps", 0)):
            best[e] = r
    return best


def status(cer: float, floor: float) -> str:
    if cer < READ_CER:
        return "read"
    if cer <= floor - PARTLY_GAP:
        return "partly"
    return "lost"


def save_small(fig, name: str):
    """Save at 150 dpi, then quantise to a 64-colour palette PNG (flat charts lose nothing; keeps files small)."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, format="png")
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(64, method=Image.Quantize.MEDIANCUT).save(OUT / name, optimize=True)


def chart_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK_1, "axes.titlecolor": INK_1,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


# --------------------------------------------------------------------------- chart 1: density vs readability
def chart_density():
    """One strip per (encoder, stage). x = chars per vision token OF THAT ENCODER (SigLIP: 16-px tokens of a
    224-px page; DeepEncoder: 64 tokens of a 512-px page; Qwen2.5-VL: 256 merged 28-px tokens of a 448-px page),
    so one rendering sits at different x for different encoders."""
    plt = chart_style()
    from matplotlib.lines import Line2D

    sig = rows("main", "controls")
    r2_raw, r2_enc = rows("round2_gpu_tp200"), rows("round2_gpu_tp1000")
    q_raw, g2 = rows("round3_cpu_tp200"), rows("g2lite")
    ft, ftb = rows("round3_ft_gpu3h"), rows("round3_ft_gpu3h_blind")

    strips = []  # (group, label, points); point = (cpt, cer, family, status, encoding)

    def probe_points(sel, cpt=lambda r: r["chars_per_token"]):
        return [(cpt(r), r["cer"], FAM_OF[r["family"]], status(r["cer"], PROBE_FLOOR), e) for e, r in sel.items()]

    strips.append(("SigLIP-B/16", "raw pixels (ceiling)", probe_points(pick(sig, encoder="identity", cond="clean"))))
    strips.append(("SigLIP-B/16", "frozen encoder, layer 6", probe_points(pick(sig, encoder="siglip_l6", cond="clean"))))
    # identity32 rows store chars per 32-px patch; the docs quote them per DeepEncoder token (4096 px^2 each)
    strips.append(("DeepSeek-OCR DeepEncoder", "raw pixels (ceiling)",
                   probe_points(pick(r2_raw, encoder="identity32", cond="clean"), cpt=lambda r: 4096 / r["px_per_char"])))
    strips.append(("DeepSeek-OCR DeepEncoder", "frozen encoder",
                   probe_points(pick(r2_enc, encoder="deepencoder", cond="clean"))))
    strips.append(("Qwen2.5-VL-3B", "raw pixels (ceiling)",
                   probe_points(pick(q_raw, encoder="identity28", cond="clean"))))
    # G2-lite: the moonshot's best arm (no merger, "direct"); the probe under-reads Latin here, so no Latin point
    gd = pick(g2, encoding="rgb3_lg_pb_h5", arm="direct")
    strips.append(("Qwen2.5-VL-3B", "frozen ViT (probe, best arm)", probe_points(gd)))
    blind = {r["encoding"]: r for r in ftb if r["cond"] == "clean"}
    pts = []
    for r in ft:
        if r["cond"] != "clean":
            continue
        e = r["encoding"]
        floor = blind[e]["cer"] if e in blind else PROBE_FLOOR  # lg_pb_h5's blind run died before its eval
        pts.append((r["chars_per_token"], r["cer"], FAM_OF[r["family"]], status(r["cer"], floor), e))
    strips.append(("Qwen2.5-VL-3B", "LLM fine-tuned (LoRA, 1000 steps)", pts))

    # y layout: groups separated by a gap
    ys, y, prev = [], 0.0, None
    for g, _, _ in strips:
        if prev is not None:
            y += 1.0 if g == prev else 1.75
        ys.append(y)
        prev = g
    fig, ax = plt.subplots(figsize=(10.4, 6.4), dpi=150)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 9000)
    ax.set_ylim(ys[-1] + 0.7, -1.25)
    for x in (1, 10, 100, 1000):
        ax.axvline(x, color=GRID, lw=0.8, zorder=0)
    ax.axvline(TEXT_CPT, color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(TEXT_CPT * 1.08, -1.05, f"plain text tokens: {TEXT_CPT} chars/token", color=INK_2, fontsize=8.5,
            va="center", zorder=2, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    ax.axvline(DEEPSEEK_OCR_CPT, color=INK_1, lw=1.4, ls=(0, (1, 2.5)), zorder=1)
    ax.text(DEEPSEEK_OCR_CPT * 1.08, -0.5, f"DeepSeek-OCR: ~97% precision at {DEEPSEEK_OCR_RATIO}x\n"
            f"compression ≈ {DEEPSEEK_OCR_CPT:.0f} chars/token", color=INK_1, fontsize=8.5, va="center",
            linespacing=1.15)
    fill = {"read": "full", "partly": "left", "lost": "none"}
    for (g, label, points), yy in zip(strips, ys):
        ax.axhline(yy, color=GRID, lw=0.6, zorder=0)
        for cpt, cer, fam, st, e in points:
            _, col, mk = FAMILY[fam]
            ax.plot(cpt, yy + FAM_DY[fam], marker=mk, ms=13 if mk == "*" else 8.5, mew=1.6, color=col,
                    mec=col, mfc=col if st != "lost" else SURFACE, fillstyle=fill[st], mfcalt=SURFACE,
                    ls="none", zorder=3)
    ax.set_yticks(ys, [s[1] for s in strips], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    starts = [i for i, s in enumerate(strips) if i == 0 or strips[i - 1][0] != s[0]]
    for i in starts:
        ax.text(-0.012, ys[i] - 0.62, strips[i][0], transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9.5, fontweight="bold", color=INK_1)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.set_xlabel("characters per vision token of that encoder (log scale; higher = denser)")

    def note(i, enc, text, dx=1.12, dy=-0.36, ha="left"):
        for cpt, cer, fam, _, e in strips[i][2]:
            if e == enc:
                ax.text(cpt * dx, ys[i] + dy, text, fontsize=8, color=INK_2, ha=ha, va="center")
    def cer_of(i, enc):
        return next(p[1] for p in strips[i][2] if p[4] == enc)

    def span(i, above):  # CER range of the points denser than `above` chars/token
        c = [p[1] for p in strips[i][2] if p[0] > above]
        return f"{min(c):.0%}-{max(c):.0%}"
    note(1, "latin_p16", f"Latin {cer_of(1, 'latin_p16'):.1%} CER at 2.1", dx=0.9, ha="right")
    note(1, "col_b1", f"everything above 10: {span(1, 10)}", dx=0.6, ha="center")
    raw_max = max(p[1] for p in strips[2][2])
    note(2, "col_b1", f"all < {np.ceil(raw_max * 100):.0f}% CER", dx=1.0, ha="center")
    note(3, "latin_p32", f"Latin {cer_of(3, 'latin_p32'):.0%} at 8.5", dx=1.0, ha="center")
    note(3, "col_b2", f"everything denser: {span(3, 9)}", dx=0.6, ha="center")
    note(4, "rgb3_lg_pb_h5", f"moonshot {cer_of(4, 'rgb3_lg_pb_h5'):.2%} at 322", dx=1.0, ha="center")
    gb = pick(rows("g2lite_blind"), encoding="rgb3_lg_pb_h5")["rgb3_lg_pb_h5"]["cer"]
    note(5, "rgb3_lg_pb_h5", f"{cer_of(5, 'rgb3_lg_pb_h5'):.0%} (empty-output blind floor {gb:.1%})", dx=1.0,
         ha="center")
    note(6, "latin_p8", f"Latin {cer_of(6, 'latin_p8'):.1%} (blind {blind['latin_p8']['cer']:.1%})", dx=1.0,
         ha="center")
    note(6, "lg_pb_h5", "the denser three: at their blind floors", dx=1.4, ha="center", dy=0.42)

    fam_handles = [Line2D([], [], marker=mk, color=col, ls="none", ms=12 if mk == "*" else 7.5, label=lab)
                   for lab, col, mk in FAMILY.values()]
    st_handles = [Line2D([], [], marker="o", color=INK_2, mfc=INK_2, ls="none", ms=7.5, label="read (CER < 5%)"),
                  Line2D([], [], marker="o", color=INK_2, fillstyle="left", mfcalt=SURFACE, ls="none", ms=7.5,
                         label="partly read (≥ 25 points below its floor)"),
                  Line2D([], [], marker="o", color=INK_2, mfc=SURFACE, ls="none", ms=7.5, mew=1.6,
                         label="not read (at or near its floor)")]
    leg1 = ax.legend(handles=fam_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                     fontsize=8.5, title="code", title_fontsize=9, alignment="left")
    ax.add_artist(leg1)
    ax.legend(handles=st_handles, loc="upper left", bbox_to_anchor=(1.01, 0.62), frameon=False, fontsize=8.5,
              title="how well it is read", title_fontsize=9, alignment="left")
    fig.suptitle("Every code reads from raw pixels; through vision encoders only low-density Latin survives",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.925, "Floor: about 73% CER (letter statistics) for the small probes, the arm's own blind "
             "control for the fine-tuned LLM. Probe CERs are upper bounds (probes are data-limited).",
             fontsize=8.5, color=INK_2)
    fig.subplots_adjust(left=0.22, right=0.72, top=0.88, bottom=0.09)
    save_small(fig, "readme_density.png")
    plt.close(fig)
    print("readme_density.png:", [(s[1], len(s[2])) for s in strips])


# --------------------------------------------------------------------------- chart 2: R3-1 short budget
def chart_r3_1():
    plt = chart_style()
    arms = ["latin_p8", "latin_p5", "lg_pb_h5", "rgb3_lg_pb_h5"]
    names = {"latin_p8": "Latin p8", "latin_p5": "Latin p5", "lg_pb_h5": "learned glyphs\nlg_pb_h5",
             "rgb3_lg_pb_h5": "moonshot\nrgb3_lg_pb_h5"}
    real = {r["encoding"]: r for r in rows("round3_ft_gpu3h") if r["cond"] == "clean"}
    blind = {r["encoding"]: r for r in rows("round3_ft_gpu3h_blind") if r["cond"] == "clean"}
    labels = [f"{names[a]}\n{real[a]['chars_per_token']:.0f} chars/token" for a in arms]
    x = np.arange(len(arms))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=150)
    for ax, key, title, fmt, top in ((axes[0], "cer", "Character error rate (lower is better)", "{:.1%}", 1.0),
                                     (axes[1], "mc_acc", "Multiple-choice accuracy (higher is better)", "{:.2f}", 1.0)):
        rv = [real[a][key] for a in arms]
        bv = [blind[a][key] if a in blind else np.nan for a in arms]
        ax.bar(x - w / 2 - 0.01, rv, w, color=FAMILY["latin"][1], label="real page images", zorder=2)
        ax.bar(x + w / 2 + 0.01, bv, w, color=BLIND_C, label="blind control (same training, images shuffled between pages)", zorder=2)
        for i, (a, b) in enumerate(zip(rv, bv)):
            ax.text(x[i] - w / 2 - 0.01, a + 0.015, fmt.format(a), ha="center", va="bottom", fontsize=8.5,
                    color=INK_1)
            if np.isnan(b):
                ax.text(x[i] + w / 2 + 0.03, 0.02, "no blind\neval (run\ndied at\nstep 500)", ha="center",
                        va="bottom", fontsize=7, color=INK_2)
            else:
                ax.text(x[i] + w / 2 + 0.01, b + 0.015, fmt.format(b), ha="center", va="bottom", fontsize=8.5,
                        color=INK_2)
        if key == "mc_acc":
            ax.axhline(0.25, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=3, label="chance (1 in 4)")
        ax.set_ylim(0, top)
        ax.set_xticks(x, labels, fontsize=8.5)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10, loc="left", color=INK_1)
        if key == "cer":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    hc = [h for h, lab in zip(h1, l1) if lab.startswith("chance")]
    fig.legend(h0 + hc, l0 + ["chance (1 in 4)"], loc="lower left", bbox_to_anchor=(0.05, 0.0), ncol=3,
               frameon=False, fontsize=8.5)
    r0 = real[arms[0]]
    fig.suptitle(f"R3-1, short budget: Qwen2.5-VL-3B with LoRA, {r0['steps']} steps, {r0['train_pages']} training "
                 f"pages, {r0['test_pages']} clean test pages", x=0.02, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.02, 0.885, "Only Latin p8 is clearly read. Latin p5, the learned glyphs and the moonshot sit at "
             "their blind floors. No arm is near 3% CER, so gate G3 is undecided.", fontsize=8.5, color=INK_2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.22, wspace=0.18)
    save_small(fig, "readme_r3_1.png")
    plt.close(fig)
    print("readme_r3_1.png:", {a: (round(real[a]["cer"], 3), round(blind[a]["cer"], 3) if a in blind else None)
                               for a in arms})


# --------------------------------------------------------------------------- charts 7-8: Claude QA v2
QA_SUMMARY = ROOT / "results" / "claude_qa" / "analysis_v2" / "summary.json"
EFFORT = {"low": ("low effort", "#2a78d6", "o"), "high": ("high effort", "#eb6834", "s")}   # validated slots 1-2


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _qa_rows() -> dict:
    return json.loads(QA_SUMMARY.read_text())


def chart_claude_qa_saving():
    """Accuracy vs token saving, simple lookups and multi-step questions side by side, one line per effort.
    x = 1 is the text arm; each image point is paired with its document's text answers."""
    plt = chart_style()
    res = _qa_rows()
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 5.0), dpi=150, sharey=True)
    panels = [("Simple lookups", ("needle", "number"), "240 questions per point"),
              ("Multi-step questions", ("complex",), "120 per point: two-hop, three-hop, sums, largest-of-three, counts")]
    for ax, (title, kinds, sub) in zip(axs, panels):
        for e, (lab, col, mk) in EFFORT.items():
            pts = sorted((r["saving"], r) for k, r in res.items() if k.startswith(e + "/") and "saving" in r)
            t_k = sum(pts[0][1]["kinds"][k][0] for k in kinds)
            n = sum(pts[0][1]["kinds"][k][2] for k in kinds)
            xs, ys, los, his = [1.0], [t_k / n], *[[v] for v in wilson(t_k, n)]
            for sav, r in pts:
                i_k = sum(r["kinds"][k][1] for k in kinds)
                lo, hi = wilson(i_k, n)
                xs.append(sav), ys.append(i_k / n), los.append(lo), his.append(hi)
            dx = -0.035 if e == "low" else 0.035
            x = np.array(xs) + dx
            y = np.array(ys) * 100
            ax.plot(x, y, color=col, lw=2, zorder=2)
            ax.errorbar(x, y, yerr=[y - np.array(los) * 100, np.array(his) * 100 - y], fmt="none", ecolor=col,
                        elinewidth=1.2, capsize=0, zorder=2)
            ax.plot(x, y, ls="none", marker=mk, ms=8, color=col, mec=SURFACE, mew=1.5, zorder=3, label=lab)
            if kinds == ("complex",):   # the two lookup lines coincide: the legend names them there
                ax.text(x[-1] + 0.08, y[-1], lab, color=INK_2, fontsize=8.5, va="center")
        ax.set_xticks([1, 2.19, 2.87, 4.15], ["text", "2.2x", "2.9x", "4.2x"])
        ax.set_xlim(0.7, 4.9)
        ax.set_ylim(0, 104)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=16)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")
        ax.set_xlabel("fewer input tokens than the same document as text")
    axs[0].set_ylabel("questions answered correctly")
    axs[0].legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.suptitle("Claude answers as well from page images at 2.2x fewer tokens; at 4.2x multi-step questions fail",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.86, "Claude Opus 5.5, 3 documents of ~700k characters (~245k text tokens), DejaVu Sans pages "
             "of 1932 px; pre-registered, 95% Wilson intervals.\nAt 2.2x fewer tokens the accuracy gap to text is at "
             "most 1.6 points (95% bound) at both efforts; at 2.9x it is lossless at high effort.",
             fontsize=8.5, color=INK_2, linespacing=1.4)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.13, wspace=0.08)
    save_small(fig, "claude_qa_saving.png")
    plt.close(fig)
    print("claude_qa_saving.png")


def chart_claude_qa_kinds():
    """At 58.8 px²/char (~4.2x fewer tokens): accuracy per question type, text vs image at both efforts."""
    plt = chart_style()
    from matplotlib.lines import Line2D
    res = _qa_rows()
    low, high = res["low/58.8"]["kinds"], res["high/58.8"]["kinds"]
    rows_ = [("needle", "Invented fact, looked up"), ("number", "Number in a quoted sentence"), ("sum", "Sum of two facts"),
             ("max", "Largest of three facts"), ("two_hop", "Two-hop chain"), ("three_hop", "Three-hop chain"),
             ("count", "Count across the whole document")]
    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=150)
    for i, (k, lab) in enumerate(rows_):
        y = i
        n = low[k][2]
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
        t = low[k][0] / n * 100
        ax.plot(t, y, marker="|", ms=14, mew=2.2, color=INK_2, ls="none", zorder=3)
        for e, d, dy in (("low", low, -0.14), ("high", high, 0.14)):
            _, col, mk = EFFORT[e]
            v = d[k][1] / n * 100
            lo, hi = wilson(d[k][1], n)
            ax.plot([lo * 100, hi * 100], [y + dy] * 2, color=col, lw=1.2, zorder=2)
            ax.plot(v, y + dy, marker=mk, ms=8, color=col, mec=SURFACE, mew=1.5, ls="none", zorder=3)
        ax.text(101.5, y, f"n = {n}", fontsize=8, color=INK_2, va="center")
    ax.set_yticks(range(len(rows_)), [lab for _, lab in rows_], fontsize=9)
    ax.set_ylim(len(rows_) - 0.5, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("questions answered correctly")
    ax.axhline(1.5, color=AXIS, lw=0.8, ls=(0, (3, 3)))
    ax.text(1, 1.62, "multi-step", fontsize=8, color=INK_2, va="top")
    ax.text(1, 1.38, "lookups", fontsize=8, color=INK_2, va="bottom")
    handles = [Line2D([], [], marker="|", ms=12, mew=2.2, color=INK_2, ls="none", label="from text (low effort)")]
    handles += [Line2D([], [], marker=mk, ms=7.5, color=col, ls="none", label=f"from images, {lab}")
                for lab, col, mk in EFFORT.values()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("At 4.2x fewer tokens single facts survive; tying facts together across the document fails",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.905, "Claude Opus 5.5, images at 58.8 px²/char, pooled over 3 documents; 95% Wilson intervals. "
             "Text scores 94-100% on every type at both efforts.", fontsize=8.5, color=INK_2)
    fig.subplots_adjust(left=0.27, right=0.93, top=0.76, bottom=0.12)
    save_small(fig, "claude_qa_kinds.png")
    plt.close(fig)
    print("claude_qa_kinds.png")


# list prices (Claude Opus 5.5, per token) and the CLI's 5-minute prompt-cache write / read multipliers
QA_PRICE_IN, QA_PRICE_OUT, QA_CACHE_WRITE = 4e-6, 20e-6, 1.25


def _qa_uncached_cost(u: dict) -> float:
    """List-price cost of a call as if nothing came from the prompt cache (every input token written once, as the
    first call of a document is). Equals the logged total_cost_usd on every uncached v2 call."""
    tok = u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"]
    return tok * QA_PRICE_IN * QA_CACHE_WRITE + u["output_tokens"] * QA_PRICE_OUT


def chart_claude_qa_cost():
    """Cost per call vs accuracy, all questions and multi-step only, one line per effort: text and the three image
    densities. Costs are list price, as if uncached, averaged over the 3 documents."""
    plt = chart_style()
    res = _qa_rows()
    cost = {}
    for f in sorted((ROOT / "results" / "claude_qa").glob("qav2_*/runs.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("cached") or r["arm"] == "none":
                continue
            key = (r["effort"], "text" if r["arm"] == "text" else r["budget"])
            cost.setdefault(key, []).append(_qa_uncached_cost(r["usage"]))
    cost = {k: float(np.mean(v)) for k, v in cost.items()}
    lab = {"text": "text", 117.6: "2.2x", 88.2: "2.9x", 58.8: "4.2x"}
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 5.0), dpi=150, sharey=True)
    for ax, (title, kinds, sub) in zip(axs, [("All questions", ("all",), "360 per point"),
                                             ("Multi-step questions", ("complex",), "120 per point")]):
        where = {}
        for e, (elab, col, mk) in EFFORT.items():
            pts = []
            for b in ("text", 117.6, 88.2, 58.8):
                r = res[f"{e}/117.6"] if b == "text" else res[f"{e}/{b}"]
                j = 0 if b == "text" else 1
                k_ = sum(r["kinds"][k][j] for k in kinds)
                n = sum(r["kinds"][k][2] for k in kinds)
                pts.append((cost[(e, b)], k_ / n, *wilson(k_, n), lab[b]))
            pts.sort()
            x = np.array([p_[0] for p_ in pts])
            y = np.array([p_[1] for p_ in pts]) * 100
            lo = np.array([p_[2] for p_ in pts]) * 100
            hi = np.array([p_[3] for p_ in pts]) * 100
            ax.plot(x, y, color=col, lw=2, zorder=2)
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=col, elinewidth=1.2, capsize=0, zorder=2)
            ax.plot(x, y, ls="none", marker=mk, ms=8, color=col, mec=SURFACE, mew=1.5, zorder=3, label=elab)
            for xi, yi, p_ in zip(x, y, pts):
                where.setdefault(p_[4], []).append((e, xi, yi))
        for name, ps in where.items():   # one label above a pair that sits at the top, else one per point
            if all(yi >= 95 for _, _, yi in ps):
                ax.text(np.mean([xi for _, xi, _ in ps]), 102.5, name, ha="center", va="bottom", fontsize=8,
                        color=INK_2)
            else:
                for e, xi, yi in ps:
                    if yi >= 95:
                        ax.text(xi, 102.5, name, ha="center", va="bottom", fontsize=8, color=INK_2)
                    elif e == "low":
                        ax.text(xi, yi - 4.5, name, ha="center", va="top", fontsize=8, color=INK_2)
                    else:
                        ax.text(xi + 0.025, yi, name, ha="left", va="center", fontsize=8, color=INK_2)
        ax.set_xlim(0.25, 1.45)
        ax.set_ylim(0, 108)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4], ["$0.40", "$0.60", "$0.80", "$1.00", "$1.20", "$1.40"])
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=16)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")
        ax.set_xlabel("cost per call: one ~245k-token document plus 120 questions")
    axs[0].set_ylabel("questions answered correctly")
    axs[0].legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("Page images at 2.2x fewer tokens cost half as much as text for the same answers",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.86, "Claude Opus 5.5 list price (\\$4 / \\$20 per million input / output tokens), as if uncached, "
             "mean of 3 documents; 95% Wilson intervals.\nHigh effort adds thinking tokens: ~4-8 cents per call more. "
             "Calls drew on a subscription quota, so these are list prices, not amounts billed.",
             fontsize=8.5, color=INK_2, linespacing=1.4)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.13, wspace=0.08)
    save_small(fig, "claude_qa_cost.png")
    plt.close(fig)
    print("claude_qa_cost.png", {f"{k[0]}/{k[1]}": round(v, 3) for k, v in cost.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["demos", "charts"], help="draw only the demo figures or only the charts")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only != "charts":
        fig_glyphs()
        fig_colour()
        fig_rgb()
        fig_moonshot()
    if args.only != "demos":
        chart_density()
        chart_r3_1()
        chart_claude_qa_saving()
        chart_claude_qa_kinds()
        chart_claude_qa_cost()
