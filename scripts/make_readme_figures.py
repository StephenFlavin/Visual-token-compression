"""Draw the README figures (docs/img/*.png).

Demo figures, for one short haiku (need torch to read the glyph bank, and the fonts from
scripts/fetch_fonts.sh; Inter is optional, labels fall back to DejaVu Sans):
1. glyphs.png   plain text vs the learned whole-pixel glyph alphabet (N=300 chunks)
2. colour.png   plain text vs one coloured 2x2-pixel block per character (encoder_probe col_b2)
3. rgb.png      three lines of text vs the same three lines overlaid in the R, G and B channels (rgb3)
Every picture is drawn at one zoom factor per figure, so widths and areas compare honestly; each figure
also shows the real 1:1 pixels ("actual size").

Charts (need only matplotlib; every number is read from encoder_probe/results/*/runs.jsonl):
4. readme_density.png  chars per vision token vs whether the code is read, per encoder and stage
5. readme_r3_1.png     R3-1 short budget: CER and MC accuracy per arm next to its blind control

    python3 scripts/make_readme_figures.py                 # all five
    python3 scripts/make_readme_figures.py --only charts   # just the two charts
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


# --------------------------------------------------------------------------- charts: shared data and style
RES = ROOT / "encoder_probe" / "results"
TEXT_CPT = 4.33        # chars per text token, Qwen tokenizer on held-out WikiText (docs/CONTEXT_BUDGET.md)
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
            va="center")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["demos", "charts"], help="draw only the demo figures or only the charts")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only != "charts":
        fig_glyphs()
        fig_colour()
        fig_rgb()
    if args.only != "demos":
        chart_density()
        chart_r3_1()
