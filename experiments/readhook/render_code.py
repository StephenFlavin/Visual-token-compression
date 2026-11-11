"""Render a text file's lines as one PNG page: monospace, line numbers, columns, long lines wrapped."""
import math, sys
from PIL import Image, ImageDraw, ImageFont
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

def render(lines, first_no, out, W=1932, H=1876, em=13.8, col_chars=100, header=""):
    f = ImageFont.truetype(FONT, round(em))
    cw = f.getlength("M"); pitch = math.ceil(em * 1.17)
    nw = 5                                       # line-number gutter, chars
    colw = (col_chars + nw + 1) * cw
    ncol = max(1, int((W - 16) // (colw + 12)))
    rows = (H - 8 - pitch) // pitch              # first row holds the header
    # visual rows: (number or None, text)
    vis = []
    for i, l in enumerate(lines):
        l = l.rstrip("\n").expandtabs(4)
        chunks = [l[j:j + col_chars] for j in range(0, max(1, len(l)), col_chars)] or [""]
        for k, c in enumerate(chunks):
            vis.append((first_no + i if k == 0 else None, c, i))
    cap = rows * ncol
    used = vis[:cap]
    if len(vis) > cap:                           # never cut a line in half
        last = used[-1][2]
        if len(vis) > cap and vis[cap][2] == last:
            used = [v for v in used if v[2] < last]
    n_lines = (used[-1][2] + 1) if used else 0
    h = min(H, 8 + pitch * (1 + math.ceil(len(used) / ncol) if ncol == 1 else 1 + min(rows, len(used))))
    h = max(56, math.ceil(h / 28) * 28)
    img = Image.new("L", (W, h), 255); d = ImageDraw.Draw(img)
    d.text((8, 4), header.format(a=first_no, b=first_no + n_lines - 1), font=f, fill=0)
    for idx, (no, txt, _) in enumerate(used):
        c, r = divmod(idx, rows)
        x = 8 + c * (colw + 12); y = 4 + pitch * (r + 1)
        if no is not None:
            d.text((x, y), f"{no:>{nw}}", font=f, fill=110)
        d.text((x + (nw + 1) * cw, y), txt, font=f, fill=0)
    img.save(out)
    return n_lines, img.size
