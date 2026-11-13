"""GlyphBank: one ink cell per symbol at supersampled resolution (WP-LG1).

Geometry (all cells share it, so composed lines follow vtc.layout conventions)
-------------------------------------------------------------------------------
* ``height_px`` = line pitch h (px at 1x); a cell is ``height_ss = h * ss`` rows tall and covers the
  line box only (the Phase-2 margin is added later by the Channel).
* Equivalent em = ``h / 1.2`` (``line_spacing`` = 1.2) unless ``em`` is given explicitly (the
  calibration passes the exact Phase-2 em, e.g. em 7 -> h = round(8.4) = 8).
* Baseline (font banks) at ``((line_spacing - 1)/2 + 0.8) * em`` from the top of the line box, i.e.
  ``0.9 * em * ss`` ss-rows: identical to ``LayoutParams.first_baseline`` minus the margin.
* Cell width: per symbol, ``widths_ss`` (ss pixels). Font banks use the glyph's advance at the
  supersampled em (``vtc.layout.glyph_advance``, rounded), i.e. proportional like Phase 2; ink
  outside [0, advance) (negative side bearings, e.g. 'j') is clipped (``clip_frac`` records it).

Representation
--------------
* learnable bank: ``logits`` (n, height_ss, max_width_ss); ``forward(tau, hard)`` = sigmoid(logits/tau)
  masked to each cell's width; ``hard=True`` -> straight-through binarisation (value = (p > .5),
  gradient = sigmoid's).
* fixed bank (``from_font(..., learnable=False)``, the default): the rendered coverage is stored
  as a buffer ``fixed_ink`` and returned unchanged (``tau``/``hard`` are ignored), so the
  calibration bank is exactly PIL's antialiased coverage at ss resolution.

Anti-degeneracy helpers (PLAN amendments; optional losses for the joint trainer)
--------------------------------------------------------------------------------
* guard band: learnable / random banks force zero ink in the outer ``guard_ss = (gx, gy)`` ss-px of
  every cell (default (1, 1): adjacent glyphs are >= 2 ss-px = 0.5 px apart and ink never touches
  the line box edge). Font banks use guard 0 (fonts have their own side bearings).
* ``min_feature_penalty(ink, ss)``: |x - open(x)| + |close(x) - x| with an ss x ss square
  structuring element (soft morphology via max-pool): zero iff every stroke, dot, gap and hole is at
  least one OUTPUT pixel wide.
* ``coverage(ink, widths_ss)``: per-glyph mean ink over its cell; ``coverage_penalty`` keeps it in
  [lo, hi] (default 0.10-0.45).
* ``GlyphBank.random_binary(...)``: frozen random binary codebook (Bernoulli at output-pixel
  resolution, nearest-upsampled to ss) - the "random code" baseline.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vtc import config
from vtc.layout import get_font, glyph_advance

LINE_SPACING = 1.2


def em_for_height(height_px: int) -> float:
    return height_px / LINE_SPACING


def baseline_ss(em: float, ss: int, line_spacing: float = LINE_SPACING) -> float:
    """Baseline row (ss pixels, fractional) inside the line box, margin excluded (vtc.layout)."""
    return ((line_spacing - 1) / 2 * em + 0.8 * em) * ss


def _cjk_dy(cjk_font: str) -> float:
    """Vertical shift (em, + = down) used by vtc SubstitutionScheme/CJKUnit (computed over the
    whole symbol pool, cjk_scale = 1)."""
    from vtc.schemes.substitution import ink_extent, load_symbols, vertical_shift
    top, bot = ink_extent(str(cjk_font), tuple(load_symbols()))
    return vertical_shift(top, bot, 1.0)


def render_font_cells(symbols: Sequence[str], font_path, height_px: int, ss: int = 4,
                      em: float | None = None, cjk_map: dict | None = None,
                      cjk_font=config.CJK_FONT) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Render each symbol with PIL at ss resolution, as vtc renders it at ``em``.

    -> (cells: list of float32 (height_ss, w_ss) coverage in [0,1], widths_ss, clip_frac).
    Tokens: if ``cjk_map`` maps the symbol to an ideograph it is drawn with ``cjk_font`` at a 1.0 em
    advance with the CJKUnit vertical shift; otherwise the string is drawn char by char with
    ``font_path`` (per-char advances, no kerning - exactly how Phase-2 Latin draws text).
    """
    from PIL import Image, ImageDraw

    em = em_for_height(height_px) if em is None else float(em)
    sem = em * ss
    H = height_px * ss
    y = baseline_ss(em, ss)
    pad = int(math.ceil(sem))                    # room for negative side bearings
    dy_cjk = _cjk_dy(str(cjk_font)) if cjk_map else 0.0
    cells, widths, clips = [], [], []
    for sym in symbols:
        ideo = cjk_map.get(sym) if cjk_map else None
        if ideo is not None:
            adv = 1.0 * sem
            draws = [(0.0, y + dy_cjk * sem, ideo, get_font(str(cjk_font), sem))]
        else:
            draws, x = [], 0.0
            f = get_font(str(font_path), sem)
            for ch in sym:
                if not ch.isspace():
                    draws.append((x, y, ch, f))
                x += glyph_advance(str(font_path), round(sem, 3), ch)
            adv = x
        w = max(1, int(round(adv)))
        img = Image.new("L", (w + 2 * pad, H + 2 * pad), 0)
        d = ImageDraw.Draw(img)
        for x, yy, ch, f in draws:
            d.text((pad + x, pad + yy), ch, font=f, fill=255, anchor="ls")
        a = np.asarray(img, dtype=np.float32) / 255.0
        cell = a[pad:pad + H, pad:pad + w]
        tot = float(a.sum())
        clips.append(0.0 if tot == 0 else 1.0 - float(cell.sum()) / tot)
        cells.append(np.ascontiguousarray(cell))
        widths.append(w)
    return cells, widths, clips


# ------------------------------------------------------------------------- anti-degeneracy helpers
def _erode(x: torch.Tensor, k: int) -> torch.Tensor:
    """min over the k x k window [i, i+k) (out-of-bounds pixels ignored)."""
    return -F.max_pool2d(F.pad(-x, (0, k - 1, 0, k - 1), value=float("-inf")), k, stride=1)


def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    """max over the k x k window (i-k, i] - the reflected element, so open/close are unshifted
    (out-of-bounds pixels ignored)."""
    return F.max_pool2d(F.pad(x, (k - 1, 0, k - 1, 0), value=float("-inf")), k, stride=1)


def min_feature_penalty(ink: torch.Tensor, ss: int = 4, mask: torch.Tensor | None = None,
                        reduction: str = "mean") -> torch.Tensor:
    """Sub-output-pixel detail energy of ``ink`` (n, H, W) at ss resolution.

    open(x) = dilate(erode(x)) removes ink features (strokes, dots) narrower than ss sub-pixels;
    close(x) = erode(dilate(x)) fills gaps/holes narrower than ss. Penalty = mean over the cell
    (``mask``, default all) of (x - open(x)) + (close(x) - x) (both >= 0). Differentiable
    (max-pool subgradients); 0 for any glyph built from >= ss x ss blocks of solid ink/background.
    """
    x = ink.unsqueeze(1) if ink.dim() == 3 else ink
    k = int(ss)
    if k <= 1:                          # ss=1: every feature is >= 1 output pixel by construction
        z = x.sum((1, 2, 3)) * 0.0
        return z if reduction == "none" else z.sum()
    r = 2 * (k - 1)                     # cells sit on background: extend with zeros, crop after
    xp = F.pad(x, (r, r, r, r))
    opened = _dilate(_erode(xp, k), k)[..., r:-r, r:-r]
    closed = _erode(_dilate(xp, k), k)[..., r:-r, r:-r]
    e = (x - opened) + (closed - x)
    if mask is not None:
        m = mask.unsqueeze(1) if mask.dim() == 3 else mask
        e = e * m
        if reduction == "none":
            return e.sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp_min(1)
        return e.sum() / m.sum().clamp_min(1)
    if reduction == "none":
        return e.mean((1, 2, 3))
    return e.mean()


def coverage(ink: torch.Tensor, widths_ss: torch.Tensor) -> torch.Tensor:
    """(n,) mean ink of each glyph over its full cell (height x width, guard band included)."""
    w = torch.as_tensor(widths_ss, dtype=ink.dtype, device=ink.device)
    return ink.sum((1, 2)) / (w * ink.shape[1]).clamp_min(1)


def coverage_penalty(ink: torch.Tensor, widths_ss: torch.Tensor, lo: float = 0.10,
                     hi: float = 0.45) -> torch.Tensor:
    """mean over glyphs of relu(lo - c)^2 + relu(c - hi)^2 (0 when every coverage is in range).
    Pass a subset (e.g. ink[1:], skipping the space glyph) if some glyphs should be blank."""
    c = coverage(ink, widths_ss)
    return (F.relu(lo - c) ** 2 + F.relu(c - hi) ** 2).mean()


def _resize_cell(cell: np.ndarray, w: int) -> np.ndarray:
    if cell.shape[1] == w:
        return cell
    from PIL import Image
    im = Image.fromarray((cell * 255).astype(np.uint8))
    return np.asarray(im.resize((w, cell.shape[0]), Image.Resampling.BILINEAR), np.float32) / 255.0


class GlyphBank(nn.Module):
    """Per-symbol ink cells (n_symbols, height_ss, max_width_ss), zero outside each cell's width.

    ``widths_px`` may be fractional; ``widths_ss = round(widths_px * ss)`` (>= 1). Pass
    ``widths_ss`` directly to avoid rounding.
    init: "random" (logits ~ N(init_bias, init_std), default N(0, 0.5): grey cells with distinct
    noise) or "font" (needs ``symbols``; logits = logit(clamp(font coverage, .05, .95)) / tau-free,
    font cell resized horizontally to each cell width, + small noise).
    """

    def __init__(self, n_symbols: int, height_px: int, widths_px=None, ss: int = 4,
                 init: str = "random", learnable: bool = True, *, widths_ss=None,
                 symbols: Sequence[str] | None = None, font_path=config.LATIN_FONT,
                 cjk_map: dict | None = None, em: float | None = None,
                 init_std: float = 0.5, init_bias: float = 0.0, seed: int = 0,
                 fixed_ink: torch.Tensor | None = None, guard_ss=None, pixel_block: bool = False):
        super().__init__()
        # pixel_block: glyph logits are tied within each ss x ss block (one value per OUTPUT pixel),
        # so learned glyphs are whole-pixel bitmaps by construction (no sub-pixel detail), while the
        # channel still places them at random sub-pixel phase.
        self.pixel_block = bool(pixel_block) and int(ss) > 1
        self.n_symbols, self.height_px, self.ss = int(n_symbols), int(height_px), int(ss)
        self.height_ss = self.height_px * self.ss
        self.em = em_for_height(self.height_px) if em is None else float(em)
        self.symbols = list(symbols) if symbols is not None else None
        self.init = init
        if widths_ss is None:
            if widths_px is None:
                raise ValueError("give widths_px or widths_ss")
            w = torch.as_tensor(widths_px, dtype=torch.float64).reshape(-1)
            if w.numel() == 1:
                w = w.expand(self.n_symbols)
            widths_ss = torch.clamp(torch.round(w * self.ss), min=1)
        widths_ss = torch.as_tensor(widths_ss, dtype=torch.long).reshape(-1)
        if widths_ss.numel() != self.n_symbols:
            raise ValueError(f"{widths_ss.numel()} widths for {self.n_symbols} symbols")
        self.register_buffer("widths_ss", widths_ss.clone())
        self.max_width_ss = int(widths_ss.max())
        if guard_ss is None:                     # default: 1 ss-px guard for learned/random banks
            guard_ss = 0 if fixed_ink is not None else 1
        gx, gy = (guard_ss, guard_ss) if isinstance(guard_ss, int) else tuple(int(g) for g in guard_ss)
        self.guard_ss = (int(gx), int(gy))
        cols = torch.arange(self.max_width_ss)[None, None, :]
        rows = torch.arange(self.height_ss)[None, :, None]
        w = widths_ss[:, None, None]
        gxe = torch.minimum(torch.full_like(w, gx), (w - 1) // 2)     # never mask a whole cell
        gye = min(gy, (self.height_ss - 1) // 2)
        self.register_buffer("cell_mask", (cols < w).float().expand(
            self.n_symbols, self.height_ss, self.max_width_ss).contiguous())
        self.register_buffer("mask", ((cols >= gxe) & (cols < w - gxe) & (rows >= gye)
                                      & (rows < self.height_ss - gye)).float())
        self.clip_frac: list[float] | None = None
        shape = (self.n_symbols, self.height_ss, self.max_width_ss)

        if fixed_ink is not None:              # fixed bank: stored coverage
            self.register_buffer("fixed_ink", fixed_ink.float().reshape(shape) * self.mask)
            self.logits = None
            self.learnable = False
            return
        self.fixed_ink = None
        g = torch.Generator().manual_seed(seed)
        if init == "random":
            logits = init_bias + init_std * torch.randn(shape, generator=g)
        elif init == "font":
            if self.symbols is None:
                raise ValueError("init='font' needs symbols")
            cells, _, _ = render_font_cells(self.symbols, font_path, self.height_px, self.ss,
                                            self.em, cjk_map)
            p = torch.zeros(shape)
            for i, c in enumerate(cells):
                w = int(widths_ss[i])
                p[i, :, :w] = torch.from_numpy(_resize_cell(c, w))
            p = p.clamp(0.05, 0.95)
            logits = torch.log(p / (1 - p)) + 0.1 * init_std * torch.randn(shape, generator=g)
        else:
            raise ValueError(f"unknown init {init!r}")
        self.logits = nn.Parameter(logits, requires_grad=bool(learnable))
        self.learnable = bool(learnable)

    # ------------------------------------------------------------------ forward
    def forward(self, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        if self.fixed_ink is not None:
            return self.fixed_ink
        z = self.logits
        if self.pixel_block:
            k = self.ss
            z = F.avg_pool2d(z.unsqueeze(1), k, ceil_mode=True).squeeze(1)
            z = z.repeat_interleave(k, 1).repeat_interleave(k, 2)[:, :self.height_ss, :self.max_width_ss]
        p = torch.sigmoid(z / tau)
        if hard:
            p = p + ((p > 0.5).to(p.dtype) - p).detach()
        return p * self.mask

    def coverage(self, ink: torch.Tensor | None = None) -> torch.Tensor:
        """(n,) per-glyph ink coverage (of ``ink`` or of ``forward()``)."""
        return coverage(self.forward() if ink is None else ink, self.widths_ss)

    def min_feature_penalty(self, ink: torch.Tensor | None = None,
                            reduction: str = "mean") -> torch.Tensor:
        return min_feature_penalty(self.forward() if ink is None else ink, self.ss,
                                   self.cell_mask, reduction)

    def coverage_penalty(self, ink: torch.Tensor | None = None, lo: float = 0.10,
                         hi: float = 0.45) -> torch.Tensor:
        return coverage_penalty(self.forward() if ink is None else ink, self.widths_ss, lo, hi)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def random_binary(cls, n_symbols: int, height_px: int, widths_px, ss: int = 4,
                      density: float = 0.3, seed: int = 0, guard_ss=1,
                      symbols: Sequence[str] | None = None, blank: Sequence[int] = (0,),
                      max_tries: int = 100) -> "GlyphBank":
        """FROZEN random binary codebook: Bernoulli(density) pixels drawn at OUTPUT-pixel resolution
        inside each cell's guard band, nearest-upsampled to ss (min feature = 1 px), all glyphs
        distinct and non-empty; symbols listed in ``blank`` (default 0 = ASCII space) get no ink."""
        widths_px_t = torch.as_tensor(widths_px, dtype=torch.float64).reshape(-1)
        if widths_px_t.numel() == 1:
            widths_px_t = widths_px_t.expand(n_symbols)
        wss = torch.clamp(torch.round(widths_px_t * ss), min=1).long()
        bank = cls(n_symbols, height_px, ss=ss, widths_ss=wss, symbols=symbols, init="random",
                   learnable=False, init_std=0.0, guard_ss=guard_ss)
        gx, gy = bank.guard_ss
        H = bank.height_ss
        g = torch.Generator().manual_seed(seed)
        ink = torch.zeros(n_symbols, H, bank.max_width_ss)
        seen = set()
        blank = set(int(b) for b in blank)
        for i in range(n_symbols):
            if i in blank:
                continue
            w = int(wss[i])
            ix, iy = max(1, w - 2 * gx), max(1, H - 2 * gy)
            # whole output-pixel blocks only (partial blocks would be sub-pixel features), centred
            bx, by = max(1, ix // ss), max(1, iy // ss)
            px, py = min(ix, bx * ss), min(iy, by * ss)
            ox, oy = gx + (ix - px) // 2, gy + (iy - py) // 2
            cell = None
            for _ in range(max_tries):
                low = (torch.rand(by, bx, generator=g) < density).float()
                cand = low.repeat_interleave(ss, 0).repeat_interleave(ss, 1)[:py, :px]
                key = cand.numpy().tobytes()
                if cand.sum() > 0 and key not in seen:
                    seen.add(key)
                    cell = cand
                    break
            if cell is None:
                raise ValueError(f"could not draw a distinct non-empty glyph for symbol {i} "
                                 f"(cell {ix}x{iy} ss-px, density {density})")
            ink[i, oy:oy + py, ox:ox + px] = cell
        out = cls(n_symbols, height_px, ss=ss, widths_ss=wss, symbols=symbols, init="random_binary",
                  learnable=False, fixed_ink=ink * bank.mask, guard_ss=guard_ss)
        return out

    @classmethod
    def from_font(cls, symbols: Sequence[str], font_path=config.LATIN_FONT, height_px: int = 8,
                  ss: int = 4, cjk_map: dict | None = None, *, em: float | None = None,
                  cjk_font=config.CJK_FONT, learnable: bool = False) -> "GlyphBank":
        """Bank whose cells are PIL renderings (cell = advance, proportional like Phase 2).

        learnable=False (default): FIXED bank, no parameters (``fixed_ink`` buffer).
        learnable=True: logits initialised from the rendering (logit of clamped coverage).
        """
        cells, widths, clips = render_font_cells(symbols, font_path, height_px, ss, em, cjk_map,
                                                 cjk_font)
        n, H, Wm = len(symbols), height_px * ss, max(widths)
        ink = torch.zeros(n, H, Wm)
        for i, c in enumerate(cells):
            ink[i, :, :c.shape[1]] = torch.from_numpy(c)
        if learnable:
            bank = cls(n, height_px, ss=ss, widths_ss=widths, symbols=symbols, em=em,
                       init="random", init_std=0.0)
            p = ink.clamp(0.02, 0.98)
            with torch.no_grad():
                bank.logits.copy_(torch.log(p / (1 - p)))
            bank.init = "font"
        else:
            bank = cls(n, height_px, ss=ss, widths_ss=widths, symbols=symbols, em=em,
                       init="font", learnable=False, fixed_ink=ink)
        bank.clip_frac = clips
        bank.font_path = str(font_path)
        return bank

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            d = {"symbols": self.symbols, "widths_ss": self.widths_ss.cpu(), "ss": self.ss,
                 "height_px": self.height_px, "em": self.em, "init": self.init,
                 "learnable": self.learnable, "ink": self.forward().detach().cpu(),
                 "guard_ss": list(self.guard_ss), "pixel_block": self.pixel_block,
                 "logits": None if self.logits is None else self.logits.detach().cpu()}
        torch.save(d, path)

    @classmethod
    def load(cls, path, learnable: bool | None = None) -> "GlyphBank":
        """Restore a bank. Learned banks come back with their logits (``learnable`` defaults to
        the saved flag; pass False to freeze); fixed banks come back fixed."""
        d = torch.load(path, map_location="cpu", weights_only=False)
        n = int(d["widths_ss"].numel())
        if d.get("logits") is None:
            return cls(n, d["height_px"], ss=d["ss"], widths_ss=d["widths_ss"], symbols=d["symbols"],
                       em=d["em"], init=d.get("init", "font"), learnable=False, fixed_ink=d["ink"],
                       guard_ss=tuple(d.get("guard_ss", (0, 0))))
        lr = d.get("learnable", True) if learnable is None else learnable
        bank = cls(n, d["height_px"], ss=d["ss"], widths_ss=d["widths_ss"], symbols=d["symbols"],
                   em=d["em"], init="random", learnable=lr, init_std=0.0,
                   guard_ss=tuple(d.get("guard_ss", (1, 1))), pixel_block=d.get("pixel_block", False))
        with torch.no_grad():
            bank.logits.copy_(d["logits"])
        bank.init = d.get("init", "random")
        return bank

    # ------------------------------------------------------------------ viewing
    def atlas_png(self, path, symbols: Sequence[str] | None = None, tau: float = 1.0,
                  hard: bool = False, cols: int = 16, zoom: int | None = None,
                  show_1x: bool = True, max_symbols: int | None = None) -> None:
        """Grid of glyph cells: ss-res cell (dark ink on white, nearest-upscaled), optionally the
        1x area-downsampled view underneath, and a label (spaces shown as '␣')."""
        from PIL import Image, ImageDraw, ImageFont

        symbols = list(symbols if symbols is not None else (self.symbols or
                                                             [str(i) for i in range(self.n_symbols)]))
        n = min(self.n_symbols, len(symbols), max_symbols or self.n_symbols)
        with torch.no_grad():
            ink = self.forward(tau, hard)[:n].cpu().numpy()
        ss = self.ss
        zoom = zoom or max(1, int(round(48 / self.height_ss)))
        cw = self.max_width_ss * zoom
        ch = self.height_ss * zoom
        lab_h = 14
        cell_w = max(cw, 40) + 6
        cell_h = ch * (2 if show_1x else 1) + lab_h + 8
        rows = math.ceil(n / cols)
        img = Image.new("L", (cols * cell_w, rows * cell_h), 255)
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(str(config.LATIN_FONT), 11)
        except OSError:
            font = ImageFont.load_default()
        for i in range(n):
            r, c = divmod(i, cols)
            x0, y0 = c * cell_w + 3, r * cell_h + 2
            w = int(self.widths_ss[i])
            a = ink[i, :, :w]
            im = Image.fromarray((255 - a * 255).round().astype(np.uint8))
            img.paste(im.resize((w * zoom, ch), Image.Resampling.NEAREST), (x0, y0))
            d.rectangle([x0 - 1, y0 - 1, x0 + w * zoom, y0 + ch], outline=200)
            yy = y0 + ch
            if show_1x:
                wp = int(math.ceil(w / ss))
                pad = np.zeros((self.height_ss, wp * ss), np.float32)
                pad[:, :w] = a
                low = pad.reshape(self.height_px, ss, wp, ss).mean(axis=(1, 3))
                im1 = Image.fromarray((255 - low * 255).round().astype(np.uint8))
                img.paste(im1.resize((wp * ss * zoom, ch), Image.Resampling.NEAREST), (x0, yy))
                yy += ch
            lab = symbols[i].replace(" ", "␣")
            lab = lab if len(lab) <= 8 else lab[:7] + "…"
            d.text((x0, yy + 1), lab, fill=0, font=font)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
