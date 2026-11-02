"""GlyphBank: one ink cell per symbol at supersampled resolution (WP-LG1).

Geometry (all cells share it, so composed lines follow vbpe.layout conventions)
-------------------------------------------------------------------------------
* ``height_px`` = line pitch h (px at 1x); a cell is ``height_ss = h * ss`` rows tall and covers the
  line box only (the Phase-2 margin is added later by the Channel).
* Equivalent em = ``h / 1.2`` (``line_spacing`` = 1.2) unless ``em`` is given explicitly (the
  calibration passes the exact Phase-2 em, e.g. em 7 -> h = round(8.4) = 8).
* Baseline (font banks) at ``((line_spacing - 1)/2 + 0.8) * em`` from the top of the line box, i.e.
  ``0.9 * em * ss`` ss-rows: identical to ``LayoutParams.first_baseline`` minus the margin.
* Cell width: per symbol, ``widths_ss`` (ss pixels). Font banks use the glyph's advance at the
  supersampled em (``vbpe.layout.glyph_advance``, rounded), i.e. proportional like Phase 2; ink
  outside [0, advance) (negative side bearings, e.g. 'j') is clipped (``clip_frac`` records it).

Representation
--------------
* learnable bank: ``logits`` (n, height_ss, max_width_ss); ``forward(tau, hard)`` = sigmoid(logits/tau)
  masked to each cell's width; ``hard=True`` -> straight-through binarisation (value = (p > .5),
  gradient = sigmoid's).
* fixed bank (``from_font(..., learnable=False)``, the default): the rendered coverage is stored
  as a buffer ``fixed_ink`` and returned unchanged (``tau``/``hard`` are ignored), so the
  calibration bank is exactly PIL's antialiased coverage at ss resolution.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from vbpe import config
from vbpe.layout import get_font, glyph_advance

LINE_SPACING = 1.2


def em_for_height(height_px: int) -> float:
    return height_px / LINE_SPACING


def baseline_ss(em: float, ss: int, line_spacing: float = LINE_SPACING) -> float:
    """Baseline row (ss pixels, fractional) inside the line box, margin excluded (vbpe.layout)."""
    return ((line_spacing - 1) / 2 * em + 0.8 * em) * ss


def _cjk_dy(cjk_font: str) -> float:
    """Vertical shift (em, + = down) used by vbpe SubstitutionScheme/CJKUnit (computed over the
    whole symbol pool, cjk_scale = 1)."""
    from vbpe.schemes.substitution import ink_extent, load_symbols, vertical_shift
    top, bot = ink_extent(str(cjk_font), tuple(load_symbols()))
    return vertical_shift(top, bot, 1.0)


def render_font_cells(symbols: Sequence[str], font_path, height_px: int, ss: int = 4,
                      em: float | None = None, cjk_map: dict | None = None,
                      cjk_font=config.CJK_FONT) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Render each symbol with PIL at ss resolution, as vbpe renders it at ``em``.

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
                 fixed_ink: torch.Tensor | None = None):
        super().__init__()
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
        cols = torch.arange(self.max_width_ss)
        self.register_buffer("mask", (cols[None, None, :] < widths_ss[:, None, None]).float())
        self.clip_frac: list[float] | None = None
        shape = (self.n_symbols, self.height_ss, self.max_width_ss)

        if fixed_ink is not None:              # fixed bank: stored coverage
            self.register_buffer("fixed_ink", fixed_ink.float().reshape(shape) * self.mask[:, :1].expand(shape))
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
        p = torch.sigmoid(self.logits / tau)
        if hard:
            p = p + ((p > 0.5).to(p.dtype) - p).detach()
        return p * self.mask

    # ------------------------------------------------------------------ constructors
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
                       em=d["em"], init=d.get("init", "font"), learnable=False, fixed_ink=d["ink"])
        lr = d.get("learnable", True) if learnable is None else learnable
        bank = cls(n, d["height_px"], ss=d["ss"], widths_ss=d["widths_ss"], symbols=d["symbols"],
                   em=d["em"], init="random", learnable=lr, init_std=0.0)
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
