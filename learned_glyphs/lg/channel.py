"""Rendering channel: supersampled line canvas -> degraded low-res line image (WP-LG1).

Pipeline per line (train mode draws fresh parameters per line; eval uses fixed mid-range values and
a seeded generator for the remaining randomness, so eval is deterministic):

1. Phase-2 margin: ``margin`` px (default 2, = RenderParams.margin) of background on every side.
2. Sub-pixel phase: shift right by sx and down by sy ss-pixels, sx, sy ~ U{0..ss-1}
   (resolution 1/ss px, like Phase 2's supersampled phase; Phase 2 has x phase only, the y phase
   is an extra guard against row-aligned "barcode" glyphs - ``phase_y=False`` disables it).
3. Area downsample by ss (avg_pool = PIL BOX resize used by vbpe.layout).
4. Resampling round trip (``scale``, harsh only): resize by a non-integer factor s (log-uniform in
   [scale]) and back to the original size, each direction with a filter drawn from ``filters``
   (torch ``area`` = box, ``bilinear``/``bicubic`` with antialias). This injects real resampling
   geometry (sub-pixel misregistration, aliasing, filter ringing) while keeping the image size,
   so widths, CTC lengths and px^2/char accounting are unchanged.
5. Gaussian blur, sigma ~ U[blur] px (separable per-sample kernels, grouped conv; 0 = identity).
6. Ink gamma x**g, g log-uniform in [gamma]; contrast: x = b + x*(c - b), c ~ U[contrast],
   background lift b ~ U[bg] (same form as vbpe.crnn.data.augment_batch).
7. Additive Gaussian noise, sigma ~ U[noise]; clamp to [0,1].
8. JPEG (``jpeg`` quality range, harsh only): differentiable approximation - 8x8 block DCT with the
   IJG luminance table scaled by quality, quantised with the Shin & Song (2017) cubic rounding
   round(z) + (z - round(z))**3 (true gradient everywhere, close to real JPEG in value).
9. 8-bit quantisation round(255 x)/255 with a straight-through gradient.

Held-out (test-only) degradations: ``Channel(preset, test_only=True)`` or the preset suffix
``"+heldout"`` (e.g. ``"harsh+heldout"``; ``"heldout"`` alone = harsh base) appends, in eval mode
only, operations never used by any training preset: a PIL LANCZOS resampling round trip at scale
0.87 and REAL PIL JPEG at quality 50 (both non-differentiable, wrapped straight-through). Calling a
test-only channel with ``train=True`` raises.

Output convention = vbpe.crnn.data.collate: (B,1,H,W) float32, ink = 1, background = 0,
H = height_px + 2*margin, widths_px = ceil(widths_ss / ss) + 2*margin. Everything except the
held-out ops is differentiable w.r.t. the canvas (quantisation via straight-through).
"""
from __future__ import annotations

import io
import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

TRAIN_FILTERS = ("area", "bilinear", "bicubic")
HELDOUT_SCALE, HELDOUT_JPEG_Q = 0.87, 50


@dataclass(frozen=True)
class ChannelParams:
    phase_x: bool = True
    phase_y: bool = True
    scale: tuple[float, float] = (1.0, 1.0)       # resampling round-trip factor (log-uniform)
    eval_scale: float | None = None               # None -> geometric mean of ``scale``
    filters: tuple[str, ...] = TRAIN_FILTERS      # resize filters drawn per line and direction
    eval_filter: str = "bilinear"
    blur: tuple[float, float] = (0.0, 0.0)        # gaussian sigma, px at 1x
    contrast: tuple[float, float] = (1.0, 1.0)    # ink level c (1 = full black)
    bg: tuple[float, float] = (0.0, 0.0)          # background lift b
    gamma: tuple[float, float] = (1.0, 1.0)       # ink gamma (log-uniform)
    noise: tuple[float, float] = (0.0, 0.0)       # additive gaussian sigma
    jpeg: tuple[float, float] = (0.0, 0.0)        # JPEG quality range (0 = off), differentiable approx
    quantize: bool = True

    def eval_value(self, name: str) -> float:
        lo, hi = getattr(self, name)
        if name == "scale" and self.eval_scale is not None:
            return float(self.eval_scale)
        return math.sqrt(lo * hi) if name in ("gamma", "scale") else (lo + hi) / 2


PRESETS: dict[str, ChannelParams] = {
    # phase + area-downsample (+ 8-bit, as Phase-2 uint8 images): Phase-2 eval rendering
    "clean": ChannelParams(),
    # Phase-2 train augmentation (contrast/bg/noise ranges of augment_batch) + mild optics
    "default": ChannelParams(blur=(0.0, 0.5), contrast=(0.7, 1.0), bg=(0.0, 0.05),
                             gamma=(0.8, 1.25), noise=(0.0, 0.06)),
    # geometry-heavy (PLAN amendments): resampling round trip at 0.8-1.25 with random filters,
    # stronger blur, photometric jitter, noise <= 0.1, JPEG q 40-90
    "harsh": ChannelParams(scale=(0.8, 1.25), eval_scale=0.9, blur=(0.3, 1.2), contrast=(0.5, 0.95),
                           bg=(0.0, 0.1), gamma=(0.7, 1.4), noise=(0.01, 0.1), jpeg=(40.0, 90.0)),
}

# IJG standard luminance quantisation table
_JPEG_Q = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)


def _dct_matrix(n: int = 8) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(n, dtype=torch.float32)[None, :]
    m = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    m[0] /= math.sqrt(2.0)
    return m


_DCT = _dct_matrix()


def jpeg_qtable(quality: torch.Tensor) -> torch.Tensor:
    """(B,) qualities 1..100 -> (B,8,8) IJG-scaled luminance tables."""
    q = quality.float().clamp(1, 100)
    s = torch.where(q < 50, 5000.0 / q, 200.0 - 2 * q)
    return torch.floor((_JPEG_Q[None] * s[:, None, None] + 50) / 100).clamp_min(1.0)


def jpeg_approx(x: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """Differentiable grey JPEG approximation of (B,1,H,W) in [0,1] at per-sample ``quality``."""
    B, _, H, W = x.shape
    ph, pw = (-H) % 8, (-W) % 8
    v = F.pad(x * 255.0 - 128.0, (0, pw, 0, ph), mode="replicate")
    Hp, Wp = H + ph, W + pw
    blk = v.reshape(B, Hp // 8, 8, Wp // 8, 8).permute(0, 1, 3, 2, 4)          # B,h,w,8,8
    D = _DCT.to(x)
    c = D @ blk @ D.T
    Q = jpeg_qtable(quality).to(x)[:, None, None]
    z = c / Q
    zr = torch.round(z)
    z = zr + (z - zr) ** 3
    rec = D.T @ (z * Q) @ D
    out = rec.permute(0, 1, 3, 2, 4).reshape(B, 1, Hp, Wp)[:, :, :H, :W]
    return (out + 128.0) / 255.0


def pil_jpeg(x: torch.Tensor, quality: int = HELDOUT_JPEG_Q) -> torch.Tensor:
    """REAL PIL JPEG (grey, white background like a rendered page), straight-through."""
    from PIL import Image
    a = x.detach().clamp(0, 1).cpu().numpy()
    out = np.empty_like(a)
    for b in range(a.shape[0]):
        im = Image.fromarray((255 - np.round(a[b, 0] * 255)).astype(np.uint8), mode="L")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out[b, 0] = (255.0 - np.asarray(Image.open(buf), dtype=np.float32)) / 255.0
    y = torch.from_numpy(out).to(x)
    return x + (y - x).detach()


def pil_resample(x: torch.Tensor, scale: float = HELDOUT_SCALE) -> torch.Tensor:
    """PIL LANCZOS round trip (down by ``scale`` and back) - a filter no training preset uses."""
    from PIL import Image
    a = x.detach().cpu().numpy().astype(np.float32)
    out = np.empty_like(a)
    H, W = a.shape[2:]
    h2, w2 = max(1, round(H * scale)), max(1, round(W * scale))
    for b in range(a.shape[0]):
        im = Image.fromarray(a[b, 0], mode="F")
        im = im.resize((w2, h2), Image.Resampling.LANCZOS).resize((W, H), Image.Resampling.LANCZOS)
        out[b, 0] = np.asarray(im, dtype=np.float32)
    y = torch.from_numpy(out).to(x).clamp(0, 1)
    return x + (y - x).detach()


def _resize(x: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if mode == "area":
        return F.interpolate(x, size=size, mode="area")
    return F.interpolate(x, size=size, mode=mode, align_corners=False, antialias=True)


def resample_roundtrip(x: torch.Tensor, scale: torch.Tensor, modes_down, modes_up) -> torch.Tensor:
    """Per-sample resize by ``scale`` and back to the original size (differentiable)."""
    B, _, H, W = x.shape
    outs = []
    for b in range(B):
        s = float(scale[b])
        if abs(s - 1.0) < 1e-6:
            outs.append(x[b:b + 1])
            continue
        size = (max(1, round(H * s)), max(1, round(W * s)))
        y = _resize(x[b:b + 1], size, modes_down[b])
        outs.append(_resize(y, (H, W), modes_up[b]))
    return torch.cat(outs, 0).clamp(0.0, 1.0)


def _gauss_kernels(sigma: torch.Tensor, radius: int) -> torch.Tensor:
    """(B,) sigmas -> (B, 2r+1) normalised 1-D kernels (sigma ~ 0 -> delta)."""
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    s = sigma.clamp_min(1e-3)[:, None]
    k = torch.exp(-0.5 * (x[None, :] / s) ** 2)
    return k / k.sum(1, keepdim=True)


def gaussian_blur(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Per-sample separable gaussian blur of (B,1,H,W) with zero (= background) padding."""
    B, _, H, W = x.shape
    smax = float(sigma.max())
    if smax < 1e-3:
        return x
    r = max(1, int(math.ceil(3 * smax)))
    k = _gauss_kernels(sigma.float(), r).to(x)
    y = x.reshape(1, B, H, W)
    y = F.conv2d(y, k[:, None, None, :], padding=(0, r), groups=B)
    y = F.conv2d(y, k[:, None, :, None], padding=(r, 0), groups=B)
    return y.reshape(B, 1, H, W)


class Channel:
    def __init__(self, preset: str = "default", ss: int = 4, margin: int = 2, eval_seed: int = 0,
                 test_only: bool = False, **overrides):
        name = preset
        if preset == "heldout":
            preset, test_only = "harsh", True
        elif preset.endswith("+heldout"):
            preset, test_only = preset[: -len("+heldout")], True
        if preset not in PRESETS:
            raise KeyError(f"unknown channel preset {name!r} (have {sorted(PRESETS)}, '+heldout')")
        self.name = name
        self.preset, self.ss, self.margin, self.eval_seed = preset, int(ss), int(margin), int(eval_seed)
        self.test_only = bool(test_only)
        self.params = replace(PRESETS[preset], **overrides)

    def describe(self) -> dict:
        d = {"preset": self.preset, "ss": self.ss, "margin": self.margin, "test_only": self.test_only,
             **asdict(self.params)}
        if self.test_only:
            d["heldout"] = {"pil_lanczos_roundtrip_scale": HELDOUT_SCALE, "pil_jpeg_quality": HELDOUT_JPEG_Q}
        return d

    def _draw(self, name: str, B: int, train: bool, gen) -> torch.Tensor:
        lo, hi = getattr(self.params, name)
        if not train:
            return torch.full((B, 1, 1, 1), self.params.eval_value(name))
        if lo == hi:
            return torch.full((B, 1, 1, 1), float(lo))
        u = torch.rand(B, 1, 1, 1, generator=gen)
        if name in ("gamma", "scale"):
            return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        return lo + u * (hi - lo)

    def __call__(self, canvas_ss: torch.Tensor, widths_ss, generator: torch.Generator | None = None,
                 train: bool = True, phase=None) -> tuple[torch.Tensor, torch.Tensor]:
        """canvas_ss (B,1,Hs,Ws) ink in [0,1] (Hs divisible by ss) -> (images, widths_px).

        ``phase``: None (random: train fresh, eval seeded), an int (same ss shift in x and y for
        every line, e.g. 0 for pixel-aligned comparisons) or a (B,2) long tensor of (sx, sy).
        """
        if train and self.test_only:
            raise ValueError(f"channel {self.name!r} is test-only (held-out degradations); "
                             "never call it with train=True")
        ss, m, p = self.ss, self.margin, self.params
        gen = generator
        if gen is None and not train:
            gen = torch.Generator().manual_seed(self.eval_seed)
        B, _, Hs, Ws = canvas_ss.shape
        if Hs % ss:
            raise ValueError(f"canvas height {Hs} not divisible by ss={ss}")
        widths_ss = torch.as_tensor(widths_ss, dtype=torch.long)
        Hout = Hs // ss + 2 * m
        Wout = -(-Ws // ss) + 2 * m
        # pad: (ss-1) phase room + margin on left/top, rest on right/bottom
        L = T = m * ss + ss - 1
        R = Wout * ss + ss - 1 - L - Ws
        Bt = Hout * ss + ss - 1 - T - Hs
        big = F.pad(canvas_ss, (L, R, T, Bt))
        if phase is None:
            sx = torch.randint(0, ss, (B,), generator=gen) if p.phase_x else torch.zeros(B, dtype=torch.long)
            sy = torch.randint(0, ss, (B,), generator=gen) if p.phase_y else torch.zeros(B, dtype=torch.long)
        elif isinstance(phase, int):
            sx = sy = torch.full((B,), phase, dtype=torch.long)
        else:
            phase = torch.as_tensor(phase, dtype=torch.long)
            sx, sy = phase[:, 0], phase[:, 1]
        crops = [big[b:b + 1, :, ss - 1 - int(sy[b]): ss - 1 - int(sy[b]) + Hout * ss,
                     ss - 1 - int(sx[b]): ss - 1 - int(sx[b]) + Wout * ss] for b in range(B)]
        x = F.avg_pool2d(torch.cat(crops, 0), ss)                          # (B,1,Hout,Wout)

        if p.scale != (1.0, 1.0):
            s = self._draw("scale", B, train, gen).reshape(B)
            if train:
                nf = len(p.filters)
                fi = torch.randint(0, nf, (2, B), generator=gen)
                down = [p.filters[int(i)] for i in fi[0]]
                up = [p.filters[int(i)] for i in fi[1]]
            else:
                down = up = [p.eval_filter] * B
            x = resample_roundtrip(x, s, down, up)
        if p.blur != (0.0, 0.0):
            x = gaussian_blur(x, self._draw("blur", B, train, gen).reshape(B))
        if p.gamma != (1.0, 1.0):
            x = x.clamp_min(1e-6) ** self._draw("gamma", B, train, gen)
        if p.contrast != (1.0, 1.0) or p.bg != (0.0, 0.0):
            c, b = self._draw("contrast", B, train, gen), self._draw("bg", B, train, gen)
            x = b + x * (c - b)
        if p.noise != (0.0, 0.0):
            s = self._draw("noise", B, train, gen)
            x = x + s * torch.randn(x.shape, generator=gen)
        x = x.clamp(0.0, 1.0)
        if p.jpeg != (0.0, 0.0):
            x = jpeg_approx(x, self._draw("jpeg", B, train, gen).reshape(B)).clamp(0.0, 1.0)
        if self.test_only and not train:
            x = pil_jpeg(pil_resample(x), HELDOUT_JPEG_Q)
        if p.quantize:
            x = x + (torch.round(x * 255.0) / 255.0 - x).detach()
        widths_px = -(-widths_ss // ss) + 2 * m
        return x.float(), widths_px
