"""Rendering channel: supersampled line canvas -> degraded low-res line image (WP-LG1).

Pipeline per line (train mode draws fresh parameters per line; eval uses fixed mid-range values and
a seeded generator for the remaining randomness, so eval is deterministic):

1. Phase-2 margin: ``margin`` px (default 2, = RenderParams.margin) of background on every side.
2. Sub-pixel phase: shift right by sx and down by sy ss-pixels, sx, sy ~ U{0..ss-1}
   (resolution 1/ss px, like Phase 2's supersampled phase; Phase 2 has x phase only, the y phase
   is an extra guard against row-aligned "barcode" glyphs - ``phase_y=False`` disables it).
3. Area downsample by ss (avg_pool = PIL BOX resize used by vbpe.layout).
4. Gaussian blur, sigma ~ U[blur] px (separable per-sample kernels, grouped conv; 0 = identity).
5. Ink gamma x**g, g log-uniform in [gamma]; contrast: x = b + x*(c - b), c ~ U[contrast],
   background lift b ~ U[bg] (same form as vbpe.crnn.data.augment_batch).
6. Additive Gaussian noise, sigma ~ U[noise]; clamp to [0,1].
7. 8-bit quantisation round(255 x)/255 with a straight-through gradient.

Output convention = vbpe.crnn.data.collate: (B,1,H,W) float32, ink = 1, background = 0,
H = height_px + 2*margin, widths_px = ceil(widths_ss / ss) + 2*margin. Everything is
differentiable w.r.t. the canvas (quantisation via straight-through).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ChannelParams:
    phase_x: bool = True
    phase_y: bool = True
    blur: tuple[float, float] = (0.0, 0.0)        # gaussian sigma, px at 1x
    contrast: tuple[float, float] = (1.0, 1.0)    # ink level c (1 = full black)
    bg: tuple[float, float] = (0.0, 0.0)          # background lift b
    gamma: tuple[float, float] = (1.0, 1.0)       # ink gamma (log-uniform)
    noise: tuple[float, float] = (0.0, 0.0)       # additive gaussian sigma
    quantize: bool = True

    def eval_value(self, name: str) -> float:
        lo, hi = getattr(self, name)
        return math.sqrt(lo * hi) if name == "gamma" else (lo + hi) / 2


PRESETS: dict[str, ChannelParams] = {
    # phase + area-downsample (+ 8-bit, as Phase-2 uint8 images): Phase-2 eval rendering
    "clean": ChannelParams(),
    # Phase-2 train augmentation (contrast/bg/noise ranges of augment_batch) + mild optics
    "default": ChannelParams(blur=(0.0, 0.5), contrast=(0.7, 1.0), bg=(0.0, 0.05),
                             gamma=(0.8, 1.25), noise=(0.0, 0.06)),
    # more blur, less contrast, more noise
    "harsh": ChannelParams(blur=(0.3, 0.9), contrast=(0.5, 0.9), bg=(0.0, 0.1),
                           gamma=(0.7, 1.4), noise=(0.02, 0.12)),
}


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
                 **overrides):
        if preset not in PRESETS:
            raise KeyError(f"unknown channel preset {preset!r} (have {sorted(PRESETS)})")
        self.preset, self.ss, self.margin, self.eval_seed = preset, int(ss), int(margin), int(eval_seed)
        self.params = replace(PRESETS[preset], **overrides)

    def describe(self) -> dict:
        return {"preset": self.preset, "ss": self.ss, "margin": self.margin, **asdict(self.params)}

    def _draw(self, name: str, B: int, train: bool, gen) -> torch.Tensor:
        lo, hi = getattr(self.params, name)
        if not train or lo == hi:
            return torch.full((B, 1, 1, 1), self.params.eval_value(name) if not train else lo)
        u = torch.rand(B, 1, 1, 1, generator=gen)
        if name == "gamma":
            return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        return lo + u * (hi - lo)

    def __call__(self, canvas_ss: torch.Tensor, widths_ss, generator: torch.Generator | None = None,
                 train: bool = True, phase=None) -> tuple[torch.Tensor, torch.Tensor]:
        """canvas_ss (B,1,Hs,Ws) ink in [0,1] (Hs divisible by ss) -> (images, widths_px).

        ``phase``: None (random: train fresh, eval seeded), an int (same ss shift in x and y for
        every line, e.g. 0 for pixel-aligned comparisons) or a (B,2) long tensor of (sx, sy).
        """
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
        if p.quantize:
            x = x + (torch.round(x * 255.0) / 255.0 - x).detach()
        widths_px = -(-widths_ss // ss) + 2 * m
        return x.float(), widths_px
