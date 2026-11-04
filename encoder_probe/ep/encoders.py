"""Frozen image encoders -> (B, 196, 768) token grids for 224x224 RGB pages, plus degradations.

  identity  raw 16x16x3 pixel patches (768 values) = the ceiling: no information lost
  siglip    google/siglip-base-patch16-224, last_hidden_state (the SigLIP family feeds PaliGemma-style VLMs)
  siglip_l<k>  same model, output of transformer layer k (1..12) without the final layernorm (control)
  clip      openai/clip-vit-base-patch16, penultimate layer, CLS dropped (LLaVA convention)
"""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

GRID = 14


def resolve_device(name: str = "auto") -> torch.device:
    """--device value -> torch.device: auto = cuda if available, else cpu (the CPU path is the default)."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda, but torch.cuda.is_available() is False (CPU-only torch wheel?)")
    return torch.device(name)


def degrade(img: np.ndarray, cond: str) -> np.ndarray:
    if cond == "clean":
        return img
    if cond.startswith("jpeg"):
        q = int(cond[4:])
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=q)   # PIL default 4:2:0 chroma subsampling
        return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    if cond.startswith("resize"):  # down to s% with bicubic and back (an API resizing the image)
        s = int(cond[6:]) / 100
        im = Image.fromarray(img)
        small = im.resize((round(224 * s), round(224 * s)), Image.Resampling.BICUBIC)
        return np.asarray(small.resize((224, 224), Image.Resampling.BICUBIC))
    raise KeyError(cond)


def patchify(x: np.ndarray) -> np.ndarray:
    """(B,224,224,3) float -> (B,196,768) in (dy, dx, ch) order."""
    B = x.shape[0]
    return x.reshape(B, GRID, 16, GRID, 16, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, GRID * GRID, 768)


class Encoder:
    def __init__(self, name: str, device: str | torch.device = "cpu"):
        self.name = name
        self.device = torch.device(device)
        self.model = None
        self.layer = int(name[len("siglip_l"):]) if name.startswith("siglip_l") else None
        if name == "siglip" or self.layer is not None:
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").eval()
            self.mean, self.std = np.array([0.5] * 3), np.array([0.5] * 3)
        elif name == "clip":
            from transformers import CLIPVisionModel
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").eval()
            self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
            self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        elif name != "identity":
            raise KeyError(name)
        if self.model is not None and self.device.type != "cpu":
            self.model = self.model.to(self.device)   # stays fp32, as the round-1 rows

    @torch.no_grad()
    def __call__(self, imgs: np.ndarray, batch: int | None = None) -> np.ndarray:
        batch = batch or (64 if self.device.type == "cuda" else 8)
        x = imgs.astype(np.float32) / 255.0
        if self.model is None:
            return patchify(x).astype(np.float32)
        out = []
        for i in range(0, len(x), batch):
            xb = (x[i:i + batch] - self.mean) / self.std
            t = torch.from_numpy(xb.astype(np.float32)).permute(0, 3, 1, 2).contiguous().to(self.device)
            if self.layer is not None:
                h = self.model(pixel_values=t, output_hidden_states=True).hidden_states[self.layer]
            elif self.name == "siglip":
                h = self.model(pixel_values=t).last_hidden_state
            else:
                h = self.model(pixel_values=t, output_hidden_states=True).hidden_states[-2][:, 1:]
            out.append(h.float().cpu().numpy())
        return np.concatenate(out)
