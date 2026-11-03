"""Round-2 frozen encoders on S x S pages -> (B, G*G, d) token grids (row-major over the G x G token grid).

  name         S    P   G   d     tokens                     what
  identity28   448  28  16  2352  256  raw 28x28x3 pixel patches (ceiling for the Qwen geometry)
  qwen25vl     448  28  16  2048  256  Qwen/Qwen2.5-VL-3B-Instruct vision tower, merger output (the LLM's
                                      image tokens: 14-px patches, 32 ViT blocks, 2x2 patch merger MLP)
  identity32   512  32  16  3072  256  raw 32x32x3 patches (ceiling for the DeepEncoder page; see ROUND2.md for
                                      why not raw 64x64 patches)
  deepencoder  512  64   8  1280  64   deepseek-ai/DeepSeek-OCR DeepEncoder, Tiny mode, projector output (the
                                      LLM's image tokens: SAM-B + 16x conv compressor + CLIP-L + linear projector)

Weights: only the vision tensors are fetched from each checkpoint (HTTP range requests, ep/hf_partial.py) into
encoder_probe/cache/weights/. Both models run in bf16 (their checkpoint dtype).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "encoder_probe" / "cache" / "weights"

QWEN_REPO, QWEN_REV = "Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3"
QWEN_SHARD = "model-00001-of-00002.safetensors"          # holds every visual.* tensor (one contiguous 1.34 GB run)
DS_REPO, DS_REV = "deepseek-ai/DeepSeek-OCR", "9f30c71f441d010e5429c532364a86705536c53a"
DS_SHARD = "model-00001-of-000001.safetensors"

GEOMS = {  # encoder -> (page side S, token cell P)
    "identity28": (448, 28),
    "qwen25vl": (448, 28),
    "identity32": (512, 32),
    "deepencoder": (512, 64),
}
DIMS = {"identity28": 28 * 28 * 3, "qwen25vl": 2048, "identity32": 32 * 32 * 3, "deepencoder": 1280}
# round-2 encodings (requested set) + Latin anchors at lower density, so that "best Latin at 3% CER" is defined
# per geometry (at 64-px tokens latin_p16 is already 36 chars/token)
ENCODINGS = ["latin_p5", "latin_p6", "latin_p8", "latin_p10", "latin_p16", "col_b1", "col_b2", "col_b4",
             "lg_free_h4", "lg_pb_h5", "rgb3_p6", "rgb3_p8", "mono_p6", "mono_p8"]
ANCHORS = {448: ["latin_p24"], 512: ["latin_p24", "latin_p32"]}


def encodings_for(encoder: str) -> list[str]:
    return ENCODINGS + ANCHORS[GEOMS[encoder][0]]


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def geometry(encoder: str) -> tuple[int, int, int]:
    S, P = GEOMS[encoder]
    return S, P, S // P


def patchify_s(x: np.ndarray, P: int) -> np.ndarray:
    """(B,S,S,3) -> (B, G*G, P*P*3), tokens row-major, values in (dy, dx, ch) order (round-1 convention)."""
    B, S = x.shape[0], x.shape[1]
    G = S // P
    return x.reshape(B, G, P, G, P, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, G * G, P * P * 3)


def qwen_pixel_values(imgs: np.ndarray, patch: int = 14, merge: int = 2, tp: int = 2) -> np.ndarray:
    """uint8 (B,S,S,3) -> Qwen2-VL processor layout (B*gh*gw, 3*tp*14*14), no resizing (S % 28 == 0).

    Same as Qwen2VLImageProcessor(Pil): rescale, CLIP mean/std, the still image repeated to tp frames, patches
    ordered so each 2x2 merge group is contiguous and the merged tokens are row-major."""
    x = (imgs.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD                 # (B,H,W,3)
    B, H, W, _ = x.shape
    assert H % (patch * merge) == 0 and W % (patch * merge) == 0
    gh, gw = H // patch, W // patch
    x = x.transpose(0, 3, 1, 2)[:, None].repeat(tp, 1)                          # (B,tp,C,H,W)
    x = x.reshape(B, tp, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)                                  # B,gh/m,gw/m,m,m,C,tp,p,p
    return np.ascontiguousarray(x.reshape(B * gh * gw, 3 * tp * patch * patch))


class QwenVision:
    """Qwen2.5-VL-3B vision tower; returns the patch-merger output (B, (S/28)^2, 2048)."""

    def __init__(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLVisionRotaryEmbedding)
        from encoder_probe.ep.hf_partial import fetch_tensors

        cfg = json.loads(Path(hf_hub_download(QWEN_REPO, "config.json", revision=QWEN_REV)).read_text())
        vc = Qwen2_5_VLVisionConfig(**cfg["vision_config"])
        vc._attn_implementation = "sdpa"
        path = fetch_tensors(QWEN_REPO, QWEN_SHARD, lambda k: k.startswith("visual."),
                             WEIGHTS / "qwen25vl3b_visual.safetensors", revision=QWEN_REV)
        with torch.device("meta"):
            m = Qwen2_5_VisionTransformerPretrainedModel(vc)
        sd = {k[len("visual."):]: v for k, v in load_file(str(path)).items()}
        m.load_state_dict(sd, strict=True, assign=True)
        m.rotary_pos_emb = Qwen2_5_VLVisionRotaryEmbedding(vc)                  # non-persistent buffers
        self.model = m.to(torch.bfloat16).eval()
        self.out_dim = vc.out_hidden_size

    @torch.no_grad()
    def __call__(self, imgs: np.ndarray, batch: int = 2) -> torch.Tensor:
        S = imgs.shape[1]
        n = (S // 28) ** 2
        out = []
        for i in range(0, len(imgs), batch):
            xb = imgs[i:i + batch]
            pv = torch.from_numpy(qwen_pixel_values(xb))
            thw = torch.tensor([[1, S // 14, S // 14]] * len(xb))
            o = self.model(hidden_states=pv, grid_thw=thw).pooler_output            # (len*n, 2048)
            assert o.shape == (len(xb) * n, self.out_dim), o.shape
            out.append(o.reshape(len(xb), n, self.out_dim).to(torch.bfloat16))
        return torch.cat(out)


class DeepEncoderVision:
    """DeepSeek-OCR DeepEncoder (Tiny mode at S = 512); returns projector tokens (B, (S/64)^2, 1280)."""

    def __init__(self, stage: str = "projector"):
        from safetensors.torch import load_file
        from encoder_probe.ep.deepencoder import DeepEncoderModel
        from encoder_probe.ep.hf_partial import fetch_tensors

        pre = ("model.sam_model.", "model.vision_model.", "model.projector.")
        path = fetch_tensors(DS_REPO, DS_SHARD, lambda k: k.startswith(pre),
                             WEIGHTS / "deepseek_ocr_deepencoder.safetensors", revision=DS_REV,
                             rename=lambda k: k[len("model."):])
        with torch.device("meta"):
            m = DeepEncoderModel()
        m.load_state_dict(load_file(str(path)), strict=True, assign=True)
        m.vision_model.embeddings.position_ids = torch.arange(257)[None]
        self.model = m.to(torch.bfloat16).eval()
        self.stage = stage
        self.out_dim = 1024 if stage == "sam" else 1280

    @torch.no_grad()
    def __call__(self, imgs: np.ndarray, batch: int = 4) -> torch.Tensor:
        out = []
        for i in range(0, len(imgs), batch):
            x = (imgs[i:i + batch].astype(np.float32) / 255.0 - 0.5) / 0.5
            t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(next(self.model.parameters()).dtype)
            o = self.model(t, stage=self.stage)
            assert o.shape == (len(x), (imgs.shape[1] // 64) ** 2, self.out_dim), o.shape
            out.append(o.to(torch.bfloat16))
        return torch.cat(out)


def load_vlm(name: str):
    if name == "qwen25vl":
        return QwenVision()
    if name == "deepencoder":
        return DeepEncoderVision()
    raise KeyError(name)
