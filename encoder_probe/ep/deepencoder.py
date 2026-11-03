"""DeepEncoder of deepseek-ai/DeepSeek-OCR (vision path only), vendored for inference on CPU.

Minimal re-implementation of the vision modules in the checkpoint's ``deepencoder.py`` (MIT License,
Copyright (c) 2023 DeepSeek), stripped of training/flash/tensor-parallel code and of the einops/easydict
dependencies; module and parameter names match the checkpoint so its tensors load with ``strict=True``.

  image (B,3,S,S), normalised with mean = std = 0.5
   -> sam_model: SAM ViT-B/16 (12 blocks, window 14, global attention in blocks 2,5,8,11, decomposed rel-pos)
      -> neck (768->256) -> net_2 (stride 2, 256->512) -> net_3 (stride 2, 512->1024)   = 16x conv compressor
      => (B, 1024, S/64, S/64)
   -> vision_model: CLIP-L/14 whose patch embeddings are replaced by the SAM output (CLS + interpolated
      abs pos, pre-LN, 24 blocks, quick-GELU)  => (B, 1 + (S/64)^2, 1024)
   -> concat(CLIP[:, 1:], SAM flattened) (B, N, 2048) -> projector (linear 2048 -> 1280)
      => (B, N, 1280): the image tokens the language model sees (it also inserts a learned newline token
         after each row and a view separator, which carry no image information and are not returned).
Tiny mode is S = 512 -> N = 64 tokens (one per 64 x 64 px); Small is S = 640 -> 100.
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------------- SAM ViT-B


def _abs_pos_sam(abs_pos: torch.Tensor, tgt: int) -> torch.Tensor:
    if abs_pos.size(1) == tgt:
        return abs_pos
    p = F.interpolate(abs_pos.permute(0, 3, 1, 2).float(), size=(tgt, tgt), mode="bicubic",
                      antialias=True, align_corners=False).to(abs_pos.dtype)
    return p.permute(0, 2, 3, 1)


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.lin1, self.lin2, self.act = nn.Linear(dim, mlp_dim), nn.Linear(mlp_dim, dim), nn.GELU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * ((x - u) / torch.sqrt(s + self.eps)) + self.bias[:, None, None]


def _rel_pos(q: int, k: int, rel: torch.Tensor) -> torch.Tensor:
    d = int(2 * max(q, k) - 1)
    if rel.shape[0] != d:
        rel = F.interpolate(rel.float().reshape(1, rel.shape[0], -1).permute(0, 2, 1), size=d,
                            mode="linear").to(rel.dtype).reshape(-1, d).permute(1, 0)
    qc = torch.arange(q)[:, None] * max(k / q, 1.0)
    kc = torch.arange(k)[None, :] * max(q / k, 1.0)
    return rel[((qc - kc) + (k - 1) * max(q / k, 1.0)).long()]


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, input_size: tuple[int, int]):
        super().__init__()
        self.num_heads = num_heads
        hd = dim // num_heads
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, hd))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, hd))

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        Rh, Rw = _rel_pos(H, H, self.rel_pos_h), _rel_pos(W, W, self.rel_pos_w)
        r_q = q.reshape(-1, H, W, q.shape[-1])
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).unsqueeze(-1).reshape(-1, H * W, H, 1)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).unsqueeze(-2).reshape(-1, H * W, 1, W)
        q, k, v = (t.view(B, self.num_heads, H * W, -1) for t in (q, k, v))
        bias = (rel_h.view(B, self.num_heads, H * W, H, 1) + rel_w.view(B, self.num_heads, H * W, 1, W))
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.view(B, self.num_heads, H * W, H * W))
        return self.proj(x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1))


def _win_part(x, ws):
    B, H, W, C = x.shape
    ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    Hp, Wp = H + ph, W + pw
    x = x.view(B, Hp // ws, ws, Wp // ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)
    return x, (Hp, Wp)


def _win_unpart(w, ws, pad_hw, hw):
    (Hp, Wp), (H, W) = pad_hw, hw
    B = w.shape[0] // (Hp * Wp // ws // ws)
    x = w.view(B, Hp // ws, Wp // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous()


class Block(nn.Module):
    def __init__(self, dim, num_heads, window_size, input_size):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1, self.norm2 = norm(dim), norm(dim)
        self.attn = Attention(dim, num_heads, input_size if window_size == 0 else (window_size, window_size))
        self.mlp = MLPBlock(dim, dim * 4)
        self.window_size = window_size

    def forward(self, x):
        sc = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad = _win_part(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = _win_unpart(x, self.window_size, pad, (H, W))
        x = sc + x
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, k=16, in_chans=3, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=k, stride=k)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


class SAMEncoder(nn.Module):
    def __init__(self, img_size=1024, patch=16, dim=768, depth=12, heads=12, out_chans=256, window=14,
                 global_idx=(2, 5, 8, 11)):
        super().__init__()
        g = img_size // patch
        self.patch_embed = PatchEmbed(patch, 3, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, g, g, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, 0 if i in global_idx else window, (g, g))
                                     for i in range(depth)])
        self.neck = nn.Sequential(nn.Conv2d(dim, out_chans, 1, bias=False), LayerNorm2d(out_chans),
                                  nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False), LayerNorm2d(out_chans))
        self.net_2 = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + _abs_pos_sam(self.pos_embed, x.size(1))
        for b in self.blocks:
            x = b(x)
        return self.net_3(self.net_2(self.neck(x.permute(0, 3, 1, 2))))


# --------------------------------------------------------------------------------------------- CLIP-L


def _abs_pos_clip(abs_pos: torch.Tensor, n_tok: int) -> torch.Tensor:
    """abs_pos (1, 1+L, C) -> (1, n_tok, C): bicubic resize of the patch grid, CLS kept."""
    a = abs_pos.squeeze(0)
    cls, old = a[:1], a[1:]
    src, tgt, dim = int(math.sqrt(a.shape[0] - 1)), int(math.sqrt(n_tok)), a.shape[-1]
    if src == tgt:
        return abs_pos
    new = F.interpolate(old.view(1, src, src, dim).permute(0, 3, 1, 2).float(), size=(tgt, tgt),
                        mode="bicubic", antialias=True, align_corners=False).to(a.dtype)
    return torch.cat([cls, new.permute(0, 2, 3, 1).reshape(tgt * tgt, dim)], 0)[None]


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class CLIPEmbeddings(nn.Module):
    def __init__(self, dim=1024, image_size=224, patch=14):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(dim))
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=False)  # unused (SAM feeds)
        n = (image_size // patch) ** 2 + 1
        self.position_embedding = nn.Embedding(n, dim)
        self.register_buffer("position_ids", torch.arange(n)[None], persistent=False)

    def forward(self, patch_embeds):
        B = patch_embeds.shape[0]
        x = patch_embeds.flatten(2).transpose(1, 2)
        x = torch.cat([self.class_embedding.expand(B, 1, -1).to(x.dtype), x], 1)
        return x + _abs_pos_clip(self.position_embedding(self.position_ids), x.size(1))


class CLIPAttention(nn.Module):
    def __init__(self, dim=1024, heads=16):
        super().__init__()
        self.heads = heads
        self.qkv_proj, self.out_proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv_proj(x).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(o.permute(0, 2, 1, 3).reshape(B, L, D))


class CLIPMLP(nn.Module):
    def __init__(self, dim=1024, hid=4096):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hid), nn.Linear(hid, dim)

    def forward(self, x):
        return self.fc2(quick_gelu(self.fc1(x)))


class CLIPBlock(nn.Module):
    def __init__(self, dim=1024, heads=16):
        super().__init__()
        self.self_attn, self.mlp = CLIPAttention(dim, heads), CLIPMLP(dim)
        self.layer_norm1, self.layer_norm2 = nn.LayerNorm(dim, eps=1e-5), nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x):
        h = x + self.self_attn(self.layer_norm1(x))
        return h + self.mlp(self.layer_norm2(h))


class CLIPTransformer(nn.Module):
    def __init__(self, layers=24):
        super().__init__()
        self.layers = nn.ModuleList([CLIPBlock() for _ in range(layers)])

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x


class CLIPVit(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = CLIPEmbeddings()
        self.transformer = CLIPTransformer()
        self.pre_layrnorm = nn.LayerNorm(1024, eps=1e-5)

    def forward(self, patch_embeds):
        return self.transformer(self.pre_layrnorm(self.embeddings(patch_embeds)))


class Projector(nn.Module):
    def __init__(self, din=2048, dout=1280):
        super().__init__()
        self.layers = nn.Linear(din, dout)

    def forward(self, x):
        return self.layers(x)


class DeepEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam_model, self.vision_model, self.projector = SAMEncoder(), CLIPVit(), Projector()

    def forward(self, x: torch.Tensor, stage: str = "projector") -> torch.Tensor:
        """x (B,3,S,S) normalised -> (B, (S/64)^2, 1280) projector tokens (``stage='sam'``: SAM+conv, 1024-d)."""
        f1 = self.sam_model(x)                                      # (B, 1024, S/64, S/64)
        sam = f1.flatten(2).permute(0, 2, 1)
        if stage == "sam":
            return sam
        f2 = self.vision_model(f1)                                  # (B, 1+N, 1024)
        return self.projector(torch.cat([f2[:, 1:], sam], -1))
