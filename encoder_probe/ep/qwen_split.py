"""Round 3 (R3-0b): the Qwen2.5-VL vision tower split at a block index k.

    head(pixels)  = patch_embed -> window reorder -> blocks[:k]          frozen, run once per page and cached
    tail(hidden)  = blocks[k:] -> patch merger -> reverse window order   trained (R3-1's vision unfreezing, k = 28)

``tail(head(x))`` is the model's own forward (transformers 5.22.1 ``Qwen2_5_VisionTransformerPretrainedModel``):
the rotary embeddings, cu_seqlens, window index and the per-block choice of window / full attention come from the
same helpers (``transformers.vision_utils``) and in the same order as its ``forward``; tested to reproduce
``pooler_output`` (tests/test_r3_0b.py, and on the real weights in ROUND3.md).

Cached hidden states are per page, (N, d) = (1024, 1280) at S = 448, in the model's **window order** (the order the
blocks see). The window permutation only moves tokens within an image, so B pages concatenated are exactly the
model's packed sequence for B images of the same grid.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.vision_utils import (get_max_seqlen, get_vision_attention_seqlens, get_vision_position_ids,
                                       get_vision_window_index)


def vit_meta(model, grid_thw: torch.Tensor) -> dict:
    """Positions and attention layout for packed images, exactly as the model's forward builds them."""
    position_ids = get_vision_position_ids(grid_thw, model.spatial_merge_size, kwargs={})
    cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, model.config, kwargs={})
    window_index, cu_window_seqlens = get_vision_window_index(
        grid_thw, spatial_merge_size=model.spatial_merge_size, window_size=model.window_size,
        patch_size=model.patch_size, kwargs={})
    max_window_seqlen = get_max_seqlen(cu_window_seqlens, model.config, kwargs={}, kwarg_name="max_window_seqlen")
    return dict(position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, window_index=window_index,
                cu_window_seqlens=cu_window_seqlens, max_window_seqlen=max_window_seqlen,
                reverse=torch.argsort(window_index))


def _pos_emb(model, meta: dict, like: torch.Tensor):
    seq_len = like.shape[0]
    pe = model.rotary_pos_emb(like, meta["position_ids"])
    return tuple(model.permute_input_for_window_attn(f, meta["window_index"], seq_len=seq_len) for f in pe)


def run_blocks(model, h: torch.Tensor, meta: dict, start: int, stop: int, pos_emb=None) -> torch.Tensor:
    """blocks[start:stop] on the packed, window-ordered sequence h (as the model's forward loop)."""
    pos_emb = _pos_emb(model, meta, h) if pos_emb is None else pos_emb
    for i in range(start, stop):
        if i in model.fullatt_block_indexes:
            cu, mx = meta["cu_seqlens"], meta["max_seqlen"]
        else:
            cu, mx = meta["cu_window_seqlens"], meta["max_window_seqlen"]
        h = model.blocks[i](h, cu_seqlens=cu, max_seqlen=mx, position_embeddings=pos_emb)
    return h


def head(model, pixel_values: torch.Tensor, grid_thw: torch.Tensor, k: int, meta: dict | None = None):
    """-> hidden states entering block k, (total patches, d), window order."""
    meta = meta or vit_meta(model, grid_thw)
    h = model.patch_embed(pixel_values)
    h = model.permute_input_for_window_attn(h, meta["window_index"], seq_len=h.shape[0])
    return run_blocks(model, h, meta, 0, k)


def tail(model, h: torch.Tensor, grid_thw: torch.Tensor, k: int, meta: dict | None = None) -> torch.Tensor:
    """hidden states entering block k (window order) -> merger tokens in the model's output (row-major) order."""
    meta = meta or vit_meta(model, grid_thw)
    h = run_blocks(model, h, meta, k, len(model.blocks))
    return model.merger(h)[meta["reverse"]]


class QwenViTTail(nn.Module):
    """Trainable copy of blocks[k:] + merger of a Qwen2.5-VL vision tower, on (B, N, d) cached page states
    (window order) -> (B, n_tok, out_dim) merger tokens (row-major), the LLM's image tokens.

    Holds references to the tower's own block / merger modules (so their forward code is the model's) plus the
    small helpers ``tail`` needs; the frozen head is not kept."""

    def __init__(self, tower, k: int, S: int):
        super().__init__()
        self.k, self.S = k, S
        self.blocks = nn.ModuleList(tower.blocks[k:])
        self.merger = tower.merger
        self.rotary_pos_emb = tower.rotary_pos_emb
        self.config = tower.config
        self.spatial_merge_size, self.window_size = tower.spatial_merge_size, tower.window_size
        self.patch_size, self.spatial_merge_unit = tower.patch_size, tower.spatial_merge_unit
        self.full = [i - k for i in tower.fullatt_block_indexes if i >= k]
        self.permute_input_for_window_attn = tower.permute_input_for_window_attn
        self.g = S // self.patch_size
        self._meta: dict[int, dict] = {}

    def meta(self, B: int, device) -> dict:
        if B not in self._meta:
            thw = torch.tensor([[1, self.g, self.g]] * B)
            self._meta[B] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vit_meta(self, thw).items()}
        return self._meta[B]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, N, d = h.shape
        meta = self.meta(B, h.device)
        x = h.reshape(B * N, d)
        pe = _pos_emb(self, meta, x)
        for j, blk in enumerate(self.blocks):
            cu, mx = ((meta["cu_seqlens"], meta["max_seqlen"]) if j in self.full
                      else (meta["cu_window_seqlens"], meta["max_window_seqlen"]))
            x = blk(x, cu_seqlens=cu, max_seqlen=mx, position_embeddings=pe)
        out = self.merger(x)[meta["reverse"]]
        return out.reshape(B, -1, out.shape[-1])
