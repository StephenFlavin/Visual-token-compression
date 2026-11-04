"""Round 3: the full Qwen2.5-VL-3B-Instruct (vision tower + LLM) for LoRA fine-tuning and long-context runs.

Shared by experiments/finetune_qwen.py (R3-1), experiments/longctx_bench.py (R3-2) and tests/test_round3.py.

* Model: transformers 5.22.1 ``Qwen2_5_VLForConditionalGeneration`` at the pinned revision of vlm_encoders.py
  (full weights, ~7.5 GB in the Hugging Face cache), bf16 on CUDA, sdpa attention. ``tiny=True`` builds a
  random-init model from a shrunken config (2 LLM layers, 6 vision blocks) with a byte-level tokenizer, so the
  whole pipeline runs on the CPU in tests without any download.
* Inputs are built without the HF processor: pixel values with ``qwen_pixel_values`` (tested equal to the
  processor in round 2), 256 ``<|image_pad|>`` tokens per 448-px page, the Qwen2.5-VL chat template written out
  by hand, and ``mm_token_type_ids`` (1 on image tokens), which transformers 5 needs to build the 3-D (M-RoPE)
  positions.
* Trainable parts (``make_trainable``): LoRA (r=16, alpha=32, dropout 0.05) on every LLM attention and MLP
  projection (q, k, v, o, gate, up, down), plus the full patch merger and the last 4 vision blocks; everything
  else frozen. Trainable tensors are kept in fp32 (bf16 autocast on CUDA).
* KV accounting: ``kv_bytes`` sums the key/value tensors of a transformers Cache; ``kv_bytes_analytic`` is
  layers x 2 x kv_heads x head_dim x positions x bytes (Qwen2.5-VL-3B: 36 x 2 x 2 x 128 x 2 B = 36,864 B/position).
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import torch
import torch.nn.functional as F

from encoder_probe.ep.vlm_encoders import QWEN_REPO, QWEN_REV, qwen_pixel_values

SYSTEM = "You are a helpful assistant."
LORA_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
PATCH, MERGE = 14, 2


def image_tokens(S: int) -> int:
    return (S // (PATCH * MERGE)) ** 2


# --------------------------------------------------------------------------------------------- tokenizers

class ByteTok:
    """Byte-level tokenizer with the Qwen special tokens (tests / --tiny only)."""
    SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
                "<|image_pad|>", "<|video_pad|>"]

    def __init__(self):
        ids = {s: 256 + i for i, s in enumerate(self.SPECIALS)}
        self.eos = self.pad = ids["<|endoftext|>"]
        self.im_start, self.im_end = ids["<|im_start|>"], ids["<|im_end|>"]
        self.vision_start, self.vision_end = ids["<|vision_start|>"], ids["<|vision_end|>"]
        self.image_pad, self.video_pad = ids["<|image_pad|>"], ids["<|video_pad|>"]
        self.vocab_size = 264
        self.name = "bytes"

    def encode(self, s: str) -> list[int]:
        return list(s.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", "replace")


class HFTok:
    """The checkpoint's tokenizer (Qwen2 BPE) behind the same small interface as ByteTok."""

    def __init__(self, tok):
        self.t = tok
        c = tok.convert_tokens_to_ids
        self.eos = c("<|endoftext|>")
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else self.eos
        self.im_start, self.im_end = c("<|im_start|>"), c("<|im_end|>")
        self.vision_start, self.vision_end = c("<|vision_start|>"), c("<|vision_end|>")
        self.image_pad, self.video_pad = c("<|image_pad|>"), c("<|video_pad|>")
        self.vocab_size = len(tok)
        self.name = f"{QWEN_REPO}@{QWEN_REV[:8]}"

    def encode(self, s: str) -> list[int]:
        return self.t.encode(s, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.t.decode(list(ids), skip_special_tokens=True)


# ------------------------------------------------------------------------------------------------- models

def tiny_config(tok, layers: int = 2, hidden: int = 64, heads: int = 4, kv_heads: int = 2, vision_depth: int = 6,
                max_positions: int = 32768):
    """A shrunken Qwen2.5-VL config with the real architecture (window + full-attention vision blocks, 2x2
    merger, M-RoPE, GQA) for CPU tests."""
    from transformers import Qwen2_5_VLConfig
    head_dim = hidden // heads
    sec = [head_dim // 2 - 2 * (head_dim // 2 * 3 // 8), head_dim // 2 * 3 // 8, head_dim // 2 * 3 // 8]
    return Qwen2_5_VLConfig(
        text_config=dict(vocab_size=tok.vocab_size, hidden_size=hidden, intermediate_size=2 * hidden,
                         num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv_heads,
                         max_position_embeddings=max_positions, rms_norm_eps=1e-6, bos_token_id=tok.eos,
                         eos_token_id=tok.im_end, pad_token_id=tok.pad,
                         rope_parameters={"rope_type": "default", "rope_theta": 1e6, "mrope_section": sec}),
        vision_config=dict(depth=vision_depth, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=hidden, patch_size=PATCH, spatial_merge_size=MERGE,
                           temporal_patch_size=2, window_size=112, fullatt_block_indexes=[2, vision_depth - 1],
                           in_chans=3, tokens_per_second=2),
        image_token_id=tok.image_pad, video_token_id=tok.video_pad, vision_start_token_id=tok.vision_start,
        vision_end_token_id=tok.vision_end, bos_token_id=tok.eos, eos_token_id=tok.im_end, pad_token_id=tok.pad,
        tie_word_embeddings=True)


def load_model(device: torch.device, tiny: bool = False, rope_scaling: dict | None = None, seed: int = 0):
    """-> (model, tokenizer). Real: bf16 on CUDA (fp32 on CPU); tiny: fp32 random init (seeded, so every
    build is identical: needed for exact resume in tests)."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    if tiny:
        tok = ByteTok()
        cfg = tiny_config(tok)
        if rope_scaling:
            cfg.text_config.rope_parameters = {**cfg.text_config.rope_parameters, **rope_scaling}
        cfg._attn_implementation = "sdpa"
        g = torch.random.get_rng_state()
        torch.manual_seed(seed)
        model = Qwen2_5_VLForConditionalGeneration(cfg)
        torch.random.set_rng_state(g)
        return model.to(device).float(), tok
    from transformers import AutoConfig, AutoTokenizer
    tok = HFTok(AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REV))
    kw = {}
    if rope_scaling:
        cfg = AutoConfig.from_pretrained(QWEN_REPO, revision=QWEN_REV)
        cfg.text_config.rope_parameters = {**cfg.text_config.rope_parameters, **rope_scaling}
        kw["config"] = cfg
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_REPO, revision=QWEN_REV, dtype=dtype, attn_implementation="sdpa", **kw)
    return model.to(device), tok


def text_cfg(model):
    return model.config.text_config


def rope_info(model) -> dict:
    t = text_cfg(model)
    return {"rope_parameters": dict(t.rope_parameters), "max_position_embeddings": t.max_position_embeddings}


def make_trainable(model, r: int = 16, alpha: int = 32, dropout: float = 0.05, vision_blocks: int = 4,
                   train_merger: bool = True):
    """LoRA on the LLM's attention + MLP projections; unfreeze the patch merger and the last ``vision_blocks``
    vision blocks; freeze everything else. Trainable tensors are cast to fp32. -> the peft model."""
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS,
                                          bias="none"))
    vis = model.model.visual
    depth = len(vis.blocks)
    mods = [vis.blocks[i] for i in range(depth - vision_blocks, depth)] + ([vis.merger] if train_merger else [])
    for m in mods:
        for p in m.parameters():
            p.requires_grad_(True)
    for n, p in pm.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return pm


def trainable_state(pm) -> dict:
    return {n: p.detach().cpu().clone() for n, p in pm.named_parameters() if p.requires_grad}


def load_trainable_state(pm, sd: dict) -> None:
    params = dict(pm.named_parameters())
    want = {n for n, p in params.items() if p.requires_grad}
    if set(sd) != want:
        raise KeyError(f"trainable mismatch: missing {sorted(want - set(sd))[:3]} extra {sorted(set(sd) - want)[:3]}")
    with torch.no_grad():
        for n, t in sd.items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))


def param_counts(pm) -> dict:
    tr = {n: p.numel() for n, p in pm.named_parameters() if p.requires_grad}
    return {"trainable": sum(tr.values()), "lora": sum(v for n, v in tr.items() if "lora_" in n),
            "vision": sum(v for n, v in tr.items() if "lora_" not in n),
            "total": sum(p.numel() for p in pm.parameters())}


def hf_model(m):
    """The Qwen2_5_VLForConditionalGeneration under a peft wrapper (or the model itself)."""
    return m.get_base_model() if hasattr(m, "get_base_model") else m


# ------------------------------------------------------------------------------------------------- inputs

def chat_ids(tok, prompt: str, n_images: int = 0, n_img_tokens: int = 256, answer: str | None = None,
             doc_text: str | None = None) -> tuple[list[int], list[int], list[int]]:
    """Qwen2.5-VL chat template, written out:
        <|im_start|>system\\n{SYSTEM}<|im_end|>\\n<|im_start|>user\\n
        [<|vision_start|><|image_pad|> x n_img_tokens <|vision_end|>] x n_images [doc_text] {prompt}<|im_end|>\\n
        <|im_start|>assistant\\n [{answer}<|im_end|>]
    -> (input ids, labels (-100 except the answer and its <|im_end|>), mm_token_type_ids)."""
    e = tok.encode
    ids = [tok.im_start] + e(f"system\n{SYSTEM}") + [tok.im_end] + e("\n") + [tok.im_start] + e("user\n")
    types = [0] * len(ids)
    for _ in range(n_images):
        ids += [tok.vision_start] + [tok.image_pad] * n_img_tokens + [tok.vision_end]
        types += [0] + [1] * n_img_tokens + [0]
    tail = e((doc_text or "") + prompt) + [tok.im_end] + e("\n") + [tok.im_start] + e("assistant\n")
    ids += tail
    types += [0] * len(tail)
    labels = [-100] * len(ids)
    if answer is not None:
        a = e(answer) + [tok.im_end]
        ids += a
        labels += a
        types += [0] * len(a)
    return ids, labels, types


def collate(samples: list[dict], tok, device, left_pad: bool = False) -> dict:
    """samples: {"imgs": list of uint8 (S,S,3), "prompt", "answer" (optional), "doc_text" (optional)}."""
    seqs, imgs = [], []
    for s in samples:
        im = s.get("imgs") or []
        n_tok = image_tokens(im[0].shape[0]) if im else 0
        seqs.append(chat_ids(tok, s["prompt"], len(im), n_tok, s.get("answer"), s.get("doc_text")))
        imgs += im
    L = max(len(x[0]) for x in seqs)
    B = len(seqs)
    ids = torch.full((B, L), tok.pad, dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    typ = torch.zeros((B, L), dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, (a, b, c) in enumerate(seqs):
        sl = slice(L - len(a), L) if left_pad else slice(0, len(a))
        ids[i, sl], lab[i, sl], typ[i, sl], att[i, sl] = torch.tensor(a), torch.tensor(b), torch.tensor(c), 1
    out = {"input_ids": ids.to(device), "attention_mask": att.to(device), "labels": lab.to(device),
           "mm_token_type_ids": typ.to(device)}
    if imgs:
        out.update(pixel_inputs(imgs, device))
    return out


def pixel_inputs(imgs: list[np.ndarray], device) -> dict:
    S = imgs[0].shape[0]
    pv = torch.from_numpy(qwen_pixel_values(np.stack(imgs)))
    thw = torch.tensor([[1, S // PATCH, S // PATCH]] * len(imgs), dtype=torch.long)
    return {"pixel_values": pv.to(device), "image_grid_thw": thw.to(device)}


# ------------------------------------------------------------------------------------------ loss / decode

def lm_loss(model, batch: dict, chunk: int = 512) -> tuple[torch.Tensor, int]:
    """Mean next-token CE over answer tokens only. The LM head runs on answer positions in checkpointed chunks,
    so the (positions x 151k) logits are never all alive at once. -> (loss, number of target tokens)."""
    m = hf_model(model)
    m.model.rope_deltas = None
    kw = {k: batch[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values",
                                "image_grid_thw") if k in batch}
    h = m.model(**kw, use_cache=False).last_hidden_state[:, :-1]
    y = batch["labels"][:, 1:]
    sel = y != -100
    hs, ys = h[sel], y[sel]
    tot = h.new_zeros((), dtype=torch.float32)

    def ce(hh, yy):
        return F.cross_entropy(m.lm_head(hh).float(), yy, reduction="sum")

    for i in range(0, len(ys), chunk):
        if torch.is_grad_enabled():
            tot = tot + torch.utils.checkpoint.checkpoint(ce, hs[i:i + chunk], ys[i:i + chunk], use_reentrant=False)
        else:
            tot = tot + ce(hs[i:i + chunk], ys[i:i + chunk])
    return tot / max(1, len(ys)), int(len(ys))


@torch.no_grad()
def generate(model, tok, samples: list[dict], device, max_new_tokens: int, autocast: bool = False) -> list[str]:
    """Greedy batched generation (left padding) with transformers' generate. -> decoded answers."""
    m = hf_model(model)
    b = collate([{**s, "answer": None} for s in samples], tok, device, left_pad=True)
    b.pop("labels")
    m.model.rope_deltas = None
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
        out = m.generate(**b, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=[tok.im_end, tok.eos],
                         pad_token_id=tok.pad, top_p=None, top_k=None, temperature=None)
    res = []
    for row in out[:, b["input_ids"].shape[1]:].tolist():
        cut = [i for i, t in enumerate(row) if t in (tok.im_end, tok.eos)]
        res.append(tok.decode(row[:cut[0]] if cut else row))
    m.model.rope_deltas = None
    return res


def kv_bytes(cache) -> tuple[int, int]:
    """-> (bytes of all key/value tensors in a transformers Cache, positions held)."""
    tot, pos = 0, 0
    for layer in cache.layers:
        for t in (layer.keys, layer.values):
            if t is not None and t.numel():
                tot += t.numel() * t.element_size()
                pos = max(pos, t.shape[-2])
    return tot, pos


def kv_bytes_analytic(model, positions: int, bytes_per: int | None = None) -> int:
    t = text_cfg(model)
    hd = getattr(t, "head_dim", None) or t.hidden_size // t.num_attention_heads
    if bytes_per is None:
        bytes_per = next(hf_model(model).model.language_model.parameters()).element_size()
    return t.num_hidden_layers * 2 * t.num_key_value_heads * hd * positions * bytes_per


def config_hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def lora_module_names(pm) -> list[str]:
    return sorted({re.sub(r"\.lora_[AB]\..*", "", n) for n, _ in pm.named_parameters() if "lora_" in n})
