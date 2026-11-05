"""G2-lite (ROUND3.md): the Qwen2.5-VL patch merger as a frozen / trained / bypassed front end of the round-2 probe.

Input: cached final-block hidden states of the frozen ViT (after all 32 blocks, before the merger), one page =
(1024, 1280) bf16 in the tower's **window order** (cache ``qwen25vl_h32``, written by ep/token_cache.py::build
with split=32; ep/qwen_split.py). The merger needs nothing else: it RMS-normalises each patch, concatenates
each run of 4 consecutive patches (one 2x2 merge group; the window permutation moves whole groups) and maps
5120 -> 5120 -> 2048; ``reverse`` (argsort of the window index, one page) puts the 256 groups back in row-major
order, which is the order of the LLM's image tokens and of the probe's 16 x 16 grid.

Arms (``Front``):
  frozen   pretrained merger, no gradient: (B, 1024, 1280) -> (B, 256, 2048); identical to the model's tokens
  trained  the same merger, initialised from pretrained, trained with the probe
  direct   no merger: the 4 patch states of each group concatenated -> (B, 256, 5120) (the 4 patches in the
           merger's own (dy, dx) order)
followed by a fixed per-dim standardisation (mean / std over the training pages: of the pretrained merger's
output for frozen and trained, of the concatenated states for direct), as run_probe2 standardises its tokens.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

ARMS = ("frozen", "trained", "direct")
MERGER_KEYS = ("ln_q.weight", "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias")


def window_reverse(S: int, patch: int, merge: int, window: int) -> torch.Tensor:
    """(n_groups,) index: merger rows of one page (window order) -> row-major token order."""
    from transformers.vision_utils import get_vision_window_index
    g = S // patch
    wi, _ = get_vision_window_index(torch.tensor([[1, g, g]]), spatial_merge_size=merge, window_size=window,
                                    patch_size=patch, kwargs={})
    return torch.argsort(wi)


def load_merger(weights: Path, prefix: str = "visual.merger."):
    """The pretrained Qwen2.5-VL-3B patch merger (fp32) from the cached visual.* safetensors, plus its geometry."""
    from safetensors import safe_open
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPatchMerger
    with safe_open(str(weights), "pt") as f:
        sd = {k: f.get_tensor(prefix + k).float() for k in MERGER_KEYS}
    out_dim, ctx = sd["mlp.2.weight"].shape[0], sd["ln_q.weight"].shape[0]
    m = Qwen2_5_VLPatchMerger(dim=out_dim, context_dim=ctx, spatial_merge_size=2)
    m.load_state_dict(sd, strict=True)
    return m.float()


class Front(nn.Module):
    """Cached pre-merger page states (B, N, c) -> standardised probe tokens (B, N/4, d) for one arm."""

    def __init__(self, arm: str, merger: nn.Module | None, reverse: torch.Tensor, c: int):
        super().__init__()
        assert arm in ARMS, arm
        self.arm = arm
        self.merger = merger if arm != "direct" else None
        if self.merger is not None:
            self.merger.requires_grad_(arm == "trained")
        self.register_buffer("reverse", reverse.clone().long(), persistent=False)
        d = 4 * c if arm == "direct" else merger.mlp[2].out_features
        self.d = d
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))

    def raw(self, h: torch.Tensor) -> torch.Tensor:
        """(B, N, c) any float dtype -> (B, N/4, d) fp32 row-major tokens, before standardisation."""
        B, N, c = h.shape
        x = h.to(self.mu.device).float()   # cached states live on the CPU; the front may be on the GPU
        if self.arm == "direct":
            return x.reshape(B, N // 4, 4 * c)[:, self.reverse]
        y = self.merger(x.reshape(B * N, c))                      # the merger's own forward (RMSNorm, view, MLP)
        return y.reshape(B, N // 4, -1)[:, self.reverse]

    @torch.no_grad()
    def fit_standardiser(self, h_train, batch: int = 8, keep: bool = False):
        """Per-dim mean / std of ``raw`` over the training pages (streamed; fp64 sums). ``keep``: also return the
        raw train tokens (fp32), so the frozen arm does not run the merger twice."""
        s = torch.zeros(self.d, dtype=torch.float64)
        s2 = torch.zeros(self.d, dtype=torch.float64)
        n, kept = 0, None
        for i in range(0, len(h_train), batch):
            r = self.raw(h_train[i:i + batch])
            if keep:   # preallocated on the CPU: no second copy at the end
                if kept is None:
                    kept = torch.empty((len(h_train), *r.shape[1:]), dtype=r.dtype)
                kept[i:i + len(r)] = r.cpu()
            x = r.double().reshape(-1, self.d)
            s += x.sum(0).cpu()
            s2 += (x * x).sum(0).cpu()
            n += len(x)
        mu = s / n
        var = (s2 / n - mu * mu).clamp_min(0) * n / max(1, n - 1)
        self.mu.copy_(mu.float())
        self.sd.copy_(var.sqrt().float().clamp_min(1e-3))
        return kept

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (self.raw(h) - self.mu) / self.sd


class G2Reader(nn.Module):
    """Front + GridProbe with the probe interface run_probe.evaluate expects (features / read / parameters)."""

    def __init__(self, front: Front | None, probe: nn.Module):
        super().__init__()
        self.front, self.probe = front, probe

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe.features(x if self.front is None else self.front(x))

    def read(self, feat, page_idx, line_idx):
        return self.probe.read(feat, page_idx, line_idx)

    @staticmethod
    def recon_loss(feat, page):
        return feat.float().sub(page).pow(2).mean()          # = GridProbe.recon_loss (MSE to the clean page)


@torch.no_grad()
def precompute(front: Front, h, batch: int = 8) -> torch.Tensor:
    """Frozen arm: the (deterministic, gradient-free) front applied once to every page -> fp32 tokens."""
    return torch.cat([front(h[i:i + batch]) for i in range(0, len(h), batch)])


def trainable_names(model: nn.Module) -> list[str]:
    return sorted(n for n, p in model.named_parameters() if p.requires_grad)


def edit_ops(pairs) -> dict:
    """CER split for (ref, hyp) pairs: substitutions, insertions (extra hyp chars), deletions (missing ref chars),
    and the hyp/ref length ratio. (sub + ins + del) / ref_chars equals the CER of run_probe.evaluate."""
    from rapidfuzz.distance import Levenshtein
    sub = ins = dele = nr = nh = 0
    for r, h in pairs:
        for tag, *_ in Levenshtein.editops(r, h):
            if tag == "replace":
                sub += 1
            elif tag == "insert":
                ins += 1
            else:
                dele += 1
        nr += len(r)
        nh += len(h)
    n = max(1, nr)
    return {"cer": (sub + ins + dele) / n, "sub": sub / n, "ins": ins / n, "del": dele / n, "ref_chars": nr,
            "hyp_chars": nh, "len_ratio": nh / n}


@torch.no_grad()
def decode_pairs(model, tok, pages, enc, classes, bf16: bool, batch_pages: int = 2) -> list[tuple[str, str]]:
    """All (ref, hyp) line pairs on ``pages``, decoded exactly as run_probe.evaluate does."""
    from encoder_probe.ep.probe import greedy
    dev = next(model.parameters()).device
    model.eval()
    out = []
    for s in range(0, len(pages), batch_pages):
        pb = list(range(s, min(len(pages), s + batch_pages)))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
            feat = model.features(tok[pb].to(dev))
        pi, li, refs = [], [], []
        for k, p in enumerate(pb):
            for i, r in enumerate(pages[p].refs):
                if r:
                    pi.append(k), li.append(i), refs.append(r)
        if not pi:
            continue
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
            lp = model.read(feat, torch.tensor(pi, device=dev), torch.tensor(li, device=dev))
        for d, r in zip(greedy(lp.float()), refs):
            out.append((r, enc.clean_hyp(enc.decode("".join(classes[j - 1] for j in d)))))
    model.train()
    return out
