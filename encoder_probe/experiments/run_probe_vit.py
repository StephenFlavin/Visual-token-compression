"""R3-0b: the round-2 probe on Qwen2.5-VL tokens with the last ViT blocks + patch merger trained (ROUND3.md).

    python3 encoder_probe/experiments/run_probe_vit.py --encoding latin_p8 --train-pages 1000 --steps 10000 \
        --run-name round3_cpu_vit4_tp1000

The frozen part of the tower (patch embed + blocks 0..K-1, K = 28 by default) is run once per page by
build_tokens2.py --split K, which caches the hidden states entering block K (cache name qwen25vl_hK, bf16, window
order). Here blocks K..31 and the merger (R3-1's vision unfreezing: 4 blocks + merger, 115.5M parameters) are trained
together with the GridProbe of run_probe2.py. Everything else is run_probe2's recipe: 2 pages x 16 lines per
step, CTC + aux MSE x10 to the clean page, AdamW (fused) with OneCycle (5% warm-up), weight decay 1e-4, grad clip 5
(over all trained parameters), seed 0, eval on 4 test pages every 500 steps and on all test pages at the end, the
round-1 evaluate(). The probe gets lr 2e-3 (run_probe2), the ViT part R3-1's --lr-vision 2e-5, both on the same
OneCycle schedule. The merger tokens are standardised with the frozen-token statistics of run_probe2 (the
qwen25vl token cache's train split: identical to the frozen run's input at step 0), stored once per cache in
``<hidden cache>.stats.pt``. Resume checkpoints every --ckpt-every steps (exact resume, tested).
Rows: encoder_probe/results/<run>/runs.jsonl, encoder = "qwen25vl_vit<n>" (n = trained blocks).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "encoder_probe" / "experiments"))
from encoder_probe.ep import token_cache  # noqa: E402
from encoder_probe.ep.geom import pages_digest  # noqa: E402
from encoder_probe.ep.probe_grid import GridProbe  # noqa: E402
from encoder_probe.ep.qwen_split import QwenViTTail  # noqa: E402
from encoder_probe.ep.vlm_encoders import GEOMS  # noqa: E402
from run_probe import evaluate  # noqa: E402

FAIR_KEYS = ("steps", "train_pages", "test_pages", "batch_pages", "lines_per_page", "lr", "lr_vision", "recon_weight",
             "split", "seed", "weight_decay", "clip", "pct_start", "cond")


def load_tail(split: int, S: int = 448, tiny: bool = False) -> QwenViTTail:
    """blocks[split:] + merger of the Qwen2.5-VL-3B vision tower in fp32 (only those tensors are materialised);
    tiny = the random-init tower of ep/qwen_ft.py (tests)."""
    if tiny:
        from encoder_probe.ep import qwen_ft
        m, _ = qwen_ft.load_model(torch.device("cpu"), tiny=True)
        return QwenViTTail(m.model.visual, split, S).float()
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLVisionRotaryEmbedding)
    from encoder_probe.ep.hf_partial import fetch_tensors
    from encoder_probe.ep.vlm_encoders import QWEN_REPO, QWEN_REV, QWEN_SHARD, WEIGHTS
    cfg = json.loads(Path(hf_hub_download(QWEN_REPO, "config.json", revision=QWEN_REV)).read_text())
    vc = Qwen2_5_VLVisionConfig(**cfg["vision_config"])
    vc._attn_implementation = "sdpa"
    path = fetch_tensors(QWEN_REPO, QWEN_SHARD, lambda k: k.startswith("visual."),
                         WEIGHTS / "qwen25vl3b_visual.safetensors", revision=QWEN_REV)
    with torch.device("meta"):
        tower = Qwen2_5_VisionTransformerPretrainedModel(vc)
    keep = tuple(f"blocks.{i}." for i in range(split, vc.depth)) + ("merger.",)
    sd = {k[len("visual."):]: v for k, v in load_file(str(path)).items() if k[len("visual."):].startswith(keep)}
    missing, unexpected = tower.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected and all(not k.startswith(keep) for k in missing), (unexpected, missing[:5])
    tower.rotary_pos_emb = Qwen2_5_VLVisionRotaryEmbedding(vc)                 # non-persistent buffers
    tail = QwenViTTail(tower, split, S)
    del tower, sd
    return tail.float()


class TunedProbe(nn.Module):
    """cached hidden states (B, N, d_vit) -> trained ViT tail -> merger tokens, standardised -> GridProbe."""

    def __init__(self, tail: QwenViTTail, probe: GridProbe, mu: torch.Tensor, sd: torch.Tensor):
        super().__init__()
        self.tail, self.probe = tail, probe
        self.register_buffer("mu", mu.reshape(1, 1, -1).float())
        self.register_buffer("sd", sd.reshape(1, 1, -1).float())

    def tokens(self, h: torch.Tensor) -> torch.Tensor:
        return (self.tail(h.float()) - self.mu) / self.sd

    def features(self, h: torch.Tensor) -> torch.Tensor:
        return self.probe.features(self.tokens(h))

    def read(self, feat, page_idx, line_idx):
        return self.probe.read(feat, page_idx, line_idx)

    recon_loss = staticmethod(GridProbe.recon_loss)


def token_stats(hpath: Path, encoding: str, cond: str, n_train: int, n_test: int, tr, te, tail, log=print):
    """(mu, sd) of the frozen merger tokens on the train pages, exactly as run_probe2 computes them; from the
    qwen25vl token cache if it exists, else from the frozen tail on the cached states. Saved next to the cache."""
    sp = hpath.with_name(hpath.name + ".stats.pt")
    if sp.exists():
        st = torch.load(sp, weights_only=False)
        if st["digest_train"] == pages_digest(tr):
            return st["mu"], st["sd"], st["source"]
    tp = token_cache.cache_path("qwen25vl", cond, encoding, n_train, n_test)
    if tp.exists():
        d = token_cache._load(tp, tr, te)
        tok, source = d["train"].float(), "qwen25vl token cache"
        del d
    else:
        log("[vit] no frozen token cache: computing the token statistics with the frozen tail", flush=True)
        h = token_cache._load(hpath, tr, te, mmap=True)["train"]
        with torch.no_grad():
            tok = torch.cat([tail(h[i:i + 2].float()) for i in range(0, len(h), 2)])
        source = "frozen tail on cached states"
    mu = tok.mean((0, 1), keepdim=True)
    sd = tok.std((0, 1), keepdim=True).clamp_min(1e-3)
    del tok
    torch.save({"mu": mu, "sd": sd, "source": source, "digest_train": pages_digest(tr)}, sp)
    return mu, sd, source


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--cond", default="clean")
    ap.add_argument("--split", type=int, default=28, help="first trained ViT block (Qwen2.5-VL-3B has 32)")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--train-pages", type=int, default=1000)
    ap.add_argument("--test-pages", type=int, default=16)
    ap.add_argument("--batch-pages", type=int, default=2)
    ap.add_argument("--lines-per-page", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3, help="probe (run_probe2)")
    ap.add_argument("--lr-vision", type=float, default=2e-5, help="ViT blocks + merger (finetune_qwen.py)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--pct-start", type=float, default=0.05)
    ap.add_argument("--recon-weight", type=float, default=10.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="round3_cpu_vit4_tp1000")
    ap.add_argument("--ckpt-every", type=int, default=200, help="resume checkpoint interval (0 = off)")
    ap.add_argument("--stop-after", type=int, default=0, help="exit after this step (resume tests)")
    ap.add_argument("--tiny", action="store_true", help="tiny random-init tower (tests)")
    ap.add_argument("--out-root", default=str(ROOT / "encoder_probe" / "results"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "encoder_probe" / "cache" / "ckpt"))
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)     # fp32 on CPU (no native bf16 here): denormals cost 5.5x (AGENTS.md)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    n_vit = 32 - a.split if not a.tiny else None
    out = Path(a.out_root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tail = load_tail(a.split, tiny=a.tiny)
    n_vit = len(tail.blocks)
    encoder = f"qwen25vl_vit{n_vit}"
    key = dict(encoding=a.encoding, encoder=encoder, cond=a.cond, steps=a.steps)
    rows = out / "runs.jsonl"
    if rows.exists():
        for l in rows.read_text().splitlines():
            if l.strip() and all(json.loads(l).get(k) == v for k, v in key.items()) and \
                    json.loads(l).get("train_pages") == a.train_pages:
                print(f"[vit] skip, done: {key}")
                return 0
    S, P = GEOMS["qwen25vl"]
    G = S // P
    enc, tr, te = token_cache.split_pages(a.encoding, S, a.train_pages, a.test_pages)
    hname = token_cache.hidden_encoder("qwen25vl", a.split)
    hpath = token_cache.cache_path(hname, a.cond, a.encoding, a.train_pages, a.test_pages)
    if not hpath.exists():
        raise SystemExit(f"[vit] missing {hpath}: run build_tokens2.py --encodings {a.encoding} "
                         f"--train-pages {a.train_pages} --split {a.split} first")
    hd = token_cache._load(hpath, tr, te, mmap=True)
    h_tr, h_te = hd["train"], hd["test"]                       # bf16, memory-mapped; batches go to fp32
    assert h_tr.shape[:2] == (len(tr), (S // 14) ** 2) and hd.get("split_block") == a.split, h_tr.shape
    mu, sd, stats_src = token_stats(hpath, a.encoding, a.cond, a.train_pages, a.test_pages, tr, te, tail)
    pix_tr = torch.from_numpy(np.stack([p.img for p in tr])).permute(0, 3, 1, 2).contiguous()   # clean, uint8
    for pg in tr:          # train images now live in pix_tr (frees ~0.6 GB at 1000 pages)
        pg.img = None
    classes = enc.classes
    cidx = {c: i + 1 for i, c in enumerate(classes)}
    tgt = [[torch.tensor([cidx[x] for x in lab], dtype=torch.long) for lab in p.labels] for p in tr]
    chars = sum(p.n_chars for p in te)
    ppc = S * S * len(te) / max(1, chars)
    cpt = chars / (len(te) * G * G)
    d_tok = mu.shape[-1]
    probe = GridProbe(enc.pitch, len(classes), d=d_tok, G=G, P=P)
    model = TunedProbe(tail, probe, mu, sd)
    vit_p = list(tail.parameters())
    n_vit_params = sum(p.numel() for p in vit_p)
    groups = [{"params": list(probe.parameters()), "lr": a.lr}, {"params": vit_p, "lr": a.lr_vision}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay, fused=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr_vision], total_steps=a.steps,
                                                pct_start=a.pct_start)
    fair = {k: getattr(a, k) for k in FAIR_KEYS}
    fair_hash = hashlib.sha1(json.dumps(fair, sort_keys=True).encode()).hexdigest()[:10]
    print(f"[vit] {key} S={S} P={P} G={G} pitch={enc.pitch} lines/page={enc.n_lines} classes={len(classes)} "
          f"train={len(tr)} test={len(te)} px2/char={ppc:.2f} chars/token={cpt:.1f} vit_params={n_vit_params} "
          f"stats={stats_src} fair_hash={fair_hash} ({time.time() - t0:.0f}s)", flush=True)
    curve, t_train, start, t_prev = [], time.time(), 1, 0.0
    tp = f"__tp{a.train_pages}"
    ck = Path(a.ckpt_dir) / f"{a.run_name}__{a.encoding}__{encoder}__{a.cond}__{a.steps}{tp}.pt"
    if a.ckpt_every and ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
        curve, start, t_prev = st["curve"], st["step"] + 1, st["t_train"]
        print(f"[vit] resumed from step {st['step']} ({ck.name})", flush=True)
    t_train -= t_prev
    params = [p for g in groups for p in g["params"]]
    for step in range(start, a.steps + 1):
        pb = rng.choice(len(tr), a.batch_pages, replace=False)
        pi, li, ys = [], [], []
        for k, p in enumerate(pb):
            ok = [i for i, t in enumerate(tgt[p]) if len(t)]
            for i in rng.choice(ok, min(a.lines_per_page, len(ok)), replace=False):
                pi.append(k), li.append(int(i)), ys.append(tgt[p][i])
        ix = torch.as_tensor(pb)
        feat = model.features(h_tr[ix])
        logp = model.read(feat, torch.tensor(pi), torch.tensor(li)).float()
        T = logp.shape[0]
        ctc = F.ctc_loss(logp, torch.cat(ys), torch.full((len(ys),), T, dtype=torch.long),
                         torch.tensor([len(y) for y in ys]), blank=0, zero_infinity=True)
        rec = model.recon_loss(feat, pix_tr[ix].float() / 127.5 - 1)
        loss = ctc + a.recon_weight * rec
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            c, _ = evaluate(model, h_te, te, enc, classes, max_pages=4, bf16=False, device="cpu")
            curve.append((step, round(c, 5)))
            print(f"[vit] step {step} ctc {ctc.item():.3f} rec {rec.item():.4f} val_cer {c:.4f} "
                  f"({(time.time() - t_train) / step:.3f} s/step, {time.time() - t0:.0f}s)", flush=True)
        stop = a.stop_after and step == a.stop_after
        if a.ckpt_every and (step % a.ckpt_every == 0 or stop) and step < a.steps:
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "curve": curve,
                        "step": step, "t_train": time.time() - t_train}, tmp)
            tmp.replace(ck)
        if stop:
            print(f"[vit] stopped after step {step} (--stop-after)", flush=True)
            return 0
    cer, samples = evaluate(model, h_te, te, enc, classes, bf16=False, device="cpu")
    row = {**key, "phase": "probe", "round": 3, "r3": "0b", "scheme": f"{enc.family}@{encoder}/{a.cond}",
           "family": enc.family, "pitch": enc.pitch, "S": S, "P": P, "G": G, "n_tokens": G * G, "d": d_tok,
           "px_per_char": ppc, "budget": round(ppc, 3), "chars_per_token": cpt, "chars_per_vlm_token": cpt,
           "cer": cer, "score_v": 2, "bf16": False, "device": "cpu", "enc_compute": hd.get("enc_compute"),
           "vit_split": a.split, "vit_blocks_trained": n_vit, "vit_params": n_vit_params, "lr_vision": a.lr_vision,
           "stats_source": stats_src, "fair": fair, "fair_hash": fair_hash, "recon_weight": a.recon_weight,
           "train_pages": len(tr), "test_pages": len(te), "lr": a.lr, "n_test_chars": chars,
           "digest_test": pages_digest(te), "curve": curve, "s_per_step": (time.time() - t_train) / a.steps,
           "train_time_s": time.time() - t0, "smoke": "smoke" in a.run_name or a.tiny, "samples": samples}
    print(f"[vit] TEST {key} cer={cer:.4f} px2/char={ppc:.2f} chars/token={cpt:.1f}", flush=True)
    for r, h in samples[:2]:
        print(f"   ref: {r[:90]!r}\n   hyp: {h[:90]!r}")
    with open(rows, "a") as f:
        f.write(json.dumps(row) + "\n")
    ck.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
