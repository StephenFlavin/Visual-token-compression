"""Test B (docs/TEXT_SLOTS.md): text-side slots against page pixels, same reader LLM, same budget.

Does a slot built from r text tokens carry more per context position than a vision token built from a rendered
page? No pixels and no vision tower: the encoder reads the LLM's own (frozen) token embeddings.

* encoder (LocalSlots): slot i = one linear map of its own r consecutive token embeddings (initialised to their
  mean, so at r = 1 the slots ARE the token embeddings and the reader copies them from step 0), after an optional
  small transformer over the window whose output starts at zero. Local slots turn copying into unpacking r tokens
  per slot; Stage 0 (text_compress.py) used m global memory slots and learned gist, not copying.
* reader: Qwen2.5-VL-3B's LLM with the same LoRA as the R3-1 image arms (r=16, alpha 32, lr 2e-4), reading
  [m slots][prompt "Repeat the text exactly:"] and rebuilding the L tokens. ``--decoder frozen``: no LoRA, the
  unmodified LLM reads the slots (the encoder alone is trained).
* ladder: ``--init-from`` warm-starts a ratio from the previous one (LoRA and mixer copied; the pooling map of
  ratio r = k * r' starts as the mean of k adjacent old slots).
* text: ``--text`` is the TRAINING text: windows of WikiText-103 with their words shuffled (default, as the
  Runs 1-2 image arms train: no language prior to lean on, so the gist shortcut of Stage 0 is closed) or natural.
* eval on held-out windows of BOTH natural and shuffled text: teacher-forced loss, greedy CER, a BLIND control
  (each window read from another window's slots) and the PRIOR (the unmodified LLM with no slots).
  reference = min(blind, prior) per window; bits per slot = (reference - loss) * tokens / slots / ln 2, the same
  measure as capacity_curve.py's bits per vision token. Per-window losses are stored so analyse_slots.py can
  bootstrap.

    python3 encoder_probe/experiments/text_slots.py --ratio 1 --smoke    # GPU: s/step, peak memory (longest sequence)
    python3 encoder_probe/experiments/text_slots.py --ratio 1 --steps 0 --run-name tb    # the untrained reader
    python3 encoder_probe/experiments/text_slots.py --ratio 1 --steps 250 --run-name tb  # gate G1
    python3 encoder_probe/experiments/text_slots.py --ratio 2 --steps 1000 --run-name tb \\
        --init-from encoder_probe/cache/ts_encoders/tb__r1__shuffled__lora.pt            # gate G2
    python3 encoder_probe/experiments/text_slots.py --ratio 4 --tiny --steps 30 --device cpu   # CPU test model

Rows: encoder_probe/results/<run>/runs.jsonl (phase "slots"); checkpoints (--ckpt-every, exact resume) in
encoder_probe/cache/ckpt/; the trained encoder + LoRA of every finished arm in encoder_probe/cache/ts_encoders/.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_compress as S0  # noqa: E402
from encoder_probe.ep import qwen_ft as Q  # noqa: E402
from encoder_probe.ep.encoders import resolve_device  # noqa: E402

FAIR_KEYS = ["seq_tokens", "batch_size", "lr", "lr_enc", "warmup", "weight_decay", "clip", "lora_r", "lora_alpha",
             "lora_dropout", "seed", "eval_windows", "val_windows", "tiny", "decoder", "enc_layers", "text"]


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ratio", type=int, required=True, help="text tokens per slot (L must divide by it)")
    ap.add_argument("--seq-tokens", type=int, default=480, help="L: text tokens per window (same for every arm)")
    ap.add_argument("--steps", type=int, default=3000, help="0 = evaluate the initial state (the r = 1 gate)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--text", choices=["natural", "shuffled"], default="shuffled", help="training text")
    ap.add_argument("--decoder", choices=["lora", "frozen"], default="lora")
    ap.add_argument("--enc-layers", type=int, default=2, help="transformer layers before pooling (0 = pooling only)")
    ap.add_argument("--init-from", default="", help="warm start from a finished arm's ts_encoders file")
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA (R3-1's value)")
    ap.add_argument("--lr-enc", type=float, default=3e-4, help="the slot encoder (new parameters)")
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--val-windows", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bf16", choices=["auto", "on", "off"], default="auto", help="auto = bf16 autocast on CUDA")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run-name", default="tb")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-root", default=str(ROOT / "encoder_probe" / "results"))
    ap.add_argument("--cache-root", default=str(ROOT / "encoder_probe" / "cache"))
    ap.add_argument("--stop-after", type=int, default=0, help="stop (after a checkpoint) at this step: tests")
    ap.add_argument("--tiny", action="store_true", help="random-init tiny Qwen2.5-VL + byte tokenizer (CPU tests)")
    ap.add_argument("--smoke", action="store_true", help="20 steps, small eval: timing and memory check")
    a = ap.parse_args(argv)
    if a.smoke:
        a.steps, a.eval_windows, a.val_windows, a.eval_every, a.ckpt_every = 20, 8, 4, 10, 0
        a.run_name = a.run_name + "_smoke"
    if a.seq_tokens % a.ratio:
        ap.error(f"--seq-tokens {a.seq_tokens} must be a multiple of --ratio {a.ratio}")
    if a.decoder == "frozen":
        a.lora_r = 0
    return a


# ------------------------------------------------------------------------------------------------- data

def shuffled_ids(tok, ids: np.ndarray, L: int, rng) -> np.ndarray:
    """The window's words in random order, re-tokenised and cut to L tokens (``ids`` is longer than L so the
    shuffled text always fills the window)."""
    words = tok.decode(ids.tolist()).split()
    out = tok.encode(" ".join(words[j] for j in rng.permutation(len(words))))
    while len(out) < L:                       # only if the re-tokenised text came out much shorter
        out = out + out
    return np.asarray(out[:L], dtype=np.int64)


def src_len(L: int, text: str) -> int:
    return L if text == "natural" else int(L * 1.5) + 8


def windows(tok, stream: np.ndarray, L: int, n: int, phase: float, text: str, seed: int) -> np.ndarray:
    w = S0.fixed_windows(stream, src_len(L, text), n, phase)
    if text == "natural":
        return w.astype(np.int64)
    return np.stack([shuffled_ids(tok, w[i], L, np.random.default_rng([seed, int(phase * 1000), i]))
                     for i in range(n)])


# ------------------------------------------------------------------------------------------------ model

def sinusoid(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : d // 2]
    return pe


class LocalSlots(nn.Module):
    """slot i = pool([e_{i,1} .. e_{i,r}]) * gain; pool starts as the mean. The optional mixer (a small pre-norm
    transformer over the whole window) adds to the embeddings through a zero-initialised projection, so the
    initial slots are exactly the pooled token embeddings."""

    def __init__(self, r: int, d: int, layers: int, max_len: int):
        super().__init__()
        self.r, self.d = r, d
        self.pool = nn.Linear(r * d, d)
        with torch.no_grad():
            self.pool.weight.copy_(torch.cat([torch.eye(d) / r] * r, 1))
            self.pool.bias.zero_()
        self.gain = nn.Parameter(torch.ones(()))
        self.mix = None
        if layers:
            heads = 8 if d % 8 == 0 and d >= 256 else 4
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.norm_in = nn.LayerNorm(d)
            self.mix = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
            self.out = nn.Linear(d, d)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
            self.register_buffer("pe", sinusoid(max_len, d), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, L, d) token embeddings -> (B, L / r, d)."""
        B, L, d = x.shape
        if self.mix is not None:
            x = x + self.out(self.mix(self.norm_in(x) + self.pe[:L].to(x.dtype)))
        return self.pool(x.reshape(B, L // self.r, self.r * d)) * self.gain

    def load_from(self, sd: dict, r_old: int) -> None:
        """Warm start from ratio r_old (r = k * r_old): the new slot starts as the mean of k adjacent old slots."""
        if self.r % r_old:
            raise ValueError(f"cannot warm-start ratio {self.r} from ratio {r_old}")
        k = self.r // r_old
        sd = dict(sd)
        sd["pool.weight"] = sd["pool.weight"].repeat(1, k) / k
        missing, extra = self.load_state_dict(sd, strict=False)
        if missing or extra:
            raise KeyError(f"encoder mismatch: missing {missing[:3]} extra {extra[:3]}")


class Reader(S0.Compressor):
    """The LLM reads [slots][prompt] and rebuilds the text, with its LoRA ON (``dec_off`` False). The encoder
    pass uses only the frozen embedding table and the slot encoder."""

    dec_off = False

    def encode(self, ids: torch.Tensor, enc: LocalSlots) -> torch.Tensor:
        with self.ac():
            return enc(self.emb(ids).float()).float()

    def empty(self, B: int) -> torch.Tensor:
        return torch.zeros(B, 0, self.emb.weight.shape[1], device=self.dev)

    def window_losses(self, ids: torch.Tensor, mem: torch.Tensor, lora_off: bool = False) -> list[float]:
        """Mean CE per window (no grad), with the adapter on (reader) or off (the unmodified LLM: the prior)."""
        self.dec_off = lora_off
        try:
            return [self.recon_loss(ids[i:i + 1], mem[i:i + 1])[0].item() for i in range(len(ids))]
        finally:
            self.dec_off = False


def build(a, dev):
    """-> (model or peft model, tokenizer, AdapterSwitch). ``--decoder frozen``: no LoRA at all."""
    if a.lora_r > 0:
        return S0.build(a, dev)
    torch.manual_seed(a.seed)
    model, tok = Q.load_model(dev, tiny=a.tiny, seed=a.seed)
    for p in model.parameters():
        p.requires_grad_(False)
    sw = S0.AdapterSwitch(model)
    if a.grad_ckpt:
        Q.hf_model(model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tok, sw


def final_path(cache_root, run: str, ratio: int, text: str, decoder: str) -> Path:
    return Path(cache_root) / "ts_encoders" / f"{run}__r{ratio}__{text}__{decoder}.pt"


def main(argv=None) -> int:
    a = parse(argv)
    dev = resolve_device(a.device)
    cuda = dev.type == "cuda"
    autocast = a.bf16 == "on" or (a.bf16 == "auto" and cuda)
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    out = Path(a.out_root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    rows_f = out / "runs.jsonl"
    key = {"phase": "slots", "ratio": a.ratio, "steps": a.steps, "seq_tokens": a.seq_tokens, "text": a.text,
           "decoder": a.decoder, "enc_layers": a.enc_layers, "init_from": Path(a.init_from).name}
    if rows_f.exists():
        for line in rows_f.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if all(r.get(k) == v for k, v in key.items()):
                print(f"[ts] skip, done: {key}")
                return 0
    cfg = {k: v for k, v in vars(a).items() if k not in ("out_root", "cache_root", "data_dir", "stop_after")}
    fair_hash = Q.config_hash({k: cfg[k] for k in FAIR_KEYS})
    L, m = a.seq_tokens, a.seq_tokens // a.ratio

    pm, tok, sw = build(a, dev)
    tcache = Path(a.cache_root) / "s0_tokens"
    lim = 300_000 if a.tiny else 0
    tr = S0.token_stream(tok, Path(a.data_dir) / "train.txt", tcache, lim)
    te = S0.token_stream(tok, Path(a.data_dir) / "test.txt", tcache, lim)
    val = torch.from_numpy(windows(tok, te, L, a.val_windows, 0.37, a.text, a.seed)).to(dev)
    evs = {t: torch.from_numpy(windows(tok, te, L, a.eval_windows, 0.0, t, a.seed)).to(dev)
           for t in ("natural", "shuffled")}
    rd = Reader(pm, tok, m, dev, autocast, sw)
    d = rd.emb.weight.shape[1]
    torch.manual_seed(a.seed)
    enc = LocalSlots(a.ratio, d, a.enc_layers, L).to(dev)
    prev = {"samples": 0, "chain": []}
    with torch.no_grad():
        ref_rms = float(rd.emb(val).float().pow(2).mean().sqrt())
        if a.init_from:
            st = torch.load(a.init_from, map_location="cpu", weights_only=False)
            if st["config"]["decoder"] != a.decoder or st["config"]["enc_layers"] != a.enc_layers:
                raise ValueError("--init-from arm has another --decoder or --enc-layers")
            enc.load_from(st["enc"], st["ratio"])
            if a.lora_r > 0:
                Q.load_trainable_state(pm, st["lora"])
            prev = {"samples": st["samples"], "chain": st["chain"]}
            ref_rms = st["slot_rms"]
        if a.ratio > 1:        # keep the slots at the scale the reader last saw (r = 1: exactly the embeddings)
            enc.gain.mul_(ref_rms / float(rd.encode(val, enc).pow(2).mean().sqrt()))
    lora_params = [p for p in pm.parameters() if p.requires_grad]
    params = lora_params + list(enc.parameters())
    opt = torch.optim.AdamW([{"params": lora_params, "lr": a.lr}, {"params": list(enc.parameters()), "lr": a.lr_enc}],
                            weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, S0.lr_lambda(max(1, a.steps), a.warmup))
    rng = np.random.default_rng(a.seed)
    n_lora, n_enc = sum(p.numel() for p in lora_params), sum(p.numel() for p in enc.parameters())
    print(f"[ts] ratio {a.ratio} {a.text} decoder={a.decoder}: L={L} slots={m} train_tokens={len(tr)} "
          f"device={dev.type} bf16={autocast} lora={n_lora} enc={n_enc} warm={Path(a.init_from).name or '-'} "
          f"fair_hash={fair_hash}", flush=True)

    def encode_all(x):
        return torch.cat([rd.encode(x[i:i + a.gen_batch], enc) for i in range(0, len(x), a.gen_batch)])

    def val_loss():
        pm.eval()
        enc.eval()
        with torch.no_grad():
            mems = encode_all(val)
            res = [float(np.mean(rd.window_losses(val, mm)))
                   for mm in (mems, mems[torch.from_numpy(S0.blind_perm(len(val))).to(dev)])]
        pm.train()
        enc.train()
        return res

    ck = Path(a.cache_root) / "ckpt" / f"ts__{a.run_name}__r{a.ratio}__{a.text}__{a.decoder}__{a.steps}.pt"
    curve, start, t_prev, ema = [], 1, 0.0, None
    if a.ckpt_every and ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        for p, v in zip(params, st["params"]):
            p.data.copy_(v.to(p.device, p.dtype))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        if cuda and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])
        curve, start, t_prev, ema = st["curve"], st["step"] + 1, st["t_train"], st["ema"]
        print(f"[ts] resumed from step {st['step']}", flush=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    pm.train()
    enc.train()
    t_train = time.time() - t_prev
    S = src_len(L, a.text)
    for step in range(start, a.steps + 1):
        starts = rng.integers(0, len(tr) - S, a.batch_size)
        if a.text == "natural":
            batch = np.stack([tr[s:s + L] for s in starts]).astype(np.int64)
        else:
            batch = np.stack([shuffled_ids(tok, tr[s:s + S], L, rng) for s in starts])
        ids = torch.from_numpy(batch).to(dev)
        opt.zero_grad(set_to_none=True)
        loss, _ = rd.recon_loss(ids, rd.encode(ids, enc))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        sched.step()
        lv = loss.item()
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if step % a.log_every == 0 or step == 1:
            print(f"[ts] step {step}/{a.steps} loss {lv:.4f} ema {ema:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t_train) / step:.2f}s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            vl, vb = val_loss()
            curve.append((step, round(ema, 5), round(vl, 5), round(vb, 5)))
            print(f"[ts] step {step} val_loss {vl:.4f} blind {vb:.4f}", flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            torch.save({"params": [p.detach().cpu() for p in params], "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if cuda else None,
                        "curve": curve, "ema": ema, "step": step, "t_train": time.time() - t_train}, tmp)
            tmp.replace(ck)
        if a.stop_after and step >= a.stop_after:
            print(f"[ts] stop after step {step} (--stop-after)", flush=True)
            return 0
    train_s = time.time() - t_train
    peak = torch.cuda.max_memory_allocated() if cuda else None

    # ----------------------------------------------------- eval: loss, greedy CER, blind control, the LLM's prior
    t_ev = time.time()
    pm.eval()
    enc.eval()
    drop = ("losses", "samples", "cer")
    evals, slot_rms = {}, None
    with torch.no_grad():
        for text, ev in evs.items():
            mems = encode_all(ev)
            if text == a.text:
                slot_rms = float(mems.pow(2).mean().sqrt())
            perm = torch.from_numpy(S0.blind_perm(len(ev))).to(dev)
            refs = [tok.decode(r.tolist()) for r in ev]
            res = {}
            for arm, mm in (("real", mems), ("blind", mems[perm])):
                hyps = []
                for i in range(0, len(ev), a.gen_batch):
                    hyps += [tok.decode(h) for h in rd.greedy(mm[i:i + a.gen_batch], L + 16)]
                pairs = list(zip(refs, hyps))
                res[arm] = {"losses": rd.window_losses(ev, mm), **S0.edit_stats(pairs),
                            "samples": [[r[:300], h[:300]] for r, h in pairs[:4]]}
            prior = rd.window_losses(ev, rd.empty(len(ev)), lora_off=True)
            real, blind = np.asarray(res["real"]["losses"]), np.asarray(res["blind"]["losses"])
            ref = np.minimum(blind, np.asarray(prior))
            n_chars = sum(len(r) for r in refs)
            cps = n_chars / (len(ev) * m)
            frac = float(1 - real.mean() / ref.mean())
            evals[text] = {
                "chars_per_slot": cps, "chars_per_text_token": n_chars / (len(ev) * L),
                "cer": res["real"]["cer"], "loss": float(real.mean()),
                "ops": {k: v for k, v in res["real"].items() if k not in drop},
                "blind_cer": res["blind"]["cer"], "blind_loss": float(blind.mean()),
                "blind_ops": {k: v for k, v in res["blind"].items() if k not in drop},
                "prior_loss": float(np.mean(prior)), "ref_loss": float(ref.mean()), "recovered_frac": frac,
                "recovered_chars_per_slot": frac * cps,
                "bits_per_slot": float((ref - real).mean() * (L + 1) / m / math.log(2)),
                "window_losses": {"real": real.tolist(), "blind": blind.tolist(), "prior": list(prior)},
                "n_eval_windows": len(ev), "n_eval_chars": n_chars,
                "samples": res["real"]["samples"], "blind_samples": res["blind"]["samples"]}
            e = evals[text]
            print(f"[ts] TEST ratio {a.ratio} trained on {a.text}, read {text} ({cps:.1f} chars/slot): cer "
                  f"{e['cer']:.4f} (blind {e['blind_cer']:.4f}) loss {e['loss']:.4f} (blind {e['blind_loss']:.4f}, "
                  f"prior {e['prior_loss']:.4f}) bits/slot {e['bits_per_slot']:.1f} recovered {frac:.3f}", flush=True)
    samples = prev["samples"] + a.steps * a.batch_size
    chain = prev["chain"] + [[a.ratio, a.steps, a.batch_size]]
    row = {**key, "mem_slots": m, "eval": evals,
           "kv_bytes_per_1M_chars": Q.kv_bytes_analytic(pm, 1, bytes_per=2) * 1e6 / evals["natural"]["chars_per_slot"],
           "cum_samples": samples, "chain": chain, "curve": curve,
           "s_per_step": train_s / max(1, a.steps - start + 1), "train_time_s": train_s,
           "eval_time_s": time.time() - t_ev, "peak_gpu_bytes": peak, "device": str(dev), "bf16": autocast,
           "lora_params": n_lora, "enc_params": n_enc, "tokenizer": tok.name, "config": cfg,
           "fair_hash": fair_hash, "tiny": a.tiny, "smoke": a.smoke}
    with open(rows_f, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[ts] DONE ratio {a.ratio}: {row['s_per_step']:.2f}s/step peak {(peak or 0) / 2**30:.1f} GiB, "
          f"{samples} training samples so far", flush=True)
    if not a.smoke:
        f = final_path(a.cache_root, a.run_name, a.ratio, a.text, a.decoder)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        torch.save({"enc": {k: v.detach().cpu() for k, v in enc.state_dict().items()},
                    "lora": Q.trainable_state(pm) if a.lora_r > 0 else {}, "ratio": a.ratio, "slot_rms": slot_rms,
                    "samples": samples, "chain": chain, "config": cfg}, tmp)
        tmp.replace(f)
    if a.ckpt_every and ck.exists():
        ck.unlink()   # finished: free the disk
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
