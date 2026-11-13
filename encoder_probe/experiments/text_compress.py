"""Stage 0 of the own-encoder plan (docs/gpu/OWN_ENCODER.md): the text-side compression ceiling.

How many characters can one input position of Qwen2.5-VL-3B's LLM carry, if a learned encoder packs them into it?
No pixels: the encoder reads the text tokens themselves, so this bounds any pixel encoder (a rendered page carries
the same information and is harder to read). ICAE-style autoencoder:

* encoder: the LLM itself with LoRA (r=64 on every attention and MLP projection), reading
  [L text tokens][m learned memory embeddings]; its last hidden states at the m memory positions, through a
  learned projector (identity init) and gain, are the m memory slots. ratio r = L / m text tokens per slot.
* decoder: the SAME LLM with the adapter switched OFF (frozen, unmodified), reading
  [m memory slots][prompt "Repeat the text exactly:"] and reconstructing the L tokens (+ <|endoftext|>).
  Loss = next-token CE on the reconstruction; gradients reach the encoder only through the slots.
* data: contiguous L-token windows of WikiText-103 (data/train.txt from scripts/build_data.py); held-out
  windows from data/test.txt (val: 16 for the loss curve; eval: 64 for CER).
* eval: teacher-forced loss, greedy reconstruction CER (op split, exact-prefix share, exact windows), and a
  BLIND control on the same trained model: each window decoded from another window's slots (derangement), i.e.
  the frozen decoder's language prior alone. No separately trained blind run is needed: the decoder is frozen and
  the encoder only ever sees its own window.
* every ratio arm shares --seq-tokens, steps, batch, LR, LoRA, seed and eval windows (``fair_hash``).

    python3 encoder_probe/experiments/text_compress.py --ratio 16 --smoke              # GPU: s/step + peak memory
    python3 encoder_probe/experiments/text_compress.py --ratio 16 --steps 3000 --run-name stage0
    python3 encoder_probe/experiments/text_compress.py --ratio 16 --tiny --steps 30 --device cpu   # CPU test model

Rows: encoder_probe/results/<run>/runs.jsonl (phase "stage0"); checkpoints (--ckpt-every, exact resume):
encoder_probe/cache/ckpt/s0__<run>__r<ratio>__<steps>.pt; --save-final keeps the trained encoder (~0.5 GB).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from encoder_probe.ep import qwen_ft as Q  # noqa: E402
from encoder_probe.ep.encoders import resolve_device  # noqa: E402

PROMPT = "\nRepeat the text exactly:\n"
FAIR_KEYS = ["seq_tokens", "steps", "batch_size", "lr", "warmup", "weight_decay", "clip", "lora_r", "lora_alpha",
             "lora_dropout", "seed", "eval_windows", "val_windows", "tiny"]
CHARS_PER_TEXT_TOKEN = 4.33   # Qwen tokenizer on held-out WikiText (docs/claude/CONTEXT_BUDGET.md); the row measures its own


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ratio", type=int, required=True, help="text tokens per memory slot (L must divide by it)")
    ap.add_argument("--seq-tokens", type=int, default=480, help="L: text tokens per window (same for every arm)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA, memory embeddings, projector")
    ap.add_argument("--warmup", type=float, default=0.05, help="linear warm-up fraction, then cosine to 0")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--val-windows", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bf16", choices=["auto", "on", "off"], default="auto", help="auto = bf16 autocast on CUDA")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run-name", default="stage0")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-root", default=str(ROOT / "encoder_probe" / "results"))
    ap.add_argument("--cache-root", default=str(ROOT / "encoder_probe" / "cache"))
    ap.add_argument("--save-final", action="store_true", help="keep the trained encoder (LoRA + slots + projector)")
    ap.add_argument("--stop-after", type=int, default=0, help="stop (after a checkpoint) at this step: tests")
    ap.add_argument("--tiny", action="store_true", help="random-init tiny Qwen2.5-VL + byte tokenizer (CPU tests)")
    ap.add_argument("--smoke", action="store_true", help="20 steps, small eval: timing and memory check")
    a = ap.parse_args(argv)
    if a.smoke:
        a.steps, a.eval_windows, a.val_windows, a.eval_every, a.ckpt_every = 20, 8, 4, 10, 0
        a.run_name = a.run_name + "_smoke"
    if a.seq_tokens % a.ratio:
        ap.error(f"--seq-tokens {a.seq_tokens} must be a multiple of --ratio {a.ratio}")
    return a


def lr_lambda(steps: int, warmup: float):
    w = max(1, int(round(warmup * steps)))

    def f(i):
        if i < w:
            return (i + 1) / w
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (i - w) / max(1, steps - w))))
    return f


# ------------------------------------------------------------------------------------------------- data

def token_stream(tok, path: Path, cache: Path, max_chars: int = 0) -> np.ndarray:
    """All of ``path`` tokenised once (cached as .npy, keyed on tokenizer and size)."""
    tag = tok.name.replace("/", "_").replace("@", "_")
    f = cache / f"{tag}__{path.stem}__{max_chars or 'all'}.npy"
    if f.exists():
        return np.load(f)
    text = path.read_text(encoding="utf-8")
    if max_chars:
        text = text[:max_chars]
    ids = []
    for i in range(0, len(text), 1_000_000):   # chunked on line breaks so tokens never straddle a cut
        j = len(text) if i + 1_000_000 >= len(text) else text.rfind("\n", i, i + 1_000_000) + 1 or i + 1_000_000
        ids += tok.encode(text[i:j])
    arr = np.asarray(ids, dtype=np.int32)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(f, arr)
    return arr


def fixed_windows(stream: np.ndarray, L: int, n: int, phase: float) -> np.ndarray:
    """n windows of L tokens at evenly spaced offsets (phase in [0, 1) shifts them: val vs eval)."""
    span = len(stream) - L
    starts = ((np.arange(n) + phase) * span / n).astype(np.int64)
    return np.stack([stream[s:s + L] for s in starts])


# ------------------------------------------------------------------------------------------------ model

class AdapterSwitch:
    """Turns the LoRA layers off for the decoder pass. Gradient checkpointing recomputes each layer during
    backward, when the adapter state may differ: ``context_fn`` snapshots the state at forward time and restores
    it for the recompute (peft's disable_adapter() context alone makes the recomputed values mismatch)."""

    def __init__(self, pm):
        from peft.tuners.lora import LoraLayer
        self.layers = [mod for mod in pm.modules() if isinstance(mod, LoraLayer)]
        self.off = False

    def set(self, off: bool):
        if off != self.off:
            for mod in self.layers:
                mod.enable_adapters(not off)
            self.off = off

    @contextlib.contextmanager
    def state(self, off: bool):
        prev = self.off
        self.set(off)
        try:
            yield
        finally:
            self.set(prev)

    def context_fn(self):
        return contextlib.nullcontext(), self.state(self.off)


class Slots(nn.Module):
    """m learned memory embeddings (encoder input) and the projector/gain from encoder hidden states to slots."""

    def __init__(self, m: int, d: int, emb_rms: float, g: torch.Generator):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(m, d, generator=g) * emb_rms)
        self.proj = nn.Linear(d, d)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(d))
            self.proj.bias.zero_()
        self.gain = nn.Parameter(torch.ones(()))

    def out(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h.float()) * self.gain


class Compressor:
    """Encoder = LLM + LoRA over [text][memory]; decoder = the same LLM with the adapter disabled."""

    dec_off = True   # the decoder pass runs with the adapter off (text_slots.py's reader keeps it on)

    def __init__(self, pm, tok, m: int, dev, autocast: bool, switch: AdapterSwitch):
        self.pm, self.tok, self.m, self.dev, self.autocast, self.sw = pm, tok, m, dev, autocast, switch
        hf = Q.hf_model(pm)
        self.lm, self.head = hf.model.language_model, hf.lm_head
        self.emb = self.lm.get_input_embeddings()
        self.prompt = torch.tensor(tok.encode(PROMPT), dtype=torch.long, device=dev)

    def ac(self):
        return torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.autocast)

    def encode(self, ids: torch.Tensor, slots: Slots) -> torch.Tensor:
        """ids (B, L) -> memory slots (B, m, d), fp32."""
        with self.ac():
            x = self.emb(ids)
            mem = slots.mem.to(x.dtype)[None].expand(ids.shape[0], -1, -1)
            h = self.lm(inputs_embeds=torch.cat([x, mem], 1), use_cache=False).last_hidden_state[:, ids.shape[1]:]
        return slots.out(h)

    def prefix(self, mem: torch.Tensor) -> torch.Tensor:
        B = mem.shape[0]
        pe = self.emb(self.prompt)[None].expand(B, -1, -1)
        return torch.cat([mem.to(pe.dtype), pe], 1)

    def recon_loss(self, ids: torch.Tensor, mem: torch.Tensor, chunk: int = 512) -> tuple[torch.Tensor, int]:
        """Mean CE of reconstructing ids (+ eos) from the slots with the frozen decoder."""
        B, L = ids.shape
        tgt = torch.cat([ids, torch.full((B, 1), self.tok.eos, dtype=ids.dtype, device=ids.device)], 1)
        with self.sw.state(self.dec_off), self.ac():
            pre = self.prefix(mem)
            inp = torch.cat([pre, self.emb(tgt[:, :-1])], 1)
            h = self.lm(inputs_embeds=inp, use_cache=False).last_hidden_state[:, pre.shape[1] - 1:]
            hs, ys = h.reshape(-1, h.shape[-1]), tgt.reshape(-1)
            tot = h.new_zeros((), dtype=torch.float32)

            def ce(hh, yy):
                return F.cross_entropy(self.head(hh).float(), yy, reduction="sum")
            for i in range(0, len(ys), chunk):
                if torch.is_grad_enabled():
                    tot = tot + torch.utils.checkpoint.checkpoint(ce, hs[i:i + chunk], ys[i:i + chunk],
                                                                  use_reentrant=False)
                else:
                    tot = tot + ce(hs[i:i + chunk], ys[i:i + chunk])
        return tot / len(ys), int(len(ys))

    @torch.no_grad()
    def greedy(self, mem: torch.Tensor, max_new: int) -> list[list[int]]:
        """Greedy reconstruction with the frozen decoder (KV cache; every prefix has the same length)."""
        B = mem.shape[0]
        out_ids = []
        done = torch.zeros(B, dtype=torch.bool, device=mem.device)
        with self.sw.state(self.dec_off), self.ac():
            o = self.lm(inputs_embeds=self.prefix(mem), use_cache=True)
            past, h = o.past_key_values, o.last_hidden_state[:, -1]
            for _ in range(max_new):
                nxt = self.head(h).float().argmax(-1)
                out_ids.append(nxt)
                done |= nxt == self.tok.eos
                if bool(done.all()):
                    break
                o = self.lm(inputs_embeds=self.emb(nxt)[:, None], past_key_values=past, use_cache=True)
                past, h = o.past_key_values, o.last_hidden_state[:, -1]
        rows = torch.stack(out_ids, 1).tolist()
        return [r[:r.index(self.tok.eos)] if self.tok.eos in r else r for r in rows]


def build(a, dev):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(a.seed)
    model, tok = Q.load_model(dev, tiny=a.tiny, seed=a.seed)
    for p in model.parameters():
        p.requires_grad_(False)
    pm = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                          target_modules=Q.LORA_TARGETS, bias="none"))
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    sw = AdapterSwitch(pm)
    if a.grad_ckpt:
        Q.hf_model(pm).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "context_fn": sw.context_fn})
    return pm, tok, sw


def trainables(pm, slots) -> list[torch.nn.Parameter]:
    return [p for p in pm.parameters() if p.requires_grad] + list(slots.parameters())


def edit_stats(pairs) -> dict:
    from encoder_probe.ep.g2lite import edit_ops
    ops = edit_ops(pairs)
    pre, exact = [], 0
    for r, h in pairs:
        k = 0
        for x, y in zip(r, h):
            if x != y:
                break
            k += 1
        pre.append(k / max(1, len(r)))
        exact += r == h
    ops["prefix_frac"] = float(np.mean(pre)) if pre else 0.0
    ops["exact_windows"] = exact / max(1, len(pairs))
    return ops


def blind_perm(n: int) -> np.ndarray:
    """Fixed derangement: window i is decoded from window (i + 1) mod n's slots."""
    return (np.arange(n) + 1) % n


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
    if rows_f.exists():
        for line in rows_f.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("phase") == "stage0" and r.get("ratio") == a.ratio and r.get("steps") == a.steps \
                    and r.get("seq_tokens") == a.seq_tokens:
                print(f"[s0] skip, done: ratio {a.ratio} steps {a.steps}")
                return 0
    cfg = {k: v for k, v in vars(a).items() if k not in ("out_root", "cache_root", "data_dir", "stop_after")}
    fair_hash = Q.config_hash({k: cfg[k] for k in FAIR_KEYS})
    L, m = a.seq_tokens, a.seq_tokens // a.ratio

    pm, tok, sw = build(a, dev)
    tcache = Path(a.cache_root) / "s0_tokens"
    lim = 300_000 if a.tiny else 0
    tr = token_stream(tok, Path(a.data_dir) / "train.txt", tcache, lim)
    te = token_stream(tok, Path(a.data_dir) / "test.txt", tcache, lim)
    val = torch.from_numpy(fixed_windows(te, L, a.val_windows, 0.37)).long().to(dev)  # no overlap with eval
    ev = torch.from_numpy(fixed_windows(te, L, a.eval_windows, 0.0)).long().to(dev)
    comp = Compressor(pm, tok, m, dev, autocast, sw)
    with torch.no_grad():
        emb_rms = float(comp.emb.weight.float().pow(2).mean().sqrt())
    g = torch.Generator().manual_seed(a.seed)
    slots = Slots(m, comp.emb.weight.shape[1], emb_rms, g).to(dev)
    with torch.no_grad():   # match the slots' scale to token embeddings (the decoder's input space)
        h = comp.encode(val[: min(4, len(val))], slots)
        slots.gain.fill_(emb_rms / float(h.pow(2).mean().sqrt()))
    params = trainables(pm, slots)
    n_lora = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.steps, a.warmup))
    rng = np.random.default_rng(a.seed)
    print(f"[s0] ratio {a.ratio}: L={L} slots={m} train_tokens={len(tr)} test_tokens={len(te)} device={dev.type} "
          f"bf16={autocast} lora={n_lora} slots_params={sum(p.numel() for p in slots.parameters())} "
          f"fair_hash={fair_hash}", flush=True)

    def val_loss():
        pm.eval()
        with torch.no_grad():
            mems = torch.cat([comp.encode(val[i:i + a.gen_batch], slots) for i in range(0, len(val), a.gen_batch)])
            res = []
            for mm in (mems, mems[torch.from_numpy(blind_perm(len(val))).to(dev)]):
                tot = n = 0
                for i in range(0, len(val), a.gen_batch):
                    lo, k = comp.recon_loss(val[i:i + a.gen_batch], mm[i:i + a.gen_batch])
                    tot += lo.item() * k
                    n += k
                res.append(tot / n)
        pm.train()
        return res

    ck = Path(a.cache_root) / "ckpt" / f"s0__{a.run_name}__r{a.ratio}__{a.steps}.pt"
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
        print(f"[s0] resumed from step {st['step']}", flush=True)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    pm.train()
    t_train = time.time() - t_prev
    for step in range(start, a.steps + 1):
        starts = rng.integers(0, len(tr) - L, a.batch_size)
        ids = torch.from_numpy(np.stack([tr[s:s + L] for s in starts])).long().to(dev)
        opt.zero_grad(set_to_none=True)
        loss, _ = comp.recon_loss(ids, comp.encode(ids, slots))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        sched.step()
        lv = loss.item()
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if step % a.log_every == 0 or step == 1:
            el = time.time() - t_train
            print(f"[s0] step {step}/{a.steps} loss {lv:.4f} ema {ema:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{el / step:.2f}s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            vl, vb = val_loss()
            curve.append((step, round(ema, 5), round(vl, 5), round(vb, 5)))
            print(f"[s0] step {step} val_loss {vl:.4f} blind {vb:.4f}", flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            torch.save({"params": [p.detach().cpu() for p in params], "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if cuda else None,
                        "curve": curve, "ema": ema, "step": step, "t_train": time.time() - t_train}, tmp)
            tmp.replace(ck)
        if a.stop_after and step >= a.stop_after:
            print(f"[s0] stop after step {step} (--stop-after)", flush=True)
            return 0
    train_s = time.time() - t_train
    peak = torch.cuda.max_memory_allocated() if cuda else None

    # ------------------------------------------------------------------ eval: loss, greedy CER, blind control
    t_ev = time.time()
    pm.eval()
    with torch.no_grad():
        mems = torch.cat([comp.encode(ev[i:i + a.gen_batch], slots) for i in range(0, len(ev), a.gen_batch)])
        perm = torch.from_numpy(blind_perm(len(ev))).to(dev)
        refs = [tok.decode(r.tolist()) for r in ev]
        res = {}
        for arm, mm in (("real", mems), ("blind", mems[perm])):
            tot = n = 0
            hyps = []
            for i in range(0, len(ev), a.gen_batch):
                lo, k = comp.recon_loss(ev[i:i + a.gen_batch], mm[i:i + a.gen_batch])
                tot += lo.item() * k
                n += k
                hyps += [tok.decode(h) for h in comp.greedy(mm[i:i + a.gen_batch], L + 16)]
            pairs = list(zip(refs, hyps))
            res[arm] = {"loss": tot / n, **edit_stats(pairs), "samples": [[r[:300], h[:300]] for r, h in pairs[:4]]}
    n_chars = sum(len(r) for r in refs)
    cps = n_chars / (len(ev) * m)
    row = {"phase": "stage0", "ratio": a.ratio, "seq_tokens": L, "mem_slots": m, "steps": a.steps,
           "chars_per_slot": cps, "chars_per_text_token": n_chars / (len(ev) * L),
           "kv_bytes_per_1M_chars": Q.kv_bytes_analytic(pm, 1, bytes_per=2) * 1e6 / cps,
           "cer": res["real"]["cer"], "loss": res["real"]["loss"], "ops": {k: v for k, v in res["real"].items()
                                                                           if k not in ("loss", "samples", "cer")},
           "blind_cer": res["blind"]["cer"], "blind_loss": res["blind"]["loss"],
           "blind_ops": {k: v for k, v in res["blind"].items() if k not in ("loss", "samples", "cer")},
           "gain": 1 - res["real"]["loss"] / res["blind"]["loss"], "n_eval_windows": len(ev),
           "n_eval_chars": n_chars, "samples": res["real"]["samples"], "blind_samples": res["blind"]["samples"],
           "curve": curve, "s_per_step": train_s / max(1, a.steps - start + 1), "train_time_s": train_s,
           "eval_time_s": time.time() - t_ev, "peak_gpu_bytes": peak, "device": str(dev), "bf16": autocast,
           "lora_params": n_lora, "tokenizer": tok.name, "config": cfg, "fair_hash": fair_hash,
           "tiny": a.tiny, "smoke": a.smoke}
    with open(rows_f, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[s0] TEST ratio {a.ratio} ({cps:.1f} chars/slot): cer {row['cer']:.4f} (blind {row['blind_cer']:.4f}) "
          f"loss {row['loss']:.4f} (blind {row['blind_loss']:.4f}) prefix {row['ops']['prefix_frac']:.3f} "
          f"exact {row['ops']['exact_windows']:.3f} {row['s_per_step']:.2f}s/step peak "
          f"{(peak or 0) / 2**30:.1f} GiB", flush=True)
    if a.save_final:
        f = Path(a.cache_root) / "s0_encoders" / f"{a.run_name}__r{a.ratio}__{a.steps}.pt"
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": [p.detach().cpu() for p in params], "config": cfg}, f)
    if a.ckpt_every and ck.exists():
        ck.unlink()   # finished: free the disk
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
