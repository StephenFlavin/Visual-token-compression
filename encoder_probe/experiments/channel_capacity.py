"""PLAN_3090 amendment 6: channel capacity of Qwen2.5-VL's vision side (trained tail + merger) into the frozen LLM.

Random bits painted as colour blocks; no language prior, no alphabet, no legibility confound. Each image carries
--bits-per-image uniform random bits; the target is the same bits as octal digits (one Qwen token each, 3 bits).
Because the source is uniform and known, the information recovered has an exact reference:

    recovered bits = H - CE = 3 x n_digits - (cross-entropy of the true digits, in bits)

a lower bound on the mutual information between image and digits (Barber-Agakov), with no blind run needed. The
swapped-image check (each eval image replaced by another sample's) must give <= 0.

* Layout: cells of --cell px (2, 4 or 7: they divide the 28-px token), --levels per RGB channel (4: 0/85/170/255,
  2 bits per channel; 2: 0/255). A cell carries 3 x log2(levels) bits. Bits are ordered token by token (token rows,
  then the cells of each token row-major), so a token's digits are contiguous in the target.
* Offered bits per vision token = (28 / cell)^2 x 3 x log2(levels): cell 7 -> 96, cell 4 -> 294, cell 2 -> 1176
  with 4 levels. The image is (side x 28) px square with side^2 = bits-per-image / bits-per-token, so every arm
  scores the same 4704 bits (1568 digits) per image: 7x7 tokens (196 px), 4x4 (112 px), 2x2 (56 px). Every vision
  token the LLM sees carries scored bits, so nothing can be parked in unscored tokens.
* Training: the Stage A recipe (amendments 3-4): LLM frozen (no LoRA), vision blocks 16-31 + merger trained at
  lr 2e-5, patch embedding frozen (``--train-patch-embed`` for an ablation), AdamW, warm-up then cosine, clip 1,
  batch 8. Fresh random images every step (no train set, nothing to memorise).
* Eval: --eval-samples fixed held-out images, teacher-forced; recovered bits per vision token with a sample
  bootstrap CI, digit accuracy (argmax), the swapped-image check, and the same at step 0 (frozen model).
* Resume: --ckpt-every N (trainables, optimizer, scheduler, RNGs, curve).

    python encoder_probe/experiments/channel_capacity.py --cell 4 --smoke --device cuda
    python encoder_probe/experiments/channel_capacity.py --cell 4 --steps 6000 --device cuda --run-name r12_chan
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder_probe.ep import qwen_ft as Q  # noqa: E402
from encoder_probe.ep.encoders import resolve_device  # noqa: E402

TOKEN = 28
PROMPT = "Read the colour cells as octal digits."
FAIR_KEYS = ("steps", "batch_size", "lr_vision", "warmup", "weight_decay", "clip", "vision_blocks", "train_merger",
             "train_patch_embed", "bits_per_image", "levels", "seed")


# --------------------------------------------------------------------------------------------- the code

def geometry(cell: int, levels: int, bits_per_image: int) -> dict:
    assert TOKEN % cell == 0, "cell must divide the 28-px token"
    assert levels in (2, 4), "levels per channel: 2 or 4"
    cpt_side = TOKEN // cell
    bits_cell = 3 * int(math.log2(levels))
    bits_tok = cpt_side * cpt_side * bits_cell
    n_tok = bits_per_image / bits_tok
    side = int(round(math.sqrt(n_tok)))
    assert side * side * bits_tok == bits_per_image, f"bits per image must be a square number of tokens x {bits_tok}"
    assert bits_per_image % 3 == 0
    return {"cell": cell, "levels": levels, "bits_per_cell": bits_cell, "bits_per_token": bits_tok, "side": side,
            "n_tokens": side * side, "S": side * TOKEN, "n_digits": bits_per_image // 3,
            "bits_per_image": bits_per_image}


def make_sample(g: dict, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """-> (uint8 image (S, S, 3), octal digit string). Bits: tokens row-major, cells row-major within a token,
    per cell R, G, B levels MSB first; the bit string is read 3 bits per octal digit."""
    L, c, side, k = g["levels"], g["cell"], g["side"], TOKEN // g["cell"]
    vals = rng.integers(0, L, (side, side, k, k, 3))          # token row, token col, cell row, cell col, channel
    nb = int(math.log2(L))
    bits = ((vals[..., None] >> np.arange(nb - 1, -1, -1)) & 1).reshape(-1)
    digits = "".join(str(int(x)) for x in bits.reshape(-1, 3) @ np.array([4, 2, 1]))
    lv = np.round(np.linspace(0, 255, L)).astype(np.uint8)
    cells = lv[vals].transpose(0, 2, 1, 3, 4).reshape(side * k, side * k, 3)   # (rows of cells, cols of cells, 3)
    img = np.repeat(np.repeat(cells, c, 0), c, 1)
    return np.ascontiguousarray(img), digits


def decode_image(g: dict, img: np.ndarray) -> str:
    """Inverse of make_sample (tests): read each cell's centre, back to octal digits."""
    L, c, side, k = g["levels"], g["cell"], g["side"], TOKEN // g["cell"]
    lv = np.round(np.linspace(0, 255, L))
    cen = img[c // 2::c, c // 2::c].astype(float)              # (side*k, side*k, 3)
    vals = np.abs(cen[..., None] - lv).argmin(-1).reshape(side, k, side, k, 3).transpose(0, 2, 1, 3, 4)
    nb = int(math.log2(L))
    bits = ((vals[..., None] >> np.arange(nb - 1, -1, -1)) & 1).reshape(-1)
    return "".join(str(int(x)) for x in bits.reshape(-1, 3) @ np.array([4, 2, 1]))


# --------------------------------------------------------------------------------------------- scoring

@torch.no_grad()
def score(pm, tok, samples: list[dict], dev, autocast: bool, mb: int = 4) -> list[dict]:
    """Per sample: CE of the digit tokens in bits (the closing <|im_end|> excluded) and argmax accuracy."""
    m = Q.hf_model(pm)
    out = []
    for i in range(0, len(samples), mb):
        b = Q.collate(samples[i:i + mb], tok, dev)
        m.model.rope_deltas = None
        kw = {k: b[k] for k in ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw")}
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=autocast):
            h = m.model(**kw, use_cache=False).last_hidden_state[:, :-1]
        y = b["labels"][:, 1:]
        for j in range(y.shape[0]):
            pos = torch.nonzero(y[j] != -100).squeeze(1)[:-1]     # drop <|im_end|>
            ce_bits, hit = 0.0, 0
            for s in range(0, len(pos), 512):
                p = pos[s:s + 512]
                logits = m.lm_head(h[j, p]).float()
                ce_bits += float(F.cross_entropy(logits, y[j, p], reduction="sum")) / math.log(2)
                hit += int((logits.argmax(-1) == y[j, p]).sum())
            out.append({"n_digits": int(len(pos)), "ce_bits": ce_bits, "hits": hit})
    return out


def boot(x: np.ndarray, reps: int = 10000, seed: int = 0) -> list[float]:
    r = np.random.default_rng(seed)
    b = x[r.integers(0, len(x), (reps, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def summarise(g: dict, real: list[dict], swap: list[dict] | None) -> dict:
    def per_tok(rs):
        return np.array([(3 * r["n_digits"] - r["ce_bits"]) / g["n_tokens"] for r in rs])
    out = {"recovered_bits_per_token": boot(per_tok(real)),
           "digit_acc": float(sum(r["hits"] for r in real) / sum(r["n_digits"] for r in real)),
           "ce_bits_per_digit": float(sum(r["ce_bits"] for r in real) / sum(r["n_digits"] for r in real))}
    if swap is not None:
        out["swap_recovered_bits_per_token"] = boot(per_tok(swap))
    out["recovered_frac"] = out["recovered_bits_per_token"][0] / g["bits_per_token"]
    return out


def evaluate(pm, tok, g, ev: list[dict], dev, autocast, mb) -> dict:
    real = score(pm, tok, ev, dev, autocast, mb)
    sw = [{**s, "imgs": ev[(i + 1) % len(ev)]["imgs"]} for i, s in enumerate(ev)]
    return summarise(g, real, score(pm, tok, sw, dev, autocast, mb))


# --------------------------------------------------------------------------------------------- main

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=int, required=True, help="block size in px: 7, 4 or 2")
    ap.add_argument("--levels", type=int, default=4, help="levels per RGB channel (2 or 4)")
    ap.add_argument("--bits-per-image", type=int, default=4704)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--smoke", action="store_true", help="200 steps, 16 eval samples, run name *_smoke")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--lr-vision", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--vision-blocks", type=int, default=16)
    ap.add_argument("--no-train-merger", dest="train_merger", action="store_false")
    ap.add_argument("--train-patch-embed", action="store_true")
    ap.add_argument("--eval-samples", type=int, default=64)
    ap.add_argument("--val-samples", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--bf16", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--run-name", default="r12_chan")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--out-root", default=str(ROOT / "encoder_probe" / "results"))
    ap.add_argument("--cache-root", default=str(ROOT / "encoder_probe" / "cache"))
    a = ap.parse_args(argv)
    if a.smoke:
        a.steps, a.eval_samples = 200, min(a.eval_samples, 16)
        if not a.run_name.endswith("_smoke"):
            a.run_name += "_smoke"
    return a


def main(argv=None) -> int:
    from finetune_qwen import lr_lambda
    a = parse(argv)
    g = geometry(a.cell, a.levels, a.bits_per_image)
    dev = resolve_device(a.device)
    cuda = dev.type == "cuda"
    autocast = a.bf16 == "on" or (a.bf16 == "auto" and cuda)
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    out = Path(a.out_root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "runs.jsonl"
    key = (a.cell, a.levels, a.steps, a.train_patch_embed)
    for l in (rows.read_text().splitlines() if rows.exists() else []):
        r = json.loads(l) if l.strip() else {}
        if (r.get("cell"), r.get("levels"), r.get("steps"), r.get("config", {}).get("train_patch_embed")) == key:
            print(f"[chan] skip, done: cell {a.cell} levels {a.levels} steps {a.steps}")
            return 0
    cfg = {k: v for k, v in vars(a).items() if k not in ("out_root", "cache_root", "stop_after")}
    fair = {k: cfg[k] for k in FAIR_KEYS}
    fair_hash = Q.config_hash(fair)

    torch.manual_seed(a.seed)
    model, tok = Q.load_model(dev, tiny=a.tiny, seed=a.seed, bf16_cpu=autocast and not cuda)
    pm = Q.make_trainable(model, 0, 32, 0.0, a.vision_blocks, a.train_merger, a.train_patch_embed)
    if a.grad_ckpt:
        Q.hf_model(pm).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": a.lr_vision}], weight_decay=a.weight_decay,
                            betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.steps, a.warmup))
    counts = Q.param_counts(pm)
    mb = a.micro_batch or a.batch_size
    erng = np.random.default_rng([a.seed, 99])                 # held-out images, never the training stream
    ev = []
    for _ in range(a.eval_samples):
        img, d = make_sample(g, erng)
        ev.append({"imgs": [img], "prompt": PROMPT, "answer": d})
    val = ev[:a.val_samples]
    print(f"[chan] cell {a.cell} levels {a.levels}: {g['bits_per_token']} bits/token offered, {g['n_tokens']} "
          f"tokens ({g['S']} px), {g['n_digits']} digits/image, device={dev.type} bf16={autocast} "
          f"trainable={counts} fair_hash={fair_hash}", flush=True)

    tag = f"{a.run_name}__c{a.cell}l{a.levels}{'_pe' if a.train_patch_embed else ''}__{a.steps}"
    ck = Path(a.cache_root) / "ckpt" / f"chan__{tag}.pt"
    rng = np.random.default_rng(a.seed)
    curve, start, t_prev, ema, frozen = [], 1, 0.0, None, None
    if a.ckpt_every and ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        Q.load_trainable_state(pm, st["trainable"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        if cuda and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])
        curve, start, t_prev, ema, frozen = st["curve"], st["step"] + 1, st["t_train"], st["ema"], st["frozen"]
        print(f"[chan] resumed from step {st['step']}", flush=True)
    if frozen is None:
        pm.eval()
        frozen = evaluate(pm, tok, g, ev, dev, autocast, mb)
        print(f"[chan] step 0 (frozen): {frozen['recovered_bits_per_token'][0]:.2f} bits/token, "
              f"swap {frozen['swap_recovered_bits_per_token'][0]:.2f}", flush=True)
    pm.train()
    t_train = time.time() - t_prev
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for step in range(start, a.steps + 1):
        samples = []
        for _ in range(a.batch_size):
            img, d = make_sample(g, rng)
            samples.append({"imgs": [img], "prompt": PROMPT, "answer": d})
        batches = [Q.collate(samples[i:i + mb], tok, dev) for i in range(0, len(samples), mb)]
        n_tot = sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)
        opt.zero_grad(set_to_none=True)
        loss_v = 0.0
        for b in batches:
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=autocast):
                loss, n = Q.lm_loss(pm, b)
            (loss * (n / n_tot)).backward()
            loss_v += loss.item() * n / n_tot
        torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        sched.step()
        ema = loss_v if ema is None else 0.95 * ema + 0.05 * loss_v
        if step % a.log_every == 0 or step == 1:
            print(f"[chan] step {step} loss {loss_v:.4f} nats/digit ema {ema:.4f} (uniform {math.log(8):.4f}) "
                  f"lr {sched.get_last_lr()[0]:.2e} ({(time.time() - t_train) / step:.2f} s/step)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            pm.eval()
            v = summarise(g, score(pm, tok, val, dev, autocast, mb), None)
            pm.train()
            curve.append((step, round(ema, 5), round(v["recovered_bits_per_token"][0], 3), round(v["digit_acc"], 4)))
            print(f"[chan] step {step} val: {v['recovered_bits_per_token'][0]:.2f} bits/token of "
                  f"{g['bits_per_token']}, digit acc {v['digit_acc']:.3f}", flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            torch.save({"trainable": Q.trainable_state(pm), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if cuda else None, "curve": curve, "ema": ema,
                        "step": step, "t_train": time.time() - t_train, "frozen": frozen}, tmp)
            tmp.replace(ck)
        if a.stop_after and step >= a.stop_after:
            print(f"[chan] stop after step {step} (--stop-after)", flush=True)
            return 0
    t_done = time.time() - t_train
    peak = torch.cuda.max_memory_allocated() if cuda else None
    pm.eval()
    trained = evaluate(pm, tok, g, ev, dev, autocast, mb)
    row = {"phase": "channel_capacity", **g, "steps": a.steps, "frozen": frozen, "trained": trained, "curve": curve,
           "s_per_step": t_done / a.steps, "train_time_s": t_done, "peak_gpu_bytes": peak, "params": counts,
           "device": dev.type, "bf16": autocast, "tiny": a.tiny, "smoke": a.smoke or "smoke" in a.run_name,
           "base": None if a.tiny else f"{Q.QWEN_REPO}@{Q.QWEN_REV}", "config": cfg, "fair": fair,
           "fair_hash": fair_hash}
    with open(rows, "a") as f:
        f.write(json.dumps(row) + "\n")
    t = trained["recovered_bits_per_token"]
    s = trained["swap_recovered_bits_per_token"]
    print(f"[chan] TEST cell {a.cell}: {t[0]:.2f} ({t[1]:.2f}-{t[2]:.2f}) bits/token recovered of "
          f"{g['bits_per_token']} offered; digit acc {trained['digit_acc']:.3f}; swap {s[0]:.2f} ({s[1]:.2f}-{s[2]:.2f})",
          flush=True)
    ck.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
