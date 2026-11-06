"""Round 3 (R3-1): LoRA fine-tune Qwen2.5-VL-3B-Instruct to read one page encoding; eval transcription CER + cloze.

    python finetune_qwen.py --encoding latin_p8 --smoke                      # 500 steps, the loop check
    python finetune_qwen.py --encoding col_b4 --steps 5000 --run-name round3_ft

Plan: encoder_probe/ROUND3.md (R3-1). Runbook (RTX 3090): docs/LOCAL_GPU.md, "Round 3 on the 3090".

* Model (ep/qwen_ft.py): LoRA r=16 (alpha 32, dropout 0.05) on every LLM attention + MLP projection, the patch
  merger and the last 4 vision blocks trainable (fp32), everything else frozen; bf16 weights + bf16 autocast on
  CUDA, gradient checkpointing. --tiny swaps in a random-init shrunken model + byte tokenizer (CPU tests only).
* Data: single 448-px pages from the round-2 renderers (``split_pages`` = get_encoding_s + the optional page
  cache, EP_PAGE_CACHE=1), any registered --encoding (latin_p*, col_b*, mono_p*, rgb3_p*, lg_*, rgb3_lg_*, ...).
  Train pages from WikiText-103 train, test pages from the held-out test (+val) text, as rounds 1-2.
* Tasks per sample (ep/r3_tasks.py): 60% transcription (whole page, or a window of whole lines with at most
  --max-target-chars target chars, so dense pages stay short sequences), 40% questions (cloze / multiple choice /
  copy a span). Answers are the decoded plain text (for codes the original characters, not class labels). Loss on
  answer tokens only.
* Eval on held-out test pages, per --eval-conds (clean, jpeg75; encoders.degrade): greedy generation of
  --eval-windows evenly spaced line windows per page -> CER (edit distance / ref chars, per-line clean_hyp as
  rounds 1-2), plus cloze and multiple-choice accuracy. Val loss on fixed test samples every --eval-every steps.
* Rows: encoder_probe/results/<run>/runs.jsonl, one per eval cond, round-2 conventions (chars_per_token,
  px_per_char, budget, device, ...) plus the full config, ``fair`` (the settings that must be identical across
  arms: steps, data, LoRA config, unfrozen layers, task mix, ...) and its hash ``fair_hash``. Skip key:
  (encoding, cond, steps). The final trainable weights go to encoder_probe/cache/r3_adapters/ (for
  longctx_bench.py --adapter).
* Resume: --ckpt-every N saves trainables + optimizer + scheduler + numpy/torch/CUDA RNG + curve; a restarted job
  continues exactly (tested on CPU) and the checkpoint is deleted once the rows are written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from encoder_probe.ep import qwen_ft as Q  # noqa: E402
from encoder_probe.ep import r3_tasks as T  # noqa: E402
from encoder_probe.ep.encoders import degrade, resolve_device  # noqa: E402
from encoder_probe.ep.geom import pages_digest  # noqa: E402
from encoder_probe.ep.token_cache import split_pages  # noqa: E402

S = 448                     # Qwen-native page: 448 = 16 x 28 -> 256 image tokens, no resizing (ROUND2.md)
FAIR_KEYS = ("steps", "batch_size", "lr", "lr_vision", "warmup", "weight_decay", "clip", "lora_r", "lora_alpha",
             "lora_dropout", "vision_blocks", "train_merger", "train_pages", "test_pages", "task_mix",
             "max_target_chars", "train_conds", "seed", "eval_windows", "eval_cloze", "eval_mc", "max_new_tokens",
             "val_samples",
             "grad_ckpt", "tiny")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true", help="500 steps, 100 train / 4 test pages, run name *_smoke")
    ap.add_argument("--batch-size", type=int, default=4, help="samples (pages) per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=0, help="samples per forward (0 = batch size); for VRAM")
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    ap.add_argument("--lr-vision", type=float, default=2e-5, help="merger + unfrozen vision blocks")
    ap.add_argument("--warmup", type=float, default=0.05, help="linear warm-up fraction, then cosine to 0")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--vision-blocks", type=int, default=4, help="last N vision blocks unfrozen")
    ap.add_argument("--no-train-merger", dest="train_merger", action="store_false")
    ap.add_argument("--train-pages", type=int, default=1000)
    ap.add_argument("--test-pages", type=int, default=16)
    ap.add_argument("--task-mix", type=float, default=0.6, help="share of transcription samples")
    ap.add_argument("--max-target-chars", type=int, default=2000,
                    help="transcription windows hold at most this many target chars (same for every arm)")
    ap.add_argument("--train-conds", nargs="+", default=["clean"], help="degradations sampled per train sample")
    ap.add_argument("--eval-conds", nargs="+", default=["clean", "jpeg75"])
    ap.add_argument("--eval-windows", type=int, default=4, help="transcription windows per test page")
    ap.add_argument("--eval-cloze", type=int, default=4, help="cloze questions per test page")
    ap.add_argument("--eval-mc", type=int, default=2, help="multiple-choice questions per test page")
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="generation cap (0 = max_target_chars // 2 + 32); per batch also <= 1.5 x ref tokens + 16")
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--val-samples", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=250, help="resume checkpoint interval (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind-images", action="store_true",
                    help="control: every page's image is replaced by another page's (seeded derangement, train and "
                         "test separately; texts and targets stay with the true page). Same settings and fair_hash as "
                         "the matched arm; run name gets _blind. Its CER is the floor from priors / label statistics")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto = cuda if available")
    ap.add_argument("--bf16", choices=["auto", "on", "off"], default="auto", help="bf16 autocast (auto = CUDA)")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--run-name", default="round3_ft")
    ap.add_argument("--tiny", action="store_true", help="random-init tiny model + byte tokenizer (tests)")
    ap.add_argument("--stop-after", type=int, default=0, help="exit after this step without eval (resume tests)")
    ap.add_argument("--out-root", default=str(ROOT / "encoder_probe" / "results"))
    ap.add_argument("--cache-root", default=str(ROOT / "encoder_probe" / "cache"))
    a = ap.parse_args(argv)
    if a.smoke:
        a.steps = 500
        a.train_pages, a.test_pages = min(a.train_pages, 100), min(a.test_pages, 4)
        a.eval_windows, a.eval_cloze, a.eval_mc = min(a.eval_windows, 2), min(a.eval_cloze, 2), min(a.eval_mc, 2)
        if not a.run_name.endswith("_smoke"):
            a.run_name += "_smoke"
    if a.blind_images and not a.run_name.endswith("_blind"):
        a.run_name += "_blind"   # own rows file, checkpoint and adapter: never mixes with the matched arm
    if not a.max_new_tokens:
        a.max_new_tokens = a.max_target_chars // 2 + 32
    return a


def lr_lambda(steps: int, warmup: float):
    import math
    w = max(1, int(round(warmup * steps)))

    def f(i):   # i = optimizer steps taken so far
        if i < w:
            return (i + 1) / w
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (i - w) / max(1, steps - w))))
    return f


def run_eval(pm, tok, enc, te, flats_te, cond, a, dev, autocast, log=print):
    """-> dict with cer, accuracies and samples for one degradation."""
    imgs = [degrade(p.img, cond) for p in te]
    trans, qs = [], []
    for k, pg in enumerate(te):
        r = np.random.default_rng([a.seed, 1000, k])
        others = flats_te[:k] + flats_te[k + 1:]
        if getattr(pg, "streams", None):   # colour-gist pages: windows and questions per stream
            for st, lines in enumerate(pg.streams):
                for (lo, hi) in T.eval_windows(pg, a.max_target_chars, a.eval_windows, lines):
                    trans.append({"imgs": [imgs[k]], "ref": "\n".join(lines[lo:hi]), "page": k, "stream": st,
                                  "prompt": T.stream_transcribe_prompt(pg.colours[st], lo, hi, len(lines))})
                for kind, n in (("cloze", a.eval_cloze), ("mc", a.eval_mc)):
                    for _ in range(n):
                        s = (T.stream_cloze_sample(pg, st, r) if kind == "cloze"
                             else T.stream_mc_sample(pg, st, others, r))
                        if s is not None:
                            qs.append({"imgs": [imgs[k]], "prompt": s[0], "ref": s[1], "task": kind, "page": k,
                                       "stream": st, "lure": s[2].get("lure")})
            continue
        for (lo, hi) in T.eval_windows(pg, a.max_target_chars, a.eval_windows):
            trans.append({"imgs": [imgs[k]], "prompt": T.transcribe_prompt(lo, hi, len(pg.refs)),
                          "ref": T.transcript(pg, lo, hi), "page": k, "lines": (lo, hi)})
        for kind, n in (("cloze", a.eval_cloze), ("mc", a.eval_mc)):
            for _ in range(n):
                s = T.cloze_sample(flats_te[k], r) if kind == "cloze" else T.mc_sample(flats_te[k], others, r)
                if s is not None:
                    qs.append({"imgs": [imgs[k]], "prompt": s[0], "ref": s[1], "task": kind, "page": k})
    pm.eval()
    errs = chars = 0
    samples, pairs = [], []
    per = {}   # stream -> counters (colour-gist pages only)

    def ps(st):
        return per.setdefault(st, {"errs": 0, "chars": 0, "mc": [], "lure": [], "cloze": []})
    for i in range(0, len(trans), a.gen_batch):
        b = trans[i:i + a.gen_batch]
        cap = min(a.max_new_tokens, int(1.5 * max(len(tok.encode(s["ref"])) for s in b)) + 16)
        hyps = Q.generate(pm, tok, b, dev, cap, autocast)
        for s, h in zip(b, hyps):
            h2 = T.clean_transcript(enc, h)
            e = Levenshtein.distance(s["ref"], h2)
            errs += e
            chars += len(s["ref"])
            if "stream" in s:
                ps(s["stream"])["errs"] += e
                ps(s["stream"])["chars"] += len(s["ref"])
            pairs.append((s["ref"], h2))
            if len(samples) < 4:
                samples.append((s["ref"][:300], h2[:300]))
    acc = {"cloze": [], "mc": []}
    qsamples = []
    for i in range(0, len(qs), a.gen_batch):
        b = qs[i:i + a.gen_batch]
        for s, h in zip(b, Q.generate(pm, tok, b, dev, 12, autocast)):
            ok = T.answer_correct(s["task"], s["ref"], h)
            acc[s["task"]].append(ok)
            if "stream" in s:
                ps(s["stream"])[s["task"]].append(ok)
                if s["task"] == "mc" and s.get("lure"):
                    ps(s["stream"])["lure"].append(T.answer_correct("mc", s["lure"], h))
            if len(qsamples) < 4:
                qsamples.append((s["prompt"][:120], s["ref"], h[:60]))
    pm.train()
    from encoder_probe.ep.g2lite import edit_ops
    ops = edit_ops(pairs)   # CER split: sub / ins / del (fractions of ref chars) and hyp/ref length ratio
    # per-window "exact prefix": share of ref chars before the first error (read-then-drift diagnostic)
    pref = [next((i for i, (x, y) in enumerate(zip(r, h)) if x != y), min(len(r), len(h))) for r, h in pairs]
    ops["prefix_frac"] = sum(pref) / max(1, chars)
    out = {"cer": errs / max(1, chars), "n_eval_chars": chars, "n_windows": len(trans), "ops": ops,
           "cloze_acc": float(np.mean(acc["cloze"])) if acc["cloze"] else None, "n_cloze": len(acc["cloze"]),
           "mc_acc": float(np.mean(acc["mc"])) if acc["mc"] else None, "n_mc": len(acc["mc"]),
           "samples": samples, "q_samples": qsamples}
    if per:
        out["per_stream"] = [{"colour": te[0].colours[st], "cer": c["errs"] / max(1, c["chars"]),
                              "n_chars": c["chars"], "mc_acc": float(np.mean(c["mc"])) if c["mc"] else None,
                              "n_mc": len(c["mc"]), "lure_rate": float(np.mean(c["lure"])) if c["lure"] else None,
                              "cloze_acc": float(np.mean(c["cloze"])) if c["cloze"] else None,
                              "n_cloze": len(c["cloze"])} for st, c in sorted(per.items())]
    log(f"[ft] eval {cond}: cer {out['cer']:.4f} ({chars} chars, {len(trans)} windows) cloze {out['cloze_acc']} "
        f"mc {out['mc_acc']}", flush=True)
    return out


def main(argv=None) -> int:
    a = parse(argv)
    dev = resolve_device(a.device)
    cuda = dev.type == "cuda"
    autocast = a.bf16 == "on" or (a.bf16 == "auto" and cuda)
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    out = Path(a.out_root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "runs.jsonl"
    done = set()
    if rows.exists():
        for l in rows.read_text().splitlines():
            r = json.loads(l) if l.strip() else {}
            if r.get("phase") == "finetune" and r.get("encoding") == a.encoding and r.get("steps") == a.steps:
                done.add(r.get("cond"))
    if set(a.eval_conds) <= done:
        print(f"[ft] skip, done: {a.encoding} steps={a.steps} conds={a.eval_conds}")
        return 0
    cfg = {k: v for k, v in vars(a).items() if k not in ("out_root", "cache_root", "stop_after")}
    fair = {k: cfg[k] for k in FAIR_KEYS}
    fair_hash = Q.config_hash(fair)

    t0 = time.time()
    enc, tr, te = split_pages(a.encoding, S, a.train_pages, a.test_pages)
    flats_tr = [T.flat_text(enc, p) for p in tr]
    flats_te = [T.flat_text(enc, p) for p in te]
    blind = None
    digest_true = pages_digest(te)   # of the true test pages (the blind control reports this one too)
    if a.blind_images:   # break image-text alignment; every path (train, val, eval) reads p.img
        brng = np.random.default_rng([a.seed, 7919])
        perms = {}
        for name, pages in (("train", tr), ("test", te)):
            n = len(pages)
            while True:
                perm = brng.permutation(n)
                if n < 2 or not np.any(perm == np.arange(n)):
                    break
            imgs = [pages[j].img for j in perm]
            for pg, im in zip(pages, imgs):
                pg.img = im
            perms[name] = perm.tolist()
        blind = {"seed": [a.seed, 7919], "perm_test": perms["test"], "perm_train_head": perms["train"][:20]}
    t_pages = time.time() - t0
    torch.manual_seed(a.seed)   # LoRA init draws from the global RNG: identical across arms and restarts
    model, tok = Q.load_model(dev, tiny=a.tiny, seed=a.seed, bf16_cpu=autocast and not cuda)
    pm = Q.make_trainable(model, a.lora_r, a.lora_alpha, a.lora_dropout, a.vision_blocks, a.train_merger)
    if a.grad_ckpt:
        Q.hf_model(pm).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm.train()
    counts = Q.param_counts(pm)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    lora_p = [p for n, p in pm.named_parameters() if p.requires_grad and "lora_" in n]
    vis_p = [p for n, p in pm.named_parameters() if p.requires_grad and "lora_" not in n]
    groups = [{"params": lora_p, "lr": a.lr}] + ([{"params": vis_p, "lr": a.lr_vision}] if vis_p else [])
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.steps, a.warmup))
    chars_te = sum(p.n_chars for p in te)
    ppc = S * S * len(te) / max(1, chars_te)
    cpt = chars_te / (len(te) * Q.image_tokens(S))
    print(f"[ft] {a.encoding} S={S} pitch={enc.pitch} lines/page={enc.n_lines} train={len(tr)} test={len(te)} "
          f"px2/char={ppc:.2f} chars/token={cpt:.1f} device={dev.type} bf16={autocast} trainable={counts} "
          f"fair_hash={fair_hash} pages={t_pages:.0f}s", flush=True)

    # fixed validation samples (teacher-forced loss), never drawn from the training rng
    vr = np.random.default_rng([a.seed, 7])
    val = []
    for i in range(a.val_samples):
        k = i % len(te)
        p, ans, _ = T.page_sample(te[k], flats_te[k], flats_te, vr, a.max_target_chars, a.task_mix)
        val.append({"imgs": [te[k].img], "prompt": p, "answer": ans})

    def val_loss():
        pm.eval()
        tot = n = 0
        with torch.no_grad():
            for i in range(0, len(val), a.gen_batch):
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=autocast):
                    l, k = Q.lm_loss(pm, Q.collate(val[i:i + a.gen_batch], tok, dev))
                tot += l.item() * k
                n += k
        pm.train()
        return tot / max(1, n)

    def val_per_sample():
        """[(loss sum, target tokens)] per fixed val sample, same order in every run of an encoding (real and
        blind share the targets): paired bootstrap CIs in analyse_colour.py."""
        pm.eval()
        out = []
        with torch.no_grad():
            for v in val:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=autocast):
                    l, k = Q.lm_loss(pm, Q.collate([v], tok, dev))
                out.append((round(l.item() * k, 4), k))
        pm.train()
        return out

    tag = f"{a.run_name}__{a.encoding}__{a.steps}"
    ck = Path(a.cache_root) / "ckpt" / f"r3ft__{tag}.pt"
    curve, start, t_prev, ema = [], 1, 0.0, None
    if a.ckpt_every and ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        Q.load_trainable_state(pm, st["trainable"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        if cuda and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])
        curve, start, t_prev, ema = st["curve"], st["step"] + 1, st["t_train"], st["ema"]
        print(f"[ft] resumed from step {st['step']} ({ck.name})", flush=True)
    t_train = time.time() - t_prev
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    mb = a.micro_batch or a.batch_size
    params = lora_p + vis_p
    for step in range(start, a.steps + 1):
        pb = rng.choice(len(tr), a.batch_size, replace=len(tr) < a.batch_size)
        samples = []
        for p in pb:
            prompt, ans, _ = T.page_sample(tr[p], flats_tr[p], flats_tr, rng, a.max_target_chars, a.task_mix)
            cond = a.train_conds[int(rng.integers(len(a.train_conds)))]
            samples.append({"imgs": [degrade(tr[p].img, cond)], "prompt": prompt, "answer": ans})
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
            print(f"[ft] step {step} loss {loss_v:.4f} ema {ema:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"({(time.time() - t_train) / step:.2f} s/step)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            vl = val_loss()
            curve.append((step, round(ema, 5), round(vl, 5)))
            print(f"[ft] step {step} val_loss {vl:.4f}", flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            torch.save({"trainable": Q.trainable_state(pm), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if cuda else None, "curve": curve, "ema": ema,
                        "step": step, "t_train": time.time() - t_train}, tmp)
            tmp.replace(ck)
        if a.stop_after and step >= a.stop_after:
            print(f"[ft] stop after step {step} (--stop-after)", flush=True)
            return 0
    t_done = time.time() - t_train
    peak = torch.cuda.max_memory_allocated() if cuda else None
    vps = val_per_sample()
    adapter = Path(a.cache_root) / "r3_adapters" / f"{tag}.pt"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trainable": Q.trainable_state(pm), "encoding": a.encoding, "steps": a.steps, "config": cfg,
                "fair": fair, "fair_hash": fair_hash, "base": None if a.tiny else f"{Q.QWEN_REPO}@{Q.QWEN_REV}",
                "lora": {"r": a.lora_r, "alpha": a.lora_alpha, "dropout": a.lora_dropout,
                         "vision_blocks": a.vision_blocks, "train_merger": a.train_merger}}, adapter)
    base = {"encoding": a.encoding, "steps": a.steps, "phase": "finetune", "round": 3,
            "family": enc.family, "pitch": enc.pitch, "S": S, "P": 28, "G": S // 28, "n_tokens": Q.image_tokens(S),
            "px_per_char": ppc, "budget": round(ppc, 3), "chars_per_token": cpt, "chars_per_vlm_token": cpt,
            "score_v": 2, "bf16": autocast, "device": dev.type, "tiny": a.tiny,
            "base": None if a.tiny else f"{Q.QWEN_REPO}@{Q.QWEN_REV}", "train_pages": len(tr),
            "test_pages": len(te), "n_test_chars": chars_te, "digest_test": digest_true, "curve": curve,
            "s_per_step": t_done / a.steps, "train_time_s": t_done, "peak_gpu_bytes": peak,
            "params": counts, "lora_modules": len(Q.lora_module_names(pm)), "adapter": str(adapter),
            "config": cfg, "fair": fair, "fair_hash": fair_hash, "smoke": a.smoke or "smoke" in a.run_name,
            "blind_images": blind, "val_per_sample": vps}
    for cond in a.eval_conds:
        if cond in done:
            continue
        te0 = time.time()
        ev = run_eval(pm, tok, enc, te, flats_te, cond, a, dev, autocast)
        row = {**base, "cond": cond, "scheme": f"{enc.family}@qwen25vl_ft/{cond}", **ev,
               "eval_time_s": time.time() - te0}
        with open(rows, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[ft] TEST {a.encoding} {cond} cer={ev['cer']:.4f} cloze={ev['cloze_acc']} chars/token={cpt:.1f}",
              flush=True)
    ck.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
