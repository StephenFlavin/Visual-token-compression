"""Round 2: train one probe (encoding, frozen VLM encoder or identity ceiling, degradation) -> CER on held-out pages.

    python3 encoder_probe/experiments/run_probe2.py --encoding latin_p8 --encoder qwen25vl --cond clean
    python3 encoder_probe/experiments/run_probe2.py --print-queue          # the round-2 queue, one command per line

Encoders and page geometry: ep/vlm_encoders.py (identity28/qwen25vl: 448 px pages, 16x16 tokens of 28 px;
identity32/deepencoder: 512 px pages, 16x16 tokens of 32 px / 8x8 tokens of 64 px). VLM tokens come from the
cache (ep/token_cache.py); identity tokens are raw patches computed here. Same training recipe as round 1
(run_probe.py): AdamW lr 2e-3 OneCycle, 3000 steps, 2 pages x 16 lines per step, aux MSE x10 on train pages,
bf16 autocast, eval on 4 test pages every 500 steps and on all test pages at the end.
Rows: encoder_probe/results/<run>/runs.jsonl (existing (encoding, encoder, cond, steps) rows are skipped).

--device auto (default) uses CUDA when available, else the CPU path (unchanged). On CUDA the probe and the
test tokens live on the GPU, the train tokens/pixels too unless they exceed --gpu-data-gb (then each batch is
copied over); bf16 autocast on CUDA; the batch sampling (numpy rng) is identical. Runbook: docs/gpu/LOCAL_GPU.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "encoder_probe" / "experiments"))
from encoder_probe.ep.encoders import degrade, resolve_device  # noqa: E402
from encoder_probe.ep.geom import pages_digest  # noqa: E402
from encoder_probe.ep.probe_grid import GridProbe  # noqa: E402
from encoder_probe.ep.token_cache import get_tokens, page_cache_path, split_pages  # noqa: E402
from encoder_probe.ep.vlm_encoders import DIMS, GEOMS, cpu_has_bf16, encodings_for, geometry, patchify_s  # noqa: E402
from run_probe import evaluate  # noqa: E402  (round-1 evaluation, unchanged; works for any probe with features/read)

QUEUE_BLOCKS = [("identity28", "clean"), ("qwen25vl", "clean"), ("qwen25vl", "jpeg75"),
                ("identity32", "clean"), ("deepencoder", "clean"), ("deepencoder", "jpeg75")]
VLMS = ("qwen25vl", "deepencoder")
PAIRED = {"identity28": "qwen25vl", "identity32": "deepencoder"}   # ceiling -> the VLM sharing its page size


def print_queue(steps: int, run: str) -> None:
    print("python3 encoder_probe/experiments/build_tokens2.py --encoders qwen25vl deepencoder --conds clean jpeg75")
    for e, c in QUEUE_BLOCKS:
        for name in encodings_for(e):
            print(f"python3 encoder_probe/experiments/run_probe2.py --encoding {name} --encoder {e} --cond {c} "
                  f"--steps {steps} --run-name {run}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoding")
    ap.add_argument("--encoder", default="qwen25vl", choices=list(DIMS))
    ap.add_argument("--cond", default="clean")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--train-pages", type=int, default=200)
    ap.add_argument("--test-pages", type=int, default=16)
    ap.add_argument("--batch-pages", type=int, default=2)
    ap.add_argument("--lines-per-page", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--recon-weight", type=float, default=10.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bf16", action="store_true", help="same as --bf16 off")
    ap.add_argument("--bf16", choices=["auto", "on", "off"], default="auto",
                    help="bf16 autocast for the probe; auto = on only if the CPU has native bf16 (AMX/AVX512-BF16)")
    ap.add_argument("--run-name", default="round2")
    ap.add_argument("--print-queue", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=200, help="resume checkpoint interval (0 = off)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto = cuda if available")
    ap.add_argument("--gpu-data-gb", type=float, default=4.0,
                    help="CUDA: keep train tokens+pixels on the GPU if they fit in this many GB, else copy per batch")
    ap.add_argument("--pages-only", action="store_true",
                    help="render the pages into the page cache (encoder_probe/cache/pages/) and exit (CPU work)")
    a = ap.parse_args(argv)
    if a.print_queue:
        print_queue(a.steps, a.run_name)
        return 0
    if a.pages_only:
        S, _, _ = geometry(a.encoder)
        if page_cache_path(a.encoding, S, a.train_pages, a.test_pages).exists():
            print(f"[probe2] pages already cached: {a.encoding} S={S} tr{a.train_pages} te{a.test_pages}")
            return 0
        t0 = time.time()
        _, tr, te = split_pages(a.encoding, S, a.train_pages, a.test_pages, cache=True)
        print(f"[probe2] pages cached: {a.encoding} S={S} train={len(tr)} test={len(te)} ({time.time() - t0:.0f}s)")
        return 0
    dev = resolve_device(a.device)
    cuda = dev.type == "cuda"
    if a.bf16 == "off" or (a.bf16 == "auto" and not (torch.cuda.is_bf16_supported() if cuda else cpu_has_bf16())):
        a.no_bf16 = True
    torch.set_num_threads(a.threads)
    # flush denormals: in fp32 on CPU they made latin probe steps 5.5x slower (17 -> 3.1 s/step, 2026-11-04)
    torch.set_flush_denormal(True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = ROOT / "encoder_probe" / "results" / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    key = dict(encoding=a.encoding, encoder=a.encoder, cond=a.cond, steps=a.steps)
    rows = out / "runs.jsonl"
    if rows.exists():
        for l in rows.read_text().splitlines():
            if l.strip() and all(json.loads(l).get(k) == v for k, v in key.items()):
                print(f"[probe2] skip, done: {key}")
                return 0
    t0 = time.time()
    S, P, G = geometry(a.encoder)
    enc, tr, te = split_pages(a.encoding, S, a.train_pages, a.test_pages)
    t_pages = time.time() - t0
    waited = 0.0
    if a.encoder in VLMS:
        d, waited = get_tokens(a.encoder, a.encoding, a.cond, S, a.train_pages, a.test_pages, tr, te, device=dev)
        tok_tr, tok_te = d["train"].float(), d["test"].float()
        enc_compute = d.get("enc_compute")
        del d
    else:
        def toks(pages):
            x = np.stack([degrade(p.img, a.cond) for p in pages]).astype(np.float32) / 255.0
            return torch.from_numpy(patchify_s(x, P))
        tok_tr, tok_te = toks(tr), toks(te)
        enc_compute = None
    assert tok_tr.shape == (len(tr), G * G, DIMS[a.encoder]), tok_tr.shape
    pix_tr = torch.from_numpy(np.stack([p.img for p in tr])).permute(0, 3, 1, 2).contiguous()   # clean, uint8
    mu = tok_tr.mean((0, 1), keepdim=True)
    sd = tok_tr.std((0, 1), keepdim=True).clamp_min(1e-3)
    tok_tr, tok_te = (tok_tr - mu) / sd, (tok_te - mu) / sd
    on_gpu = False
    if cuda:   # the train set is small (5000 DeepEncoder pages: 1.6 GB tokens + 3.9 GB uint8 pixels)
        tok_te = tok_te.to(dev)
        on_gpu = (tok_tr.numel() * tok_tr.element_size() + pix_tr.numel()) / 1e9 <= a.gpu_data_gb
        if on_gpu:
            tok_tr, pix_tr = tok_tr.to(dev), pix_tr.to(dev)
        for pg in tr:   # train images now live in pix_tr: frees ~0.8 MB/page of host RAM for big --train-pages
            pg.img = None
    t_feat = time.time() - t0
    classes = enc.classes
    cidx = {c: i + 1 for i, c in enumerate(classes)}
    tgt = [[torch.tensor([cidx[x] for x in lab], dtype=torch.long) for lab in p.labels] for p in tr]
    chars = sum(p.n_chars for p in te)
    ppc = S * S * len(te) / max(1, chars)
    cpt = chars / (len(te) * G * G)
    print(f"[probe2] {key} S={S} P={P} G={G} pitch={enc.pitch} lines/page={enc.n_lines} classes={len(classes)} "
          f"test_pages={len(te)} px2/char={ppc:.2f} chars/token={cpt:.1f} pages={t_pages:.0f}s "
          f"wait={waited:.0f}s feat={t_feat:.0f}s bf16={not a.no_bf16} device={dev.type}"
          f"{' train-data-on-gpu=' + str(on_gpu) if cuda else ''}", flush=True)
    probe = GridProbe(enc.pitch, len(classes), d=DIMS[a.encoder], G=G, P=P).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=a.lr, weight_decay=1e-4, fused=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    curve, t_train, start, t_prev = [], time.time(), 1, 0.0
    # resume checkpoint: the container can restart at any time (AGENTS.md), so save training state
    # every --ckpt-every steps and continue from it; deleted once the row is written
    # non-default page counts get their own checkpoint (two --train-pages runs may share a --run-name)
    tp = "" if a.train_pages == 200 else f"__tp{a.train_pages}"
    ck = (ROOT / "encoder_probe" / "cache" / "ckpt" /
          f"{a.run_name}__{a.encoding}__{a.encoder}__{a.cond}__{a.steps}{tp}.pt")
    if a.ckpt_every and ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)   # state dicts copy onto dev's params
        probe.load_state_dict(st["probe"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
        if cuda and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])
        curve, start, t_prev = st["curve"], st["step"] + 1, st["t_train"]
        print(f"[probe2] resumed from step {st['step']} ({ck.name})", flush=True)
    t_train -= t_prev

    def take(t, ix):   # rows ix of a train tensor, on dev (a no-op copy-free index on the CPU path)
        return t[torch.as_tensor(ix, device=t.device)].to(dev, non_blocking=True)

    for step in range(start, a.steps + 1):
        pb = rng.choice(len(tr), a.batch_pages, replace=False)
        pi, li, ys = [], [], []
        for k, p in enumerate(pb):
            ok = [i for i, t in enumerate(tgt[p]) if len(t)]
            for i in rng.choice(ok, min(a.lines_per_page, len(ok)), replace=False):
                pi.append(k), li.append(int(i)), ys.append(tgt[p][i])
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=not a.no_bf16):
            feat = probe.features(take(tok_tr, pb))
            logp = probe.read(feat, torch.tensor(pi, device=dev), torch.tensor(li, device=dev))
        logp = logp.float()
        T = logp.shape[0]
        # targets on logp's device; the length tensors may stay on the CPU (ctc_loss reads them on the host)
        ctc = F.ctc_loss(logp, torch.cat(ys).to(dev), torch.full((len(ys),), T, dtype=torch.long),
                         torch.tensor([len(y) for y in ys]), blank=0, zero_infinity=True)
        rec = probe.recon_loss(feat, take(pix_tr, pb).float() / 127.5 - 1)
        loss = ctc + a.recon_weight * rec
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            c, _ = evaluate(probe, tok_te, te, enc, classes, max_pages=4, bf16=not a.no_bf16, device=dev)
            curve.append((step, round(c, 5)))
            print(f"[probe2] step {step} ctc {ctc.item():.3f} rec {rec.item():.4f} val_cer {c:.4f} "
                  f"({(time.time() - t_train) / step:.3f} s/step, {time.time() - t0:.0f}s)", flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            torch.save({"probe": probe.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if cuda else None, "curve": curve,
                        "step": step, "t_train": time.time() - t_train}, tmp)
            tmp.replace(ck)
    cer, samples = evaluate(probe, tok_te, te, enc, classes, bf16=not a.no_bf16, device=dev)
    row = {**key, "phase": "probe", "round": 2, "scheme": f"{enc.family}@{a.encoder}/{a.cond}",
           "family": enc.family, "pitch": enc.pitch, "S": S, "P": P, "G": G, "n_tokens": G * G,
           "d": DIMS[a.encoder], "px_per_char": ppc, "budget": round(ppc, 3), "chars_per_token": cpt,
           "chars_per_vlm_token": GEOMS[PAIRED.get(a.encoder, a.encoder)][1] ** 2 / ppc,
           "cer": cer, "score_v": 2, "bf16": not a.no_bf16, "device": dev.type, "enc_compute": enc_compute,
           "recon_weight": a.recon_weight,
           "train_pages": len(tr), "test_pages": len(te), "lr": a.lr, "n_test_chars": chars,
           "digest_test": pages_digest(te), "curve": curve, "wait_s": waited,
           "s_per_step": (time.time() - t_train) / a.steps, "train_time_s": time.time() - t0,
           "smoke": "smoke" in a.run_name, "samples": samples}
    print(f"[probe2] TEST {key} cer={cer:.4f} px2/char={ppc:.2f} chars/token={cpt:.1f}", flush=True)
    for r, h in samples[:2]:
        print(f"   ref: {r[:90]!r}\n   hyp: {h[:90]!r}")
    with open(rows, "a") as f:
        f.write(json.dumps(row) + "\n")
    ck.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
