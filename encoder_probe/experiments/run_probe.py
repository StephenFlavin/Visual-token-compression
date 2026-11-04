"""Train one probe: (encoding, frozen encoder, degradation) -> CER on held-out pages.

    python3 encoder_probe/experiments/run_probe.py --encoding col_b2 --encoder siglip --cond clean
    python3 encoder_probe/experiments/run_probe.py --print-grid   # one command per line, for the job queue

Rows go to encoder_probe/results/<run>/runs.jsonl (existing (encoding, encoder, cond, steps) rows are skipped).
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
from encoder_probe.ep.encoders import Encoder, degrade  # noqa: E402
from encoder_probe.ep.pages import H, W, get_encoding, make_pages  # noqa: E402
from encoder_probe.ep.probe import Probe, edit_distance, greedy  # noqa: E402

GRID_ENCODINGS = ["latin_p4", "latin_p5", "latin_p6", "latin_p8", "latin_p10",
                  "col_b1", "col_b2", "col_b3", "col_b4",
                  "lg_free_w23_h3", "lg_free_h4", "lg_pb_h4", "lg_pb_h5"]
GRID_RUNS = [("identity", "clean"), ("siglip", "clean"), ("siglip", "jpeg75"), ("identity", "jpeg75")]


def evaluate(probe, tok, pages, enc, classes, max_pages=None, batch_pages=2, bf16=True):
    probe.eval()
    errs = n = 0
    samples = []
    idx = list(range(len(pages) if max_pages is None else min(max_pages, len(pages))))
    with torch.no_grad():
        for s in range(0, len(idx), batch_pages):
            pb = idx[s:s + batch_pages]
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                feat = probe.features(tok[pb])
            pi, li, refs = [], [], []
            for k, p in enumerate(pb):
                for i, r in enumerate(pages[p].refs):
                    if r:
                        pi.append(k), li.append(i), refs.append(r)
            if not pi:
                continue
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                lp = probe.read(feat, torch.tensor(pi), torch.tensor(li))
            dec = greedy(lp.float())
            for d, r in zip(dec, refs):
                hyp = enc.clean_hyp(enc.decode("".join(classes[j - 1] for j in d)))
                errs += edit_distance(r, hyp)
                n += len(r)
                if len(samples) < 3:
                    samples.append((r, hyp))
    probe.train()
    return errs / max(1, n), samples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoding")
    ap.add_argument("--encoder", default="siglip")
    ap.add_argument("--cond", default="clean")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--train-pages", type=int, default=200)
    ap.add_argument("--test-pages", type=int, default=16)
    ap.add_argument("--batch-pages", type=int, default=2)
    ap.add_argument("--lines-per-page", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--recon-weight", type=float, default=10.0, help="aux MSE weight (feature ch 0-2 -> clean page)")
    ap.add_argument("--free-channels", type=int, default=0, help="gated feature channels besides the 3 pixel ones")
    ap.add_argument("--token-mlp", type=int, default=0, help="capacity control: residual token MLP blocks")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast for the probe (AMX: ~1.6x faster)")
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--print-grid", action="store_true")
    a = ap.parse_args(argv)
    if a.print_grid:
        for e, c in GRID_RUNS:
            for name in GRID_ENCODINGS:
                print(f"python3 encoder_probe/experiments/run_probe.py --encoding {name} --encoder {e} "
                      f"--cond {c} --steps {a.steps} --run-name {a.run_name}")
        return 0
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
            r = json.loads(l)
            # colour rows scored before the hypothesis-stripping fix (no score_v) are redone
            stale = a.encoding.startswith("col_") and r.get("score_v", 1) < 2
            if all(r.get(k) == v for k, v in key.items()) and not stale:
                print(f"[probe] skip, done: {key}")
                return 0
    t0 = time.time()
    enc = get_encoding(a.encoding)
    tr = make_pages(enc, "train", a.train_pages)
    te = make_pages(enc, "test", a.test_pages)
    model = Encoder(a.encoder)
    def toks(pages):
        imgs = np.stack([degrade(p.img, a.cond) for p in pages])
        return torch.from_numpy(model(imgs))
    tok_tr, tok_te = toks(tr), toks(te)
    pix_tr = torch.from_numpy(np.stack([p.img for p in tr])).permute(0, 3, 1, 2).contiguous()  # clean, uint8
    mu = tok_tr.mean((0, 1), keepdim=True)
    sd = tok_tr.std((0, 1), keepdim=True).clamp_min(1e-3)
    tok_tr, tok_te = (tok_tr - mu) / sd, (tok_te - mu) / sd
    del model
    t_feat = time.time() - t0
    classes = enc.classes
    cidx = {c: i + 1 for i, c in enumerate(classes)}
    tgt = [[torch.tensor([cidx[x] for x in lab], dtype=torch.long) for lab in p.labels] for p in tr]
    chars = sum(p.n_chars for p in te)
    ppc = H * W * len(te) / max(1, chars)
    print(f"[probe] {key} pitch={enc.pitch} lines/page={enc.n_lines} classes={len(classes)} "
          f"px2/char={ppc:.2f} chars/token={256 / ppc:.1f} feat_time={t_feat:.0f}s", flush=True)
    probe = Probe(enc.pitch, len(classes), c=3 + a.free_channels, token_mlp=a.token_mlp)
    opt = torch.optim.AdamW(probe.parameters(), lr=a.lr, weight_decay=1e-4, fused=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    curve = []
    for step in range(1, a.steps + 1):
        pb = rng.choice(len(tr), a.batch_pages, replace=False)
        pi, li, ys = [], [], []
        for k, p in enumerate(pb):
            ok = [i for i, t in enumerate(tgt[p]) if len(t)]
            for i in rng.choice(ok, min(a.lines_per_page, len(ok)), replace=False):
                pi.append(k), li.append(int(i)), ys.append(tgt[p][i])
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=not a.no_bf16):
            feat = probe.features(tok_tr[pb])
            logp = probe.read(feat, torch.tensor(pi), torch.tensor(li))
        logp = logp.float()
        T = logp.shape[0]
        ctc = F.ctc_loss(logp, torch.cat(ys), torch.full((len(ys),), T, dtype=torch.long),
                         torch.tensor([len(y) for y in ys]), blank=0, zero_infinity=True)
        rec = probe.recon_loss(feat, pix_tr[pb].float() / 127.5 - 1)
        loss = ctc + a.recon_weight * rec
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            c, _ = evaluate(probe, tok_te, te, enc, classes, max_pages=4, bf16=not a.no_bf16)
            curve.append((step, round(c, 5)))
            print(f"[probe] step {step} ctc {ctc.item():.3f} rec {rec.item():.4f} val_cer {c:.4f} ({time.time() - t0:.0f}s)", flush=True)
    cer, samples = evaluate(probe, tok_te, te, enc, classes, bf16=not a.no_bf16)
    row = {**key, "phase": "probe", "scheme": f"{enc.family}@{a.encoder}/{a.cond}", "family": enc.family,
           "pitch": enc.pitch, "px_per_char": ppc, "budget": round(ppc, 3), "chars_per_token": 256 / ppc,
           "cer": cer, "score_v": 2, "bf16": not a.no_bf16, "recon_weight": a.recon_weight, "free_channels": a.free_channels, "token_mlp": a.token_mlp, "train_pages": a.train_pages, "lr": a.lr, "n_test_chars": chars, "curve": curve, "train_time_s": time.time() - t0,
           "samples": samples}
    print(f"[probe] TEST {key} cer={cer:.4f} px2/char={ppc:.2f} chars/token={256 / ppc:.1f}", flush=True)
    for r, h in samples[:2]:
        print(f"   ref: {r[:90]!r}\n   hyp: {h[:90]!r}")
    with open(rows, "a") as f:
        f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
