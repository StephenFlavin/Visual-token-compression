"""G2-lite (ROUND3.md): does a trained Qwen2.5-VL patch merger let dense codes through? One (arm, encoding) run.

    python3 encoder_probe/experiments/run_g2lite.py --arm trained --encoding rgb3_lg_pb_h5
    python3 encoder_probe/experiments/run_g2lite.py --print-queue          # cache builds + runs, pre-registered order

Input: the frozen ViT's final-block, pre-merger hidden states (all 32 blocks, (1024, 1280) bf16 per 448-px page,
window order), cached once per encoding as ``qwen25vl_h32`` (ep/token_cache.py::build with split=32, resumable in
chunks; the queue builds them with build_tokens2.py; a run whose cache is missing waits for a builder holding the
cache lock, or builds it itself). Arms (ep/g2lite.py::Front):
  frozen   pretrained merger, frozen (its output is precomputed once: no dropout, so this equals running it per
           step) + probe                                  (R3-0 at this budget, the control)
  trained  merger initialised from pretrained, trained jointly with the probe     (G2)
  direct   no merger: the 4 concatenated patch states, 5120-d per token + probe   (upper bound of the ViT output)
Every arm: the round-2 GridProbe (G = 16, P = 28) and recipe of run_probe2.py: CTC + aux MSE x10 to the clean page,
AdamW lr 2e-3 (all trainable parameters, merger included), wd 1e-4, OneCycle (5% warm-up), grad-norm clip 5 over
all trainable parameters, 2 pages x 16 lines per step, 3000 steps, seed 0, 500 train / 16 test pages; fp32 on CPUs
without native bf16 (+ flush denormals); eval on 4 test pages every 500 steps and on all 16 at the end (``cer``).
The probe is initialised from the same seed in every arm (re-seeded after the merger is built).
Rows: encoder_probe/results/<run>/runs.jsonl keyed by (encoding, arm, steps, train_pages, seed); resume
checkpoints every --ckpt-every steps under encoder_probe/cache/ckpt/ (exact resume, tested).
--tiny: a random-init tiny Qwen2.5-VL tower (6 blocks, ep/qwen_ft.py) for the CPU tests, no downloads.
"""
from __future__ import annotations

import argparse
import copy
import gc
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
from encoder_probe.ep import flock  # noqa: E402
from encoder_probe.ep import token_cache as TC  # noqa: E402
from encoder_probe.ep.g2lite import ARMS, Front, G2Reader, load_merger, trainable_names, window_reverse  # noqa: E402
from encoder_probe.ep.geom import pages_digest  # noqa: E402
from encoder_probe.ep.probe_grid import GridProbe  # noqa: E402
from encoder_probe.ep.vlm_encoders import WEIGHTS, cpu_has_bf16  # noqa: E402
from run_probe import evaluate  # noqa: E402

S, P, G = 448, 28, 16
CODES = ["rgb3_lg_pb_h5", "latin_p8", "lg_pb_h5", "latin_p5", "latin_p6"]
# run order: the pre-registered six with the amendment's Latin sweep (ROUND3.md, 2026-11-04 ~21:50): moonshot
# trained > latin_p8 trained > latin_p5 trained > moonshot frozen > latin_p6 trained > the rest
ORDER = [("trained", "rgb3_lg_pb_h5"), ("trained", "latin_p8"), ("trained", "latin_p5"), ("frozen", "rgb3_lg_pb_h5"),
         ("trained", "latin_p6"), ("direct", "rgb3_lg_pb_h5"), ("trained", "lg_pb_h5"), ("frozen", "latin_p8"),
         ("frozen", "latin_p5"), ("frozen", "lg_pb_h5"), ("direct", "lg_pb_h5"), ("direct", "latin_p8"),
         ("direct", "latin_p5"), ("frozen", "latin_p6"), ("direct", "latin_p6")]


def print_queue(a) -> None:
    """Each cache build (lean, --build-only) goes right before the first run that needs it."""
    built = set()
    for arm, e in ORDER:
        if e not in built:
            built.add(e)
            print(f"python3 encoder_probe/experiments/run_g2lite.py --build-only --encoding {e} "
                  f"--train-pages {a.train_pages} --test-pages {a.test_pages} --threads 1")
        print(f"python3 encoder_probe/experiments/run_g2lite.py --arm {arm} --encoding {e} --steps {a.steps} "
              f"--train-pages {a.train_pages} --run-name {a.run_name} --threads 1")


class TinyVision:
    """QwenVision's __call__ on the tiny random-init tower (tests)."""

    def __new__(cls, tower):
        from encoder_probe.ep.vlm_encoders import QwenVision
        v = QwenVision.__new__(QwenVision)
        v.model, v.device, v.dtype = tower, torch.device("cpu"), torch.float32
        v.compute, v.out_dim = "cpu/float32", tower.merger.mlp[2].out_features
        return v


def get_hidden(a, enc_name, tr, te, split, vision_fn, log=print):
    """-> (hidden cache dict, token cache dict, seconds waited). Waits for a builder holding the cache lock;
    builds the cache here if none is running."""
    hp = TC.cache_path(TC.hidden_encoder("qwen25vl", split), "clean", enc_name, a.train_pages, a.test_pages)
    tp = TC.cache_path("qwen25vl", "clean", enc_name, a.train_pages, a.test_pages)
    t0, last = time.time(), 0.0
    while not (hp.exists() and tp.exists()):
        if TC.build_locked(tp):
            if time.time() - last > 600:
                log(f"[g2] waiting for the builder of {hp.name} ({time.time() - t0:.0f}s)", flush=True)
                last = time.time()
            time.sleep(a.poll)
            continue
        log(f"[g2] no builder: building {hp.parent.parent.name}/{hp.name} here", flush=True)
        TC.build(vision_fn(), "qwen25vl", enc_name, "clean", S, a.train_pages, a.test_pages, tr, te, log=log,
                 split=split)
    waited = time.time() - t0
    dh = torch.load(hp, weights_only=False, mmap=True)
    dt = torch.load(tp, weights_only=False, mmap=True)
    for d, p in ((dh, hp), (dt, tp)):
        if d["digest_train"] != pages_digest(tr) or d["digest_test"] != pages_digest(te):
            raise RuntimeError(f"{p}: cached states were made from different pages")
    assert dh.get("split_block") == split and dh.get("order") == "window", (dh.get("split_block"), dh.get("order"))
    return dh, dt, waited


def towers(a):
    """-> (split block, window size, ViT factory, pretrained-merger factory) for the real or the tiny tower."""
    if a.tiny:
        from encoder_probe.ep import qwen_ft
        tower = qwen_ft.load_model(torch.device("cpu"), tiny=True)[0].model.visual.eval()
        return (len(tower.blocks), tower.window_size, lambda: TinyVision(tower),
                lambda: copy.deepcopy(tower.merger).float())

    def vision_fn():
        from encoder_probe.ep.vlm_encoders import QwenVision
        return QwenVision()
    return 32, 112, vision_fn, lambda: load_merger(WEIGHTS / "qwen25vl3b_visual.safetensors")


class LeanVision:
    """Wraps a vision tower for token_cache.build: drops the tower right after it has encoded ``last`` (the final
    missing chunk), so the build's final concatenation does not run next to ~2.7 GB of fp32 ViT weights."""

    def __init__(self, inner, last: np.ndarray | None):
        self.inner, self.last, self.compute = inner, last, getattr(inner, "compute", None)

    def __call__(self, imgs, **kw):
        r = self.inner(imgs, **kw)
        if self.last is not None and imgs.shape == self.last.shape and np.array_equal(imgs, self.last):
            self.inner = None
            gc.collect()
        return r


def build_only(a) -> int:
    """Build the (tokens + hidden-state) caches of one encoding with the label strings dropped first."""
    cache_root = Path(a.cache_root)
    TC.CACHE, TC.PAGES = cache_root / "tokens", cache_root / "pages"
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    split, _, vision_fn, _ = towers(a)
    hp = TC.cache_path(TC.hidden_encoder("qwen25vl", split), "clean", a.encoding, a.train_pages, a.test_pages)
    tp = TC.cache_path("qwen25vl", "clean", a.encoding, a.train_pages, a.test_pages)
    t0 = time.time()
    while not (hp.exists() and tp.exists()) and TC.build_locked(tp):
        time.sleep(a.poll)
    if hp.exists() and tp.exists():
        print(f"[g2] cache ready: {hp.name} (waited {time.time() - t0:.0f}s)", flush=True)
        return 0
    _, tr, te = TC.split_pages(a.encoding, S, a.train_pages, a.test_pages, cache=True)
    for p in tr + te:   # the build only needs the images (digests); labels of dense codes are ~1.5 GB
        p.labels = p.refs = None
    gc.collect()
    # chunks are extracted in order (train, then test), so the test chunk is the last one still to do
    last = np.stack([TC.degrade(p.img, "clean") for p in te])
    TC.build(LeanVision(vision_fn(), last), "qwen25vl", a.encoding, "clean", S, a.train_pages, a.test_pages,
             tr, te, split=split, chunk=a.chunk)
    print(f"[g2] built {hp.name} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--encoding")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--train-pages", type=int, default=500)
    ap.add_argument("--test-pages", type=int, default=16)
    ap.add_argument("--batch-pages", type=int, default=2)
    ap.add_argument("--lines-per-page", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--recon-weight", type=float, default=10.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-pages", type=int, default=4, help="test pages for the curve (the row's cer uses all)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", choices=["auto", "on", "off"], default="auto",
                    help="bf16 autocast for the probe; auto = on only with native bf16 (AMX/AVX512-BF16)")
    ap.add_argument("--run-name", default="g2lite")
    ap.add_argument("--ckpt-every", type=int, default=200, help="resume checkpoint interval (0 = off)")
    ap.add_argument("--stop-after", type=int, default=0, help="exit after this step's checkpoint (resume tests)")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--print-queue", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="tiny random-init tower (tests)")
    ap.add_argument("--build-only", action="store_true",
                    help="build the encoding's hidden-state cache (lean: no label strings, ViT freed before the "
                         "final concatenation) and exit; waits instead if another builder holds the cache")
    ap.add_argument("--chunk", type=int, default=25, help="--build-only: pages per resumable chunk (keep it fixed: "
                    "a chunk is reused only if its first page and page digest match)")
    ap.add_argument("--out-root", default=str(ROOT / "encoder_probe" / "results"))
    ap.add_argument("--cache-root", default=str(ROOT / "encoder_probe" / "cache"))
    a = ap.parse_args(argv)
    if a.print_queue:
        print_queue(a)
        return 0
    if a.build_only:
        return build_only(a)
    assert a.arm and a.encoding, "--arm and --encoding are required"
    bf16 = a.bf16 == "on" or (a.bf16 == "auto" and cpu_has_bf16())
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)   # fp32 on CPU: denormals made probes 5.5x slower (AGENTS.md)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    cache_root = Path(a.cache_root)
    TC.CACHE, TC.PAGES = cache_root / "tokens", cache_root / "pages"
    out = Path(a.out_root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    key = dict(encoding=a.encoding, arm=a.arm, steps=a.steps, train_pages=a.train_pages, seed=a.seed)
    rows = out / "runs.jsonl"
    if rows.exists():
        for l in rows.read_text().splitlines():
            if l.strip() and all(json.loads(l).get(k) == v for k, v in key.items()):
                print(f"[g2] skip, done: {key}")
                return 0
    ck = cache_root / "ckpt" / f"g2lite__{a.run_name}__{a.arm}__{a.encoding}__{a.steps}__tp{a.train_pages}__s{a.seed}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    run_lock = open(ck.with_suffix(".lock"), "a")   # held for the process lifetime (released by the kernel)
    if not flock.try_lock(run_lock):
        print(f"[g2] skip, running in another process: {key}")
        return 0
    t0 = time.time()
    split, window, vision_fn, merger_fn = towers(a)
    enc, tr, te = TC.split_pages(a.encoding, S, a.train_pages, a.test_pages, cache=True)
    t_pages = time.time() - t0
    dh, dt, waited = get_hidden(a, a.encoding, tr, te, split, vision_fn)
    h_tr, h_te = dh["train"], dh["test"]
    N, c = h_tr.shape[1:]
    assert h_tr.shape == (len(tr), (S // 14) ** 2, c) and N == 4 * G * G, h_tr.shape
    reverse = window_reverse(S, 14, 2, window)
    # check on the test pages: pretrained merger on the cached (bf16) states vs the cached model tokens (computed
    # from the fp32 states by the tower's own forward)
    with torch.no_grad():
        chk_front = Front("frozen", merger_fn(), reverse, c)
        mt = torch.cat([chk_front.raw(h_te[i:i + 4]) for i in range(0, len(h_te), 4)])
        ref = dt["test"].float()
        merger_check = {"max_abs": (mt - ref).abs().max().item(),
                        "rel_l2": ((mt - ref).norm() / ref.norm()).item(),
                        "corr": float(np.corrcoef(mt.flatten().numpy(), ref.flatten().numpy())[0, 1])}
        del chk_front, mt, ref
    front = Front(a.arm, None if a.arm == "direct" else merger_fn(), reverse, c)
    if a.arm == "frozen":
        raw_tr = front.fit_standardiser(h_tr, keep=True)
        with torch.no_grad():
            tok_tr = (raw_tr - front.mu) / front.sd
            tok_te = torch.cat([front(h_te[i:i + 4]) for i in range(0, len(h_te), 4)])
        del raw_tr
    else:
        front.fit_standardiser(h_tr)
        tok_tr, tok_te = h_tr, h_te
    enc_compute = dh.get("enc_compute")
    digest_test = pages_digest(te)
    pix_tr = torch.from_numpy(np.stack([p.img for p in tr])).permute(0, 3, 1, 2).contiguous()   # clean, uint8
    for p in tr:
        p.img = None
    t_feat = time.time() - t0
    classes = enc.classes
    cidx = {ch: i + 1 for i, ch in enumerate(classes)}
    tgt = [[torch.tensor([cidx[x] for x in lab], dtype=torch.long) for lab in p.labels] for p in tr]
    for p in tr:   # the label strings of 500 dense pages hold ~1.5 GB (rgb3_lg: ~560 two-char labels per line)
        p.labels = p.refs = None
    gc.collect()
    chars = sum(p.n_chars for p in te)
    ppc = S * S * len(te) / max(1, chars)
    cpt = chars / (len(te) * G * G)
    torch.manual_seed(a.seed)   # same probe init in every arm (the merger build above consumed the RNG)
    probe = GridProbe(enc.pitch, len(classes), d=front.d, G=G, P=P)
    model = G2Reader(None if a.arm == "frozen" else front, probe)
    params = [p for p in model.parameters() if p.requires_grad]
    n_merger = sum(p.numel() for p in front.parameters() if p.requires_grad) if a.arm == "trained" else 0
    print(f"[g2] {key} pitch={enc.pitch} lines/page={enc.n_lines} classes={len(classes)} d={front.d} "
          f"train={len(tr)} test={len(te)} px2/char={ppc:.2f} chars/token={cpt:.1f} pages={t_pages:.0f}s "
          f"wait={waited:.0f}s feat={t_feat:.0f}s bf16={bf16} trainable={sum(p.numel() for p in params)} "
          f"(merger {n_merger}) merger_check={merger_check}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4, fused=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    curve, train_curve, ctcs, start, t_prev = [], [], [], 1, 0.0
    if a.ckpt_every and ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        curve, train_curve, ctcs = st["curve"], st["train_curve"], st["ctcs"]
        start, t_prev = st["step"] + 1, st["t_train"]
        print(f"[g2] resumed from step {st['step']} ({ck.name})", flush=True)
    t_train = time.time() - t_prev
    for step in range(start, a.steps + 1):
        pb = rng.choice(len(tr), a.batch_pages, replace=False)
        pi, li, ys = [], [], []
        for k, p in enumerate(pb):
            ok = [i for i, t in enumerate(tgt[p]) if len(t)]
            for i in rng.choice(ok, min(a.lines_per_page, len(ok)), replace=False):
                pi.append(k), li.append(int(i)), ys.append(tgt[p][i])
        ix = torch.as_tensor(pb)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            feat = model.features(tok_tr[ix])
            logp = model.read(feat, torch.tensor(pi), torch.tensor(li))
        logp = logp.float()
        T = logp.shape[0]
        ctc = F.ctc_loss(logp, torch.cat(ys), torch.full((len(ys),), T, dtype=torch.long),
                         torch.tensor([len(y) for y in ys]), blank=0, zero_infinity=True)
        rec = GridProbe.recon_loss(feat, pix_tr[ix].float() / 127.5 - 1)
        loss = ctc + a.recon_weight * rec
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()
        ctcs.append(round(ctc.item(), 5))
        if step % a.eval_every == 0 or step == a.steps:
            cv, _ = evaluate(model, tok_te, te, enc, classes, max_pages=a.eval_pages, bf16=bf16, device="cpu")
            curve.append((step, round(cv, 5)))
            k0 = train_curve[-1][0] if train_curve else 0
            train_curve.append((step, round(float(np.mean(ctcs[k0:step])), 5)))
            print(f"[g2] step {step} ctc {ctc.item():.3f} (mean {train_curve[-1][1]:.3f}) rec {rec.item():.4f} "
                  f"val_cer {cv:.4f} ({(time.time() - t_train) / step:.3f} s/step, {time.time() - t0:.0f}s)",
                  flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "curve": curve, "train_curve": train_curve,
                        "ctcs": ctcs, "step": step, "t_train": time.time() - t_train}, tmp)
            tmp.replace(ck)
            if a.stop_after and step >= a.stop_after:
                print(f"[g2] stop after step {step} (--stop-after)", flush=True)
                return 0
    s_step = (time.time() - t_train) / a.steps
    cer, samples = evaluate(model, tok_te, te, enc, classes, bf16=bf16, device="cpu")
    cfg = {k: v for k, v in vars(a).items() if k not in ("out_root", "cache_root", "stop_after", "print_queue",
                                                         "poll")}
    row = {**key, "phase": "probe", "round": 3, "gate": "G2-lite", "encoder": f"qwen25vl_h{split}", "cond": "clean",
           "scheme": f"{enc.family}@g2lite_{a.arm}", "family": enc.family, "pitch": enc.pitch, "S": S, "P": P,
           "G": G, "n_tokens": G * G, "d": front.d, "px_per_char": ppc, "budget": round(ppc, 3),
           "chars_per_token": cpt, "chars_per_vlm_token": P * P / ppc, "cer": cer, "score_v": 2,
           "train_ctc": float(np.mean(ctcs[-100:])), "train_ctc_last": ctcs[-1], "train_curve": train_curve,
           "bf16": bf16, "device": "cpu", "enc_compute": enc_compute, "recon_weight": a.recon_weight,
           "test_pages": len(te), "lr": a.lr, "n_test_chars": chars, "digest_test": digest_test, "curve": curve,
           "trainable": sorted({n.split(".")[0] + "." + n.split(".")[1] for n in trainable_names(model)}),
           "params": {"probe": sum(p.numel() for p in probe.parameters()), "merger_trainable": n_merger},
           "standardise": "merger_pretrained" if a.arm != "direct" else "concat_states",
           "merger_check": merger_check, "config": cfg, "tiny": a.tiny, "wait_s": waited, "s_per_step": s_step,
           "train_time_s": time.time() - t0, "smoke": "smoke" in a.run_name or a.tiny, "samples": samples}
    print(f"[g2] TEST {key} cer={cer:.4f} chars/token={cpt:.1f} train_ctc={row['train_ctc']:.3f}", flush=True)
    for r, h in samples[:2]:
        print(f"   ref: {r[:90]!r}\n   hyp: {h[:90]!r}")
    with open(rows, "a") as f:
        f.write(json.dumps(row) + "\n")
    ck.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
