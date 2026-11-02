#!/usr/bin/env python3
"""Phase 2: train one CRNN line reader from scratch per (scheme, em); write held-out CER rows.

  python experiments/phase2_train.py --run-name main                    # full default grid
  python experiments/phase2_train.py --smoke --schemes latin --ems 8 5  # quick benchmark

Stages: (1) pre-render every (scheme, em) dataset into results/phase2/cache/ using
``--render-workers`` processes; (2) train ``--parallel`` runs concurrently, each with
``--threads`` torch threads. Rows are appended to results/phase2/<run>/runs.jsonl as runs finish;
(scheme, em) pairs already in runs.jsonl are skipped (resume), unless --no-resume.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vbpe.crnn.train import TrainConfig, default_out, load_data, train_run  # noqa: E402

DEFAULT_SCHEMES = ["latin", "latin_condensed", "subst300", "subst1000", "quad3", "quad4"]
# latin already reaches CER ~2% at em 5 (smoke), so the grid starts at em 4; quad needs the top end
DEFAULT_EMS = [4, 5, 6, 7, 8, 10]
DEFAULT_STEPS = 3500


def _train_job(cfg_dict: dict, out_dir: str) -> dict:
    cfg = TrainConfig(**cfg_dict)
    log_path = Path(out_dir) / "logs" / f"{cfg.scheme}_em{cfg.em:g}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        def log(msg: str) -> None:
            fh.write(msg + "\n")
            fh.flush()
            print(msg, flush=True)
        return train_run(cfg, Path(out_dir), log=log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schemes", nargs="+", default=DEFAULT_SCHEMES)
    ap.add_argument("--ems", nargs="+", type=float, default=DEFAULT_EMS)
    ap.add_argument("--steps", type=int, default=None, help=f"default {DEFAULT_STEPS} (smoke: 600)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=None, help="default 250 (smoke: 100)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per run")
    ap.add_argument("--parallel", type=int, default=4, help="concurrent training runs")
    ap.add_argument("--render-workers", type=int, default=None, help="default threads*parallel")
    ap.add_argument("--n-train", type=int, default=None, help="default 40000 (smoke: 8000)")
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--supersample", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=None, help="BiLSTM hidden size override")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="small budget; allows the built-in fallback text if the corpus is missing")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--prerender-only", action="store_true")
    a = ap.parse_args(argv)

    steps = a.steps or (600 if a.smoke else DEFAULT_STEPS)
    run = a.run_name or ("smoke" if a.smoke else time.strftime("run_%Y%m%d_%H%M"))
    out = default_out(run)
    out.mkdir(parents=True, exist_ok=True)
    model_kw = {"hidden": a.hidden} if a.hidden else {}
    cfgs = [TrainConfig(scheme=s, em=em, steps=steps, batch=a.batch, lr=a.lr,
                        eval_every=a.eval_every or (100 if a.smoke else 250),
                        n_train=a.n_train or (8000 if a.smoke else 40_000), n_val=a.n_val,
                        n_test=a.n_test, threads=a.threads, seed=a.seed, allow_fallback=a.smoke,
                        render={"supersample": a.supersample}, model=model_kw)
            for s in a.schemes for em in a.ems]
    done = set()
    if not a.no_resume and (out / "runs.jsonl").exists():
        for line in open(out / "runs.jsonl"):
            r = json.loads(line)
            done.add((r["scheme"], float(r["em"])))
    cfgs = [c for c in cfgs if (c.scheme, float(c.em)) not in done]
    # big ems / dense schemes first: better packing of the parallel pool
    cfgs.sort(key=lambda c: -c.em)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    print(f"[phase2] run={run} jobs={len(cfgs)} (skipped {len(done)} done) steps={steps}", flush=True)

    t0 = time.time()
    rw = a.render_workers or max(1, a.threads * a.parallel)
    for c in cfgs:
        load_data(c, workers=rw)
    print(f"[phase2] pre-render done in {time.time() - t0:.0f}s", flush=True)
    if a.prerender_only:
        return 0

    rows = []
    if a.parallel <= 1:
        for c in cfgs:
            rows.append(_train_job(asdict(c), str(out)))
    else:
        with ProcessPoolExecutor(a.parallel, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_train_job, asdict(c), str(out)): c for c in cfgs}
            for f in as_completed(futs):
                c = futs[f]
                try:
                    rows.append(f.result())
                except Exception as e:  # keep the grid going
                    print(f"[phase2] FAILED {c.scheme} em={c.em}: {e!r}", flush=True)
    print(f"[phase2] {len(rows)} runs in {time.time() - t0:.0f}s -> {out / 'runs.jsonl'}")
    for r in sorted(rows, key=lambda r: (r["scheme"], r["em"])):
        print(f"  {r['scheme']:16s} em={r['em']:<4g} px/char={r['px_per_char']:6.1f} "
              f"cer={r['cer']:.4f} sym_cer={r['symbol_cer']:.4f} best@{r['best_step']} "
              f"{r['train_time_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
