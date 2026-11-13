#!/usr/bin/env python3
"""E2 transfer + E3 bridge for saved glyph banks (WP-LG3).

  # E2: fresh CRNN on each frozen bank (Phase-2 budget), rows -> learned_glyphs/results/<run>/runs.jsonl
  python3 learned_glyphs/experiments/transfer_eval.py --banks 'learned_glyphs/results/e1*/banks/*.pt' \\
      --run-name e2_transfer --parallel 2 [--channel harsh]

  # E3: score banks with the unchanged Phase-2 pipeline at several ems (runtime scheme hook)
  python3 learned_glyphs/experiments/transfer_eval.py --bridge --banks ... --ems 4 5 6 7 8 --run-name e3_bridge

  # E3 via the parent CLI (needs the parent hook; prints the patch if it is missing)
  python3 learned_glyphs/experiments/transfer_eval.py --bridge-cmd --banks ... --ems 4 5 6

  # report: summary.md / cer_curves.png / decision (S1, S2, GO) + atlas
  python3 learned_glyphs/experiments/transfer_eval.py --report [--atlas]

Jobs already present in the run's runs.jsonl (same bank sha, mode, channel, em) are skipped.
Every job uses 1 torch thread; --parallel processes run concurrently (spawn).
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

LG_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = LG_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learned_glyphs.lg import transfer as TR  # noqa: E402
from learned_glyphs.lg.scheme import (BRIDGE_PATCH, bank_sha8, phase2_hook_installed,  # noqa: E402
                                      scheme_name_for)


def expand_banks(specs: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in specs:
        hits = sorted(glob.glob(s)) or sorted(glob.glob(str(REPO_ROOT / s)))
        if not hits and Path(s).is_dir():
            hits = sorted(glob.glob(str(Path(s) / "**" / "*.pt"), recursive=True))
        if not hits:
            raise SystemExit(f"no bank matches {s!r}")
        out += [Path(h).resolve() for h in hits]
    return list(dict.fromkeys(out))


def done_keys(run_dir: Path) -> set:
    f = run_dir / "runs.jsonl"
    keys = set()
    if f.exists():
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                keys.add((r.get("bank_sha8"), r.get("train_mode"), r.get("channel"),
                          round(float(r["em"]), 3) if r.get("train_mode") == "bridge" else None))
    return keys


def _pool_run(fn, jobs, parallel: int) -> list[dict]:
    rows = []
    if parallel <= 1 or len(jobs) <= 1:
        for j in jobs:
            rows += fn(j)
        return rows
    with ProcessPoolExecutor(parallel, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for f in as_completed(futs):
            try:
                rows += f.result()
            except Exception as e:  # keep the grid going
                print(f"[transfer_eval] FAILED {futs[f][0]}: {e!r}", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--banks", nargs="*", default=[], help="bank .pt paths / globs / run dirs")
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--channel", default="default", help="transfer channel preset (clean|default|harsh)")
    ap.add_argument("--seed", type=int, default=1, help="reader seed (transfer); bridge uses 0 like Phase 2")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-root", default=None, help="default learned_glyphs/results")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--bridge", action="store_true", help="E3 in-process (runtime hook) instead of E2")
    ap.add_argument("--bridge-cmd", action="store_true",
                    help="print (and with --execute run) the parent Phase-2 command per bank")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--ems", nargs="+", type=float, default=None,
                    help="bridge ems (default: Phase-2 grid 4 5 6 7 8 + each bank's native em)")
    ap.add_argument("--n-train", type=int, default=40_000, help="bridge: Phase-2 n_train")
    ap.add_argument("--allow-fallback", action="store_true", help="bridge smoke: pydoc fallback text")
    ap.add_argument("--report", action="store_true", help="write learned_glyphs/results/summary/")
    ap.add_argument("--atlas", action="store_true", help="with --report: atlas of --banks")
    ap.add_argument("--include-smoke", action="store_true")
    a = ap.parse_args(argv)

    banks = expand_banks(a.banks) if a.banks else []
    out_root = Path(a.out_root) if a.out_root else TR.RESULTS

    if a.bridge_cmd:
        if not banks:
            raise SystemExit("--bridge-cmd needs --banks")
        names = [scheme_name_for(b) for b in banks]
        ems = a.ems or [4, 5, 6, 7, 8]
        cmd = (f"python3 experiments/phases/phase2_train.py --schemes {' '.join(shlex.quote(n) for n in names)} "
               f"--ems {' '.join(f'{e:g}' for e in ems)} --steps {a.steps} --parallel {max(1, a.parallel)} "
               f"--threads 1 --run-name {a.run_name or 'lg_bridge'}")
        print(cmd)
        if not phase2_hook_installed():
            print("\n# parent hook missing: vtc/crnn/data.py::resolve_scheme does not know 'lg:' names.\n"
                  "# Apply this patch (or use --bridge, which installs the same hook at runtime):\n")
            print(BRIDGE_PATCH)
            return 2
        if a.execute:
            import subprocess
            return subprocess.call(cmd, shell=True, cwd=REPO_ROOT)
        return 0

    if a.bridge:
        run = a.run_name or time.strftime("bridge_%Y%m%d_%H%M")
        run_dir = out_root / run
        done = set() if a.no_resume else done_keys(run_dir)
        jobs = []
        for b in banks:
            from learned_glyphs.lg.scheme import make_lg_scheme
            ems = a.ems or sorted({4.0, 5.0, 6.0, 7.0, 8.0, round(make_lg_scheme(scheme_name_for(b)).native_em, 3)})
            sha = bank_sha8(b)
            for em in ems:
                if (sha, "bridge", "phase2", round(float(em), 3)) in done:
                    continue
                kw = dict(steps=a.steps, run_name=run, out_root=str(out_root), n_train=a.n_train,
                          n_val=a.n_val, n_test=a.n_test, batch=a.batch, eval_every=a.eval_every,
                          threads=1, seed=0, allow_fallback=a.allow_fallback)
                jobs.append((str(b), float(em), kw))
        jobs.sort(key=lambda j: -j[1])                      # big ems first (slowest)
        print(f"[transfer_eval] bridge run={run} jobs={len(jobs)} steps={a.steps}", flush=True)
        rows = _pool_run(TR.bridge_job, jobs, a.parallel)
    elif banks:
        run = a.run_name or time.strftime("transfer_%Y%m%d_%H%M")
        run_dir = out_root / run
        done = set() if a.no_resume else done_keys(run_dir)
        jobs = []
        for b in banks:
            if (bank_sha8(b), "transfer", a.channel, None) in done:
                continue
            cfg = TR.TransferConfig(bank_path=str(b), channel=a.channel, steps=a.steps, batch=a.batch,
                                    eval_every=a.eval_every, n_val=a.n_val, n_test=a.n_test,
                                    seed=a.seed, run_name=run, out_root=str(out_root))
            jobs.append((str(b), asdict(cfg)))
        print(f"[transfer_eval] transfer run={run} jobs={len(jobs)} steps={a.steps}", flush=True)
        rows = _pool_run(TR.run_job, jobs, a.parallel)
    else:
        rows = []

    for r in sorted(rows, key=lambda r: (r["scheme"], r["em"])):
        print(f"  {r['scheme']:18s} {r['train_mode']:8s} em={r['em']:<6.3g} h={r['height_px']:<5.3g} "
              f"px/char={r['px_per_char']:6.2f} cer={r['cer']:.4f} sym_cer={r['symbol_cer']:.4f} "
              f"{r['train_time_s']:.0f}s  [{r.get('source_tag')}]")

    if a.report:
        from learned_glyphs.lg import report
        out = out_root / "summary"
        if a.atlas and banks:
            p = report.atlas_comparison([(b.stem, b) for b in banks], out / "atlas_comparison.png")
            print(f"wrote {p}")
        report.write_report(out, include_smoke=a.include_smoke)
        print(f"wrote {out}/summary.md, cer_curves.png, decision.json")
    elif not banks:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
