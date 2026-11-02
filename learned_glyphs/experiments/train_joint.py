#!/usr/bin/env python3
"""E1: jointly train glyph banks + CRNN readers through the channel; write rows to
learned_glyphs/results/<run>/runs.jsonl (one per config, Phase-2 row schema with phase="lg").

  # one run
  python3 learned_glyphs/experiments/train_joint.py --n-tokens 300 --height-px 7 --run-name try
  # E1 grid: {fixed, learned} x {N=0, N=300} x h in {5, 6, 7, 10} (= round(1.2 em), em 4,5,6,8)
  nohup python3 learned_glyphs/experiments/train_joint.py --grid e1 --parallel 4 --run-name e1 \
      > learned_glyphs/results/e1.log 2>&1 &
  # quick check (300 steps, small test set)
  python3 learned_glyphs/experiments/train_joint.py --smoke --run-name smoke_joint

Every JointConfig field is a flag (``--lr-glyph 0.05``, ``--no-learn-glyphs``, ``--model '{"hidden": 96}'``).
Grid presets set n_tokens / learn_glyphs / height_px per run; ``--heights`` / ``--n-tokens-list`` /
``--modes`` override the preset axes. Configs whose tag is already in runs.jsonl are skipped
(resume) unless ``--no-resume``. ``--parallel`` runs a process pool, each run with ``--threads``
torch threads (default 1); per-run logs in results/<run>/logs/<tag>.log.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_glyphs.lg.train import JointConfig, run_job  # noqa: E402

GRIDS = {
    # heights = round(1.2 * em) for em 4, 5, 6, 8 (PLAN E1)
    "e1": {"modes": ["fixed", "learned"], "n_tokens": [0, 300], "heights": [5, 6, 7, 10]},
}
SMOKE = dict(steps=300, eval_every=100, n_test=200, n_val=100, n_expand_probe=1000)


def _field_type(f):
    hints = typing.get_type_hints(JointConfig)
    t = hints[f.name]
    args = [a for a in typing.get_args(t) if a is not type(None)]
    return args[0] if args else t


def add_config_args(ap: argparse.ArgumentParser) -> None:
    for f in fields(JointConfig):
        flag = "--" + f.name.replace("_", "-")
        default = f.default if f.default is not MISSING else None
        t = _field_type(f)
        if t is bool:
            ap.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=None,
                            help=f"(default {default})")
        elif t is dict:
            ap.add_argument(flag, dest=f.name, type=json.loads, default=None,
                            help="JSON dict (default {})")
        else:
            ap.add_argument(flag, dest=f.name, type=t if t in (int, float, str) else str,
                            default=None, help=f"(default {default})")


def base_config(a: argparse.Namespace) -> JointConfig:
    kw = {f.name: getattr(a, f.name) for f in fields(JointConfig) if getattr(a, f.name) is not None}
    if a.smoke:
        for k, v in SMOKE.items():
            kw.setdefault(k, v)
    kw.setdefault("run_name", a.run_name or ("smoke_joint" if a.smoke else "joint"))
    return JointConfig(**kw)


def expand_grid(base: JointConfig, a: argparse.Namespace) -> list[JointConfig]:
    if not a.grid:
        return [base]
    g = dict(GRIDS[a.grid])
    if a.modes:
        g["modes"] = a.modes
    if a.n_tokens_list:
        g["n_tokens"] = a.n_tokens_list
    if a.heights:
        g["heights"] = a.heights
    out = []
    # largest heights first (slowest runs start first -> better packing)
    for h in sorted(g["heights"], reverse=True):
        for n in g["n_tokens"]:
            for m in g["modes"]:
                out.append(replace(base, height_px=int(h), n_tokens=int(n),
                                   learn_glyphs=(m == "learned")))
    return out


def done_tags(out_dir: Path) -> set[str]:
    p = out_dir / "runs.jsonl"
    if not p.exists():
        return set()
    tags = set()
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                tags.add(json.loads(line)["tag"])
            except (ValueError, KeyError):
                pass
    return tags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--grid", choices=sorted(GRIDS), default=None)
    ap.add_argument("--modes", nargs="+", choices=["fixed", "learned"], default=None)
    ap.add_argument("--n-tokens-list", nargs="+", type=int, default=None)
    ap.add_argument("--heights", nargs="+", type=int, default=None)
    ap.add_argument("--parallel", type=int, default=1, help="concurrent runs (process pool)")
    ap.add_argument("--smoke", action="store_true", help=f"small budget: {SMOKE}")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the configs and exit")
    a = ap.parse_args(argv)

    base = base_config(a)
    cfgs = expand_grid(base, a)
    out_dir = base.out_dir()
    done = set() if a.no_resume else done_tags(out_dir)
    todo = [c for c in cfgs if c.tag not in done]
    print(f"[train_joint] run={base.run_name} out={out_dir} configs={len(cfgs)} "
          f"done={len(cfgs) - len(todo)} todo={len(todo)} parallel={a.parallel}", flush=True)
    for c in todo:
        print(f"  {c.tag}: steps={c.steps} batch={c.batch} lr_reader={c.lr_reader} "
              f"lr_glyph={c.lr_glyph} tau={c.tau_start}->{c.tau_end} hard_from={c.hard_from} "
              f"channel={c.channel}", flush=True)
    if a.dry_run or not todo:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grid_config.json").write_text(json.dumps(
        {"argv": sys.argv[1:], "configs": [asdict(c) for c in cfgs]}, indent=1))
    t0 = time.time()
    failed = 0
    if a.parallel <= 1:
        for c in todo:
            run_job(asdict(c))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.parallel, mp_context=ctx) as ex:
            futs = {ex.submit(run_job, asdict(c)): c.tag for c in todo}
            for fut in as_completed(futs):
                try:
                    row = fut.result()[0]
                    print(f"[train_joint] DONE {futs[fut]} cer={row['cer']:.4f} "
                          f"px/char={row['px_per_char']:.2f} ({time.time() - t0:.0f}s)", flush=True)
                except Exception as e:  # keep the rest of the grid going
                    failed += 1
                    print(f"[train_joint] FAILED {futs[fut]}: {e!r}", flush=True)
    print(f"[train_joint] finished {len(todo) - failed}/{len(todo)} in {time.time() - t0:.0f}s",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
