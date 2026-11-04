"""Core-filling scheduler for Phase-2 runs.

Launches one ``phase2_train.py`` process per (scheme, em) job, longest (largest em) first, keeping
``--cores`` training processes busy in total. Busy cores already used by OTHER phase2_train
processes (e.g. an earlier grid still finishing) are measured from /proc CPU deltas and subtracted.
All jobs append to the same run (default phase2_main); phase2_train resumes per (scheme, em).

    nohup python3 experiments/phase2_driver.py > results/phase2_driver.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100   # no /proc on Windows: see _train_pids

# Full Phase-2 grid. Finished (scheme, em) rows are skipped by phase2_train, so re-running this
# driver after a crash/reclaim resumes the grid (in-flight runs restart from step 0).
DEFAULT_JOBS = ([("latin", e) for e in (4, 5, 6, 7, 8)]
                + [(s, e) for s in ("latin_condensed", "subst300", "subst1000") for e in (5, 6, 7, 8, 10)]
                + [(q, e) for q in ("quad3", "quad4") for e in (8, 10, 12)])


def _cpu_ticks(pid: int) -> int | None:
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])  # utime + stime
    except (OSError, IndexError, ValueError):
        return None


def _train_pids(exclude: set[int]) -> list[int]:
    out = []
    if not Path("/proc").is_dir():   # Windows: no Phase-2 processes to subtract
        return out
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in exclude:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        # training work happens in the parent (parallel=1) or in multiprocessing pool workers
        if "phase2_train.py" in cmd or "multiprocessing.spawn" in cmd:
            out.append(int(d.name))
    return out


def busy_external(exclude: set[int], window: float = 5.0) -> int:
    pids = _train_pids(exclude)
    t0 = {p: _cpu_ticks(p) for p in pids}
    time.sleep(window)
    busy = 0
    for p, a in t0.items():
        b = _cpu_ticks(p)
        if a is not None and b is not None and (b - a) / TICKS / window > 0.5:
            busy += 1
    return busy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--run-name", default="phase2_main")
    a = ap.parse_args(argv)

    jobs = sorted(DEFAULT_JOBS, key=lambda j: -j[1])  # largest em (slowest) first
    log_dir = ROOT / "results" / "phase2" / a.run_name / "driver"
    log_dir.mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple[subprocess.Popen, tuple[str, int]]] = {}

    def own_pids() -> set[int]:
        pids = set()
        for p, _ in running.values():
            pids.add(p.pid)
            try:  # children of our own jobs (render workers) are ours too
                for c in Path(f"/proc/{p.pid}/task/{p.pid}/children").read_text().split():
                    pids.add(int(c))
            except OSError:
                pass
        return pids

    while jobs or running:
        for pid in [p for p, (proc, _) in running.items() if proc.poll() is not None]:
            proc, job = running.pop(pid)
            print(f"[driver] done {job} rc={proc.returncode}", flush=True)
        ext = busy_external(own_pids() | {os.getpid()})
        free = a.cores - ext - len(running)
        while free > 0 and jobs:
            s, e = jobs.pop(0)
            cmd = [sys.executable, "experiments/phase2_train.py", "--schemes", s, "--ems", str(e),
                   "--steps", str(a.steps), "--parallel", "1", "--threads", "1",
                   "--render-workers", "1", "--run-name", a.run_name]
            log = open(log_dir / f"{s}_em{e}.log", "a")
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "OMP_NUM_THREADS": "1"})
            running[proc.pid] = (proc, (s, e))
            print(f"[driver] start {(s, e)} (external busy={ext}, running={len(running)}, left={len(jobs)})",
                  flush=True)
            free -= 1
        time.sleep(15)
    print("[driver] all jobs finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
