"""Tiny CPU job queue: run single-core shell commands from a queue file, keeping --cores busy.

Each non-empty, non-comment line of the queue file is one shell command (run from the repo root,
OMP_NUM_THREADS=1). Lines are claimed in order; new lines may be appended while the queue runs.
Cores used by OTHER training processes (phase2_train.py, phase2 pool workers) are measured from
/proc CPU deltas and subtracted, so this queue backfills whatever another grid leaves idle.
With --after-phase2, nothing starts until the Phase-2 driver has launched its last job.

    nohup python3 scripts/job_queue.py results/queue.txt > results/queue.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "phases"))
from phase2_driver import busy_external  # noqa: E402


def phase2_all_started() -> bool:
    log = ROOT / "results" / "phase2_driver.log"
    if not log.exists():
        return True
    txt = log.read_text()
    return "left=0)" in txt or "all jobs finished" in txt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("queue")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--after-phase2", action="store_true")
    ap.add_argument("--log-dir", default="results/queue_logs")
    a = ap.parse_args(argv)
    qpath, log_dir = Path(a.queue), ROOT / a.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    claimed = 0
    running: dict[int, tuple[subprocess.Popen, str]] = {}
    idle_rounds = 0
    while True:
        for pid in [p for p, (proc, _) in running.items() if proc.poll() is not None]:
            proc, cmd = running.pop(pid)
            print(f"[queue] done rc={proc.returncode}: {cmd}", flush=True)
        lines = [l.strip() for l in qpath.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
        pending = lines[claimed:]
        if not pending and not running:
            idle_rounds += 1
            if idle_rounds >= 4:  # ~1 min with nothing to do -> exit
                print("[queue] empty, exiting", flush=True)
                return 0
        else:
            idle_rounds = 0
        if pending and (not a.after_phase2 or phase2_all_started()):
            own = {p for p in running}
            for p, _ in list(running.values()):
                try:
                    own |= {int(c) for c in Path(f"/proc/{p.pid}/task/{p.pid}/children").read_text().split()}
                except OSError:
                    pass
            ext = busy_external(own | {os.getpid()})
            free = a.cores - ext - len(running)
            while free > 0 and pending:
                cmd = pending.pop(0)
                claimed += 1
                log = open(log_dir / f"job{claimed:03d}.log", "a")
                log.write(f"$ {cmd}\n")
                log.flush()
                proc = subprocess.Popen(cmd, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "OMP_NUM_THREADS": "1"})
                running[proc.pid] = (proc, cmd)
                print(f"[queue] start job{claimed:03d} (ext busy={ext}, running={len(running)}): {cmd}",
                      flush=True)
                free -= 1
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
