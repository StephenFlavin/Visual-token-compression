"""Merge Phase-2 runs: for each (scheme, em) keep the row with the most training steps
(phase2_long 10.5k-step reruns supersede phase2_main 3.5k rows) -> results/phase2/merged/runs.jsonl.

    python3 scripts/merge_phase2.py && python3 -m vbpe.analysis results/phase2/merged/runs.jsonl --out results/phase2/merged
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = [ROOT / "results/phase2" / r / "runs.jsonl" for r in (sys.argv[1:] or ["phase2_main", "phase2_long"])]
best: dict = {}
for f in SRC:
    if not f.exists():
        continue
    for line in f.open():
        r = json.loads(line)
        k = (r["scheme"], float(r["em"]))
        if k not in best or r.get("steps", 0) >= best[k].get("steps", 0):
            best[k] = r | {"source_run": f.parent.name}
out = ROOT / "results/phase2/merged"
out.mkdir(parents=True, exist_ok=True)
with (out / "runs.jsonl").open("w") as fh:
    for k in sorted(best):
        fh.write(json.dumps(best[k]) + "\n")
for (s, e), r in sorted(best.items()):
    print(f"{s:16s} em{e:<5g} ppc={r['px_per_char']:6.2f} cer={100 * r['cer']:6.2f}% steps={r.get('steps')} ({r['source_run']})")
