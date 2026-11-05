"""G2-lite table: codes x arms (test CER on 16 pages, chars/token, final train CTC) from results/g2lite/runs.jsonl.

    python3 encoder_probe/experiments/analyse_g2lite.py [runs.jsonl ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("frozen", "trained", "direct", "trained@2e-3", "blind")
# "trained" = the amended arm (merger LR 1e-4, ROUND3.md amendment 2); "trained@2e-3" = as registered (collapsed)


def load(paths):
    rows = {}
    for p in paths:
        for l in Path(p).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                arm = r["arm"]
                if arm == "trained" and r.get("merger_lr") is None:
                    arm = "trained@2e-3"
                old = rows.get((r["encoding"], arm))
                if old is not None and "ops" not in r and old.get("ops") and old["cer"] == r["cer"]:
                    continue                                 # keep the exact-resume row with the op split
                rows[(r["encoding"], arm)] = r               # later rows win
    return rows


def table(rows) -> str:
    codes = sorted({e for e, _ in rows}, key=lambda e: next(r["chars_per_token"] for (x, _), r in rows.items()
                                                            if x == e))
    out = ["| code | chars/token | " + " | ".join(ARMS) + " |", "|---|---:|" + "---|" * len(ARMS)]
    for e in codes:
        cpt = next(r["chars_per_token"] for (x, _), r in rows.items() if x == e)
        cells = []
        for a in ARMS:
            r = rows.get((e, a))
            if r is None:
                cells.append("not run")
                continue
            o = r.get("ops")
            op = (f"; del {100 * o['del']:.0f} sub {100 * o['sub']:.0f} len {o['len_ratio']:.2f}"
                  f" correct {100 * (o['len_ratio'] - o['sub'] - o['ins']):.0f}%") if o else ""
            cells.append(f"{100 * r['cer']:.1f}% (ctc {r['train_ctc']:.2f}{op}; curve "
                         + " ".join(f"{100 * c:.0f}" for _, c in r["curve"]) + ")")
        out.append(f"| {e} | {cpt:.1f} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None) -> int:
    paths = (argv if argv is not None else sys.argv[1:]) or [ROOT / "encoder_probe" / "results" / "g2lite" /
                                                              "runs.jsonl"]
    print(table(load(paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
