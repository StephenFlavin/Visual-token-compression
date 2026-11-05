"""G2-lite table: codes x arms (test CER on 16 pages, chars/token, final train CTC) from results/g2lite/runs.jsonl.

    python3 encoder_probe/experiments/analyse_g2lite.py [runs.jsonl ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("frozen", "trained", "direct")


def load(paths):
    rows = {}
    for p in paths:
        for l in Path(p).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                rows[(r["encoding"], r["arm"])] = r          # later rows win
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
            cells.append("not run" if r is None else
                         f"{100 * r['cer']:.1f}% (ctc {r['train_ctc']:.2f}; curve "
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
