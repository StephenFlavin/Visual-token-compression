"""Summarise encoder-probe rows: CER tables, C@3% / C@1% (chars per vision token), ceiling survival.

    python3 encoder_probe/experiments/analyse.py encoder_probe/results/{main,controls}/runs.jsonl [--json out.json]

C@x = 256 / P@x, with P@x from vtc.analysis.p_at on (budget = px^2/char, CER) per family curve.
'<' marks left-censored values (already below x at the densest encoding in the family: the true C@x is
at least the printed value); '-' = never reached within the family's range.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vtc.analysis import CurveTable, crossing, p_at, usable_rows  # noqa: E402

FAMILIES = ["latin", "lg_free", "lg_pb", "colour"]
SETTINGS = [("identity", "clean"), ("siglip", "clean"), ("siglip_l6", "clean"), ("siglip_l6", "jpeg75"),
            ("identity", "jpeg75"), ("siglip", "jpeg75"), ("clip", "clean"), ("clip", "jpeg75")]


def c_at(rows, scheme, thr):
    """(chars/token, censor flag) at CER threshold ``thr`` for one family curve."""
    t = CurveTable(rows, scheme)
    if t.empty:
        return None, "missing"
    p = p_at(rows, scheme, thr)
    _, cens = crossing(t.x, t.mean(), thr, t.counts())
    return (None if p is None else 256 / p), cens


def fmt_c(v, cens):
    if cens == "missing":
        return "n/a"
    if v is None:
        return "-"
    return f"{'>=' if cens == 'left' else ''}{v:.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", default=[str(ROOT / f"encoder_probe/results/{d}/runs.jsonl")
                                                 for d in ("main", "controls")])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = []
    for p in a.runs:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    best = {}
    for r in usable_rows(rows):   # later rows win; a colour row scored before the strip fix never wins
        k = (r["encoding"], r["encoder"], r["cond"])
        if k in best and best[k].get("score_v", 1) > r.get("score_v", 1):
            continue
        best[k] = r
    rows = list(best.values())
    by = best
    settings = [s for s in SETTINGS if any((r["encoder"], r["cond"]) == s for r in rows)]
    encs = sorted({r["encoding"] for r in rows}, key=lambda e: (FAMILIES.index(by_fam(rows, e)), -ppc(rows, e)))

    out = {"cer": {}, "c_at": {}}
    print("## CER by encoding\n")
    print("| encoding | px²/char | chars/token | " + " | ".join(f"{e}/{c}" for e, c in settings) + " |")
    print("|---|---:|---:|" + "---:|" * len(settings))
    for e in encs:
        x = ppc(rows, e)
        cells = []
        for s in settings:
            r = by.get((e, *s))
            cells.append("" if r is None else f"{100 * r['cer']:.2f}%")
            if r is not None:
                out["cer"][f"{e}@{s[0]}/{s[1]}"] = r["cer"]
        print(f"| {e} | {x:.2f} | {256 / x:.1f} | " + " | ".join(cells) + " |")

    for thr in (0.10, 0.03, 0.01):
        print(f"\n## C@{100 * thr:g}% (chars per vision token; text tokens ≈ 4 chars/token)\n")
        print("| family | " + " | ".join(f"{e}/{c}" for e, c in settings) + " |")
        print("|---|" + "---:|" * len(settings))
        for f in FAMILIES:
            cells = []
            for s in settings:
                sch = f"{f}@{s[0]}/{s[1]}"
                v, cens = c_at(rows, sch, thr)
                out["c_at"][f"{sch}@{thr}"] = {"chars_per_token": v, "censored": cens}
                cells.append(fmt_c(v, cens))
            print(f"| {f} | " + " | ".join(cells) + " |")

    print("\n## Ceiling survival: CER through the encoder vs identity/clean (same encoding)\n")
    others = [s for s in settings if s != ("identity", "clean")]
    print("| encoding | identity/clean | " + " | ".join(f"{e}/{c}" for e, c in others) + " |")
    print("|---|---:|" + "---:|" * len(others))
    for e in encs:
        base = by.get((e, "identity", "clean"))
        cells = [("" if by.get((e, *s)) is None else f"{100 * by[(e, *s)]['cer']:.2f}%") for s in others]
        print(f"| {e} | {'' if base is None else f'{100 * base['cer']:.2f}%'} | " + " | ".join(cells) + " |")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


def by_fam(rows, e):
    return next(r["family"] for r in rows if r["encoding"] == e)


def ppc(rows, e):
    v = [r["px_per_char"] for r in rows if r["encoding"] == e]
    return sum(v) / len(v)


if __name__ == "__main__":
    raise SystemExit(main())
