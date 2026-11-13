"""Summarise round-2 probe rows (run_probe2.py): CER per encoding and C@x per family, per setting.

    python3 encoder_probe/experiments/analyse2.py encoder_probe/results/round2*/runs.jsonl

A setting is (encoder, cond, train_pages). chars/VLM token = the paired VLM's P^2 / px^2 per char (DeepEncoder:
4096, Qwen: 784), so identity32 rows read on DeepEncoder's token axis. C@x = P^2 / P@x with P@x from
vtc.analysis.p_at per family curve ('>=' = left-censored, '-' = never reached). The guard column is the
identity ceiling of the same geometry (ROUND2.md: a row is interpreted only if its ceiling is < 10% CER).
Duplicate (setting, encoding) rows: the later row wins. Smoke rows are dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vtc.analysis import CurveTable, crossing, p_at, usable_rows  # noqa: E402

FAMILIES = ["latin", "mono", "rgb3", "lg_pb", "lg_free", "colour"]
VLM_P2 = {"identity28": 784, "qwen25vl": 784, "identity32": 4096, "deepencoder": 4096}
CEILING = {"qwen25vl": "identity28", "deepencoder": "identity32"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    best = {}
    for p in a.runs:
        for l in Path(p).read_text().splitlines():
            r = json.loads(l) if l.strip() else None
            if r and not r.get("smoke") and r.get("round") == 2:
                best[(r["encoder"], r["cond"], r.get("train_pages", 200), r["encoding"])] = r
    rows = usable_rows(best.values())
    best = {k: r for k, r in best.items() if r in rows}
    settings = sorted({k[:3] for k in best}, key=lambda s: (s[0].startswith("identity"), s))
    encs = sorted({r["encoding"] for r in rows},
                  key=lambda e: (FAMILIES.index(next(r["family"] for r in rows if r["encoding"] == e)),
                                 -next(r["px_per_char"] for r in rows if r["encoding"] == e)))
    head = " | ".join(f"{e}/{c}/tp{n}" for e, c, n in settings)
    print("## Test CER by encoding\n")
    print(f"| encoding | chars/VLM token | {head} |")
    print("|---|---:|" + "---:|" * len(settings))
    for e in encs:
        r0 = next(r for r in rows if r["encoding"] == e)
        cells = []
        for s in settings:
            r = best.get((*s, e))
            cells.append("" if r is None else f"{100 * r['cer']:.2f}%")
        print(f"| {e} | {r0['chars_per_vlm_token']:.1f} | " + " | ".join(cells) + " |")

    for thr in (0.10, 0.03):
        print(f"\n## C@{100 * thr:g}% (chars per VLM token; text tokens ~ 4 chars/token)\n")
        print(f"| family | {head} |")
        print("|---|" + "---:|" * len(settings))
        for f in FAMILIES:
            cells = []
            for s in settings:
                sch = f"{f}|{s}"
                rs = [{**r, "scheme": sch} for r in rows if r["family"] == f and
                      (r["encoder"], r["cond"], r.get("train_pages", 200)) == s]
                t = CurveTable(rs, sch)
                if t.empty:
                    cells.append("n/a")
                    continue
                pv = p_at(rs, sch, thr)
                _, cens = crossing(t.x, t.mean(), thr, t.counts())
                cells.append("-" if pv is None else f"{'>=' if cens == 'left' else ''}{VLM_P2[s[0]] / pv:.1f}")
            print(f"| {f} | " + " | ".join(cells) + " |")

    print("\n## Informativeness guard: identity ceiling, same geometry (< 10% to interpret; same page count if run,"
          " else the largest page count run)\n")
    for s in settings:
        if s[0] in CEILING:
            for e in encs:
                r = best.get((*s, e))
                cs = sorted((k[2] != s[2], -k[2]) + (v,) for k, v in best.items()
                            if k[0] == CEILING[s[0]] and k[1] == "clean" and k[3] == e)
                c = cs[0][-1] if cs else None
                if r is None:
                    continue
                g = "no ceiling row" if c is None else f"ceiling {100 * c['cer']:.2f}% (tp{c['train_pages']})" + (
                    "" if c["cer"] < 0.10 else "  -> NOT interpretable")
                print(f"- {s[0]}/{s[1]}/tp{s[2]} {e}: {100 * r['cer']:.2f}% ({g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
