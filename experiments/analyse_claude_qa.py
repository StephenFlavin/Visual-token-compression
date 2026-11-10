"""Pooled analysis of the Claude QA v2 runs (results/claude_qa/summary.md, "v2: pre-registered").

Each run dir holds one document's rows (text, none, image at one px²/char) for one effort. Image arms are paired
with the same document's text arm (same effort) question by question. Pooled over documents, per (effort, budget):
accuracy per question kind (Wilson 95% CI), the paired difference text - image (Newcombe's score interval for
paired proportions, 95%), exact McNemar on the discordant pairs, and the pre-registered verdicts.
Arms whose input tokens fall short of the image-token formula (pages dropped) are excluded and listed.

    python3 experiments/analyse_claude_qa.py results/claude_qa/qav2_*
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


MARGIN = 0.02      # pre-registered: "no measurable loss" = upper 95% bound of (text - image) < 2 points


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.0
    return n10, n01, p


def paired_diff_ci(pairs: list[tuple[str, bool, bool]]) -> tuple[float, float, float]:
    """pairs: (doc, text correct, image correct) -> (text - image, 95% CI) by Newcombe's score method for paired
    proportions (method 10). Unlike a bootstrap it stays informative with no discordant pairs: 0 of 100 gives an
    upper bound of ~3.7%, 0 of 360 ~1.1%."""
    n = len(pairs)
    a = sum(t and i for _, t, i in pairs)
    b = sum(t and not i for _, t, i in pairs)
    c = sum(i and not t for _, t, i in pairs)
    d = n - a - b - c
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / den if den else 0.0
    delta = p1 - p2
    lo = delta - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = delta + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return float(delta), float(lo), float(hi)


def load(dirs: list[Path]) -> tuple[dict, list]:
    """-> {(effort, budget): [(doc, text_row, image_row)]}, none rows by effort, excluded arms."""
    arms = defaultdict(list)
    excluded = []
    for d in dirs:
        rows = [json.loads(x) for x in (d / "runs.jsonl").read_text().splitlines() if x.strip()]
        by = {r["arm"]: r for r in rows}
        doc = d.name
        if "image" in by and "text" in by:
            im = by["image"]
            key = (im["effort"], im["budget"])
            if im.get("incomplete"):
                excluded.append((d.name, im["input_tokens"], im["image_tokens_formula"]))
            else:
                arms[key].append((doc.split("_b")[0], by["text"], im))
        if "none" in by:
            arms[(by["none"]["effort"], "none")].append((doc.split("_b")[0], None, by["none"]))
    return arms, excluded


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/claude_qa/analysis_v2")
    a = ap.parse_args(argv)
    arms, excluded = load([Path(x) for x in a.dirs if (Path(x) / "runs.jsonl").exists()])
    md = ["# Claude QA v2: pooled over documents", "",
          f"Lossless (pre-registered) = upper 95% bound of paired (text - image) accuracy < {MARGIN:.0%}.", ""]
    res = {}
    for key in sorted(arms, key=lambda k: (k[0], str(k[1]))):
        effort, budget = key
        entries = arms[key]
        if budget == "none":
            per = [p for _, _, r in entries for p in r["per_question"]]
            res[f"{effort}/none"] = {"acc": sum(p["correct"] for p in per) / len(per), "n": len(per)}
            continue
        pairs, cpairs, kinds = [], [], defaultdict(lambda: [0, 0, 0])   # kind -> [text correct, image correct, n]
        tok_t = tok_i = 0
        for doc, t, i in entries:
            tq = {p["id"]: p for p in t["per_question"]}
            tok_t += t["input_tokens"]
            tok_i += i["input_tokens"]
            for p in i["per_question"]:
                q = tq[p["id"]]
                pairs.append((doc, bool(q["correct"]), bool(p["correct"])))
                if p["kind"] == "complex":
                    cpairs.append(pairs[-1])
                for k in {"all", p["kind"], p.get("sub") or p["kind"]}:
                    kinds[k][0] += q["correct"]
                    kinds[k][1] += p["correct"]
                    kinds[k][2] += 1
        d = paired_diff_ci(pairs)
        n10, n01, pm = mcnemar([x[1] for x in pairs], [x[2] for x in pairs])
        dc = paired_diff_ci(cpairs) if cpairs else (float("nan"),) * 3
        r = {"effort": effort, "budget": budget, "docs": len(entries), "n": len(pairs),
             "tokens_text": tok_t, "tokens_image": tok_i, "saving": tok_t / tok_i,
             "text_minus_image": d, "complex_text_minus_image": dc, "mcnemar": [n10, n01, pm],
             "lossless": d[2] < MARGIN, "kinds": {k: v for k, v in kinds.items()}}
        res[f"{effort}/{budget}"] = r
        md += [f"## effort {effort}, image at {budget} px²/char: {len(entries)} documents, {len(pairs)} paired questions, "
               f"{r['saving']:.2f}x fewer input tokens", "",
               f"* text - image: {d[0]:+.3f} (95% CI {d[1]:+.3f} to {d[2]:+.3f}); McNemar text-only {n10}, image-only "
               f"{n01}, p = {pm:.3g}; **{'LOSSLESS' if r['lossless'] else 'loss not excluded'}** (margin {MARGIN:.0%})",
               f"* complex only: text - image {dc[0]:+.3f} (95% CI {dc[1]:+.3f} to {dc[2]:+.3f})", "",
               "| kind | n | text | image |", "|---|---|---|---|"]
        for k in ["all", "needle", "number", "complex", "two_hop", "three_hop", "sum", "max", "count"]:
            if k in kinds:
                tc, ic, n = kinds[k]
                lo_t, hi_t = wilson(tc, n)
                lo_i, hi_i = wilson(ic, n)
                md.append(f"| {k} | {n} | {tc / n:.1%} ({lo_t:.0%}-{hi_t:.0%}) | {ic / n:.1%} ({lo_i:.0%}-{hi_i:.0%}) |")
        md.append("")
    for k, v in res.items():
        if k.endswith("/none"):
            md.append(f"* no document, effort {k.split('/')[0]}: {v['acc']:.1%} of {v['n']} questions")
    if excluded:
        md += ["", "Excluded (pages dropped): " + "; ".join(f"{n} ({t:,} of {f:,.0f} tokens)" for n, t, f in excluded)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
