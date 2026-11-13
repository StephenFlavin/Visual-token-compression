"""Non-uniform resolution for text: long-document Q&A with the page density graded by position in the document.

DeepSeek-OCR proposes (untested) a memory that forgets: old context re-rendered at ever lower resolution, so recent
text is sharp and expensive and old text blurry and cheap. Here the v2 long-document Q&A suite (same 3 documents,
120 questions each, same system prompt and prompt, Opus 5.5 ``--effort high``, 1932-px pages) is given with the
document cut into three equal parts by characters, each part rendered at its own px²/char:

* ``recent_sharp``  first (oldest) third 58.8, middle 88.2, last (most recent) third 117.6;
* ``old_sharp``     the reverse (the control that separates position from density).

Pages stay in document order; every page holds text of one third only (tiers change at page boundaries). The
uniform 117.6 / 88.2 / 58.8 arms and text are the existing ``qav2_d{d}_high_b*`` rows (paired by question).

    python3 experiments/claude_qa/claude_qa_gradient.py check            # free: identity with v2, pages, tokens per tier
    python3 experiments/claude_qa/claude_qa_gradient.py run --yes        # the calls (resumable; rows appended per call)
    python3 experiments/claude_qa/claude_qa_gradient.py analyse          # works on whatever rows exist
Pre-registration and results: results/claude_qa/gradient/README.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude_qa"))
import analyse_claude_qa as A  # noqa: E402
import claude_qa_render as R  # noqa: E402
from vtc import config  # noqa: E402

OUT = config.RESULTS / "claude_qa" / "gradient"
QA = config.RESULTS / "claude_qa"
DOCS = [(0, 0), (1150, 1), (2480, 2)]          # (paragraph offset, seed) of qav2_d0/d1/d2
CHARS = 700000
PAGE_PX = 1932
ARMS = {"recent_sharp": (58.8, 88.2, 117.6), "old_sharp": (117.6, 88.2, 58.8)}
UNIFORM = {117.6: "b118", 88.2: "b88", 58.8: "b59"}
REPEATS = [0, 1]
CAP_USD = 12.0
EST_PER_CALL = 1.0       # list price, upper estimate: a 1932-px v2 call at high effort cost $0.13-0.73
SHARP = 117.6
DENSE = 58.8
LOOKUP = ("needle", "number")
SUBS = ("two_hop", "three_hop", "sum", "max", "count")
REFUSAL = re.compile(r"usage policy|acceptable use|violat|I can(?:no|')t help with|I won't|refus", re.I)


def salt(rep: int) -> str | None:
    return None if rep == 0 else f"gradient{rep}"


# --------------------------------------------------------------------------- document, tiers, pages
def build_doc(d: int) -> tuple[str, list[dict]]:
    """Document d of the v2 suite, exactly as ``claude_qa_render.py --chars 700000 --n-complex 40`` built it."""
    off, seed = DOCS[d]
    return R.make_questions(R.build_document(CHARS, off), 40, 40, seed, 40)


def anchor_spans(doc: str, q: dict) -> list[tuple[int, int]]:
    """Character spans of the facts a question needs. A number question's span runs from its quoted opening to the
    end of that sentence (so the number is inside it)."""
    anchors = q.get("anchors") or [q["fact"]]
    out = []
    for a in anchors:
        s = doc.find(a)
        assert s >= 0 and doc.count(a) == 1, (q["id"], a)
        e = s + len(a)
        if q["kind"] == "number":      # to the end of the sentence, and at least past the number asked for
            num = next((m for m in R.NUM.finditer(doc, e, e + 2000) if R.norm_num(m.group()) == q["answer"]), None)
            assert num is not None, q["id"]
            m = re.search(r"[.!?](?:\s|$)", doc[num.end():num.end() + 2000])
            e = num.end() + (m.end() if m else 0)
        out.append((s, e))
    return out


def split_thirds(doc: str, spans: list[tuple[int, int]]) -> list[int]:
    """[0, c1, c2, len(doc)]: two cuts at len/3 and 2 len/3, each moved to the next space, and past any fact span it
    would cut (a fact is never split across two densities)."""
    bounds = [0]
    for k in (1, 2):
        c = len(doc) * k // 3
        moved = True
        while moved:
            moved = False
            for s, e in spans:
                if s < c < e:
                    c, moved = e, True
            sp = doc.find(" ", c)
            c = len(doc) if sp < 0 else sp + 1
            moved = moved or any(s < c < e for s, e in spans)
        bounds.append(c)
    bounds.append(len(doc))
    return bounds


def tier_of(pos: int, bounds: list[int]) -> int:
    return 0 if pos < bounds[1] else (1 if pos < bounds[2] else 2)


def render_tiered(doc: str, bounds: list[int], budgets: tuple, render=None) -> tuple[list, list[dict]]:
    """Each third rendered on its own pages with ``render_pages`` at its density, in document order."""
    render = render or (lambda s, b: R.render_pages(s, b, PAGE_PX))
    pages, info = [], []
    for t in range(3):
        p, inf = render(doc[bounds[t]:bounds[t + 1]], budgets[t])
        pages += p
        info += [{**x, "tier": t, "budget": budgets[t]} for x in inf]
    return pages, info


def annotate(doc: str, qs: list[dict], bounds: list[int]) -> list[dict]:
    """Per question: the tier of every fact it needs (lookups have one)."""
    out = []
    for q in qs:
        tiers = [tier_of(s, bounds) for s, _ in anchor_spans(doc, q)]
        out.append({**q, "tiers": tiers, "tier": tiers[0] if len(tiers) == 1 else None})
    return out


def call_spec(d: int, arm: str, cache: dict | None = None) -> dict:
    cache = cache if cache is not None else {}
    if d not in cache:
        doc, qs = build_doc(d)
        spans = [sp for q in qs for sp in anchor_spans(doc, q)]
        bounds = split_thirds(doc, spans)
        cache[d] = {"doc": doc, "qs": annotate(doc, qs, bounds), "bounds": bounds}
    c = cache[d]
    if arm not in c:
        c[arm] = render_tiered(c["doc"], c["bounds"], ARMS[arm])
    pages, info = c[arm]
    return {"d": d, "arm": arm, "doc": c["doc"], "qs": c["qs"], "bounds": c["bounds"], "pages": pages, "info": info,
            "vt": sum(p["vision_tokens"] for p in info), "system": R.SYSTEM,
            "prompt": R.prompt_for("image", c["doc"], c["qs"], len(pages))}


# --------------------------------------------------------------------------- existing rows
def v2_rows(d: int) -> dict:
    """{'text': row, 117.6: row, 88.2: row, 58.8: row} from qav2_d{d}_high_b*."""
    out = {}
    for b, suf in UNIFORM.items():
        p = QA / f"qav2_d{d}_high_{suf}" / "runs.jsonl"
        for x in p.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if r["arm"] == "image":
                    out[b] = r
                elif r["arm"] == "text":
                    out.setdefault("text", r)
    return out


def load_rows() -> list[dict]:
    rows = {}
    p = OUT / "runs.jsonl"
    for x in p.read_text().splitlines() if p.exists() else []:
        if x.strip():
            r = json.loads(x)
            rows[(r["doc"], r["arm"], r["repeat"])] = r
    return [rows[k] for k in sorted(rows)]


def spent() -> float:
    p = OUT / "spend.jsonl"
    return sum(json.loads(x)["cost_usd"] for x in p.read_text().splitlines() if x.strip()) if p.exists() else 0.0


def est_per_call() -> float:
    """Upper estimate of one call's list price: $1.00, or 1.3x the dearest call so far if that is higher."""
    p = OUT / "spend.jsonl"
    costs = [json.loads(x)["cost_usd"] for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return max([EST_PER_CALL] + [1.3 * c for c in costs])


def make_row(spec: dict, rep: int, rec: dict) -> dict:
    qs = spec["qs"]
    ans = R.parse_answers(rec.get("text", ""))
    per = [{"id": q["id"], "kind": q["kind"], "sub": q.get("sub"), "pos": q["pos"], "answer": q["answer"],
            "tiers": q["tiers"], "tier": q["tier"],
            "densities": [ARMS[spec["arm"]][t] for t in q["tiers"]], "given": ans.get(q["id"]),
            "correct": ans.get(q["id"]) is not None and R.norm_num(ans[q["id"]]) == q["answer"]} for q in qs]
    incomplete = bool(rec.get("input_tokens") is not None and rec["input_tokens"] < 0.97 * spec["vt"])
    refused = bool(rec.get("stop_reason") == "refusal" or (rec.get("error") and REFUSAL.search(rec["error"]))
                   or (not ans and REFUSAL.search(rec.get("text", ""))))
    row = {"arm": spec["arm"], "doc": spec["d"], "repeat": rep, "cache_salt": salt(rep),
           "budgets": list(ARMS[spec["arm"]]), "bounds": spec["bounds"], "chars": len(spec["doc"]),
           "n_pages": len(spec["pages"]), "pages_per_tier": [sum(x["tier"] == t for x in spec["info"]) for t in range(3)],
           "image_tokens_formula": spec["vt"],
           "image_tokens_per_tier": [sum(x["vision_tokens"] for x in spec["info"] if x["tier"] == t) for t in range(3)],
           "effort": "high", "model": rec.get("model"), "input_tokens": rec.get("input_tokens"),
           "output_tokens": rec.get("output_tokens"), "usage": rec.get("usage"), "cost_usd": rec.get("cost_usd"),
           "latency_s": rec.get("latency_s"), "cached": rec.get("cached"), "error": rec.get("error"),
           "stop_reason": rec.get("stop_reason"), "cache_key": rec.get("cache_key"), "incomplete": incomplete,
           "refused": refused, "page_px": PAGE_PX, "n_questions": len(qs),
           "acc_all": float(np.mean([p["correct"] for p in per])),
           "acc": {k: float(np.mean([p["correct"] for p in per if p["kind"] == k])) for k in sorted({q["kind"] for q in qs})},
           "per_question": per, "raw": rec.get("text", "")}
    row["unanswered"], row["disengaged"] = A.engagement(row)
    return row


def todo_jobs(rows: list[dict], docs) -> list[tuple[int, str, int]]:
    have = {(r["doc"], r["arm"], r["repeat"]) for r in rows}
    return [(d, arm, rep) for rep in REPEATS for d in docs for arm in ARMS if (d, arm, rep) not in have]


# --------------------------------------------------------------------------- check (free)
def cmd_check(a) -> None:
    cache: dict = {}
    res = {}
    for d in a.docs:
        stored = json.loads((QA / f"qav2_d{d}_high_b88" / "questions.json").read_text())
        v2 = v2_rows(d)
        for arm in ARMS:
            s = call_spec(d, arm, cache)
            assert [(q["id"], q["question"], q["answer"]) for q in s["qs"]] == \
                   [(q["id"], q["question"], q["answer"]) for q in stored], f"d{d}: questions differ from v2"
            assert len(s["doc"]) == v2[88.2]["chars"], "document differs from v2"
            pt = [sum(x["tier"] == t for x in s["info"]) for t in range(3)]
            tt = [sum(x["vision_tokens"] for x in s["info"] if x["tier"] == t) for t in range(3)]
            ppc = [float(np.mean([x["px_per_char"] for x in s["info"] if x["tier"] == t])) for t in range(3)]
            ref = v2[88.2]["image_tokens_formula"]
            lk = [sum(q["kind"] in LOOKUP and q["tier"] == t for q in s["qs"]) for t in range(3)]
            res[f"d{d} {arm}"] = {"pages": len(s["pages"]), "pages_per_tier": pt, "tokens_per_tier": tt,
                                  "tokens": s["vt"], "uniform88_tokens": ref, "ratio_to_uniform88": s["vt"] / ref,
                                  "px_per_char_per_tier": ppc, "bounds": s["bounds"], "lookups_per_tier": lk}
            print(f"[check] d{d} {arm}: {len(s['pages'])} pages {pt}, image tokens {s['vt']:,} {tt} "
                  f"(uniform 88.2: {ref:,}; ratio {s['vt'] / ref:.3f}), px²/char per tier "
                  + ", ".join(f"{v:.1f}" for v in ppc) + f"; thirds at {s['bounds'][1:3]}; lookups per tier {lk}",
                  flush=True)
        uni = {b: v2[b]["image_tokens_formula"] for b in UNIFORM}
        print(f"[check] d{d} uniform image tokens (v2 rows): " + ", ".join(f"{b}: {v:,}" for b, v in uni.items()))
    jobs = todo_jobs(load_rows(), a.docs)
    print(f"[check] {len(jobs)} calls to make, at most ${est_per_call():.2f} each = ${len(jobs) * est_per_call():.2f}; "
          f"spent ${spent():.2f}; cap ${CAP_USD:.0f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dry_run.json").write_text(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- run
def cmd_run(a) -> None:
    from vtc import claude_eval
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = todo_jobs(load_rows(), a.docs)
    if a.limit:
        jobs = jobs[:a.limit]
    e = est_per_call()
    print(f"[run] {len(jobs)} calls, at most ${e:.2f} each; spent ${spent():.2f}; cap ${CAP_USD:.0f}",
          flush=True)
    if not a.yes or not jobs:
        print("[run] add --yes to make the calls" if jobs else "[run] nothing to do", flush=True)
        return
    cache: dict = {}
    specs = {(d, arm): call_spec(d, arm, cache) for d, arm, _ in jobs}
    lock = threading.Lock()
    state = {"inflight": 0.0, "stop": False}

    def one(job):
        d, arm, rep = job
        with lock:
            if state["stop"]:
                return
            if spent() + state["inflight"] + e > CAP_USD:
                print(f"[run] d{d} {arm} r{rep}: skipped, could cross the ${CAP_USD:.0f} cap", flush=True)
                return
            state["inflight"] += e
        spec = specs[(d, arm)]
        rec = claude_eval.transcribe(spec["pages"], spec["system"], spec["prompt"], effort="high", timeout=1800.0,
                                     hires=True, cache_salt=salt(rep))
        row = make_row(spec, rep, rec)
        with lock:
            state["inflight"] -= e
            if not rec.get("cached"):
                with open(OUT / "spend.jsonl", "a") as f:
                    f.write(json.dumps({"what": f"d{d} {arm} r{rep}", "cost_usd": rec.get("cost_usd", 0.0),
                                        "input_tokens": rec.get("input_tokens"),
                                        "output_tokens": rec.get("output_tokens"), "error": rec.get("error")}) + "\n")
            if row["error"] and not row["refused"]:
                if re.search(r"usage limit|limit reached|quota", row["error"], re.I):
                    state["stop"] = True
                print(f"[run] d{d} {arm} r{rep}: ERROR (not stored, rerun): {row['error'][:300]}", flush=True)
                return
            with open(OUT / "runs.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"[run] d{d} {arm} r{rep}: all {row['acc_all']:.3f} {row['acc']}; in {row['input_tokens']:,} "
              f"(formula {row['image_tokens_formula']:,}) out {row['output_tokens']:,} ${row['cost_usd']:.2f} "
              f"{row['latency_s']:.0f}s incomplete {row['incomplete']} refused {row['refused']} "
              f"unanswered {row['unanswered']} disengaged {row['disengaged']}; spent ${spent():.2f}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, jobs))
    print(f"[run] done; spent ${spent():.2f}", flush=True)


# --------------------------------------------------------------------------- analysis
def valid(r: dict) -> bool:
    return not (r.get("incomplete") or r.get("refused") or r.get("unanswered") or r.get("disengaged") or r.get("error"))


def acc(xs: list[bool]) -> dict:
    k, n = int(sum(xs)), len(xs)
    lo, hi = A.wilson(k, n)
    return {"k": k, "n": n, "acc": k / n if n else float("nan"), "lo": lo, "hi": hi}


def tiers_class(tiers: list[int], budgets: tuple) -> str:
    """'dense' if any fact sits in the 58.8 tier, else 'sharp'."""
    return "dense" if any(budgets[t] == DENSE for t in tiers) else "sharp"


def by_tier(per: list[dict], tier_of_q: dict) -> dict:
    """Lookup accuracy by the third their fact is in."""
    return {t: acc([p["correct"] for p in per if p["kind"] in LOOKUP and tier_of_q[p["id"]] == t]) for t in range(3)}


def analyse(rows: list[dict], v2: dict, qtiers: dict) -> dict:
    """rows: this test's rows; v2: {doc: v2_rows(doc)}; qtiers: {doc: {qid: [tiers]}} (from the document split).
    -> results dict (all numbers the README, summary and figure use)."""
    good = [r for r in rows if valid(r)]
    res = {"invalid": [(r["doc"], r["arm"], r["repeat"]) for r in rows if not valid(r)], "arms": {}, "tests": {},
           "calls": [{"doc": r["doc"], "arm": r["arm"], "repeat": r["repeat"], "acc_all": r["acc_all"],
                      "input_tokens": r["input_tokens"], "image_tokens_formula": r["image_tokens_formula"],
                      "output_tokens": r["output_tokens"], "cost_usd": r["cost_usd"], "latency_s": r["latency_s"],
                      "valid": valid(r)} for r in rows]}
    docs = sorted(v2)

    # pooled per-question outcomes of every arm: list of (doc, per_question entry)
    pool: dict = {}
    for d in docs:
        for k in ("text", 117.6, 88.2, 58.8):
            if k in v2[d]:
                name = "text" if k == "text" else f"uniform {k}"
                pool.setdefault(name, []).append((d, v2[d][k]))
    for r in good:
        pool.setdefault(r["arm"], []).append((r["doc"], r))
    budgets = {**{f"uniform {b}": (b, b, b) for b in UNIFORM}, **ARMS, "text": None}

    for name, entries in pool.items():
        per = [(d, p) for d, r in entries for p in r["per_question"]]
        x = {"calls": len(entries), "docs": sorted({d for d, _ in entries}),
             "input_tokens_mean": float(np.mean([r["input_tokens"] for _, r in entries])),
             "cost_mean": float(np.mean([r["cost_usd"] for _, r in entries])),
             "all": acc([p["correct"] for _, p in per]),
             "lookup": acc([p["correct"] for _, p in per if p["kind"] in LOOKUP]),
             "complex": acc([p["correct"] for _, p in per if p["kind"] == "complex"]),
             "per_call": [r["acc_all"] for _, r in entries],
             "complex_by_sub": {s: acc([p["correct"] for _, p in per if p.get("sub") == s]) for s in SUBS}}
        x["lookup_by_tier"] = {t: acc([p["correct"] for d, p in per if p["kind"] in LOOKUP and qtiers[d][p["id"]][0] == t])
                               for t in range(3)}
        if budgets[name] is not None:
            b = budgets[name]
            x["budgets"] = list(b)
            for arm_b, lab in ((ARMS["recent_sharp"], "complex_by_recent_sharp_tiers"),
                               (ARMS["old_sharp"], "complex_by_old_sharp_tiers")):
                x[lab] = {c: acc([p["correct"] for d, p in per if p["kind"] == "complex"
                                  and tiers_class(qtiers[d][p["id"]], arm_b) == c]) for c in ("sharp", "dense")}
        res["arms"][name] = x

    # paired tests on the first call of each document (repeat 0); and on every call (each paired with the same v2 row)
    def pairs_of(arm: str, ref: str, reps) -> list:
        out = []
        for r in good:
            if r["arm"] != arm or r["repeat"] not in reps:
                continue
            refrow = v2[r["doc"]]["text" if ref == "text" else float(ref.split()[1])]
            rq = {p["id"]: p["correct"] for p in refrow["per_question"]}
            out += [(r["doc"], bool(rq[p["id"]]), bool(p["correct"])) for p in r["per_question"]]
        return out
    for arm in ARMS:
        for ref in ("text", "uniform 88.2", "uniform 117.6", "uniform 58.8"):
            for lab, reps in (("first_call", (0,)), ("all_calls", tuple(REPEATS))):
                pr = pairs_of(arm, ref, reps)
                if not pr:
                    continue
                diff = A.paired_diff_ci(pr)                    # ref - arm
                n10, n01, pm = A.mcnemar([x[1] for x in pr], [x[2] for x in pr])
                res["tests"][f"{arm} vs {ref} ({lab})"] = {
                    "n": len(pr), "arm_minus_ref": -diff[0], "lo": -diff[2], "hi": -diff[1],
                    "ref_only": n10, "arm_only": n01, "p": pm}

    # pre-registered predictions (point estimates pooled over every valid call of each arm)
    ar = res["arms"]
    pred = {}
    if all(a in ar for a in ARMS) and all(f"uniform {b}" in ar for b in UNIFORM):
        n1 = []
        for arm, b in ARMS.items():
            for t in range(3):
                mix, uni = ar[arm]["lookup_by_tier"][t]["acc"], ar[f"uniform {b[t]}"]["lookup_by_tier"][t]["acc"]
                n1.append({"arm": arm, "tier": t, "density": b[t], "mixed": mix, "uniform": uni, "diff": mix - uni})
        pred["N1"] = {"pass": all(abs(x["diff"]) <= 0.03 for x in n1), "cells": n1}
        dn2 = ar["recent_sharp"]["all"]["acc"] - ar["old_sharp"]["all"]["acc"]
        pred["N2"] = {"pass": abs(dn2) <= 0.03, "recent_minus_old": dn2}
        lo59, hi88 = ar["uniform 58.8"]["all"]["acc"], ar["uniform 88.2"]["all"]["acc"]
        pred["N3"] = {"pass": all(lo59 <= ar[a]["all"]["acc"] < hi88 for a in ARMS),
                      "uniform58.8": lo59, "uniform88.2": hi88, **{a: ar[a]["all"]["acc"] for a in ARMS}}
        n4 = {}
        for arm in ARMS:
            lab = f"complex_by_{arm}_tiers"
            u = ar["uniform 117.6"][lab]
            m = ar[arm][lab]
            n4[arm] = {c: {"mixed": m[c]["acc"], "uniform117.6": u[c]["acc"], "n_mixed": m[c]["n"],
                           "n_uniform": u[c]["n"], "diff": m[c]["acc"] - u[c]["acc"]} for c in ("sharp", "dense")}
        pred["N4"] = {"pass": all(n4[a]["dense"]["diff"] <= -0.10 and n4[a]["sharp"]["diff"] > -0.10 for a in ARMS),
                      "arms": n4}
    res["predictions"] = pred
    return res


def fmt(x: dict) -> str:
    return f"{100 * x['acc']:.1f}% ({x['k']}/{x['n']})" if x["n"] else "-"


def summary_md(res: dict) -> str:
    ar = res["arms"]
    order = [a for a in ("text", "uniform 117.6", "uniform 88.2", "uniform 58.8", "recent_sharp", "old_sharp") if a in ar]
    md = ["# Graded resolution: summary (generated by `claude_qa_gradient.py analyse`)", "",
          "Pooled over every valid call of each arm (uniform arms and text: the v2 rows, one call per document).", "",
          "| arm | densities (thirds, old to new) | calls | input tokens / call | cost / call | all | lookups | multi-step |",
          "|---|---|---|---|---|---|---|---|"]
    for a in order:
        x = ar[a]
        dens = "text" if a == "text" else " / ".join(f"{b:g}" for b in x["budgets"])
        md.append(f"| {a} | {dens} | {x['calls']} | {x['input_tokens_mean']:,.0f} | ${x['cost_mean']:.2f} | "
                  f"{fmt(x['all'])} | {fmt(x['lookup'])} | {fmt(x['complex'])} |")
    md += ["", "Lookups (needle + number questions) by the third their fact is in (density drawn at in brackets):", "",
           "| arm | first third | middle third | last third |", "|---|---|---|---|"]
    for a in order:
        x = ar[a]
        cells = []
        for t in range(3):
            c = fmt(x["lookup_by_tier"][t])
            cells.append(c + (f" [{x['budgets'][t]:g}]" if "budgets" in x else ""))
        md.append(f"| {a} | " + " | ".join(cells) + " |")
    md += ["", "Multi-step questions by the tiers their facts touch. 'dense' = at least one fact in a third that the "
           "named mixed arm draws at 58.8; 'sharp' = all facts in its 88.2 / 117.6 thirds. Each arm is scored on the "
           "same question split:", "",
           "| arm | split by recent_sharp: sharp | dense | split by old_sharp: sharp | dense |", "|---|---|---|---|---|"]
    for a in order:
        x = ar[a]
        if "complex_by_recent_sharp_tiers" not in x:
            continue
        r, o = x["complex_by_recent_sharp_tiers"], x["complex_by_old_sharp_tiers"]
        md.append(f"| {a} | {fmt(r['sharp'])} | {fmt(r['dense'])} | {fmt(o['sharp'])} | {fmt(o['dense'])} |")
    md += ["", "Multi-step questions by type:", "", "| arm | " + " | ".join(SUBS) + " |", "|---|" + "---|" * len(SUBS)]
    for a in order:
        md.append(f"| {a} | " + " | ".join(fmt(ar[a]["complex_by_sub"][s]) for s in SUBS) + " |")
    md += ["", "Paired tests (arm - reference, points; Newcombe 95% interval; exact McNemar on discordant pairs):", "",
           "| comparison | pairs | arm - ref | 95% interval | ref-only right | arm-only right | p |", "|---|---|---|---|---|---|---|"]
    for k, t in res["tests"].items():
        md.append(f"| {k} | {t['n']} | {100 * t['arm_minus_ref']:+.1f} | {100 * t['lo']:+.1f} to {100 * t['hi']:+.1f} | "
                  f"{t['ref_only']} | {t['arm_only']} | {t['p']:.3g} |")
    md += ["", "Calls:", "", "| doc | arm | repeat | all | input tokens | image-token formula | output | cost | s | valid |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["calls"]:
        md.append(f"| d{c['doc']} | {c['arm']} | {c['repeat']} | {100 * c['acc_all']:.1f}% | {c['input_tokens']:,} | "
                  f"{c['image_tokens_formula']:,} | {c['output_tokens']:,} | ${c['cost_usd']:.2f} | {c['latency_s']:.0f} | "
                  f"{c['valid']} |")
    md += ["", f"Invalid calls: {res['invalid'] or 'none'}", "", "## Pre-registered predictions", ""]
    p = res["predictions"]
    if p:
        md.append("* **N1 " + ("PASS" if p["N1"]["pass"] else "FAIL") + "** (each third within 3 points of the uniform arm "
                  "at its density): " + "; ".join(f"{c['arm']} third {c['tier'] + 1} @{c['density']:g}: "
                                                   f"{100 * c['mixed']:.1f} vs {100 * c['uniform']:.1f} ({100 * c['diff']:+.1f})"
                                                   for c in p["N1"]["cells"]))
        md.append(f"* **N2 {'PASS' if p['N2']['pass'] else 'FAIL'}** (recent_sharp - old_sharp overall within 3 points): "
                  f"{100 * p['N2']['recent_minus_old']:+.1f}")
        md.append(f"* **N3 {'PASS' if p['N3']['pass'] else 'FAIL'}** (both mixed arms in [uniform 58.8, uniform 88.2)): "
                  f"58.8 {100 * p['N3']['uniform58.8']:.1f}, 88.2 {100 * p['N3']['uniform88.2']:.1f}, "
                  + ", ".join(f"{a} {100 * p['N3'][a]:.1f}" for a in ARMS))
        md.append(f"* **N4 {'PASS' if p['N4']['pass'] else 'FAIL'}** (multi-step with a 58.8 fact loses >= 10 points "
                  "against uniform 117.6 on the same questions; all-sharp ones do not): " + "; ".join(
                      f"{a}: dense {100 * v['dense']['mixed']:.1f} vs {100 * v['dense']['uniform117.6']:.1f} "
                      f"({100 * v['dense']['diff']:+.1f}, n={v['dense']['n_mixed']}), sharp {100 * v['sharp']['mixed']:.1f} vs "
                      f"{100 * v['sharp']['uniform117.6']:.1f} ({100 * v['sharp']['diff']:+.1f}, n={v['sharp']['n_mixed']})"
                      for a, v in p["N4"]["arms"].items()))
    else:
        md.append("Not judged yet: rows for every arm are needed.")
    return "\n".join(md) + "\n"


def question_tiers(docs) -> dict:
    out = {}
    for d in docs:
        doc, qs = build_doc(d)
        bounds = split_thirds(doc, [sp for q in qs for sp in anchor_spans(doc, q)])
        out[d] = {q["id"]: q["tiers"] for q in annotate(doc, qs, bounds)}
    return out


def cmd_analyse(a) -> None:
    rows = load_rows()
    v2 = {d: v2_rows(d) for d in a.docs}
    tiers_file = OUT / "question_tiers.json"
    if tiers_file.exists():
        qt = {int(k): v for k, v in json.loads(tiers_file.read_text()).items()}
    else:
        qt = question_tiers(a.docs)
        OUT.mkdir(parents=True, exist_ok=True)
        tiers_file.write_text(json.dumps(qt, indent=1))
    res = analyse(rows, v2, qt)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    md = summary_md(res)
    (OUT / "summary.md").write_text(md)
    print(md)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--docs", type=int, nargs="+", default=[0, 1, 2])
        if name == "run":
            p.add_argument("--yes", action="store_true")
            p.add_argument("--workers", type=int, default=6)
            p.add_argument("--limit", type=int, default=0, help="make at most this many calls")
    a = ap.parse_args(argv)
    {"check": cmd_check, "run": cmd_run, "analyse": cmd_analyse}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
