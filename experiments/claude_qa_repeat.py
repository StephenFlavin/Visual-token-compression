"""Repeat calls on wording-sensitive questions: is the 117.6 px²/char (2.2x) pages' small-word loss in Test A
call-to-call variation, or a real weakness of 2.2x pages?

Test A (``claude_qa_prune.py``) made one call per arm and document. Here the same calls are repeated, byte for byte
(same 3 documents, 160 questions, prompts, page images, model and ``--effort high``): 4 more calls per document for
``page118`` and for ``page88`` (5 per document and density with Test A's) and 1 more per document for ``text`` (2 with
Test A's). A repeat differs from Test A's call only in the response cache key (``claude_eval``'s ``cache_salt`` =
the repeat index); the request is unchanged. Calls are the unit of analysis (answers of one type are correlated within
a call): per density, the mean small-word accuracy over the 15 calls and a bootstrap over whole calls within document,
against text.

    python3 experiments/claude_qa_repeat.py check            # free: identity with Test A, cost estimate
    python3 experiments/claude_qa_repeat.py locate           # free: where each small-word sentence sits on the pages
    python3 experiments/claude_qa_repeat.py run --yes        # the 27 calls (resumable; rows appended per call)
    python3 experiments/claude_qa_repeat.py analyse
Pre-registration and results: results/claude_qa/repeat/README.md.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import claude_qa_prune as P  # noqa: E402
import claude_qa_render as R  # noqa: E402
from vbpe import config  # noqa: E402

OUT = config.RESULTS / "claude_qa" / "repeat"
ARMS = {"page118": 117.6, "page88": 88.2, "text": None}
NEW_REPEATS = {"page118": [1, 2, 3, 4], "page88": [1, 2, 3, 4], "text": [1]}
CAP_USD = 25.0
MARGIN = 0.02            # 2 points
N_BOOT = 10000


def salt(rep: int) -> str | None:
    return None if rep == 0 else f"repeat{rep}"


# --------------------------------------------------------------------------- calls (byte-identical to Test A)
def call_spec(d: int, arm: str, cache: dict | None = None) -> dict:
    """Everything Test A sent for (document, arm): images, system prompt, prompt. ``cache`` reuses rendered pages."""
    cache = cache if cache is not None else {}
    if d not in cache:
        cache[d] = {"bundle": P.build_doc(d)}
    doc, qs = cache[d]["bundle"]
    if arm not in cache[d]:
        cache[d][arm] = R.render_pages(doc, ARMS[arm], P.PAGE_PX) if ARMS[arm] else ([], [])
    pages, info = cache[d][arm]
    return {"d": d, "arm": arm, "doc": doc, "qs": qs, "pages": pages, "info": info,
            "vt": sum(p["vision_tokens"] for p in info), "system": R.SYSTEM,
            "prompt": P.prompt(arm, doc, qs, len(pages))}


def transcribe_kwargs(spec: dict, rep: int) -> dict:
    return {"images": spec["pages"], "system": spec["system"], "prompt": spec["prompt"], "effort": "high",
            "timeout": 1800.0, "hires": bool(spec["pages"]), "cache_salt": salt(rep)}


def test_a_rows() -> dict:
    """{(doc, arm): Test A's row} for the three arms repeated here."""
    out = {}
    for d in range(3):
        p = P.OUT / f"d{d}" / "runs.jsonl"
        for x in p.read_text().splitlines() if p.exists() else []:
            if x.strip():
                r = json.loads(x)
                if r["arm"] in ARMS:
                    out[(d, r["arm"])] = r
    return out


def load_rows() -> list[dict]:
    """All calls, Test A's as repeat 0, new ones from runs.jsonl (the last row per (doc, arm, repeat) wins)."""
    rows = {}
    for (d, arm), r in test_a_rows().items():
        rows[(d, arm, 0)] = {**r, "repeat": 0, "source": "test_a"}
    p = OUT / "runs.jsonl"
    for x in p.read_text().splitlines() if p.exists() else []:
        if x.strip():
            r = json.loads(x)
            rows[(r["doc"], r["arm"], r["repeat"])] = r
    return [rows[k] for k in sorted(rows)]


def spent() -> float:
    p = OUT / "spend.jsonl"
    return sum(json.loads(x)["cost_usd"] for x in p.read_text().splitlines() if x.strip()) if p.exists() else 0.0


def make_row(spec: dict, rep: int, rec: dict) -> dict:
    """One row per call, the fields of Test A's rows plus repeat and cache salt."""
    from analyse_claude_qa import engagement
    qs = spec["qs"]
    ans = P.parse_answers(rec.get("text", ""))
    per = [{"id": q["id"], "kind": q["kind"], "sub": q.get("sub"), "pos": q["pos"], "answer": q["answer"],
            "lure": q.get("lure"), "given": ans.get(q["id"]),
            "correct": ans.get(q["id"]) is not None and R.norm_num(ans[q["id"]]) == q["answer"]} for q in qs]
    incomplete = bool(spec["pages"] and rec.get("input_tokens") is not None
                      and rec["input_tokens"] < 0.97 * spec["vt"])
    refused = bool(rec.get("stop_reason") == "refusal" or (rec.get("error") and P.REFUSAL.search(rec["error"]))
                   or (not ans and P.REFUSAL.search(rec.get("text", ""))))
    row = {"arm": spec["arm"], "doc": spec["d"], "repeat": rep, "cache_salt": salt(rep), "source": "repeat",
           "chars": len(spec["doc"]), "n_pages": len(spec["pages"]), "image_tokens_formula": spec["vt"],
           "effort": "high", "model": rec.get("model"), "input_tokens": rec.get("input_tokens"),
           "output_tokens": rec.get("output_tokens"), "usage": rec.get("usage"), "cost_usd": rec.get("cost_usd"),
           "latency_s": rec.get("latency_s"), "cached": rec.get("cached"), "error": rec.get("error"),
           "stop_reason": rec.get("stop_reason"), "cache_key": rec.get("cache_key"), "incomplete": incomplete,
           "refused": refused, "page_px": P.PAGE_PX if spec["pages"] else 0, "n_questions": len(qs),
           "acc_all": float(np.mean([p["correct"] for p in per])),
           "acc": {k: float(np.mean([p["correct"] for p in per if p["kind"] == k])) for k in sorted({q["kind"] for q in qs})},
           "per_question": per, "raw": rec.get("text", "")}
    row["unanswered"], row["disengaged"] = engagement(row)
    return row


def todo_jobs(rows: list[dict], docs) -> list[tuple[int, str, int]]:
    have = {(r["doc"], r["arm"], r["repeat"]) for r in rows}
    return [(d, arm, rep) for d in docs for arm, reps in NEW_REPEATS.items() for rep in reps if (d, arm, rep) not in have]


def estimate(jobs, ta: dict) -> float:
    """List price per job = Test A's measured cost of the same call (x 1.1 for longer thinking)."""
    return sum(1.1 * ta[(d, arm)]["cost_usd"] for d, arm, _ in jobs)


# --------------------------------------------------------------------------- free checks
def cmd_check(a) -> None:
    """Questions, prompts and page images are identical to Test A's: the unsalted cache key of each call must be
    the key under which Test A's response is stored, and the rendered pages must equal Test A's saved PNGs."""
    from PIL import Image

    from vbpe import claude_eval
    ta = test_a_rows()
    cache: dict = {}
    for d in a.docs:
        stored = json.loads((P.OUT / f"d{d}" / "questions.json").read_text())
        for arm in ARMS:
            s = call_spec(d, arm, cache)
            assert [q["question"] for q in s["qs"]] == [q["question"] for q in stored], (d, arm)
            pngs = [claude_eval.png_bytes(im) for im in s["pages"]]
            key = claude_eval.cache_key(claude_eval.DEFAULT_MODEL, s["system"], s["prompt"], pngs, "high")
            hit = (claude_eval.CACHE_DIR / f"{key}.json").exists()
            same_px = True
            for i, im in enumerate(s["pages"]):
                f = P.OUT / f"d{d}" / "images" / arm / f"page_{i:02d}.png"
                same_px &= f.exists() and np.array_equal(np.asarray(Image.open(f)), np.asarray(im))
            print(f"[check] d{d} {arm}: {len(s['pages'])} pages, {s['vt']:,} image tokens (Test A row: "
                  f"{ta[(d, arm)]['image_tokens_formula']:,}), Test A's response cached under this key: {hit}, "
                  f"pages equal Test A's PNGs: {same_px}", flush=True)
            assert hit and same_px and s["vt"] == ta[(d, arm)]["image_tokens_formula"], "not identical to Test A"
    rows = load_rows()
    jobs = todo_jobs(rows, a.docs)
    est = estimate(jobs, ta)
    print(f"[check] {len(jobs)} calls to make, estimate ${est:.2f} (Test A's cost per call x 1.1); spent here "
          f"${spent():.2f}, cap ${CAP_USD:.0f}", flush=True)


def page_lines(doc: str, budget: float) -> list[list[str]]:
    """The text of every rendered line, per page, exactly as ``render_pages`` lays it out (same chunks and em)."""
    from vbpe import metrics
    from vbpe.layout import LayoutParams, fits
    from vbpe.schemes import get_scheme
    pages, info = R.render_pages(doc, budget, P.PAGE_PX)
    sch = get_scheme("latin")
    max_chars = int(P.PAGE_PX * P.PAGE_PX / budget * 0.97)
    while True:                       # the chunk size render_pages ended with
        chunks = R.chunk(doc, max_chars)
        if [len(metrics.normalize(c)) for c in chunks] == [x["chars"] for x in info]:
            break
        assert len(chunks) <= len(info) + 5 and max_chars > 1000, "could not reproduce the page chunks"
        max_chars = int(max_chars * 0.95)
    out = []
    for c, x in zip(chunks, info):
        lines = fits(sch.encode(c), x["em"], x["W"], x["H"], LayoutParams())
        assert lines is not None and len(lines) == x["lines"]
        out.append(["".join(u.source for u in ln) for ln in lines])
    return out


def locate(lines: list[list[str]], sentence: str) -> dict:
    """Where ``sentence`` sits: page, first/last line, lines on that page, split across lines / pages."""
    flat, where = [], []
    for pi, pg in enumerate(lines):
        for li, t in enumerate(pg):
            for _ in t:
                where.append((pi, li))
            flat.append(t)
            where.append((pi, li))     # a virtual separator after each line
            flat.append("\n")
    s = "".join(flat)
    pat = r"\s*".join(re.escape(w) for w in sentence.split())
    m = re.search(pat, s)
    if not m:
        return {"found": False}
    (p0, l0), (p1, l1) = where[m.start()], where[m.end() - 1]
    return {"found": True, "page": p0, "line": l0, "end_page": p1, "end_line": l1, "lines_on_page": len(lines[p0]),
            "split_page": p0 != p1, "n_lines": (l1 - l0 + 1) if p0 == p1 else None,
            "top": l0 == 0, "bottom": (p1 == p0 and l1 == len(lines[p0]) - 1) or p0 != p1}


def cmd_locate(a) -> None:
    res = {}
    for d in a.docs:
        doc, qs = P.build_doc(d)
        for arm in ("page118", "page88"):
            lines = page_lines(doc, ARMS[arm])
            for q in qs:
                if q["kind"] == "small":
                    res[f"d{d} {arm} {q['id']}"] = {"doc": d, "arm": arm, "id": q["id"], "sub": q["sub"],
                                                   **locate(lines, q["anchors"][0])}
            ps = [v for v in res.values() if v["doc"] == d and v["arm"] == arm and v["sub"] == "passive"]
            for v in ps:
                print(f"[locate] d{d} {arm} {v['id']}: page {v['page']} line {v['line']}/{v['lines_on_page']} "
                      f"lines {v['n_lines']} split_page {v['split_page']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "locations.json").write_text(json.dumps(res, indent=1))
    found = [v for v in res.values() if v["found"]]
    print(f"[locate] {len(found)}/{len(res)} sentences found; split across pages: "
          f"{sum(v['split_page'] for v in found)}; on 2+ lines: {sum((v['n_lines'] or 2) > 1 for v in found)}")


# --------------------------------------------------------------------------- run
def cmd_run(a) -> None:
    from vbpe import claude_eval
    OUT.mkdir(parents=True, exist_ok=True)
    ta = test_a_rows()
    jobs = todo_jobs(load_rows(), a.docs)
    if a.limit:
        jobs = jobs[:a.limit]
    est = estimate(jobs, ta)
    print(f"[run] {len(jobs)} calls, estimate ${est:.2f}; spent here ${spent():.2f}; cap ${CAP_USD:.0f}", flush=True)
    if spent() + est > CAP_USD:
        print("[run] estimate would cross the cap: stopping", flush=True)
        return
    if not a.yes or not jobs:
        print("[run] add --yes to make the calls" if jobs else "[run] nothing to do", flush=True)
        return
    cache: dict = {}
    specs = {(d, arm): call_spec(d, arm, cache) for d, arm, _ in jobs}
    lock = threading.Lock()
    inflight = {"est": 0.0}

    def one(job):
        d, arm, rep = job
        e = 1.1 * ta[(d, arm)]["cost_usd"]
        with lock:
            if spent() + inflight["est"] + e > CAP_USD:
                print(f"[run] d{d} {arm} r{rep}: skipped, would cross the cap", flush=True)
                return
            inflight["est"] += e
        spec = specs[(d, arm)]
        rec = claude_eval.transcribe(**transcribe_kwargs(spec, rep))
        row = make_row(spec, rep, rec)
        with lock:
            inflight["est"] -= e
            if not rec.get("cached"):
                with open(OUT / "spend.jsonl", "a") as f:
                    f.write(json.dumps({"what": f"d{d} {arm} r{rep}", "cost_usd": rec.get("cost_usd", 0.0),
                                        "input_tokens": rec.get("input_tokens"),
                                        "output_tokens": rec.get("output_tokens"), "error": rec.get("error")}) + "\n")
            if row["error"] and not row["refused"]:
                print(f"[run] d{d} {arm} r{rep}: ERROR (not stored, rerun): {row['error'][:300]}", flush=True)
                return
            with open(OUT / "runs.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        sm = [p for p in row["per_question"] if p["kind"] == "small"]
        print(f"[run] d{d} {arm} r{rep}: all {row['acc_all']:.3f} small {np.mean([p['correct'] for p in sm]):.3f} "
              f"passive {sum(p['correct'] for p in sm if p['sub'] == 'passive')}/5; in {row['input_tokens']:,} "
              f"(formula {row['image_tokens_formula']:,}) out {row['output_tokens']:,} ${row['cost_usd']:.2f} "
              f"{row['latency_s']:.0f}s incomplete {row['incomplete']} refused {row['refused']} "
              f"unanswered {row['unanswered']} disengaged {row['disengaged']}; spent ${spent():.2f}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, jobs))
    print(f"[run] done; spent here ${spent():.2f}", flush=True)


# --------------------------------------------------------------------------- analysis
def valid(r: dict) -> bool:
    return not (r.get("incomplete") or r.get("refused") or r.get("unanswered") or r.get("disengaged") or r.get("error"))


def call_score(r: dict, group: str) -> float:
    f = {"small": lambda p: p["kind"] == "small", "original": lambda p: p["kind"] != "small",
         "all": lambda p: True}[group]
    return float(np.mean([p["correct"] for p in r["per_question"] if f(p)]))


def boot_diff(arm: dict, ref: dict, n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """arm, ref: {doc: [per-call scores]}. Mean over the arm's calls minus mean over the reference's calls (each
    document weighted by its number of calls), and a 95% percentile interval from resampling whole calls with
    replacement within each document, independently for the two arms."""
    docs = sorted(set(arm) & set(ref))
    a = [np.asarray(arm[d], float) for d in docs]
    b = [np.asarray(ref[d], float) for d in docs]
    point = float(np.concatenate(a).mean() - np.concatenate(b).mean())
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for i in range(n):
        x = np.concatenate([v[rng.integers(0, len(v), len(v))] for v in a]).mean()
        y = np.concatenate([v[rng.integers(0, len(v), len(v))] for v in b]).mean()
        bs[i] = x - y
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def verdict(point: float, lo: float, hi: float, margin: float = MARGIN) -> str:
    """Pre-registered: loss if the interval excludes 0 and the mean loss is >= margin; no loss shown if the worst
    case (lower end) is within the margin; otherwise not settled."""
    if hi < 0 and point <= -margin:
        return "loss"
    if lo >= -margin:
        return "no loss shown"
    return "not settled"


def cmd_analyse(a) -> None:
    rows = load_rows()
    good = [r for r in rows if valid(r)]
    bad = [r for r in rows if not valid(r)]
    locs = json.loads((OUT / "locations.json").read_text()) if (OUT / "locations.json").exists() else {}
    res, md = {"invalid": [(r["doc"], r["arm"], r["repeat"]) for r in bad]}, []
    by = {}
    for r in good:
        by.setdefault(r["arm"], {}).setdefault(r["doc"], []).append(r)
    for arm in ARMS:
        calls = by.get(arm, {})
        x = {"n_calls": sum(len(v) for v in calls.values()),
             "input_tokens": float(np.mean([r["input_tokens"] for v in calls.values() for r in v])),
             "cost_new": float(sum(r["cost_usd"] for v in calls.values() for r in v if r["source"] == "repeat")),
             "per_call": {str(d): [{"repeat": r["repeat"], "small": call_score(r, "small"),
                                    "original": call_score(r, "original"),
                                    "passive_right": sum(p["correct"] for p in r["per_question"] if p["sub"] == "passive"),
                                    "small_wrong": [(p["id"], p["sub"], p["given"], p["answer"], p["lure"])
                                                    for p in r["per_question"] if p["kind"] == "small" and not p["correct"]],
                                    "orig_wrong": [(p["id"], p["kind"], p["given"], p["answer"])
                                                   for p in r["per_question"] if p["kind"] != "small" and not p["correct"]]}
                                   for r in sorted(v, key=lambda r: r["repeat"])] for d, v in sorted(calls.items())}}
        allc = [r for v in calls.values() for r in v]
        x["all_passives_wrong"] = sum(all(not p["correct"] for p in r["per_question"] if p["sub"] == "passive")
                                      for r in allc)
        smalls = [p for r in allc for p in r["per_question"] if p["kind"] == "small"]
        x["by_sub"] = {s: float(np.mean([p["correct"] for p in smalls if p["sub"] == s]))
                       for s in sorted({p["sub"] for p in smalls})}
        x["by_kind"] = {k: float(np.mean([p["correct"] for r in allc for p in r["per_question"] if p["kind"] == k]))
                        for k in ("needle", "number", "complex", "small")}
        x["lure_share"] = float(np.mean([p["given"] is not None and R.norm_num(p["given"]) == p["lure"] for p in smalls]))
        x["wrong_is_lure"] = (float(np.mean([R.norm_num(p["given"] or "") == p["lure"] for p in smalls if not p["correct"]]))
                              if any(not p["correct"] for p in smalls) else None)
        for g in ("small", "original", "all"):
            sc = {d: [call_score(r, g) for r in v] for d, v in calls.items()}
            x[f"mean_{g}"] = float(np.mean([s for v in sc.values() for s in v]))
            x[f"sd_{g}"] = float(np.std([s for v in sc.values() for s in v], ddof=1)) if x["n_calls"] > 1 else None
            if arm != "text" and "text" in by:
                tsc = {d: [call_score(r, g) for r in v] for d, v in by["text"].items()}
                pt, lo, hi = boot_diff(sc, tsc)
                x[f"diff_{g}"] = {"point": pt, "lo": lo, "hi": hi, "verdict": verdict(pt, lo, hi)}
        # passive answers by where the sentence sits (top / bottom line of a page, split over lines or pages)
        if locs and arm != "text":
            loc_rows = []
            for r in allc:
                for p in r["per_question"]:
                    if p["sub"] == "passive":
                        lo_ = locs.get(f"d{r['doc']} {arm} {p['id']}", {})
                        loc_rows.append((lo_, p["correct"]))
            x["passive_by_place"] = {
                k: (float(np.mean([c for lo_, c in loc_rows if f(lo_)])) if any(f(lo_) for lo_, _ in loc_rows) else None,
                    sum(f(lo_) for lo_, _ in loc_rows))
                for k, f in {"split_page": lambda v: v.get("split_page"), "top_line": lambda v: v.get("top"),
                             "bottom_line": lambda v: v.get("bottom"),
                             "one_line": lambda v: v.get("n_lines") == 1,
                             "two_plus_lines": lambda v: (v.get("n_lines") or 2) > 1}.items()}
        res[arm] = x
    # same-call agreement: does a call's passive outcome follow the call (all or none) or the question?
    for arm in ("page118", "page88"):
        if arm in res:
            pc = [c["passive_right"] for v in res[arm]["per_call"].values() for c in v]
            res[arm]["passive_right_hist"] = {k: pc.count(k) for k in range(6)}
    out = OUT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    md.append("| density | calls | input tokens (mean) | small-word mean | − text (95% CI) | verdict | calls with "
              "every passive wrong | original 120 mean | − text (95% CI) | verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        if arm not in res:
            continue
        x = res[arm]
        dd = lambda g: (f"{100 * x[f'diff_{g}']['point']:+.1f} ({100 * x[f'diff_{g}']['lo']:+.1f} to "  # noqa: E731
                        f"{100 * x[f'diff_{g}']['hi']:+.1f})" if f"diff_{g}" in x else "(reference)")
        vv = lambda g: x[f"diff_{g}"]["verdict"] if f"diff_{g}" in x else ""  # noqa: E731
        md.append(f"| {arm} | {x['n_calls']} | {x['input_tokens']:,.0f} | {x['mean_small']:.1%} | {dd('small')} | "
                  f"{vv('small')} | {x['all_passives_wrong']} of {x['n_calls']} | {x['mean_original']:.1%} | "
                  f"{dd('original')} | {vv('original')} |")
    md.append("")
    md.append("Per-call small-word accuracy (document: calls in repeat order, Test A's first; passives right of 5 "
              "in brackets):")
    md.append("")
    for arm in ARMS:
        if arm in res:
            md.append(f"* {arm}: " + "; ".join(
                f"d{d} " + ", ".join(f"{c['small']:.1%} ({c['passive_right']})" for c in v)
                for d, v in res[arm]["per_call"].items()))
    md.append("")
    md.append("Per-call original-120 accuracy:")
    md.append("")
    for arm in ARMS:
        if arm in res:
            md.append(f"* {arm}: " + "; ".join(f"d{d} " + ", ".join(f"{c['original']:.1%}" for c in v)
                                                for d, v in res[arm]["per_call"].items()))
    md.append("")
    subs = sorted({s for x in res.values() if isinstance(x, dict) and "by_sub" in x for s in x["by_sub"]})
    md.append("| density | " + " | ".join(subs) + " | lure answers (of small-word) |")
    md.append("|---|" + "---|" * (len(subs) + 1))
    for arm in ARMS:
        if arm in res:
            md.append(f"| {arm} | " + " | ".join(f"{res[arm]['by_sub'].get(s, float('nan')):.0%}" for s in subs)
                      + f" | {res[arm]['lure_share']:.1%} |")
    md.append("")
    for arm in ("page118", "page88"):
        if arm in res and "passive_by_place" in res[arm]:
            md.append(f"* {arm} passives right, by place: " + ", ".join(
                f"{k} {v[0]:.0%} (n={v[1]})" if v[0] is not None else f"{k} - (n=0)"
                for k, v in res[arm]["passive_by_place"].items())
                + f"; calls by passives right (0..5): {res[arm]['passive_right_hist']}")
    md.append(f"* invalid calls: {res['invalid'] or 'none'}")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "locate", "run", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--docs", type=int, nargs="+", default=[0, 1, 2])
        if name == "run":
            p.add_argument("--yes", action="store_true")
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--limit", type=int, default=0, help="make at most this many calls")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"check": cmd_check, "locate": cmd_locate, "run": cmd_run, "analyse": cmd_analyse}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
