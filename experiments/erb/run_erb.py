"""EnterpriseRAG-Bench answers from Opus 5.5: text against page images (docs/claude/ERB_PLAN.md).

One single-shot call per question and arm, no tools, effort low, the benchmark's own answer prompt
(erb_common.ANSWER_GEN_PROMPT) and context format. Arms:

  closed  no documents (the corpus is synthetic, so this should score near zero; its input also measures the
          fixed prompt, which Pmatch's budget subtracts, so it runs wherever Pmatch does)
  T10     the official BM25 top-10 as text: our counterpart of the leaderboard's "BM25 + GPT-5.4"
  P10     the same 10 documents, same formatting, as page images at 88.2 px²/char (2.9x)
  Pmatch  as many documents of the BM25 ranking (official top-10, then results/erb/retrieval.jsonl) as fit, as
          pages, in the context tokens T10 used for the same question (measured: T10 input - closed input), max 50

Resumable like experiments/rag/run_rag.py: each finished call is appended to results/erb/runs.jsonl at once and cached
on disk (vtc.claude_eval, keyed on model + prompt + images); a re-run skips finished (question, arm) rows. Failed
calls are not recorded or cached; when most of a batch fails (e.g. the usage limit) the run stops and prints how to
resume. Pmatch is built after T10 and closed exist for the question (it needs their measured token counts).

    python3 experiments/erb/run_erb.py --pilot --dry-run      # contexts, token and cost estimates, no calls
    python3 experiments/erb/run_erb.py --pilot                # 50 stratified questions, 4 arms
    python3 experiments/erb/run_erb.py --arms T10 P10 Pmatch  # all 500
    python3 experiments/erb/run_erb.py --export               # results/erb/answers_<arm>.jsonl for the evaluator
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import erb_common as E  # noqa: E402

sys.path[:0] = [str(E.ROOT), str(E.ROOT / "experiments" / "readhook")]
MODEL, EFFORT, BUDGET = "claude-opus-5-5", "low", 88.2
ARMS = ["closed", "T10", "P10", "Pmatch"]
CPT = 2.9                                  # chars per text token: used only for dry-run estimates
MAX_DOCS = 50
SYSTEM = "You are a helpful assistant."
RUNS = E.OUT / "runs.jsonl"
PILOT = E.OUT / "pilot_qids.json"


def cost(u: dict) -> float:
    return (u.get("input_tokens", 0) * 4 + u.get("output_tokens", 0) * 20 + u.get("cache_creation_input_tokens", 0) * 5
            + u.get("cache_read_input_tokens", 0) * 0.2) / 1e6


def pilot_ids(n: int = 50, seed: int = 0) -> list[str]:
    """Stratified by question type: at least 3 per type, the rest proportional; fixed once written."""
    if PILOT.exists():
        return json.loads(PILOT.read_text())
    by = defaultdict(list)
    for q in E.questions():
        by[q["question_type"]].append(q["question_id"])
    rng = random.Random(seed)
    total = sum(len(v) for v in by.values())
    quota = {t: max(3, round(n * len(v) / total)) for t, v in by.items()}
    while sum(quota.values()) > n:                                   # trim the largest strata
        quota[max(quota, key=quota.get)] -= 1
    ids = sorted(i for t, v in by.items() for i in rng.sample(v, quota[t]))
    E.OUT.mkdir(parents=True, exist_ok=True)
    PILOT.write_text(json.dumps(ids, indent=0))
    return ids


def ranking() -> dict[str, list[str]]:
    p = E.OUT / "retrieval.jsonl"
    if not p.exists():
        raise SystemExit("results/erb/retrieval.jsonl missing: run experiments/erb/build_bm25.py first")
    return {r["question_id"]: r["ranked"] for r in map(json.loads, p.read_text().splitlines())}


def flow(uuids: list[str]) -> str:
    return " ".join(E.format_context_documents(uuids).split())


def page_images(uuids: list[str]):
    from vtchook import render
    return [pg.image for pg in render.render(flow(uuids), BUDGET)]


def page_tokens(uuids: list[str]) -> int:
    from vtchook import render
    return render.page_tokens(flow(uuids), BUDGET)


def pages_note(n: int) -> str:
    return (f"[The {n} retrieved documents are shown in the page images above, in rank order. Each starts with "
            f"\"--- Document i (ID: ...) ---\" and its title; line breaks are shown as spaces.]")


def fit_docs(ranked: list[str], budget_tokens: float) -> int:
    """Largest n (10 <= n <= MAX_DOCS) whose pages cost <= budget_tokens."""
    n = 10
    while n < min(MAX_DOCS, len(ranked)) and page_tokens(ranked[: n + 1]) <= budget_tokens:
        n += 1
    return n


def build_job(arm: str, q: dict, ranked: list[str], budget: float | None) -> dict:
    if arm == "closed":
        docs, imgs, ctx = [], [], "(no documents were retrieved)"
    elif arm == "T10":
        docs, imgs = ranked[:10], []
        ctx = E.format_context_documents(docs)
    elif arm == "P10":
        docs = ranked[:10]
        imgs = page_images(docs)
        ctx = pages_note(len(docs))
    elif arm == "Pmatch":
        docs = ranked[: fit_docs(ranked, budget)]
        imgs = page_images(docs)
        ctx = pages_note(len(docs))
    else:
        raise ValueError(arm)
    return {"docs": docs, "images": imgs,
            "prompt": E.ANSWER_GEN_PROMPT.format(context_documents=ctx, question=q["question"])}


def gold_recall(q: dict, docs: list[str]) -> float | None:
    g = set(q.get("expected_doc_ids") or [])
    return len(g & set(docs)) / len(g) if g else None


def load_rows() -> dict[tuple[str, str], dict]:
    if not RUNS.exists():
        return {}
    return {(r["question_id"], r["arm"]): r for r in map(json.loads, RUNS.read_text().splitlines())}


def measured_budget(rows: dict, qid: str) -> float | None:
    t, c = rows.get((qid, "T10")), rows.get((qid, "closed"))
    return (t["input_tokens"] - c["input_tokens"]) if t and c else None


def run(qs: list[dict], arms: list[str], a) -> bool:
    from vtc import claude_eval as CE
    rk = ranking()
    E.OUT.mkdir(parents=True, exist_ok=True)
    for phase in (["closed", "T10", "P10"], ["Pmatch"]):          # Pmatch needs T10 and closed measured first
        rows = load_rows()
        jobs = []
        for q in qs:
            for arm in [x for x in phase if x in arms]:
                if (q["question_id"], arm) in rows:
                    continue
                budget = measured_budget(rows, q["question_id"]) if arm == "Pmatch" else None
                if arm == "Pmatch" and budget is None:
                    print(f"[erb] skip Pmatch {q['question_id']}: needs T10 and closed rows first", flush=True)
                    continue
                j = build_job(arm, q, rk[q["question_id"]], budget)
                jobs.append({"meta": {"q": q, "arm": arm, "docs": j["docs"], "budget": budget,
                                      "n_pages": len(j["images"])},
                             "images": j["images"], "system": SYSTEM, "prompt": j["prompt"], "model": MODEL,
                             "effort": EFFORT, "hires": True, "timeout": 600.0, "max_retries": 2})
        print(f"[erb] {'/'.join(phase)}: {len(jobs)} calls to do", flush=True)
        for b in range(0, len(jobs), a.batch):
            batch = jobs[b: b + a.batch]
            res = CE.transcribe_many(batch, workers=a.workers, progress=False)
            errs = 0
            for r in res:
                m = r["meta"]
                if r.get("error") or not (r.get("text") or "").strip():
                    errs += 1
                    continue
                u = r.get("usage") or {}
                q = m["q"]
                row = {"question_id": q["question_id"], "type": q["question_type"], "arm": m["arm"],
                       "document_ids": m["docs"], "n_docs": len(m["docs"]), "n_pages": m["n_pages"],
                       "budget_tokens": m["budget"], "gold_recall": gold_recall(q, m["docs"]),
                       "answer": r["text"].strip(), "cached": r.get("cached"), "input_tokens": r.get("input_tokens"),
                       "output_tokens": r.get("output_tokens"), "usage": u, "cost_list": cost(u),
                       "cost_usd": r.get("cost_usd"), "latency_s": r.get("latency_s"), "model": r.get("model")}
                with open(RUNS, "a") as f:
                    f.write(json.dumps(row) + "\n")
            print(f"[erb] {min(b + a.batch, len(jobs))}/{len(jobs)} calls, {errs} failed in this batch", flush=True)
            if errs >= max(3, len(batch) // 2):
                err = next((r.get("error") for r in res if r.get("error")), "empty answers")
                print(f"[erb] STOPPED: {errs}/{len(batch)} failed (last error: {str(err)[:300]}). Nothing failed was "
                      f"recorded or cached; re-run the same command to resume.", flush=True)
                return False
    return True


def dry_run(qs: list[dict], arms: list[str]) -> None:
    rk = ranking()
    s = defaultdict(list)
    for q in qs:
        r = rk[q["question_id"]]
        t10 = len(E.format_context_documents(r[:10]))
        est = t10 / CPT
        s["T10_chars"].append(t10)
        s["T10_tokens_est"].append(est)
        s["P10_page_tokens"].append(page_tokens(r[:10]))
        n = fit_docs(r, est)
        s["Pmatch_docs_est"].append(n)
        s["Pmatch_page_tokens"].append(page_tokens(r[:n]))
        s["recall_T10"].append(gold_recall(q, r[:10]))
        s["recall_Pmatch_est"].append(gold_recall(q, r[:n]))
    fix = 6000                                       # Claude Code CLI's fixed prompt per call, cache write
    out = 250                                        # answer tokens (estimate)
    per = {"closed": fix * 5e-6 + out * 2e-5,
           "T10": (fix + st.mean(s["T10_tokens_est"])) * 5e-6 + out * 2e-5,
           "P10": (fix + st.mean(s["P10_page_tokens"])) * 5e-6 + out * 2e-5,
           "Pmatch": (fix + st.mean(s["Pmatch_page_tokens"])) * 5e-6 + out * 2e-5}
    print(f"[erb] dry run on {len(qs)} questions (text tokens estimated at {CPT} chars/token):")
    for k, v in s.items():
        v = [x for x in v if x is not None]
        print(f"  {k:20s} mean {st.mean(v):10.2f}  median {st.median(v):10.2f}  max {max(v):10.2f}")
    tot = sum(per[x] for x in arms) * len(qs)
    print("  est. $ per call: " + ", ".join(f"{k} {v:.3f}" for k, v in per.items() if k in arms) +
          f"; answers for {len(qs)} questions x {len(arms)} arms: ~${tot:.0f} (list price)")


def export() -> None:
    rows = load_rows()
    for arm in ARMS:
        R = sorted((r for (q, x), r in rows.items() if x == arm), key=lambda r: r["question_id"])
        if R:
            with open(E.OUT / f"answers_{arm}.jsonl", "w") as f:
                for r in R:
                    f.write(json.dumps({"question_id": r["question_id"], "answer": r["answer"],
                                        "document_ids": r["document_ids"]}) + "\n")
            print(f"[erb] {E.OUT / f'answers_{arm}.jsonl'}: {len(R)} answers")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="the 50 stratified questions in results/erb/pilot_qids.json")
    ap.add_argument("--arms", nargs="+", default=None, help=f"default: all of {ARMS} with --pilot, else T10 P10 Pmatch "
                    "(+ closed, which Pmatch's budget needs)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    if a.export:
        export()
        return 0
    qs = E.questions()
    if a.pilot:
        keep = set(pilot_ids())
        qs = [q for q in qs if q["question_id"] in keep]
    qs = qs[: a.limit] if a.limit else qs
    arms = a.arms or ARMS
    if "Pmatch" in arms and "closed" not in arms:
        arms = ["closed"] + arms                    # its token budget is T10 - closed for the same question
    if a.dry_run:
        dry_run(qs, arms)
        return 0
    return 0 if run(qs, arms, a) else 2


if __name__ == "__main__":
    raise SystemExit(main())
