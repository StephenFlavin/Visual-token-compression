"""RAG with page images (docs/RAG_PLAN.md; predictions committed before any call, 2026-11-11).

One single-shot Opus call per question and arm, no tools, low effort. Retrieval: BM25 over ~1,200-character passages.
Arms: closed (no passages), T5 (top-5 as text), P5 (same top-5 as pages at 88.2 px²/char), Pmatch (as many top-ranked
passages as fit in T5's estimated text tokens, as pages), T15 (top-15 as text), hybrid (top-3 as text + the next ranks
as pages within T5's estimated tokens).

Resumable: every finished call is appended to results/rag/runs.jsonl at once and successful responses are cached on
disk (vbpe.claude_eval, keyed on model + prompt + images); data preparation is deterministic, so a re-run rebuilds the
same prompts, skips finished (dataset, question, arm) rows and serves cached calls free. Failed calls (e.g. a usage
limit) are neither recorded nor cached; when most of a batch fails the run stops and prints the command to resume.

    python3 experiments/rag/run_rag.py --stages A B C        # A planted facts, B MultiHop-RAG, C HotpotQA
    python3 experiments/rag/run_rag.py --analyse
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "experiments" / "readhook")]
OUT = ROOT / "results" / "rag"
MODEL, EFFORT, BUDGET = "claude-opus-5-5", "low", 88.2
ARMS = ["closed", "T5", "P5", "Pmatch", "T15", "hybrid"]
CPT = 2.9                      # chars per Claude text token on English prose (measured in the QA tests)
P_IN, P_OUT, P_W, P_R = 4, 20, 5, 0.2   # $ per million tokens (Opus 5.5 list, fitted from CLI costs)

SYSTEM = ("You answer one question using the retrieved documents provided (as text, as page images, or both). Reply "
          "with the shortest exact answer only: a name, number, date, or yes / no, with no explanation. If the "
          "documents do not contain the answer, reply exactly: Insufficient information")
SYSTEM_CLOSED = ("You answer one question. Reply with the shortest exact answer only: a name, number, date, or yes / "
                 "no, with no explanation. If you do not know, reply exactly: Insufficient information")


# --------------------------------------------------------------------------------------------- retrieval
WORD = re.compile(r"[a-z0-9]+")
STOP = set("a an the of and or in on at to for by with from is are was were be been as that this which who what when "
           "where how did does do it its into than then there their they he she his her".split())


def toks(s: str) -> list[str]:
    return [w for w in WORD.findall(s.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.tf = [Counter(toks(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b
        self.post = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                self.post[w].append(i)

    def top(self, query: str, k: int) -> list[int]:
        sc = defaultdict(float)
        for w in set(toks(query)):
            for i in self.post.get(w, ()):
                f = self.tf[i][w]
                sc[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return [i for i, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))[:k]]


def passages_from(title: str, text: str, size: int = 1200) -> list[str]:
    """~size-char passages at sentence boundaries, each headed by its document title."""
    sents = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > size:
            out.append(cur)
            cur = ""
        cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return [f"[{title}] {p}" for p in out]


# --------------------------------------------------------------------------------------------- datasets
def ds_planted(n_per_corpus: int = 40) -> tuple[list[str], list[dict]]:
    """The deep-research corpora (invented facts planted in real Wikipedia text); questions answerable from a few
    passages: lookups (needle, number) and cross-source (conflict, chain, total)."""
    import claude_qa_research as R
    corp = R.split_corpora(R.articles())
    passages, qs = [], []
    for k, srcs in enumerate(corp):
        doc, src_txt, qq = R.build(srcs, seed=k)
        for (title, _), txt in zip(srcs, src_txt):
            passages += passages_from(title, txt.split("] ", 1)[1] if txt.startswith("[Source") else txt)
        rng = random.Random(k)
        want = {"needle": 10, "number": 10, "conflict": 7, "chain": 7, "total": 6}
        for kind, n in want.items():
            pool = [q for q in qq if q["kind"] == kind]
            for q in rng.sample(pool, min(n, len(pool)))[: n_per_corpus]:
                qs.append({"id": f"c{k}_{q['id']}", "question": q["question"], "answer": q["answer"], "type": kind,
                           "gold_spans": q["anchors"], "numeric": True})
    return passages, qs


def ds_multihop(n: int = 150) -> tuple[list[str], list[dict]]:
    from datasets import load_dataset
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    passages = []
    for a in corpus:
        passages += passages_from(f"{a['title']} ({a['source']}, {str(a['published_at'])[:10]})", a["body"])
    q = list(load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train"))
    by = defaultdict(list)
    for r in q:
        by[r["question_type"]].append(r)
    rng = random.Random(0)
    per = n // len(by)
    qs = []
    for t in sorted(by):
        for i, r in enumerate(rng.sample(by[t], per)):
            ev = r["evidence_list"] if isinstance(r["evidence_list"], list) else eval(r["evidence_list"])  # noqa: S307
            qs.append({"id": f"mh_{t}_{i}", "question": r["query"], "answer": r["answer"], "type": t,
                       "gold_spans": [e["fact"][:80] for e in ev if e.get("fact")], "numeric": False})
    return passages, qs


def ds_hotpot(n: int = 100, pool: int = 2000) -> tuple[list[str], list[dict]]:
    from datasets import load_dataset
    d = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = random.Random(0)
    idx = rng.sample(range(len(d)), pool)
    rows = [d[i] for i in idx]
    seen, passages = set(), []
    for r in rows:
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            if title not in seen:
                seen.add(title)
                passages.append(f"[{title}] " + " ".join(s.strip() for s in sents))
    hard = [r for r in rows if r["level"] == "hard"]
    qs = [{"id": f"hp_{r['id']}", "question": r["question"], "answer": r["answer"], "type": r["type"],
           "gold_titles": sorted(set(r["supporting_facts"]["title"])), "numeric": False} for r in hard[:n]]
    return passages, qs


# --------------------------------------------------------------------------------------------- contexts
def text_block(ps: list[str]) -> str:
    return "\n\n".join(f"Document {i + 1}: {p}" for i, p in enumerate(ps))


def page_images(ps: list[str], start: int = 1):
    from vbhook import render
    flow = " ".join(f"Document {i + start}: {' '.join(p.split())}" for i, p in enumerate(ps))
    return [pg.image for pg in render.render(flow, BUDGET)]


def page_tokens(ps: list[str], start: int = 1) -> int:
    from vbhook import render
    flow = " ".join(f"Document {i + start}: {' '.join(p.split())}" for i, p in enumerate(ps))
    return render.page_tokens(flow, BUDGET)


def build_job(arm: str, q: dict, ranked: list[str]) -> dict:
    t5_tok = len(text_block(ranked[:5])) / CPT
    qline = f"Question: {q['question']}"
    if arm == "closed":
        return {"images": [], "system": SYSTEM_CLOSED, "prompt": qline, "n_passages": 0, "n_text": 0, "n_pages": 0}
    if arm in ("T5", "T15"):
        k = 5 if arm == "T5" else 15
        return {"images": [], "system": SYSTEM, "prompt": f"Retrieved documents:\n\n{text_block(ranked[:k])}\n\n{qline}",
                "n_passages": len(ranked[:k]), "n_text": len(ranked[:k]), "n_pages": 0}
    if arm == "P5":
        imgs = page_images(ranked[:5])
        return {"images": imgs, "system": SYSTEM, "n_passages": len(ranked[:5]), "n_text": 0, "n_pages": len(imgs),
                "prompt": f"The retrieved documents are in the page image(s) above.\n\n{qline}"}
    if arm == "Pmatch":
        n = 5
        while n < min(15, len(ranked)) and page_tokens(ranked[: n + 1]) <= t5_tok:
            n += 1
        imgs = page_images(ranked[:n])
        return {"images": imgs, "system": SYSTEM, "n_passages": n, "n_text": 0, "n_pages": len(imgs),
                "prompt": f"The retrieved documents are in the page image(s) above.\n\n{qline}"}
    if arm == "hybrid":
        head = text_block(ranked[:3])
        rest_budget = t5_tok - len(head) / CPT
        m = 3
        while m < min(15, len(ranked)) and page_tokens(ranked[3: m + 1], start=4) <= rest_budget:
            m += 1
        imgs = page_images(ranked[3:m], start=4) if m > 3 else []
        return {"images": imgs, "system": SYSTEM, "n_passages": m, "n_text": 3, "n_pages": len(imgs),
                "prompt": (f"Retrieved documents 1-3:\n\n{head}\n\n" + (f"Documents 4-{m} are in the page image(s) "
                           "above.\n\n" if imgs else "") + qline)}
    raise ValueError(arm)


# --------------------------------------------------------------------------------------------- scoring
def norm(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return float(p == g)
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def score(q: dict, text: str) -> dict:
    pred = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    if q["numeric"]:
        import claude_qa_render as Q
        m = re.search(r"-?\d[\d,]*(?:\.\d+)?", pred)
        ok = bool(m) and Q.norm_num(m.group(0)) == q["answer"]
        return {"pred": pred, "em": ok, "f1": float(ok)}
    gold = q["answer"]
    if norm(gold).startswith("insufficient"):
        ok = norm(pred).startswith("insufficient")
        return {"pred": pred, "em": ok, "f1": float(ok)}
    return {"pred": pred, "em": norm(pred) == norm(gold), "f1": f1(pred, gold)}


def recall(q: dict, ranked: list[str], k: int) -> float:
    top = ranked[:k]
    if "gold_titles" in q:
        return sum(any(p.startswith(f"[{t}]") for p in top) for t in q["gold_titles"]) / max(1, len(q["gold_titles"]))
    spans = [" ".join(s.split())[:60].lower() for s in q.get("gold_spans", []) if s]
    flat = [" ".join(p.split()).lower() for p in top]
    return sum(any(s in p for p in flat) for s in spans) / max(1, len(spans))


# --------------------------------------------------------------------------------------------- run
STAGES = {"A": ("planted", ds_planted, 60), "B": ("multihop", ds_multihop, 150), "C": ("hotpot", ds_hotpot, 100)}


def run_stage(stage: str, a) -> bool:
    from vbpe import claude_eval as CE
    name, fn, n = STAGES[stage]
    passages, qs = fn()
    if name == "planted":
        qs = [q for i, q in enumerate(qs) if i % 2 == 0][:n]          # 60 of the 120, spread over the 3 corpora
    bm = BM25(passages)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = OUT / "runs.jsonl"
    done = {(r["dataset"], r["qid"], r["arm"]) for r in map(json.loads, runs.read_text().splitlines())} if runs.exists() else set()
    jobs = []
    for q in qs:
        ranked = [passages[i] for i in bm.top(q["question"], 15)]
        for arm in ARMS:
            if (name, q["id"], arm) in done:
                continue
            j = build_job(arm, q, ranked)
            jobs.append({"meta": {"q": q, "arm": arm, "rec": {k: j[k] for k in ("n_passages", "n_text", "n_pages")},
                                  "recall5": recall(q, ranked, 5), "recall15": recall(q, ranked, 15),
                                  "recall_used": recall(q, ranked, j["n_passages"])},
                         "images": j["images"], "system": j["system"], "prompt": j["prompt"], "model": MODEL,
                         "effort": EFFORT, "hires": True, "timeout": 600.0, "max_retries": 2})
    print(f"[rag] stage {stage} ({name}): {len(passages):,} passages, {len(qs)} questions, {len(jobs)} calls to do", flush=True)
    for b in range(0, len(jobs), a.batch):
        batch = jobs[b: b + a.batch]
        res = CE.transcribe_many(batch, workers=a.workers, progress=False)
        errs = 0
        for r in res:
            m = r["meta"]
            if r.get("error") or not (r.get("text") or "").strip():
                errs += 1
                continue                                          # not recorded: retried on the next run
            u = r.get("usage") or {}
            row = {"dataset": name, "qid": m["q"]["id"], "type": m["q"]["type"], "arm": m["arm"], **m["rec"],
                   "recall5": m["recall5"], "recall15": m["recall15"], "recall_used": m["recall_used"],
                   "answer": m["q"]["answer"], **score(m["q"], r.get("text", "")), "cached": r.get("cached"),
                   "input_tokens": r.get("input_tokens"), "output_tokens": r.get("output_tokens"),
                   "usage": u, "cost_usd": r.get("cost_usd"), "latency_s": r.get("latency_s")}
            with open(runs, "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"[rag] stage {stage}: {min(b + a.batch, len(jobs))}/{len(jobs)} calls, {errs} failed in this batch",
              flush=True)
        if errs >= max(3, len(batch) // 2):
            err = next((r.get("error") for r in res if r.get("error")), "empty answers")
            print(f"[rag] STOPPED: {errs}/{len(batch)} calls failed (last error: {str(err)[:300]}). Nothing failed was "
                  f"recorded or cached. Resume with: python3 experiments/rag/run_rag.py --stages "
                  f"{' '.join(a.stages[a.stages.index(stage):])}", flush=True)
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build retrieval and contexts, print stats, no calls")
    a = ap.parse_args()
    if a.analyse:
        from analyse_rag import main as am
        return am()
    if a.dry_run:
        for s in a.stages:
            name, fn, n = STAGES[s]
            passages, qs = fn()
            if name == "planted":
                qs = [q for i, q in enumerate(qs) if i % 2 == 0][:n]
            bm = BM25(passages)
            st = defaultdict(list)
            for q in qs[:20]:
                ranked = [passages[i] for i in bm.top(q["question"], 15)]
                st["recall5"].append(recall(q, ranked, 5))
                st["recall15"].append(recall(q, ranked, 15))
                for arm in ("T5", "Pmatch", "hybrid"):
                    j = build_job(arm, q, ranked)
                    st[f"{arm}_passages"].append(j["n_passages"])
                st["T5_tok"].append(len(text_block(ranked[:5])) / CPT)
                st["P5_tok"].append(page_tokens(ranked[:5]))
            print(s, name, len(passages), "passages", len(qs), "questions |",
                  {k: round(sum(v) / len(v), 2) for k, v in st.items()}, flush=True)
        return 0
    for s in a.stages:
        if not run_stage(s, a):
            return 2
    print("[rag] all stages done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
