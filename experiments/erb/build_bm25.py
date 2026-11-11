"""BM25 ranking of the full EnterpriseRAG-Bench corpus for the 500 questions, deeper than the official top-10.

The official baseline ("BM25 + GPT-5.4" on the leaderboard) is OpenSearch BM25 over one field, title + "\\n\\n" +
content, with the standard analyzer (StandardTokenizer + lower case, no stop words), k1 = 1.2, b = 0.75, a `match`
query on the question, top 10. Its 10 ids per question are published (answers_bm25.jsonl), so the text arm uses them
as they are. The budget-matched page arm needs ranks 11-50, which only a re-run of the same retrieval gives: this
script reimplements Lucene's BM25 (Lucene >= 8 form, score = idf * tf / (tf + k1 * (1 - b + b * dl / avgdl)),
idf = ln(1 + (N - n + 0.5) / (n + 0.5)), document length stored lossily as Lucene's one-byte norm) and checks
itself against the published top-10 (overlap@10 is printed and saved). The final ranking of a question is the
official top-10 followed by this ranking's documents not already in it.

Only the question terms are indexed (a few thousand), so the pass is light: ~0.5 GB of postings, ~5-10 min on 4 cores.

    python3 experiments/erb/build_bm25.py [--workers 4] [--depth 60]
Output: results/erb/retrieval.jsonl (one row per question), data/erb/bm25_postings.npz (re-queryable cache).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from array import array
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import erb_common as E  # noqa: E402

K1, B = 1.2, 0.75
# UAX#29-like words: letters/digits/underscore runs, joined across ' and . between word characters (3.5, don't, e.g)
TOK = re.compile(r"\w+(?:['’.]\w+)*")
CACHE = E.ROOT / "data" / "erb" / "bm25_postings.npz"


def analyze(text: str) -> list[str]:
    return [t for t in TOK.findall(text.lower()) if len(t) <= 255]


# Lucene SmallFloat.intToByte4 / byte4ToInt: the lossy one-byte document length BM25 actually uses.
def _long_to_int4(i: int) -> int:
    nb = i.bit_length()
    if nb < 4:
        return i
    shift = nb - 4
    return ((i >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(i: int) -> int:
    bits, shift = i & 0x07, (i >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


_FREE = 255 - _long_to_int4(2 ** 31 - 1)
LENGTH_TABLE = np.array([b if b < _FREE else _FREE + _int4_to_long(b - _FREE) for b in range(256)], dtype=np.float64)


def norm_byte(n: int) -> int:
    return n if n < _FREE else _FREE + _long_to_int4(n - _FREE)


_Q: frozenset = frozenset()


def _init(q):
    global _Q
    _Q = q


def _scan(chunk: list[tuple[int, str]]):
    """Lengths of every document in the chunk and postings for the query terms only."""
    docs, dls, post = array("i"), array("i"), {}
    bad = 0
    for gid, rel in chunk:
        try:
            t, c = E.extract_document_content(json.loads((E.SOURCES / rel).read_text()))
        except Exception:
            bad += 1                                              # the official indexer skips these too
            continue
        toks = analyze(f"{t}\n\n{c}")
        docs.append(gid)
        dls.append(len(toks))
        for w, n in Counter(toks).items():
            if w in _Q:
                p = post.get(w)
                if p is None:
                    p = post[w] = (array("i"), array("i"))
                p[0].append(gid)
                p[1].append(n)
    return docs, dls, post, bad


def build(workers: int) -> dict:
    qs = E.questions()
    qterms = frozenset(w for q in qs for w in analyze(q["question"]))
    items = list(E.uuid_index().items())
    uuids = [u for u, _ in items]
    jobs = [(i, rel) for i, (_, rel) in enumerate(items)]
    chunks = [jobs[i: i + 2000] for i in range(0, len(jobs), 2000)]
    t0 = time.time()
    dl = np.full(len(items), -1, dtype=np.int64)
    post: dict[str, tuple[list, list]] = {}
    bad = 0
    with Pool(workers, initializer=_init, initargs=(qterms,)) as pool:
        for k, (docs, dls, p, b) in enumerate(pool.imap_unordered(_scan, chunks)):
            dl[np.frombuffer(docs, dtype=np.int32)] = np.frombuffer(dls, dtype=np.int32)
            bad += b
            for w, (d, n) in p.items():
                L = post.setdefault(w, ([], []))
                L[0].append(np.frombuffer(d, dtype=np.int32).copy())
                L[1].append(np.frombuffer(n, dtype=np.int32).copy())
            if k % 25 == 0:
                print(f"[bm25] {k + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)
    terms = sorted(post)
    arrs = {"uuids": np.array(uuids), "dl": dl, "terms": np.array(terms)}
    for i, w in enumerate(terms):
        arrs[f"d{i}"] = np.concatenate(post[w][0])
        arrs[f"n{i}"] = np.concatenate(post[w][1])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **arrs)
    print(f"[bm25] indexed {int((dl >= 0).sum()):,} docs ({bad} unparseable), {len(terms):,} query terms, "
          f"{time.time() - t0:.0f}s -> {CACHE}", flush=True)
    return dict(np.load(CACHE))


def rank(z: dict, depth: int) -> dict[str, list[tuple[str, float]]]:
    uuids, dl = z["uuids"], z["dl"]
    ok = dl >= 0
    N = int(ok.sum())
    avgdl = dl[ok].sum() / N
    lens = LENGTH_TABLE[np.array([norm_byte(int(x)) if x >= 0 else 0 for x in dl])]
    denom_base = K1 * (1 - B + B * lens / avgdl)
    tid = {w: i for i, w in enumerate(z["terms"])}
    out = {}
    for q in E.questions():
        s = np.zeros(len(uuids))
        for w, mult in Counter(analyze(q["question"])).items():   # repeated query terms are repeated clauses
            if w not in tid:
                continue
            d, n = z[f"d{tid[w]}"], z[f"n{tid[w]}"].astype(np.float64)
            idf = math.log(1 + (N - len(d) + 0.5) / (len(d) + 0.5))
            s[d] += mult * idf * n / (n + denom_base[d])
        s[~ok] = -1
        top = np.argpartition(-s, depth)[:depth]
        top = top[np.lexsort((top, -s[top]))]                     # score desc, then index order (Lucene: doc id)
        out[q["question_id"]] = [(str(uuids[i]), float(s[i])) for i in top]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--depth", type=int, default=60)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    z = dict(np.load(CACHE)) if CACHE.exists() and not a.rebuild else build(a.workers)
    ours = rank(z, a.depth)
    off = E.official_top10()
    E.OUT.mkdir(parents=True, exist_ok=True)
    ov, top1 = [], []
    with open(E.OUT / "retrieval.jsonl", "w") as f:
        for q in E.questions():
            qid = q["question_id"]
            o = off[qid]
            mine = [u for u, _ in ours[qid]]
            merged = o + [u for u in mine if u not in o]
            ov.append(len(set(o) & set(mine[:10])) / 10)
            top1.append(o[0] == mine[0])
            gold = set(q["expected_doc_ids"] or [])
            rec = {k: (len(gold & set(merged[:k])) / len(gold) if gold else None) for k in (10, 20, 30, 40, 50)}
            f.write(json.dumps({"question_id": qid, "type": q["question_type"], "official_top10": o,
                                "ours": ours[qid], "ranked": merged[:50], "overlap10": ov[-1],
                                "recall": rec}) + "\n")
    rows = [json.loads(l) for l in (E.OUT / "retrieval.jsonl").read_text().splitlines()]
    def mrec(k):
        v = [r["recall"][str(k)] if str(k) in r["recall"] else r["recall"][k] for r in rows]
        v = [x for x in v if x is not None]
        return sum(v) / len(v)
    print(f"[bm25] overlap@10 with the official top-10: mean {np.mean(ov):.3f}, exact 10/10 on "
          f"{sum(x == 1 for x in ov)}/{len(ov)}; same top-1 on {sum(top1)}/{len(top1)}")
    print("[bm25] recall of gold docs in the merged ranking: " +
          ", ".join(f"@{k} {mrec(k):.3f}" for k in (10, 20, 30, 40, 50)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
