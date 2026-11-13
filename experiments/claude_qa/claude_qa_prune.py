"""Test A (docs/background/REVIEW_2026-11-11.md): can extractive pruning of the *text* replace page images, or stack with them?

Same 3 documents as the v2 Claude QA runs (``qav2_d0/1/2``: 700k chars, offsets 0 / 1150 / 2480, seeds 0 / 1 / 2) plus
40 planted "small-word" facts per document whose numeric answer is right only if function words are read (negation,
"all but", passive, comparatives, before/after, from/to, "instead of"); each such sentence holds a second number that
answers the question under the wrong reading. Pruner: LLMLingua-2 (xlm-roberta-large, MeetingBank), run once per
document; its per-word keep probabilities are cached, so any keep rate is a cheap re-threshold.

Arms (Opus, effort high, one call per arm and document, all 160 questions in one call):
  none, text, page118 / page88 / page59 (pages at 117.6 / 88.2 / 58.8 px²/char), ptext22 (pruned text, tokens matched to
  page118), ptext15 (lightly pruned text, ~1.5x fewer tokens than text), ppage15 (ptext15's document as 117.6 pages).

    python3 experiments/claude_qa/claude_qa_prune.py build             # documents, questions, pruning probabilities (CPU, cached)
    python3 experiments/claude_qa/claude_qa_prune.py check             # free survival checks + token / cost estimate
    python3 experiments/claude_qa/claude_qa_prune.py calibrate --yes   # small unscored calls: Claude tokens per char of pruned text
    python3 experiments/claude_qa/claude_qa_prune.py run --docs 0 --yes
    python3 experiments/claude_qa/claude_qa_prune.py analyse
Pre-registration and results: results/claude_qa/prune/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude_qa"))
import claude_qa_render as R  # noqa: E402
from vbpe import config  # noqa: E402

OUT = config.RESULTS / "claude_qa" / "prune"
CACHE = config.RESULTS / "claude_qa" / "prune" / "cache"          # git-ignored (results/**/cache/)
DOCS = [(0, 0), (1150, 1), (2480, 2)]                            # (paragraph offset, seed) of qav2_d0/d1/d2
CHARS = 700000
PRUNER = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
PAGE_PX = 1932
BUDGETS = {"page118": 117.6, "page88": 88.2, "page59": 58.8}
ARMS = ["none", "text", "page118", "page88", "page59", "ptext22", "ptext15", "ppage15"]
NOTE = "Words have been removed from this document to shorten it."
CAP_USD = 25.0
PRICE_IN, PRICE_OUT = 5.0e-6, 20.0e-6        # Opus 5.5 list price: input as a cache write (1.25 x $4/M), output $20/M

# --------------------------------------------------------------------------- small-word facts
# (sub, fact template, question template, function words the answer depends on)
SMALL = [
    ("negation", "The museum at {place} did not receive {lure} maps from {who}; it received {ans}.",
     "How many maps did the museum at {place} receive from {who}?", ["not"]),
    ("negated_count", "Of the {total} coins that {who} catalogued at {place}, {k} were not silver.",
     "How many of the coins that {who} catalogued at {place} were silver?", ["not"]),
    ("all_but", "{who} planted {total} pear trees at {place}, and all but {k} of them died in the first winter.",
     "How many of the pear trees that {who} planted at {place} died in the first winter?", ["all", "but"]),
    ("passive", "At the horse fair in {place}, {who} was paid {n1} crowns by {who2}, and {who2} was paid {n2} crowns "
                "by {who3}.",
     "How many crowns did {who2} pay at the horse fair in {place}?", ["was", "by"]),
    ("comparative", "The choir of {place} has {n1} members, {d} {dir} than the choir of {place2}.",
     "How many members does the choir of {place2} have?", ["{dir}", "than"]),
    ("before_after", "{w1} the flood, the granary at {place} held {n1} sacks of rye; {w2} it, the granary held {n2}.",
     "How many sacks of rye did the granary at {place} hold {when} the flood?", ["{w1l}", "{w2}"]),
    ("from_to", "{n1} crates of slate were shipped from {place} to {place2}, and {n2} crates from {place2} to {place}.",
     "How many crates of slate were shipped to {dest}?", ["from", "to"]),
    ("instead_of", "The ferry at {place} carried {ans} passengers on its first crossing, instead of the {lure} that "
                   "{who} had planned.",
     "How many passengers did the ferry at {place} carry on its first crossing?", ["instead", "of"]),
]


def _two(rng, lo=30, hi=990) -> tuple[int, int]:
    while True:
        a, b = (int(x) for x in rng.integers(lo, hi, 2))
        if abs(a - b) >= 20:
            return a, b


def small_fact(sub_i: int, rng, names) -> dict:
    """One planted small-word fact. ``names()`` -> a fresh (person, place) pair. -> question dict (with the fact as
    its anchor, the answer, the lure = the answer under the wrong reading, and the function words it depends on)."""
    sub, fact, q, crit = SMALL[sub_i]
    who, place = names()
    v = {"who": who, "place": place}
    if sub in ("negation", "instead_of"):
        v["ans"], v["lure"] = _two(rng)
        ans, lure = v["ans"], v["lure"]
    elif sub in ("negated_count", "all_but"):
        v["total"] = int(rng.integers(200, 960))
        v["k"] = int(rng.integers(12, v["total"] // 3))
        ans, lure = v["total"] - v["k"], v["k"]
    elif sub == "passive":
        v["who2"], v["who3"] = names()[0], names()[0]
        v["n1"], v["n2"] = _two(rng)
        ans, lure = v["n1"], v["n2"]
    elif sub == "comparative":
        v["place2"] = names()[1]
        v["n1"] = int(rng.integers(150, 900))
        v["d"] = int(rng.integers(11, 120))
        v["dir"] = ["fewer", "more"][int(rng.integers(2))]
        ans, lure = (v["n1"] + v["d"], v["n1"] - v["d"]) if v["dir"] == "fewer" else (v["n1"] - v["d"], v["n1"] + v["d"])
    elif sub == "before_after":
        v["n1"], v["n2"] = _two(rng)
        v["w1"], v["w2"] = ["Before", "after"] if rng.integers(2) else ["After", "before"]
        v["w1l"] = v["w1"].lower()
        v["when"] = [v["w1l"], v["w2"]][int(rng.integers(2))]
        ans, lure = (v["n1"], v["n2"]) if v["when"] == v["w1l"] else (v["n2"], v["n1"])
    elif sub == "from_to":
        v["place2"] = names()[1]
        v["n1"], v["n2"] = _two(rng)
        v["dest"] = [v["place"], v["place2"]][int(rng.integers(2))]
        ans, lure = (v["n1"], v["n2"]) if v["dest"] == v["place2"] else (v["n2"], v["n1"])
    else:
        raise ValueError(sub)
    text = fact.format(**v)
    return {"kind": "small", "sub": sub, "answer": str(ans), "lure": str(lure), "anchors": [text],
            "question": q.format(**v), "critical": [c.format(**v) for c in crit]}


def make_small(n: int, n_paras: int, seed: int, taken_names: str) -> list[tuple[int, dict]]:
    """``n`` small-word facts (n/8 per type, shuffled), placed before evenly spaced paragraphs half-way between the
    needles' positions. Own RNG stream ([seed, 2]); names never occur in ``taken_names`` (the original document).
    -> [(paragraph index, question)]."""
    rng = np.random.default_rng([seed, 2])
    used: set = set()

    def names():
        while True:
            who, place = R.invented(rng, used)
            if who not in taken_names and place not in taken_names and who.split()[1] not in taken_names:
                return who, place
    subs = [i % len(SMALL) for i in range(n)]
    rng.shuffle(subs)
    step = (n_paras - 1) / (n + 1)
    at = (np.linspace(0, n_paras - 1, n + 2)[1:-1] + step / 2).astype(int)
    out = []
    for k, (i, s) in enumerate(zip(at, subs)):
        q = small_fact(int(s), rng, names)
        q["id"] = f"W{k + 1:02d}"
        out.append((int(i), q))
    return out


def build_doc(d: int, n_small: int = 40) -> tuple[str, list[dict]]:
    """Document d of the v2 suite with the small-word facts added. The original 120 questions (ids, wording, answers,
    anchors) are those of qav2_d{d}; only their ``pos`` moves a little."""
    off, seed = DOCS[d]
    paras = R.build_document(CHARS, off)
    doc0, _ = R.make_questions(paras, 40, 40, seed, 40)
    small = make_small(n_small, len(paras), seed, doc0)
    paras2 = list(paras)
    for i, q in small:
        paras2[i] = q["anchors"][0] + " " + paras2[i]
    doc, qs = R.make_questions(paras2, 40, 40, seed, 40)
    for _, q in small:
        assert doc.count(q["anchors"][0]) == 1
        q["pos"] = doc.find(q["anchors"][0]) / len(doc)
        qs.append(q)
    return doc, qs


def verify_original(d: int, qs: list[dict]) -> None:
    stored = json.loads((config.RESULTS / "claude_qa" / f"qav2_d{d}_high_b118" / "questions.json").read_text())
    key = lambda q: (q["id"], q["question"], q["answer"], q.get("anchors"))  # noqa: E731
    assert [key(q) for q in qs[:120]] == [key(q) for q in stored], f"document {d}: original questions changed"


# --------------------------------------------------------------------------- pruning (LLMLingua-2, cached)
def load_compressor():
    """LLMLingua-2 with a stand-in for its OpenAI tokenizer (tiktoken's file host is refused here): token counts use
    the pruner's own xlm-roberta tokenizer. In LLMLingua-2 that tokenizer only weights words in the per-chunk keep
    threshold and reports counts."""
    import tiktoken
    import torch
    torch.set_num_threads(4)
    torch.set_flush_denormal(True)

    class _Stub:
        tok = None

        def encode(self, text):
            return _Stub.tok.tokenize(text.replace("▁", " ")) or [text]
    orig = tiktoken.encoding_for_model
    tiktoken.encoding_for_model = lambda name: _Stub()
    try:
        from llmlingua import PromptCompressor
        pc = PromptCompressor(model_name=PRUNER, use_llmlingua2=True, device_map="cpu")
    finally:
        tiktoken.encoding_for_model = orig
    _Stub.tok = pc.tokenizer
    return pc


def word_weight(pc, word: str) -> int:
    return len(pc.oai_tokenizer.encode(word))


def word_probs(pc, text: str, log=print) -> list[dict]:
    """LLMLingua-2's chunking and per-word keep probabilities (mean over a word's tokens, digit tokens forced to 1),
    exactly as compress_prompt_llmlingua2 computes them. -> [{"words", "probs", "w"}] per 512-token chunk."""
    import time

    import torch
    import torch.nn.functional as F
    from llmlingua.utils import TokenClfDataset
    chunks = pc._PromptCompressor__chunk_context(text, chunk_end_tokens={".", "\n"})
    # as the library calls it (llmlingua 0.2.2): without model_name, so the xlm-roberta model gets BERT's
    # [CLS]/[SEP]/[PAD] strings, which its tokenizer maps to <unk>. Kept to stay off-the-shelf.
    ds = TokenClfDataset(chunks, tokenizer=pc.tokenizer, max_len=pc.max_seq_len)
    out, t0 = [], time.time()
    bs = 8
    with torch.no_grad():
        for b0 in range(0, len(ds), bs):
            items = [ds[i] for i in range(b0, min(len(ds), b0 + bs))]
            ids = torch.stack([x["ids"] for x in items]).long()
            mask = torch.stack([x["mask"] for x in items]).long() == 1
            probs = F.softmax(pc.model(input_ids=ids, attention_mask=mask).logits, dim=-1)[:, :, 1]
            for j in range(ids.shape[0]):
                act = torch.masked_select(probs[j], mask[j]).numpy()
                toks = pc.tokenizer.convert_ids_to_tokens(torch.masked_select(ids[j], mask[j]).tolist())
                words, tp, _ = pc._PromptCompressor__merge_token_to_word(
                    tokens=toks, token_probs=list(act), force_tokens=[], token_map={}, force_reserve_digit=True)
                out.append({"words": words, "probs": [float(sum(p) / len(p)) for p in tp],
                            "w": [word_weight(pc, w) for w in words]})
            if (b0 // bs) % 10 == 0:
                log(f"[prune] chunk {b0 + len(items)}/{len(ds)} {time.time() - t0:.0f}s")
    return out


def keep_labels(chunks: list[dict], rate: float, digit_words: bool = True) -> list[list[bool]]:
    """LLMLingua-2's token-level filter, re-applied to cached probabilities: per chunk, a word is kept if its
    probability exceeds the int(100 * (1 - rate) + 1)-th percentile of the chunk's probabilities weighted by its
    token count. ``digit_words``: a word containing a digit is always kept (the library forces digit *tokens* but
    averages them with other tokens of the same word, so ',512' of '4,512' could be dropped)."""
    reduce_rate = max(0.0, 1.0 - rate)
    out = []
    for c in chunks:
        p = np.array(c["probs"], dtype=float)
        if digit_words:
            p = np.where([bool(re.search(r"\d", w)) for w in c["words"]], 1.0, p)
        if reduce_rate <= 0 or not len(p):
            out.append([True] * len(p))
            continue
        thr = np.percentile(np.repeat(p, c["w"]), int(100 * reduce_rate + 1))
        out.append([bool(x > thr or (thr == 1.0 and x == thr)) for x in p])
    return out


def piece(w: str) -> str:
    return " " + w[1:] if w.startswith("▁") else w


def assemble(chunks: list[dict], labels: list[list[bool]] | None = None) -> tuple[str, list[tuple[int, int, bool]]]:
    """Concatenate words (all, with keep flags) -> (full reconstructed text, [(start, end, kept)] per word); the pruned
    text is ``pruned_text(...)``. Word-initial pieces carry their leading space (xlm-roberta's '▁')."""
    parts, spans, pos = [], [], 0
    for ci, c in enumerate(chunks):
        for wi, w in enumerate(c["words"]):
            s = piece(w)
            parts.append(s)
            spans.append((pos, pos + len(s), True if labels is None else labels[ci][wi]))
            pos += len(s)
    return "".join(parts), spans


def pruned_text(chunks: list[dict], labels: list[list[bool]]) -> str:
    s = "".join(piece(w) for c, lab in zip(chunks, labels) for w, k in zip(c["words"], lab) if k)
    return re.sub(r" {2,}", " ", s).strip()


def prob_cache(d: int, doc: str) -> Path:
    h = hashlib.sha256((PRUNER + doc).encode()).hexdigest()[:16]
    return CACHE / f"probs_d{d}_{h}.json"


# --------------------------------------------------------------------------- survival (free check)
def norm_anchor(pc_tok, s: str) -> str:
    return pc_tok.convert_tokens_to_string(pc_tok.tokenize(s))


def survival(full: str, spans: list[tuple[int, int, bool]], q: dict, anchors_norm: list[str]) -> dict:
    """For one question: are its anchors' numbers, names (capitalised words), opening words (sentence-anchor
    questions) and function words (small-word questions) all kept? Matching is on whole words of the reconstructed
    text. -> {"located", "number", "names", "opening", "function", "all"} (None = does not apply)."""
    starts = np.array([s for s, _, _ in spans])
    res = {"located": True, "number": True, "names": True, "opening": None, "function": None}
    words_in = []
    for a in anchors_norm:
        i = full.find(a)
        if i < 0 or full.count(a) != 1:
            res["located"] = False
            continue
        j0 = max(0, int(np.searchsorted(starts, i, side="right")) - 1)
        ws = []
        for s, e, k in spans[j0:]:
            if s >= i + len(a):
                break
            if e > i:
                ws.append((full[s:e].strip(), k))
        words_in.append(ws)
        for w, k in ws:
            if re.search(r"\d", w) and not k:
                res["number"] = False
            if is_invented(w) and not k:
                res["names"] = False
    if q["kind"] == "number" and words_in:
        res["opening"] = all(k for _, k in words_in[0])
        res["opening_share"] = float(np.mean([k for _, k in words_in[0]]))
        res["names"] = None
    if q["kind"] == "small" and words_in:
        need = [c.lower() for c in q["critical"]]
        flat = [(re.sub(r"\W", "", w).lower(), k) for ws in words_in for w, k in ws]
        res["function"] = all(all(k for w, k in flat if w == c) and any(w == c for w, _ in flat) for c in need)
    res["all"] = bool(res["located"] and all(v for k, v in res.items()
                                             if k not in ("located", "opening_share") and v is not None))
    return res


_INVENTED = re.compile("^(?:" + "|".join(s.capitalize() for s in R.SYL) + ")(?:" + "|".join(R.SYL) + ")$")


def is_invented(w: str) -> bool:
    """A word of an invented name (first names of the harness, or two harness syllables), punctuation stripped."""
    w = re.sub(r"[^\w]", "", w)
    return w in R.FIRST or bool(_INVENTED.match(w))


def pruned_span(full: str, spans, a: str) -> str:
    i = full.find(a)
    if i < 0:
        return ""
    return "".join(full[s:e] for s, e, k in spans if k and e > i and s < i + len(a)).strip()


# --------------------------------------------------------------------------- prompts and parsing
def prompt(arm: str, doc: str, qs: list[dict], n_pages: int) -> str:
    base = R.prompt_for({"none": "none", "text": "text", "ptext22": "text", "ptext15": "text"}.get(arm, "image"),
                        doc, qs, n_pages)
    if arm in ("ptext22", "ptext15", "ppage15"):
        base = base.replace("\n\nQuestions:\n", f"\n\n{NOTE}\n\nQuestions:\n", 1)
    return base


def parse_answers(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*\**([NSCW]\d{2})\**\s*[:\-]\s*(.+)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


REFUSAL = re.compile(r"usage policy|acceptable use|violat|I can(?:no|')t help with|I won't|refus", re.I)


def est_cost(input_tokens: float, output_tokens: float) -> float:
    return input_tokens * PRICE_IN + output_tokens * PRICE_OUT


# --------------------------------------------------------------------------- commands
def load_state() -> dict:
    p = OUT / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(st: dict) -> None:
    (OUT / "state.json").write_text(json.dumps(st, indent=1))


def spent() -> float:
    p = OUT / "spend.jsonl"
    if not p.exists():
        return 0.0
    return sum(json.loads(x)["cost_usd"] for x in p.read_text().splitlines() if x.strip())


def log_spend(what: str, rec: dict) -> None:
    if rec.get("cached"):
        return
    with open(OUT / "spend.jsonl", "a") as f:
        f.write(json.dumps({"what": what, "cost_usd": rec.get("cost_usd", 0.0), "input_tokens": rec.get("input_tokens"),
                            "output_tokens": rec.get("output_tokens"), "error": rec.get("error")}) + "\n")


def cmd_build(a) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    pc = None
    for d in a.docs:
        doc, qs = build_doc(d)
        verify_original(d, qs)
        dd = OUT / f"d{d}"
        dd.mkdir(parents=True, exist_ok=True)
        (dd / "questions.json").write_text(json.dumps(qs, indent=1))
        print(f"[prune] document {d}: {len(doc):,} chars, {len(qs)} questions (original 120 verified)", flush=True)
        cp = prob_cache(d, doc)
        if cp.exists():
            continue
        pc = pc or load_compressor()
        chunks = word_probs(pc, doc)
        full, _ = assemble(chunks)
        tmp = cp.with_suffix(".tmp")
        tmp.write_text(json.dumps({"chunks": chunks, "doc_chars": len(doc), "full_chars": len(full)}))
        tmp.replace(cp)
        print(f"[prune] document {d}: {len(chunks)} chunks cached -> {cp}", flush=True)


def doc_bundle(d: int, pc_tok=None) -> dict:
    doc, qs = build_doc(d)
    chunks = json.loads(prob_cache(d, doc).read_text())["chunks"]
    return {"doc": doc, "qs": qs, "chunks": chunks}


def rate_for_chars(chunks: list[dict], target_chars: int) -> float:
    """Smallest keep rate (1% steps, the library's percentile resolution) whose pruned text reaches target_chars."""
    lo, hi = 0.05, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if len(pruned_text(chunks, keep_labels(chunks, mid))) < target_chars:
            lo = mid
        else:
            hi = mid
    return round(hi, 4)


def pruned_docs(b: dict, st: dict) -> dict:
    """-> {"ptext22": text, "ptext15": text} at the calibrated rates (state.json)."""
    out = {}
    for arm in ("ptext22", "ptext15"):
        rate = st.get("rates", {}).get(f"d{b['d']}", {}).get(arm)
        if rate is not None:
            out[arm] = pruned_text(b["chunks"], keep_labels(b["chunks"], rate))
    return out


def cmd_check(a) -> None:
    """Free checks: per pruned document, the share of questions whose anchors keep their numbers, names, opening
    words and function words; token and cost estimate of every arm."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PRUNER)
    st = load_state()
    cpt = st.get("chars_per_token", {})
    report, total_cost = [], 0.0
    for d in a.docs:
        b = doc_bundle(d)
        b["d"] = d
        full, spans0 = assemble(b["chunks"])
        anchors = {q["id"]: [norm_anchor(tok, x) for x in (q.get("anchors") or [q["fact"]])] for q in b["qs"]}
        ratesd = st.get("rates", {}).get(f"d{d}", {})
        n_pages118 = None
        rows = {}
        for arm, rate in ratesd.items():
            labels = keep_labels(b["chunks"], rate)
            _, spans = assemble(b["chunks"], labels)
            text = pruned_text(b["chunks"], labels)
            surv = {q["id"]: survival(full, spans, q, anchors[q["id"]]) for q in b["qs"]}
            by = {}
            for q in b["qs"]:
                for k in ("all", q["kind"]):
                    by.setdefault(k, []).append(surv[q["id"]]["all"])
            examples = [{"id": q["id"], "sub": q["sub"], "original": q["anchors"][0],
                         "pruned": pruned_span(full, spans, anchors[q["id"]][0]), "survives": surv[q["id"]]}
                        for q in b["qs"] if q["kind"] == "small"]
            rows[arm] = {"rate": rate, "chars": len(text), "keep_chars": len(text) / len(b["doc"]),
                         "survive": {k: float(np.mean(v)) for k, v in by.items()},
                         "survive_parts": {p: float(np.mean([s[p] for s in surv.values() if s.get(p) is not None]))
                                           for p in ("located", "number", "names", "opening", "opening_share", "function")},
                         "examples": examples}
            (OUT / f"d{d}" / f"{arm}_survival.json").write_text(json.dumps({"survival": surv, **rows[arm]}, indent=1))
        report.append((d, len(b["doc"]), rows))
        # token / cost estimate
        if cpt:
            n_q = len(b["qs"])
            ov = st.get("prompt_overhead", 7500)
            est = {"none": ov, "text": len(b["doc"]) / cpt["full"] + ov}
            for arm, bud in BUDGETS.items():
                _, info = R.render_pages(b["doc"], bud, PAGE_PX)
                est[arm] = sum(p["vision_tokens"] for p in info) + ov
                if arm == "page118":
                    n_pages118 = len(info)
            for arm in ("ptext22", "ptext15"):
                if arm in rows:
                    est[arm] = rows[arm]["chars"] / cpt.get(arm, cpt["full"]) + ov
            if "ptext15" in rows:
                txt = pruned_docs(b, st)["ptext15"]
                _, info = R.render_pages(txt, 117.6, PAGE_PX)
                est["ppage15"] = sum(p["vision_tokens"] for p in info) + ov
            outk = {"none": 3000, "text": 8000, "ptext22": 9000, "ptext15": 8000, "page118": 8000, "page88": 9000,
                    "page59": 12000, "ppage15": 10000}
            c = {k: est_cost(v, outk[k]) for k, v in est.items()}
            total_cost += sum(c.values())
            print(f"[check] document {d}: {n_q} questions, 117.6 pages: {n_pages118}; estimated input tokens / $:",
                  "; ".join(f"{k} {v:,.0f} ${c[k]:.2f}" for k, v in est.items()), flush=True)
    for d, n, rows in report:
        for arm, r in rows.items():
            print(f"[check] d{d} {arm}: rate {r['rate']}, kept chars {r['keep_chars']:.3f} ({r['chars']:,}); "
                  "all needed words kept: " + ", ".join(f"{k} {v:.2f}" for k, v in r["survive"].items())
                  + "; parts: " + ", ".join(f"{k} {v:.2f}" for k, v in r["survive_parts"].items()), flush=True)
    if total_cost:
        print(f"[check] estimated cost of all arms on {len(a.docs)} documents: ${total_cost:.2f} "
              f"(spent so far ${spent():.2f}, cap ${CAP_USD:.0f})", flush=True)


CAL_SYSTEM = "Answer briefly."
CAL_PROMPT = "<document>\n{x}\n</document>\n\nReply with the single word OK."


def cmd_calibrate(a) -> None:
    """Small unscored calls (effort low) on an excerpt of document 0: Claude input tokens per character of full text
    and of text pruned at the two candidate rates. Rates are then set per document so that ptext22's pruned
    document has as many Claude tokens as page118's image tokens, and ptext15's total is text's total / 1.5."""
    from vbpe import claude_eval
    st = load_state()
    b = doc_bundle(0)
    full, spans0 = assemble(b["chunks"])
    lo, hi = len(full) // 2, len(full) // 2 + a.excerpt
    sel = [i for i, (s, e, _) in enumerate(spans0) if s >= lo and e <= hi]

    def excerpt(rate):
        labels = keep_labels(b["chunks"], rate) if rate < 1 else None
        _, spans = assemble(b["chunks"], labels)
        s = "".join(full[spans[i][0]:spans[i][1]] for i in sel if spans[i][2])
        return re.sub(r" {2,}", " ", s).strip()

    def tokens(x, what):
        rec = claude_eval.transcribe([], CAL_SYSTEM, CAL_PROMPT.format(x=x), effort="low")
        log_spend(what, rec)
        assert rec["error"] is None, rec["error"]
        return rec["input_tokens"]
    if not a.yes:
        print("[cal] add --yes to make the calibration calls (~$0.4)")
        return
    ov = tokens("", "calibrate overhead")
    res = {"overhead": ov}
    for name, rate in (("full", 1.0), ("ptext22", a.rate22), ("ptext15", a.rate15)):
        x = excerpt(rate)
        t = tokens(x, f"calibrate {name}") - ov
        res[name] = {"rate": rate, "chars": len(x), "tokens": t, "cpt": len(x) / t}
        print(f"[cal] {name}: rate {rate}, {len(x):,} chars, {t:,} tokens, {len(x) / t:.3f} chars/token", flush=True)
    qblock = "Questions:\n" + "\n".join(f"{q['id']}: {q['question']}" for q in b["qs"])
    res["questions"] = tokens(qblock, "calibrate question block") - ov
    st["calibration"] = res
    st["prompt_overhead"] = ov + res["questions"]
    print(f"[cal] overhead {ov}, question block {res['questions']} tokens", flush=True)
    st["chars_per_token"] = {k: res[k]["cpt"] for k in ("full", "ptext22", "ptext15")}
    save_state(st)


def cmd_set_rates(a) -> None:
    """Set each document's keep rates from the calibration: ptext22's pruned document = page118's image tokens;
    ptext15's total input = (text's total input) / 1.5 (both totals with the same prompt overhead)."""
    st = load_state()
    cpt = st["chars_per_token"]
    ov = st.get("prompt_overhead", 7500)
    st.setdefault("rates", {})
    for d in a.docs:
        b = doc_bundle(d)
        _, info = R.render_pages(b["doc"], 117.6, PAGE_PX)
        img = sum(p["vision_tokens"] for p in info)
        text_tok = len(b["doc"]) / cpt["full"]
        t22 = img * cpt["ptext22"]
        t15 = ((text_tok + ov) / 1.5 - ov) * cpt["ptext15"]
        st["rates"][f"d{d}"] = {"ptext22": rate_for_chars(b["chunks"], int(t22)),
                                "ptext15": rate_for_chars(b["chunks"], int(t15))}
        print(f"[rates] d{d}: page118 image tokens {img:,}; text ~{text_tok:,.0f} tokens; target chars ptext22 "
              f"{t22:,.0f}, ptext15 {t15:,.0f} -> rates {st['rates'][f'd{d}']}", flush=True)
    save_state(st)


def cmd_run(a) -> None:
    from analyse_claude_qa import engagement
    from vbpe import claude_eval
    st = load_state()
    stopped = set(st.get("stopped_arms", []))
    for d in a.docs:
        b = doc_bundle(d)
        b["d"] = d
        qs = b["qs"]
        dd = OUT / f"d{d}"
        done = set()
        if (dd / "runs.jsonl").exists():
            done = {json.loads(x)["arm"] for x in (dd / "runs.jsonl").read_text().splitlines() if x.strip()}
        pdocs = pruned_docs(b, st)
        jobs = []
        for arm in a.arms:
            if arm in done or arm in stopped:
                print(f"[run] d{d} {arm}: {'done' if arm in done else 'stopped (refused earlier)'}", flush=True)
                continue
            if arm in ("text", "none"):
                doc, pages, info = b["doc"], [], []
            elif arm in BUDGETS:
                doc = b["doc"]
                pages, info = R.render_pages(doc, BUDGETS[arm], PAGE_PX)
            elif arm in ("ptext22", "ptext15"):
                doc, pages, info = pdocs[arm], [], []
            elif arm == "ppage15":
                doc = pdocs["ptext15"]
                pages, info = R.render_pages(doc, 117.6, PAGE_PX)
            vt = sum(p["vision_tokens"] for p in info)
            pr = prompt(arm, doc, qs, len(pages))
            est_in = vt + (len(pr) / st["chars_per_token"].get(arm, st["chars_per_token"]["full"]) if not pages else
                           st.get("prompt_overhead", 7500))
            jobs.append({"arm": arm, "doc": doc, "pages": pages, "info": info, "vt": vt, "prompt": pr,
                         "est": est_cost(est_in, 12000)})
        need = sum(j["est"] for j in jobs)
        if spent() + need > CAP_USD:
            print(f"[run] d{d}: spent ${spent():.2f} + estimate ${need:.2f} > cap ${CAP_USD}: stopping", flush=True)
            return
        if not a.yes:
            print(f"[run] d{d}: {len(jobs)} calls, estimate ${need:.2f}; add --yes", flush=True)
            continue
        for j in jobs:
            (dd / "images" / j["arm"]).mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(j["pages"]):
                im.save(dd / "images" / j["arm"] / f"page_{i:02d}.png")
            if j["arm"] in ("ptext22", "ptext15"):
                (CACHE / f"d{d}_{j['arm']}.txt").write_text(j["doc"])
        tj = [{"images": j["pages"], "system": R.SYSTEM_NONE if j["arm"] == "none" else R.SYSTEM, "prompt": j["prompt"],
               "effort": "high", "timeout": 1800.0, "hires": bool(j["pages"]), "meta": j["arm"]} for j in jobs]
        recs = claude_eval.transcribe_many(tj, workers=a.workers)
        for j, rec in zip(jobs, recs):
            arm = j["arm"]
            log_spend(f"d{d} {arm}", rec)
            ans = parse_answers(rec.get("text", ""))
            per = [{"id": q["id"], "kind": q["kind"], "sub": q.get("sub"), "pos": q["pos"], "answer": q["answer"],
                    "lure": q.get("lure"), "given": ans.get(q["id"]),
                    "correct": ans.get(q["id"]) is not None and R.norm_num(ans[q["id"]]) == q["answer"]} for q in qs]
            incomplete = bool(j["pages"] and rec.get("input_tokens") and rec["input_tokens"] < 0.97 * j["vt"])
            refused = bool(rec.get("stop_reason") == "refusal" or (rec.get("error") and REFUSAL.search(rec["error"]))
                           or (not ans and REFUSAL.search(rec.get("text", ""))))
            row = {"arm": arm, "doc": d, "chars": len(j["doc"]), "n_pages": len(j["pages"]),
                   "image_tokens_formula": j["vt"], "effort": "high", "model": rec.get("model"),
                   "input_tokens": rec.get("input_tokens"), "output_tokens": rec.get("output_tokens"),
                   "usage": rec.get("usage"), "cost_usd": rec.get("cost_usd"), "latency_s": rec.get("latency_s"),
                   "cached": rec.get("cached"), "error": rec.get("error"), "stop_reason": rec.get("stop_reason"),
                   "incomplete": incomplete, "refused": refused, "page_px": PAGE_PX if j["pages"] else 0,
                   "rate": st.get("rates", {}).get(f"d{d}", {}).get("ptext15" if arm == "ppage15" else arm),
                   "n_questions": len(qs), "acc_all": float(np.mean([p["correct"] for p in per])),
                   "acc": {k: float(np.mean([p["correct"] for p in per if p["kind"] == k]))
                           for k in sorted({q["kind"] for q in qs})},
                   "per_question": per, "raw": rec.get("text", "")}
            row["unanswered"], row["disengaged"] = engagement(row)
            if row["error"] and not refused:
                print(f"[run] d{d} {arm}: ERROR (not stored, rerun): {row['error'][:300]}", flush=True)
                continue
            with open(dd / "runs.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if refused:
                st = load_state()
                st.setdefault("stopped_arms", []).append(arm)
                save_state(st)
            print(f"[run] d{d} {arm}: acc {row['acc_all']:.3f} " + ", ".join(f"{k} {v:.2f}" for k, v in row["acc"].items())
                  + f"; in {row['input_tokens']:,} out {row['output_tokens']:,} ${row['cost_usd']:.2f} "
                  f"{row['latency_s']:.0f}s incomplete {incomplete} refused {refused} unanswered {row['unanswered']}",
                  flush=True)
        print(f"[run] spent so far ${spent():.2f}", flush=True)


def cmd_analyse(a) -> None:
    from analyse_claude_qa import MARGIN, mcnemar, paired_diff_ci, wilson
    rows = {}
    for d in range(3):
        p = OUT / f"d{d}" / "runs.jsonl"
        if p.exists():
            for x in p.read_text().splitlines():
                if x.strip():
                    r = json.loads(x)
                    rows[(d, r["arm"])] = r
    docs = sorted({d for d, _ in rows})
    valid = lambda r: not (r.get("incomplete") or r.get("refused") or r.get("unanswered") or r.get("disengaged"))  # noqa
    res, md = {}, []
    groups = {"all": lambda p: True, "original": lambda p: p["kind"] != "small", "small": lambda p: p["kind"] == "small",
              "needle": lambda p: p["kind"] == "needle", "number": lambda p: p["kind"] == "number",
              "complex": lambda p: p["kind"] == "complex"}
    for arm in ARMS:
        have = [d for d in docs if (d, arm) in rows]
        if not have:
            continue
        rs = [rows[(d, arm)] for d in have]
        r = {"docs": have, "invalid": [d for d, x in zip(have, rs) if not valid(x)],
             "input_tokens": float(np.mean([x["input_tokens"] for x in rs])),
             "cost_usd": float(sum(x["cost_usd"] for x in rs if not x.get("cached")))}
        tdocs = [d for d in have if (d, "text") in rows]
        if tdocs:
            r["ratio_vs_text"] = float(np.mean([rows[(d, "text")]["input_tokens"] / rows[(d, arm)]["input_tokens"]
                                                for d in tdocs]))
        for g, f in groups.items():
            per = [p for x in rs for p in x["per_question"] if f(p)]
            k = sum(p["correct"] for p in per)
            r[f"acc_{g}"] = (k / len(per), *wilson(k, len(per)), len(per)) if per else None
            if arm != "text" and tdocs:
                pairs = []
                for d in tdocs:
                    tq = {p["id"]: p for p in rows[(d, "text")]["per_question"]}
                    pairs += [(d, bool(tq[p["id"]]["correct"]), bool(p["correct"]))
                              for p in rows[(d, arm)]["per_question"] if f(p)]
                if pairs:
                    diff = paired_diff_ci(pairs)
                    n10, n01, pm = mcnemar([x[1] for x in pairs], [x[2] for x in pairs])
                    r[f"diff_{g}"] = {"arm_minus_text": -diff[0], "lo": -diff[2], "hi": -diff[1],
                                      "text_only": n10, "arm_only": n01, "p": pm, "n": len(pairs),
                                      "lossless": diff[2] < MARGIN,
                                      "loss": diff[1] > 0 and pm < 0.05}
        smalls = [p for x in rs for p in x["per_question"] if p["kind"] == "small"]
        r["small_lure"] = (float(np.mean([p["given"] is not None and R.norm_num(p["given"]) == p["lure"] for p in smalls]))
                           if smalls else None)
        subs = sorted({p["sub"] for p in smalls})
        r["small_by_sub"] = {s: float(np.mean([p["correct"] for p in smalls if p["sub"] == s])) for s in subs}
        r["by_sub"] = {}
        for p in (p for x in rs for p in x["per_question"]):
            key = p.get("sub") or p["kind"]
            r["by_sub"].setdefault(key, []).append(p["correct"])
        r["by_sub"] = {k: float(np.mean(v)) for k, v in r["by_sub"].items()}
        res[arm] = r
    # pages vs ppage15 (P3)
    comp = {}
    for ref in ("page59", "page88"):
        common = [d for d in docs if (d, ref) in rows and (d, "ppage15") in rows]
        for g in ("all", "original"):
            pairs = []
            for d in common:
                rq = {p["id"]: p for p in rows[(d, ref)]["per_question"]}
                pairs += [(d, bool(rq[p["id"]]["correct"]), bool(p["correct"]))
                          for p in rows[(d, "ppage15")]["per_question"] if groups[g](p)]
            if pairs:
                diff = paired_diff_ci(pairs)
                n10, n01, pm = mcnemar([x[1] for x in pairs], [x[2] for x in pairs])
                comp[f"ppage15_vs_{ref}_{g}"] = {"diff": -diff[0], "lo": -diff[2], "hi": -diff[1], "ref_only": n10,
                                                "ppage_only": n01, "p": pm, "n": len(pairs)}
    out = OUT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"arms": res, "ppage15_vs_pages": comp}, indent=1))
    md.append("| arm | input tokens (mean) | x fewer than text | all | original 120 | small-word | arm - text, all (95% CI) "
              "| original | small-word | invalid docs |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for arm, r in res.items():
        f = lambda g: f"{r[f'acc_{g}'][0]:.1%}" if r.get(f"acc_{g}") else "-"  # noqa: E731
        dd = lambda g: (f"{100 * r[f'diff_{g}']['arm_minus_text']:+.1f} ({100 * r[f'diff_{g}']['lo']:+.1f} to "  # noqa
                        f"{100 * r[f'diff_{g}']['hi']:+.1f})" if f"diff_{g}" in r else "-")
        md.append(f"| {arm} | {r['input_tokens']:,.0f} | {r.get('ratio_vs_text', float('nan')):.2f} | {f('all')} | "
                  f"{f('original')} | {f('small')} | {dd('all')} | {dd('original')} | {dd('small')} | "
                  f"{r['invalid'] or ''} |")
    md.append("")
    for k, v in comp.items():
        md.append(f"* {k}: {100 * v['diff']:+.1f} points (95% CI {100 * v['lo']:+.1f} to {100 * v['hi']:+.1f}), "
                  f"ref-only {v['ref_only']}, ppage15-only {v['ppage_only']}, McNemar p = {v['p']:.3g}, n = {v['n']}")
    md.append("")
    md.append("Accuracy per question type:")
    md.append("")
    subs = sorted({s for r in res.values() for s in r["by_sub"]})
    md.append("| arm | " + " | ".join(subs) + " |")
    md.append("|---|" + "---|" * len(subs))
    for arm, r in res.items():
        md.append(f"| {arm} | " + " | ".join(f"{r['by_sub'].get(s, float('nan')):.0%}" for s in subs) + " |")
    md.append("")
    md.append("Small-word questions answered with the lure (the wrong-reading number): "
              + ", ".join(f"{arm} {r['small_lure']:.0%}" for arm, r in res.items() if r.get("small_lure") is not None))
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "check", "calibrate", "set-rates", "run", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--docs", type=int, nargs="+", default=[0, 1, 2])
        p.add_argument("--yes", action="store_true")
        if name == "calibrate":
            p.add_argument("--excerpt", type=int, default=60000)
            p.add_argument("--rate22", type=float, default=0.45)
            p.add_argument("--rate15", type=float, default=0.66)
        if name == "run":
            p.add_argument("--arms", nargs="+", default=ARMS)
            p.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"build": cmd_build, "check": cmd_check, "calibrate": cmd_calibrate, "set-rates": cmd_set_rates, "run": cmd_run,
     "analyse": cmd_analyse}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
