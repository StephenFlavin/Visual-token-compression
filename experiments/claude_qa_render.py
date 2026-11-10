"""Does Claude answer questions as well from a rendered document as from its text? (user request, 2026-11-10)

Phase 1 showed Claude *transcribes* natural Latin at 29.4 px²/char with 3% CER (~6x fewer tokens than text). This
tests *use*: the same long document is given to headless Claude three ways, with the same questions:

* ``text``   the document as text in the prompt;
* ``image``  the same string rendered as DejaVu Sans Latin pages at --budget px²/char (Phase 1's renderer; each page
             <= 1.15 MP and 1568 px, so the API does not downsample), in order;
* ``none``   no document (the prior / memorisation floor: WikiText is Wikipedia, which Claude may know).

Questions (all with exact answers, scored automatically):
* ``needle``  --n-needles invented facts inserted between paragraphs at evenly spaced positions (invented people
              and places, so only the document can answer);
* ``number``  --n-number real sentences from the document that contain one number; the question quotes the
              sentence's opening words and asks for the number.

Tokens come from the CLI's usage report (input incl. cache creation / read). Calls go through
vbpe.claude_eval (isolated, no tools, cached on disk). Paired exact McNemar test, text vs image.

    python3 experiments/claude_qa_render.py --chars 500000 --dry-run      # build pages + questions, no calls
    python3 experiments/claude_qa_render.py --chars 500000 --yes
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vbpe import config, corpus, metrics  # noqa: E402
from vbpe.layout import render_page  # noqa: E402

FIRST = ["Alma", "Bertil", "Cora", "Dmitri", "Edda", "Floris", "Greta", "Henrik", "Ilse", "Joram", "Katja",
         "Lorcan", "Maren", "Niels", "Odile", "Pavo", "Quilla", "Rasmus", "Sabine", "Tobin", "Ulla", "Viggo",
         "Wenna", "Yorick", "Zelda", "Anselm", "Brita", "Cosmo", "Dagny", "Elof", "Fenna", "Gisli", "Hedda",
         "Ivo", "Jutta", "Kasimir", "Liesl", "Matthis", "Nora", "Osric"]
SYL = ["var", "len", "mor", "tis", "bak", "ell", "dra", "sund", "kel", "oth", "rim", "ast", "vel", "nor", "gav",
       "bry", "lun", "dek", "wen", "tor", "fal", "mir", "osk", "eby", "hal", "quen", "zar", "ith", "pel", "rud"]
TEMPLATES = [
    ("The parish archive of {place} holds {n} letters written by {who}.",
     "How many letters written by {who} does the parish archive of {place} hold?"),
    ("In the survey of {place}, {who} recorded {n} species of moth.",
     "How many species of moth did {who} record in the survey of {place}?"),
    ("The bridge at {place} was rebuilt by {who} using {n} tonnes of granite.",
     "How many tonnes of granite did {who} use to rebuild the bridge at {place}?"),
]
NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def invented(rng, used: set) -> tuple[str, str]:
    while True:
        last = "".join(rng.choice(SYL, 2)).capitalize()
        place = "".join(rng.choice(SYL, 2)).capitalize()
        who = f"{rng.choice(FIRST)} {last}"
        if who not in used and place not in used and last != place:
            used.update((who, place))
            return who, place


def build_document(n_chars: int, offset: int, splits=("test", "val", "train")):
    """Held-out test text first, then validation, then train (only if a very long document needs it)."""
    paras = [p for sp in splits for p in corpus.read_paragraphs(sp)][offset:]
    out, total = [], 0
    for p in paras:
        out.append(p)
        total += len(p) + 1
        if total >= n_chars:
            break
    return out


def norm_num(s: str) -> str:
    return re.sub(r"[^\d.]", "", s).rstrip(".")


def make_questions(paras: list[str], n_needles: int, n_number: int, seed: int):
    rng = np.random.default_rng(seed)
    used: set = set()
    # needles: inserted before evenly spaced paragraphs
    at = np.linspace(0, len(paras) - 1, n_needles + 2)[1:-1].astype(int)
    needles = {}
    for k, i in enumerate(at):
        who, place = invented(rng, used)
        n = int(rng.integers(120, 9800))
        while 1000 <= n <= 2100:            # not year-like
            n = int(rng.integers(120, 9800))
        fact, q = TEMPLATES[k % len(TEMPLATES)]
        needles[int(i)] = {"id": f"N{k + 1:02d}", "kind": "needle", "fact": fact.format(place=place, n=f"{n:,}", who=who),
                           "question": q.format(place=place, who=who), "answer": str(n)}
    doc_paras = []
    for i, p in enumerate(paras):
        if i in needles:
            doc_paras.append(needles[i]["fact"])
        doc_paras.append(p)
    doc = " ".join(doc_paras)
    # number questions: real sentences with exactly one number (>= 2 digits) after >= 6 words
    cands = []
    for s in (s for p in paras for s in corpus.split_sentences(p)):
        nums = NUM.findall(s)
        if len(nums) != 1 or len(norm_num(nums[0]).replace(".", "")) < 2 or re.search(r"\d", NUM.sub("", s)):
            continue
        m = NUM.search(s)
        opening = s[:m.start()].split()
        if len(opening) < 6:
            continue
        opening = " ".join(opening[:10])
        if doc.count(opening) != 1 or NUM.search(opening):
            continue
        cands.append({"opening": opening, "answer": norm_num(nums[0]), "pos": doc.find(opening)})
    cands.sort(key=lambda c: c["pos"])
    pick = np.linspace(0, len(cands) - 1, n_number).astype(int) if len(cands) >= n_number else range(len(cands))
    qs = list(needles.values())
    for k, j in enumerate(pick):
        c = cands[j]
        qs.append({"id": f"S{k + 1:02d}", "kind": "number", "answer": c["answer"],
                   "question": f'Find the sentence that begins "{c["opening"]}". What number does that sentence '
                               "contain?"})
    for q in qs:
        anchor = q.get("fact") or q["question"].split('"')[1]
        q["pos"] = doc.find(anchor) / len(doc)
    return doc, qs


def chunk(doc: str, max_chars: int) -> list[str]:
    out, i = [], 0
    while i < len(doc):
        j = min(len(doc), i + max_chars)
        if j < len(doc):
            j = doc.rfind(" ", i, j) + 1 or j
        out.append(doc[i:j].strip())
        i = j
    return out


def render_pages(doc: str, budget: float) -> tuple[list, list[dict]]:
    from vbpe.schemes import get_scheme
    sys.path.insert(0, str(ROOT / "experiments"))
    from phase1_sweep import canvas_size
    sch = get_scheme("latin")
    max_chars = int(config.CLAUDE_MAX_AREA / budget * 0.97)
    while True:
        pages, info = [], []
        ok = True
        for c in chunk(doc, max_chars):
            n = len(metrics.normalize(c))
            W, H = canvas_size(budget, n)
            if max(W, H) > config.CLAUDE_MAX_EDGE or W * H > config.CLAUDE_MAX_AREA:
                ok = False
                break
            img, inf = render_page(sch.encode(c), W, H)
            if not inf["fit"]:
                ok = False
                break
            pages.append(img)
            info.append({"W": W, "H": H, "chars": n, "em": inf["em"], "lines": inf["n_lines"],
                         "px_per_char": W * H / n, "vision_tokens": metrics.vision_tokens(W, H)})
        if ok:
            return pages, info
        max_chars = int(max_chars * 0.95)


SYSTEM = ("You answer questions about a document. Use only the document provided. Answer every question with "
          "the exact value only (a number), one line per question in the form 'ID: answer'. If you cannot find "
          "the answer, give your best guess.")
SYSTEM_NONE = ("Answer every question with a number only, one line per question in the form 'ID: answer'. If you "
               "do not know, give your best guess.")


def prompt_for(arm: str, doc: str, qs: list[dict], n_pages: int) -> str:
    ql = "\n".join(f"{q['id']}: {q['question']}" for q in qs)
    if arm == "text":
        return f"<document>\n{doc}\n</document>\n\nQuestions:\n{ql}"
    if arm == "image":
        return (f"The document is the {n_pages} page images above, in reading order (the text runs on from one page "
                f"to the next).\n\nQuestions:\n{ql}")
    return f"Questions:\n{ql}"


def parse_answers(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*\**([NS]\d{2})\**\s*[:\-]\s*(.+)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    from math import comb
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n10, n01) + 1)) / 2 ** n) if n else 1.0
    return {"text_only": n10, "image_only": n01, "p": p}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chars", type=int, default=500000)
    ap.add_argument("--offset", type=int, default=0, help="first test paragraph")
    ap.add_argument("--budget", type=float, default=29.4, help="px²/char (Phase 1: Claude's 3%% CER point)")
    ap.add_argument("--n-needles", type=int, default=40)
    ap.add_argument("--n-number", type=int, default=40)
    ap.add_argument("--arms", nargs="+", default=["none", "image", "text"])
    ap.add_argument("--effort", default="low")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="qa500k")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    out = config.RESULTS / "claude_qa" / a.run_name
    (out / "images").mkdir(parents=True, exist_ok=True)
    paras = build_document(a.chars, a.offset)
    doc, qs = make_questions(paras, a.n_needles, a.n_number, a.seed)
    pages, pinfo = render_pages(doc, a.budget)
    for i, im in enumerate(pages):
        im.save(out / "images" / f"page_{i:02d}.png")
    vt = sum(p["vision_tokens"] for p in pinfo)
    (out / "questions.json").write_text(json.dumps(qs, indent=1))
    (out / "pages.json").write_text(json.dumps(pinfo, indent=1))
    print(f"[qa] document {len(doc):,} chars ({len(paras)} paragraphs + {a.n_needles} needles); "
          f"{len(qs)} questions; {len(pages)} pages, {vt:,.0f} image tokens, "
          f"px²/char {np.mean([p['px_per_char'] for p in pinfo]):.1f}, em {np.mean([p['em'] for p in pinfo]):.2f}; "
          f"text ~{metrics.text_tokens_est(doc):,.0f} tokens (estimate)", flush=True)
    if a.dry_run:
        return 0
    if not a.yes:
        print("[qa] add --yes to call Claude", flush=True)
        return 0
    from vbpe import claude_eval
    rows = []
    for arm in a.arms:
        imgs = pages if arm == "image" else []
        rec = claude_eval.transcribe(imgs, SYSTEM_NONE if arm == "none" else SYSTEM,
                                     prompt_for(arm, doc, qs, len(pages)), effort=a.effort, timeout=1800.0)
        ans = parse_answers(rec.get("text", ""))
        per = [{"id": q["id"], "kind": q["kind"], "pos": q["pos"], "answer": q["answer"], "given": ans.get(q["id"]),
                "correct": ans.get(q["id"]) is not None and norm_num(ans[q["id"]]) == q["answer"]} for q in qs]
        row = {"arm": arm, "run": a.run_name, "chars": len(doc), "budget": a.budget, "n_pages": len(pages),
               "image_tokens_formula": vt, "effort": a.effort, "model": rec.get("model"),
               "input_tokens": rec.get("input_tokens"), "output_tokens": rec.get("output_tokens"),
               "usage": rec.get("usage"), "cost_usd": rec.get("cost_usd"), "latency_s": rec.get("latency_s"),
               "cached": rec.get("cached"), "error": rec.get("error"),
               "acc": {k: float(np.mean([p["correct"] for p in per if p["kind"] == k])) for k in ("needle", "number")},
               "acc_all": float(np.mean([p["correct"] for p in per])), "per_question": per,
               "raw": rec.get("text", "")}
        rows.append(row)
        print(f"[qa] {arm}: needle {row['acc']['needle']:.2f}, number {row['acc']['number']:.2f}; input tokens "
              f"{row['input_tokens']:,}, output {row['output_tokens']:,}, ${row['cost_usd']:.2f}, "
              f"{row['latency_s']:.0f}s, error {row['error']}", flush=True)
        with open(out / "runs.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    by = {r["arm"]: r for r in rows}
    if "text" in by and "image" in by:
        t = [p["correct"] for p in by["text"]["per_question"]]
        i = [p["correct"] for p in by["image"]["per_question"]]
        m = mcnemar_exact(t, i)
        print(f"[qa] text vs image (paired, exact McNemar): text-only correct {m['text_only']}, image-only "
              f"{m['image_only']}, p = {m['p']:.3g}; tokens text {by['text']['input_tokens']:,} vs image "
              f"{by['image']['input_tokens']:,} ({by['text']['input_tokens'] / max(1, by['image']['input_tokens']):.1f}x)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
