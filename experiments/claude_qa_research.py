"""Claude QA, deep-research variant (user request, 2026-11-10): can a research orchestrator keep its sources as page
images, alongside short text notes, instead of as full text?

Each corpus is ~40 real Wikipedia articles (WikiText-103 test + validation, in order), each a "source" with a visible
header "[Source k: Title]", with invented facts planted inside them so that only the sources can answer. One call per
arm and corpus answers the same 120 questions (exact numbers, scored automatically):

* ``text``         all sources as text;
* ``pages``        the same string as 2.2x pages (117.6 px²/char, 1932-px pages, as Claude QA v2);
* ``notes``        research notes only: one call per source (same model, low effort) reads the source as text and
                   writes <= 250 words of notes, without seeing any question; the orchestrator gets all notes;
* ``notes_pages``  the notes as text plus all sources as 2.2x pages (the "notes + page archive" design);
* ``none``         nothing (the prior).

Questions per corpus (60 lookups, 60 synthesis):
* lookups: 30 invented facts (``needle``), 30 numbers in real sentences (``number``);
* synthesis, 12 each: ``citation`` (which source contains the sentence that begins "..."; answer = source number),
  ``conflict`` (two sources give different values for the same thing; answer = the difference), ``chain`` (person in a
  needle fact -> chairs a rowing club in another source -> its boats in a third), ``source_count`` (how many sources
  mention an organisation; 3-9), ``total`` (sum of an organisation's grants reported across 2-4 sources).
Cross-source facts sit in different sources at least 1/8 of the corpus apart.

    python3 experiments/claude_qa_research.py --dry-run          # build corpora, pages and questions; no calls
    python3 experiments/claude_qa_research.py --yes              # notes calls, then the 5 arms x 3 corpora
    python3 experiments/claude_qa_research.py --analyse          # tables from runs.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from vbpe import config, corpus  # noqa: E402
import claude_qa_render as Q  # noqa: E402
import analyse_claude_qa as A  # noqa: E402

OUT = config.RESULTS / "claude_qa" / "research"
BUDGET, PAGE_PX = 117.6, 1932
ARMS = ["none", "notes", "notes_pages", "pages", "text"]
LOOKUP = ("needle", "number")
SYNTH = ("citation", "conflict", "chain", "source_count", "total")
NOTES_WORDS = 250

SYSTEM = ("You answer questions about a set of sources. Use only the sources provided. Answer every question with "
          "the exact value only (a number), one line per question in the form 'ID: answer'. If you cannot find "
          "the answer, give your best guess.")
NOTES_SYSTEM = ("You are a research assistant. You write notes on one source at a time for a colleague who will later "
                "answer factual questions about many sources from your notes.")


# ----------------------------------------------------------------------------------------------- corpora
def articles() -> list[tuple[str, list[str]]]:
    """(title, cleaned paragraphs) for every article of the raw WikiText-103 test then validation split."""
    out = []
    for sp in ("test", "val"):
        for line in corpus._raw_split(sp):
            if line.startswith(" = ") and not line.startswith(" = = "):
                title = corpus.clean_paragraph(line.strip().strip("=").strip()) or line.strip(" =\n")
                out.append((title, []))
            elif out:
                p = corpus.clean_paragraph(line)
                if p and not line.lstrip().startswith("="):
                    out[-1][1].append(p)
    return [(t, ps) for t, ps in out if sum(map(len, ps)) > 2000]


def split_corpora(arts, n: int = 3) -> list[list[tuple[str, list[str]]]]:
    total = sum(sum(map(len, ps)) for _, ps in arts)
    target, out, cur, size = total / n, [], [], 0
    for a in arts:
        cur.append(a)
        size += sum(map(len, a[1]))
        if size >= target and len(out) < n - 1:
            out.append(cur)
            cur, size = [], 0
    out.append(cur)
    return out


# ----------------------------------------------------------------------------------------------- questions
def build(sources, seed: int):
    """-> (document string, per-source strings, questions)."""
    rng = np.random.default_rng([seed, 7])
    used: set = set()
    S = len(sources)
    far = max(2, S // 8)
    extra: dict[tuple[int, int], list[str]] = {}

    def put(fact: str, s: int) -> None:
        p = int(rng.integers(0, len(sources[s][1]) + 1))
        extra.setdefault((s, p), []).append(fact)

    def apart(k: int) -> list[int]:
        for _ in range(100000):
            ss = [int(x) for x in rng.choice(S, k, replace=False)]
            if all(abs(a - b) >= far for i, a in enumerate(ss) for b in ss[i + 1:]):
                return ss
        raise RuntimeError("no spread of sources")

    def number() -> int:
        n = int(rng.integers(120, 9800))
        while 1000 <= n <= 2100:
            n = int(rng.integers(120, 9800))
        return n

    qs = []
    needles = []
    for k in range(30):
        who, place = Q.invented(rng, used)
        n, t = number(), k % len(Q.TEMPLATES)
        fact, q = Q.TEMPLATES[t]
        s = int(rng.integers(0, S))
        f = fact.format(place=place, n=f"{n:,}", who=who)
        put(f, s)
        needles.append({"who": who, "place": place, "t": t, "s": s})
        qs.append({"id": f"N{k + 1:02d}", "kind": "needle", "answer": str(n), "anchors": [f], "srcs": [s],
                   "question": q.format(place=place, who=who)})
    for k in range(12):                                                     # conflict
        _, place = Q.invented(rng, used)
        who, _ = Q.invented(rng, used)
        a, b = apart(2)
        n1 = int(rng.integers(300, 9000))
        n2 = n1 + int(rng.choice([-1, 1])) * int(rng.integers(40, 900))
        f1 = f"The {place} reservoir holds {n1:,} megalitres of water."
        f2 = f"A later survey by {who} put the capacity of the {place} reservoir at {n2:,} megalitres."
        put(f1, a)
        put(f2, b)
        qs.append({"id": f"C{k + 1:02d}", "kind": "conflict", "answer": str(abs(n1 - n2)), "anchors": [f1, f2],
                   "srcs": [a, b], "question": f"The sources give two different capacities for the {place} reservoir. "
                                               "What is the difference between them, in megalitres?"})
    for k, x in enumerate([needles[int(i)] for i in rng.choice(len(needles), 12, replace=False)]):   # chain
        _, club = Q.invented(rng, used)
        m = int(rng.integers(12, 480))
        f1, f2 = f"{x['who']} chairs the {club} Rowing Club.", f"The {club} Rowing Club owns {m} boats."
        for _ in range(100000):
            a, b = apart(2)
            if min(abs(a - x["s"]), abs(b - x["s"])) >= far:
                break
        put(f1, a)
        put(f2, b)
        qs.append({"id": f"H{k + 1:02d}", "kind": "chain", "answer": str(m), "anchors": [f1, f2],
                   "srcs": [x["s"], a, b],
                   "question": f"How many boats does the rowing club chaired by {Q.HEAD[x['t']].format(place=x['place'])} "
                               "own?"})
    for k in range(12):                                                     # source_count
        _, org = Q.invented(rng, used)
        ss = [int(s) for s in rng.choice(S, int(rng.integers(3, 10)), replace=False)]
        fs = []
        for s in ss:
            who, _ = Q.invented(rng, used)
            fs.append(f"{who} was a fellow of the {org} Institute.")
            put(fs[-1], s)
        qs.append({"id": f"M{k + 1:02d}", "kind": "source_count", "answer": str(len(ss)), "anchors": fs,
                   "srcs": sorted(ss), "question": f"How many different sources mention the {org} Institute?"})
    for k in range(12):                                                     # total
        _, org = Q.invented(rng, used)
        ss = apart(int(rng.integers(2, 5)))
        fs, tot = [], 0
        for s in ss:
            who, _ = Q.invented(rng, used)
            g = int(rng.integers(150, 4000))
            tot += g
            fs.append(f"The {org} Trust gave {who} a grant of {g:,} crowns.")
            put(fs[-1], s)
        qs.append({"id": f"T{k + 1:02d}", "kind": "total", "answer": str(tot), "anchors": fs, "srcs": sorted(ss),
                   "question": f"What is the total, in crowns, of all the grants the {org} Trust gave according to the "
                               "sources?"})
    # assemble
    src_txt = []
    for s, (title, ps) in enumerate(sources):
        body = []
        for p_i in range(len(ps) + 1):
            body += extra.get((s, p_i), [])
            if p_i < len(ps):
                body.append(ps[p_i])
        src_txt.append(f"[Source {s + 1}: {title}] " + " ".join(body))
    doc = " ".join(src_txt)
    # real sentences: numbers (lookups) and citations, spread evenly over the corpus
    num_c, cit_c = [], []
    for s, (_, ps) in enumerate(sources):
        for sent in (x for p in ps for x in corpus.split_sentences(p)):
            words = sent.split()
            if len(words) < 12:
                continue
            nums = Q.NUM.findall(sent)
            m = Q.NUM.search(sent)
            if (len(nums) == 1 and len(Q.norm_num(nums[0]).replace(".", "")) >= 2
                    and not re.search(r"\d", Q.NUM.sub("", sent))):
                op = sent[:m.start()].split()
                if len(op) >= 6:
                    op = " ".join(op[:10])
                    if doc.count(op) == 1 and not Q.NUM.search(op):
                        num_c.append({"opening": op, "answer": Q.norm_num(nums[0]), "s": s, "pos": doc.find(op)})
                        continue
            op = " ".join(words[:10])
            if doc.count(op) == 1 and '"' not in op:
                cit_c.append({"opening": op, "s": s, "pos": doc.find(op)})
    num_c.sort(key=lambda c: c["pos"])
    for k, j in enumerate(np.linspace(0, len(num_c) - 1, 30).astype(int)):
        c = num_c[j]
        qs.append({"id": f"S{k + 1:02d}", "kind": "number", "answer": c["answer"], "anchors": [c["opening"]],
                   "srcs": [c["s"]], "question": f'Find the sentence that begins "{c["opening"]}". What number does '
                                                 "that sentence contain?"})
    by_src = {}
    for c in sorted(cit_c, key=lambda c: c["pos"]):
        by_src.setdefault(c["s"], []).append(c)
    picks = [by_src[s][len(by_src[s]) // 2] for s in sorted(by_src)]
    for k, j in enumerate(np.linspace(0, len(picks) - 1, 12).astype(int)):
        c = picks[j]
        qs.append({"id": f"R{k + 1:02d}", "kind": "citation", "answer": str(c["s"] + 1), "anchors": [c["opening"]],
                   "srcs": [c["s"]], "question": f'Which source contains the sentence that begins "{c["opening"]}"? '
                                                 "Answer with the source number."})
    for q in qs:
        assert all(doc.count(a) == 1 for a in q["anchors"]), (q["id"], q["anchors"])
        q["pos"] = max(doc.find(a) for a in q["anchors"]) / len(doc)
    return doc, src_txt, qs


# ----------------------------------------------------------------------------------------------- prompts
def notes_prompt(src: str) -> str:
    return (f"Write research notes on the source below, at most {NOTES_WORDS} words. Record its main claims and the "
            "specific details a later question could ask about: names of people, places and organisations, figures "
            "with their units, and dates. Start with the source's number and title as given in its header.\n\n"
            f"<source>\n{src}\n</source>")


def arm_prompt(arm: str, doc: str, notes: str, qs, n_pages: int) -> str:
    ql = "\n".join(f"{q['id']}: {q['question']}" for q in qs)
    pages = (f"The sources are the {n_pages} page images above, in reading order (the text runs on from one page to the "
             "next); each source starts with a bracketed header such as [Source 3: Title].")
    notes_block = f"<notes>\n{notes}\n</notes>"
    return {"text": f"<sources>\n{doc}\n</sources>\n\nQuestions:\n{ql}",
            "pages": f"{pages}\n\nQuestions:\n{ql}",
            "notes": ("Research notes on each source, written earlier from its full text, are below. The full sources "
                      f"are not available.\n\n{notes_block}\n\nQuestions:\n{ql}"),
            "notes_pages": (f"{pages} Research notes on each source, written earlier from its full text, are below."
                            f"\n\n{notes_block}\n\nQuestions:\n{ql}"),
            "none": f"Questions:\n{ql}"}[arm]


def parse(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*\**([NSCHMTR]\d{2})\**\s*[:\-]\s*(.+)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


# ----------------------------------------------------------------------------------------------- run
def prepare(effort_notes: str):
    corpora = split_corpora(articles())
    built = []
    for k, srcs in enumerate(corpora):
        d = OUT / f"corpus{k}"
        (d / "images").mkdir(parents=True, exist_ok=True)
        doc, src_txt, qs = build(srcs, seed=k)
        pages, pinfo = Q.render_pages(doc, BUDGET, PAGE_PX)
        for i, im in enumerate(pages):
            im.save(d / "images" / f"page_{i:02d}.png")
        (d / "questions.json").write_text(json.dumps(qs, indent=1))
        (d / "pages.json").write_text(json.dumps(pinfo, indent=1))
        (d / "sources.json").write_text(json.dumps([t for t, _ in srcs], indent=1))
        vt = sum(p["vision_tokens"] for p in pinfo)
        print(f"[dr] corpus {k}: {len(srcs)} sources, {len(doc):,} chars, {len(qs)} questions "
              f"({sum(q['kind'] in LOOKUP for q in qs)} lookups), {len(pages)} pages, {vt:,} image tokens", flush=True)
        built.append({"k": k, "doc": doc, "src": src_txt, "qs": qs, "pages": pages, "vt": vt})
    return built


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--effort", default="high")
    ap.add_argument("--notes-effort", default="low")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args(argv)
    if a.analyse:
        return analyse()
    built = prepare(a.notes_effort)
    if a.dry_run or not a.yes:
        print("[dr] dry run" if a.dry_run else "[dr] add --yes to call Claude", flush=True)
        return 0
    from vbpe import claude_eval as CE
    # 1. notes, one call per source, without the questions
    jobs = [{"images": [], "system": NOTES_SYSTEM, "prompt": notes_prompt(s), "effort": a.notes_effort,
             "timeout": 600.0, "meta": (b["k"], i)} for b in built for i, s in enumerate(b["src"])]
    recs = CE.transcribe_many(jobs, workers=8)
    bad = [r["meta"] for r in recs if r.get("error") or not r.get("text")]
    errs = sorted({str(r.get("error"))[:200] for r in recs if r.get("error")})
    assert not bad, f"notes failed ({len(bad)}): {bad[:5]} ...; errors: {errs[:3]}"
    for b in built:
        rs = [r for r in recs if r["meta"][0] == b["k"]]
        b["notes"] = "\n\n".join(r["text"].strip() for r in rs)
        b["notes_cost"] = sum(r.get("cost_usd") or 0 for r in rs)
        b["notes_out"] = sum(r.get("output_tokens") or 0 for r in rs)
        (OUT / f"corpus{b['k']}" / "notes.md").write_text(b["notes"])
        print(f"[dr] corpus {b['k']}: notes {len(b['notes'].split()):,} words, ${b['notes_cost']:.2f}", flush=True)
    # 2. the arms
    jobs = []
    for b in built:
        for arm in ARMS:
            imgs = b["pages"] if arm in ("pages", "notes_pages") else []
            jobs.append({"images": imgs, "system": Q.SYSTEM_NONE if arm == "none" else SYSTEM,
                         "prompt": arm_prompt(arm, b["doc"], b["notes"], b["qs"], len(b["pages"])),
                         "effort": a.effort, "timeout": 1800.0, "hires": True, "meta": (b["k"], arm)})
    recs = CE.transcribe_many(jobs, workers=a.workers)
    for r in recs:
        k, arm = r["meta"]
        b = built[k]
        ans = parse(r.get("text", ""))
        per = [{"id": q["id"], "kind": q["kind"], "answer": q["answer"], "given": ans.get(q["id"]), "pos": q["pos"],
                "correct": ans.get(q["id"]) is not None and Q.norm_num(ans[q["id"]]) == q["answer"]} for q in b["qs"]]
        img = arm in ("pages", "notes_pages")
        incomplete = bool(img and r.get("input_tokens") and r["input_tokens"] < 0.97 * b["vt"])
        row = {"corpus": k, "arm": arm, "effort": a.effort, "model": r.get("model"),
               "input_tokens": r.get("input_tokens"), "output_tokens": r.get("output_tokens"), "usage": r.get("usage"),
               "cost_usd": r.get("cost_usd"), "latency_s": r.get("latency_s"), "cached": r.get("cached"),
               "error": r.get("error"), "incomplete": incomplete, "image_tokens_formula": b["vt"] if img else 0,
               "notes_cost_usd": b["notes_cost"] if arm.startswith("notes") else 0.0,
               "notes_output_tokens": b["notes_out"] if arm.startswith("notes") else 0,
               "acc_all": float(np.mean([p["correct"] for p in per])), "per_question": per, "raw": r.get("text", "")}
        with open(OUT / "runs.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[dr] corpus {k} {arm}: {row['acc_all']:.3f}, input {row['input_tokens']:,}, ${row['cost_usd']:.2f}, "
              f"{row['latency_s']:.0f}s{', INCOMPLETE' if incomplete else ''}{', ' + row['error'] if row['error'] else ''}",
              flush=True)
    return analyse()


# ----------------------------------------------------------------------------------------------- analysis
def analyse() -> int:
    rows = [json.loads(x) for x in (OUT / "runs.jsonl").read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:                                  # later rows win (a re-run replaces an earlier one)
        by[(r["corpus"], r["arm"])] = r
    corp = sorted({k for k, _ in by})
    bad = [(k, a) for (k, a), r in by.items() if r.get("incomplete") or r.get("error") or A.engagement(r)[1]]
    L = ["# Claude QA, deep-research variant: results", "",
         f"{len(corp)} corpora x 120 questions; arms paired with `text` question by question (Newcombe 95% interval "
         "for arm - text, exact McNemar). Lossless = lower bound of (arm - text) above -2 points.", ""]
    if bad:
        L += [f"Flagged arms (excluded): {bad}", ""]
    groups = [("all", None), ("lookups", LOOKUP), ("synthesis", SYNTH)] + [(k, (k,)) for k in LOOKUP + SYNTH]
    L += ["| arm | input tokens (mean) | fewer tokens | list-price $ per corpus (incl. notes) | " +
          " | ".join(g for g, _ in groups[:3]) + " | arm - text, all (95% CI) | McNemar p | verdict |",
          "|---" * 10 + "|"]
    summary = {}
    t_tok = np.mean([by[(k, "text")]["input_tokens"] for k in corp])
    for arm in ARMS:
        rs = [by[(k, arm)] for k in corp if (k, arm) in by and (k, arm) not in bad]
        if not rs:
            continue
        acc = {}
        for g, kinds in groups:
            ok = [p["correct"] for r in rs for p in r["per_question"] if kinds is None or p["kind"] in kinds]
            acc[g] = (sum(ok) / len(ok), *A.wilson(sum(ok), len(ok)), len(ok))
        tok = float(np.mean([r["input_tokens"] for r in rs]))
        cost = float(np.mean([r["cost_usd"] + r.get("notes_cost_usd", 0) for r in rs]))
        diff = {}
        if arm != "text":
            for g, kinds in groups:
                pairs = [(k, pt["correct"], pa["correct"]) for k in corp if (k, arm) in by and (k, arm) not in bad
                         for pt, pa in zip(by[(k, "text")]["per_question"], by[(k, arm)]["per_question"])
                         if kinds is None or pt["kind"] in kinds]
                d, lo, hi = A.paired_diff_ci(pairs)
                n10, n01, p = A.mcnemar([t for _, t, _ in pairs], [i for _, _, i in pairs])
                diff[g] = {"arm_minus_text": -d, "lo": -hi, "hi": -lo, "text_only": n10, "arm_only": n01, "p": p}
        summary[arm] = {"acc": acc, "tokens": tok, "saving": t_tok / tok, "cost": cost, "diff": diff,
                        "latency": float(np.median([r["latency_s"] for r in rs if not r.get("cached")] or [0]))}
        if arm == "text":
            ci, p, verdict = "-", "-", "reference"
        else:
            x = diff["all"]
            ci = f"{100 * x['arm_minus_text']:+.1f} ({100 * x['lo']:+.1f} to {100 * x['hi']:+.1f})"
            p = f"{x['p']:.2g}"
            verdict = ("lossless" if x["lo"] > -A.MARGIN else
                       "loss" if x["hi"] < 0 and x["p"] < 0.05 else "not shown lossless")
        summary[arm]["verdict"] = verdict
        L.append(f"| {arm} | {tok:,.0f} | {t_tok / tok:.2f}x | {cost:.2f} | "
                 + " | ".join(f"{100 * acc[g][0]:.1f}%" for g, _ in groups[:3]) + f" | {ci} | {p} | {verdict} |")
    L += ["", "Per question type (accuracy, pooled; n per type in brackets):", "",
          "| arm | " + " | ".join(f"{g} ({summary['text']['acc'][g][3]})" for g, _ in groups[3:]) + " |",
          "|---" * (1 + len(groups) - 3) + "|"]
    for arm, s in summary.items():
        L.append(f"| {arm} | " + " | ".join(f"{100 * s['acc'][g][0]:.0f}%" for g, _ in groups[3:]) + " |")
    L += ["", "Arm - text by group (points, 95% CI):", "", "| arm | lookups | synthesis |", "|---|---|---|"]
    for arm, s in summary.items():
        if arm != "text":
            L.append(f"| {arm} | " + " | ".join(
                f"{100 * s['diff'][g]['arm_minus_text']:+.1f} ({100 * s['diff'][g]['lo']:+.1f} to "
                f"{100 * s['diff'][g]['hi']:+.1f})" for g in ("lookups", "synthesis")) + " |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
