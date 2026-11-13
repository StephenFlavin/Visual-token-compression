"""Page images against text on a LOCAL vision-language model (docs/gpu/LOCAL_QWEN.md).

The Claude QA test (docs/claude/CLAUDE_QA.md) pointed at an OpenAI-compatible server on this machine, e.g. llama.cpp's
llama-server running Qwen3.8-27B with its vision projector. Same documents, planted facts, questions and scoring;
smaller documents (200k chars, 100 questions: a 24 GB card holds far less context) and the questions sent in
batches after the document, so the server's prefix cache processes the document once per arm.

    python3 experiments/claude_qa/local_qa.py run --dry-run                       # sizes only, no server
    python3 experiments/claude_qa/local_qa.py run --doc 0 --arms text page118     # needs the server (--base-url)
    python3 experiments/claude_qa/local_qa.py run --doc 0 1 2                     # all arms: none text page118 page88 page59
    python3 experiments/claude_qa/local_qa.py speed --label plain                 # generation speed (run once per server setup)
    python3 experiments/claude_qa/local_qa.py analyse                             # table, verdicts -> results/local_qa/<run>/summary.md

Rows: results/local_qa/<run>/runs.jsonl (one per document and arm; resumable: finished arms are skipped) and
speed.jsonl. No Claude calls.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude_qa"))
import analyse_claude_qa as AN  # noqa: E402
import claude_qa_render as R  # noqa: E402
from vtc import config, metrics  # noqa: E402

PATCH = 32                     # Qwen3.8: 16-px patches merged 2x2 -> one vision token per 32 x 32 px
BUDGETS = {"page118": 117.6, "page88": 88.2, "page59": 58.8}     # px² per character, as in the Claude test
DOC_OFFSETS = (0, 400, 800)    # first paragraph of documents 0-2 (200k chars use ~320-360 paragraphs)
THINK = re.compile(r"<think>.*?</think>", re.S)
SPEED_PROMPT = ("Explain how a refrigerator works, step by step, in about 300 words. Then list five common faults "
                "and how to recognise each one.")


def vision_tokens(w: int, h: int, patch: int = PATCH) -> int:
    return math.ceil(w / patch) * math.ceil(h / patch)


def render_pages(doc: str, budget: float, page_px: int, patch: int = PATCH) -> tuple[list, list[dict]]:
    """Square pages of side ``page_px`` (a multiple of ``patch``), the last page cut to the height its text needs.
    claude_qa_render.render_pages with the token grid as a parameter."""
    from vtc.layout import render_page
    from vtc.schemes import get_scheme
    assert page_px % patch == 0
    sch = get_scheme("latin")
    max_chars = int(page_px * page_px / budget * 0.97)
    while True:
        pages, info, ok = [], [], True
        for c in R.chunk(doc, max_chars):
            n = len(metrics.normalize(c))
            H = min(page_px, max(2 * patch, math.ceil(n * budget / page_px / patch) * patch))
            img, inf = render_page(sch.encode(c), page_px, H)
            if not inf["fit"]:
                ok = False
                break
            pages.append(img)
            info.append({"W": page_px, "H": H, "chars": n, "em": inf["em"], "px_per_char": page_px * H / n,
                         "vision_tokens": vision_tokens(page_px, H, patch)})
        if ok:
            return pages, info
        max_chars = int(max_chars * 0.95)


def data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def user_content(arm: str, doc: str, urls: list[str], qs: list[dict]) -> list[dict]:
    """The document first (identical for every batch of an arm, so the server reuses its processed prefix), then
    this batch's questions. Same wording as the Claude test's prompts."""
    ql = "\n".join(f"{q['id']}: {q['question']}" for q in qs)
    if arm == "text":
        return [{"type": "text", "text": f"<document>\n{doc}\n</document>\n\nQuestions:\n{ql}"}]
    if arm == "none":
        return [{"type": "text", "text": f"Questions:\n{ql}"}]
    parts = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    return parts + [{"type": "text", "text": f"The document is the {len(urls)} page images above, in reading order "
                                             f"(the text runs on from one page to the next).\n\nQuestions:\n{ql}"}]


def chat(base_url: str, model: str, system: str, content: list[dict], max_tokens: int, extra: dict,
         timeout: float) -> dict:
    """One /v1/chat/completions call -> {text, reasoning, usage, timings, latency_s, error}."""
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "cache_prompt": True,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}], **extra}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}", "latency_s": time.time() - t0}
    except Exception as e:   # noqa: BLE001  (connection refused, timeout: reported in the row, never cached)
        return {"error": f"{type(e).__name__}: {e}", "latency_s": time.time() - t0}
    msg = out["choices"][0]["message"]
    return {"text": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
            "finish": out["choices"][0].get("finish_reason"), "usage": out.get("usage") or {},
            "timings": out.get("timings") or {}, "latency_s": time.time() - t0, "error": None}


def build(doc_i: int, chars: int, seed: int):
    paras = R.build_document(chars, DOC_OFFSETS[doc_i])
    # 40 invented facts, 40 numbers in real sentences, 20 multi-step (two-hop, sums, largest-of-three, counts):
    # the 40-question multi-step set of the Claude test needs a document of 300k chars or more
    return R.make_questions(paras, 40, 40, seed + doc_i, 20)


def run_arm(a, arm: str, doc: str, qs: list[dict], pages: list, pinfo: list[dict]) -> dict:
    urls = [data_url(p) for p in pages]
    system = R.SYSTEM_NONE if arm == "none" else R.SYSTEM
    answers, calls, raw = {}, [], []
    for i in range(0, len(qs), a.questions_per_call):
        batch = qs[i:i + a.questions_per_call]
        rec = chat(a.base_url, a.model, system, user_content(arm, doc, urls, batch), a.max_tokens, a.extra,
                   a.timeout)
        if rec["error"]:
            return {"error": rec["error"]}
        text = THINK.sub("", rec["text"])
        answers.update(R.parse_answers(text))
        raw.append(text)
        t = rec["timings"]
        calls.append({"prompt_tokens": rec["usage"].get("prompt_tokens"),
                      "completion_tokens": rec["usage"].get("completion_tokens"), "finish": rec["finish"],
                      "latency_s": round(rec["latency_s"], 2), "prompt_n": t.get("prompt_n"),
                      "cache_n": t.get("cache_n"), "prompt_per_second": t.get("prompt_per_second"),
                      "predicted_n": t.get("predicted_n"), "predicted_per_second": t.get("predicted_per_second"),
                      "reasoning_chars": len(rec["reasoning"])})
        print(f"[local] {arm} questions {i + 1}-{i + len(batch)}: prompt {calls[-1]['prompt_tokens']} tokens, "
              f"{calls[-1]['latency_s']:.0f}s", flush=True)
    per = [{"id": q["id"], "kind": q["kind"], "sub": q.get("sub"), "pos": q["pos"], "answer": q["answer"],
            "given": answers.get(q["id"]),
            "correct": answers.get(q["id"]) is not None and R.norm_num(answers[q["id"]]) == q["answer"]} for q in qs]
    vt = sum(p["vision_tokens"] for p in pinfo)
    tok = max((c["prompt_tokens"] or 0) for c in calls)
    gen = [c["predicted_per_second"] for c in calls if c["predicted_per_second"]]
    kinds = sorted({q["kind"] for q in qs})
    subs = sorted({q["sub"] for q in qs if q.get("sub")})
    return {"arm": arm, "run": a.run_name, "chars": len(doc), "budget": BUDGETS.get(arm),
            "n_pages": len(pages), "image_tokens_formula": vt, "model": a.model,
            "input_tokens": tok, "output_tokens": sum(c["completion_tokens"] or 0 for c in calls),
            # every page must reach the model at full size: fewer prompt tokens than the pages need = downscaled
            "incomplete": bool(pages and tok < 0.97 * vt),
            "latency_s": sum(c["latency_s"] for c in calls), "prefill_tokens_per_s": calls[0]["prompt_per_second"],
            "gen_tokens_per_s": float(np.mean(gen)) if gen else None, "calls": calls, "error": None,
            "acc": {k: float(np.mean([p["correct"] for p in per if p["kind"] == k])) for k in kinds},
            "acc_sub": {k: float(np.mean([p["correct"] for p in per if p["sub"] == k])) for k in subs},
            "acc_all": float(np.mean([p["correct"] for p in per])), "n_questions": len(qs),
            "per_question": per, "raw": "\n".join(raw), "page_px": a.page_px, "extra": a.extra,
            "questions_per_call": a.questions_per_call}


def cmd_run(a) -> int:
    out = config.RESULTS / "local_qa" / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    rows_f = out / "runs.jsonl"
    done = set()
    if rows_f.exists():
        done = {(r["doc"], r["arm"]) for r in map(json.loads, rows_f.read_text().splitlines()) if not r.get("error")}
    for d in a.doc:
        doc, qs = build(d, a.chars, a.seed)
        (out / f"questions_d{d}.json").write_text(json.dumps(qs, indent=1))
        print(f"[local] document {d}: {len(doc):,} chars, {len(qs)} questions, text ~{len(doc) / 4.3:,.0f} tokens "
              "(estimate at 4.3 chars per token)", flush=True)
        for arm in a.arms:
            pages, pinfo = render_pages(doc, BUDGETS[arm], a.page_px) if arm in BUDGETS else ([], [])
            if pages:
                vt = sum(p["vision_tokens"] for p in pinfo)
                print(f"[local]   {arm}: {len(pages)} pages, {vt:,} vision tokens, "
                      f"{len(doc) / vt:.1f} chars per vision token, largest page {max(p['vision_tokens'] for p in pinfo)}"
                      " tokens", flush=True)
            if a.dry_run or (d, arm) in done:
                continue
            row = run_arm(a, arm, doc, qs, pages, pinfo)
            row["doc"] = d
            if row.get("error"):
                print(f"[local] {arm} document {d}: ERROR {row['error']} (not recorded; rerun to retry)", flush=True)
                return 1
            with open(rows_f, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[local] {arm} document {d}: all {row['acc_all']:.3f} "
                  + ", ".join(f"{k} {v:.2f}" for k, v in row["acc"].items())
                  + f"; input {row['input_tokens']:,} tokens, prefill {row['prefill_tokens_per_s']} tok/s, "
                  f"generation {row['gen_tokens_per_s']} tok/s, {row['latency_s']:.0f}s"
                  + ("; INVALID: fewer prompt tokens than the pages need (downscaled or dropped)"
                     if row["incomplete"] else ""), flush=True)
    return 0


def cmd_speed(a) -> int:
    out = config.RESULTS / "local_qa" / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    rates = []
    for i in range(a.repeats):
        rec = chat(a.base_url, a.model, "You are a helpful assistant.",
                   [{"type": "text", "text": SPEED_PROMPT + f" (run {i + 1})"}], a.max_tokens, a.extra, a.timeout)
        if rec["error"]:
            print(f"[local] speed: ERROR {rec['error']}", flush=True)
            return 1
        t, n = rec["timings"], rec["usage"].get("completion_tokens") or 0
        rate = t.get("predicted_per_second") or (n / rec["latency_s"] if rec["latency_s"] else None)
        rates.append(rate)
        print(f"[local] speed {a.label} run {i + 1}: {n} tokens, {rate:.1f} tok/s", flush=True)
    row = {"label": a.label, "model": a.model, "rates": rates, "mean_tokens_per_s": float(np.mean(rates)),
           "extra": a.extra, "max_tokens": a.max_tokens}
    with open(out / "speed.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[local] speed {a.label}: {row['mean_tokens_per_s']:.1f} tok/s (mean of {a.repeats})", flush=True)
    return 0


LOOKUP = ("needle", "number")


def summarise(rows: list[dict], speed: list[dict]) -> str:
    """Markdown: one line per arm (pooled over documents), paired with the text arm of the same document."""
    by = {(r["doc"], r["arm"]): r for r in rows}
    docs = sorted({r["doc"] for r in rows})
    arms = [x for x in ("none", "text", "page118", "page88", "page59") if any((d, x) in by for d in docs)]
    out = ["| arm | documents | input tokens (mean) | fewer than text | chars per vision token | all | lookups | "
           "multi-step | arm − text, all (95% CI) | lookups | multi-step | unanswered | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    notes = []
    for arm in arms:
        rs = [by[(d, arm)] for d in docs if (d, arm) in by]
        per = [p for r in rs for p in r["per_question"]]
        acc = lambda f: np.mean([p["correct"] for p in per if f(p)])   # noqa: E731
        tok = np.mean([r["input_tokens"] for r in rs])
        cells = {"all": lambda p: True, "lookups": lambda p: p["kind"] in LOOKUP,
                 "multi": lambda p: p["kind"] == "complex"}
        diffs, verdict, ratio = ["", "", ""], "", ""
        paired = [d for d in docs if (d, arm) in by and (d, "text") in by]
        if arm not in ("text",) and paired:
            for j, f in enumerate(cells.values()):
                pairs = [(str(d), t["correct"], i["correct"]) for d in paired
                         for t, i in zip(by[(d, "text")]["per_question"], by[(d, arm)]["per_question"]) if f(t)]
                delta, lo, hi = AN.paired_diff_ci(pairs)        # text − arm
                diffs[j] = f"{-100 * delta:+.1f} ({-100 * hi:+.1f} to {-100 * lo:+.1f})"
                if j == 0 and arm != "none":
                    verdict = "lossless" if 100 * hi < 2 else ("loss" if 100 * lo > 0 else "not shown lossless")
            ratio = f"{np.mean([by[(d, 'text')]['input_tokens'] for d in paired]) / tok:.2f}x"
        if any(r["incomplete"] for r in rs):
            verdict = "INVALID (pages downscaled or dropped)"
        cpt = (f"{np.mean([r['chars'] / r['image_tokens_formula'] for r in rs]):.1f}" if arm in BUDGETS else "")
        unanswered = sum(AN.engagement(r)[0] for r in rs)
        out.append(f"| {arm} | {len(rs)} | {tok:,.0f} | {ratio} | {cpt} | {100 * acc(cells['all']):.1f}% | "
                   f"{100 * acc(cells['lookups']):.1f}% | {100 * acc(cells['multi']):.1f}% | {diffs[0]} | {diffs[1]} "
                   f"| {diffs[2]} | {unanswered} | {verdict} |")
        pre = [r["prefill_tokens_per_s"] for r in rs if r.get("prefill_tokens_per_s")]
        gen = [r["gen_tokens_per_s"] for r in rs if r.get("gen_tokens_per_s")]
        if pre or gen:
            notes.append(f"* {arm}: prompt processing {np.mean(pre):,.0f} tokens/s, generation "
                         f"{np.mean(gen):.1f} tokens/s, {np.mean([r['latency_s'] for r in rs]):.0f} s per document"
                         if pre and gen else f"* {arm}: {np.mean([r['latency_s'] for r in rs]):.0f} s per document")
    md = ["# Page images on a local model (local_qa.py)", "\n".join(out),
          "Verdict rule (as in docs/claude/CLAUDE_QA.md): lossless = the 95% upper bound of the loss against text is "
          "below 2 points; loss = the interval excludes 0.", "## Speed", "\n".join(notes) or "(no timings returned)"]
    if speed:
        md.append("\n".join(f"* generation, {s['label']}: {s['mean_tokens_per_s']:.1f} tokens/s "
                            f"({', '.join(f'{x:.1f}' for x in s['rates'])})" for s in speed))
    return "\n\n".join(md) + "\n"


def cmd_analyse(a) -> int:
    out = config.RESULTS / "local_qa" / a.run_name
    rows = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines() if x.strip()]
    sp = out / "speed.jsonl"
    speed = [json.loads(x) for x in sp.read_text().splitlines() if x.strip()] if sp.exists() else []
    md = summarise(rows, speed)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "speed", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--run-name", default="qwen38_27b")
        p.add_argument("--base-url", default="http://127.0.0.1:8080")
        p.add_argument("--model", default="qwen3.8-27b", help="the name the server expects (llama-server ignores it)")
        p.add_argument("--extra", type=json.loads, default={},
                       help='JSON merged into every request, e.g. \'{"chat_template_kwargs": {"enable_thinking": false}}\'')
        p.add_argument("--timeout", type=float, default=3600.0)
        p.add_argument("--max-tokens", type=int, default=3000 if name == "run" else 600)
    r = sub.choices["run"]
    r.add_argument("--doc", type=int, nargs="+", default=[0], choices=[0, 1, 2])
    r.add_argument("--arms", nargs="+", default=["none", "text", "page118", "page88", "page59"],
                   choices=["none", "text", *BUDGETS])
    r.add_argument("--chars", type=int, default=200000,
                   help="document size: ~48k text tokens; the smallest the question generator supports")
    r.add_argument("--page-px", type=int, default=1600, help="page side, a multiple of 32 (1600 = 2500 tokens)")
    r.add_argument("--questions-per-call", type=int, default=20)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--dry-run", action="store_true")
    s = sub.choices["speed"]
    s.add_argument("--label", required=True, help="e.g. plain, mtp")
    s.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "speed": cmd_speed, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
