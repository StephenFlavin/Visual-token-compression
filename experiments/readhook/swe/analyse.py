"""Where a coding agent's tokens and dollars go, and how much page images can save (2026-11-11).

Reads every finished run of the Django (results/readhook_swe) and Multilingual (results/readhook_swe_ml) experiments
from its stream.jsonl. Prices are fitted exactly from the CLI's per-model cost (100 Django runs, max error 1e-15):
input $4, output $20, cache write $5 (1.25x), cache read $0.20 (0.05x) per million tokens (Opus 5.5, list).

Every API call re-sends the whole context; new content is paid once as a cache write and then as a cache read on each
later call. So a tool result of T tokens first sent at call k of N costs T x ($5 + $0.20 x (N - k)) in total, and the
cost of a run splits into: output, the starting context (system prompt, tools, issue: the first call), tool results
(what page images can shrink) and the rest (the agent's own messages, hook notes) re-sent as context.
Tool-result tokens: text by length / 2.6 chars per token (measured range for code and logs 2.1-2.9); images by the
image-token formula from their pixel size.
"""
from __future__ import annotations

import base64
import io
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCR = Path("/tmp/claude-0/-home-user-visual-bpe/9e274458-7352-5bb1-b191-52e96364b3f6/scratchpad/swe")
P_IN, P_OUT, P_W, P_R = 4e-6, 20e-6, 5e-6, 0.2e-6
CPT = 2.6


def runs():
    """(dataset, task, arm, stream path, row) for every recorded run."""
    out = []
    dj = {(json.loads(l)["id"], json.loads(l)["arm"]): json.loads(l)
          for l in open(ROOT / "results/readhook_swe/runs.jsonl")}
    for (i, a), r in dj.items():
        out.append(("django", i, a, SCR / "runs" / i / a / "stream.jsonl", r))
    ml = ROOT / "results/readhook_swe_ml/runs.jsonl"
    if ml.exists():
        for l in open(ml):
            r = json.loads(l)
            out.append(("rust/java", r["id"], r["arm"], SCR / "ml" / r["repo"].replace("/", "__") / "runs" / r["id"] / r["arm"]
                        / "stream.jsonl", r))
    return [x for x in out if x[3].exists()]


def img_tokens(b64: str) -> int:
    from PIL import Image
    w, h = Image.open(io.BytesIO(base64.b64decode(b64))).size
    return math.ceil(w / 28) * math.ceil(h / 28)


def analyse(path: Path) -> dict:
    ev = [json.loads(l) for l in open(path) if l.strip().startswith("{")]
    calls, seen = [], set()                     # one entry per API call: its usage
    results = []                                # (call index the result is first sent with, tokens, is_image)
    for e in ev:
        if e.get("type") == "assistant":
            m = e["message"]
            if m.get("id") not in seen and m.get("usage"):
                seen.add(m["id"])
                calls.append(m["usage"])
        elif e.get("type") == "user":
            for c in e["message"].get("content", []) if isinstance(e["message"].get("content"), list) else []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    body = c.get("content")
                    items = body if isinstance(body, list) else [{"type": "text", "text": str(body or "")}]
                    for it in items:
                        if it.get("type") == "image":
                            results.append((len(calls), img_tokens(it["source"]["data"]), True))
                        else:
                            results.append((len(calls), len(it.get("text", "")) / CPT, False))
    res = next((e for e in ev if e.get("type") == "result"), {})
    u = (res.get("modelUsage") or {}).get("claude-opus-5-5") or {}
    N = len(calls)
    w = sum(c.get("cache_creation_input_tokens", 0) for c in calls)
    r = sum(c.get("cache_read_input_tokens", 0) for c in calls)
    i = sum(c.get("input_tokens", 0) for c in calls)
    o = u.get("outputTokens", 0)
    first = calls[0] if calls else {}
    base = first.get("cache_creation_input_tokens", 0) + first.get("cache_read_input_tokens", 0) + first.get("input_tokens", 0)
    cost_out = o * P_OUT
    cost_in = w * P_W + r * P_R + i * P_IN
    cost_base = base * P_W * 0 + sum(min(base, c.get("cache_read_input_tokens", 0) + c.get("cache_creation_input_tokens", 0)
                                         + c.get("input_tokens", 0)) for c in calls) * P_R \
        + (first.get("cache_creation_input_tokens", 0) * P_W + first.get("input_tokens", 0) * P_IN)
    cost_tool = sum(t * (P_W + P_R * max(0, N - k)) for k, t, _ in results)
    tool_tok = sum(t for _, t, _ in results)
    img = [(k, t) for k, t, isimg in results if isimg]
    return {"calls": N, "cost": res.get("total_cost_usd") or 0.0, "cost_out": cost_out, "cost_in": cost_in,
            "cost_base": cost_base, "cost_tool": cost_tool, "cost_rest": max(0.0, cost_in - cost_base - cost_tool),
            "out": o, "write": w, "read": r, "uncached": i, "base": base, "tool_tokens": tool_tok,
            "image_tokens": sum(t for _, t in img), "image_cost": sum(t * (P_W + P_R * max(0, N - k)) for k, t in img),
            "thinking": u.get("thinkingTokens", 0)}


def main() -> int:
    if not SCR.is_dir():
        sys.exit(f"transcripts not found at {SCR} (they were in an earlier session's scratchpad and are gone); "
                 "not overwriting results/readhook_swe/analysis")
    rows = []
    for ds, i, a, p, row in runs():
        x = analyse(p)
        x.update(dataset=ds, id=i, arm=a, resolved=row.get("resolved"),
                 image_text_tokens=row.get("image_text_tokens", 0), image_tokens_hook=row.get("image_tokens", 0))
        rows.append(x)
    out = ROOT / "results" / "readhook_swe" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_run.json").write_text(json.dumps(rows, indent=1))
    L = []
    for ds in ("django", "rust/java", "all"):
        for arm in ("text", "vbhook"):
            R = [x for x in rows if x["arm"] == arm and (ds == "all" or x["dataset"] == ds)]
            if not R:
                continue
            C = sum(x["cost"] for x in R)
            s = lambda k: sum(x[k] for x in R)  # noqa: E731
            L.append({"dataset": ds, "arm": arm, "runs": len(R), "resolved": sum(bool(x["resolved"]) for x in R),
                      "cost": round(C, 2), "share_output": s("cost_out") / C, "share_start_context": s("cost_base") / C,
                      "share_tool_results": s("cost_tool") / C, "share_other_context": s("cost_rest") / C,
                      "median_calls": st.median(x["calls"] for x in R), "median_start_context_tokens": st.median(x["base"] for x in R),
                      "tool_result_tokens": round(s("tool_tokens")), "output_tokens": s("out"), "thinking_tokens": s("thinking"),
                      "image_tokens": s("image_tokens"), "image_cost": round(s("image_cost"), 2),
                      "image_text_tokens": s("image_text_tokens")})
    (out / "summary.json").write_text(json.dumps(L, indent=1))
    for d in L:
        print(f"{d['dataset']:9} {d['arm']:6} runs {d['runs']:3} resolved {d['resolved']:3} ${d['cost']:6.2f} | output "
              f"{100 * d['share_output']:4.1f}% start-ctx {100 * d['share_start_context']:4.1f}% tool-results "
              f"{100 * d['share_tool_results']:4.1f}% other-ctx {100 * d['share_other_context']:4.1f}% | calls "
              f"{d['median_calls']} start {d['median_start_context_tokens']:,.0f} tok | tool tok {d['tool_result_tokens']:,} "
              f"out {d['output_tokens']:,} (thinking {d['thinking_tokens']:,}) | images {d['image_tokens']:,} tok "
              f"(${d['image_cost']}) vs text {d['image_text_tokens']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
