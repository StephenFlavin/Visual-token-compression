#!/usr/bin/env python3
"""PreToolUse on Read: when a page image of the minified file costs fewer tokens than the text, rewrite Read's
file_path to that page (PNG). One page per Read; `offset` picks the page holding that line."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vbhook import common, policy, render
from vbhook.minify import minify

ev = json.load(sys.stdin)
ti = ev.get("tool_input", {})
p = ti.get("file_path", "")
if not p or p.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".ipynb")) or not os.path.isfile(p):
    sys.exit(0)
try:
    text = open(p, encoding="utf-8").read()
except (UnicodeDecodeError, OSError):
    sys.exit(0)
hint = None
base = os.path.basename(p)
if os.path.dirname(os.path.abspath(p)) == os.path.abspath(common.CACHE) and base.count(".") >= 2:
    hint = base.split(".")[-2]                      # saved command output: out_<hash>.<kind>.txt
if hint:                                             # saved command output: page by output line
    budget = render.DEFAULT_BUDGET
    start = max(1, int(ti.get("offset") or 1))
    win = common.output_window(text, start, hint, budget)
    if win is None:
        sys.exit(0)
    flow, m, n = win
    pg = render.render(flow, budget)[0]
    total = text.count("\n") + 1
    out = common.cache_path(f"{p}:{start}:{budget}", f".o{start}.png")
    pg.image.save(out)
    more = f" Read it with offset={start + n} for the next lines." if start + n <= total else ""
    common.log({"tool": "Read", "path": p, "decision": "image", "kind": m.kind, "image_tokens": pg.tokens,
                "lines": [start, start + n - 1]})
    common.emit({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "allow", "updatedInput": {"file_path": out},
        "additionalContext": (f"This Read shows output lines {start}-{start + n - 1} of {total} of a saved command "
                              f"output as an image ({m.kind}). " + common.FORMAT_NOTE + " " + common.legend(m.kind)
                              + more)}})
res = minify(text, p, hint)
if res is None:
    common.log({"tool": "Read", "path": p, "decision": "text", "why": "no minifier"})
    sys.exit(0)
flow, m = res
budget = render.DEFAULT_BUDGET
plan = render.plan(flow, budget)                       # page chunks; only the page shown is rendered
lines = text.split("\n")
want = max(1, int(ti.get("offset") or 1))
k = max([i for i, (_, fl) in enumerate(plan) if (fl or 1) <= want] or [0])
pg = render.render_one(plan[k][0], budget)
a = plan[k][1] or 1
b = (plan[k + 1][1] - 1) if k + 1 < len(plan) and plan[k + 1][1] else len(lines)
lim = ti.get("limit")
req_hi = min(len(lines), want + int(lim) - 1) if lim else min(len(lines), want + 1999)
span = "\n".join(lines[want - 1:req_hi])
t_tok = policy.text_tokens(span, m.cpt, read_format=True)
if not policy.use_image(pg.tokens, t_tok):
    common.log({"tool": "Read", "path": p, "decision": "text", "kind": m.kind, "image_tokens": pg.tokens,
                "text_tokens": round(t_tok)})
    sys.exit(0)
pages = plan
out = common.cache_path(f"{p}:{hash(text)}:{budget}:{k}", f".p{k + 1}.png")
pg.image.save(out)
nxt = (f" The file continues: Read it with offset={pages[k + 1][1]} for page {k + 2} of {len(pages)}."
       if k + 1 < len(pages) and pages[k + 1][1] else "")
common.log({"tool": "Read", "path": p, "decision": "image", "kind": m.kind, "page": k + 1, "pages": len(pages),
            "image_tokens": pg.tokens, "text_tokens": round(t_tok), "lines": [a, b]})
common.emit({"hookSpecificOutput": {
    "hookEventName": "PreToolUse", "permissionDecision": "allow", "updatedInput": {"file_path": out},
    "additionalContext": (f"This Read shows {base} lines {a}-{b} of {len(lines)} as an image (page {k + 1} of "
                          f"{len(pages)}). " + common.format_note(text) + " " + common.legend(m.kind) + nxt)}})
