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
res = minify(text, p, hint)
if res is None:
    common.log({"tool": "Read", "path": p, "decision": "text", "why": "no minifier"})
    sys.exit(0)
flow, m = res
budget = render.DEFAULT_BUDGET
pages = render.render(flow, budget)
lines = text.split("\n")
want = max(1, int(ti.get("offset") or 1))
k = max([i for i, pg in enumerate(pages) if (pg.first_line or 1) <= want] or [0])
pg = pages[k]
a = pg.first_line or 1
b = (pages[k + 1].first_line - 1) if k + 1 < len(pages) and pages[k + 1].first_line else len(lines)
lim = ti.get("limit")
req_hi = min(len(lines), want + int(lim) - 1) if lim else min(len(lines), want + 1999)
span = "\n".join(lines[want - 1:req_hi])
t_tok = policy.text_tokens(span, m.cpt, read_format=True)
if not policy.use_image(pg.tokens, t_tok):
    common.log({"tool": "Read", "path": p, "decision": "text", "kind": m.kind, "image_tokens": pg.tokens,
                "text_tokens": round(t_tok)})
    sys.exit(0)
out = common.cache_path(f"{p}:{hash(text)}:{budget}:{k}", f".p{k + 1}.png")
pg.image.save(out)
nxt = (f" The file continues: Read it with offset={pages[k + 1].first_line} for page {k + 2} of {len(pages)}."
       if k + 1 < len(pages) and pages[k + 1].first_line else "")
common.log({"tool": "Read", "path": p, "decision": "image", "kind": m.kind, "page": k + 1, "pages": len(pages),
            "image_tokens": pg.tokens, "text_tokens": round(t_tok), "lines": [a, b]})
common.emit({"hookSpecificOutput": {
    "hookEventName": "PreToolUse", "permissionDecision": "allow", "updatedInput": {"file_path": out},
    "additionalContext": (f"This Read shows {base} lines {a}-{b} of {len(lines)} as an image (page {k + 1} of "
                          f"{len(pages)}). " + common.FORMAT_NOTE + nxt)}})
