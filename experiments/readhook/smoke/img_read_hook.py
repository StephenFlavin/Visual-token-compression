#!/usr/bin/env python3
"""Image arm, PreToolUse on Read: show a .py file as pages of minified code (pymin, comments kept) rendered by the
Phase 1 / Claude QA renderer at 117.6 px²/char on 1932-px pages; rewrite Read's file_path to the page holding the
requested line. Other files are left alone."""
import hashlib, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "experiments")]
ev = json.load(sys.stdin)
ti = ev.get("tool_input", {})
p = ti.get("file_path", "")
if not p.endswith(".py") or not os.path.isfile(p):
    sys.exit(0)
src = open(p, encoding="utf-8").read()
import pymin
try:
    flow = pymin.reflow(src)
except Exception:                                  # unparsable (e.g. after a bad edit): indentation by spaces
    parts = []
    for no, l in enumerate(src.splitlines(), 1):
        if l.strip():
            ind = len(l) - len(l.lstrip(" "))
            parts.append(f"§{no} {'»' * (ind // 4)}{l.strip()}")
    flow = " ".join(parts)
from vbpe import config, metrics
from vbpe.layout import render_page
from vbpe.schemes import get_scheme
import claude_qa_render as Q
BUDGET, W = 117.6, 1932
sch = get_scheme("latin")
cache = os.environ["READHOOK_CACHE"]
os.makedirs(cache, exist_ok=True)
max_chars = int(W * W / BUDGET * 0.97)
while True:
    chunks = Q.chunk(flow, max_chars)
    pages, ok = [], True
    for c in chunks:
        n = len(metrics.normalize(c))
        H = min(W, max(56, -(-int(n * BUDGET / W) // 28) * 28 + 28))
        img, inf = render_page(sch.encode(c), W, H)
        if not inf["fit"]:
            ok = False
            break
        pages.append(img)
    if ok:
        break
    max_chars = int(max_chars * 0.95)
first = [int(m.group(1)) if (m := re.search(r"§(\d+) ", c)) else 1 for c in chunks]
want = max(1, int(ti.get("offset") or 1))
k = max([i for i, f in enumerate(first) if f <= want] or [0])
key = hashlib.sha1(src.encode()).hexdigest()[:12]
out = os.path.join(cache, f"{os.path.basename(p)}.{key}.p{k + 1}.png")
pages[k].save(out)
nxt = f" The file continues on page {k + 2}: Read it with offset={first[k + 1]}." if k + 1 < len(pages) else ""
with open(os.environ["READHOOK_LOG"], "a") as fh:
    fh.write(json.dumps({"path": p, "page": k + 1, "pages": len(pages), "size": pages[k].size,
                         "vision_tokens": metrics.vision_tokens(*pages[k].size), "src_chars": len(src)}) + "\n")
ctx = (f"This Read shows {os.path.basename(p)} as an image (page {k + 1} of {len(pages)}) of minified Python: each "
       "line starts with §<line number>; each » before the code is one indentation level (4 spaces); blank lines "
       "are removed and continuation lines are joined to their line." + nxt +
       " When you Edit or Write, use normal Python with spaces; Edit matches old_string ignoring whitespace.")
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                         "updatedInput": {"file_path": out}, "additionalContext": ctx}}))
