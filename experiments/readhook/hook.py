#!/usr/bin/env python3
"""PreToolUse hook on Read: render the requested lines of a text file to a PNG and point Read at it."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_code import render
CACHE = os.environ.get("READHOOK_CACHE", "/tmp/readhook_cache")
LOG = os.environ.get("READHOOK_LOG")
ev = json.load(sys.stdin)
ti = ev.get("tool_input", {})
p = ti.get("file_path", "")
if not p or p.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".ipynb")) or not os.path.isfile(p):
    sys.exit(0)
try:
    lines = open(p, encoding="utf-8").read().splitlines()
except (UnicodeDecodeError, OSError):
    sys.exit(0)
start = max(1, int(ti.get("offset") or 1))
os.makedirs(CACHE, exist_ok=True)
key = hashlib.sha1(f"{p}:{os.path.getmtime(p)}:{start}".encode()).hexdigest()[:16]
out = os.path.join(CACHE, f"{os.path.basename(p)}.{start}.{key}.png")
n, size = render(lines[start - 1:], start, out,
                 header=f"{p}  lines {{a}}-{{b}} of {len(lines)}")
end = start + n - 1
more = f" The file continues: Read it with offset={end + 1} for the next lines." if end < len(lines) else ""
if LOG:
    with open(LOG, "a") as fh:
        fh.write(json.dumps({"path": p, "start": start, "end": end, "of": len(lines), "png": out, "size": size}) + "\n")
print(json.dumps({"hookSpecificOutput": {
    "hookEventName": "PreToolUse", "permissionDecision": "allow",
    "updatedInput": {"file_path": out},
    "additionalContext": f"This Read shows {p} as an image of lines {start}-{end} of {len(lines)}.{more}"}}))
