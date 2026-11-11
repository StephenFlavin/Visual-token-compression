#!/usr/bin/env python3
"""PostToolUse on Bash: replace long command output (e.g. test failures, logs) with a rendered image.

Tested 2026-11-11 (Claude Code 2.1.322, Haiku 5.5): `updatedToolOutput` with a `data:image/png;base64,...` stdout
and `isImage: true` reaches the model as an image, and it read values from it correctly. (Rewriting the command in a
PreToolUse hook to a wrapper that prints the data URI also works.) Short output stays text: an image costs at
least ~140 tokens (a 1932 x 56 strip), more than a line like "2 passed in 0.01s".
"""
import base64, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_code import render
MIN_CHARS = int(os.environ.get("BASH_IMG_MIN_CHARS", "800"))
ev = json.load(sys.stdin)
r = ev.get("tool_response") or {}
if not isinstance(r, dict) or r.get("isImage"):
    sys.exit(0)
text = (r.get("stdout") or "") + (("\n" + r["stderr"]) if r.get("stderr") else "")
if len(text) < MIN_CHARS:
    sys.exit(0)
lines = text.splitlines()
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
    path = f.name
n, _ = render(lines, 1, path, header="command output, lines {a}-{b} of %d" % len(lines))
b64 = base64.b64encode(open(path, "rb").read()).decode()
os.unlink(path)
more = "" if n >= len(lines) else f" (only lines 1-{n} of {len(lines)} fit on the image)"
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
      "additionalContext": "The command's output is shown as an image" + more + ".",
      "updatedToolOutput": {"stdout": "data:image/png;base64," + b64, "stderr": "", "interrupted": bool(r.get("interrupted")),
                            "isImage": True}}}))
