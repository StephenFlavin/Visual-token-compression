#!/usr/bin/env python3
"""Text arm, PreToolUse on Read: log only (path and size), so both arms record what was read."""
import json, os, sys
ev = json.load(sys.stdin)
p = ev.get("tool_input", {}).get("file_path", "")
if os.path.isfile(p):
    src = open(p, encoding="utf-8", errors="replace").read()
    with open(os.environ["READHOOK_LOG"], "a") as fh:
        fh.write(json.dumps({"path": p, "src_chars": len(src), "lines": src.count("\n") + 1,
                             "read_text_tokens_est": round((len(src) + 7 * (src.count("\n") + 1)) / 2.65)}) + "\n")
