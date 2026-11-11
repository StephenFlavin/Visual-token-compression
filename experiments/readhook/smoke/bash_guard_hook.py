#!/usr/bin/env python3
"""Both arms, PreToolUse on Bash: allow only `python3 -m pytest ...` (no chaining), so file contents cannot be
dumped as text (Claude Code auto-approves read-only commands such as cat and ls)."""
import json, re, sys
cmd = json.load(sys.stdin).get("tool_input", {}).get("command", "").strip()
if re.fullmatch(r"python3? -m pytest(?:[ \t]+[\w\-\.=/:\[\]]+)*", cmd):
    sys.exit(0)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                         "permissionDecisionReason": "Only `python3 -m pytest [options]` can run in "
                                         "this environment (no other commands, no chaining). Use Read to view files."}}))
