#!/usr/bin/env python3
"""PreToolUse on Bash: make the call succeed whatever the command's exit status, so that the PostToolUse hook (which
does not run for failed calls) can turn long output into an image. The status is appended to the output as
`[exit code N]` (removed again by the PostToolUse hook when N is 0). Only the command is rewritten: no permission decision is made here, so the user's permission rules
apply as before."""
import json, sys
ev = json.load(sys.stdin)
ti = dict(ev.get("tool_input") or {})
cmd = ti.get("command", "")
if not cmd.strip() or ti.get("run_in_background") or "[exit code $?]" in cmd:
    sys.exit(0)
ti["command"] = f'{cmd}\necho "[exit code $?]"'     # a line of its own (heredocs); the post hook strips code 0
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": ti}}))
