#!/usr/bin/env python3
"""Image arm, PreToolUse on Edit and Write for .py files. The model saw minified code, so:
Edit: find old_string in the real file ignoring whitespace (and any copied § tags / » markers); if exactly one match,
replace old_string with the file's real text and re-indent new_string to the matched block's indentation.
Write: turn » markers into 4 spaces and drop § tags if the model copied them."""
import json, os, re, sys, textwrap
ev = json.load(sys.stdin)
tool, ti = ev.get("tool_name"), dict(ev.get("tool_input", {}))
p = ti.get("file_path", "")
log = os.environ.get("READHOOK_LOG")
if not p.endswith(".py"):
    sys.exit(0)
TAG = re.compile(r"§\d+ ?")

def unmark(s):
    s = TAG.sub("", s)
    return re.sub(r"(?m)^(\s*)(»+)", lambda m: m.group(1) + "    " * len(m.group(2)), s)

def emit(new_input, note):
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps({"tool": tool, "note": note}) + "\n")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                             "updatedInput": new_input}}))
    sys.exit(0)

if tool == "Write":
    ti["content"] = unmark(ti.get("content", ""))
    emit(ti, "write")
if tool != "Edit" or not os.path.isfile(p):
    sys.exit(0)
text = open(p, encoding="utf-8").read()
old = unmark(ti.get("old_string", "")).replace("»", "")
new = unmark(ti.get("new_string", ""))
if old in text and text.count(old) == 1:
    ti["old_string"], ti["new_string"] = old, new
    emit(ti, "exact")
toks = old.split()
if not toks:
    sys.exit(0)
pat = re.compile(r"\s*".join(map(re.escape, toks)))
ms = list(pat.finditer(text))
if len(ms) != 1:
    if log:
        with open(log, "a") as fh:
            fh.write(json.dumps({"tool": tool, "note": f"no unique match ({len(ms)})"}) + "\n")
    sys.exit(0)                                      # let Edit fail normally
m = ms[0]
ls = text.rfind("\n", 0, m.start()) + 1
prefix = text[ls:m.start()]
if prefix.strip() == "":                             # match starts a line: re-indent the replacement block
    body = textwrap.dedent(new.strip("\n"))
    ind = prefix
    new_block = "\n".join((ind + l) if l.strip() else l for l in body.split("\n"))
    ti["old_string"], ti["new_string"] = text[ls:m.end()], new_block
else:
    ti["old_string"], ti["new_string"] = text[m.start():m.end()], new
emit(ti, "fuzzy")
