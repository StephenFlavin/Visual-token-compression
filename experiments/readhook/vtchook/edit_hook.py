#!/usr/bin/env python3
"""PreToolUse on Edit and Write. The model may have seen the file minified, so:
Edit: if old_string is not found verbatim, find it ignoring whitespace (and any copied § tags / » markers); with
exactly one match, use the file's real text as old_string and re-indent new_string to the matched block.
Write: turn copied » markers into the file's indent unit and drop § tags."""
import json, math, os, re, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vtchook import common

ev = json.load(sys.stdin)
tool, ti = ev.get("tool_name"), dict(ev.get("tool_input", {}))
p = ti.get("file_path", "")
TAG = re.compile(r"§\d+ ?")


def indent_unit(text: str) -> str:
    if "\n\t" in text:
        return "\t"
    steps = [len(l) - len(l.lstrip(" ")) for l in text.splitlines() if l.strip() and l.startswith(" ")]
    g = 0
    for s in steps:
        g = math.gcd(g, s)
    return " " * (g if g in (2, 3, 4, 8) else 4)


def unmark(s: str, unit: str) -> str:
    s = TAG.sub("", s)
    return re.sub(r"(?m)^([ \t]*)(»+)", lambda m: m.group(1) + unit * len(m.group(2)), s)


existing = open(p, encoding="utf-8").read() if os.path.isfile(p) else ""
unit = indent_unit(existing)
if tool == "Write":
    c = ti.get("content", "")
    if "§" in c or "»" in c:
        ti["content"] = unmark(c, unit)
        common.log({"tool": "Write", "path": p, "note": "unmarked"})
        common.emit({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                            "updatedInput": ti}})
    sys.exit(0)
if tool != "Edit" or not existing:
    sys.exit(0)
old = unmark(ti.get("old_string", ""), unit).replace("»", "")
new = unmark(ti.get("new_string", ""), unit)
if existing.count(old) == 1:
    if old != ti.get("old_string") or new != ti.get("new_string"):
        ti["old_string"], ti["new_string"] = old, new
        common.emit({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                            "updatedInput": ti}})
    common.log({"tool": "Edit", "path": p, "note": "exact"})
    sys.exit(0)
chars = re.sub(r"\s+", "", old)                      # minified views drop spaces inside lines too (a+b)
if not chars or ti.get("replace_all"):
    sys.exit(0)
ms = list(re.finditer(r"\s*".join(map(re.escape, chars)), existing))
if len(ms) != 1:
    common.log({"tool": "Edit", "path": p, "note": f"no unique match ({len(ms)})"})
    sys.exit(0)                                       # Edit fails normally and the model retries
m = ms[0]
ls = existing.rfind("\n", 0, m.start()) + 1
prefix = existing[ls:m.start()]
if prefix.strip() == "":                              # match starts its line: re-indent the replacement block
    body = textwrap.dedent(new.strip("\n"))
    ti["old_string"] = existing[ls:m.end()]
    ti["new_string"] = "\n".join((prefix + l) if l.strip() else l for l in body.split("\n"))
else:
    ti["old_string"], ti["new_string"] = existing[m.start():m.end()], new
common.log({"tool": "Edit", "path": p, "note": "fuzzy"})
common.emit({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow", "updatedInput": ti}})
