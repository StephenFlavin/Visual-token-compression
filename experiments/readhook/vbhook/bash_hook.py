#!/usr/bin/env python3
"""PostToolUse on Bash: show long output as a page image when that is cheaper than the text.

File dumps (cat / head / sed -n 'a,bp' / nl of one file) use the file's own minifier with real line numbers; other
output uses the output minifiers (kind from the command line, else sniffed). Output longer than one page: the first
page is shown and the full output is saved to a file the model can Read page by page."""
import base64, io, json, os, re, shlex, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vbhook import common, policy, render
from vbhook.minify import minify, REGISTRY

ev = json.load(sys.stdin)
cmd = (ev.get("tool_input") or {}).get("command", "")
r = ev.get("tool_response") or {}
if not isinstance(r, dict) or r.get("isImage"):
    sys.exit(0)
text = (r.get("stdout") or "") + (("\n" + r["stderr"]) if r.get("stderr") else "")
if len(text) < 400:                                   # an image costs >= ~140 tokens: short output stays text
    sys.exit(0)


def file_dump(c: str):
    """-> (path, first line) when the command prints (part of) one file, else None."""
    try:
        a = shlex.split(c)
    except ValueError:
        return None
    if any(x in c for x in ("|", ";", "&&", ">", "<")) or not a:
        return None
    files = [x for x in a[1:] if not x.startswith("-") and os.path.isfile(x)]
    if len(files) != 1:
        return None
    if a[0] in ("cat", "nl", "bat") or (a[0] == "head" and "-c" not in a):
        return files[0], 1
    if a[0] == "sed" and "-n" in a:
        m = re.search(r"(\d+)\s*,\s*\d+\s*p", c)
        return files[0], int(m.group(1)) if m else 1
    return None


res, path, first = None, None, 1
fd = file_dump(cmd)
if fd and not cmd.strip().startswith("nl"):
    path, first = fd
    res = minify(text, path)
if res is None:
    hint = None
    try:
        from vbhook.minify.outputs import kind_for_command
        hint = kind_for_command(cmd)
    except Exception:  # noqa: BLE001
        pass
    res = minify(text, None, hint)
if res is None:
    common.log({"tool": "Bash", "cmd": cmd[:120], "decision": "text", "why": "no minifier", "chars": len(text)})
    sys.exit(0)
flow, m = res
flow = common.shift_tags(flow, first)
pages = render.render(flow)
t_tok = policy.text_tokens(text, m.cpt)
first_page = pages[0]
covered = first_page.chars / max(1, len(flow))
if not policy.use_image(first_page.tokens, t_tok * min(1.0, covered)):
    common.log({"tool": "Bash", "cmd": cmd[:120], "decision": "text", "kind": m.kind,
                "image_tokens": first_page.tokens, "text_tokens": round(t_tok)})
    sys.exit(0)
buf = io.BytesIO()
first_page.image.save(buf, format="PNG")
note = f"The command's output is shown as an image ({m.kind}). " + common.FORMAT_NOTE
if len(pages) > 1:
    saved = common.cache_path(cmd + text, f".{m.kind}.txt")
    saved = saved.replace(os.path.basename(saved), "out_" + os.path.basename(saved))
    open(saved, "w").write(text)
    nl = pages[1].first_line
    note += (f" Only the first of {len(pages)} pages is shown; the full output is saved at {saved}: Read it "
             + (f"with offset={nl} " if nl else "") + "for the rest.")
common.log({"tool": "Bash", "cmd": cmd[:120], "decision": "image", "kind": m.kind, "pages": len(pages),
            "image_tokens": first_page.tokens, "text_tokens": round(t_tok)})
common.emit({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": note,
             "updatedToolOutput": {"stdout": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                                   "stderr": "", "interrupted": bool(r.get("interrupted")), "isImage": True}}})
