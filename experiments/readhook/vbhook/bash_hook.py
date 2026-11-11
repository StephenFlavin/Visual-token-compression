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
EXIT0 = re.compile(r"\n?\[exit code 0\]\n?$")             # added by bash_pre_hook; only non-zero codes matter
if EXIT0.search(r.get("stdout") or ""):
    r = dict(r, stdout=EXIT0.sub("", r["stdout"]))
    text = (r.get("stdout") or "") + (("\n" + r["stderr"]) if r.get("stderr") else "")
    if len(text) < 400:
        common.emit({"hookSpecificOutput": {"hookEventName": "PostToolUse", "updatedToolOutput": {
            "stdout": r["stdout"], "stderr": r.get("stderr") or "", "interrupted": bool(r.get("interrupted")),
            "isImage": False}}})
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


budget = render.DEFAULT_BUDGET
res, path, first = None, None, 1
fd = file_dump(cmd)
if fd and not cmd.strip().startswith("nl"):
    path, first = fd
    res = minify(text, path)
saved_note = ""
mixed = None
if res is not None:                                   # a file dump: tags are the file's line numbers
    flow, m = res
    flow = common.shift_tags(flow, first)
    pages = render.render(flow, budget)
    page, n_pages = pages[0], len(pages)
    if n_pages > 1 and pages[1].first_line:
        saved_note = (f" Only the first of {n_pages} pages is shown; Read {path} with offset={pages[1].first_line} "
                      "for the rest.")
    shown_frac = page.chars / max(1, len(flow))
if res is None:
    mixed = common.mixed_flow(cmd, text, ev.get("cwd") or os.getcwd())
if res is None and mixed is not None:                 # compound command that printed whole files
    flow, kinds = mixed
    pages = render.render(flow, budget)
    if len(pages) == 1:
        page = pages[0]
        t_tok = policy.text_tokens(text, 2.3)
        if policy.use_image(page.tokens, t_tok):
            buf = io.BytesIO()
            page.image.save(buf, format="PNG")
            note = ("The command's output is shown as an image: each file it printed starts with ▶ <path> and is "
                    "minified with its own line numbers (" + ", ".join(sorted(set(kinds))) + "). " + common.FORMAT_NOTE
                    + common.BASH_READ_NOTE)
            common.log({"tool": "Bash", "cmd": cmd[:120], "decision": "image", "kind": "mixed:" + "+".join(kinds),
                        "image_tokens": page.tokens, "text_tokens": round(t_tok)})
            common.emit({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": note,
                         "updatedToolOutput": {"stdout": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                                               "stderr": "", "interrupted": bool(r.get("interrupted")), "isImage": True}}})
if res is None:
    hint = None
    try:
        from vbhook.minify.outputs import kind_for_command
        hint = kind_for_command(cmd)
    except Exception:  # noqa: BLE001
        pass
    full = common.minify_output(text, hint)
    if full is None:
        common.log({"tool": "Bash", "cmd": cmd[:120], "decision": "text", "why": "no minifier", "chars": len(text)})
        sys.exit(0)
    m = full[1]
    win = common.output_window(text, 1, m.kind, budget)
    if win is None:
        sys.exit(0)
    flow, m, n = win
    pages = render.render(flow, budget)
    page, n_pages = pages[0], 1
    total = text.count("\n") + 1
    shown_frac = n / total
    if n < total:
        saved = common.cache_path(cmd + text, f".{m.kind}.txt")
        saved = os.path.join(os.path.dirname(saved), "out_" + os.path.basename(saved))
        open(saved, "w").write(text)
        saved_note = (f" Only output lines 1-{n} of {total} are shown; the full output is saved at {saved}: Read it "
                      f"with offset={n + 1} for the next lines.")
t_tok = policy.text_tokens(text, m.cpt) * min(1.0, shown_frac)
if not policy.use_image(page.tokens, t_tok):
    common.log({"tool": "Bash", "cmd": cmd[:120], "decision": "text", "kind": m.kind, "image_tokens": page.tokens,
                "text_tokens": round(t_tok)})
    sys.exit(0)
buf = io.BytesIO()
page.image.save(buf, format="PNG")
note = (f"The command's output is shown as an image ({m.kind}). " + common.FORMAT_NOTE + " " + common.legend(m.kind)
        + saved_note + (common.BASH_READ_NOTE if path else ""))
common.log({"tool": "Bash", "cmd": cmd[:120], "decision": "image", "kind": m.kind, "image_tokens": page.tokens,
            "text_tokens": round(t_tok)})
common.emit({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": note,
             "updatedToolOutput": {"stdout": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                                   "stderr": "", "interrupted": bool(r.get("interrupted")), "isImage": True}}})
