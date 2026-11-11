"""Shared helpers for the hook scripts: paths, logging, caching, the format note shown to the model."""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), os.path.abspath(os.path.join(HERE, "..", "..", ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)

CACHE = os.environ.get("VBHOOK_CACHE") or os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "/tmp"), ".vbhook")
LOG = os.environ.get("VBHOOK_LOG")

FORMAT_NOTE = ("It is minified: §<n> marks the start of original line n; each » before a line is one level of "
               "indentation where indentation matters; other whitespace (indentation, blank lines, line breaks inside "
               "statements, spaces around operators) is removed. When you Edit or Write, use the file's real "
               "formatting; Edit matches old_string ignoring whitespace.")


def log(rec: dict) -> None:
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def cache_path(key: str, ext: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, hashlib.sha1(key.encode()).hexdigest()[:16] + ext)


def emit(obj: dict) -> None:
    print(json.dumps(obj))
    sys.exit(0)


def shift_tags(flow: str, first_line: int) -> str:
    """Renumber §n tags when the text started at line `first_line` of its file (e.g. sed -n '40,90p')."""
    if first_line == 1:
        return flow
    import re
    return re.sub(r"§(\d+) ", lambda m: f"§{int(m.group(1)) + first_line - 1} ", flow)


def legend(kind: str) -> str:
    """Extra notation of a kind (e.g. the caret brackets of tracebacks, the grep and diff layouts)."""
    try:
        from vbhook.minify.outputs import LEGENDS
        return LEGENDS.get(kind, "")
    except Exception:  # noqa: BLE001
        return ""


def output_kinds() -> list[str]:
    try:
        from vbhook.minify.outputs import LEGENDS
        return list(LEGENDS)
    except Exception:  # noqa: BLE001
        return []


def minify_output(text: str, hint: str | None):
    """Minify command output: the kind from the command line, else the first OUTPUT kind whose sniff matches (file
    sniffers such as the Dockerfile one must not claim output), else the generic log kind."""
    from vbhook.minify import REGISTRY, minify
    tried = []
    for k in ([hint] if hint else []) + [k for k in output_kinds() if REGISTRY[k].sniff and REGISTRY[k].sniff(text)] + ["log"]:
        if k in REGISTRY and k not in tried:
            tried.append(k)
            res = minify(text, None, k)
            if res:
                return res
    return None


def mixed_flow(cmd: str, text: str, cwd: str) -> tuple[str, list[str]] | None:
    """Output of a compound command that prints whole files (`ls && cat a.py && cat test*.py`): each file whose content
    appears verbatim gets its own minifier (tags = its line numbers) under a `▶ path` header; the rest is minified as
    output. -> (flow, kinds) or None when no file content is found."""
    import glob, shlex
    from vbhook.minify import minify
    try:
        toks = shlex.split(cmd.replace("\n", " ; "), posix=True)
    except ValueError:
        return None
    paths = []
    for t in toks:
        if t.startswith("-") or len(t) > 300:
            continue
        cands = glob.glob(os.path.join(cwd, t)) if any(c in t for c in "*?[") else [os.path.join(cwd, t)]
        for c in cands:
            if os.path.isfile(c) and c not in paths:
                paths.append(c)
    spans = []
    for pth in paths:
        try:
            body = open(pth, encoding="utf-8").read().rstrip("\n")
        except (OSError, UnicodeDecodeError):
            continue
        i = text.find(body) if len(body) >= 80 else -1
        if i >= 0 and not any(a < i + len(body) and i < b for a, b, _, _ in spans):
            spans.append((i, i + len(body), pth, body))
    if not spans:
        return None
    spans.sort()
    parts, kinds, pos = [], [], 0
    for a, b, pth, body in spans:
        gap = text[pos:a].strip("\n")
        if gap.strip():
            g = minify_output(gap, None)
            parts.append(g[0] if g else " ".join(gap.split()))
            kinds.append(g[1].kind if g else "text")
        f = minify(body, pth)
        rel = os.path.relpath(pth, cwd)
        parts.append(f"▶ {rel} " + (f[0] if f else " ".join(body.split())))
        kinds.append(f[1].kind if f else "text")
        pos = b
    tail = text[pos:].strip("\n")
    if tail.strip():
        g = minify_output(tail, None)
        parts.append(g[0] if g else " ".join(tail.split()))
        kinds.append(g[1].kind if g else "text")
    return " ".join(parts), kinds


def output_window(text: str, start: int, hint: str | None, budget: float):
    """The most output lines from `start` (1-based) whose flow fits one page -> (flow, minifier, n lines), or None.
    Output is paged by output line, because some kinds' § tags are file line numbers (grep, git diff)."""
    from vbhook import render
    lines = text.split("\n")
    cap = int(render.page_chars(budget) * 0.95)
    lo, hi, best = 1, len(lines) - start + 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        res = minify_output("\n".join(lines[start - 1:start - 1 + mid]), hint)
        if res and len(res[0]) <= cap:
            best, lo = (res[0], res[1], mid), mid + 1
        else:
            hi = mid - 1
    return best
