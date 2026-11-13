#!/usr/bin/env python3
"""PreCompact hook: archive the whole pre-compaction transcript as page images before Claude Code compacts it.

Reads the hook input ({"transcript_path", "session_id", "cwd", "trigger"}) from stdin, turns every message of the
transcript into one flow in order (the user's prompts, the agent's text, each tool call with its result), renders it
at 88.2 px²/char (vtchook) and writes

    <cwd>/.claude/compact_archive/<session>-<n>/page_NN.png, INDEX.txt
    <cwd>/.claude/compact_archive/LATEST          (the directory just written; read by sessionstart_archive.py)

Files read and command output keep their line structure through vtchook's minifiers (`§n` = original line n, `»` =
one indentation level), so code, configs and logs stay readable and line numbers stay quotable. Never blocks
compaction: any error is logged to <cwd>/.claude/compact_archive/errors.log and the hook exits 0.

settings.json:
  {"hooks": {"PreCompact": [{"hooks": [{"type": "command", "timeout": 300,
     "command": "python3 /path/to/precompact_archive.py"}]}]}}
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[3]), str(HERE.parents[2] / "readhook")]
BUDGET = 88.2
LINE_NO = re.compile(r"(?m)^\s*\d+[\t→]")
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(x.get("text", "") for x in content or [] if isinstance(x, dict) and x.get("type") == "text")


def _one_line(s: str) -> str:
    return " ".join(s.split())


def entry_flows(transcript: list[dict], cwd: str) -> list[tuple[str, str]]:
    """-> [(index label, flow)] in transcript order."""
    from vtchook.common import minify_output, shift_tags
    from vtchook.minify import minify
    calls, out, done, seen = {}, [], set(), set()              # forks copy history: dedupe results and texts
    for e in transcript:
        if e.get("isCompactSummary") or e.get("isMeta"):
            continue
        m = e.get("message") or {}
        key = (m.get("id") or e.get("uuid"), json.dumps(m.get("content"), sort_keys=True)[:2000])
        if m.get("role") != "user" or isinstance(m.get("content"), str):
            if key in seen:
                continue
            seen.add(key)
        role, content = m.get("role"), m.get("content")
        if role == "user" and isinstance(content, str):
            t = _one_line(REMINDER.sub("", content))
            if t:
                out.append((f"user: {t[:60]}", f"▶ user: {t}"))
            continue
        if not isinstance(content, list):
            continue
        for c in content:
            ty = c.get("type")
            if ty == "text" and role == "assistant" and c.get("text", "").strip():
                t = _one_line(c["text"])
                out.append((f"assistant: {t[:60]}", f"▶ assistant: {t}"))
            elif ty == "text" and role == "user":
                t = _one_line(REMINDER.sub("", c.get("text", "")))
                if t:
                    out.append((f"user: {t[:60]}", f"▶ user: {t}"))
            elif ty == "tool_use":
                calls[c["id"]] = (c.get("name", "tool"), c.get("input") or {})
            elif ty == "tool_result" and c.get("tool_use_id") in calls and c["tool_use_id"] not in done:
                done.add(c["tool_use_id"])
                name, inp = calls[c["tool_use_id"]]
                body = REMINDER.sub("", _text(c.get("content")))
                if name == "Read":
                    path = inp.get("file_path", "")
                    rel = os.path.relpath(path, cwd) if path.startswith("/") else path
                    raw = LINE_NO.sub("", body)
                    res = minify(raw.rstrip("\n"), path) if raw.strip() else None
                    if res:
                        flow = res[0]
                        if inp.get("offset"):
                            flow = shift_tags(flow, int(inp["offset"]))
                    else:                                    # no minifier (or it refused the file): keep a line tag on
                        first = int(inp.get("offset") or 1)  # every line so line numbers stay quotable
                        flow = " ".join(f"§{first + i} {' '.join(x.split())}"
                                        for i, x in enumerate(raw.rstrip(chr(10)).split(chr(10))) if x.strip())
                    out.append((f"Read {rel}", f"▶ Read {rel} {flow}"))
                elif name == "Bash":
                    cmd = _one_line(inp.get("command", ""))
                    res = minify_output(body, None) if body.strip() else None
                    out.append((f"Bash: {cmd[:90]}", f"▶ Bash: {cmd} ▶ output: " + (res[0] if res else _one_line(body))))
                else:
                    arg = _one_line(json.dumps(inp))[:300]
                    out.append((f"{name} {arg[:80]}", f"▶ {name} {arg} ▶ result: {_one_line(body)}"))
    uniq, texts = [], set()
    for label, flow in out:                                      # identical messages copied under new ids
        if label.startswith(("user:", "assistant:")):
            if flow in texts:
                continue
            texts.add(flow)
        uniq.append((label, flow))
    return uniq


def build(transcript_path: str, cwd: str, session_id: str) -> Path:
    from vtchook import render
    tr = [json.loads(l) for l in open(transcript_path) if l.strip()]
    flows = entry_flows(tr, cwd)
    root = Path(cwd) / ".claude" / "compact_archive"
    n = 1 + sum(1 for p in root.glob(f"{session_id}-*") if p.is_dir()) if root.exists() else 1
    d = root / f"{session_id}-{n}"
    d.mkdir(parents=True, exist_ok=True)
    cap = render.page_chars(BUDGET)
    pages, index, cur, cur_labels, carry = [], [], "", [], None
    for label, flow in flows:                                   # pack entries into pages; long entries split
        rest, first = flow, True
        while rest:
            room = cap - len(cur) - (1 if cur else 0)
            if room < 200 and cur:
                pages.append(cur)
                index.append(([f"(continues {carry})"] if carry and not cur.startswith("▶") else []) + cur_labels)
                cur, cur_labels = "", []
                continue
            take = render.chunk(rest, room)[0] if len(rest) > room else rest
            cur = (cur + " " + take) if cur else take
            if first:
                cur_labels.append(label)
            carry, rest, first = label, rest[len(take):].lstrip(), False
            if rest:
                pages.append(cur)
                index.append(cur_labels)
                cur, cur_labels = "", []
    if cur:
        pages.append(cur)
        index.append(cur_labels)
    tokens = 0
    for k, text in enumerate(pages, 1):
        pg = render.render_one(text, BUDGET)
        pg.image.save(d / f"page_{k:02d}.png")
        tokens += pg.tokens
    lines = [f"page_{k:02d}.png: " + ("; ".join(labs) if labs else "(continues the previous page)")
             for k, labs in enumerate(index, 1)]
    (d / "INDEX.txt").write_text(
        "Archive of this session's transcript before compaction, in order. Each item starts with ▶: user and "
        "assistant messages, and every tool call with its full result (files read and command output keep their "
        "line structure: §n marks original line n, » one indentation level). Pages list the items that start on "
        "them; an item that does not fit continues on the next page.\n\n" + "\n".join(lines) + "\n")
    (d / "meta.json").write_text(json.dumps({"pages": len(pages), "image_tokens": tokens, "items": len(flows),
                                             "chars": sum(map(len, pages)), "transcript": transcript_path}))
    (root / "LATEST").write_text(str(d))
    return d


def main() -> int:
    t0 = time.time()
    try:
        inp = json.load(sys.stdin)
    except Exception:                                                # noqa: BLE001
        return 0
    cwd = inp.get("cwd") or os.getcwd()
    try:
        d = build(inp["transcript_path"], cwd, inp.get("session_id", "session"))
        meta = json.loads((d / "meta.json").read_text())
        meta["seconds"] = round(time.time() - t0, 1)
        (d / "meta.json").write_text(json.dumps(meta))
    except Exception:                                                # noqa: BLE001
        root = Path(cwd) / ".claude" / "compact_archive"
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "errors.log", "a") as f:
            f.write(traceback.format_exc() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
