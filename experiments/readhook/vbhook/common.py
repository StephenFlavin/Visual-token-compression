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
