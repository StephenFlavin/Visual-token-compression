#!/usr/bin/env python3
"""SessionStart hook (matcher "compact"): after a compaction, tell the model where the PreCompact hook archived the
pre-compaction transcript as page images (precompact_archive.py). Prints nothing if there is no archive.

settings.json:
  {"hooks": {"SessionStart": [{"matcher": "compact", "hooks": [{"type": "command",
     "command": "python3 /path/to/sessionstart_archive.py"}]}]}}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    try:
        inp = json.load(sys.stdin)
    except Exception:                                                # noqa: BLE001
        inp = {}
    if inp.get("source") not in (None, "compact"):
        return 0
    root = Path(inp.get("cwd") or os.getcwd()) / ".claude" / "compact_archive"
    try:
        d = Path((root / "LATEST").read_text().strip())
        n = json.loads((d / "meta.json").read_text())["pages"]
    except Exception:                                                # noqa: BLE001
        return 0
    note = (f"This conversation was compacted. The complete transcript from before the compaction (every message, "
            f"every file read and every command output, word for word) is archived as page images: {d}/page_01.png "
            f"... page_{n:02d}.png, in order. {d}/INDEX.txt lists what each page holds. The summary above leaves out "
            f"most details: when you need an exact detail from before the compaction (a value, a line, a name, a "
            f"number), read INDEX.txt and open the relevant pages with the Read tool instead of guessing.")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": note}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
