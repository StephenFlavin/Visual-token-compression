#!/usr/bin/env python3
"""Check that paths named in the tracked markdown files exist.

Errors (exit code 1): relative markdown links and images, `[text](target)` and `src="..."` / `href="..."`,
whose target is not in the working tree. http(s) / mailto links and anchor-only links are skipped; an anchor after
a path is dropped before the check.

Warnings (exit code stays 0): inline-code spans that begin with a repository path (`docs/`, `experiments/`,
`results/`, `scripts/`, `tests/`, `encoder_probe/`, `learned_glyphs/`, `vtc/`) that does not exist. Some of these
name git-ignored outputs or examples, so they are listed for a human to read. Patterns with `*`, `<...>` or
`{a,b}` are checked up to their fixed prefix directory.

    python3 scripts/check_links.py                    # all tracked .md files
    python3 scripts/check_links.py --exclude README.md AGENTS.md
    python3 scripts/check_links.py --no-warnings docs/README.md
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("docs/", "experiments/", "results/", "scripts/", "tests/", "encoder_probe/", "learned_glyphs/", "vtc/")
LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
ATTR = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']")
CODE = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^\s*(```|~~~)")
PATH_HEAD = re.compile(r"[A-Za-z0-9_./\-]+")


def tracked_markdown() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\n") if f and (ROOT / f).is_file()]


def link_targets(text: str):
    """(line number, target) of every markdown link, image and src/href attribute outside code fences."""
    fenced = False
    for n, line in enumerate(text.split("\n"), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        bare = CODE.sub("", line)                       # links inside inline code are examples
        for m in list(LINK.finditer(bare)) + list(ATTR.finditer(bare)):
            yield n, m.group(1)


def code_paths(text: str):
    """(line number, path) of inline-code spans that start with a repository path."""
    fenced = False
    for n, line in enumerate(text.split("\n"), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        for m in CODE.finditer(line):
            span = m.group(1).strip()
            if not span.startswith(PREFIXES):
                continue
            head = PATH_HEAD.match(span).group(0)
            rest = span[len(head):]
            if rest[:1] in ("*", "<", "{", "[", "$"):   # a pattern: keep its fixed directory only
                head = head.rsplit("/", 1)[0] if "/" in head else head
            head = head.split("::")[0].rstrip(".,:;")
            yield n, head


def is_external(target: str) -> bool:
    return bool(re.match(r"[a-zA-Z][a-zA-Z0-9+.\-]*:", target)) or target.startswith(("#", "//"))


def check(files: list[str], warnings: bool = True):
    broken, warned = [], []
    for f in files:
        text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
        base = (ROOT / f).parent
        for n, target in link_targets(text):
            if is_external(target):
                continue
            path = target.split("#", 1)[0].split("?", 1)[0]
            if not path:
                continue
            dest = (ROOT / path.lstrip("/")) if path.startswith("/") else (base / path)
            if not dest.exists():
                broken.append((f, n, target))
        if warnings:
            for n, path in code_paths(text):                # repo-root paths; package docs also name their own
                if not (ROOT / path).exists() and not (base / path).exists():
                    warned.append((f, n, path))
    return broken, warned


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="markdown files to check (default: every tracked .md)")
    ap.add_argument("--exclude", nargs="*", default=[], help="files to leave out")
    ap.add_argument("--no-warnings", action="store_true", help="skip the inline-code path check")
    a = ap.parse_args(argv)
    files = [f for f in (a.files or tracked_markdown()) if f not in set(a.exclude)]
    broken, warned = check(files, warnings=not a.no_warnings)
    for f, n, p in warned:
        print(f"warning: {f}:{n}: inline path does not exist: {p}")
    for f, n, t in broken:
        print(f"BROKEN: {f}:{n}: {t}")
    print(f"{len(files)} files, {len(broken)} broken links, {len(warned)} inline-path warnings")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
