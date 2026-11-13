"""Repo atlas: a whole repository as page images, for one up-front read (docs/claude/REPO_ATLAS_PLAN.md).

Every tracked source file is minified with the coding hooks' minifiers (experiments/readhook/vbhook: only
whitespace is removed, indentation that carries meaning is kept as ``»``), joined into one flow with a header per
file, and rendered as pages at the hooks' density (88.2 px² per character). Next to the pages it writes an index
(which pages hold which file) that costs a few hundred text tokens.

    python3 experiments/atlas/repo_atlas.py stats <repo dir> [<repo dir> ...]         # sizes only, no rendering
    python3 experiments/atlas/repo_atlas.py build <repo dir> --out <dir>               # pages + index.json + index.txt

Not one image: a model reads pages of at most 1932 px, and a request above ~175k image tokens silently loses its
first pages (docs/GOTCHAS.md), so ``fits`` in the stats means the atlas is at most FIT_TOKENS image tokens.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "readhook")]
from vbhook import render  # noqa: E402
from vbhook.minify import minify  # noqa: E402

FIT_TOKENS = 150_000        # under the ~175k image tokens that arrived whole, with room for the task and tools
MAX_FILE_BYTES = 200_000    # larger tracked files are data or generated code
SKIP_DIRS = {"node_modules", "vendor", "vendored", "third_party", "third-party", "dist", "build", ".git", "target",
             "__pycache__", ".github", "fixtures", "testdata", "snapshots"}
TEST_DIRS = {"tests", "test", "testing", "benches", "benchmarks", "examples", "docs", "doc"}
SKIP_NAMES = re.compile(r"(\.lock$|-lock\.json$|\.min\.(js|css)$|\.map$|\.svg$|\.ipynb$|\.csv$|\.tsv$|\.jsonl$|"
                        r"^LICENSE|^COPYING|^CHANGELOG|^CHANGES|^HISTORY|^AUTHORS|\.po$|\.pot$|\.snap$)", re.I)
TEST_NAMES = re.compile(r"(^test_.*\.py$|_test\.(py|go|rs)$|Tests?\.java$|\.test\.[jt]sx?$|\.spec\.[jt]sx?$|^conftest\.py$)")
HEADER = "▶{path}◀ "


def tracked(repo: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, check=True).stdout
    return sorted(x for x in out.splitlines() if x)


def wanted(path: str, include_tests: bool) -> bool:
    parts = path.split("/")
    if any(p in SKIP_DIRS for p in parts[:-1]) or SKIP_NAMES.search(parts[-1]):
        return False
    if not include_tests and (any(p in TEST_DIRS for p in parts[:-1]) or TEST_NAMES.search(parts[-1])):
        return False
    return True


def thin_tags(flow: str, every: int) -> str:
    """Keep the line number on every ``every``-th tagged line; the others keep a bare ``§`` as the statement
    separator (a line number per line costs ~10% of a code flow)."""
    if every <= 1:
        return flow
    return re.sub(r"§(\d+) ", lambda m: m.group(0) if int(m.group(1)) % every == 1 else "§ ", flow)


def collect(repo: Path, include_tests: bool = False, tag_every: int = 10) -> tuple[list[dict], list[str]]:
    """-> ([{path, chars, flow, text_tokens_est, kind}], [skipped paths that a minifier does not cover])."""
    files, uncovered = [], []
    for path in tracked(repo):
        f = repo / path
        if not wanted(path, include_tests) or not f.is_file() or f.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not text.strip():
            continue
        got = minify(text, path)
        if got is None:
            uncovered.append(path)
            continue
        flow, m = got
        files.append({"path": path, "chars": len(text), "flow": thin_tags(flow, tag_every), "kind": m.kind,
                      "text_tokens_est": len(text) / m.cpt})
    return files, uncovered


def atlas_flow(files: list[dict]) -> str:
    return " ".join(HEADER.format(path=f["path"]) + f["flow"] for f in files)


def stats(repo: Path, budget: float, include_tests: bool, tag_every: int) -> dict:
    files, uncovered = collect(repo, include_tests, tag_every)
    flow = atlas_flow(files)
    tok = render.page_tokens(flow, budget) if flow else 0
    text_tok = sum(f["text_tokens_est"] for f in files)
    return {"repo": repo.resolve().name, "files": len(files), "uncovered": len(uncovered), "source_chars": sum(f["chars"] for f in files),
            "flow_chars": len(flow), "image_tokens": tok, "pages": -(-len(flow) // render.page_chars(budget)) if flow else 0,
            "text_tokens_est": round(text_tok), "fewer_tokens": round(text_tok / tok, 2) if tok else None,
            "fits": 0 < tok <= FIT_TOKENS}


def build(repo: Path, out: Path, budget: float, include_tests: bool, tag_every: int) -> dict:
    files, uncovered = collect(repo, include_tests, tag_every)
    flow = atlas_flow(files)
    out.mkdir(parents=True, exist_ok=True)
    pages, index, n = [], [], 0
    for chunk in render.chunk(flow, render.page_chars(budget)):
        paths = re.findall(r"▶([^◀]+)◀", chunk)
        for pg in render.render(chunk, budget):          # a chunk that does not fit one page comes back as several
            n += 1
            pg.image.save(out / f"page_{n:03d}.png")
            pages.append({"page": n, "tokens": pg.tokens, "size": list(pg.image.size), "starts": paths})
    last = None
    for p in pages:                                       # file -> pages: from the page it starts on to the next start
        for path in p["starts"]:
            if last:
                index[-1]["pages"][1] = p["page"]
            index.append({"path": path, "pages": [p["page"], p["page"]]})
            last = path
    if index:
        index[-1]["pages"][1] = pages[-1]["page"]
    meta = {"repo": repo.resolve().name, "budget": budget, "files": len(files), "uncovered": uncovered,
            "image_tokens": sum(p["tokens"] for p in pages), "pages": pages, "index": index,
            "text_tokens_est": round(sum(f["text_tokens_est"] for f in files))}
    (out / "index.json").write_text(json.dumps(meta, indent=1))
    (out / "index.txt").write_text("\n".join(
        f"{e['path']}: page {e['pages'][0]}" + (f"-{e['pages'][1]}" if e["pages"][1] != e["pages"][0] else "")
        for e in index) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stats", "build"):
        p = sub.add_parser(name)
        p.add_argument("repos", nargs="+", type=Path)
        p.add_argument("--budget", type=float, default=88.2, help="px² per character (the coding hooks' density)")
        p.add_argument("--include-tests", action="store_true", help="also tests, examples and docs")
        p.add_argument("--tag-every", type=int, default=10, help="line number on every n-th line (1 = every line)")
    sub.choices["build"].add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "stats":
        print("| repo | files | source chars | pages | image tokens | text tokens (est.) | fewer tokens | fits one request |")
        print("|---|---|---|---|---|---|---|---|")
        for r in a.repos:
            s = stats(r, a.budget, a.include_tests, a.tag_every)
            print(f"| {s['repo']} | {s['files']} | {s['source_chars']:,} | {s['pages']} | {s['image_tokens']:,} | "
                  f"{s['text_tokens_est']:,} | {s['fewer_tokens']}x | {'yes' if s['fits'] else 'no'} |")
        return 0
    meta = build(a.repos[0], a.out, a.budget, a.include_tests, a.tag_every)
    print(f"[atlas] {meta['repo']}: {meta['files']} files -> {len(meta['pages'])} pages, {meta['image_tokens']:,} image "
          f"tokens (text ~{meta['text_tokens_est']:,}); {len(meta['uncovered'])} files without a minifier; {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
