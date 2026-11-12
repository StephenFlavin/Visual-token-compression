"""Repo atlas, stage 1: where does the change go? (docs/claude/REPO_ATLAS_PLAN.md, results/repo_atlas/README.md)

One question per SWE-bench task: the issue, then "list the files that must change, most likely first". Arms:

  tree   the paths of the repository's source files (the atlas's file set)
  map    per file, its class / function / method signature lines (``signatures``)
  atlas  the source files as page images (repo_atlas.build) plus index.txt as text
  text   the same files as plain text with a path header per file
  agent  Claude Code headless in an export of the checkout, read-only tools (Read, Grep, Glob), turn limit

    python3 experiments/atlas/repo_atlas_eval.py select   --work DIR            # pick the tasks (no model calls)
    python3 experiments/atlas/repo_atlas_eval.py prepare  --work DIR            # checkouts, atlases, contexts, sizes
    python3 experiments/atlas/repo_atlas_eval.py estimate --work DIR            # list-price estimate per arm
    python3 experiments/atlas/repo_atlas_eval.py run      --work DIR --arms tree map atlas text [--cap 12]
    python3 experiments/atlas/repo_atlas_eval.py run      --work DIR --arms agent --cap 12
    python3 experiments/atlas/repo_atlas_eval.py analyse

Clones, checkouts, page images and agent transcripts live in --work (outside the repo); rows, the selection, the
context sizes and the atlas indexes go to results/repo_atlas/. Rows are appended per (task, arm); finished pairs
are skipped, so ``run`` resumes. One-shot calls go through vbpe.claude_eval (cache, clean env).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "atlas")]
import repo_atlas as RA  # noqa: E402

OUT = ROOT / "results" / "repo_atlas"
REPOS = {"psf/requests": "princeton-nlp/SWE-bench_Verified", "gin-gonic/gin": "SWE-bench/SWE-bench_Multilingual",
         "sharkdp/bat": "SWE-bench/SWE-bench_Multilingual", "preactjs/preact": "SWE-bench/SWE-bench_Multilingual"}
PER_REPO = 2
TEXT_REPOS = {"psf/requests", "gin-gonic/gin"}
ARMS = ("tree", "map", "atlas", "text", "agent")
MODEL = "claude-opus-5-5"
EFFORT = "high"
AGENT_TURNS = 40
PAGE_ARRIVAL = 0.97
DOC_EXT = re.compile(r"\.(md|rst|txt|adoc|markdown)$", re.I)

SYSTEM = ("You are an expert software engineer. You decide where in a repository a change has to be made. "
          "Answer only in the format asked for.")
INSTRUCTION = ("List the files that must change to fix this issue, most likely first: at most 5 paths, exactly as they "
               "appear in the repository, one per line as `FILE: <path>`. Then one line `PLAN: <one sentence>`.")
AGENT_INTRO = ("The current directory is a checkout of the {repo} repository. Use your tools to look at the code as "
               "much as you need; do not change anything.")


# --------------------------------------------------------------------------- pure functions (tested)
def patch_files(patch: str) -> list[str]:
    """Paths a unified diff touches, in order (the b/ side; the a/ side for deletions)."""
    out = []
    for m in re.finditer(r"^diff --git a/(\S+) b/(\S+)$", patch, re.M):
        p = m.group(2) if m.group(2) != "/dev/null" else m.group(1)
        if p not in out:
            out.append(p)
    return out


def is_gold(path: str) -> bool:
    """A source file: not a test, doc, changelog, lock file or anything else repo_atlas leaves out."""
    if DOC_EXT.search(path):
        return False
    return RA.wanted(path, include_tests=False)


def gold_files(patch: str) -> list[str]:
    return [p for p in patch_files(patch) if is_gold(p)]


_DECL = [re.compile(p) for p in (
    r"\bdef\s+(\w+)", r"\bclass\s+([A-Za-z_]\w*)", r"\bfunc\s+(?:\([^)]*\)\s*)?(\w+)", r"\bfn\s+(\w+)",
    r"\bfunction\s*\*?\s*(\w+)", r"\b(?:struct|enum|trait|interface|union)\s+([A-Za-z_]\w*)",
    r"\btype\s+([A-Za-z_]\w*)\s+(?:struct|interface)", r"([A-Za-z_$][\w$]*)\s*[:=]\s*(?:async\s+)?function\b",
    r"([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>")]


def decl_names(line: str) -> list[str]:
    names = []
    for rx in _DECL:
        for m in rx.finditer(line):
            if m.group(1) not in names and m.group(1) not in ("function", "if", "for", "while", "return"):
                names.append(m.group(1))
    return names


def gold_functions(patch: str) -> list[str]:
    """Functions/classes the gold patch touches in its gold files: names in the hunk headers' context (the enclosing
    definition git found above the hunk) and declarations on changed (+/-) lines."""
    names, cur = [], None
    for line in patch.splitlines():
        m = re.match(r"^diff --git a/\S+ b/(\S+)$", line)
        if m:
            cur = m.group(1) if is_gold(m.group(1)) else None
            continue
        if cur is None or line.startswith(("+++", "---")):
            continue
        if line.startswith("@@"):
            ctx = line.split("@@", 2)[2] if line.count("@@") >= 2 else ""
            found = decl_names(ctx)
        elif line[:1] in "+-":
            found = decl_names(line[1:])
        else:
            continue
        names += [n for n in found if n not in names]
    return names


def _clip(s: str, n: int = 200) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s if len(s) <= n else s[: n - 1] + "…"


def py_signatures(text: str) -> list[str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except SyntaxError:
        return [_clip(m.group(0)) for m in re.finditer(r"^[ \t]*(?:async\s+)?(?:def|class)\s+\w+.*$", text, re.M)]
    lines = text.splitlines()
    out = []

    def walk(node, depth):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                b = ch.body[0]                            # the header runs up to the first statement
                seg = lines[ch.lineno - 1: b.lineno]
                seg[-1] = seg[-1][: b.col_offset] if b.lineno > ch.lineno or len(seg) == 1 else seg[-1]
                if b.lineno > ch.lineno:
                    seg = seg[:-1] if not seg[-1].strip() else seg
                head = re.sub(r"\s*:\s*(#[^\n]*)?$", "", " ".join(l.strip() for l in seg))
                out.append("  " * depth + _clip(head))
                walk(ch, depth + 1)
            elif not isinstance(ch, (ast.expr, ast.stmt)) or isinstance(ch, (ast.If, ast.Try, ast.With)):
                walk(ch, depth)
    walk(tree, 0)
    return out


_SIG = {
    "go": [r"^func\s.*", r"^type\s+\w+.*"],
    "rust": [r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+\w+.*",
             r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|type|mod|union)\s+\w+.*", r"^\s*(?:unsafe\s+)?impl\b.*",
             r"^\s*macro_rules!\s*\w+.*"],
    "js": [r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b.*",
           r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\b.*",
           r"^\s*(?:export\s+)?(?:const|let|var)\s+[\w$]+\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[\w$]+\s*=>).*",
           r"^\s*[\w$.]+\.prototype\.[\w$]+\s*=.*",
           r"^\s*(?:export\s+)?(?:declare\s+)?(?:interface|type|enum|namespace)\s+[\w$]+.*",
           r"^\s+(?:static\s+|async\s+|get\s+|set\s+)*(?!(?:if|for|while|switch|catch|function|return|else)\b)[\w$]+\s*\([^)]*\)\s*\{.*"],
}
_LANG = {".go": "go", ".rs": "rust", ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js", ".ts": "js", ".tsx": "js"}


def signatures(path: str, text: str) -> list[str]:
    """The map's lines for one file. Python: every class/def header (all nesting levels, decorators and docstrings
    left out) from ``ast``. Go: lines starting ``func`` or ``type``. Rust: fn / struct / enum / trait / type / mod /
    union / impl / macro_rules lines. JS/TS: function, class, arrow/function assignments, prototype assignments,
    interface / type / enum / namespace, and indented method heads. Other files: none (path only). First line of
    each declaration only, opening brace removed, at most 200 characters."""
    if path.endswith((".py", ".pyi")):
        return py_signatures(text)
    lang = _LANG.get(Path(path).suffix)
    if not lang:
        return []
    rx = re.compile("|".join(f"(?:{p})" for p in _SIG[lang]))
    out = []
    for line in text.splitlines():
        if rx.fullmatch(line):
            ind = len(line) - len(line.lstrip())
            body = re.sub(r"\s*\{\s*(\}\s*)?$", "", line.strip())
            out.append(" " * min(ind, 8) + _clip(body))
    return out


def build_map(files: list[tuple[str, str]]) -> str:
    """files = [(path, text)] -> the map: the path, then its signature lines indented by two spaces."""
    parts = []
    for path, text in files:
        parts.append(path + "\n" + "".join("  " + s + "\n" for s in signatures(path, text)))
    return "".join(parts)


def norm_path(p: str, cwd: str | None = None) -> str:
    p = p.strip().strip("`'\"*").strip()
    if cwd and p.startswith(cwd.rstrip("/") + "/"):
        p = p[len(cwd.rstrip("/")) + 1:]
    p = re.sub(r"(:\d+)+(-\d+)?$", "", p)               # path:line
    p = re.sub(r"\s+\(.*\)$", "", p)                    # path (note)
    p = re.sub(r"^(\./|[ab]/(?=\S))", "", p)
    return p.lstrip("/")


def parse_answer(text: str, cwd: str | None = None) -> tuple[list[str], str]:
    """-> (up to 5 distinct paths from ``FILE:`` lines, in order; the PLAN sentence or '')."""
    files, plan = [], ""
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(?:[-*]\s*|\d+[.)]\s*)?\**FILE\**\s*:\s*(.+?)\s*$", line, re.I)
        if m:
            p = norm_path(m.group(1), cwd)
            if p and p not in files and len(files) < 5:
                files.append(p)
            continue
        m = re.match(r"^\s*\**PLAN\**\s*:\s*(.+?)\s*$", line, re.I)
        if m and not plan:
            plan = m.group(1)
    return files, plan


def score(files: list[str], plan: str, gold: list[str], funcs: list[str], repo_paths: set[str] | None = None) -> dict:
    top = files[:5]
    hit = [g for g in gold if g in top]
    text = plan + "\n" + "\n".join(top)
    named = [f for f in funcs if re.search(rf"(?<![\w$]){re.escape(f)}(?![\w$])", text)]
    return {"all_gold_top5": bool(gold) and len(hit) == len(gold), "first_right": bool(top) and top[0] in gold,
            "gold_share": len(hit) / len(gold) if gold else 0.0, "n_named": len(top),
            "funcs_named": named, "funcs_share": len(named) / len(funcs) if funcs else None,
            "not_in_repo": [f for f in top if repo_paths is not None and f not in repo_paths]}


def image_tokens(sizes: list[tuple[int, int]]) -> int:
    return sum(-(-w // 28) * -(-h // 28) for w, h in sizes)


# --------------------------------------------------------------------------- tasks and checkouts
def sh(cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)


def load_tasks(work: Path) -> list[dict]:
    cache = work / "tasks_all.json"
    if not cache.exists():
        from datasets import load_dataset
        rows = []
        for ds in sorted(set(REPOS.values())):
            rows += [dict(r) for r in load_dataset(ds, split="test") if r["repo"] in REPOS and REPOS[r["repo"]] == ds]
        cache.write_text(json.dumps(rows))
    return json.loads(cache.read_text())


def clone(work: Path, repo: str) -> Path:
    d = work / "clones" / repo.split("/")[1]
    if not d.exists():
        sh(["git", "clone", "-q", f"https://github.com/{repo}.git", str(d)])
    return d


def checkout(work: Path, t: dict) -> Path:
    """A worktree at the base commit (for the file set, the atlas and the contexts)."""
    d = work / "co" / t.get("instance_id", t.get("id"))
    if not d.exists():
        c = clone(work, t["repo"])
        sh(["git", "-C", str(c), "worktree", "add", "-q", "--detach", str(d), t["base_commit"]])
    return d


def export(work: Path, t: dict) -> Path:
    """A plain copy of the tree at the base commit without .git (the agent's cwd: no history to read)."""
    d = work / "export" / t.get("instance_id", t.get("id")) / t["repo"].split("/")[1]
    if not d.exists():
        d.mkdir(parents=True)
        c = clone(work, t["repo"])
        arch = subprocess.run(["git", "-C", str(c), "archive", t["base_commit"]], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=arch, check=True)
    return d


def select(work: Path) -> dict:
    """Per repository: tasks ordered by the number of gold files (source files in the patch, new files included;
    most first; ties: lowest instance id); keep the first PER_REPO whose atlas at the base commit is at most
    RA.FIT_TOKENS image tokens. Scored gold = the gold files that exist at the base commit (``gold``); files the fix
    creates (``gold_new``) cannot be named as they appear in the repository and are reported, not scored."""
    tasks = load_tasks(work)
    chosen, skipped = [], []
    for repo in REPOS:
        ts = sorted((t for t in tasks if t["repo"] == repo), key=lambda t: (-len(gold_files(t["patch"])), t["instance_id"]))
        n = 0
        for t in ts:
            if n == PER_REPO:
                break
            co = checkout(work, t)
            st = RA.stats(co, 88.2, False, 10)
            at_base = set(sh(["git", "-C", str(co), "ls-files"]).stdout.split())
            g = gold_files(t["patch"])
            entry = {"id": t["instance_id"], "repo": repo, "base_commit": t["base_commit"],
                     "gold": [x for x in g if x in at_base], "gold_new": [x for x in g if x not in at_base],
                     "patch_files": patch_files(t["patch"]),
                     "funcs": gold_functions(t["patch"]), "atlas_image_tokens": st["image_tokens"]}
            if st["image_tokens"] > RA.FIT_TOKENS:
                skipped.append({**entry, "why": "atlas too large"})
                continue
            chosen.append(entry)
            n += 1
    sel = {"rule": select.__doc__.strip(), "tasks": chosen, "skipped": skipped,
           "pool": {r: len([t for t in tasks if t["repo"] == r]) for r in REPOS}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel


def selection() -> list[dict]:
    return json.loads((OUT / "selection.json").read_text())["tasks"]


def task_rows(work: Path) -> dict[str, dict]:
    return {t["instance_id"]: t for t in load_tasks(work)}


# --------------------------------------------------------------------------- contexts
def contexts(work: Path, s: dict) -> dict:
    """Build (once) every arm's context for one selected task; returns sizes and paths."""
    t_dir = work / "ctx" / s["id"]
    meta_p = t_dir / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())
    co = checkout(work, s)
    files, uncovered = RA.collect(co, False, 10)
    paths = [f["path"] for f in files]
    texts = [(p, (co / p).read_text(encoding="utf-8")) for p in paths]
    t_dir.mkdir(parents=True, exist_ok=True)
    name = s["repo"]
    tree = f"Repository {name}: its {len(paths)} source files (tests, docs and data files left out).\n\n" + "\n".join(paths) + "\n"
    rmap = (f"Repository {name}: a repo map of its {len(paths)} source files (tests, docs and data files left out). Per "
            "file: its path, then the signature line of every class, function and method in it, indented.\n\n"
            + build_map(texts))
    text = (f"Repository {name}: the full text of its {len(paths)} source files (tests, docs and data files left out), "
            "each after a line `=== <path> ===`.\n\n" + "".join(f"=== {p} ===\n{x}\n" for p, x in texts))
    atlas_dir = t_dir / "atlas"
    am = RA.build(co, atlas_dir, 88.2, False, 10)
    index = (atlas_dir / "index.txt").read_text()
    atlas = (f"The images above are the {len(paths)} source files of the repository {name} (tests, docs and data files "
             "left out), in this order, each starting with a header ▶path◀. The code is minified: §<n> marks the "
             "start of original line n (shown on every 10th line, a bare § starts the others); each » before a line "
             "is one level of indentation where indentation matters; blank lines and other indentation are not "
             "shown. Index (file: page):\n\n" + index)
    for k, v in (("tree", tree), ("map", rmap), ("text", text), ("atlas", atlas)):
        (t_dir / f"{k}.txt").write_text(v)
    all_paths = sh(["git", "-C", str(co), "ls-files"]).stdout.split()
    meta = {"id": s["id"], "files": len(paths), "uncovered": uncovered, "chars": {k: len(v) for k, v in
            (("tree", tree), ("map", rmap), ("text", text), ("atlas_index", atlas))},
            "pages": len(am["pages"]), "image_tokens": am["image_tokens"],
            "page_sizes": [p["size"] for p in am["pages"]], "atlas_dir": str(atlas_dir),
            "gold_outside_fileset": [g for g in s["gold"] if g not in paths],
            "gold_missing_at_base": [g for g in s["gold"] if g not in all_paths]}
    (t_dir / "repo_paths.json").write_text(json.dumps(all_paths))
    meta_p.write_text(json.dumps(meta, indent=1))
    idx = OUT / "index"
    idx.mkdir(parents=True, exist_ok=True)
    shutil.copy(atlas_dir / "index.txt", idx / f"{s['id']}.txt")
    return meta


def prompt_for(arm: str, ctx_text: str, issue: str, repo: str) -> str:
    if arm == "agent":
        return AGENT_INTRO.format(repo=repo) + f"\n\n<issue>\n{issue}\n</issue>\n\n" + INSTRUCTION
    return ctx_text + f"\n<issue>\n{issue}\n</issue>\n\n" + INSTRUCTION


# --------------------------------------------------------------------------- running
_lock = threading.Lock()
_alock = threading.Lock()
_reserved = [0.0]


def reserve(cost: float, cap: float) -> bool:
    """Book ``cost`` against the cap before a call; False (nothing booked) if spent + booked + cost > cap."""
    with _lock:
        if spent() + _reserved[0] + cost > cap:
            return False
        _reserved[0] += cost
        return True


def release(cost: float) -> None:
    with _lock:
        _reserved[0] -= cost


def job_bound(job: dict, meta: dict) -> float:
    """A generous upper bound on one one-shot call's list price: everything written to cache, 8k output tokens."""
    return (len(job["prompt"]) / 3.0 + meta["image_tokens"]) * PRICE_WRITE * 1.1 + 8000 * PRICE_OUT


def append(name: str, row: dict) -> None:
    with _alock:
        with open(OUT / name, "a") as f:
            f.write(json.dumps(row) + "\n")


def done_pairs() -> set:
    p = OUT / "runs.jsonl"
    return {(r["id"], r["arm"]) for r in map(json.loads, p.read_text().splitlines())} if p.exists() else set()


def spent() -> float:
    p = OUT / "spend.jsonl"
    return sum(r["cost_usd"] for r in map(json.loads, p.read_text().splitlines())) if p.exists() else 0.0


def finish_row(s: dict, arm: str, text: str, rec: dict, cwd: str | None = None) -> dict:
    files, plan = parse_answer(text, cwd)
    repo_paths = set(json.loads((Path(rec.pop("_ctx_dir")) / "repo_paths.json").read_text()))
    sc = score(files, plan, s["gold"], s["funcs"], repo_paths)
    return {"id": s["id"], "repo": s["repo"], "arm": arm, "gold": s["gold"], "funcs": s["funcs"], "files": files,
            "plan": plan, **sc, "refused_or_empty": not files, **rec, "answer": text}


def one_shot_jobs(work: Path, arms: list[str], todo: set) -> list[dict]:
    from PIL import Image
    tr = task_rows(work)
    jobs = []
    for s in selection():
        meta = contexts(work, s)
        for arm in arms:
            if arm == "agent" or (s["id"], arm) not in todo or (arm == "text" and s["repo"] not in TEXT_REPOS):
                continue
            ctx = (work / "ctx" / s["id"] / f"{arm}.txt").read_text()
            imgs = [Image.open(Path(meta["atlas_dir"]) / f"page_{i + 1:03d}.png") for i in range(meta["pages"])] \
                if arm == "atlas" else []
            jobs.append({"images": imgs, "system": SYSTEM, "prompt": prompt_for(arm, ctx, tr[s["id"]]["problem_statement"],
                         s["repo"]), "model": MODEL, "effort": EFFORT, "hires": True, "timeout": 900.0,
                         "meta": {"s": s, "arm": arm, "ctx_dir": str(work / "ctx" / s["id"]),
                                  "image_tokens": meta["image_tokens"] if arm == "atlas" else 0}})
    return jobs


def run_one_shot(work: Path, arms: list[str], cap: float, workers: int) -> None:
    from vbpe import claude_eval as CE
    todo = {(s["id"], a) for s in selection() for a in arms} - done_pairs()
    jobs = one_shot_jobs(work, arms, todo)
    print(f"[ra] {len(jobs)} one-shot calls to make; spent so far ${spent():.2f} (cap ${cap})", flush=True)
    if not jobs:
        return
    def run(job):
        meta = job.pop("meta")
        bound = job_bound(job, meta)
        if not reserve(bound, cap):
            print(f"[ra] SKIP {meta['s']['id']} {meta['arm']}: ${spent():.2f} spent + in flight + ${bound:.2f} > cap", flush=True)
            return
        try:
            rec = CE.transcribe(**job)
        finally:
            release(bound)
        if not rec.get("cached") and rec.get("cost_usd"):
            append("spend.jsonl", {"what": f"{meta['s']['id']} {meta['arm']}", "cost_usd": rec["cost_usd"],
                                   "input_tokens": rec.get("input_tokens"), "output_tokens": rec.get("output_tokens"),
                                   "error": rec.get("error")})
        if rec.get("error"):
            print(f"[ra] ERROR {meta['s']['id']} {meta['arm']}: {rec['error'][:300]}", flush=True)
            return                                          # not recorded: retried on the next start
        keep = {k: rec.get(k) for k in ("model", "input_tokens", "output_tokens", "cost_usd", "latency_s", "cached",
                                        "usage", "thinking_tokens", "stop_reason", "cache_key")}
        keep["seconds"] = rec.get("latency_s")
        keep["turns"] = 1
        keep["image_tokens_formula"] = meta["image_tokens"]
        keep["pages_arrived"] = (meta["arm"] != "atlas" or (rec.get("input_tokens") or 0) >= PAGE_ARRIVAL * meta["image_tokens"])
        keep["_ctx_dir"] = meta["ctx_dir"]
        row = finish_row(meta["s"], meta["arm"], rec.get("text", ""), keep)
        append("runs.jsonl", row)
        print(f"[ra] {row['id']:28} {row['arm']:6} all={row['all_gold_top5']!s:5} first={row['first_right']!s:5} "
              f"share={row['gold_share']:.2f} in={row['input_tokens']:,} ${row['cost_usd']:.3f} {row['seconds']:.0f}s "
              f"{'' if row['pages_arrived'] else 'PAGES DROPPED'} files={row['files']}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, jobs))


def agent_once(work: Path, s: dict, issue: str) -> dict:
    from vbpe import claude_eval as CE
    cwd = export(work, s)
    cmd = [CE.CLAUDE_BIN, "-p", prompt_for("agent", "", issue, s["repo"]), "--output-format", "stream-json", "--verbose",
           "--model", MODEL, "--effort", EFFORT, "--tools", "Read,Grep,Glob", "--setting-sources", "",
           "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence", "--max-turns", str(AGENT_TURNS)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=CE.build_env(), cwd=cwd, timeout=1800,
                           stdin=subprocess.DEVNULL)
        stdout, rc, err = p.stdout, p.returncode, p.stderr
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc, err = -9, "timeout"
    secs = time.time() - t0
    tdir = work / "agent_streams"
    tdir.mkdir(exist_ok=True)
    (tdir / f"{s['id']}.jsonl").write_text(stdout)
    ev = [json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]
    res = next((e for e in reversed(ev) if e.get("type") == "result"), None)
    calls = [c for e in ev if e.get("type") == "assistant" for c in e["message"]["content"] if c.get("type") == "tool_use"]
    denied = [str(d)[:200] for d in (res or {}).get("permission_denials") or []]
    outside = [c["input"] for c in calls if any(isinstance(v, str) and v.startswith("/") and not v.startswith(str(cwd))
                                                 for v in (c.get("input") or {}).values())]
    if res is None or (res.get("is_error") and res.get("subtype") not in ("error_max_turns",)):
        return {"error": f"rc={rc} {str((res or {}).get('result'))[:300]} {err[-300:]}",
                "cost_usd": float((res or {}).get("total_cost_usd") or 0.0)}
    u = res.get("usage") or {}
    text = res.get("result") or ""
    if not text:                                            # max turns: take the last assistant text, if any
        texts = [c["text"] for e in ev if e.get("type") == "assistant" for c in e["message"]["content"] if c.get("type") == "text"]
        text = texts[-1] if texts else ""
    names = [c["name"] for c in calls]
    mu = res.get("modelUsage") or {}
    return {"model": max(mu, key=lambda m: mu[m].get("outputTokens", 0)) if mu else MODEL,
            "input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0), "usage": {k: u.get(k, 0) for k in ("input_tokens",
            "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")},
            "cost_usd": float(res.get("total_cost_usd") or 0.0), "seconds": round(secs, 1), "turns": res.get("num_turns"),
            "subtype": res.get("subtype"), "stop_reason": res.get("stop_reason"), "cached": False,
            "tools": {k: names.count(k) for k in sorted(set(names))}, "tool_calls": len(calls),
            "outside_cwd": outside[:10], "permission_denials": denied[:10], "text": text, "cwd": str(cwd),
            "model_usage": {m: {"in": v.get("inputTokens", 0) + v.get("cacheCreationInputTokens", 0) + v.get(
                "cacheReadInputTokens", 0), "out": v.get("outputTokens", 0), "cost": v.get("costUSD", 0.0)}
                for m, v in (res.get("modelUsage") or {}).items()}}


def run_agent(work: Path, cap: float, workers: int) -> None:
    tr = task_rows(work)
    todo = [s for s in selection() if (s["id"], "agent") not in done_pairs()]
    print(f"[ra] {len(todo)} agent runs to make; spent so far ${spent():.2f} (cap ${cap})", flush=True)
    worst = [1.5]                                           # assumed until measured: an agent run's cost bound

    def run(s):
        bound = worst[0]
        if not reserve(bound, cap):
            print(f"[ra] SKIP agent {s['id']}: ${spent():.2f} spent + in flight + ${bound:.2f} > cap", flush=True)
            return
        try:
            contexts(work, s)
            rec = agent_once(work, s, tr[s["id"]]["problem_statement"])
        finally:
            release(bound)
        if rec.get("cost_usd"):
            append("spend.jsonl", {"what": f"{s['id']} agent", "cost_usd": rec["cost_usd"],
                                   "input_tokens": rec.get("input_tokens"), "output_tokens": rec.get("output_tokens"),
                                   "error": rec.get("error")})
            worst[0] = max(1.0, rec["cost_usd"] * 2)
        if rec.get("error"):
            print(f"[ra] ERROR agent {s['id']}: {rec['error'][:300]}", flush=True)
            return
        text = rec.pop("text")
        cwd = rec.pop("cwd")
        rec["_ctx_dir"] = str(work / "ctx" / s["id"])
        rec["image_tokens_formula"] = 0
        rec["pages_arrived"] = True
        row = finish_row(s, "agent", text, rec, cwd)
        append("runs.jsonl", row)
        print(f"[ra] {row['id']:28} agent  all={row['all_gold_top5']!s:5} first={row['first_right']!s:5} "
              f"share={row['gold_share']:.2f} turns={row['turns']} in={row['input_tokens']:,} ${row['cost_usd']:.3f} "
              f"{row['seconds']:.0f}s tools={row['tools']} files={row['files']}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))


# --------------------------------------------------------------------------- estimate and analysis
PRICE_IN, PRICE_WRITE, PRICE_OUT = 4e-6, 5e-6, 20e-6     # per token, list; fitted to results/claude_qa/repeat rows


def estimate(work: Path, agent_cost: float = 1.0, out_tokens: int = 4000) -> dict:
    est = {a: 0.0 for a in ARMS}
    for s in selection():
        meta = contexts(work, s)
        issue = len(task_rows(work)[s["id"]]["problem_statement"]) / 3.5
        for arm, chars in (("tree", meta["chars"]["tree"]), ("map", meta["chars"]["map"]), ("text", meta["chars"]["text"])):
            if arm == "text" and s["repo"] not in TEXT_REPOS:
                continue
            est[arm] += (chars / 3.2 + issue) * PRICE_WRITE + out_tokens * PRICE_OUT
        est["atlas"] += (meta["image_tokens"] + meta["chars"]["atlas_index"] / 3.2 + issue) * PRICE_WRITE + out_tokens * PRICE_OUT
        est["agent"] += agent_cost
    est["total"] = sum(est.values())
    return {k: round(v, 2) for k, v in est.items()}


def analyse() -> str:
    rows = [json.loads(l) for l in (OUT / "runs.jsonl").read_text().splitlines()]
    sel = selection()
    ids = [s["id"] for s in sel]
    by = {(r["id"], r["arm"]): r for r in rows}
    new = {s["id"]: set(s.get("gold_new", [])) for s in sel}
    for r in rows:                                          # a correctly named new file is not a wrong path
        r["new_named"] = [f for f in r["files"] if f in new.get(r["id"], ())]
        r["not_in_repo"] = [f for f in r["not_in_repo"] if f not in new.get(r["id"], ())]
        r["first_right_or_new"] = r["first_right"] or bool(r["files"] and r["files"][0] in new.get(r["id"], ()))
    lines = ["| arm | tasks | all gold files in top 5 | first file right (or a new file the fix creates) | "
             "mean gold share | functions named | new files named | mean input tokens | mean cost | mean seconds | "
             "mean turns | wrong paths | pages dropped |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        rs = [by[(i, arm)] for i in ids if (i, arm) in by]
        if not rs:
            continue
        n = len(rs)
        fs = [r["funcs_share"] for r in rs if r["funcs_share"] is not None]
        mean = lambda k: sum(r.get(k) or 0 for r in rs) / n  # noqa: E731
        lines.append(f"| {arm} | {n} | {sum(r['all_gold_top5'] for r in rs)} of {n} | {sum(r['first_right'] for r in rs)} of {n} "
                     f"({sum(r['first_right_or_new'] for r in rs)}) | {mean('gold_share'):.2f} | {sum(fs) / len(fs):.2f} | "
                     f"{sum(len(r['new_named']) for r in rs)} of {sum(len(new[r['id']]) for r in rs)} | {mean('input_tokens'):,.0f} | ${mean('cost_usd'):.3f} | "
                     f"{mean('seconds'):.0f} | {mean('turns'):.1f} | {sum(len(r['not_in_repo']) for r in rs)} | "
                     f"{sum(not r['pages_arrived'] for r in rs)} |")
    lines += ["", "| task | gold files | " + " | ".join(ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    for s in sel:
        cells = []
        for arm in ARMS:
            r = by.get((s["id"], arm))
            cells.append("" if r is None else ("all" if r["all_gold_top5"] else f"{round(r['gold_share'] * len(s['gold']))}/{len(s['gold'])}")
                         + (" (1st)" if r and r["first_right"] else ""))
        lines.append(f"| {s['id']} | {len(s['gold'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["select", "prepare", "estimate", "run", "analyse"])
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("REPO_ATLAS_WORK", "/tmp/repo_atlas_work")))
    ap.add_argument("--arms", nargs="+", default=["tree", "map", "atlas", "text"], choices=ARMS)
    ap.add_argument("--cap", type=float, default=12.0, help="stop launching calls once this much (list price) is spent")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--agent-cost", type=float, default=1.0, help="assumed cost per agent run for the estimate")
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "select":
        sel = select(a.work)
        for s in sel["tasks"]:
            print(f"{s['id']:28} gold={s['gold']} funcs={s['funcs']} atlas={s['atlas_image_tokens']:,}")
        for s in sel["skipped"]:
            print(f"skipped {s['id']}: {s['why']}")
    elif a.cmd == "prepare":
        sizes = {}
        for s in selection():
            m = contexts(a.work, s)
            sizes[s["id"]] = {k: m[k] for k in ("files", "pages", "image_tokens", "chars", "gold_outside_fileset",
                                                "gold_missing_at_base")} | {"uncovered": len(m["uncovered"])}
            print(s["id"], json.dumps(sizes[s["id"]]))
        (OUT / "contexts.json").write_text(json.dumps(sizes, indent=1))
    elif a.cmd == "estimate":
        print(json.dumps(estimate(a.work, a.agent_cost), indent=1))
    elif a.cmd == "run":
        one = [x for x in a.arms if x != "agent"]
        if one:
            run_one_shot(a.work, one, a.cap, a.workers)
        if "agent" in a.arms:
            run_agent(a.work, a.cap, min(a.workers, 4))
        print(f"[ra] spent ${spent():.2f} list price", flush=True)
    else:
        print(analyse())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
