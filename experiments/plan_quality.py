"""Plan-quality test: an orchestrator reads the repository once and writes a brief per ticket for worker agents.

Design: docs/claude/PLAN_QUALITY_TEST.md. Pre-registration, ticket lists, rows and results: results/plan_quality/README.md.
Tickets come from SWE-rebench (the fresh-task test's source, experiments/repo_atlas_swe.py); its funnel rows are
reused, its runner functions (checkouts, environments, grading) are imported, nothing of it is edited.

Planners (Opus 5.5, high effort, one instruction and brief format):
  tree   the file list            map    the repo map (signatures per file)
  atlas  the page images + index  scout  Claude Code with read-only tools (Read, Grep, Glob) on an export
Layer 1: one brief per ticket, scored against the real fix (files, functions, invented paths and symbols, size;
         the approach judged blind by Opus 5.5 at low effort, one call per brief).
Layer 2: groups of tickets of one repository planned together on one commit (COLLIDES_WITH, ORDER).
Layer 3: Claude Code workers with edit tools, graded by the ticket's own tests: Opus high with the atlas brief, Opus
         high with the tree brief, Sonnet medium and Haiku low with the atlas brief (no-brief rows: repo_atlas_swe).

    python3 experiments/plan_quality.py select                 # fit + closed-book screen beyond the first 80 (cheap)
    python3 experiments/plan_quality.py prepare                # contexts, groups, size estimate (no calls)
    python3 experiments/plan_quality.py plan  --layer 1 [--ids ...] [--planners ...] --cap 45
    python3 experiments/plan_quality.py judge [--ids ...] --cap 45
    python3 experiments/plan_quality.py plan  --layer 2 --cap 45
    python3 experiments/plan_quality.py work  --cap 45 [--max-tickets N]
    python3 experiments/plan_quality.py extend --cap 45               # layer 1 beyond the layer-3 tickets, whole tickets
    python3 experiments/plan_quality.py analyse

Rows: results/plan_quality/{funnel,briefs,judge,groups,work,spend}.jsonl, tickets.json. Clones, exports, venvs,
pages and transcripts in --work (scratchpad), never in the repository. Every step skips finished rows (resumable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

OUT = ROOT / "results" / "plan_quality"
SWE_OUT = ROOT / "results" / "repo_atlas_swe"          # read only
DEFAULT_WORK = Path(os.environ.get("PQ_WORK", "/tmp/claude-0/-home-user-visual-bpe/"
                                   "588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/pq"))
MODEL = "claude-opus-5-5"
PLAN_EFFORT = "high"
JUDGE_EFFORT = "low"
TARGET = 40
PLANNERS = ("tree", "map", "atlas", "scout")
GROUP_PLANNERS = ("tree", "map", "atlas")
SCOUT_TURNS = 40
PAGE_ARRIVAL = 0.97
SEED = 20261112
TIERS = {"haiku": ("claude-haiku-5-5", "low"), "sonnet": ("claude-sonnet-5-5", "medium"),
         "opus": ("claude-opus-5-5", "high")}
TIER_RANK = {"haiku": 1, "sonnet": 2, "opus": 3}
SIZE_RANK = {"S": 1, "M": 2, "L": 3}
WORK_ARMS = {"opus_atlas": ("opus", "atlas"), "opus_tree": ("opus", "tree"), "sonnet_atlas": ("sonnet", "atlas"),
             "haiku_atlas": ("haiku", "atlas")}
PRICE_WRITE, PRICE_OUT = 5e-6, 20e-6
NO_MEMORY = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
FIX_CHARS = 24_000                                       # the judge sees at most this much of the real fix

# --------------------------------------------------------------------------- prompts (pre-registered verbatim)
SYSTEM_PLAN = ("You are the planning lead for a team of coding agents. You look at a repository and write a short "
               "brief per ticket for the worker agent that will implement it. Answer only in the format asked for.")
BRIEF_FORMAT = """TICKET: <id>
SUMMARY: <one sentence>
FILES: <path> : <what changes>            (at most 6, most important first)
FUNCTIONS: <names>
APPROACH: <at most 3 sentences>
RISKS: <at most 2>
COLLIDES_WITH: <ticket ids that touch the same files, or none>
ORDER: <tickets that should land first, or none>
SIZE: S | M | L
WORKER: <haiku | sonnet | opus> / <low | medium | high>
CONFIDENCE: low | medium | high"""
INSTRUCTION = """Write a brief for the worker agent that will implement {which}. The worker gets the ticket and your brief, nothing else; it reads the files it edits and may depart from your brief where the code disagrees.

Use exactly this format, one field per line; repeat the FILES: line once per file:

{fmt}

Rules:
* Paths exactly as they appear in the repository. A file or function that does not exist yet and has to be created: write " (new)" after it.
* FUNCTIONS: comma-separated, as function or Class.method.
* SIZE: S = one file, a local change, no design choice; M = two or three files, or one decision that is not obvious; L = four or more files, or a change across modules.
* WORKER: the model and reasoning level that should implement the ticket. Default per size: S = haiku / low, M = sonnet / medium, L = opus / high. Choose the cheapest worker you expect to resolve the ticket; if you choose other than the default for the size, say why in RISKS.
{batch_rule}"""
SINGLE_RULE = "* This is the only ticket in this batch: COLLIDES_WITH: none and ORDER: none."
GROUP_RULE = ("* There are {n} tickets in this batch, all against this same commit. Write one brief per ticket, in the "
              "order given, with a blank line between briefs. COLLIDES_WITH: the ids of other tickets in this batch whose "
              "change touches any of the same files. ORDER: the ids of tickets in this batch that should land before "
              "this one.")
TREE_HEAD = "Repository {repo}: its {n} source files (tests, docs and data files left out).\n\n"
MAP_HEAD = ("Repository {repo}: a repo map of its {n} source files (tests, docs and data files left out). Per file: its "
            "path, then the signature line of every class, function and method in it, indented.\n\n")
ATLAS_HEAD = ("The images above are the {n} source files of the repository {repo} (tests, docs and data files left out), "
              "in this order, each starting with a header ▶path◀. The code is minified: §<n> marks the start of original "
              "line n (shown on every 10th line, a bare § starts the others); each » before a line is one level of "
              "indentation where indentation matters; blank lines and other indentation are not shown. Index (file: "
              "page):\n\n")
SCOUT_HEAD = ("The current directory is a checkout of the {repo} repository. Use your tools (Read, Grep, Glob) to look "
              "at the code as much as you need; do not change anything. Then reply with the brief only.\n\n")

SYSTEM_JUDGE = """You grade a brief that a planner wrote for a coding ticket, against the change that actually fixed the ticket. The brief was written before the fix existed. It does not have to match the fix line by line: a different approach that would also work counts as right. Judge three things:
CAUSE: does the brief identify the actual cause of the problem (for a feature request: what has to be added)?
PLACE: does it point to the right place in the code: the files and functions the fix changes, or others where the change would work as well?
OUTCOME: would a competent engineer who follows the brief arrive at the behaviour the ticket asks for?
Answer each with yes, partly or no. Reply in exactly four lines:
CAUSE: yes|partly|no
PLACE: yes|partly|no
OUTCOME: yes|partly|no
REASON: <one sentence>"""
JUDGE_PROMPT = """<ticket>
{issue}
</ticket>

<real_fix note="source files only; tests, docs and changelogs left out{trunc}">
{fix}
</real_fix>

<brief>
{brief}
</brief>"""

WORK_BRIEF = ("\n\nA planner who looked at the repository wrote this brief for the issue. Use it as a starting point; read "
              "the files before you edit them, and depart from the brief where the code disagrees.\n\n<brief>\n{brief}\n"
              "</brief>")

KEYS = ("TICKET", "SUMMARY", "FILES", "FUNCTIONS", "APPROACH", "RISKS", "COLLIDES_WITH", "ORDER", "SIZE", "WORKER",
        "CONFIDENCE")
CONTENT_KEYS = ("SUMMARY", "FILES", "FUNCTIONS", "APPROACH", "RISKS")


# --------------------------------------------------------------------------- pure functions (tested)
def split_briefs(text: str) -> list[str]:
    """A reply holding several briefs -> one text per brief (a brief starts at a line beginning ``TICKET:``)."""
    parts, cur = [], []
    for line in (text or "").splitlines():
        if re.match(r"^\s*[*#>-]*\s*\**TICKET\**\s*:", line, re.I) and cur and any("TICKET" in l.upper() for l in cur):
            parts.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur and "\n".join(cur).strip():
        parts.append("\n".join(cur))
    return [p for p in parts if re.search(r"TICKET\**\s*:", p, re.I)] or ([text] if (text or "").strip() else [])


_KEY_RX = re.compile(r"^\s*[*#>-]*\s*\**(" + "|".join(KEYS) + r")\**\s*:\s*(.*)$", re.I)
_PATH_RX = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*`?(?P<path>[\w.@+-]+(?:/[\w.@+-]+)*/?)`?\s*(?P<new>\(new\))?\s*"
                      r"(?:(?::|—|–|\s-)\s*(?P<what>.*))?$", re.I)
_NEW_RX = re.compile(r"\(new\)|\bnew (file|module)\b|\bcreate[sd]?\b", re.I)


def _ids(v: str) -> list[str]:
    if re.match(r"^\s*(none|n/a|-)?\s*\.?\s*$", v or "", re.I) or re.match(r"^\s*none\b", v or "", re.I):
        return []
    return list(dict.fromkeys(re.findall(r"\bT\d+\b", v)))


def parse_brief(text: str) -> dict:
    """One brief -> fields. FILES: one entry per FILES line (or continuation line), {path, what, new}. FUNCTIONS:
    [{name, new}]. WORKER: {model, effort}. ``problems``: departures from the format (see validate_brief)."""
    fields: dict[str, list[str]] = {}
    cur = None
    for line in (text or "").splitlines():
        m = _KEY_RX.match(line)
        if m:
            cur = m.group(1).upper()
            fields.setdefault(cur, []).append(m.group(2).strip())
        elif cur and line.strip():
            fields[cur].append(line.strip())
    one = lambda k: " ".join(x for x in fields.get(k, []) if x).strip()  # noqa: E731
    files = []
    for entry in fields.get("FILES", []):
        if not entry.strip():
            continue
        m = _PATH_RX.match(entry.replace("**", ""))
        if not m or "/" not in m.group("path") and "." not in m.group("path"):
            continue
        what = (m.group("what") or "").strip()
        path = m.group("path").strip().lstrip("./") if m.group("path").startswith("./") else m.group("path").strip()
        if path.lower() in ("none", "n/a"):
            continue
        files.append({"path": path, "what": what, "new": bool(m.group("new")) or bool(_NEW_RX.search(entry))})
    funcs = []
    for item in re.split(r"[,;\n]", "\n".join(fields.get("FUNCTIONS", []))):
        item = item.strip()
        new = "(new)" in item.lower()
        m = re.search(r"[A-Za-z_][\w.]*", re.sub(r"\(new\)", "", item, flags=re.I).replace("`", ""))
        if not m or m.group(0).lower() in ("none", "n", "a"):
            continue
        name = m.group(0).strip(".")
        if name and name not in [f["name"] for f in funcs]:
            funcs.append({"name": name, "new": new})
    w = re.search(r"\b(haiku|sonnet|opus)\b\s*/\s*\b(low|medium|high|xhigh|max)\b", one("WORKER"), re.I)
    size = re.search(r"\b([SML])\b", one("SIZE"))
    conf = re.search(r"\b(low|medium|high)\b", one("CONFIDENCE"), re.I)
    b = {"ticket": (re.search(r"\bT\d+\b", one("TICKET")) or [None])[0], "summary": one("SUMMARY"), "files": files,
         "functions": funcs, "approach": one("APPROACH"), "risks": one("RISKS"),
         "collides": _ids(one("COLLIDES_WITH")), "order": _ids(one("ORDER")),
         "size": size.group(1) if size else None,
         "worker": {"model": w.group(1).lower(), "effort": w.group(2).lower()} if w else None,
         "confidence": conf.group(1).lower() if conf else None, "keys": [k for k in KEYS if k in fields]}
    b["problems"] = validate_brief(b)
    return b


def validate_brief(b: dict) -> list[str]:
    p = [f"missing {k}" for k in KEYS if k not in b["keys"]]
    if not b["files"]:
        p.append("no files")
    if len(b["files"]) > 6:
        p.append("more than 6 files")
    if b["size"] is None:
        p.append("bad SIZE")
    if b["worker"] is None:
        p.append("bad WORKER")
    if b["confidence"] is None:
        p.append("bad CONFIDENCE")
    if len(re.findall(r"[.!?](\s|$)", b["approach"])) > 3:
        p.append("APPROACH over 3 sentences")
    return p


def path_exists(path: str, repo_paths: set[str]) -> bool:
    """A file in the repository, or a directory holding one."""
    p = path.strip("/")
    return p in repo_paths or any(x.startswith(p + "/") for x in repo_paths)


def symbol_exists(name: str, words: set[str]) -> bool:
    """A named function / Class.method exists when its last component occurs as an identifier in the repository."""
    last = name.split(".")[-1]
    return last in words


def invented(b: dict, repo_paths: set[str], words: set[str]) -> dict:
    """Named paths and symbols absent from the repository at the commit, except those the brief marks as new."""
    paths = [f for f in b["files"]]
    syms = [f for f in b["functions"]]
    miss_p = [f["path"] for f in paths if not f["new"] and not path_exists(f["path"], repo_paths)]
    miss_s = [f["name"] for f in syms if not f["new"] and not symbol_exists(f["name"], words)]
    return {"named": len(paths) + len(syms), "paths_named": len(paths), "symbols_named": len(syms),
            "paths_missing": miss_p, "symbols_missing": miss_s, "missing": len(miss_p) + len(miss_s),
            "new_marked": sum(f["new"] for f in paths) + sum(f["new"] for f in syms)}


def is_source(path: str) -> bool:
    from repo_atlas_eval import is_gold
    return is_gold(path)


def file_scores(b: dict, gold: list[str], gold_all: list[str]) -> dict:
    """gold = the fix's source files that exist at the base commit (scored); gold_all also holds files the fix
    creates. Recall over gold; 'wrong' = named source files that are in neither (tests and docs a brief names are
    counted apart, not as wrong)."""
    named = [f["path"].strip("/") for f in b["files"]]
    src = [p for p in named if is_source(p)]
    hit = [g for g in gold if g in named]
    wrong = [p for p in src if p not in gold_all]
    return {"gold_share": len(hit) / len(gold) if gold else None, "gold_hit": hit,
            "new_named": [p for p in named if p in gold_all and p not in gold], "named_src": len(src),
            "wrong": wrong, "wrong_share": len(wrong) / len(src) if src else None,
            "non_source_named": [p for p in named if not is_source(p)], "first_right": bool(named) and named[0] in gold_all}


def func_score(b: dict, funcs: list[str]) -> dict:
    """Share of the fix's changed functions named anywhere in FUNCTIONS, FILES descriptions or APPROACH."""
    text = " ".join([f["name"] for f in b["functions"]] + [f["what"] for f in b["files"]] + [b["approach"]])
    named = [f for f in funcs if re.search(rf"(?<![\w$]){re.escape(f)}(?![\w$])", text)]
    return {"funcs_named": named, "funcs_share": len(named) / len(funcs) if funcs else None}


def fix_size(patch: str) -> dict:
    """The real fix's size over its source files: files, functions (repo_atlas_eval.gold_functions), lines changed."""
    from repo_atlas_eval import gold_files, gold_functions
    lines, cur = 0, None
    for line in patch.splitlines():
        m = re.match(r"^diff --git a/\S+ b/(\S+)$", line)
        if m:
            cur = is_source(m.group(1))
            continue
        if cur and line[:1] in "+-" and not line.startswith(("+++", "---")):
            lines += 1
    return {"files": len(gold_files(patch)), "functions": len(gold_functions(patch)), "lines": lines}


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def composite_size(sizes: list[dict]) -> list[float]:
    """Mean of the ranks of files, functions and lines changed (the primary 'real size' for prediction 5)."""
    rs = [ranks([s[k] for s in sizes]) for k in ("files", "functions", "lines")]
    return [statistics.mean(r[i] for r in rs) for i in range(len(sizes))]


def real_size_class(files: int) -> str:
    """The design's own size definitions applied to the fix's source files (S one file, M two or three, L 4+)."""
    return "S" if files <= 1 else "M" if files <= 3 else "L"


def measured_tier(resolved: dict[str, bool]) -> str:
    """Cheapest of haiku (low), sonnet (medium), opus (high) whose run resolved the ticket; 'none' if none did."""
    for t in ("haiku", "sonnet", "opus"):
        if resolved.get(t):
            return t
    return "none"


def tier_verdict(planned: str | None, measured: str) -> str:
    if planned is None:
        return "no choice"
    if measured == "none":
        return "unresolvable"
    d = TIER_RANK[planned] - TIER_RANK[measured]
    return "right" if d == 0 else "too large" if d > 0 else "too small"


def policies(rows: list[dict]) -> dict:
    """rows: {planned, runs: {tier: {resolved, cost_usd}}} per ticket -> per policy: resolved, cost."""
    out = {}
    for name, pick in (("always opus high", lambda r: "opus"), ("always haiku low", lambda r: "haiku"),
                       ("always sonnet medium", lambda r: "sonnet"), ("planner's choice", lambda r: r["planned"])):
        res, cost, n = 0, 0.0, 0
        for r in rows:
            t = pick(r)
            if t is None or t not in r["runs"]:
                continue
            n += 1
            res += bool(r["runs"][t]["resolved"])
            cost += r["runs"][t]["cost_usd"] or 0.0
        out[name] = {"tickets": n, "resolved": res, "cost_usd": round(cost, 4)}
    return out


def true_pairs(gold_by_ticket: dict[str, list[str]]) -> set[frozenset]:
    ids = list(gold_by_ticket)
    return {frozenset((a, b)) for i, a in enumerate(ids) for b in ids[i + 1:]
            if set(gold_by_ticket[a]) & set(gold_by_ticket[b])}


def flagged_pairs(collides: dict[str, list[str]]) -> set[frozenset]:
    """A pair is flagged when either brief names the other in COLLIDES_WITH (ids outside the batch ignored)."""
    ids = set(collides)
    return {frozenset((a, b)) for a, lst in collides.items() for b in lst if b in ids and b != a}


def collision_score(flagged: set, true: set) -> dict:
    return {"true": len(true), "flagged": len(flagged), "found": len(flagged & true), "wrong": len(flagged - true),
            "missed": len(true - flagged)}


def parse_judge(text: str) -> dict | None:
    v = {"yes": 2, "partly": 1, "no": 0}
    out = {}
    for k in ("CAUSE", "PLACE", "OUTCOME"):
        m = re.search(rf"^\s*\**{k}\**\s*:\s*\**\s*(yes|partly|no)\b", text or "", re.I | re.M)
        if not m:
            return None
        out[k.lower()] = m.group(1).lower()
    out["score"] = sum(v[out[k]] for k in ("cause", "place", "outcome"))
    m = re.search(r"^\s*\**REASON\**\s*:\s*(.+)$", text or "", re.I | re.M)
    out["reason"] = m.group(1).strip() if m else ""
    return out


def brief_content(text: str) -> str:
    """The brief lines a judge or a worker sees: SUMMARY, FILES, FUNCTIONS, APPROACH, RISKS (no ticket id, batch
    fields, size, worker or confidence)."""
    keep, cur = [], None
    for line in (text or "").splitlines():
        m = _KEY_RX.match(line)
        if m:
            cur = m.group(1).upper()
        if cur in CONTENT_KEYS and line.strip():
            keep.append(line.rstrip())
    return "\n".join(keep)


def source_patch(patch: str) -> str:
    """The per-file sections of a unified diff whose path is a source file."""
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    return "".join(p for p in parts if (m := re.match(r"diff --git a/(\S+) b/(\S+)", p)) and is_source(m.group(2)))


def make_groups(tickets: list[dict], size: int = 3) -> list[list[dict]]:
    """Tickets of one repository, oldest first, in consecutive groups of ``size``; a remainder of 1-2 joins the last
    group. Repositories with fewer than ``size`` tickets form no group."""
    by: dict[str, list[dict]] = {}
    for t in tickets:
        by.setdefault(t["repo"], []).append(t)
    out = []
    for repo in sorted(by):
        ts = sorted(by[repo], key=lambda t: (t["created_at"], t["id"]))
        if len(ts) < size:
            continue
        gs = [ts[i:i + size] for i in range(0, len(ts) - len(ts) % size, size)]
        gs[-1] += ts[len(ts) - len(ts) % size:]
        out += gs
    return out


def memory_slug(cwd: str) -> str:
    """Claude Code's per-project directory name under ~/.claude/projects."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


# --------------------------------------------------------------------------- io
_alock = threading.Lock()
_rlock = threading.Lock()
_reserved = [0.0]
STOP = threading.Event()
USAGE_LIMIT = re.compile(r"hit your .{0,20}limit|usage limit|rate limit reached|resets \d", re.I)


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def append(name: str, row: dict) -> None:
    with _alock:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / name, "a") as f:
            f.write(json.dumps(row) + "\n")


def spent() -> float:
    return sum(r.get("cost_usd") or 0 for r in jl(OUT / "spend.jsonl"))


def book(what: str, cost: float, **kw) -> None:
    if cost:
        append("spend.jsonl", {"what": what, "cost_usd": cost, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw})


def reserve(cost: float, cap: float) -> bool:
    with _rlock:
        if spent() + _reserved[0] + cost > cap:
            return False
        _reserved[0] += cost
        return True


def release(cost: float) -> None:
    with _rlock:
        _reserved[0] -= cost


def limit_hit(text: str | None) -> bool:
    if text and USAGE_LIMIT.search(text):
        if not STOP.is_set():
            print(f"[pq] USAGE LIMIT: {text[:300]} -- stopping, nothing more is launched", flush=True)
        STOP.set()
        return True
    return False


def S():
    import repo_atlas_swe
    return repo_atlas_swe


def task_map(work: Path) -> dict[str, dict]:
    return S().task_map(work)


# --------------------------------------------------------------------------- selection
def screen_call(t: dict, co: Path) -> dict:
    """The fresh-task test's closed-book screen, same prompt (so its cached answers are reused)."""
    import repo_atlas as RA
    import repo_atlas_eval as E
    from vbpe import claude_eval as CE
    paths = [x["path"] for x in RA.collect(co, False, 10)[0]]
    tree = TREE_HEAD.format(repo=t["repo"], n=len(paths)) + "\n".join(paths) + "\n"
    rec = CE.transcribe([], E.SYSTEM, E.prompt_for("tree", tree, t["problem_statement"], t["repo"]), model=MODEL,
                        effort="high", timeout=600.0, max_retries=1)
    if not rec.get("cached"):
        book(f"{t['instance_id']} screen", rec.get("cost_usd") or 0.0)
    files, plan = E.parse_answer(rec.get("text", ""))
    return {"files": files, "plan": plan, "error": rec.get("error"), "cached": rec.get("cached")}


def select(work: Path, workers: int) -> None:
    sw = S()
    rows = sw.load_rows(work)
    cands = sw.candidate_order(rows)
    prior = {r["id"]: r for r in sw.jl(SWE_OUT / "funnel.jsonl")}
    mine = {r["id"]: r for r in jl(OUT / "funnel.jsonl") if r.get("stage") != "screen error"}
    lock = threading.Lock()
    state = {"passed": sum(r.get("passed", False) for r in mine.values())}
    todo = [t for t in cands if t["instance_id"] not in mine]
    print(f"[sel] {len(cands)} candidates; {len(mine)} recorded, {state['passed']} passed; {len(todo)} to go", flush=True)

    def one(t):
        if STOP.is_set():
            return
        with lock:
            if state["passed"] >= TARGET:
                return
        iid = t["instance_id"]
        p = prior.get(iid)
        row = {"id": iid, "repo": t["repo"], "created_at": t["created_at"], "n_gold": len(sw.gold_files(t["patch"])),
               "passed": False}
        if p and ("screen" in p or p.get("stage") == "atlas too large"):
            row.update({k: p.get(k) for k in ("image_tokens", "files", "source_chars", "gold", "gold_new")})
            row["source"] = "repo_atlas_swe funnel"
            if p.get("stage") == "atlas too large":
                row["stage"] = "atlas too large"
            else:
                row["screen"] = p["screen"]
                row["stage"] = "screen: closed-book names every gold file" if p["screen"]["drop"] else "passed"
        else:
            row["source"] = "this test"
            try:
                f = sw.fit(work, t)
                row.update({k: f[k] for k in ("image_tokens", "files", "source_chars", "gold", "gold_new")})
                if not f["fits"]:
                    row["stage"] = "atlas too large"
                else:
                    s = screen_call(t, Path(f["checkout"]))
                    row["screen"] = {"files": s["files"], "plan": s["plan"], "drop": sw.screen_drop(s["files"], f["gold"]),
                                     "error": s["error"]}
                    if s["error"]:
                        row["stage"] = "screen error"
                        limit_hit(s["error"])
                    else:
                        row["stage"] = "screen: closed-book names every gold file" if row["screen"]["drop"] else "passed"
            except Exception as ex:  # noqa: BLE001
                row["stage"] = f"error: {str(ex)[-300:]}"
            finally:
                shutil.rmtree(work / "co" / iid, ignore_errors=True)
        row["passed"] = row["stage"] == "passed"
        with lock:
            state["passed"] += row["passed"]
        append("funnel.jsonl", row)
        print(f"[sel] {row['created_at'][:10]} {iid:48} gold={row['n_gold']} tok={row.get('image_tokens')} -> "
              f"{row['stage']} ({row['source']}; passed {state['passed']})", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


def layer3_ids() -> list[str]:
    """The fresh-task test's compared tasks, in its run order (newest first), the contaminated one left out."""
    sw = S()
    bad = json.loads((SWE_OUT / "contaminated.json").read_text())
    rows = [r for r in sw.jl(SWE_OUT / "funnel.jsonl") if r.get("accepted")]
    order = [r["id"] for r in sorted(rows, key=lambda r: (r["created_at"], r["id"]), reverse=True)]
    return [i for i in order if i not in bad]


def prepare(work: Path) -> dict:
    """tickets.json: every ticket that passed (layer 1-2 order: the layer-3 tickets first in the fresh-task order,
    then the others newest first), its gold files, functions and real size; the groups for layer 2."""
    from repo_atlas_eval import gold_functions
    sw = S()
    tm = task_map(work)
    passed = [r for r in jl(OUT / "funnel.jsonl") if r.get("passed")]
    l3 = layer3_ids()
    newest = sorted(passed, key=lambda r: (r["created_at"], r["id"]), reverse=True)
    order = [i for i in l3 if any(r["id"] == i for r in passed)] + [r["id"] for r in newest if r["id"] not in l3]
    fun = {r["id"]: r for r in passed}
    tickets = []
    for i in order:
        t = tm[i]
        meta = contexts(work, t)
        tickets.append({"id": i, "repo": t["repo"], "created_at": t["created_at"][:10], "base_commit": t["base_commit"],
                        "gold": fun[i]["gold"], "gold_new": fun[i]["gold_new"], "gold_all": sw.gold_files(t["patch"]),
                        "funcs": gold_functions(t["patch"]), "size": fix_size(t["patch"]),
                        "real_class": real_size_class(len(sw.gold_files(t["patch"]))),
                        "layer3": i in l3, "files_in_atlas": meta["n_files"], "pages": meta["pages"],
                        "image_tokens_formula": meta["image_tokens_formula"], "map_chars": meta["map_chars"],
                        "tree_chars": meta["tree_chars"]})
    groups = []
    for g in make_groups(tickets):
        base = g[0]
        store = sw.git_store(work, tm[base["id"]])
        at = set(subprocess.run(["git", "-C", str(store), "ls-tree", "-r", "--name-only", base["base_commit"]],
                                capture_output=True, text=True, check=True).stdout.split())
        keep = [t for t in g if all(x in at for x in t["gold"])]
        dropped = [t["id"] for t in g if t not in keep]
        groups.append({"gid": f"G{len(groups) + 1}", "repo": base["repo"], "commit_of": base["id"],
                       "base_commit": base["base_commit"], "tickets": [t["id"] for t in keep], "dropped": dropped,
                       "usable": len(keep) >= 3})
    sel = {"tickets": tickets, "groups": groups, "layer3": l3,
           "layer2_evaluable": sum(g["usable"] for g in groups) >= 3}
    (OUT / "tickets.json").write_text(json.dumps(sel, indent=1))
    return sel


def tickets() -> dict:
    return json.loads((OUT / "tickets.json").read_text())


# --------------------------------------------------------------------------- contexts
def contexts(work: Path, t: dict) -> dict:
    """Tree, map, atlas pages, repository paths and identifiers at the ticket's base commit; an export without git
    for the scout. Built once."""
    import repo_atlas as RA
    import repo_atlas_eval as E
    sw = S()
    iid = t["instance_id"]
    d = work / "ctx" / iid
    meta_p = d / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())
    co = sw.fresh_repo(work, t, work / "co" / iid / t["repo"].split("/")[1])
    files, _ = RA.collect(co, False, 10)
    paths = [f["path"] for f in files]
    texts = [(p, (co / p).read_text(encoding="utf-8")) for p in paths]
    d.mkdir(parents=True, exist_ok=True)
    (d / "tree.txt").write_text(TREE_HEAD.format(repo=t["repo"], n=len(paths)) + "\n".join(paths) + "\n")
    (d / "map.txt").write_text(MAP_HEAD.format(repo=t["repo"], n=len(paths)) + E.build_map(texts))
    am = RA.build(co, d / "atlas", 88.2, False, 10)
    (d / "atlas.txt").write_text(ATLAS_HEAD.format(repo=t["repo"], n=len(paths)) + (d / "atlas" / "index.txt").read_text())
    all_paths = subprocess.run(["git", "ls-files"], cwd=co, capture_output=True, text=True, check=True).stdout.split()
    words = set()
    for p in all_paths:
        f = co / p
        try:
            if f.is_file() and f.stat().st_size < 3_000_000:
                words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            pass
    (d / "repo_paths.json").write_text(json.dumps(all_paths))
    (d / "words.json").write_text(json.dumps(sorted(words)))
    meta = {"n_files": len(paths), "pages": len(am["pages"]), "image_tokens_build": am["image_tokens"],
            "image_tokens_formula": E.image_tokens([tuple(p["size"]) for p in am["pages"]]),
            "map_chars": len((d / "map.txt").read_text()), "tree_chars": len((d / "tree.txt").read_text()),
            "atlas_dir": str(d / "atlas")}
    meta_p.write_text(json.dumps(meta, indent=1))
    shutil.rmtree(work / "co" / iid, ignore_errors=True)
    return meta


def export(work: Path, t: dict) -> Path:
    """The tree at the base commit without .git, in a neutral directory (no ticket id, no arm name)."""
    sw = S()
    tag = hashlib.sha1(("scout" + t["instance_id"]).encode()).hexdigest()[:10]
    d = work / "x" / tag / t["repo"].split("/")[1]
    if not d.exists():
        d.mkdir(parents=True)
        store = sw.git_store(work, t)
        arch = subprocess.run(["git", "-C", str(store), "archive", t["base_commit"]], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=arch, check=True)
    return d


def repo_sets(work: Path, iid: str) -> tuple[set, set]:
    d = work / "ctx" / iid
    return set(json.loads((d / "repo_paths.json").read_text())), set(json.loads((d / "words.json").read_text()))


# --------------------------------------------------------------------------- planner calls
def ticket_block(items: list[tuple[str, str]]) -> str:
    return "\n\n".join(f'<ticket id="{tid}">\n{issue}\n</ticket>' for tid, issue in items)


def plan_prompt(ctx_text: str, items: list[tuple[str, str]]) -> str:
    group = len(items) > 1
    which = "each ticket below" if group else "the ticket below"
    rule = GROUP_RULE.format(n=len(items)) if group else SINGLE_RULE
    return (ctx_text + "\n" + ticket_block(items) + "\n\n"
            + INSTRUCTION.format(which=which, fmt=BRIEF_FORMAT, batch_rule=rule))


def memory_check_and_clear(cwd: Path) -> dict:
    """Auto-memory is off (CLAUDE_CODE_DISABLE_AUTO_MEMORY=1); also remove the run's project store, and record
    whether anything was written to it."""
    p = Path.home() / ".claude" / "projects" / memory_slug(str(cwd))
    mem = p / "memory"
    info = {"project_dir": p.exists(), "memory_files": sorted(x.name for x in mem.iterdir()) if mem.is_dir() else []}
    shutil.rmtree(p, ignore_errors=True)
    return info


def scout_once(work: Path, t: dict, items: list[tuple[str, str]]) -> dict:
    from vbpe import claude_eval as CE
    cwd = export(work, t)
    memory_check_and_clear(cwd)
    prompt = SCOUT_HEAD.format(repo=t["repo"]) + plan_prompt("", items).lstrip("\n")
    cmd = [CE.CLAUDE_BIN, "-p", "--output-format", "stream-json", "--verbose", "--model", MODEL, "--effort",
           PLAN_EFFORT, "--tools", "Read,Grep,Glob", "--permission-prompts", "none", "--setting-sources", "",
           "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence", "--max-turns",
           str(SCOUT_TURNS), "--append-system-prompt", SYSTEM_PLAN]
    t0 = time.time()
    try:
        p = subprocess.run(cmd + [prompt], capture_output=True, text=True, cwd=cwd, timeout=1800,
                           env={**CE.build_env(), **S().NO_INDEX, **NO_MEMORY}, stdin=subprocess.DEVNULL)
        stdout, rc, err = p.stdout, p.returncode, p.stderr
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc, err = -9, "timeout"
    secs = time.time() - t0
    mem = memory_check_and_clear(cwd)
    tdir = work / "transcripts" / "scout"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{t['instance_id']}_{items[0][0]}.jsonl").write_text(stdout)
    ev = [json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]
    tot = S().stream_totals(ev)
    res = next((e for e in reversed(ev) if e.get("type") == "result"), None)
    calls = [c for e in ev if e.get("type") == "assistant" for c in e["message"]["content"] if c.get("type") == "tool_use"]
    outside = [json.dumps(c.get("input"))[:200] for c in calls
               if any(isinstance(v, str) and v.startswith("/") and not v.startswith(str(cwd))
                      for v in (c.get("input") or {}).values())]
    text = (res or {}).get("result") or ""
    if not text:
        texts = [c["text"] for e in ev if e.get("type") == "assistant" for c in e["message"]["content"] if c.get("type") == "text"]
        text = texts[-1] if texts else ""
    err_txt = None
    if res is None or (res.get("is_error") and res.get("subtype") != "error_max_turns"):
        err_txt = f"rc={rc} {str((res or {}).get('result'))[:300]} {err[-300:]}"
    names = [c["name"] for c in calls]
    return {"text": text, "error": err_txt, "cost_usd": tot["cost_usd"], "input_tokens": tot["input_tokens"],
            "output_tokens": tot["output_tokens"], "usage": tot["usage"], "turns": tot["turns"], "results": tot["results"],
            "seconds": round(secs, 1), "subtype": (res or {}).get("subtype"),
            "tools": {k: names.count(k) for k in sorted(set(names))}, "outside_cwd": outside[:10],
            "permission_denials": [str(x)[:200] for x in (res or {}).get("permission_denials") or []][:10],
            "memory": mem, "model_usage": {m: round(v.get("costUSD", 0.0), 4) for m, v in ((res or {}).get("modelUsage") or {}).items()}}


def one_shot(work: Path, planner: str, t: dict, items: list[tuple[str, str]], meta: dict) -> dict:
    from PIL import Image
    from vbpe import claude_eval as CE
    d = work / "ctx" / t["instance_id"]
    ctx = (d / f"{planner}.txt").read_text()
    imgs = [Image.open(Path(meta["atlas_dir"]) / f"page_{i + 1:03d}.png") for i in range(meta["pages"])] \
        if planner == "atlas" else []
    rec = CE.transcribe(imgs, SYSTEM_PLAN, plan_prompt(ctx, items), model=MODEL, effort=PLAN_EFFORT, hires=True,
                        timeout=1200.0, max_retries=1)
    out = {k: rec.get(k) for k in ("text", "error", "cost_usd", "input_tokens", "output_tokens", "usage", "cached",
                                   "thinking_tokens", "stop_reason", "model", "model_usage")}
    out["seconds"] = rec.get("latency_s")
    out["turns"] = 1
    if planner == "atlas":
        out["image_tokens_formula"] = meta["image_tokens_formula"]
        out["pages_arrived"] = (rec.get("input_tokens") or 0) >= PAGE_ARRIVAL * meta["image_tokens_formula"]
    return out


def call_bound(planner: str, meta: dict, items: int, scout_bound: list) -> float:
    if planner == "scout":
        return scout_bound[0]
    chars = {"tree": meta["tree_chars"], "map": meta["map_chars"], "atlas": 3000}[planner] + 4000 * items
    img = meta["image_tokens_formula"] if planner == "atlas" else 0
    return (chars / 3.0 + img) * PRICE_WRITE * 1.1 + 6000 * items * PRICE_OUT


def run_plans(work: Path, layer: int, planners: list[str], ids: list[str] | None, cap: float, workers: int) -> None:
    tm = task_map(work)
    sel = tickets()
    done = {(r["layer"], r["unit"], r["planner"]) for r in jl(OUT / "briefs.jsonl")}
    if layer == 1:
        units = [(x["id"], [x["id"]]) for x in sel["tickets"] if not ids or x["id"] in ids]
        planners = [p for p in planners]
    else:
        if not sel["layer2_evaluable"]:
            print("[pq] layer 2 not evaluable (fewer than three usable groups)", flush=True)
            return
        units = [(g["gid"], g["tickets"]) for g in sel["groups"] if g["usable"] and (not ids or g["gid"] in ids)]
        planners = [p for p in planners if p in GROUP_PLANNERS]
    jobs = [(u, members, p) for u, members in units for p in planners if (layer, u, p) not in done]
    scout_rows = [r["cost_usd"] for r in jl(OUT / "briefs.jsonl") if r["planner"] == "scout" and r.get("cost_usd")]
    scout_bound = [max(1.5, 2 * max(scout_rows)) if scout_rows else 1.5]
    print(f"[pq] layer {layer}: {len(jobs)} planner calls to make; spent ${spent():.2f} (cap ${cap})", flush=True)

    def one(job):
        unit, members, planner = job
        if STOP.is_set():
            return
        base = tm[members[0]]                               # layer 2: the group's commit is its first ticket's
        meta = contexts(work, base)
        items = [(f"T{k + 1}", tm[m]["problem_statement"]) for k, m in enumerate(members)]
        bound = call_bound(planner, meta, len(items), scout_bound)
        if not reserve(bound, cap):
            print(f"[pq] SKIP {unit} {planner}: ${spent():.2f} spent + in flight + ${bound:.2f} > cap ${cap}", flush=True)
            return
        try:
            rec = scout_once(work, base, items) if planner == "scout" else one_shot(work, planner, base, items, meta)
        finally:
            release(bound)
        if not rec.get("cached"):
            book(f"L{layer} {unit} {planner}", rec.get("cost_usd") or 0.0, error=rec.get("error"))
        if planner == "scout" and rec.get("cost_usd"):
            scout_bound[0] = max(scout_bound[0], 2 * rec["cost_usd"])
        if rec.get("error"):
            limit_hit(rec.get("error"))
            print(f"[pq] ERROR {unit} {planner}: {str(rec.get('error'))[:300]} (not recorded)", flush=True)
            return
        row = {"layer": layer, "unit": unit, "tickets": members, "ids": [i for i, _ in items], "planner": planner,
               "commit": base["base_commit"], **rec}
        append("briefs.jsonl", row)
        bs = [parse_brief(x) for x in split_briefs(rec["text"])]
        print(f"[pq] L{layer} {unit:45} {planner:5} briefs={len(bs)} problems={[b['problems'] for b in bs]} "
              f"${rec.get('cost_usd') or 0:.3f} {rec.get('seconds')}s turns={rec.get('turns')}"
              f"{'' if rec.get('pages_arrived', True) else ' PAGES DROPPED'}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))


# --------------------------------------------------------------------------- judge
def run_judge(work: Path, ids: list[str] | None, cap: float, workers: int) -> None:
    from vbpe import claude_eval as CE
    tm = task_map(work)
    done = {(r["unit"], r["planner"]) for r in jl(OUT / "judge.jsonl")}
    rows = [r for r in jl(OUT / "briefs.jsonl") if r["layer"] == 1 and (r["unit"], r["planner"]) not in done
            and (not ids or r["unit"] in ids)]
    random.Random(SEED).shuffle(rows)
    print(f"[pq] {len(rows)} briefs to judge; spent ${spent():.2f} (cap ${cap})", flush=True)

    def one(r):
        if STOP.is_set():
            return
        t = tm[r["unit"]]
        fix = source_patch(t["patch"])
        trunc = ""
        if len(fix) > FIX_CHARS:
            fix, trunc = fix[:FIX_CHARS] + "\n[... truncated]", f"; truncated to its first {FIX_CHARS} characters"
        content = brief_content(split_briefs(r["text"])[0] if r["text"] else "")
        prompt = JUDGE_PROMPT.format(issue=t["problem_statement"], fix=fix, trunc=trunc, brief=content or "(empty)")
        bound = len(prompt) / 3.0 * PRICE_WRITE * 1.1 + 3000 * PRICE_OUT
        if not reserve(bound, cap):
            print(f"[pq] SKIP judge {r['unit']} {r['planner']}: cap", flush=True)
            return
        try:
            rec = CE.transcribe([], SYSTEM_JUDGE, prompt, model=MODEL, effort=JUDGE_EFFORT, timeout=600.0, max_retries=1)
        finally:
            release(bound)
        if not rec.get("cached"):
            book(f"judge {r['unit']} {r['planner']}", rec.get("cost_usd") or 0.0, error=rec.get("error"))
        if rec.get("error"):
            limit_hit(rec["error"])
            print(f"[pq] judge ERROR {r['unit']} {r['planner']}: {rec['error'][:200]}", flush=True)
            return
        j = parse_judge(rec["text"])
        append("judge.jsonl", {"unit": r["unit"], "planner": r["planner"], "judge": j, "text": rec["text"],
                               "cost_usd": rec.get("cost_usd"), "cached": rec.get("cached"),
                               "input_tokens": rec.get("input_tokens"), "output_tokens": rec.get("output_tokens"),
                               "fix_truncated": bool(trunc)})
        print(f"[pq] judge {r['unit']:45} {r['planner']:5} {j and j['score']} ${rec.get('cost_usd') or 0:.3f}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, rows))


# --------------------------------------------------------------------------- layer 3 workers
LEAK_EXTRA = r"|/pq/(ds|ctx|co|git|x|transcripts)\b|/home/user/visual-bpe|plan_quality|repo_atlas"
# (the run's own checkout and venv live under /pq/runs/<hash>/: not a leak)


def leak_hits(commands: list[str], pkg: str) -> list[str]:
    rx = re.compile(S().LEAK.pattern.replace("{pkg}", re.escape(pkg)) + LEAK_EXTRA, re.I)
    return [c[:200] for c in commands if rx.search(c)]


def apply_bytes(repo: Path, patch: str) -> bool:
    if not patch.strip():
        return True
    f = repo.parent / f"apply_{os.getpid()}_{threading.get_ident()}.diff"
    f.write_bytes(patch.encode("utf-8", "surrogateescape"))
    ok = subprocess.run(["git", "apply", "--whitespace=nowarn", str(f)], cwd=repo, capture_output=True).returncode == 0
    f.unlink(missing_ok=True)
    return ok


def worker_once(work: Path, t: dict, arm: str, brief_text: str, broken: set) -> dict:
    from vbpe import claude_eval as CE
    sw = S()
    tier, _ = WORK_ARMS[arm]
    model, effort = TIERS[tier]
    tag = hashlib.sha1(("w" + arm + t["instance_id"]).encode()).hexdigest()[:10]
    base = work / "runs" / tag
    repo = base / t["repo"].split("/")[1]
    venv = base / "venv"
    sw.fresh_repo(work, t, repo)
    env_info = sw.make_env(t, repo, venv)
    text = sw.PROMPT.format(repo=t["repo"], venv=str(venv), issue=t["problem_statement"]) + WORK_BRIEF.format(brief=brief_text)
    memory_check_and_clear(repo)
    cmd = [CE.CLAUDE_BIN, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", model, "--effort", effort, "--tools", "Bash,Read,Edit,Write", "--allowedTools",
           "Bash,Read,Edit,Write", "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
           "--no-session-persistence", "--max-turns", str(sw.MAX_TURNS)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=CE.build_message([], text), capture_output=True,
                           env={**sw.agent_env(CE.build_env()), **NO_MEMORY}, cwd=repo, timeout=sw.AGENT_TIMEOUT)
        stdout, rc = p.stdout.decode("utf-8", "replace"), p.returncode
    except subprocess.TimeoutExpired as e:
        stdout, rc = (e.stdout or b"").decode("utf-8", "replace"), -9
    secs = time.time() - t0
    mem = memory_check_and_clear(repo)
    tr = work / "transcripts" / "work" / t["instance_id"]
    tr.mkdir(parents=True, exist_ok=True)
    (tr / f"{arm}.jsonl").write_text(stdout)
    ev = [json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]
    res = next((e for e in reversed(ev) if e.get("type") == "result"), {})
    asst = [e for e in ev if e.get("type") == "assistant"]
    calls = [c for e in asst for c in e["message"]["content"] if c.get("type") == "tool_use"]
    cmds = [str((c.get("input") or {}).get("command", "")) for c in calls if c["name"] == "Bash"]
    outside = [json.dumps(c.get("input"))[:200] for c in calls if c["name"] in ("Read", "Edit", "Write")
               and not str((c.get("input") or {}).get("file_path", "")).startswith(str(repo))]
    tot = sw.stream_totals(ev)
    book(f"L3 {t['instance_id']} {arm}", tot["cost_usd"])     # booked before grading, so an error cannot lose it
    sw.sh(["git", "add", "-A"], cwd=repo)
    # bytes: a patch need not be UTF-8 (pyfakefs-1286 haiku); surrogateescape keeps every byte through the filter
    patch = subprocess.run(["git", "diff", "--cached", "--binary", "HEAD"], cwd=repo, capture_output=True).stdout \
        .decode("utf-8", "surrogateescape")
    graded = sw.exclude_files(patch, set(sw.patch_paths(t["test_patch"])))
    (tr / f"{arm}.patch").write_bytes(patch.encode("utf-8", "surrogateescape"))
    sw.reset(repo)
    applied = apply_bytes(repo, graded)
    if applied and sw.git_apply(repo, t["test_patch"]):
        log, _ = sw.run_tests(t, repo, venv)
        g = sw.grade_status(sw.parse_pytest_log(log), t["FAIL_TO_PASS"], t["PASS_TO_PASS"], broken)
    else:
        g = {"resolved": False, "note": "patch does not apply"}
    shutil.rmtree(base, ignore_errors=True)
    gold = sw.gold_files(t["patch"])
    names = [c["name"] for c in calls]
    return {"id": t["instance_id"], "arm": arm, "tier": tier, "model": model, "effort": effort,
            "resolved": bool(g.get("resolved")), "grade": g, "patch_applied": applied, "patch_files": sw.patch_paths(graded),
            "patch_lines": graded.count("\n"), "gold": gold, "gold_touched": sw.touched_share(graded, gold),
            "turns": tot["turns"], "cost_usd": tot["cost_usd"], "results": tot["results"], "subtype": res.get("subtype"),
            "is_error": res.get("is_error"), "rc": rc, "seconds": round(secs), "env_seconds": env_info["seconds"],
            "input_tokens": tot["input_tokens"], "output_tokens": tot["output_tokens"], "usage": tot["usage"],
            "brief_chars": len(brief_text), "tools": {k: names.count(k) for k in sorted(set(names))},
            "outside_repo_edits": outside[:10], "leak_suspects": leak_hits(cmds, t["repo"].split("/")[1]),
            "permission_denials": [str(x)[:200] for x in res.get("permission_denials") or []][:10], "memory": mem,
            "model_usage": {m: round(v.get("costUSD", 0.0), 4) for m, v in (res.get("modelUsage") or {}).items()},
            "final": (res.get("result") or "")[-300:]}


def layer1_brief(unit: str, planner: str) -> str | None:
    for r in jl(OUT / "briefs.jsonl"):
        if r["layer"] == 1 and r["unit"] == unit and r["planner"] == planner:
            bs = split_briefs(r["text"])
            return brief_content(bs[0]) if bs else None
    return None


def run_work(work: Path, cap: float, max_tickets: int | None, workers: int) -> None:
    tm = task_map(work)
    vrows = {v["id"]: v for v in jl(SWE_OUT / "validate.jsonl") if v.get("valid")}
    order = tickets()["layer3"]
    if max_tickets:
        order = order[:max_tickets]
    done = {(r["id"], r["arm"]) for r in jl(OUT / "work.jsonl")}
    per = {}
    for r in jl(OUT / "work.jsonl"):
        per[r["id"]] = per.get(r["id"], 0) + (r["cost_usd"] or 0)
    bound = [max([3.0] + [1.5 * v for v in per.values()])]
    print(f"[work] {len(order)} tickets; spent ${spent():.2f} (cap ${cap}); per-ticket bound ${bound[0]:.2f}", flush=True)
    for tid in order:
        todo = [a for a in WORK_ARMS if (tid, a) not in done]
        if not todo:
            continue
        if STOP.is_set():
            print("[work] STOP: usage limit", flush=True)
            break
        started = len(todo) < len(WORK_ARMS)
        if not started and spent() + bound[0] > cap:
            print(f"[work] STOP before {tid}: ${spent():.2f} spent + bound ${bound[0]:.2f} > cap ${cap}", flush=True)
            break
        briefs = {p: layer1_brief(tid, p) for p in ("atlas", "tree")}
        if not all(briefs.values()):
            print(f"[work] SKIP {tid}: missing layer-1 brief ({[p for p, b in briefs.items() if not b]})", flush=True)
            continue
        t = tm[tid]
        broken = set(vrows[tid].get("env_broken_p2p") or [])

        def one(arm):
            if STOP.is_set():
                return None
            for attempt in range(2):
                try:
                    row = worker_once(work, t, arm, briefs[WORK_ARMS[arm][1]], broken)
                except Exception as ex:  # noqa: BLE001
                    print(f"[work] {tid} {arm} ERROR {str(ex)[-300:]}", flush=True)
                    continue
                if USAGE_LIMIT.search(row["final"] or "") and (row["is_error"] or row["rc"]):
                    limit_hit(row["final"])
                    print(f"[work] {tid} {arm}: USAGE LIMIT, not recorded", flush=True)
                    return None
                if row["subtype"] is None and row["cost_usd"] == 0 and attempt == 0:
                    print(f"[work] {tid} {arm}: no result event (rc={row['rc']}); retry", flush=True)
                    continue
                append("work.jsonl", row)
                print(f"[work] {tid:45} {arm:12} resolved={row['resolved']!s:5} {row['grade'].get('f2p')} "
                      f"{row['grade'].get('p2p')} turns={row['turns']} ${row['cost_usd']:.2f} {row['seconds']}s "
                      f"leaks={len(row['leak_suspects'])} mem={row['memory']}", flush=True)
                return row
        with ThreadPoolExecutor(min(workers, len(todo))) as ex:
            list(ex.map(one, todo))
        per = {}
        for r in jl(OUT / "work.jsonl"):
            per[r["id"]] = per.get(r["id"], 0) + (r["cost_usd"] or 0)
        bound[0] = max([3.0] + [1.5 * v for v in per.values()])
        print(f"[work] spent ${spent():.2f}", flush=True)


def ticket_projection(x: dict) -> float:
    """Projected list price of one more layer-1 ticket (four planners + four judge calls) from the rows so far: the
    mean of tree, map, scout and judge calls; the atlas call as its measured cost per image token times this
    ticket's image tokens, plus the mean atlas output cost."""
    rows = [r for r in jl(OUT / "briefs.jsonl") if r["layer"] == 1 and r.get("cost_usd") and not r.get("cached")]
    sel = {t["id"]: t for t in tickets()["tickets"]}
    m = lambda p: statistics.mean(r["cost_usd"] for r in rows if r["planner"] == p)  # noqa: E731
    at = [r for r in rows if r["planner"] == "atlas"]
    per_tok = statistics.mean((r["cost_usd"] - (r.get("output_tokens") or 0) * PRICE_OUT)
                              / sel[r["unit"]]["image_tokens_formula"] for r in at)
    out = statistics.mean((r.get("output_tokens") or 0) * PRICE_OUT for r in at)
    jd = statistics.mean(r["cost_usd"] for r in jl(OUT / "judge.jsonl") if r.get("cost_usd"))
    return m("tree") + m("map") + m("scout") + per_tok * x["image_tokens_formula"] + out + 4 * jd


def extend(work: Path, cap: float, workers: int) -> None:
    """Layer 1 beyond the layer-3 tickets, in order, one whole ticket at a time (four planners, then its four judge
    calls), started only when spent + 1.2 x its projection stays under the cap."""
    for x in tickets()["tickets"]:
        have = {r["planner"] for r in jl(OUT / "briefs.jsonl") if r["layer"] == 1 and r["unit"] == x["id"]}
        judged = {r["planner"] for r in jl(OUT / "judge.jsonl") if r["unit"] == x["id"]}
        if have >= set(PLANNERS) and judged >= set(PLANNERS):
            continue
        if STOP.is_set():
            break
        proj = ticket_projection(x)
        if not have and spent() + 1.2 * proj > cap:
            print(f"[extend] STOP before {x['id']}: ${spent():.2f} + 1.2 x ${proj:.2f} > cap ${cap}", flush=True)
            break
        run_plans(work, 1, list(PLANNERS), [x["id"]], cap, workers)
        run_judge(work, [x["id"]], cap, workers)
        print(f"[extend] {x['id']} done (projected ${proj:.2f}); spent ${spent():.2f}", flush=True)


# --------------------------------------------------------------------------- estimate
def estimate(work: Path, out_tokens: int = 3000) -> dict:
    sel = tickets()
    est = {p: 0.0 for p in PLANNERS}
    measured = {}
    for r in jl(OUT / "briefs.jsonl"):
        if r["layer"] == 1 and r.get("cost_usd"):
            measured.setdefault(r["planner"], []).append(r["cost_usd"])
    for x in sel["tickets"]:
        est["tree"] += (x["tree_chars"] / 3.2 + 1500) * PRICE_WRITE + out_tokens * PRICE_OUT
        est["map"] += (x["map_chars"] / 3.2 + 1500) * PRICE_WRITE + out_tokens * PRICE_OUT
        est["atlas"] += (x["image_tokens_formula"] + 2500) * PRICE_WRITE + out_tokens * PRICE_OUT
        est["scout"] += statistics.mean(measured["scout"]) if measured.get("scout") else 0.4
    est["judge"] = statistics.mean(r["cost_usd"] for r in jl(OUT / "judge.jsonl") if r.get("cost_usd")) * 4 * len(sel["tickets"]) \
        if jl(OUT / "judge.jsonl") else 0.05 * 4 * len(sel["tickets"])
    est["layer2"] = sum((next(x for x in sel["tickets"] if x["id"] == g["commit_of"])["image_tokens_formula"]
                         + 4000 * len(g["tickets"])) * PRICE_WRITE + 2000 * len(g["tickets"]) * PRICE_OUT * 3
                        for g in sel["groups"] if g["usable"]) if sel["layer2_evaluable"] else 0.0
    est["measured_means"] = {k: round(statistics.mean(v), 4) for k, v in measured.items()}
    est["total_layers_1_2"] = sum(v for k, v in est.items() if isinstance(v, float))
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in est.items()}


# --------------------------------------------------------------------------- analysis
def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def scored_layer1(work: Path) -> list[dict]:
    sel = {x["id"]: x for x in tickets()["tickets"]}
    judge = {(r["unit"], r["planner"]): r["judge"] for r in jl(OUT / "judge.jsonl")}
    out = []
    for r in jl(OUT / "briefs.jsonl"):
        if r["layer"] != 1 or r["unit"] not in sel:
            continue
        x = sel[r["unit"]]
        bs = split_briefs(r["text"])
        b = parse_brief(bs[0] if bs else "")
        paths, words = repo_sets(work, r["unit"])
        out.append({"id": r["unit"], "planner": r["planner"], "brief": b, "files": file_scores(b, x["gold"], x["gold_all"]),
                    "funcs": func_score(b, x["funcs"]), "invented": invented(b, paths, words),
                    "judge": judge.get((r["unit"], r["planner"])), "cost_usd": r.get("cost_usd") or 0.0,
                    "seconds": r.get("seconds") or 0.0, "turns": r.get("turns"), "pages_arrived": r.get("pages_arrived", True),
                    "size_real": x["size"], "real_class": x["real_class"], "layer3": x["layer3"],
                    "outside_cwd": r.get("outside_cwd") or [], "memory": r.get("memory")})
    return out


def analyse(work: Path) -> str:
    sw = S()
    sel = tickets()
    L1 = scored_layer1(work)
    by = {(r["id"], r["planner"]): r for r in L1}
    ids = [x["id"] for x in sel["tickets"] if all((x["id"], p) in by for p in PLANNERS)]
    L = [f"## Layer 1: {len(ids)} tickets with all four briefs", "",
         "| planner | briefs | fix files named (mean) | named source files not in the fix | first file in the fix | "
         "fix functions named | invented paths + symbols (of all named) | format problems | judge score (0-6) | "
         "judge OUTCOME yes | mean cost | mean seconds | mean turns |", "|---|" + "---|" * 12]
    summ = {}
    for p in PLANNERS:
        rs = [by[(i, p)] for i in ids]
        if not rs:
            continue
        gs = [r["files"]["gold_share"] for r in rs if r["files"]["gold_share"] is not None]
        ws = sum(len(r["files"]["wrong"]) for r in rs) / max(1, sum(r["files"]["named_src"] for r in rs))
        fs = [r["funcs"]["funcs_share"] for r in rs if r["funcs"]["funcs_share"] is not None]
        inv = sum(r["invented"]["missing"] for r in rs) / max(1, sum(r["invented"]["named"] for r in rs))
        js = [r["judge"]["score"] for r in rs if r["judge"]]
        jo = [r["judge"]["outcome"] == "yes" for r in rs if r["judge"]]
        summ[p] = {"gold_share": statistics.mean(gs) if gs else None, "invented": inv,
                   "judge": statistics.mean(js) if js else None}
        L.append(f"| {p} | {len(rs)} | {_pct(summ[p]['gold_share'])} | {_pct(ws)} | "
                 f"{sum(r['files']['first_right'] for r in rs)} of {len(rs)} | {_pct(statistics.mean(fs) if fs else None)} | "
                 f"{sum(r['invented']['missing'] for r in rs)} of {sum(r['invented']['named'] for r in rs)} ({_pct(inv)}) | "
                 f"{sum(bool(r['brief']['problems']) for r in rs)} | {statistics.mean(js) if js else float('nan'):.2f} | "
                 f"{sum(jo)} of {len(jo)} | ${statistics.mean(r['cost_usd'] for r in rs):.3f} | "
                 f"{statistics.mean(r['seconds'] for r in rs):.0f} | {statistics.mean((r['turns'] or 1) for r in rs):.1f} |")
    # paired comparisons
    def paired(a, b, key):
        xs = [(key(by[(i, a)]), key(by[(i, b)])) for i in ids]
        xs = [(x, y) for x, y in xs if x is not None and y is not None]
        if not xs:
            return "n/a"
        d = [x - y for x, y in xs]
        s = sw.paired_sign([x for x, _ in xs], [y for _, y in xs])
        return (f"mean difference {100 * statistics.mean(d):+.1f} points over {len(xs)}; {a} higher on {s['b_lower']}, "
                f"lower on {s['a_lower']}, sign test p = {s['p']:.3f}")
    L += ["", "Paired (layer 1):",
          f"* fix files named, atlas vs tree: {paired('atlas', 'tree', lambda r: r['files']['gold_share'])}",
          f"* fix files named, scout vs atlas: {paired('scout', 'atlas', lambda r: r['files']['gold_share'])}",
          f"* fix files named, map vs tree: {paired('map', 'tree', lambda r: r['files']['gold_share'])}",
          f"* fix functions named, atlas vs tree: {paired('atlas', 'tree', lambda r: r['funcs']['funcs_share'])}",
          f"* judge score / 6, atlas vs tree: {paired('atlas', 'tree', lambda r: r['judge'] and r['judge']['score'] / 6)}",
          f"* judge score / 6, scout vs atlas: {paired('scout', 'atlas', lambda r: r['judge'] and r['judge']['score'] / 6)}"]
    inv_rows = [(r["id"], r["planner"], r["invented"]["paths_missing"], r["invented"]["symbols_missing"])
                for r in L1 if r["invented"]["missing"]]
    L += ["", "Invented paths and symbols (named, absent from the repository, not marked new):"]
    L += [f"* {i} {p}: paths {pm} symbols {sm}" for i, p, pm, sm in inv_rows] or ["* none"]
    # sizing
    L += ["", "### Sizing (layer 1)", "", "| planner | rank correlation SIZE vs real size (composite) | vs files | vs functions | "
          "vs lines | SIZE S / M / L | real class (design's own rule) M / L | SIZE = real class |", "|---|---|---|---|---|---|---|---|"]
    for p in PLANNERS:
        rs = [by[(i, p)] for i in ids if by[(i, p)]["brief"]["size"]]
        if len(rs) < 3:
            continue
        sz = [SIZE_RANK[r["brief"]["size"]] for r in rs]
        comp = composite_size([r["size_real"] for r in rs])
        c = lambda k: spearman(sz, [r["size_real"][k] for r in rs])  # noqa: E731
        f = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
        cnt = {s: sum(r["brief"]["size"] == s for r in rs) for s in "SML"}
        rc = {s: sum(r["real_class"] == s for r in rs) for s in "SML"}
        L.append(f"| {p} | {f(spearman(sz, comp))} | {f(c('files'))} | {f(c('functions'))} | {f(c('lines'))} | "
                 f"{cnt['S']} / {cnt['M']} / {cnt['L']} | {rc['M']} / {rc['L']} | "
                 f"{sum(r['brief']['size'] == r['real_class'] for r in rs)} of {len(rs)} |")
    ra = [by[(i, "atlas")] for i in ids if by[(i, "atlas")]["brief"]["size"]]
    if ra:
        L += ["", "Confusion, atlas planner (rows: SIZE; columns: real class by the fix's source files):", "",
              "| SIZE | real S | real M | real L |", "|---|---|---|---|"]
        for s in "SML":
            L.append(f"| {s} | " + " | ".join(str(sum(r['brief']['size'] == s and r['real_class'] == c for r in ra))
                                             for c in "SML") + " |")
    summ["atlas_size_rho"] = spearman([SIZE_RANK[r["brief"]["size"]] for r in ra],
                                      composite_size([r["size_real"] for r in ra])) if len(ra) >= 3 else None
    # layer 2
    L += ["", "## Layer 2: tickets that collide", ""]
    if not sel["layer2_evaluable"]:
        L.append(f"Not evaluable: {sum(g['usable'] for g in sel['groups'])} usable group(s) of three or more tickets "
                 f"({', '.join(g['repo'] + ' ' + str(len(g['tickets'])) for g in sel['groups']) or 'none'}).")
    else:
        gold = {x["id"]: x["gold_all"] for x in sel["tickets"]}
        L += ["| group | repository | tickets | true colliding pairs | " + " | ".join(f"{p}: found / wrong" for p in GROUP_PLANNERS)
              + " |", "|---|---|---|---|" + "---|" * len(GROUP_PLANNERS)]
        rows2 = {(r["unit"], r["planner"]): r for r in jl(OUT / "briefs.jsonl") if r["layer"] == 2}
        tot = {p: {"found": 0, "wrong": 0, "true": 0} for p in GROUP_PLANNERS}
        for g in sel["groups"]:
            if not g["usable"]:
                continue
            idmap = {f"T{k + 1}": tid for k, tid in enumerate(g["tickets"])}
            tp = true_pairs({k: gold[v] for k, v in idmap.items()})
            cells = []
            for p in GROUP_PLANNERS:
                r = rows2.get((g["gid"], p))
                if not r:
                    cells.append("")
                    continue
                bs = [parse_brief(x) for x in split_briefs(r["text"])]
                col = {b["ticket"]: b["collides"] for b in bs if b["ticket"] in idmap}
                for k in idmap:
                    col.setdefault(k, [])
                sc = collision_score(flagged_pairs(col), tp)
                for k in ("found", "wrong", "true"):
                    tot[p][k] += sc[k]
                cells.append(f"{sc['found']} / {sc['wrong']}")
            L.append(f"| {g['gid']} | {g['repo']} | {len(g['tickets'])} | {len(tp)} | " + " | ".join(cells) + " |")
        L.append("| total | | | " + str(tot["atlas"]["true"]) + " | "
                 + " | ".join(f"{tot[p]['found']} / {tot[p]['wrong']}" for p in GROUP_PLANNERS) + " |")
        summ["collide"] = tot
    # layer 3
    L += ["", "## Layer 3: workers", ""]
    W = {(r["id"], r["arm"]): r for r in jl(OUT / "work.jsonl")}
    plain = {r["id"]: r for r in sw.jl(SWE_OUT / "runs.jsonl") if r["arm"] == "plain"}
    l3 = [i for i in sel["layer3"] if all((i, a) in W for a in WORK_ARMS)]
    L.append(f"Tickets with all four worker runs: {len(l3)} of {len(sel['layer3'])}.")
    if l3:
        L += ["", "| arm | tickets | resolved | median turns | mean cost | median seconds | gold files touched |",
              "|---|---|---|---|---|---|---|"]
        arms = {"opus, no brief (repo_atlas_swe plain)": {i: plain[i] for i in l3}}
        arms.update({f"{a}": {i: W[(i, a)] for i in l3} for a in WORK_ARMS})
        for name, rs in arms.items():
            v = list(rs.values())
            L.append(f"| {name} | {len(v)} | {sum(r['resolved'] for r in v)} | {statistics.median(r['turns'] or 0 for r in v):g} | "
                     f"${statistics.mean(r['cost_usd'] for r in v):.2f} | {statistics.median(r['seconds'] for r in v):.0f} | "
                     f"{statistics.mean(r['gold_touched'] for r in v):.2f} |")
        g = lambda a, k: [W[(i, a)][k] for i in l3]  # noqa: E731
        pl = lambda k: [plain[i][k] for i in l3]  # noqa: E731
        m1 = sw.mcnemar(g("opus_atlas", "resolved"), pl("resolved"))
        m2 = sw.mcnemar(g("opus_atlas", "resolved"), g("opus_tree", "resolved"))
        s1 = sw.paired_sign(g("opus_atlas", "turns"), pl("turns"))
        s2 = sw.paired_sign(g("opus_atlas", "turns"), g("opus_tree", "turns"))
        c1 = sw.paired_sign(g("opus_atlas", "cost_usd"), pl("cost_usd"))
        L += ["", "Paired (layer 3):",
              f"* resolved, opus+atlas brief vs no brief: brief only {m1['only_a']}, no brief only {m1['only_b']}, p = {m1['p']:.3f}",
              f"* resolved, opus+atlas brief vs opus+tree brief: atlas only {m2['only_a']}, tree only {m2['only_b']}, p = {m2['p']:.3f}",
              f"* turns, opus+atlas brief vs no brief: median {statistics.median(g('opus_atlas', 'turns')):g} vs "
              f"{statistics.median(pl('turns')):g}; brief lower on {s1['a_lower']} of {s1['n']} untied, p = {s1['p']:.3f}",
              f"* turns, opus+atlas brief vs opus+tree brief: atlas lower on {s2['a_lower']} of {s2['n']} untied, p = {s2['p']:.3f}",
              f"* cost, opus+atlas brief vs no brief: brief lower on {c1['a_lower']} of {c1['n']}, p = {c1['p']:.3f}"]
        summ["l3"] = {"n": len(l3), "atlas_res": sum(g("opus_atlas", "resolved")), "plain_res": sum(pl("resolved")),
                      "atlas_turns": statistics.median(g("opus_atlas", "turns")), "plain_turns": statistics.median(pl("turns")),
                      "turns_test": s1}
        # right-sizing
        rows = []
        L += ["", "### Right-sizing (atlas planner's WORKER against the measured tier)", "",
              "| ticket | planner SIZE | planner WORKER | haiku low | sonnet medium | opus high | measured tier | verdict |",
              "|---|---|---|---|---|---|---|---|"]
        for i in l3:
            b = by[(i, "atlas")]["brief"]
            planned = b["worker"]["model"] if b["worker"] else None
            runs = {WORK_ARMS[a][0]: W[(i, a)] for a in ("haiku_atlas", "sonnet_atlas", "opus_atlas")}
            mt = measured_tier({k: v["resolved"] for k, v in runs.items()})
            v = tier_verdict(planned, mt)
            rows.append({"planned": planned, "runs": runs, "verdict": v})
            cell = lambda r: f"{'yes' if r['resolved'] else 'no'} ${r['cost_usd']:.2f}"  # noqa: E731
            L.append(f"| {i} | {b['size']} | {b['worker'] and b['worker']['model'] + ' / ' + b['worker']['effort']} | "
                     f"{cell(runs['haiku'])} | {cell(runs['sonnet'])} | {cell(runs['opus'])} | {mt} | {v} |")
        vc = {k: sum(r["verdict"] == k for r in rows) for k in ("right", "too small", "too large", "unresolvable", "no choice")}
        L += ["", "Verdicts: " + ", ".join(f"{k} {v}" for k, v in vc.items())]
        pol = policies(rows)
        L += ["", "| policy | tickets | resolved | total worker cost |", "|---|---|---|---|"]
        L += [f"| {k} | {v['tickets']} | {v['resolved']} | ${v['cost_usd']:.2f} |" for k, v in pol.items()]
        summ["policies"] = pol
        summ["verdicts"] = vc
        for r in W.values():                                 # re-apply the current pattern to stored commands
            r["leak_suspects"] = leak_hits(r.get("leak_suspects") or [], r["id"].split("__")[1].rsplit("-", 1)[0])
        leaks = [(r["id"], r["arm"], r["leak_suspects"]) for r in W.values() if r.get("leak_suspects")]
        L += ["", f"Leak suspects in worker runs: {len(leaks)}"] + [f"* {x}" for x in leaks]
        mem = [(r["id"], r["arm"], r["memory"]) for r in W.values() if (r.get("memory") or {}).get("memory_files")]
        L += [f"Worker runs that wrote auto-memory files: {len(mem)}"] + [f"* {x}" for x in mem]
    # predictions
    L += ["", "## Predictions", ""]
    a, t_, s = summ.get("atlas", {}), summ.get("tree", {}), summ.get("scout", {})
    def pv(cond):
        return "not evaluable" if cond is None else ("PASS" if cond else "FAIL")
    p1 = None if a.get("gold_share") is None or t_.get("gold_share") is None else a["gold_share"] - t_["gold_share"] >= 0.05
    p2 = None if a.get("invented") is None or s.get("invented") is None else (a["invented"] <= 0.02 and a["invented"] - s["invented"] <= 0.02)
    p3 = None if a.get("gold_share") is None or s.get("gold_share") is None else abs(s["gold_share"] - a["gold_share"]) <= 0.05 or s["gold_share"] > a["gold_share"]
    col = summ.get("collide")
    p4 = None if not col else (col["atlas"]["found"] > col["tree"]["found"] and col["atlas"]["wrong"] <= col["tree"]["wrong"])
    p5 = None if summ.get("atlas_size_rho") is None else summ["atlas_size_rho"] >= 0.3
    pol = summ.get("policies")
    p6 = None if not pol else (pol["planner's choice"]["resolved"] >= pol["always opus high"]["resolved"] - 1
                               and pol["planner's choice"]["cost_usd"] <= 0.7 * pol["always opus high"]["cost_usd"])
    l3s = summ.get("l3")
    p7 = None if not l3s else (l3s["atlas_turns"] < l3s["plain_turns"] and abs(l3s["atlas_res"] - l3s["plain_res"]) <= 2)
    for k, v, d in ((1, p1, f"atlas {_pct(a.get('gold_share'))} vs tree {_pct(t_.get('gold_share'))} fix files named"),
                    (2, p2, f"invented: atlas {_pct(a.get('invented'))}, scout {_pct(s.get('invented'))}"),
                    (3, p3, f"scout {_pct(s.get('gold_share'))} vs atlas {_pct(a.get('gold_share'))}"),
                    (4, p4, f"collisions {col}"), (5, p5, f"atlas SIZE rank correlation {summ.get('atlas_size_rho')}"),
                    (6, p6, f"policies {pol}"), (7, p7, f"layer 3 {l3s}")):
        L.append(f"* P{k}: {pv(v)} ({d})")
    # cost
    sp = jl(OUT / "spend.jsonl")
    cat = {}
    for r in sp:
        w = r["what"]
        k = ("screen" if w.endswith("screen") else "smoke" if w.startswith("smoke") else "judge" if w.startswith("judge")
             else "L3 " + w.split()[2] if w.startswith("L3") else f"L{w[1]} " + w.split()[-1] if w.startswith("L") else w)
        cat[k] = cat.get(k, 0) + r["cost_usd"]
    L += ["", "## Spend (list price)", "", "| what | $ |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in sorted(cat.items())]
    L.append(f"| **total** | **{spent():.2f}** |")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["select", "prepare", "estimate", "plan", "judge", "work", "extend", "analyse"])
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cap", type=float, default=45.0)
    ap.add_argument("--layer", type=int, default=1, choices=[1, 2])
    ap.add_argument("--planners", nargs="+", default=list(PLANNERS), choices=PLANNERS)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--max-tickets", type=int)
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    w = min(a.workers, 2)
    if a.cmd == "select":
        select(a.work, w)
    elif a.cmd == "prepare":
        sel = prepare(a.work)
        print(f"{len(sel['tickets'])} tickets; groups: {[(g['gid'], g['repo'], len(g['tickets']), g['usable']) for g in sel['groups']]}"
              f"; layer 2 evaluable: {sel['layer2_evaluable']}")
        print(json.dumps(estimate(a.work), indent=1))
    elif a.cmd == "estimate":
        print(json.dumps(estimate(a.work), indent=1))
    elif a.cmd == "plan":
        run_plans(a.work, a.layer, a.planners, a.ids, a.cap, w)
    elif a.cmd == "judge":
        run_judge(a.work, a.ids, a.cap, w)
    elif a.cmd == "work":
        run_work(a.work, a.cap, a.max_tickets, w)
    elif a.cmd == "extend":
        extend(a.work, a.cap, w)
    else:
        txt = analyse(a.work)
        (OUT / "summary.md").write_text(txt + "\n")
        print(txt)
    print(f"[pq] spent ${spent():.2f} list price", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
