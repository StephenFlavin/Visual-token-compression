"""Repo atlas on hard coding tasks, again (REPO_ATLAS_PLAN.md Q1): plain vs atlas, two fresh runs per task and arm,
auto-memory off, package index blocked for every run.

Same 16 SWE-rebench tasks as the first test (``results/repo_atlas_swe/``, smolvm-172 left out there for a leak), same
prompt, tools, model, effort, atlas pages and grading: everything comes from ``experiments/atlas/repo_atlas_swe.py``
(imported, not edited). What differs: ``CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`` in the child env, each run's project
store under ``~/.claude/projects/`` checked and deleted before and after, the package index blocked from the first
run, two independent runs per task and arm in a shuffled order (all first runs before any second run), each run in
its own neutral directory (a hash of task, arm and run number).

    python3 experiments/atlas/repo_atlas_swe_rerun.py check                  # memory check of the first test's transcripts
    python3 experiments/atlas/repo_atlas_swe_rerun.py validate [--workers 2]  # rebuild + validate the 16 environments
    python3 experiments/atlas/repo_atlas_swe_rerun.py order                   # print the shuffled run order
    python3 experiments/atlas/repo_atlas_swe_rerun.py run --cap 40 [--max-runs N] [--workers 2]
    python3 experiments/atlas/repo_atlas_swe_rerun.py analyse

Rows: results/repo_atlas_swe_rerun/{validate,runs,spend}.jsonl. Work dir (checkouts, venvs, transcripts) in the
scratchpad; it links the first test's atlas builds (``ra_swe/ctx``) and today's git stores (``pq/git``).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "atlas")]

import repo_atlas_swe as sw  # noqa: E402

OUT = ROOT / "results" / "repo_atlas_swe_rerun"
FIRST = ROOT / "results" / "repo_atlas_swe"                 # read only
SCRATCH = Path("/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad")
DEFAULT_WORK = Path(os.environ.get("RA_SWE2_WORK", str(SCRATCH / "swe_rerun")))
ARMS = ("plain", "atlas")
REPS = (1, 2)
SEED = 20261112
NO_MEMORY = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
BOUND_FLOOR = {"plain": 1.5, "atlas": 2.5}                 # reserved before a run until measured (first test max:
#                                                            plain $0.77, atlas $1.63)


# --------------------------------------------------------------------------- pure functions (tested)
def memory_slug(cwd: str) -> str:
    """Claude Code's per-project directory name under ~/.claude/projects (as experiments/atlas/plan_quality.py)."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def run_dir_tag(iid: str, arm: str, rep: int) -> str:
    """Neutral directory name: neither the arm nor the task appears in the run's paths."""
    return hashlib.sha1(f"rerun|{iid}|{arm}|{rep}".encode()).hexdigest()[:10]


def run_order(ids: list[str], seed: int = SEED) -> list[tuple[str, str, int]]:
    """All (task, arm) first runs in a shuffled order, then all second runs in another shuffled order, so a test cut
    short by the cap is still one full replication."""
    rng = random.Random(seed)
    out = []
    for rep in REPS:
        jobs = [(i, a, rep) for i in ids for a in ARMS]
        rng.shuffle(jobs)
        out += jobs
    return out


def smoke_first(order: list[tuple[str, str, int]]) -> list[tuple[str, str, int]]:
    """The first plain and the first atlas job of the order move to the front (the smoke runs, scored like the rest)."""
    firsts = []
    for a in ARMS:
        j = next(x for x in order if x[1] == a)
        firsts.append(j)
    return firsts + [x for x in order if x not in firsts]


LEAK_EXTRA = (r"|/ra_swe\b|/pq/|/swe_rerun/(ds|ctx|git|co|validate|transcripts)\b|/home/user/visual-bpe|"
              r"repo_atlas|plan_quality|\.claude/projects|MEMORY\.md|/memory/")
OUTSIDE = re.compile(r"/tmp/claude-0|scratchpad|/root/\.claude")


def own_prefixes(base: Path, repo: Path) -> list[str]:
    """The run's own directory and the CLI's own per-run temp dir (background-task output files live in
    /tmp/claude-0/<slug of the cwd>/<session>/tasks/)."""
    return [str(base), "/tmp/claude-0/" + memory_slug(str(repo))]


def leak_hits(commands: list[str], pkg: str, own) -> list[str]:
    """The first test's pattern (a release of the package, pip download/index, pypi.org, git fetch/clone, curl/wget,
    github, the dataset) plus any path into the scratch directories, the repository of this harness or a memory
    store. The run's own directory (``own``: checkout + venv) is removed from each command first, so its venv path
    (which sits in the scratchpad) is not a hit."""
    rx = re.compile(sw.LEAK.pattern.replace("{pkg}", re.escape(pkg)) + LEAK_EXTRA, re.I)
    out = []
    for c in commands:
        s = c
        for o in ([own] if isinstance(own, str) else own or []):
            s = s.replace(o, "<RUN>")
        if rx.search(s) or OUTSIDE.search(s):
            out.append(c[:200])
    return out


def per_task(rows: list[dict], ids: list[str]) -> dict:
    """{(id, arm): [resolved run 1, resolved run 2]} for tasks with both runs of the arm."""
    by = {}
    for r in rows:
        by.setdefault((r["id"], r["arm"]), {})[r["rep"]] = r
    return {(i, a): [bool(by[(i, a)][k]["resolved"]) for k in REPS] for i in ids for a in ARMS
            if (i, a) in by and all(k in by[(i, a)] for k in REPS)}


def bootstrap_diff(a: list[int], b: list[int], n: int = 10000, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval over tasks for sum(a) - sum(b) (resolved task-runs per task, paired)."""
    rng = random.Random(seed)
    k = len(a)
    if not k:
        return (0.0, 0.0)
    ds = sorted(sum(a[j] - b[j] for j in (rng.randrange(k) for _ in range(k))) for _ in range(n))
    return (ds[int(0.025 * n)], ds[int(0.975 * n) - 1])


# --------------------------------------------------------------------------- io
_alock = threading.Lock()
_rlock = threading.Lock()
_reserved = [0.0]
STOP = threading.Event()


def jl(path: Path) -> list[dict]:
    return sw.jl(path)


def append(name: str, row: dict) -> None:
    with _alock:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / name, "a") as f:
            f.write(json.dumps(row) + "\n")


def spent() -> float:
    return sum(r.get("cost_usd") or 0 for r in jl(OUT / "spend.jsonl"))


def reserve(cost: float, cap: float) -> bool:
    with _rlock:
        if spent() + _reserved[0] + cost > cap:
            return False
        _reserved[0] += cost
        return True


def release(cost: float) -> None:
    with _rlock:
        _reserved[0] -= cost


def bound(arm: str) -> float:
    cs = [r["cost_usd"] for r in jl(OUT / "runs.jsonl") if r["arm"] == arm]
    first = [r["cost_usd"] for r in jl(FIRST / "runs.jsonl") if r["arm"] == arm]
    return max([BOUND_FLOOR[arm], 1.5 * max(first or [0])] + [1.5 * c for c in cs])


def task_ids() -> list[str]:
    """The first test's 16 compared tasks (accepted, minus contaminated), its order (newest first)."""
    bad = json.loads((FIRST / "contaminated.json").read_text())
    rows = [r for r in jl(FIRST / "funnel.jsonl") if r.get("accepted")]
    order = [r["id"] for r in sorted(rows, key=lambda r: (r["created_at"], r["id"]), reverse=True)]
    return [i for i in order if i not in bad]


def valid_ids() -> list[str]:
    v = {r["id"]: r for r in jl(OUT / "validate.jsonl")}
    return [i for i in task_ids() if v.get(i, {}).get("valid")]


def setup_work(work: Path) -> None:
    """Link the first test's atlas builds and dataset, and today's git stores (no re-download)."""
    work.mkdir(parents=True, exist_ok=True)
    for name, src in (("ctx", SCRATCH / "ra_swe" / "ctx"), ("ds", SCRATCH / "ra_swe" / "ds"),
                      ("git", SCRATCH / "pq" / "git")):
        d = work / name
        if not d.exists() and src.exists():
            d.symlink_to(src)


def memory_check_and_clear(cwd: Path) -> dict:
    """Auto-memory is off (CLAUDE_CODE_DISABLE_AUTO_MEMORY=1); record whether the run's project store exists and
    holds memory files, then delete it."""
    p = Path.home() / ".claude" / "projects" / memory_slug(str(cwd))
    mem = p / "memory"
    files = sorted(str(x.relative_to(p)) for x in p.rglob("*") if x.is_file()) if p.is_dir() else []
    info = {"project_dir": p.exists(), "memory_files": sorted(x.name for x in mem.iterdir()) if mem.is_dir() else [],
            "files": files[:20]}
    shutil.rmtree(p, ignore_errors=True)
    return info


# --------------------------------------------------------------------------- first test: memory check (free)
def check_first(transcripts: Path) -> dict:
    """Search the first test's transcripts for any tool call or tool result touching a memory store or ~/.claude,
    and list each run's auto-memory path from its init event and what that store holds now."""
    out = {"runs": 0, "tool_hits": [], "stores": {}, "missing": not transcripts.exists()}
    rx = re.compile(r"\.claude/|MEMORY\.md|/memory/", re.I)
    for f in sorted(transcripts.glob("*/*.jsonl")):
        out["runs"] += 1
        key = f"{f.parent.name} {f.stem}"
        for line in f.read_text().splitlines():
            if not line.startswith("{"):
                continue
            e = json.loads(line)
            if e.get("type") == "system" and e.get("subtype") == "init":
                mp = (e.get("memory_paths") or {}).get("auto")
                if mp:
                    p = Path(mp)
                    out["stores"][key] = {"auto": mp, "exists": p.exists(),
                                          "files": sorted(x.name for x in p.iterdir()) if p.is_dir() else []}
            if e.get("type") == "assistant":
                for c in e["message"]["content"]:
                    if c.get("type") == "tool_use" and rx.search(json.dumps(c.get("input"))):
                        out["tool_hits"].append({"run": key, "tool": c["name"], "input": json.dumps(c.get("input"))[:300]})
            if e.get("type") == "user":
                for c in (e.get("message") or {}).get("content") or []:
                    if isinstance(c, dict) and c.get("type") == "tool_result" and rx.search(json.dumps(c.get("content"))):
                        out["tool_hits"].append({"run": key, "tool": "result", "input": json.dumps(c.get("content"))[:300]})
    paths = [v["auto"] for v in out["stores"].values()]
    out["distinct_store_paths"] = len(set(paths))
    out["stores_with_files"] = {k: v for k, v in out["stores"].items() if v["files"]}
    return out


# --------------------------------------------------------------------------- validation
def validate(work: Path, workers: int) -> None:
    tm = sw.task_map(work)
    done = {r["id"] for r in jl(OUT / "validate.jsonl")}
    todo = [i for i in task_ids() if i not in done]
    first = {r["id"]: r for r in jl(FIRST / "validate.jsonl") if r.get("valid")}

    def one(i):
        v = sw.validate(work, tm[i])
        old = set(first[i].get("env_broken_p2p") or [])
        new = set(v.get("env_broken_p2p") or [])
        v["first_env_broken_p2p"] = sorted(old)
        v["env_broken_same_as_first"] = old == new
        append("validate.jsonl", v)
        print(f"[val] {i:45} valid={v.get('valid')} {v.get('why') or ''} install={v.get('install_s')}s "
              f"tests={v.get('test_s')}s broken={len(new)} (first {len(old)})", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


# --------------------------------------------------------------------------- agent runs
def apply_bytes(repo: Path, patch: str) -> bool:
    if not patch.strip():
        return True
    f = repo.parent / f"apply_{os.getpid()}_{threading.get_ident()}.diff"
    f.write_bytes(patch.encode("utf-8", "surrogateescape"))
    ok = subprocess.run(["git", "apply", "--whitespace=nowarn", str(f)], cwd=repo, capture_output=True).returncode == 0
    f.unlink(missing_ok=True)
    return ok


def run_one(work: Path, t: dict, arm: str, rep: int, broken: set) -> dict:
    """sw.run_one with: auto-memory off and its store checked before and after, a neutral directory per run, the
    spend booked before grading, and the patch handled as bytes. Prompt, images, tools and flags are sw's."""
    from vbpe import claude_eval as CE
    iid = t["instance_id"]
    meta = sw.contexts(work, t)
    base = work / "runs" / run_dir_tag(iid, arm, rep)
    repo = base / t["repo"].split("/")[1]
    venv = base / "venv"
    if base.exists():
        shutil.rmtree(base)
    sw.fresh_repo(work, t, repo)
    env_info = sw.make_env(t, repo, venv)
    ctx = work / "ctx" / iid
    text = sw.first_message(arm, t["repo"], str(venv), t["problem_statement"],
                            atlas_index=(Path(meta["atlas_dir"]) / "index.txt").read_text() if arm == "atlas" else "",
                            n_files=meta["n_files"])
    pngs = [(Path(meta["atlas_dir"]) / f"page_{i + 1:03d}.png").read_bytes() for i in range(meta["pages"])] \
        if arm == "atlas" else []
    stdin = CE.build_message(pngs, text)
    cmd = [CE.CLAUDE_BIN, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", sw.MODEL, "--effort", sw.EFFORT, "--tools", "Bash,Read,Edit,Write", "--allowedTools",
           "Bash,Read,Edit,Write", "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
           "--no-session-persistence", "--max-turns", str(sw.MAX_TURNS)]
    mem_before = memory_check_and_clear(repo)
    env = {**sw.agent_env(CE.build_env()), **NO_MEMORY}
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, env=env, cwd=repo, timeout=sw.AGENT_TIMEOUT)
        stdout, rc = p.stdout.decode("utf-8", "replace"), p.returncode
    except subprocess.TimeoutExpired as e:
        stdout, rc = (e.stdout or b"").decode("utf-8", "replace"), -9
    secs = time.time() - t0
    mem_after = memory_check_and_clear(repo)
    tr = work / "transcripts" / iid
    tr.mkdir(parents=True, exist_ok=True)
    (tr / f"{arm}_r{rep}.jsonl").write_text(stdout)
    ev = [json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]
    res = next((e for e in reversed(ev) if e.get("type") == "result"), {})
    init = next((e for e in ev if e.get("type") == "system" and e.get("subtype") == "init"), {})
    asst = [e for e in ev if e.get("type") == "assistant"]
    first_u = (asst[0]["message"].get("usage") or {}) if asst else {}
    first_in = sum(first_u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    calls = [c for e in asst for c in e["message"]["content"] if c.get("type") == "tool_use"]
    cmds = [str((c.get("input") or {}).get("command", "")) for c in calls if c["name"] == "Bash"]
    paths = [str((c.get("input") or {}).get("file_path", "")) for c in calls if c["name"] in ("Read", "Edit", "Write")]
    outside = [x for x in paths if not x.startswith(str(base))]
    tot = sw.stream_totals(ev)
    append("spend.jsonl", {"what": f"{iid} {arm} r{rep}", "cost_usd": tot["cost_usd"],
                           "input_tokens": tot["input_tokens"], "output_tokens": tot["output_tokens"],
                           "t": time.strftime("%Y-%m-%dT%H:%M:%S")})       # booked before grading
    sw.sh(["git", "add", "-A"], cwd=repo)
    patch = subprocess.run(["git", "diff", "--cached", "--binary", "HEAD"], cwd=repo, capture_output=True).stdout \
        .decode("utf-8", "surrogateescape")
    graded = sw.exclude_files(patch, set(sw.patch_paths(t["test_patch"])))
    (tr / f"{arm}_r{rep}.patch").write_bytes(patch.encode("utf-8", "surrogateescape"))
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
    pkg = t["repo"].split("/")[1]
    return {"id": iid, "arm": arm, "rep": rep, "model": sw.MODEL, "effort": sw.EFFORT,
            "resolved": bool(g.get("resolved")), "grade": g, "patch_applied": applied,
            "patch_files": sw.patch_paths(graded), "patch_lines": graded.count("\n"), "gold": gold,
            "gold_touched": sw.touched_share(graded, gold), "turns": tot["turns"], "cost_usd": tot["cost_usd"],
            "results": tot["results"], "subtype": res.get("subtype"), "is_error": res.get("is_error"), "rc": rc,
            "seconds": round(secs), "env_seconds": env_info["seconds"], "input_tokens": tot["input_tokens"],
            "output_tokens": tot["output_tokens"], "usage": tot["usage"], "first_turn_input": first_in,
            "image_tokens_formula": meta["image_tokens_formula"] if arm == "atlas" else 0,
            "pages_arrived": arm != "atlas" or first_in >= sw.PAGE_ARRIVAL * meta["image_tokens_formula"],
            "pages": meta["pages"] if arm == "atlas" else 0, "tools": {k: names.count(k) for k in sorted(set(names))},
            "outside_paths": outside[:10], "leak_suspects": leak_hits(cmds + outside, pkg, own_prefixes(base, repo)),
            "memory_before": mem_before, "memory_after": mem_after,
            "memory_path_init": (init.get("memory_paths") or {}).get("auto"),
            "cli_version": init.get("claude_code_version"),
            "permission_denials": [str(x)[:200] for x in res.get("permission_denials") or []][:10],
            "model_usage": {m: round(v.get("costUSD", 0.0), 4) for m, v in (res.get("modelUsage") or {}).items()},
            "final": (res.get("result") or "")[-300:], "dir_tag": run_dir_tag(iid, arm, rep), "repo_dir": repo.name}


def run(work: Path, cap: float, max_runs: int | None, workers: int) -> None:
    tm = sw.task_map(work)
    vrows = {v["id"]: v for v in jl(OUT / "validate.jsonl") if v.get("valid")}
    order = smoke_first(run_order(task_ids()))
    order = [j for j in order if j[0] in vrows]
    done = {(r["id"], r["arm"], r["rep"]) for r in jl(OUT / "runs.jsonl")}
    todo = [j for j in order if j not in done]
    if max_runs is not None:
        todo = todo[:max_runs]
    print(f"[run] {len(order)} jobs, {len(done)} done, {len(todo)} to run now; spent ${spent():.2f} (cap ${cap})",
          flush=True)
    rep1_left = {j for j in order if j[2] == 1 and j not in done}
    lock = threading.Lock()

    def one(job):
        iid, arm, rep = job
        if STOP.is_set():
            return
        if rep == 2:                                   # every first run is finished before any second run
            while True:
                with lock:
                    if not rep1_left:
                        break
                if STOP.is_set():
                    return
                time.sleep(10)
        b = bound(arm)
        if not reserve(b, cap):
            print(f"[run] SKIP {iid} {arm} r{rep}: ${spent():.2f} spent + reserved + bound ${b:.2f} > cap ${cap}",
                  flush=True)
            STOP.set()
            return
        try:
            for attempt in range(2):
                try:
                    row = run_one(work, tm[iid], arm, rep, set(vrows[iid].get("env_broken_p2p") or []))
                except Exception as ex:  # noqa: BLE001
                    print(f"[run] {iid} {arm} r{rep} ERROR {str(ex)[-300:]}", flush=True)
                    continue
                if sw.USAGE_LIMIT.search(row["final"] or "") and (row["is_error"] or row["rc"]):
                    STOP.set()
                    append("spend.jsonl", {"what": f"{iid} {arm} r{rep}: usage-limit stop, not recorded", "cost_usd": 0})
                    print(f"[run] {iid} {arm} r{rep}: USAGE LIMIT, not recorded: {row['final'][-200:]}", flush=True)
                    return
                if row["subtype"] is None and row["cost_usd"] == 0 and attempt == 0:
                    print(f"[run] {iid} {arm} r{rep}: no result event (rc={row['rc']}); retry", flush=True)
                    continue
                append("runs.jsonl", row)
                print(f"[run] {iid:45} {arm:5} r{rep} resolved={row['resolved']!s:5} {row['grade'].get('f2p')} "
                      f"{row['grade'].get('p2p')} turns={row['turns']} ${row['cost_usd']:.2f} {row['seconds']}s "
                      f"first_in={row['first_turn_input']:,}{'' if row['pages_arrived'] else ' PAGES DROPPED'} "
                      f"leaks={len(row['leak_suspects'])} mem={row['memory_before']['memory_files']}/"
                      f"{row['memory_after']['memory_files']} spent=${spent():.2f}", flush=True)
                return
        finally:
            release(b)
            with lock:
                rep1_left.discard(job)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"[run] finished; spent ${spent():.2f}{' (STOPPED)' if STOP.is_set() else ''}", flush=True)


# --------------------------------------------------------------------------- analysis
def analyse() -> str:
    rows = jl(OUT / "runs.jsonl")
    ids = valid_ids()
    pt = per_task(rows, ids)
    full = [i for i in ids if all((i, a) in pt for a in ARMS)]
    first = {(r["id"], r["arm"]): r for r in jl(FIRST / "runs.jsonl")}
    L = [f"Tasks validated: {len(ids)} of {len(task_ids())}; with both runs of both arms: {len(full)}; "
         f"rows: {len(rows)}", ""]
    L += ["| arm | task-runs resolved | tasks resolved in both / one / neither | median turns (runs) | "
          "mean cost per run | median cost per run | median seconds | median input tokens | pages dropped | "
          "first test (1 run): resolved of 16, median turns, mean cost |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a and r["id"] in full]
        if not rs:
            continue
        k = [sum(pt[(i, a)]) for i in full]
        fr = [first[(i, a)] for i in full if (i, a) in first]
        med = lambda key: statistics.median(r[key] or 0 for r in rs)  # noqa: E731
        L.append(f"| {a} | {sum(r['resolved'] for r in rs)} of {len(rs)} | {k.count(2)} / {k.count(1)} / {k.count(0)} | "
                 f"{med('turns'):g} | ${statistics.mean(r['cost_usd'] for r in rs):.2f} | ${med('cost_usd'):.2f} | "
                 f"{med('seconds'):.0f} | {med('input_tokens'):,.0f} | {sum(not r['pages_arrived'] for r in rs)} | "
                 f"{sum(r['resolved'] for r in fr)} of {len(fr)}, {statistics.median(r['turns'] for r in fr):g}, "
                 f"${statistics.mean(r['cost_usd'] for r in fr):.2f} |")
    by = {(r["id"], r["arm"], r["rep"]): r for r in rows}
    L += ["", "| task | plain r1 / r2 (turns, cost) | atlas r1 / r2 (turns, cost) | first test plain / atlas |",
          "|---|---|---|---|"]
    yn = lambda r: "yes" if r["resolved"] else "no"  # noqa: E731
    for i in ids:
        cells = []
        for a in ARMS:
            c = []
            for k in REPS:
                r = by.get((i, a, k))
                c.append(f"{yn(r)} ({r['turns']}, ${r['cost_usd']:.2f})" if r else "-")
            cells.append(" / ".join(c))
        f = " / ".join(yn(first[(i, a)]) if (i, a) in first else "-" for a in ARMS)
        L.append(f"| {i} | {cells[0]} | {cells[1]} | {f} |")
    if full:
        kp = [sum(pt[(i, "plain")]) for i in full]
        ka = [sum(pt[(i, "atlas")]) for i in full]
        mt = lambda a: [statistics.mean(by[(i, a, k)]["turns"] for k in REPS) for i in full]  # noqa: E731
        mc = lambda a: [statistics.mean(by[(i, a, k)]["cost_usd"] for k in REPS) for i in full]  # noqa: E731
        ms = lambda a: [statistics.mean(by[(i, a, k)]["seconds"] for k in REPS) for i in full]  # noqa: E731
        s_res = sw.paired_sign(kp, ka)          # a_lower = plain lower = atlas higher
        both = sw.mcnemar([x == 2 for x in ka], [x == 2 for x in kp])
        s_turn = sw.paired_sign(mt("atlas"), mt("plain"))
        s_cost = sw.paired_sign(mc("atlas"), mc("plain"))
        s_sec = sw.paired_sign(ms("atlas"), ms("plain"))
        lo, hi = bootstrap_diff(ka, kp)
        diff = sum(ka) - sum(kp)
        n_runs = 2 * len(full)
        L += ["", f"Paired tests over {len(full)} tasks (exact):",
              f"* resolved runs per task (0-2), atlas vs plain: atlas higher on {s_res['a_lower']}, plain higher on "
              f"{s_res['b_lower']}, tied {len(full) - s_res['n']}; sign test p = {s_res['p']:.3f}",
              f"* resolved in both runs: atlas only {both['only_a']}, plain only {both['only_b']}; exact McNemar "
              f"p = {both['p']:.3f}",
              f"* difference in resolved task-runs (atlas - plain): {diff:+d} of {n_runs} ({100 * diff / n_runs:+.1f} "
              f"points); bootstrap 95% interval over tasks {lo:+.0f} .. {hi:+.0f}",
              f"* mean turns per task, atlas vs plain: atlas lower on {s_turn['a_lower']} of {s_turn['n']} untied, "
              f"sign test p = {s_turn['p']:.4f}; medians {statistics.median(mt('atlas')):g} vs "
              f"{statistics.median(mt('plain')):g}",
              f"* mean cost per task: atlas lower on {s_cost['a_lower']} of {s_cost['n']}, p = {s_cost['p']:.4f}; "
              f"mean per run ${statistics.mean(mc('atlas')):.2f} vs ${statistics.mean(mc('plain')):.2f} "
              f"(ratio {statistics.mean(mc('atlas')) / max(1e-9, statistics.mean(mc('plain'))):.2f})",
              f"* mean seconds per task: atlas lower on {s_sec['a_lower']} of {s_sec['n']}, p = {s_sec['p']:.3f}"]
        L += ["", "Run-to-run agreement (tasks whose run 1 and run 2 differ in resolved):"]
        for a in ARMS:
            fl = [i for i in full if pt[(i, a)][0] != pt[(i, a)][1]]
            L.append(f"* {a}: {len(fl)} of {len(full)} flip ({', '.join(fl) or 'none'})")
        L.append("* each run 1 alone, atlas vs plain: " + ", ".join(
            f"run {k}: atlas {sum(by[(i, 'atlas', k)]['resolved'] for i in full)} vs plain "
            f"{sum(by[(i, 'plain', k)]['resolved'] for i in full)}" for k in REPS))
        a1 = sum(ka) >= sum(kp)
        a2 = diff < 6
        a3 = s_turn["p"] < 0.05 and s_turn["a_lower"] > s_turn["n"] - s_turn["a_lower"]
        ratio = statistics.mean(mc("atlas")) / max(1e-9, statistics.mean(mc("plain")))
        L += ["", "Predictions:",
              f"* A1 atlas resolves at least as many task-runs as plain: {'PASS' if a1 else 'FAIL'} "
              f"({sum(ka)} vs {sum(kp)})",
              f"* A2 gap under 19 points (atlas - plain < 6 of 32 task-runs): {'PASS' if a2 else 'FAIL'} "
              f"({diff:+d}, {100 * diff / n_runs:+.1f} points)",
              f"* A3 fewer turns with the atlas (sign test p < 0.05): {'PASS' if a3 else 'FAIL'} "
              f"(lower on {s_turn['a_lower']} of {s_turn['n']}, p = {s_turn['p']:.4f})",
              f"* A4 atlas costs at least twice plain: {'PASS' if ratio >= 2 else 'FAIL'} ({ratio:.2f}x)"]
    for r in rows:                     # re-apply the current pattern (amendment 1: the CLI's own task-output dir)
        b = DEFAULT_WORK / "runs" / r["dir_tag"]
        r["leak_suspects"] = leak_hits(r.get("leak_suspects") or [], r["id"].split("__")[1].split("-")[0],
                                       own_prefixes(b, b / r["repo_dir"]))
    leaks = [(r["id"], r["arm"], r["rep"], r["leak_suspects"]) for r in rows if r.get("leak_suspects")]
    mem = [(r["id"], r["arm"], r["rep"]) for r in rows
           if r["memory_before"]["memory_files"] or r["memory_after"]["memory_files"]]
    store = [(r["id"], r["arm"], r["rep"], r["memory_after"]["files"]) for r in rows if r["memory_after"]["files"]]
    L += ["", f"Leak suspects (commands or paths reaching the web, a release, git history, the scratch dirs, a memory "
              f"store): {len(leaks)} runs"]
    L += [f"  * {x[0]} {x[1]} r{x[2]}: {x[3][:3]}" for x in leaks]
    L.append(f"Memory: runs with a memory file before or after: {len(mem)} of {len(rows)}; runs whose project store "
             f"held any file after the run: {len(store)}")
    L += [f"  * {x[0]} {x[1]} r{x[2]}: {x[3][:5]}" for x in store]
    L.append(f"Runs with more than one result event: {sum((r['results'] or 0) > 1 for r in rows)}; "
             f"timeouts (rc -9): {sum(r['rc'] == -9 for r in rows)}; errors: {sum(bool(r['is_error']) for r in rows)}")
    L.append(f"\nSpent (list price, all agent calls): ${spent():.2f}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["check", "validate", "order", "run", "analyse"])
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cap", type=float, default=40.0)
    ap.add_argument("--max-runs", type=int)
    a = ap.parse_args(argv)
    setup_work(a.work)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "check":
        r = check_first(SCRATCH / "ra_swe" / "transcripts")
        (OUT / "first_test_memory_check.json").write_text(json.dumps(r, indent=1))
        print(json.dumps({k: v for k, v in r.items() if k != "stores"}, indent=1))
    elif a.cmd == "validate":
        validate(a.work, a.workers)
    elif a.cmd == "order":
        for j in smoke_first(run_order(task_ids())):
            print(*j)
    elif a.cmd == "run":
        run(a.work, a.cap, a.max_runs, min(a.workers, 3))
    else:
        txt = analyse()
        (OUT / "summary.md").write_text(txt + "\n")
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
