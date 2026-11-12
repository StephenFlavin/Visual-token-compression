"""Repo atlas, hard fresh tasks: does the whole repository as page images at the start change what Claude Code solves?

Benchmark: SWE-rebench leaderboard (``nebius/SWE-rebench-leaderboard``, Python tasks collected monthly from GitHub),
newest first. No Docker: each task's repository is fetched at its base commit only (``git fetch --depth 1``), the
working copy is a fresh one-commit git repository (no history, nothing after the base reachable), and the task's
own install command runs in a uv virtualenv (``pip`` goes through uv where it can). Grading runs the task's own
test command and parses it with SWE-bench's pytest parser; resolved = all FAIL_TO_PASS and all PASS_TO_PASS that
pass with the gold patch here (env-broken ones left out, for every arm alike).

Arms (Claude Code headless, Opus 5.5, --effort high, tools Bash/Read/Edit/Write, max 250 turns, 5400 s):
  plain  the task prompt only
  map    a repo map (signatures per file, repo_atlas_eval.build_map) before the task prompt
  atlas  the atlas pages (repo_atlas.build) as images, a note on the format, index.txt, one sentence on how to use
         them, then the task prompt

    python3 experiments/atlas/repo_atlas_swe.py inspect                 # dataset sizes and dates (no calls)
    python3 experiments/atlas/repo_atlas_swe.py select [--workers 2]    # fit, closed-book screen, local validation
    python3 experiments/atlas/repo_atlas_swe.py run --cap 35 [--ids ...] [--max-tasks N]
    python3 experiments/atlas/repo_atlas_swe.py analyse

Rows: results/repo_atlas_swe/{funnel,validate,screen,runs,spend}.jsonl. Clones, venvs, pages and transcripts in
--work (scratchpad), never in the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

OUT = ROOT / "results" / "repo_atlas_swe"
DEFAULT_WORK = Path(os.environ.get("RA_SWE_WORK", "/tmp/claude-0/-home-user-visual-bpe/"
                                   "588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/ra_swe"))
HF = "https://huggingface.co/datasets/nebius/SWE-rebench-leaderboard/resolve/main/data/test-00000-of-00001.parquet"
ARMS = ("plain", "map", "atlas")
MODEL = "claude-opus-5-5"
EFFORT = "high"
MAX_TURNS = 250
AGENT_TIMEOUT = 5400
MIN_GOLD = 3
FIT_TOKENS = 150_000
MAX_TASKS = 24
MAX_CANDIDATES = 80
INSTALL_TIMEOUT = 480        # with the two test runs: about 10 minutes per task
TEST_TIMEOUT = 300
PAGE_ARRIVAL = 0.97
UNSUPPORTED = re.compile(r"\b(apt-get|apt|conda|mamba|brew|uv sync|uv run|docker|sudo)\b")
UV = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")

ATLAS_SENTENCE = ("The page images show the repository's source at the start of the task; read them for orientation "
                  "and edit from the files, not from the pages.")
PROMPT = """You are working in a checkout of the {repo} repository in the current directory. Resolve the issue below by
changing the source code; do not modify the existing tests. A Python virtual environment with this checkout installed
(editable) and its test dependencies is at {venv}: run tests with `{venv}/bin/python -m pytest <paths>`. When the issue
is resolved, reply DONE.

<issue>
{issue}
</issue>"""


# --------------------------------------------------------------------------- pure functions (tested)
def gold_files(patch: str) -> list[str]:
    from repo_atlas_eval import gold_files as gf
    return gf(patch)


def parse_pytest_log(log: str) -> dict[str, str]:
    """SWE-bench's parse_log_pytest: lines starting with a status word -> {test id: status}."""
    st = {}
    for line in log.split("\n"):
        if line.startswith(("FAILED", "PASSED", "SKIPPED", "ERROR", "XFAIL")):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) > 1:
                st[parts[1]] = parts[0]
    return st


def grade_status(st: dict, f2p: list[str], p2p: list[str], broken: set | None = None) -> dict:
    ok = lambda n: st.get(n) in ("PASSED", "XFAIL")  # noqa: E731
    p2p = [x for x in p2p if x not in (broken or set())]
    return {"resolved": all(map(ok, f2p)) and all(map(ok, p2p)), "f2p": f"{sum(map(ok, f2p))}/{len(f2p)}",
            "p2p": f"{sum(map(ok, p2p))}/{len(p2p)}", "f2p_failing": [x for x in f2p if not ok(x)][:5],
            "p2p_failing": [x for x in p2p if not ok(x)][:5]}


def patch_paths(patch: str) -> list[str]:
    from repo_atlas_eval import patch_files
    return patch_files(patch)


def touched_share(model_patch: str, gold: list[str]) -> float:
    """Share of the gold source files that the model's patch changes."""
    got = set(patch_paths(model_patch))
    return sum(g in got for g in gold) / len(gold) if gold else 0.0


def candidate_order(rows: list[dict], min_gold: int = MIN_GOLD) -> list[dict]:
    """Rule 1 and 3: Python tasks (the whole leaderboard is Python), newest first, gold patch changing at least
    ``min_gold`` source files; one task per (base commit, patch) (the same PR can appear under a renamed repo)."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (str(r["created_at"]), r["instance_id"]), reverse=True):
        key = hashlib.sha1((r["base_commit"] + r["patch"]).encode()).hexdigest()
        if key in seen or len(gold_files(r["patch"])) < min_gold:
            continue
        seen.add(key)
        out.append(r)
    return out


def install_supported(cmd: str) -> bool:
    """Install commands that need system packages, conda or a uv-managed project environment cannot run here."""
    return not UNSUPPORTED.search(cmd or "")


def screen_drop(files: list[str], gold: list[str]) -> bool:
    """Rule 5: drop the task when the closed-book answer already names every (existing) gold file in its top 5."""
    return bool(gold) and all(g in files[:5] for g in gold)


def first_message(arm: str, repo: str, venv: str, issue: str, map_text: str = "", atlas_index: str = "",
                  n_files: int = 0) -> str:
    """The text of the first user message (the atlas arm's images go before it)."""
    task = PROMPT.format(repo=repo, venv=venv, issue=issue)
    if arm == "plain":
        return task
    if arm == "map":
        return (f"Repository {repo}: a repo map of its {n_files} source files (tests, docs and data files left out). "
                "Per file: its path, then the signature line of every class, function and method in it, indented.\n\n"
                + map_text + "\n" + task)
    if arm == "atlas":
        return (f"The images above are the {n_files} source files of the repository {repo} (tests, docs and data files "
                "left out), in this order, each starting with a header ▶path◀. The code is minified: §<n> marks the "
                "start of original line n (shown on every 10th line, a bare § starts the others); each » before a "
                "line is one level of indentation where indentation matters; blank lines and other indentation are "
                f"not shown. {ATLAS_SENTENCE} Index (file: page):\n\n" + atlas_index + "\n" + task)
    raise ValueError(arm)


def exclude_files(patch: str, drop: set[str]) -> str:
    """Remove the per-file sections of a unified git diff whose path is in ``drop``."""
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    keep = []
    for p in parts:
        m = re.match(r"diff --git a/(\S+) b/(\S+)", p)
        if m and (m.group(1) in drop or m.group(2) in drop):
            continue
        keep.append(p)
    return "".join(keep)


LEAK = re.compile(r"(pip3?\s+(download|index)\b|pypi\.org|pip3?\s+(download|install)\s+(?!-e\b|-r\b|--editable)[^\n]*\b{pkg}\b|git\s+(fetch|pull|clone|remote)|"
                  r"curl\s|wget\s|github\.com|huggingface|parquet|SWE-?rebench|site-packages/{pkg}|"
                  r"ra_swe/(ds|transcripts|ctx|git|co|validate)\b|ra_swe/?\s|ra_swe/runs/?\s)", re.I)
# (the run's own venv path contains ra_swe/runs/<task>/<hash>/venv; a one-commit repository's git log is harmless)


def leak_hits(commands: list[str], pkg: str) -> list[str]:
    """Agent commands that could reach the fix: a newer release of the package, git history, the web, the dataset."""
    rx = re.compile(LEAK.pattern.replace("{pkg}", re.escape(pkg)), re.I)
    return [c[:200] for c in commands if rx.search(c)]


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test p for k of n at p = 0.5 (McNemar on discordant pairs; sign test)."""
    if n == 0:
        return 1.0
    pk = math.comb(n, k) / 2 ** n
    return min(1.0, sum(math.comb(n, i) / 2 ** n for i in range(n + 1) if math.comb(n, i) / 2 ** n <= pk + 1e-12))


def paired_sign(a: list[float], b: list[float]) -> dict:
    """Sign test of a vs b, pairs with a tie left out: {n, a_lower, b_lower, p}."""
    d = [x - y for x, y in zip(a, b) if x != y]
    lo = sum(x < 0 for x in d)
    return {"n": len(d), "a_lower": lo, "b_lower": len(d) - lo, "p": binom_two_sided(lo, len(d))}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    return {"only_a": only_a, "only_b": only_b, "p": binom_two_sided(only_a, only_a + only_b)}


# --------------------------------------------------------------------------- shell, data, checkouts
def sh(cmd, cwd=None, timeout=1800, env=None, check=False, inp=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=inp is None or isinstance(inp, str), timeout=timeout,
                       env=env, shell=isinstance(cmd, str), input=inp, stdin=None if inp is not None else subprocess.DEVNULL)
    if check and p.returncode:
        raise RuntimeError(f"{cmd}: {str(p.stderr)[-800:]}")
    return p


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


_alock = threading.Lock()


def append(name: str, row: dict) -> None:
    with _alock:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / name, "a") as f:
            f.write(json.dumps(row) + "\n")


def load_rows(work: Path) -> list[dict]:
    p = work / "ds" / "rebench_test.parquet"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        sh(["curl", "-sSL", "-o", str(p), HF], check=True, timeout=600)
    import pandas as pd
    d = pd.read_parquet(p)
    rows = []
    for r in d.to_dict("records"):
        r["created_at"] = str(r["created_at"])
        r["FAIL_TO_PASS"] = list(r["FAIL_TO_PASS"])
        r["PASS_TO_PASS"] = list(r["PASS_TO_PASS"])
        ic = dict(r["install_config"])
        r["install_config"] = {k: (list(v) if hasattr(v, "tolist") else v) for k, v in ic.items()}
        r.pop("meta", None)
        rows.append(r)
    return rows


def task_map(work: Path) -> dict[str, dict]:
    return {r["instance_id"]: r for r in load_rows(work)}


def git_store(work: Path, t: dict) -> Path:
    """A bare repository holding only the base commit (shallow fetch of that one commit)."""
    d = work / "git" / (t["repo"].replace("/", "__") + ".git")
    if not d.exists():
        sh(["git", "init", "-q", "--bare", str(d)], check=True)
    if sh(["git", "-C", str(d), "cat-file", "-e", t["base_commit"] + "^{commit}"]).returncode:
        sh(["git", "-C", str(d), "fetch", "-q", "--depth", "1", f"https://github.com/{t['repo']}.git", t["base_commit"]],
           check=True, timeout=1800)
    return d


def fresh_repo(work: Path, t: dict, dest: Path) -> Path:
    """The tree at the base commit as a new one-commit repository: no history, no later objects."""
    store = git_store(work, t)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    arch = subprocess.run(["git", "-C", str(store), "archive", t["base_commit"]], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arch, check=True)
    for c in (["git", "init", "-q"], ["git", "config", "user.email", "dev@example.invalid"],
              ["git", "config", "user.name", "dev"], ["git", "add", "-A", "-f"],
              ["git", "commit", "-q", "--no-verify", "-m", "base"]):
        sh(c, cwd=dest, check=True)
    return dest


def pip_shim(venv: Path) -> None:
    """``pip install ...`` in the venv goes through ``uv pip install`` (fast); anything uv refuses falls back to the
    real pip."""
    real = venv / "bin" / "pip-real"
    pip = venv / "bin" / "pip"
    if pip.exists() and not real.exists():
        pip.rename(real)
    script = f"""#!/bin/bash
if [ "$1" = "install" ]; then
  shift
  args=()
  for a in "$@"; do [ "$a" = "--break-system-packages" ] || args+=("$a"); done
  "{UV}" pip install --python "{venv}/bin/python" "${{args[@]}}" && exit 0
  exec "{real}" install "$@"
fi
exec "{real}" "$@"
"""
    for name in ("pip", "pip3"):
        (venv / "bin" / name).write_text(script)
        (venv / "bin" / name).chmod(0o755)


def tool_env(venv: Path) -> dict:
    env = dict(os.environ, VIRTUAL_ENV=str(venv), PATH=f"{venv}/bin:" + os.environ.get("PATH", ""),
               PIP_DISABLE_PIP_VERSION_CHECK="1", LANG="C.UTF-8", LC_ALL="C.UTF-8", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return env


def make_env(t: dict, repo: Path, venv: Path) -> dict:
    """uv venv with the task's Python, then its install command in the repository. Commits what the install wrote
    (``env`` commit) so the agent's diff holds only its own changes."""
    ic = t["install_config"]
    t0 = time.time()
    if venv.exists():
        shutil.rmtree(venv)
    sh([UV, "venv", "-q", "--seed", "-p", str(ic.get("python") or "3.12"), str(venv)], check=True, timeout=600)
    pip_shim(venv)
    env = tool_env(venv)
    log = []
    for c in list(ic.get("pre_install") or []) + [ic.get("install") or ""]:
        if c.strip():
            p = sh(["bash", "-c", c], cwd=repo, env=env, timeout=INSTALL_TIMEOUT)
            log.append(f"$ {c}\nrc={p.returncode}\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    if ic.get("pip_packages"):
        sh(["bash", "-c", "pip install " + " ".join(f"'{x}'" for x in ic["pip_packages"])], cwd=repo, env=env,
           timeout=INSTALL_TIMEOUT)
    sh([str(venv / "bin" / "python"), "-m", "pip", "--version"], env=env)
    if sh(["bash", "-c", "command -v pytest || python -m pytest --version"], env=env).returncode:
        sh(["bash", "-c", "pip install pytest"], cwd=repo, env=env, timeout=INSTALL_TIMEOUT)
    sh(["git", "add", "-A"], cwd=repo)
    sh(["git", "commit", "-q", "--no-verify", "--allow-empty", "-m", "env"], cwd=repo)
    return {"seconds": round(time.time() - t0), "log": "\n".join(log)[-3000:]}


def test_cmd(t: dict) -> str:
    c = t["install_config"]["test_cmd"]
    return re.sub(r"^\s*uv run\s+", "", c)


def run_tests(t: dict, repo: Path, venv: Path) -> tuple[str, float]:
    t0 = time.time()
    try:
        p = sh(["bash", "-c", test_cmd(t)], cwd=repo, env=tool_env(venv), timeout=TEST_TIMEOUT)
        log = p.stdout + "\n" + p.stderr
    except subprocess.TimeoutExpired as e:
        log = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")) + "\nTIMEOUT"
    return log, time.time() - t0


def git_apply(repo: Path, patch: str) -> bool:
    if not patch.strip():
        return True
    f = repo.parent / f"apply_{os.getpid()}_{threading.get_ident()}.diff"
    f.write_text(patch)
    ok = sh(["git", "apply", "--whitespace=nowarn", str(f)], cwd=repo).returncode == 0
    f.unlink(missing_ok=True)
    return ok


def reset(repo: Path) -> None:
    sh(["git", "reset", "-q", "--hard", "HEAD"], cwd=repo)
    sh(["git", "clean", "-qfd"], cwd=repo)


# --------------------------------------------------------------------------- selection: fit, screen, validate
def fit(work: Path, t: dict) -> dict:
    import repo_atlas as RA
    co = fresh_repo(work, t, work / "co" / t["instance_id"] / t["repo"].split("/")[1])
    st = RA.stats(co, 88.2, False, 10)
    at_base = set(sh(["git", "ls-files"], cwd=co).stdout.split())
    g = gold_files(t["patch"])
    return {"image_tokens": st["image_tokens"], "files": st["files"], "source_chars": st["source_chars"],
            "fits": 0 < st["image_tokens"] <= FIT_TOKENS, "gold": [x for x in g if x in at_base],
            "gold_new": [x for x in g if x not in at_base], "checkout": str(co)}


def screen(work: Path, t: dict, f: dict) -> dict:
    """Closed-book call: the issue and the file list only (the pilot's ``tree`` arm)."""
    import repo_atlas as RA
    import repo_atlas_eval as E
    from vbpe import claude_eval as CE
    co = Path(f["checkout"])
    paths = [x["path"] for x in RA.collect(co, False, 10)[0]]
    tree = (f"Repository {t['repo']}: its {len(paths)} source files (tests, docs and data files left out).\n\n"
            + "\n".join(paths) + "\n")
    rec = CE.transcribe([], E.SYSTEM, E.prompt_for("tree", tree, t["problem_statement"], t["repo"]), model=MODEL,
                        effort=EFFORT, timeout=600.0)
    if not rec.get("cached") and rec.get("cost_usd"):
        append("spend.jsonl", {"what": f"{t['instance_id']} screen", "cost_usd": rec["cost_usd"],
                               "input_tokens": rec.get("input_tokens"), "output_tokens": rec.get("output_tokens")})
    files, plan = E.parse_answer(rec.get("text", ""))
    return {"files": files, "plan": plan, "drop": screen_drop(files, f["gold"]), "error": rec.get("error"),
            "cost_usd": rec.get("cost_usd"), "cached": rec.get("cached")}


def validate(work: Path, t: dict) -> dict:
    out = {"id": t["instance_id"]}
    base = work / "validate" / t["instance_id"]
    repo = base / t["repo"].split("/")[1]
    t0 = time.time()
    try:
        fresh_repo(work, t, repo)
        e = make_env(t, repo, base / "venv")
        out["install_s"] = e["seconds"]
        if e["seconds"] > INSTALL_TIMEOUT:
            out.update(valid=False, why="install too slow")
            return out
        test_files = set(patch_paths(t["test_patch"]))
        assert git_apply(repo, t["test_patch"]), "test patch does not apply"
        log, s1 = run_tests(t, repo, base / "venv")
        st0 = parse_pytest_log(log)
        out["unpatched"] = grade_status(st0, t["FAIL_TO_PASS"], t["PASS_TO_PASS"])
        reset(repo)
        assert git_apply(repo, exclude_files(t["patch"], test_files)), "gold patch does not apply"
        assert git_apply(repo, t["test_patch"]), "test patch does not apply after gold"
        log, s2 = run_tests(t, repo, base / "venv")
        st = parse_pytest_log(log)
        p2p, f2p = t["PASS_TO_PASS"], t["FAIL_TO_PASS"]
        out["env_broken_p2p"] = [x for x in p2p if st.get(x) not in ("PASSED", "XFAIL")]
        out["gold"] = grade_status(st, f2p, p2p, set(out["env_broken_p2p"]))
        out["test_s"] = round(s1 + s2)
        f2p_gold = all(st.get(x) in ("PASSED", "XFAIL") for x in f2p)
        f2p_base_all = all(st0.get(x) in ("PASSED", "XFAIL") for x in f2p)
        out["valid"] = f2p_gold and not f2p_base_all and len(out["env_broken_p2p"]) <= len(p2p) / 2
        if not out["valid"]:
            out["why"] = ("gold fails FAIL_TO_PASS" if not f2p_gold else "base passes FAIL_TO_PASS" if f2p_base_all
                          else "over half of PASS_TO_PASS env-broken")
            out["log_tail"] = log[-1200:]
            out["install_log"] = e["log"][-1500:]
    except Exception as ex:  # noqa: BLE001
        out.update(valid=False, why="error", error=str(ex)[-600:])
    finally:
        out["seconds"] = round(time.time() - t0)
        shutil.rmtree(base, ignore_errors=True)
    return out


def select(work: Path, workers: int, min_gold: int) -> None:
    rows = load_rows(work)
    cands = candidate_order(rows, min_gold)
    funnel = {r["id"]: r for r in jl(OUT / "funnel.jsonl")}
    tried = len(funnel)
    accepted = sum(r.get("accepted", False) for r in funnel.values())
    todo = [t for t in cands if t["instance_id"] not in funnel]
    print(f"[sel] {len(cands)} candidates with >= {min_gold} gold files; {tried} tried, {accepted} accepted", flush=True)
    lock = threading.Lock()
    state = {"tried": tried, "accepted": accepted}

    def one(t):
        with lock:
            if state["accepted"] >= MAX_TASKS or state["tried"] >= MAX_CANDIDATES:
                return
            state["tried"] += 1
        row = {"id": t["instance_id"], "repo": t["repo"], "created_at": t["created_at"],
               "n_gold": len(gold_files(t["patch"])), "min_gold": min_gold, "accepted": False}
        try:
            if not install_supported(t["install_config"].get("install", "") + " " +
                                     " ".join(t["install_config"].get("pre_install") or [])):
                row["stage"] = "install needs system packages, conda or uv project env"
            else:
                f = fit(work, t)
                row.update({k: f[k] for k in ("image_tokens", "files", "source_chars", "fits", "gold", "gold_new")})
                if not f["fits"]:
                    row["stage"] = "atlas too large"
                else:
                    s = screen(work, t, f)
                    row["screen"] = {k: s[k] for k in ("files", "plan", "drop", "error")}
                    if s.get("error"):
                        row["stage"] = "screen error"
                    elif s["drop"]:
                        row["stage"] = "screen: closed-book names every gold file"
                    else:
                        v = validate(work, t)
                        append("validate.jsonl", v)
                        row["validate"] = {k: v.get(k) for k in ("valid", "why", "install_s", "test_s", "seconds")}
                        row["env_broken_p2p"] = len(v.get("env_broken_p2p") or [])
                        row["stage"] = "accepted" if v.get("valid") else f"validation: {v.get('why')}"
                        row["accepted"] = bool(v.get("valid"))
                shutil.rmtree(work / "co" / t["instance_id"], ignore_errors=True)
        except Exception as ex:  # noqa: BLE001
            row["stage"] = f"error: {str(ex)[-300:]}"
        with lock:
            state["accepted"] += row["accepted"]
        append("funnel.jsonl", row)
        print(f"[sel] {row['created_at'][:10]} {row['id']:45} gold={row['n_gold']} tok={row.get('image_tokens')} "
              f"-> {row['stage']}  (accepted {state['accepted']}, tried {state['tried']})", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


def accepted_ids() -> list[str]:
    """Accepted tasks, newest first."""
    rows = [r for r in jl(OUT / "funnel.jsonl") if r.get("accepted")]
    return [r["id"] for r in sorted(rows, key=lambda r: (r["created_at"], r["id"]), reverse=True)]


# --------------------------------------------------------------------------- agent runs
_slock = threading.Lock()


def spent() -> float:
    return sum(r["cost_usd"] or 0 for r in jl(OUT / "spend.jsonl"))


def contexts(work: Path, t: dict) -> dict:
    """Map text and atlas pages for one task (built once from a base checkout)."""
    import repo_atlas as RA
    import repo_atlas_eval as E
    d = work / "ctx" / t["instance_id"]
    meta_p = d / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())
    co = fresh_repo(work, t, work / "co" / t["instance_id"] / t["repo"].split("/")[1])
    files, _ = RA.collect(co, False, 10)
    texts = [(f["path"], (co / f["path"]).read_text(encoding="utf-8")) for f in files]
    d.mkdir(parents=True, exist_ok=True)
    (d / "map.txt").write_text(E.build_map(texts))
    am = RA.build(co, d / "atlas", 88.2, False, 10)
    meta = {"n_files": len(files), "pages": len(am["pages"]), "image_tokens_build": am["image_tokens"],
            "image_tokens_formula": E.image_tokens([tuple(p["size"]) for p in am["pages"]]),
            "map_chars": len((d / "map.txt").read_text()), "atlas_dir": str(d / "atlas")}
    meta_p.write_text(json.dumps(meta, indent=1))
    shutil.rmtree(work / "co" / t["instance_id"], ignore_errors=True)
    return meta


NO_INDEX = {"PIP_NO_INDEX": "1", "PIP_INDEX_URL": "http://127.0.0.1:9/simple", "UV_OFFLINE": "1",
            "UV_INDEX_URL": "http://127.0.0.1:9/simple", "UV_DEFAULT_INDEX": "http://127.0.0.1:9/simple"}


def agent_env(env: dict) -> dict:
    """Amendment 1 (after the pilot): no package index for the agent, so it cannot fetch a release of the package
    that already holds the fix (the pilot's smolvm-172 map run did). Same for every arm."""
    return {**env, **NO_INDEX}


def contaminated() -> dict[str, str]:
    """Tasks left out of the paired comparisons: an arm read a release or upstream copy of the fix (confirmed by
    reading the transcript; results/repo_atlas_swe/contaminated.json)."""
    p = OUT / "contaminated.json"
    return json.loads(p.read_text()) if p.exists() else {}


def stream_totals(ev: list[dict]) -> dict:
    """Turns and tokens over every result event of a run. A run whose agent left a background job emits a further
    result (a new short turn) when the job ends: num_turns and usage are per result, total_cost_usd is cumulative."""
    res = [e for e in ev if e.get("type") == "result"]
    keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
    usage = {k: sum((r.get("usage") or {}).get(k, 0) for r in res) for k in keys}
    return {"turns": sum(r.get("num_turns") or 0 for r in res) if res else None, "usage": usage,
            "input_tokens": sum(usage[k] for k in keys[:3]), "output_tokens": usage["output_tokens"],
            "cost_usd": float((res[-1].get("total_cost_usd") if res else 0) or 0.0), "results": len(res)}


def repair(work: Path) -> int:
    """Recompute turns, tokens and cost of stored rows from their transcripts (rows written before stream_totals)."""
    rows, n = jl(OUT / "runs.jsonl"), 0
    for r in rows:
        f = work / "transcripts" / r["id"] / f"{r['arm']}.jsonl"
        if not f.exists():
            continue
        tot = stream_totals([json.loads(l) for l in f.read_text().splitlines() if l.strip().startswith("{")])
        if tot["results"] > 1 or r.get("turns") != tot["turns"]:
            r.update({k: tot[k] for k in ("turns", "usage", "input_tokens", "output_tokens", "cost_usd")})
            r["results"] = tot["results"]
            n += 1
    (OUT / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return n


def run_one(work: Path, t: dict, arm: str, broken: set) -> dict:
    from vbpe import claude_eval as CE
    meta = contexts(work, t)
    tag = hashlib.sha1(arm.encode()).hexdigest()[:6]          # neutral directory name: the arm is not in any path
    base = work / "runs" / t["instance_id"] / tag
    repo = base / t["repo"].split("/")[1]
    venv = base / "venv"
    fresh_repo(work, t, repo)
    env_info = make_env(t, repo, venv)
    ctx = work / "ctx" / t["instance_id"]
    text = first_message(arm, t["repo"], str(venv), t["problem_statement"],
                         map_text=(ctx / "map.txt").read_text() if arm == "map" else "",
                         atlas_index=(Path(meta["atlas_dir"]) / "index.txt").read_text() if arm == "atlas" else "",
                         n_files=meta["n_files"])
    pngs = [(Path(meta["atlas_dir"]) / f"page_{i + 1:03d}.png").read_bytes() for i in range(meta["pages"])] \
        if arm == "atlas" else []
    stdin = CE.build_message(pngs, text)
    cmd = [CE.CLAUDE_BIN, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", MODEL, "--effort", EFFORT, "--tools", "Bash,Read,Edit,Write", "--allowedTools",
           "Bash,Read,Edit,Write", "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
           "--no-session-persistence", "--max-turns", str(MAX_TURNS)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, env=agent_env(CE.build_env()), cwd=repo,
                           timeout=AGENT_TIMEOUT)
        stdout, rc = p.stdout.decode("utf-8", "replace"), p.returncode
    except subprocess.TimeoutExpired as e:
        stdout, rc = (e.stdout or b"").decode("utf-8", "replace"), -9
    secs = time.time() - t0
    tr = work / "transcripts" / t["instance_id"]
    tr.mkdir(parents=True, exist_ok=True)
    (tr / f"{arm}.jsonl").write_text(stdout)
    ev = [json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]
    res = next((e for e in reversed(ev) if e.get("type") == "result"), {})
    asst = [e for e in ev if e.get("type") == "assistant"]
    first_u = (asst[0]["message"].get("usage") or {}) if asst else {}
    first_in = sum(first_u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    calls = [c for e in asst for c in e["message"]["content"] if c.get("type") == "tool_use"]
    cmds = [str((c.get("input") or {}).get("command", "")) for c in calls if c["name"] == "Bash"]
    outside = [json.dumps(c.get("input"))[:200] for c in calls if c["name"] in ("Read", "Edit", "Write")
               and not str((c.get("input") or {}).get("file_path", "")).startswith(str(repo))]
    # the model's patch: everything it changed, minus the files the task's test patch replaces
    sh(["git", "add", "-A"], cwd=repo)
    patch = sh(["git", "diff", "--cached", "HEAD"], cwd=repo).stdout
    test_files = set(patch_paths(t["test_patch"]))
    graded_patch = exclude_files(patch, test_files)
    (tr / f"{arm}.patch").write_text(patch)
    reset(repo)
    applied = git_apply(repo, graded_patch)
    if applied and git_apply(repo, t["test_patch"]):
        log, _ = run_tests(t, repo, venv)
        g = grade_status(parse_pytest_log(log), t["FAIL_TO_PASS"], t["PASS_TO_PASS"], broken)
    else:
        g = {"resolved": False, "note": "patch does not apply"}
    shutil.rmtree(base, ignore_errors=True)
    tot = stream_totals(ev)
    gold = gold_files(t["patch"])
    names = [c["name"] for c in calls]
    pkg = t["repo"].split("/")[1]
    return {"id": t["instance_id"], "arm": arm, "model": MODEL, "effort": EFFORT, "resolved": bool(g.get("resolved")),
            "grade": g, "patch_applied": applied, "patch_files": patch_paths(graded_patch),
            "patch_lines": graded_patch.count("\n"), "gold": gold, "gold_touched": touched_share(graded_patch, gold),
            "turns": tot["turns"], "cost_usd": tot["cost_usd"], "results": tot["results"],
            "subtype": res.get("subtype"), "is_error": res.get("is_error"), "rc": rc, "seconds": round(secs),
            "env_seconds": env_info["seconds"],
            "input_tokens": tot["input_tokens"], "output_tokens": tot["output_tokens"], "usage": tot["usage"],
            "first_turn_input": first_in, "image_tokens_formula": meta["image_tokens_formula"] if arm == "atlas" else 0,
            "pages_arrived": arm != "atlas" or first_in >= PAGE_ARRIVAL * meta["image_tokens_formula"],
            "pages": meta["pages"] if arm == "atlas" else 0, "map_chars": meta["map_chars"] if arm == "map" else 0,
            "tools": {k: names.count(k) for k in sorted(set(names))}, "outside_repo_edits": outside[:10],
            "leak_suspects": leak_hits(cmds, pkg), "permission_denials": [str(x)[:200] for x in res.get("permission_denials") or []][:10],
            "model_usage": {m: round(v.get("costUSD", 0.0), 4) for m, v in (res.get("modelUsage") or {}).items()},
            "final": (res.get("result") or "")[-300:]}


USAGE_LIMIT = re.compile(r"hit your .*limit|usage limit|rate limit|resets \d", re.I)
STOP: dict = {}


def run(work: Path, cap: float, ids: list[str] | None, max_tasks: int | None, workers: int) -> None:
    tm = task_map(work)
    vrows = {v["id"]: v for v in jl(OUT / "validate.jsonl") if v.get("valid")}
    order = ids or accepted_ids()
    if max_tasks:
        order = order[:max_tasks]
    done = {(r["id"], r["arm"]) for r in jl(OUT / "runs.jsonl")}
    bound = [4.0]                                             # assumed per-task cost (3 arms) until measured
    for r in _per_task_costs():
        bound[0] = max(bound[0], 1.5 * r)
    print(f"[run] {len(order)} tasks in order; spent ${spent():.2f} (cap ${cap}); per-task bound ${bound[0]:.2f}", flush=True)
    for tid in order:
        todo = [a for a in ARMS if (tid, a) not in done]
        if not todo:
            continue
        if STOP.get("why"):
            print(f"[run] STOP: usage limit ({STOP['why'][-200:]})", flush=True)
            break
        started = any((tid, a) in done for a in ARMS)
        if not started and spent() + bound[0] > cap:
            print(f"[run] STOP before {tid}: ${spent():.2f} spent + bound ${bound[0]:.2f} > cap ${cap}", flush=True)
            break
        t = tm[tid]
        broken = set(vrows[tid].get("env_broken_p2p") or [])
        contexts(work, t)

        def one(arm):
            for attempt in range(2):
                try:
                    row = run_one(work, t, arm, broken)
                except Exception as ex:  # noqa: BLE001
                    print(f"[run] {tid} {arm} ERROR {str(ex)[-300:]}", flush=True)
                    continue
                append("spend.jsonl", {"what": f"{tid} {arm}", "cost_usd": row["cost_usd"],
                                       "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"]})
                if USAGE_LIMIT.search(row["final"] or "") and (row["is_error"] or row["rc"]):
                    STOP["why"] = row["final"]
                    print(f"[run] {tid} {arm}: USAGE LIMIT, not recorded: {row['final'][-200:]}", flush=True)
                    return None
                if row["subtype"] is None and row["cost_usd"] == 0 and attempt == 0:
                    print(f"[run] {tid} {arm}: no result event (rc={row['rc']}); retry", flush=True)
                    continue
                append("runs.jsonl", row)
                print(f"[run] {tid:45} {arm:5} resolved={row['resolved']!s:5} {row['grade'].get('f2p')} "
                      f"{row['grade'].get('p2p')} gold={row['gold_touched']:.2f} turns={row['turns']} "
                      f"${row['cost_usd']:.2f} {row['seconds']}s first_in={row['first_turn_input']:,}"
                      f"{'' if row['pages_arrived'] else ' PAGES DROPPED'} leaks={len(row['leak_suspects'])}", flush=True)
                return row
        with ThreadPoolExecutor(min(workers, len(todo))) as ex:
            list(ex.map(one, todo))
        costs = _per_task_costs()
        if costs:
            bound[0] = max(4.0, 1.5 * max(costs))
        shutil.rmtree(work / "runs" / tid, ignore_errors=True)
        print(f"[run] spent ${spent():.2f}", flush=True)


def _per_task_costs() -> list[float]:
    by: dict[str, float] = {}
    for r in jl(OUT / "runs.jsonl"):
        by[r["id"]] = by.get(r["id"], 0) + (r["cost_usd"] or 0)
    return list(by.values())


# --------------------------------------------------------------------------- analysis
def analyse() -> str:
    rows = jl(OUT / "runs.jsonl")
    by = {(r["id"], r["arm"]): r for r in rows}
    full = [i for i in dict.fromkeys(r["id"] for r in rows) if all((i, a) in by for a in ARMS)]
    bad = contaminated()
    ids = [i for i in full if i not in bad]
    L = [f"Complete tasks (all three arms): {len(full)}; left out as contaminated: {len(full) - len(ids)} "
         f"({', '.join(f'{k}: {v}' for k, v in bad.items() if k in full) or 'none'}); compared: {len(ids)}", "",
         "| arm | tasks | resolved | mean share of gold files touched | median turns | mean cost | median cost | "
         "median seconds | median input tokens | median output tokens | pages dropped |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        rs = [by[(i, a)] for i in ids]
        if not rs:
            continue
        med = lambda k: statistics.median(r[k] or 0 for r in rs)  # noqa: E731
        L.append(f"| {a} | {len(rs)} | {sum(r['resolved'] for r in rs)} | "
                 f"{statistics.mean(r['gold_touched'] for r in rs):.2f} | {med('turns'):g} | "
                 f"${statistics.mean(r['cost_usd'] for r in rs):.2f} | ${med('cost_usd'):.2f} | {med('seconds'):.0f} | "
                 f"{med('input_tokens'):,.0f} | {med('output_tokens'):,.0f} | {sum(not r['pages_arrived'] for r in rs)} |")
    L += ["", "| task | gold files | " + " | ".join(f"{a}: resolved / gold touched / turns / cost" for a in ARMS) + " |",
          "|---|---|" + "---|" * len(ARMS)]
    for i in full:
        cells = []
        for a in ARMS:
            r = by[(i, a)]
            cells.append(f"{'yes' if r['resolved'] else 'no'} / {r['gold_touched']:.2f} / {r['turns']} / ${r['cost_usd']:.2f}")
        L.append(f"| {i}{' (contaminated, left out)' if i in bad else ''} | {len(by[(i, ARMS[0])]['gold'])} | "
                 + " | ".join(cells) + " |")
    if ids:
        g = lambda a, k: [by[(i, a)][k] for i in ids]  # noqa: E731
        L += ["", "Paired tests (exact):"]
        for a, b in (("atlas", "plain"), ("map", "plain"), ("atlas", "map")):
            m = mcnemar(g(a, "resolved"), g(b, "resolved"))
            L.append(f"* resolved, {a} vs {b}: {a} only {m['only_a']}, {b} only {m['only_b']}, McNemar p = {m['p']:.3f}")
        for k in ("turns", "cost_usd", "seconds", "input_tokens"):
            for a, b in (("atlas", "plain"), ("map", "plain")):
                s = paired_sign(g(a, k), g(b, k))
                d = statistics.median(x - y for x, y in zip(g(a, k), g(b, k)))
                L.append(f"* {k}, {a} vs {b}: median paired difference {d:+.3g}; {a} lower in {s['a_lower']} of {s['n']} "
                         f"untied, sign test p = {s['p']:.3f}")
        for a, b in (("atlas", "plain"),):
            ga, gb = g(a, "gold_touched"), g(b, "gold_touched")
            L.append(f"* gold files touched, {a} vs {b}: {a} higher on {sum(x > y for x, y in zip(ga, gb))}, lower on "
                     f"{sum(x < y for x, y in zip(ga, gb))}, equal on {sum(x == y for x, y in zip(ga, gb))}; means "
                     f"{statistics.mean(ga):.2f} vs {statistics.mean(gb):.2f}")
        for r in rows:                                   # re-apply the current pattern to stored commands
            r["leak_suspects"] = leak_hits(r.get("leak_suspects") or [], r["id"].split("__")[1].split("-")[0])
        leaks = [(r["id"], r["arm"], r["leak_suspects"]) for r in rows if r.get("leak_suspects")]
        L.append(f"* leak suspects (commands reaching the web, git history, a release of the package): {len(leaks)} runs")
        for x in leaks:
            L.append(f"  * {x[0]} {x[1]}: {x[2][:3]}")
    L.append(f"\nSpent (list price, all calls incl. screen): ${spent():.2f}")
    return "\n".join(L)


def inspect(work: Path) -> None:
    rows = load_rows(work)
    import collections
    months = collections.Counter(r["created_at"][:7] for r in rows)
    print(f"{len(rows)} tasks, {len({r['repo'] for r in rows})} repositories, created {min(r['created_at'] for r in rows)}"
          f" .. {max(r['created_at'] for r in rows)}")
    print("by month:", dict(sorted(months.items())))
    ng = collections.Counter(min(len(gold_files(r["patch"])), 5) for r in rows)
    print("gold source files (5 = 5+):", dict(sorted(ng.items())))
    print("candidates >= 3:", len(candidate_order(rows, 3)), " >= 2:", len(candidate_order(rows, 2)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["inspect", "select", "run", "analyse", "repair"])
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-gold", type=int, default=MIN_GOLD)
    ap.add_argument("--cap", type=float, default=35.0)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--max-tasks", type=int)
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "inspect":
        inspect(a.work)
    elif a.cmd == "select":
        select(a.work, a.workers, a.min_gold)
    elif a.cmd == "repair":
        print(f"[repair] {repair(a.work)} rows recomputed from their transcripts")
    elif a.cmd == "run":
        run(a.work, a.cap, a.ids, a.max_tasks, min(a.workers, 2))
    else:
        txt = analyse()
        (OUT / "summary.md").write_text(txt + "\n")
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
