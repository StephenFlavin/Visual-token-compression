"""SWE-bench Verified (Django subset) without Docker: text vs vbhook (code and output as page images), 2026-11-11.

Each task: check out the base commit, build a uv virtualenv with the version's Python, install Django editable plus
tests/requirements/py3.txt (line by line, failures ignored). Grading follows SWE-bench: apply the model patch (its
changes under tests/ are dropped), apply the task's test patch, run tests/runtests.py on the test modules the test
patch touches, parse the log with swebench's Django parser; resolved = every FAIL_TO_PASS and PASS_TO_PASS test passes.
`validate` checks the environment first: the gold patch must resolve the task and the unpatched code must not.

    harness-venv/bin/python run_swe.py validate            # no Claude calls
    harness-venv/bin/python run_swe.py run --arms text vbhook [--ids ...]
    harness-venv/bin/python run_swe.py report
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parent)]
WORK = Path(os.environ.get("SWE_WORK", "/tmp/claude-0/-home-user-visual-bpe/9e274458-7352-5bb1-b191-52e96364b3f6/scratchpad/swe"))
OUT = ROOT / "results" / "readhook_swe"
PY = {"4.0": "3.8", "4.1": "3.9", "4.2": "3.9", "5.0": "3.11"}
UV = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")


def sh(cmd, cwd=None, timeout=1800, env=None, check=False):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env, shell=isinstance(cmd, str))
    if check and p.returncode:
        raise RuntimeError(f"{cmd}: {p.stderr[-800:]}")
    return p


def tasks(ids=None):
    from datasets import load_dataset
    d = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    rows = [r for r in d if r["repo"] == "django/django" and r["version"] in PY and r["difficulty"] == "1-4 hours"]
    rows.sort(key=lambda r: r["instance_id"])
    return [r for r in rows if not ids or r["instance_id"] in ids]


def checkout(t, dest: Path):
    cache = WORK / "django-cache"
    if not cache.exists():
        sh(["git", "clone", "-q", "https://github.com/django/django.git", str(cache)], check=True, timeout=3600)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "clone", "-q", "--shared", "--no-checkout", str(cache), str(dest)], check=True)
    sh(["git", "checkout", "-q", t["base_commit"]], cwd=dest, check=True)
    sh(["git", "config", "user.email", "swe@example.invalid"], cwd=dest)
    sh(["git", "config", "user.name", "swe"], cwd=dest)


def make_env(t, repo: Path) -> Path:
    venv = repo.parent / "venv"
    sh([UV, "venv", "-q", "-p", PY[t["version"]], str(venv)], check=True)
    py = str(venv / "bin" / "python")
    sh([UV, "pip", "install", "-q", "-p", py, "-e", str(repo)], check=True, timeout=1800)
    req = repo / "tests" / "requirements" / "py3.txt"
    if req.exists():
        for line in req.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line and not line.startswith("-"):
                sh([UV, "pip", "install", "-q", "-p", py, line], timeout=900)
    return venv


def git_apply(repo: Path, patch: str) -> bool:
    f = repo.parent / f"patch_{abs(hash(patch))}.diff"
    f.write_text(patch)
    return sh(["git", "apply", "--whitespace=nowarn", str(f)], cwd=repo).returncode == 0


def labels(test_patch: str) -> list[str]:
    files = re.findall(r"^diff --git a/(\S+) b/", test_patch, re.M)
    return [f[len("tests/"):-3].replace("/", ".") for f in files if f.startswith("tests/") and f.endswith(".py")]


def run_tests(t, repo: Path, venv: Path) -> str:
    env = dict(os.environ, LANG="en_US.UTF-8", LC_ALL="en_US.UTF-8", PYTHONIOENCODING="utf8")
    p = sh([str(venv / "bin" / "python"), "tests/runtests.py", "--settings=test_sqlite", "--parallel", "1",
            "--verbosity", "2", *labels(t["test_patch"])], cwd=repo, env=env, timeout=2400)
    return p.stdout + "\n" + p.stderr


def grade(t, log: str) -> dict:
    from swebench.harness.log_parsers.python import parse_log_django
    st = parse_log_django(log, None)
    ok = lambda name: st.get(name) in ("PASSED", "XFAIL")  # noqa: E731
    f2p, p2p = json.loads(t["FAIL_TO_PASS"]), json.loads(t["PASS_TO_PASS"])
    return {"resolved": all(map(ok, f2p)) and all(map(ok, p2p)), "f2p": f"{sum(map(ok, f2p))}/{len(f2p)}",
            "p2p": f"{sum(map(ok, p2p))}/{len(p2p)}", "f2p_failing": [x for x in f2p if not ok(x)][:5],
            "p2p_failing": [x for x in p2p if not ok(x)][:5]}


def validate(t) -> dict:
    out = {"id": t["instance_id"]}
    base = WORK / "validate" / t["instance_id"]
    repo = base / "repo"
    checkout(t, repo)
    venv = make_env(t, repo)
    assert git_apply(repo, t["test_patch"]), "test patch does not apply"
    out["unpatched"] = grade(t, run_tests(t, repo, venv))
    assert git_apply(repo, t["patch"]), "gold patch does not apply"
    out["gold"] = grade(t, run_tests(t, repo, venv))
    out["valid"] = out["gold"]["resolved"] and not out["unpatched"]["resolved"]
    shutil.rmtree(base, ignore_errors=True)
    return out


PROMPT = """You are working in a checkout of the Django repository in the current directory. Resolve the issue below by
changing the source code; do not modify the existing tests. A Python environment with this checkout installed in
editable mode is at {venv}: run tests with `{venv}/bin/python tests/runtests.py --settings=test_sqlite --parallel 1 <test labels>`
(for example `{venv}/bin/python tests/runtests.py --settings=test_sqlite --parallel 1 queries`). When the issue is
resolved, reply DONE.

<issue>
{issue}
</issue>"""


def run_one(t, arm: str, model: str, effort: str) -> dict:
    from vbpe import claude_eval as CE
    base = WORK / "runs" / t["instance_id"] / arm
    repo = base / "repo"
    checkout(t, repo)
    venv = make_env(t, repo)
    if arm == "vbhook":
        from vbhook.install import settings as vb_settings
        (repo / ".claude").mkdir()
        (repo / ".claude" / "settings.local.json").write_text(json.dumps(vb_settings(
            budget=float(os.environ.get("VBHOOK_BUDGET", "88.2")), cache=str(base / "cache"), log=str(base / "hook.jsonl"),
            python=shutil.which("python3", path="/usr/local/bin:/usr/bin") or "python3")))   # hooks need the renderer's deps
        sh("echo .claude/ >> .git/info/exclude", cwd=repo)
    cmd = [CE.CLAUDE_BIN, "-p", PROMPT.format(venv=venv, issue=t["problem_statement"]), "--output-format", "stream-json",
           "--verbose", "--model", model, "--effort", effort, "--tools", "Bash,Read,Edit,Write",
           "--allowedTools", "Bash,Read,Edit,Write", "--setting-sources", "project,local", "--strict-mcp-config",
           "--no-session-persistence", "--max-turns", "250"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=CE.build_env(), cwd=repo, timeout=5400)
        stdout = p.stdout
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    (base / "stream.jsonl").write_text(stdout)
    ev = [json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]
    res = next((e for e in ev if e.get("type") == "result"), {})
    # the model's patch: everything but tests/ (SWE-bench applies the task's own test patch)
    sh(["git", "add", "-A"], cwd=repo)
    patch = sh(["git", "diff", "--cached", t["base_commit"], "--", ".", ":(exclude)tests"], cwd=repo).stdout
    (base / "model.patch").write_text(patch)
    sh(["git", "reset", "-q", "--hard", t["base_commit"]], cwd=repo)
    sh(["git", "clean", "-qfd", "-e", ".claude"], cwd=repo)
    applied = (not patch.strip()) or git_apply(repo, patch)
    g = grade(t, run_tests(t, repo, venv)) if applied and git_apply(repo, t["test_patch"]) else {"resolved": False}
    hook = [json.loads(l) for l in (base / "hook.jsonl").read_text().splitlines()] if (base / "hook.jsonl").exists() else []
    tools = [c["name"] for e in ev if e.get("type") == "assistant" for c in e["message"]["content"] if c.get("type") == "tool_use"]
    u = res.get("usage") or {}
    img = [h for h in hook if h.get("decision") == "image"]
    return {"id": t["instance_id"], "arm": arm, "model": model, "effort": effort, "resolved": g.get("resolved"), "grade": g,
            "patch_applied": applied, "patch_lines": patch.count("\n"), "turns": res.get("num_turns"),
            "cost_usd": res.get("total_cost_usd"), "is_error": res.get("is_error"), "wall_s": round(time.time() - t0),
            "input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
            "output_tokens": u.get("output_tokens"), "tools": {k: tools.count(k) for k in set(tools)},
            "image_views": len(img), "image_tokens": sum(h["image_tokens"] for h in img),
            "image_text_tokens": sum(h.get("text_tokens", 0) for h in img),
            "text_decisions": sum(1 for h in hook if h.get("decision") == "text"),
            "final": (res.get("result") or "")[-300:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "run", "report"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--arms", nargs="+", default=["text", "vbhook"])
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "validate":
        with ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(lambda t: (lambda: validate(t))() if True else None, tasks(a.ids)):
                with open(OUT / "validate.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"[validate] {r['id']} valid={r['valid']} gold={r['gold']['f2p']},{r['gold']['p2p']} "
                      f"unpatched f2p={r['unpatched']['f2p']}", flush=True)
        return 0
    if a.cmd == "run":
        valid = {json.loads(l)["id"] for l in (OUT / "validate.jsonl").read_text().splitlines() if json.loads(l)["valid"]}
        ts = [t for t in tasks(a.ids) if t["instance_id"] in valid]
        jobs = [(t, arm) for t in ts for arm in a.arms]
        with ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(lambda j: run_one(j[0], j[1], a.model, a.effort), jobs):
                with open(OUT / "runs.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"[swe] {r['id']} {r['arm']:6} resolved={r['resolved']} {r['grade'].get('f2p')} {r['grade'].get('p2p')} "
                      f"turns={r['turns']} ${r['cost_usd'] or 0:.2f} in={r['input_tokens']:,} images={r['image_views']} "
                      f"({r['image_tokens']:,} vs {r['image_text_tokens']:,}) {r['wall_s']}s", flush=True)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
