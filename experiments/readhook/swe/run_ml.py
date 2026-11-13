"""SWE-bench Multilingual, Rust and Java (Maven/Gradle) without Docker: text vs vtchook, 2026-11-11.

Tasks: the Rust and Java tasks that the two Opus submissions on the SWE-bench Multilingual leaderboard both resolved
(mini-SWE-agent v2.0.0a0 with Opus 4.6 and Opus 4.5), hardest first (mean agent steps). Lombok's tasks need Apache Ant,
which is not installed here, and are left out.

Per repository (one at a time, to bound disk): a blobless clone; each validation or agent run gets a git worktree at
the base commit. Rust builds share one target dir per repository (debug info off) through a .cargo/config.toml above
the worktrees; it is deleted when the repository is done. Grading runs the task's own eval_script (SWE-bench
Multilingual) with /testbed pointed at the worktree and `mvnd` replaced by `mvn`, parsed by swebench's parser for the
task; resolved = all FAIL_TO_PASS and PASS_TO_PASS pass, leaving out PASS_TO_PASS tests that fail with the gold patch
here (env-broken, as in run_swe.py). Validation: the gold patch must pass FAIL_TO_PASS, the base must not, and at most
half of PASS_TO_PASS may be env-broken.

    harness-venv/bin/python run_ml.py select
    harness-venv/bin/python run_ml.py all [--repos ...]      # validate + both arms, repository by repository
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parent)]
SCR = Path("/tmp/claude-0/-home-user-visual-token-compression/9e274458-7352-5bb1-b191-52e96364b3f6/scratchpad")
WORK = SCR / "swe" / "ml"
OUT = ROOT / "results" / "readhook_swe_ml"
RUST = ["burntsushi/ripgrep", "sharkdp/bat", "tokio-rs/axum", "tokio-rs/tokio", "uutils/coreutils", "nushell/nushell",
        "astral-sh/ruff"]
JAVA = ["google/gson", "javaparser/javaparser", "reactivex/rxjava", "apache/lucene", "apache/druid"]
PASS_ENV = ("JAVA_TOOL_OPTIONS", "JAVA_HOME", "CARGO_HTTP_CAINFO", "CARGO_HOME", "RUSTUP_HOME", "GRADLE_USER_HOME")
ENV_BROKEN: dict[str, set] = {}


def sh(cmd, cwd=None, timeout=3600, env=None, check=False):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env, shell=isinstance(cmd, str),
                       stdin=subprocess.DEVNULL)
    if check and p.returncode:
        raise RuntimeError(f"{cmd}: {p.stderr[-800:]}")
    return p


def dataset():
    from datasets import load_dataset
    return {r["instance_id"]: r for r in load_dataset("SWE-bench/SWE-bench_Multilingual", split="test")}


def select() -> list[str]:
    subs = [json.loads((SCR / f).read_text()) for f in ("ml_opus46.json", "ml_opus45.json")]
    d = dataset()
    rows = [r for r in d.values() if r["repo"] in RUST + JAVA and all(s.get(r["instance_id"], {}).get("resolved") for s in subs)]
    steps = lambda r: sum(s[r["instance_id"]].get("api_calls", 0) for s in subs) / len(subs)  # noqa: E731
    rows.sort(key=lambda r: (-steps(r), r["instance_id"]))
    sel = {"source": "SWE-bench/experiments evaluation/multilingual 20260213_mini-v2.0.0a0_claude-4-6-opus and "
                     "20260213_mini-v2.0.0a0_claude-4-5-opus (both resolved); Lombok (Ant) left out",
           "ranked": [r["instance_id"] for r in rows],
           "meta": {r["instance_id"]: {"repo": r["repo"], "opus_steps": steps(r)} for r in rows}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel["ranked"]


def slug(repo: str) -> str:
    return repo.replace("/", "__")


def repo_dir(repo: str) -> Path:
    base = WORK / slug(repo)
    cache = base / "cache"
    if not (cache / ".git").exists():
        shutil.rmtree(cache, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", f"https://github.com/{repo}.git", str(cache)],
           check=True, timeout=7200)
    if repo in RUST:
        (base / ".cargo").mkdir(exist_ok=True)
        (base / ".cargo" / "config.toml").write_text(
            f'[build]\ntarget-dir = "{base / "target"}"\n[profile.dev]\ndebug = 0\n[profile.test]\ndebug = 0\n')
    return base


def worktree(t, dest: Path) -> Path:
    cache = repo_dir(t["repo"]) / "cache"
    if dest.exists():
        sh(["git", "-C", str(cache), "worktree", "remove", "--force", str(dest)])
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sh(["git", "-C", str(cache), "worktree", "add", "-q", "--detach", "-f", str(dest), t["base_commit"]], check=True, timeout=3600)
    sh(["git", "config", "user.email", "swe@example.invalid"], cwd=dest)
    sh(["git", "config", "user.name", "swe"], cwd=dest)
    return dest


def drop_worktree(t, dest: Path):
    sh(["git", "-C", str(repo_dir(t["repo"]) / "cache"), "worktree", "remove", "--force", str(dest)])
    shutil.rmtree(dest, ignore_errors=True)


def tool_env() -> dict:
    return dict(os.environ, LANG="C.UTF-8", LC_ALL="C.UTF-8")


def run_eval(t, repo: Path) -> str:
    lines = [l for l in t["eval_script"].replace("/testbed", str(repo)).splitlines()
             if "git config --global" not in l]
    script = "\n".join(lines).replace("mvnd ", "mvn ")
    f = repo.parent / f"eval_{os.getpid()}_{abs(hash(script))}.sh"
    f.write_text(script)
    try:
        p = sh(["bash", str(f)], cwd=repo, env=tool_env(), timeout=5400)
        return p.stdout + "\n" + p.stderr
    except subprocess.TimeoutExpired as e:
        return str(e.stdout or "") + "\nTIMEOUT"


def parse(t, log: str) -> dict:
    from swebench.harness.log_parsers import java, rust
    fn = getattr(rust, t["log_parser"], None) or getattr(java, t["log_parser"])
    return fn(log, None)


def grade(t, log: str) -> dict:
    st = parse(t, log)
    ok = lambda n: st.get(n) in ("PASSED", "XFAIL")  # noqa: E731
    f2p = json.loads(t["FAIL_TO_PASS"]) if isinstance(t["FAIL_TO_PASS"], str) else t["FAIL_TO_PASS"]
    p2p = json.loads(t["PASS_TO_PASS"]) if isinstance(t["PASS_TO_PASS"], str) else t["PASS_TO_PASS"]
    p2p = [x for x in p2p if x not in ENV_BROKEN.get(t["instance_id"], set())]
    return {"resolved": all(map(ok, f2p)) and all(map(ok, p2p)), "f2p": f"{sum(map(ok, f2p))}/{len(f2p)}",
            "p2p": f"{sum(map(ok, p2p))}/{len(p2p)}", "f2p_failing": [x for x in f2p if not ok(x)][:5],
            "p2p_failing": [x for x in p2p if not ok(x)][:5], "parsed": len(st)}


def git_apply(repo: Path, patch: str) -> bool:
    f = repo.parent / f"patch_{os.getpid()}_{abs(hash(patch))}.diff"
    f.write_text(patch)
    return sh(["git", "apply", "--whitespace=nowarn", str(f)], cwd=repo).returncode == 0


def validate(t) -> dict:
    out = {"id": t["instance_id"], "repo": t["repo"]}
    repo = worktree(t, WORK / slug(t["repo"]) / "validate" / t["instance_id"] / "repo")
    t0 = time.time()
    try:
        out["unpatched"] = grade(t, run_eval(t, repo))
        sh(["git", "reset", "-q", "--hard", t["base_commit"]], cwd=repo)
        sh(["git", "clean", "-qfd"], cwd=repo)
        assert git_apply(repo, t["patch"]), "gold patch does not apply"
        log = run_eval(t, repo)
        st = parse(t, log)
        p2p = json.loads(t["PASS_TO_PASS"]) if isinstance(t["PASS_TO_PASS"], str) else t["PASS_TO_PASS"]
        f2p = json.loads(t["FAIL_TO_PASS"]) if isinstance(t["FAIL_TO_PASS"], str) else t["FAIL_TO_PASS"]
        out["env_broken_p2p"] = [x for x in p2p if st.get(x) not in ("PASSED", "XFAIL")]
        out["gold_f2p"] = f"{sum(st.get(x) in ('PASSED', 'XFAIL') for x in f2p)}/{len(f2p)}"
        out["valid"] = (all(st.get(x) in ("PASSED", "XFAIL") for x in f2p)
                        and out["unpatched"]["f2p"] != f"{len(f2p)}/{len(f2p)}"
                        and len(out["env_broken_p2p"]) <= len(p2p) / 2)
        if not out["valid"]:
            out["log_tail"] = log[-1500:]
    finally:
        out["seconds"] = round(time.time() - t0)
        drop_worktree(t, repo)
    return out


PROMPT = """You are working in a checkout of the {repo} repository in the current directory. Resolve the issue below by
changing the source code; do not modify the existing tests. The project's build and test tools are installed
({tools}); run the relevant tests with the project's usual commands. When the issue is resolved, reply DONE.

<issue>
{issue}
</issue>"""


def run_one(t, arm: str, model: str, effort: str) -> dict:
    from vtc import claude_eval as CE
    base = WORK / slug(t["repo"]) / "runs" / t["instance_id"] / arm
    shutil.rmtree(base, ignore_errors=True)
    repo = worktree(t, base / "repo")
    if arm == "vtchook":
        from vtchook.install import settings as vb_settings
        (repo / ".claude").mkdir(exist_ok=True)
        (repo / ".claude" / "settings.local.json").write_text(json.dumps(vb_settings(
            budget=float(os.environ.get("VTCHOOK_BUDGET", "88.2")), cache=str(base / "cache"), log=str(base / "hook.jsonl"),
            python=shutil.which("python3", path="/usr/local/bin:/usr/bin") or "python3")))
    tools = "cargo, rustc" if t["repo"] in RUST else "Java 21, mvn, and the Gradle wrapper ./gradlew where the project has one"
    env = CE.build_env()
    env.update({k: os.environ[k] for k in PASS_ENV if k in os.environ})
    cmd = [CE.CLAUDE_BIN, "-p", PROMPT.format(repo=t["repo"], tools=tools, issue=t["problem_statement"]),
           "--output-format", "stream-json", "--verbose", "--model", model, "--effort", effort,
           "--tools", "Bash,Read,Edit,Write", "--allowedTools", "Bash,Read,Edit,Write",
           "--setting-sources", "project,local", "--strict-mcp-config", "--no-session-persistence", "--max-turns", "250"]
    t0 = time.time()
    try:
        stdout = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=repo, timeout=7200,
                                stdin=subprocess.DEVNULL).stdout   # no inherited stdin: `grep -r` without a path reads it
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout) or ""
    (base / "stream.jsonl").write_text(stdout)
    ev = [json.loads(l) for l in stdout.splitlines() if l.strip().startswith("{")]
    res = next((e for e in ev if e.get("type") == "result"), {})
    sh(["git", "add", "-A", "--", ".", ":(exclude).claude"], cwd=repo)
    patch = sh(["git", "diff", "--cached", t["base_commit"]], cwd=repo).stdout
    (base / "model.patch").write_text(patch)
    sh(["git", "reset", "-q", "--hard", t["base_commit"]], cwd=repo)
    sh(["git", "clean", "-qfd", "-e", ".claude"], cwd=repo)
    applied = (not patch.strip()) or git_apply(repo, patch)
    g = grade(t, run_eval(t, repo)) if applied else {"resolved": False, "note": "model patch does not apply"}
    hook = [json.loads(l) for l in (base / "hook.jsonl").read_text().splitlines()] if (base / "hook.jsonl").exists() else []
    drop_worktree(t, repo)
    shutil.rmtree(base / "cache", ignore_errors=True)
    tl = [c["name"] for e in ev if e.get("type") == "assistant" for c in e["message"]["content"] if c.get("type") == "tool_use"]
    u = res.get("usage") or {}
    img = [h for h in hook if h.get("decision") == "image"]
    return {"id": t["instance_id"], "repo": t["repo"], "arm": arm, "model": model, "effort": effort,
            "resolved": g.get("resolved"), "grade": g, "patch_applied": applied, "patch_lines": patch.count("\n"),
            "turns": res.get("num_turns"), "cost_usd": res.get("total_cost_usd"), "is_error": res.get("is_error"),
            "wall_s": round(time.time() - t0),
            "input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
            "output_tokens": u.get("output_tokens"), "tools": {k: tl.count(k) for k in set(tl)},
            "uncached_input": u.get("input_tokens", 0), "cache_write": u.get("cache_creation_input_tokens", 0),
            "cache_read": u.get("cache_read_input_tokens", 0),
            "image_views": len(img), "image_tokens": sum(h["image_tokens"] for h in img),
            "image_text_tokens": sum(h.get("text_tokens", 0) for h in img),
            "kinds": sorted({h.get("kind", "") for h in img}), "final": (res.get("result") or "")[-300:]}


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def append(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def do_repo(repo: str, ts: list, a) -> None:
    print(f"[ml] === {repo}: {len(ts)} tasks", flush=True)
    repo_dir(repo)
    done_v = {r["id"] for r in jl(OUT / "validate.jsonl")}

    def safe_validate(t):
        try:
            return validate(t)
        except Exception as e:  # noqa: BLE001
            return {"id": t["instance_id"], "repo": repo, "valid": False, "error": str(e)[-500:]}
    with ThreadPoolExecutor(a.vworkers) as ex:
        for r in ex.map(safe_validate, [t for t in ts if t["instance_id"] not in done_v]):
            append(OUT / "validate.jsonl", r)
            print(f"[validate] {r['id']} valid={r['valid']} gold f2p={r.get('gold_f2p')} env-broken p2p="
                  f"{len(r.get('env_broken_p2p', []))} unpatched f2p={r.get('unpatched', {}).get('f2p')} "
                  f"{r.get('seconds', '')}s {(r.get('error') or '')[:120]}", flush=True)
    if a.validate_only:
        return
    vrows = {r["id"]: r for r in jl(OUT / "validate.jsonl")}
    ENV_BROKEN.update({k: set(v.get("env_broken_p2p", [])) for k, v in vrows.items()})
    done = {(r["id"], r["arm"]) for r in jl(OUT / "runs.jsonl")}
    jobs = [(t, arm) for t in ts if vrows.get(t["instance_id"], {}).get("valid") for arm in a.arms
            if (t["instance_id"], arm) not in done]

    def safe_run(j):
        try:
            return run_one(j[0], j[1], a.model, a.effort)
        except Exception as e:  # noqa: BLE001
            return {"id": j[0]["instance_id"], "arm": j[1], "error": str(e)[-500:]}
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(safe_run, jobs):
            if r.get("error"):
                print(f"[ml] {r['id']} {r['arm']} ERROR {r['error'][:200]}", flush=True)
                continue
            append(OUT / "runs.jsonl", r)
            print(f"[ml] {r['id']} {r['arm']:6} resolved={r['resolved']} {r['grade'].get('f2p')} {r['grade'].get('p2p')} "
                  f"turns={r['turns']} ${r['cost_usd'] or 0:.2f} in={r['input_tokens']:,} images={r['image_views']} "
                  f"({r['image_tokens']:,} vs {r['image_text_tokens']:,}) {r['wall_s']}s", flush=True)
    shutil.rmtree(WORK / slug(repo) / "target", ignore_errors=True)       # disk: build artefacts per repository


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "all"])
    ap.add_argument("--repos", nargs="*")
    ap.add_argument("--arms", nargs="+", default=["text", "vtchook"])
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--vworkers", type=int, default=2)
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "select":
        ids = select()
        print(len(ids), "tasks")
        return 0
    sel = json.loads((OUT / "selection.json").read_text())
    d = dataset()
    for repo in a.repos or (RUST + JAVA):
        ts = [d[i] for i in sel["ranked"] if sel["meta"][i]["repo"] == repo]
        if ts:
            do_repo(repo, ts, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
