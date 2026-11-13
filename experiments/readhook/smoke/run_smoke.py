"""Smoke test: refactor-for-performance tasks, Read as text vs Read as minified page images (2026-11-11).

    python3 experiments/readhook/smoke/run_smoke.py --arms text image --reps 1
"""
from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE)]
from vtc import claude_eval as CE  # noqa: E402
import checks  # noqa: E402

OUT = ROOT / "results" / "readhook_smoke"
PROMPT = ("`{t}.py` contains a slow function `{fn}`. Refactor `{fn}` for performance so that it scales to large "
          "inputs (roughly linear or n log n). Its behaviour must stay exactly the same, including ordering, "
          "tie-breaking and edge cases; do not change the other functions. Read the module before changing it. Run "
          "the tests with `python3 -m pytest -q` (do not modify the tests). When you are done, reply DONE.")


def settings(arm: str, work: Path) -> dict:
    env = f"READHOOK_LOG={work}/hook.jsonl READHOOK_CACHE={work}/.cache"
    guard = {"matcher": "Bash", "hooks": [{"type": "command", "command": f"python3 {HERE}/bash_guard_hook.py"}]}
    if arm == "vtchook":                    # the full hook set (Read, Edit/Write, Bash output), no Bash guard
        sys.path.insert(0, str(HERE.parent))
        from vtchook.install import settings as vb_settings
        return vb_settings(budget=float(os.environ.get("VTCHOOK_BUDGET", "88.2")), cache=f"{work}/.cache",
                           log=f"{work}/hook.jsonl")
    if arm == "text":
        return {"hooks": {"PreToolUse": [guard, {"matcher": "Read", "hooks": [
            {"type": "command", "command": f"{env} python3 {HERE}/log_read_hook.py"}]}]}}
    return {"hooks": {"PreToolUse": [guard,
        {"matcher": "Read", "hooks": [{"type": "command", "command": f"{env} python3 {HERE}/img_read_hook.py"}]},
        {"matcher": "Edit|Write", "hooks": [{"type": "command", "command": f"{env} python3 {HERE}/img_edit_hook.py"}]}]}}


def run_one(task: str, arm: str, rep: int, model: str, effort: str) -> dict:
    fn = checks.TASKS[task]
    base = Path(tempfile.mkdtemp(prefix=f"smoke_{task}_{arm}_"))
    work = base / "repo"
    shutil.copytree(HERE / "tasks" / task, work)
    (work / ".claude").mkdir()
    (work / ".claude" / "settings.local.json").write_text(json.dumps(settings(arm, base)))
    cmd = [CE.CLAUDE_BIN, "-p", PROMPT.format(t=task, fn=fn), "--output-format", "stream-json", "--verbose",
           "--model", model, "--effort", effort, "--tools", "Read,Edit,Write,Bash",
           "--allowedTools", "Read,Edit,Write,Bash(python3 -m pytest:*)", "--setting-sources", "project,local",
           "--strict-mcp-config", "--no-session-persistence", "--max-turns", "40"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=CE.build_env(), cwd=work, timeout=3600)
    ev = [json.loads(l) for l in p.stdout.splitlines() if l.strip().startswith("{")]
    res = next((e for e in ev if e.get("type") == "result"), {})
    tools, errors, bash_chars = [], [], 0
    pending = {}
    for e in ev:
        if e.get("type") == "assistant":
            for c in e["message"]["content"]:
                if c.get("type") == "tool_use":
                    tools.append(c["name"])
                    pending[c["id"]] = c["name"]
        if e.get("type") == "user":
            for c in e["message"]["content"]:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    name = pending.get(c.get("tool_use_id"), "?")
                    body = c.get("content")
                    txt = body if isinstance(body, str) else " ".join(x.get("text", "") for x in body or [] if isinstance(x, dict))
                    if c.get("is_error"):
                        errors.append({"tool": name, "msg": txt[:200]})
                    if name == "Bash":
                        bash_chars += len(txt)
    hook = [json.loads(l) for l in (base / "hook.jsonl").read_text().splitlines()] if (base / "hook.jsonl").exists() else []
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=work,
                           capture_output=True, text=True, timeout=600)
    eq = checks.equivalence(task, work / f"{task}.py")
    try:
        be = checks.bench(task, work / f"{task}.py") if eq.get("ok") else None
    except Exception as e:  # noqa: BLE001
        be = {"error": str(e)}
    u = res.get("usage") or {}
    row = {"task": task, "arm": arm, "rep": rep, "model": model, "effort": effort,
           "tests_pass": tests.returncode == 0, "equivalent": eq.get("ok"), "eq": eq, "bench": be,
           "turns": res.get("num_turns"), "cost_usd": res.get("total_cost_usd"),
           "input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
           "output_tokens": u.get("output_tokens"), "tools": tools, "tool_errors": errors, "bash_output_chars": bash_chars,
           "reads": [h for h in hook if "path" in h], "edit_hook": [h for h in hook if "tool" in h],
           "final": (res.get("result") or "")[-300:], "is_error": res.get("is_error"), "workdir": str(work)}
    if arm == "vtchook":
        row["reads"] = [h for h in hook if h.get("tool") == "Read"]
        row["edit_hook"] = [h for h in hook if h.get("tool") in ("Edit", "Write")]
        row["bash_hook"] = [h for h in hook if h.get("tool") == "Bash"]
        row["read_tokens"] = sum(h.get("image_tokens", 0) if h.get("decision") == "image" else h.get("text_tokens", 0)
                                 for h in row["reads"])
    else:
        row["read_tokens"] = sum(h.get("vision_tokens") or h.get("read_text_tokens_est", 0) for h in row["reads"])
    (base / "stream.jsonl").write_text(p.stdout)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["text", "image"])
    ap.add_argument("--tasks", nargs="+", default=list(checks.TASKS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(t, arm, r) for r in range(a.reps) for t in a.tasks for arm in a.arms]
    with ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(lambda j: run_one(*j, a.model, a.effort), jobs):
            with open(OUT / "runs.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            b = row["bench"] or {}
            print(f"[smoke] {row['task']:10} {row['arm']:5} tests {row['tests_pass']} equiv {row['equivalent']} "
                  f"speedup {b.get('speedup')} read tokens {row['read_tokens']} turns {row['turns']} "
                  f"${row['cost_usd'] or 0:.2f} tool errors {len(row['tool_errors'])} fails {row['eq'].get('fails')}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
