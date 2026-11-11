"""Compaction with a page-image archive, at scale and through real hooks (results/compaction_scale/README.md).

Per session (sessions.py: research, code or logs material):
  1. reading: a headless Claude Code session reads the material (Read or Bash, low effort); the material is deleted.
  2. D: a fork answers 4 turns of 10 questions, then a task turn (write handoff.md with 5 values).
  3. A: a fork runs /compact with no hooks, then the same turns.
  4. B: a fork runs /compact with hooks/precompact_archive.py and hooks/sessionstart_archive.py installed as project
     hooks, then the same turns. No prompt mentions the archive.
Questions and turns are identical across arms. Cost per invocation from its usage at Opus 5.5 list prices; /compact
reports no usage of its own, so its cost is the change in the session's CLI total.

Resumable: every finished invocation is appended to results/compaction_scale/runs.jsonl and the session ids are kept in
<work>/<session>/state.json; a re-run skips finished steps and continues forks by id. A failed invocation (e.g. the
usage limit) stops that session; re-run the same command to continue.

    python3 experiments/compaction/run_scale.py --sessions research:0 code:0 logs:0      # pilot
    python3 experiments/compaction/run_scale.py --sessions all
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import run_compact as RC  # noqa: E402  (claude(), transcript(); MODEL = Opus 5.5)
import sessions as S  # noqa: E402

WORK = RC.WORK.parent / "compaction_scale"
OUT = RC.ROOT / "results" / "compaction_scale"
ARMS = ["D", "A", "B"]
_lock = threading.Lock()

Q_PROMPT = """Answer these questions about the material from earlier in this session (the files are no longer on disk).
Answer each with the exact value only (a number), one line per question in the form 'ID: answer'. If you cannot
find an answer, give your best guess.

Questions:
{questions}"""

TASK_PROMPT = """Continue the work: create the file handoff.md in the current directory with one line per item in the
form 'ID: value', for the items below. Use the exact values from the material earlier in this session (the files are
no longer on disk). Then reply DONE.

Items:
{questions}"""


def hooks_settings() -> dict:
    py = sys.executable
    return {"hooks": {
        "PreCompact": [{"hooks": [{"type": "command", "timeout": 600,
                                   "command": f"{py} {HERE / 'hooks' / 'precompact_archive.py'}"}]}],
        "SessionStart": [{"matcher": "compact", "hooks": [{"type": "command", "timeout": 60,
                                                            "command": f"{py} {HERE / 'hooks' / 'sessionstart_archive.py'}"}]}]}}


def parse(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*[-*]?\s*\**([A-Z]\d{2})\**\s*[:\-=]\s*\**(.+?)\**\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def num(s: str | None) -> float | None:
    if s is None:
        return None
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s.replace(" ", ""))
    try:
        return float(m.group(0).replace(",", "")) if m else None
    except ValueError:
        return None


def score(qs: list[dict], text: str) -> list[dict]:
    ans = parse(text)
    return [{"id": q["id"], "kind": q["kind"], "answer": q["answer"], "given": ans.get(q["id"]),
             "correct": num(ans.get(q["id"])) is not None and abs(num(ans[q["id"]]) - float(q["answer"])) < 1e-9}
            for q in qs]


def log_row(row: dict) -> None:
    with _lock, open(OUT / "runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[scale] {row['session']} {row['arm']} {row['step']}: cost ${row['cost']:.3f}"
          + (f", {sum(p['correct'] for p in row['scores'])}/{len(row['scores'])} correct" if row.get("scores") else "")
          + f", reads {row['reads']} (archive {row.get('archive_reads', 0)}), {row['wall_s']}s", flush=True)


def keep(r: dict) -> dict:
    return {x: r[x] for x in ("cost", "cli_session_total", "turns", "wall_s", "reads", "output_tokens", "cache_read",
                              "cache_write", "uncached", "is_error")}


def failed(r: dict) -> bool:
    return bool(r.get("is_error")) or not r.get("session_id")


def run_session(name: str, a) -> None:
    kind, k = name.split(":")
    d = WORK / f"{kind}{k}"
    proj = d / "proj"
    stf = d / "state.json"
    st = json.loads(stf.read_text()) if stf.exists() else {}
    save = lambda: stf.write_text(json.dumps(st, indent=1))  # noqa: E731
    if "read_sid" not in st:
        if d.exists():
            shutil.rmtree(d)
        proj.mkdir(parents=True)
        mat = S.BUILDERS[kind](int(k), proj)
        (d / "material.json").write_text(json.dumps({x: mat[x] for x in ("kind", "files", "questions", "task")}))
        (OUT / "material").mkdir(parents=True, exist_ok=True)
        (OUT / "material" / f"{kind}{k}.json").write_text(json.dumps(
            {x: mat[x] for x in ("kind", "files", "questions", "task")}, indent=1))
        tools = "Bash" if kind == "logs" else "Read"
        r = RC.claude([], mat["read_prompt"], proj, tools, "low", timeout=7200)
        if failed(r):
            print(f"[scale] {name} reading FAILED: {r.get('result', '')[:300]} {r.get('stderr', '')[-300:]}", flush=True)
            return
        for x in mat["remove"]:
            shutil.rmtree(proj / x, ignore_errors=True)
        st.update(read_sid=r["session_id"], read_total=r.get("cli_session_total"))
        save()
        log_row({"session": name, "kind": kind, "arm": "-", "step": "read", **keep(r)})
    mat = json.loads((d / "material.json").read_text())
    qs, task = mat["questions"], mat["task"]
    turns = [qs[i:i + 10] for i in range(0, 40, 10)]
    for arm in a.arms:
        sk = f"{arm}_sid"
        if arm in ("A", "B", "Bfix") and sk not in st:
            sp = proj / ".claude" / "settings.json"
            if arm.startswith("B"):
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.write_text(json.dumps(hooks_settings()))
            # fork first (a one-line turn), then compact that session: with --fork-session on /compact itself the
            # PreCompact hook gets the new fork's transcript path before the file exists
            f0 = st.get(f"{arm}_fork")
            if not f0:
                r0 = RC.claude(["--resume", st["read_sid"], "--fork-session"], "Reply OK.", proj, "", "low")
                if failed(r0):
                    print(f"[scale] {name} {arm} fork FAILED: {r0.get('result', '')[:300]}", flush=True)
                    return
                f0 = st[f"{arm}_fork"] = r0["session_id"]
                st[f"{arm}_fork_total"] = r0.get("cli_session_total")
                save()
                log_row({"session": name, "kind": kind, "arm": arm, "step": "fork", **keep(r0)})
            r = RC.claude(["--resume", f0], "/compact", proj, "", "high", timeout=3600)
            if failed(r):
                print(f"[scale] {name} {arm} compaction FAILED: {r.get('result', '')[:300]}", flush=True)
                return
            st[sk] = r["session_id"]
            extra = {}
            if arm.startswith("B"):
                latest = proj / ".claude" / "compact_archive" / "LATEST"
                meta = json.loads((Path(latest.read_text().strip()) / "meta.json").read_text()) if latest.exists() else {}
                tr = json.dumps(RC.transcript(r["session_id"]))
                extra = {"archive": meta, "hook_note_in_transcript": "is archived as page images" in tr,
                         "hook_errors": (proj / ".claude" / "compact_archive" / "errors.log").exists()}
                st["archive"] = meta
            comp_cost = (r.get("cli_session_total") or 0) - (st.get(f"{arm}_fork_total") or 0)
            save()
            log_row({"session": name, "kind": kind, "arm": arm, "step": "compact", **keep(r),
                     "cost": round(comp_cost, 4), "usage_cost": r["cost"], **extra})
        for t, tq in enumerate(turns + [task]):
            step = f"turn{t + 1}" if t < len(turns) else "task"
            if st.get(f"{arm}_{step}"):
                continue
            prompt = (TASK_PROMPT if step == "task" else Q_PROMPT).format(
                questions="\n".join(f"{q['id']}: {q['question']}" for q in tq))
            if arm == "D" and sk not in st:
                args = ["--resume", st["read_sid"], "--fork-session"]
            else:
                args = ["--resume", st[sk]]
            (proj / "handoff.md").unlink(missing_ok=True)
            r = RC.claude(args, prompt, proj, "Read,Write" if step == "task" else "Read", "high", timeout=3600)
            if failed(r):
                print(f"[scale] {name} {arm} {step} FAILED: {r.get('result', '')[:300]}", flush=True)
                return
            st.setdefault(sk, r["session_id"])
            text = r.get("result", "")
            if step == "task":
                hf = proj / "handoff.md"
                text = hf.read_text() if hf.exists() else text
            sc = score(tq, text)
            st[f"{arm}_{step}"] = True
            save()
            log_row({"session": name, "kind": kind, "arm": arm, "step": step, **keep(r), "scores": sc,
                     "archive_reads": sum("compact_archive" in p for p in r["read_paths"]),
                     "transcript_reads": sum(p.endswith(".jsonl") for p in r["read_paths"])})
        if arm.startswith("B"):
            shutil.rmtree(proj / ".claude", ignore_errors=True)       # B's hooks and archive leave with B
        elif arm == "A":
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["research:0", "code:0", "logs:0"])
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=ARMS, help="Bfix: arm B again with the fixed PreCompact hook "
                    "(post-hoc, 2026-11-11: files the minifier refuses lost their line numbers)")
    a = ap.parse_args()
    names = [f"{k}:{i}" for i in range(3) for k in ("research", "code", "logs")] if a.sessions == ["all"] else a.sessions
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        list(ex.map(lambda n: run_session(n, a), names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
