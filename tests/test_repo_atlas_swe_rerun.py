"""Pure functions of experiments/atlas/repo_atlas_swe_rerun.py: run order, neutral dirs, leak scan, memory store, stats."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "atlas")]

import repo_atlas_swe_rerun as R  # noqa: E402


def test_run_order_all_first_runs_before_second_and_fixed():
    ids = [f"t{i}" for i in range(16)]
    o = R.run_order(ids)
    assert len(o) == 64 and len(set(o)) == 64
    assert all(j[2] == 1 for j in o[:32]) and all(j[2] == 2 for j in o[32:])
    assert o == R.run_order(ids)                       # fixed seed
    assert o[:32] != sorted(o[:32])                   # shuffled
    assert [j[:2] for j in o[:32]] != [j[:2] for j in o[32:]]   # second runs in another order


def test_smoke_first_puts_one_job_per_arm_in_front_and_keeps_all():
    o = R.run_order([f"t{i}" for i in range(5)])
    s = R.smoke_first(o)
    assert sorted(s) == sorted(o)
    assert {s[0][1], s[1][1]} == {"plain", "atlas"} and s[0][2] == s[1][2] == 1


def test_run_dir_tag_neutral_and_distinct():
    tags = {R.run_dir_tag("org__pkg-1", a, k) for a in R.ARMS for k in R.REPS}
    assert len(tags) == 4
    for t in tags:
        assert "plain" not in t and "atlas" not in t and "pkg" not in t


def test_memory_slug_matches_claude_code():
    assert R.memory_slug("/tmp/a_b/c.d") == "-tmp-a-b-c-d"


def test_memory_check_and_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cwd = Path("/x/run/repo")
    store = tmp_path / ".claude" / "projects" / R.memory_slug(str(cwd))
    assert R.memory_check_and_clear(cwd) == {"project_dir": False, "memory_files": [], "files": []}
    (store / "memory").mkdir(parents=True)
    (store / "memory" / "MEMORY.md").write_text("note")
    info = R.memory_check_and_clear(cwd)
    assert info["project_dir"] and info["memory_files"] == ["MEMORY.md"] and info["files"] == ["memory/MEMORY.md"]
    assert not store.exists()


def test_leak_hits_ignore_own_venv_but_flag_scratch_and_web():
    own = "/tmp/claude-0/x/scratchpad/swe_rerun/runs/abc123"
    cmds = [f"{own}/venv/bin/python -m pytest tests -q",                 # own venv: fine
            f"cd {own}/pkg && git log --oneline",                        # one-commit history: fine
            "ls /tmp/claude-0/x/scratchpad/ra_swe/transcripts",          # old transcripts: leak
            "cat /tmp/claude-0/x/scratchpad/swe_rerun/runs/zzz/pkg/a.py",  # another run's dir: leak
            "pip download pkg==2.0",
            "curl https://example.com",
            "cat ~/.claude/projects/foo/memory/MEMORY.md",
            "grep -r foo /home/user/visual-bpe"]
    hits = R.leak_hits(cmds, "pkg", own)
    assert cmds[0][:200] not in hits and cmds[1][:200] not in hits
    assert all(c[:200] in hits for c in cmds[2:])


def test_per_task_and_bootstrap():
    rows = [{"id": "a", "arm": arm, "rep": k, "resolved": r} for arm, k, r in
            [("plain", 1, True), ("plain", 2, False), ("atlas", 1, True), ("atlas", 2, True)]]
    rows += [{"id": "b", "arm": "plain", "rep": 1, "resolved": False}]
    pt = R.per_task(rows, ["a", "b"])
    assert pt == {("a", "plain"): [True, False], ("a", "atlas"): [True, True]}
    lo, hi = R.bootstrap_diff([2, 2, 2], [0, 0, 0])
    assert lo == hi == 6
    lo, hi = R.bootstrap_diff([1, 0, 2, 1], [1, 1, 1, 1])
    assert lo <= 0 <= hi


def test_leak_hits_ignore_cli_task_output_dir():
    base = Path("/tmp/claude-0/x/scratchpad/swe_rerun/runs/abc123")
    repo = base / "pkg"
    own = R.own_prefixes(base, repo)
    cli = "/tmp/claude-0/" + R.memory_slug(str(repo)) + "/sess/tasks/b1.output"
    assert R.leak_hits([f"tail -3 {cli}"], "pkg", own) == []
    assert R.leak_hits(["tail -3 /tmp/claude-0/-tmp-other-run/sess/tasks/b1.output"], "pkg", own)
