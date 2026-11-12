"""Pure functions of experiments/repo_atlas_swe.py: selection rule, gold files, grading, scoring, first message."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import repo_atlas_swe as S  # noqa: E402


def diff(*paths):
    return "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-x\n+y\n" for p in paths)


def row(iid, created, paths, base="b1"):
    return {"instance_id": iid, "created_at": created, "patch": diff(*paths), "base_commit": base}


def test_gold_files_leave_out_tests_docs_changelog():
    p = diff("CHANGELOG.md", "pkg/a.py", "tests/test_a.py", "docs/x.rst", "pkg/b.py")
    assert S.gold_files(p) == ["pkg/a.py", "pkg/b.py"]


def test_candidate_order_newest_first_min_gold_and_dedup():
    rows = [row("old", "2026-01-01", ["a.py", "b.py", "c.py"], "x"),
            row("new", "2026-05-01", ["a.py", "b.py", "c.py"], "y"),
            row("two", "2026-06-01", ["a.py", "b.py"], "z"),
            row("dup", "2026-04-01", ["a.py", "b.py", "c.py"], "y")]   # same base and patch as "new"
    assert [r["instance_id"] for r in S.candidate_order(rows, 3)] == ["new", "old"]
    assert [r["instance_id"] for r in S.candidate_order(rows, 2)] == ["two", "new", "old"]


def test_parse_pytest_log_matches_swebench():
    log = ("PASSED tests/t.py::test_a\nFAILED tests/t.py::test_b - AssertionError: x\nERROR tests/t.py::test_c\n"
           "something PASSED tests/t.py::test_d\nXFAIL tests/t.py::test_e - reason\n")
    st = S.parse_pytest_log(log)
    assert st == {"tests/t.py::test_a": "PASSED", "tests/t.py::test_b": "FAILED", "tests/t.py::test_c": "ERROR",
                  "tests/t.py::test_e": "XFAIL"}


def test_grade_status_leaves_out_env_broken():
    st = {"f": "PASSED", "p1": "PASSED", "p2": "FAILED"}
    assert not S.grade_status(st, ["f"], ["p1", "p2"])["resolved"]
    g = S.grade_status(st, ["f"], ["p1", "p2"], {"p2"})
    assert g["resolved"] and g["p2p"] == "1/1"
    assert not S.grade_status({"f": "FAILED"}, ["f"], [])["resolved"]


def test_touched_share():
    assert S.touched_share(diff("a.py", "z.py"), ["a.py", "b.py"]) == 0.5
    assert S.touched_share("", ["a.py"]) == 0.0


def test_exclude_files():
    p = diff("a.py", "tests/t.py", "b.py")
    out = S.exclude_files(p, {"tests/t.py"})
    assert S.patch_paths(out) == ["a.py", "b.py"]


def test_screen_drop():
    assert S.screen_drop(["a.py", "b.py", "c.py"], ["b.py", "a.py"])
    assert not S.screen_drop(["a.py"], ["a.py", "b.py"])
    assert not S.screen_drop(["a.py", "x", "y", "z", "w", "b.py"], ["a.py", "b.py"])   # only the top 5 count
    assert not S.screen_drop(["a.py"], [])


def test_install_supported():
    assert S.install_supported('pip install -e ".[test]" -q')
    assert not S.install_supported("apt-get update && apt-get install -y x && pip install -e .")
    assert not S.install_supported("conda install -y gmt && pip install -e .")
    assert not S.install_supported("pip install uv && uv sync")


def test_first_message_arms_share_the_task_and_differ_only_in_context():
    kw = dict(repo="o/r", venv="/v", issue="Bug X")
    plain = S.first_message("plain", **kw)
    m = S.first_message("map", **kw, map_text="a.py\n  def f()\n", n_files=3)
    a = S.first_message("atlas", **kw, atlas_index="a.py: page 1\n", n_files=3)
    assert plain.startswith("You are working") and "<issue>\nBug X\n</issue>" in plain
    assert m.endswith(plain) and a.endswith(plain)
    assert "def f()" in m and "def f()" not in a
    assert S.ATLAS_SENTENCE in a and "a.py: page 1" in a and S.ATLAS_SENTENCE not in m


def test_exact_tests():
    assert S.binom_two_sided(0, 0) == 1.0
    assert abs(S.binom_two_sided(0, 5) - 0.0625) < 1e-9
    assert abs(S.binom_two_sided(2, 4) - 1.0) < 1e-9
    m = S.mcnemar([True, True, False, False], [True, False, False, False])
    assert m["only_a"] == 1 and m["only_b"] == 0 and m["p"] == 1.0
    s = S.paired_sign([1, 2, 3, 4], [2, 2, 5, 9])
    assert s["n"] == 3 and s["a_lower"] == 3 and abs(s["p"] - 0.25) < 1e-9


def test_leak_hits():
    cmds = ["/v/bin/python -m pytest tests", "pip install -U schemathesis", "pip install -e .", "git log --oneline",
            "curl https://example.com", "cat ../../ra_swe/ds/x", "git fetch origin",
            "V=/s/ra_swe/runs/t/68c46e/venv/bin/python; $V -m pytest tests"]
    hits = S.leak_hits(cmds, "schemathesis")
    assert hits == ["pip install -U schemathesis", "curl https://example.com", "cat ../../ra_swe/ds/x", "git fetch origin"]
