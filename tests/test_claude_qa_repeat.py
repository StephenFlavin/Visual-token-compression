"""Tests for experiments/claude_qa_repeat.py (no Claude calls, no rendering)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import claude_qa_repeat as Q  # noqa: E402


def test_salt_only_for_repeats():
    assert Q.salt(0) is None and Q.salt(1) == "repeat1" and Q.salt(1) != Q.salt(2)


def test_transcribe_kwargs_same_request_except_salt():
    spec = {"pages": [], "system": "S", "prompt": "P"}
    a, b = Q.transcribe_kwargs(spec, 0), Q.transcribe_kwargs(spec, 3)
    assert a.pop("cache_salt") is None and b.pop("cache_salt") == "repeat3"
    assert a == b and a["effort"] == "high" and a["hires"] is False


def test_todo_jobs_skips_done_and_counts_27():
    jobs = Q.todo_jobs([], [0, 1, 2])
    assert len(jobs) == 27
    done = [{"doc": 0, "arm": "page118", "repeat": 1}, {"doc": 2, "arm": "text", "repeat": 1}]
    left = Q.todo_jobs(done, [0, 1, 2])
    assert len(left) == 25 and (0, "page118", 1) not in left and (2, "text", 1) not in left


def test_boot_diff_point_and_interval():
    arm = {0: [1.0, 1.0, 0.875, 1.0, 1.0], 1: [0.875] * 5, 2: [1.0] * 5}
    ref = {0: [1.0, 1.0], 1: [1.0, 1.0], 2: [1.0, 1.0]}
    pt, lo, hi = Q.boot_diff(arm, ref, n=2000)
    assert pt == pytest.approx((sum(sum(v) for v in arm.values()) / 15) - 1.0)
    assert lo <= pt <= hi <= 0
    # document 1 always loses: the interval excludes 0
    assert hi < 0
    # no spread at all: a point interval
    pt, lo, hi = Q.boot_diff({0: [0.9, 0.9]}, {0: [1.0]}, n=100)
    assert lo == pytest.approx(-0.1) and hi == pytest.approx(-0.1)


def test_verdict_rules():
    assert Q.verdict(-0.05, -0.08, -0.01) == "loss"
    assert Q.verdict(-0.01, -0.019, 0.0) == "no loss shown"
    assert Q.verdict(-0.015, -0.03, -0.001) == "not settled"     # excludes 0 but mean loss < 2 points
    assert Q.verdict(-0.03, -0.06, 0.01) == "not settled"


def test_locate_lines_and_pages():
    lines = [["Alpha beta gamma ", "delta. The museum at X did "], ["not receive 5 maps. Other ", "text."]]
    v = Q.locate(lines, "The museum at X did not receive 5 maps.")
    assert v["found"] and v["page"] == 0 and v["line"] == 1 and v["split_page"] and v["end_page"] == 1
    v = Q.locate(lines, "Alpha beta gamma delta.")
    assert v["n_lines"] == 2 and v["top"] and not v["split_page"]
    assert Q.locate(lines, "absent sentence")["found"] is False
