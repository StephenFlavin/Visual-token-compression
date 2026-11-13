"""Pure functions of experiments/atlas/repo_atlas_swe_sonnet.py (Q3): isolation from Q1, order, bound, gate questions and
scoring. No network, no Claude calls."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "atlas")]

import repo_atlas_swe as sw  # noqa: E402
import repo_atlas_swe_rerun as rr  # noqa: E402
import repo_atlas_swe_sonnet as S  # noqa: E402


def test_own_dirs_and_tags_never_q1s():
    assert S.DEFAULT_WORK != rr.DEFAULT_WORK and S.OUT != rr.OUT
    assert S.DEFAULT_WORK.name == "swe_sonnet" and S.OUT.name == "swe_sonnet"
    for a in S.ARMS:
        for k in S.REPS:
            assert S.run_dir_tag("org__pkg-1", a, k) != rr.run_dir_tag("org__pkg-1", a, k)
    tags = {S.run_dir_tag("org__pkg-1", a, k) for a in S.ARMS for k in S.REPS}
    assert len(tags) == 4 and not any("plain" in t or "atlas" in t or "pkg" in t for t in tags)


def test_patch_rr_points_q1_module_here(monkeypatch):
    for mod, name in ((sw, "MODEL"), (sw, "EFFORT"), (rr, "OUT"), (rr, "run_dir_tag")):
        monkeypatch.setattr(mod, name, getattr(mod, name))       # restored after the test
    S.patch_rr()
    assert sw.MODEL == "claude-sonnet-5-5" and sw.EFFORT == "high"
    assert rr.OUT == S.OUT and rr.run_dir_tag is S.run_dir_tag


def test_order_fixed_first_runs_first_and_not_q1s():
    ids = [f"t{i}" for i in range(16)]
    o = S.run_order(ids)
    assert len(set(o)) == 64 and o == S.run_order(ids)
    assert all(j[2] == 1 for j in o[:32]) and all(j[2] == 2 for j in o[32:])
    assert o != rr.run_order(ids)


def test_bound():
    assert S.bound("plain", []) == 1.0 and S.bound("atlas", []) == 2.0
    assert S.bound("atlas", [0.5, 2.0]) == 3.0


SRC = '''"""Doc."""
LIMIT = 0o777
NAME = "x"


def alpha(a, b=1, *rest, c, **kw):
    return a


class Beta:
    def gamma(self, x):
        return x
'''


def test_gate_candidates_answers_from_source():
    flow = "§1 Doc § LIMIT = 0o777 § NAME = 'x' § def alpha(a, b=1, *rest, c, **kw): § »return a § class Beta: §11 »def gamma(self, x): § »»return x"
    c = S.gate_candidates("m.py", SRC, flow)
    after = {x["line"]: x["answer"] for x in c if x["kind"] == "after_line"}
    assert after == {1: "alpha", 11: "gamma"}                        # first def/class at or after the tagged line
    const = {x["name"]: x for x in c if x["kind"] == "constant"}
    assert const["LIMIT"]["answer"] == "0o777" and const["LIMIT"]["value"] == 511 and const["NAME"]["value"] == "x"
    params = {x["name"]: x["answer"] for x in c if x["kind"] == "params"}
    assert params == {"alpha": "a, b, rest, c, kw", "gamma": "self, x"}


def test_page_of_anchor_and_pick_spread():
    flow = "▶a.py◀ §1 def f(x): § »return x ▶b.py◀ §1 X = 1 § def g(y, z): § »pass"
    chunks = ["▶a.py◀ §1 def f(x): § »return x", "▶b.py◀ §1 X = 1 §", "def g(y, z): § »pass"]
    assert S.page_of_anchor(flow, chunks, "a.py", "def f(") == 1
    assert S.page_of_anchor(flow, chunks, "b.py", "def g(") == 3
    assert S.page_of_anchor(flow, chunks, "a.py", "def g(") is None          # not in that file
    cands = [{"kind": k, "path": f"p{i}.py", "page": p} for i, (k, p) in
             enumerate([("params", 1), ("params", 2), ("after_line", 5), ("constant", 9), ("params", 9)])]
    got = S.pick_questions(cands, 9, 3, 1)
    assert {g["third"] for g in got} == {"early", "middle", "late"}
    assert got == S.pick_questions(cands, 9, 3, 1)


def test_score_and_parse():
    assert S.score({"kind": "after_line", "answer": "gamma"}, " `gamma` ")
    assert S.score({"kind": "after_line", "answer": "gamma"}, "gamma()")
    assert not S.score({"kind": "after_line", "answer": "gamma"}, "alpha")
    assert S.score({"kind": "constant", "answer": "0o777", "value": 511}, "511")
    assert S.score({"kind": "constant", "answer": "0o777", "value": 511}, "0o777")
    assert S.score({"kind": "constant", "answer": '"x"', "value": "x"}, "'x'")
    assert S.score({"kind": "params", "answer": "self, rows, args, kwargs"}, "self, rows, *args, **kwargs")
    assert not S.score({"kind": "params", "answer": "self, x"}, "x, self")
    a = S.parse_answers("Q1: alpha\n**Q2**: 3\nQ3) self, x\nnoise\nQ9: no", 3)
    assert a == {1: "alpha", 2: "3", 3: "self, x"}


def test_gate_summary_rule():
    rows = [{"ok": True, "model": "m", "label": "88.2", "correct": 8, "n": 10, "pages_arrived": True},
            {"ok": True, "model": "m", "label": "88.2", "correct": 8, "n": 10, "pages_arrived": True},
            {"ok": True, "model": "n", "label": "88.2", "correct": 10, "n": 10, "pages_arrived": True}]
    g = S.gate_summary(rows)
    assert g[("m", "88.2")]["passes"] and g[("m", "88.2")]["share"] == 0.8
    assert not g[("n", "88.2")]["passes"]                               # one atlas only
