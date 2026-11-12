"""Pure functions of experiments/plan_quality.py: brief parsing and validation, existence checks, file / function
scores, fix size, rank correlation, tiers, policies, collisions, judge parsing, groups."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import plan_quality as P  # noqa: E402

BRIEF = """TICKET: T1
SUMMARY: Fix the resolver so it honours the cache.
FILES: src/pkg/resolver.py : check the cache before fetching
FILES: src/pkg/cache.py (new) : a small cache module
FILES: tests/test_resolver.py : add a test
FUNCTIONS: Resolver.resolve, fetch_url, make_cache (new)
APPROACH: Look up the cache first. Fall back to fetch_url. Store the result.
RISKS: cache invalidation; opus chosen because the change crosses modules
COLLIDES_WITH: none
ORDER: none
SIZE: M
WORKER: sonnet / medium
CONFIDENCE: high"""


def diff(*paths, body="-x\n+y\n"):
    return "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@ def foo():\n{body}" for p in paths)


def test_parse_brief_fields():
    b = P.parse_brief(BRIEF)
    assert b["ticket"] == "T1"
    assert [f["path"] for f in b["files"]] == ["src/pkg/resolver.py", "src/pkg/cache.py", "tests/test_resolver.py"]
    assert [f["new"] for f in b["files"]] == [False, True, False]
    assert [f["name"] for f in b["functions"]] == ["Resolver.resolve", "fetch_url", "make_cache"]
    assert [f["new"] for f in b["functions"]] == [False, False, True]
    assert b["size"] == "M" and b["worker"] == {"model": "sonnet", "effort": "medium"} and b["confidence"] == "high"
    assert b["collides"] == [] and b["order"] == []
    assert b["problems"] == []


def test_parse_brief_tolerates_markdown_and_continuation_lines():
    text = ("**TICKET:** T2\n**SUMMARY:** x\nFILES:\n- `a/b.py` : one\n- c/d.py: two\n**FUNCTIONS:** `f()`, `G.h`\n"
            "APPROACH: a.\nRISKS: none\nCOLLIDES_WITH: T1, T3\nORDER: T1\nSIZE: L\nWORKER: Opus / High\nCONFIDENCE: Low")
    b = P.parse_brief(text)
    assert [f["path"] for f in b["files"]] == ["a/b.py", "c/d.py"]
    assert [f["name"] for f in b["functions"]] == ["f", "G.h"]
    assert b["collides"] == ["T1", "T3"] and b["order"] == ["T1"]
    assert b["size"] == "L" and b["worker"] == {"model": "opus", "effort": "high"} and b["confidence"] == "low"


def test_validate_brief_flags_problems():
    b = P.parse_brief("TICKET: T1\nSUMMARY: x\nSIZE: XL\nWORKER: gpt / high\n")
    assert "missing FILES" in b["problems"] and "bad SIZE" in b["problems"] and "bad WORKER" in b["problems"]
    many = "TICKET: T1\n" + "".join(f"FILES: p{i}.py : x\n" for i in range(7))
    assert "more than 6 files" in P.parse_brief(many)["problems"]


def test_split_briefs():
    text = "TICKET: T1\nSUMMARY: a\n\nTICKET: T2\nSUMMARY: b\n\n**TICKET:** T3\nSUMMARY: c"
    parts = P.split_briefs(text)
    assert len(parts) == 3 and [P.parse_brief(p)["ticket"] for p in parts] == ["T1", "T2", "T3"]
    assert P.split_briefs("") == []


def test_existence_checks():
    paths = {"src/pkg/resolver.py", "src/pkg/util.py"}
    assert P.path_exists("src/pkg/resolver.py", paths) and P.path_exists("src/pkg", paths)
    assert not P.path_exists("src/pkg/resolve.py", paths)
    words = {"Resolver", "resolve", "fetch_url"}
    assert P.symbol_exists("Resolver.resolve", words) and not P.symbol_exists("Resolver.lookup", words)
    b = P.parse_brief(BRIEF)
    inv = P.invented(b, paths, words)
    # cache.py is marked new; tests/test_resolver.py does not exist and is not marked new; make_cache is new
    assert inv["paths_missing"] == ["tests/test_resolver.py"] and inv["symbols_missing"] == []
    assert inv["named"] == 6 and inv["missing"] == 1


def test_file_and_function_scores():
    b = P.parse_brief(BRIEF)
    s = P.file_scores(b, gold=["src/pkg/resolver.py", "src/pkg/other.py"], gold_all=["src/pkg/resolver.py",
                      "src/pkg/other.py", "src/pkg/cache.py"])
    assert s["gold_share"] == 0.5 and s["gold_hit"] == ["src/pkg/resolver.py"]
    assert s["new_named"] == ["src/pkg/cache.py"] and s["wrong"] == [] and s["named_src"] == 2
    assert s["non_source_named"] == ["tests/test_resolver.py"] and s["first_right"]
    s2 = P.file_scores(b, gold=["src/x.py"], gold_all=["src/x.py"])
    assert s2["gold_share"] == 0.0 and s2["wrong_share"] == 1.0
    f = P.func_score(b, ["resolve", "fetch_url", "store"])
    assert f["funcs_named"] == ["resolve", "fetch_url"] and abs(f["funcs_share"] - 2 / 3) < 1e-9


def test_fix_size_counts_source_lines_only():
    p = diff("pkg/a.py", "pkg/b.py", body="-x\n+y\n+z\n") + diff("tests/test_a.py", body="+t\n+t\n+t\n+t\n")
    s = P.fix_size(p)
    assert s["files"] == 2 and s["lines"] == 6 and s["functions"] == 1


def test_spearman_and_ranks():
    assert P.ranks([3, 1, 1, 2]) == [4, 1.5, 1.5, 3]
    assert abs(P.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
    assert abs(P.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-9
    assert P.spearman([1, 1, 1], [1, 2, 3]) is None
    assert P.spearman([1, 2], [1, 2]) is None
    comp = P.composite_size([{"files": 3, "functions": 1, "lines": 10}, {"files": 5, "functions": 4, "lines": 90}])
    assert comp == [1.0, 2.0]


def test_size_class_and_tiers():
    assert [P.real_size_class(n) for n in (1, 2, 3, 4, 9)] == ["S", "M", "M", "L", "L"]
    assert P.measured_tier({"haiku": False, "sonnet": True, "opus": True}) == "sonnet"
    assert P.measured_tier({"haiku": False, "sonnet": False, "opus": False}) == "none"
    assert P.tier_verdict("opus", "sonnet") == "too large"
    assert P.tier_verdict("haiku", "opus") == "too small"
    assert P.tier_verdict("sonnet", "sonnet") == "right"
    assert P.tier_verdict("sonnet", "none") == "unresolvable"
    assert P.tier_verdict(None, "opus") == "no choice"


def test_policies():
    run = lambda r, c: {"resolved": r, "cost_usd": c}  # noqa: E731
    rows = [{"planned": "haiku", "runs": {"haiku": run(True, 0.1), "sonnet": run(True, 0.3), "opus": run(True, 1.0)}},
            {"planned": "opus", "runs": {"haiku": run(False, 0.1), "sonnet": run(False, 0.3), "opus": run(True, 1.0)}},
            {"planned": None, "runs": {"haiku": run(False, 0.1), "sonnet": run(False, 0.3), "opus": run(False, 1.0)}}]
    p = P.policies(rows)
    assert p["always opus high"] == {"tickets": 3, "resolved": 2, "cost_usd": 3.0}
    assert p["always haiku low"] == {"tickets": 3, "resolved": 1, "cost_usd": 0.3}
    assert p["planner's choice"] == {"tickets": 2, "resolved": 2, "cost_usd": 1.1}


def test_collisions():
    tp = P.true_pairs({"T1": ["a.py", "b.py"], "T2": ["b.py"], "T3": ["c.py"]})
    assert tp == {frozenset(("T1", "T2"))}
    fl = P.flagged_pairs({"T1": ["T3"], "T2": ["T1", "T9"], "T3": []})
    assert fl == {frozenset(("T1", "T3")), frozenset(("T1", "T2"))}
    assert P.collision_score(fl, tp) == {"true": 1, "flagged": 2, "found": 1, "wrong": 1, "missed": 0}


def test_parse_judge():
    j = P.parse_judge("CAUSE: yes\nPLACE: partly\nOUTCOME: no\nREASON: misses the generator.")
    assert j == {"cause": "yes", "place": "partly", "outcome": "no", "score": 3, "reason": "misses the generator."}
    assert P.parse_judge("**CAUSE:** Yes\n**PLACE:** yes\n**OUTCOME:** yes")["score"] == 6
    assert P.parse_judge("I cannot tell") is None


def test_brief_content_and_source_patch():
    c = P.brief_content(BRIEF)
    assert "TICKET" not in c and "SIZE" not in c and "WORKER" not in c and c.startswith("SUMMARY:")
    assert "FILES: src/pkg/resolver.py" in c and "RISKS:" in c
    p = diff("pkg/a.py") + diff("tests/test_a.py") + diff("CHANGELOG.md")
    sp = P.source_patch(p)
    assert "pkg/a.py" in sp and "tests/test_a.py" not in sp and "CHANGELOG" not in sp


def test_make_groups():
    t = lambda i, repo, d: {"id": i, "repo": repo, "created_at": d}  # noqa: E731
    ts = [t(f"s{k}", "o/sql", f"2025-0{k}") for k in range(1, 8)] + [t("a1", "o/a", "1"), t("a2", "o/a", "2")]
    gs = P.make_groups(ts)
    assert [[x["id"] for x in g] for g in gs] == [["s1", "s2", "s3"], ["s4", "s5", "s6", "s7"]]
    assert P.make_groups(ts[:5])[0][0]["id"] == "s1" and len(P.make_groups(ts[:5])) == 1


def test_prompts_keep_the_design_format():
    for line in ("SIZE: S | M | L", "WORKER: <haiku | sonnet | opus> / <low | medium | high>",
                 "FILES: <path> : <what changes>", "COLLIDES_WITH:"):
        assert line in P.BRIEF_FORMAT
    p = P.plan_prompt("CTX", [("T1", "issue one")])
    assert p.startswith("CTX\n<ticket id=\"T1\">") and P.SINGLE_RULE in p
    g = P.plan_prompt("CTX", [("T1", "a"), ("T2", "b"), ("T3", "c")])
    assert "There are 3 tickets" in g and "each ticket below" in g


def test_memory_slug():
    assert P.memory_slug("/tmp/a_b/c.d") == "-tmp-a-b-c-d"
