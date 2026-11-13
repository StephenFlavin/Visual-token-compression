"""repo_atlas_qa.py: categories, batching, answer parsing, entity extraction and recall, judge parsing. No network."""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "atlas"))
import repo_atlas_qa as Q  # noqa: E402


def test_category():
    assert Q.category("What is the purpose of X?") == "what"
    assert Q.category("In the session class, how does merging work?") == "how"
    assert Q.category("At which point, where is it called?") == "where"
    assert Q.category("Why does it fail?") == "why"
    assert Q.category("Explain it.") == "other"


def test_batches_balanced_fixed_and_complete():
    by = {c: [f"Q{i}" for i in range(k * 12 + 1, k * 12 + 13)] for k, c in enumerate(Q.CATS)}
    b1, b2 = Q.make_batches(by), Q.make_batches(by)
    assert b1 == b2
    assert [len(b) for b in b1] == [8] * 6
    flat = [q for b in b1 for q in b]
    assert sorted(flat) == sorted(q for v in by.values() for q in v)
    cat = {q: c for c, v in by.items() for q in v}
    for b in b1:
        assert set(Counter(cat[q] for q in b).values()) == {2}
    assert Q.make_batches(by, seed=1) != b1


def test_batches_uneven_categories():
    by = {"what": ["Q1", "Q2", "Q3"], "why": ["Q4"], "how": ["Q5", "Q6"]}
    b = Q.make_batches(by, size=4)
    assert sorted(q for x in b for q in x) == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


def test_parse_answers_formats():
    text = ("Q3: First answer\nspans two lines.\n\n**Q10:** Second `foo.py`.\n- Q7: third\nQ1. fourth\n"
            "Not an id line: Q3 is mentioned here.")
    got = Q.parse_answers(text, ["Q3", "Q10", "Q7", "Q1", "Q2"])
    assert got["Q3"] == "First answer\nspans two lines."
    assert got["Q10"] == "Second `foo.py`."
    assert got["Q7"] == "third"
    assert got["Q1"].startswith("fourth")
    assert "Q2" not in got


def test_parse_answers_does_not_confuse_q1_and_q10():
    got = Q.parse_answers("Q10: ten\nQ1: one", ["Q1", "Q10"])
    assert got == {"Q10": "ten", "Q1": "one"}


def test_parse_answers_empty():
    assert Q.parse_answers("", ["Q1"]) == {}
    assert Q.parse_answers("I cannot answer.", ["Q1"]) == {}


REF = ("The flag `_got_registered_once` in `src/flask/sansio/blueprints.py` is set by `Blueprint.register`. "
       "`self.deferred_functions` collects callbacks; `_check_setup_finished` raises `AssertionError`. "
       "See also blueprints.py and the DefaultJSONProvider in json/provider.py, e.g. via app.json.dumps(). "
       "The `json` module and `True` are plain words.\n```python\nsecret_inner_name = 1\n```")


def test_extract_entities():
    e = Q.extract_entities(REF)
    for x in ("_got_registered_once", "src/flask/sansio/blueprints.py", "Blueprint.register", "deferred_functions",
              "_check_setup_finished", "AssertionError", "DefaultJSONProvider", "json/provider.py", "app.json.dumps"):
        assert x in e, x
    assert "blueprints.py" not in e                 # a shorter form of the full path
    assert "self.deferred_functions" not in e
    assert "json" not in e and "True" not in e and "e.g" not in e
    assert "secret_inner_name" not in e             # code blocks are skipped
    assert len(e) == len(set(e))


def test_entity_variants_and_recall():
    assert "requests/utils.py" in Q.entity_variants("src/requests/utils.py")
    assert "utils.py" in Q.entity_variants("src/requests/utils.py")
    assert "Session.merge_environment_settings" in Q.entity_variants("requests.Session.merge_environment_settings")
    assert "merge_environment_settings" in Q.entity_variants("requests.Session.merge_environment_settings")
    assert "Session" not in Q.entity_variants("requests.Session")
    ents = ["src/requests/utils.py", "get_environ_proxies", "Session.request", "TestThing"]
    ans = "In requests/utils.py, get_environ_proxies calls should_bypass_proxies; Session.request merges."
    assert Q.entity_recall(ents, ans) == 0.75
    assert Q.entity_recall(ents, "get_environ_proxies_x") == 0.0   # whole tokens only
    assert Q.entity_recall([], "x") is None


def test_cross_file():
    assert Q.cross_file("see src/a/x.py and src/a/y.py")
    assert not Q.cross_file("see src/a/x.py and again x.py, plus docs/index.rst")


def test_parse_judge():
    s = Q.parse_judge('```json\n{"correctness": 12, "completeness": 10, "relevance": 18, "clarity": 17, "reasoning": 14}\n```')
    assert s["total"] == 71 and s["correctness"] == 12
    assert Q.parse_judge('{"correctness": 0, "completeness": 10, "relevance": 18, "clarity": 17, "reasoning": 14}') is None
    assert Q.parse_judge('{"correctness": 12}') is None
    assert Q.parse_judge("no json") is None


def test_judge_prompt_is_the_benchmarks():
    assert "You are a STRICT and RIGOROUS evaluator" in Q.JUDGE_SYSTEM
    assert '"reasoning": <1-20>' in Q.JUDGE_SYSTEM
    assert "{question}" in Q.JUDGE_INPUT and "{candidate}" in Q.JUDGE_INPUT


def test_bootstrap_ci():
    m, lo, hi = Q.bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and lo < m < hi
    m, lo, hi = Q.bootstrap_ci([5.0] * 10)
    assert lo == hi == 5.0


def test_read_commits(tmp_path):
    (tmp_path / "repo_commit.txt").write_text("https://github.com/psf/requests 46e939b\n\n"
                                              "https://github.com/streamlink/streamlink ab1f365")
    assert Q.read_commits(tmp_path) == {"requests": ("psf/requests", "46e939b"),
                                        "streamlink": ("streamlink/streamlink", "ab1f365")}


def test_is_test_entity():
    assert Q.is_test_entity("tests/test_utils.py") and Q.is_test_entity("test_guess_by_bom")
    assert Q.is_test_entity("TestGuessJSONUTF.test_encoded") and Q.is_test_entity("conftest.py")
    assert not Q.is_test_entity("src/requests/utils.py") and not Q.is_test_entity("_pytest.python.Metafunc")


def test_limit_hit_sets_stop():
    Q._stop.clear()
    assert not Q.limit_hit("rc=1 api_error 500")
    assert not Q._stop.is_set()
    assert Q.limit_hit("rc=1 You've hit your 5-hour limit · resets 9:10am (UTC)")
    assert Q._stop.is_set()
    Q._stop.clear()
