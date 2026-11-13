"""Pure parts of experiments/claude_qa/claude_qa_prune.py: small-word facts, keep thresholds, survival check, parsing."""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "claude_qa"))
import claude_qa_prune as P  # noqa: E402

NUM = re.compile(r"\d+")


def _names_factory(seed=0):
    rng = np.random.default_rng(seed)
    used = set()
    return rng, lambda: P.R.invented(rng, used)


def test_small_facts_have_answer_and_lure_and_critical_words():
    rng, names = _names_factory()
    for i in range(len(P.SMALL)):
        for _ in range(20):
            q = P.small_fact(i, rng, names)
            fact = q["anchors"][0]
            nums = NUM.findall(fact)
            assert len(nums) == 2, fact                      # one answer-side number and one lure-side number
            assert q["answer"] != q["lure"]
            assert int(q["answer"]) > 0 and int(q["lure"]) > 0
            words = [w.lower() for w in re.findall(r"\w+", fact)]
            assert all(c.lower() in words for c in q["critical"]), (fact, q["critical"])
            if q["sub"] in ("negation", "instead_of", "passive", "before_after", "from_to"):
                assert set(nums) == {q["answer"], q["lure"]}
            if q["sub"] in ("negated_count", "all_but"):
                total, k = map(int, nums)
                assert int(q["answer"]) == total - k and int(q["lure"]) == k


def test_small_answers_follow_the_wording():
    rng, names = _names_factory(1)
    for _ in range(30):
        q = P.small_fact(4, rng, names)                       # comparative
        n1, d = map(int, NUM.findall(q["anchors"][0]))
        assert int(q["answer"]) == (n1 + d if "fewer" in q["anchors"][0] else n1 - d)
        q = P.small_fact(6, rng, names)                       # from/to
        f = q["anchors"][0]
        m = re.match(r"(\d+) crates of slate were shipped from (\w+) to (\w+), and (\d+) crates", f)
        dest = q["question"].rsplit(" ", 1)[1].rstrip("?")
        assert q["answer"] == (m.group(1) if dest == m.group(3) else m.group(4))


def test_make_small_is_balanced_and_deterministic():
    a = P.make_small(40, 1200, 0, "")
    b = P.make_small(40, 1200, 0, "")
    assert [q for _, q in a] == [q for _, q in b]
    subs = [q["sub"] for _, q in a]
    assert all(subs.count(s) == 5 for s, *_ in P.SMALL)
    assert len({q["id"] for _, q in a}) == 40
    assert all(0 <= i < 1200 for i, _ in a)


def _chunks():
    return [{"words": ["▁The", "▁cat", "▁did", "▁not", "▁eat", "▁4", ",512", "▁fish", "."],
             "probs": [0.1, 0.9, 0.2, 0.3, 0.8, 1.0, 0.4, 0.95, 0.05], "w": [1] * 9}]


def test_keep_labels_rate_and_digits():
    ch = _chunks()
    assert all(P.keep_labels(ch, 1.0)[0])
    lab = P.keep_labels(ch, 0.5)[0]
    assert lab[5] and lab[6]                                  # a word with a digit is always kept
    assert not P.keep_labels(ch, 0.5, digit_words=False)[0][6]
    assert P.pruned_text(ch, [lab]) == "cat 4,512 fish"
    assert sum(P.keep_labels(ch, 0.3)[0]) <= sum(lab) <= sum(P.keep_labels(ch, 0.8)[0])


def test_assemble_and_survival():
    ch = _chunks()
    full, spans = P.assemble(ch)
    assert full == " The cat did not eat 4,512 fish."
    lab = P.keep_labels(ch, 0.5)
    _, sp = P.assemble(ch, lab)
    q = {"kind": "small", "critical": ["not"]}
    s = P.survival(full, sp, q, ["The cat did not eat 4,512 fish."])
    assert s["located"] and s["number"] and s["function"] is False and s["all"] is False
    s = P.survival(full, P.assemble(ch)[1], q, ["The cat did not eat 4,512 fish."])
    assert s["all"]
    assert P.survival(full, sp, q, ["no such sentence"])["located"] is False
    assert P.pruned_span(full, sp, "did not eat 4,512 fish") == "4,512 fish"


def test_prompt_note_and_parse():
    qs = [{"id": "W01", "question": "How many?"}]
    p = P.prompt("ptext22", "doc", qs, 0)
    assert P.NOTE in p and p.index(P.NOTE) < p.index("Questions:") and "<document>" in p
    assert P.NOTE not in P.prompt("text", "doc", qs, 0)
    assert P.NOTE in P.prompt("ppage15", "doc", qs, 3) and "3 page images" in P.prompt("ppage15", "doc", qs, 3)
    assert P.parse_answers("W01: 245\n**N02**: 1,200\nC03 - 7") == {"W01": "245", "N02": "1,200", "C03": "7"}
