"""claude_qa_salient: the rule-based detector, the two-size renderer, token matching and scoring. No network."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "claude_qa"))
import claude_qa_salient as S  # noqa: E402
from vtc import metrics  # noqa: E402


def marked(doc: str) -> list[str]:
    spans, why = S.detect_words(doc)
    return [doc[s:e] for (s, e), w in zip(spans, why) if w]


def reasons(doc: str) -> dict:
    spans, why = S.detect_words(doc)
    return {doc[s:e]: w for (s, e), w in zip(spans, why)}


# --------------------------------------------------------------------------- detector
def test_numbers_in_any_form_and_padding():
    doc = ("the cost rose by 12.5 % to $1,200 in the 1990s . it came third of seven and sold "
           "twenty-five copies on the 11th day")
    r = reasons(doc)
    for w in ("12.5", "%", "$1,200", "1990s", "third", "seven", "twenty-five", "11th"):
        assert "number" in r[w], w
    # one word of padding on each side of a number, not across a sentence end
    assert "number_pad" in r["sold"] and "number_pad" in r["copies"] and "number_pad" in r["day"]
    assert not r["it"] and not r["cost"] and not r["rose"]


def test_padding_stops_at_sentence_end():
    r = reasons("he left early. 42 people stayed. then rain")
    assert "number" in r["42"] and "number_pad" in r["people"]
    assert not r["early."] and not r["then"]


def test_names_mid_sentence_and_recurring_sentence_starts():
    doc = ("Boulter starred in a play by Simon Stephens. The play was long. Later the critic praised Boulter. "
           "Often it rained.")
    r = reasons(doc)
    assert "name" in r["Simon"] and "name" in r["Stephens."]
    assert "name" in r["Boulter"]              # sentence start, but capitalised mid-sentence elsewhere, never lower case
    assert not r["The"] and not r["Later"] and not r["Often"]       # sentence starts that are not names
    assert not r["play"] and not r["critic"]


def test_sentence_start_word_seen_in_lower_case_is_not_a_name():
    doc = "During the war he left. It happened during the Siege of Paris."
    r = reasons(doc)
    assert not r["During"]                     # "during" occurs in lower case
    assert "name" in r["Siege"] and "name" in r["Paris."]


def test_flip_words_and_phrases():
    doc = ("it did not arrive and never left ; all but 37 died ; at least one ; more than half ; fewer stayed "
           "before and after ; instead of the plan ; he wasn't there ; only one")
    r = reasons(doc)
    for w in ("not", "never", "all", "but", "at", "least", "more", "than", "fewer", "before", "after", "instead",
              "of", "wasn't", "only"):
        assert "flip" in r[w], w
    assert not r["arrive"] and not r["left"] and not r["the"]


def test_by_from_to_only_after_a_transfer_verb():
    doc = ("He was paid 40 crowns by Anna. The road runs from the hill to the sea. "
           "The crates were shipped from the port to the town.")
    r = reasons(doc)
    assert "flip" in r["by"]
    spans, why = S.detect_words(doc)
    words = [(doc[s:e], w) for (s, e), w in zip(spans, why)]
    froms = [w for t, w in words if t == "from"]
    tos = [w for t, w in words if t == "to"]
    assert froms[0] == set() and tos[0] == set()          # "runs from ... to": no transfer verb
    assert "flip" in froms[1] and "flip" in tos[1]        # "shipped from ... to"


def test_plain_filler_is_not_salient():
    doc = "the river was wide and the water was cold, so they waited on the bank for a while."
    assert marked(doc) == []
    assert not S.salient_mask(doc).any()


def test_mask_is_whole_words_and_joins_salient_neighbours():
    doc = "the archive of Brykel holds 516 letters written by Maren Quentor."
    m = S.salient_mask(doc)
    assert all(m[doc.find("Brykel"):doc.find("Brykel") + 6])
    s, e = doc.find("Brykel holds 516 letters"), doc.find("Brykel holds 516 letters") + len("Brykel holds 516 letters")
    assert m[s:e].all()                                   # spaces between salient words are salient
    assert not m[doc.find("archive")] and not m[doc.find(" written") ]


def test_coverage():
    doc = "a b 123 c"
    m = S.salient_mask(doc)
    assert S.coverage(doc, m, ["123"]) == 1.0
    assert S.coverage(doc, m, ["a"]) == 0.0


# --------------------------------------------------------------------------- renderer
def toy(n: int = 4000) -> str:
    rng = np.random.default_rng(0)
    words = ["river", "wide", "water", "cold", "bank", "while", "Anna", "Paris", "1,234", "not", "the", "and"]
    return " ".join(rng.choice(words, n))


def test_layout_keeps_order_and_both_sizes():
    doc = toy()
    f = S.Flow(doc, S.salient_mask(doc))
    pages = S.layout(f, 13.76, 5.0)
    idx = [j for p in pages for a, b, _, _ in p["lines"] for j in range(a, b)]
    assert idx == list(range(len(f.words)))               # every word once, in reading order across pages
    ems = {round(m, 2) for p in pages for _, _, m, _ in p["lines"]}
    assert 13.76 in ems                                   # lines with a salient word take the large pitch
    _, wems, _ = f.widths(13.76, 5.0)
    assert set(wems) == {13.76, 5.0}
    for p in pages:
        assert p["H"] % 28 == 0 and p["H"] <= 1932
        tops = [base - 0.9 * m for _, _, m, base in p["lines"]]
        for (a, b, m, base), t1 in zip(p["lines"][:-1], tops[1:]):
            assert t1 == pytest.approx(base - 0.9 * m + 1.2 * m)      # pitch = 1.2 x the line's largest em
        a, b, m, base = p["lines"][-1]
        assert base + 0.25 * m + S.MARGIN <= p["H"]


def test_lines_fit_the_width():
    doc = toy()
    f = S.Flow(doc, S.salient_mask(doc))
    ww, _, sp = f.widths(13.76, 5.0)
    for p in S.layout(f, 13.76, 5.0):
        for a, b, _, _ in p["lines"]:
            assert sum(ww[a:b]) + sum(sp[a:b - 1]) <= S.PAGE_W - 2 * S.MARGIN + 1e-6


def test_tokens_formula():
    pages = [{"H": 1932, "lines": []}, {"H": 560, "lines": []}]
    assert S.tokens_of(pages) == metrics.vision_tokens(1932, 1932) + metrics.vision_tokens(1932, 560)


def test_solve_em_matches_target_within_tolerance():
    doc = toy(30000)
    f = S.Flow(doc, S.salient_mask(doc))
    target = S.tokens_of(S.layout(f, 13.76, 13.76)) // 2
    em = S.solve_em(f, target, 13.76, "fill")
    tok = S.tokens_of(S.layout(f, 13.76, em))
    assert tok <= target and tok >= (1 - S.TOL) * target
    assert S.tokens_of(S.layout(f, 13.76, em + 0.2)) > target


def test_draw_both_sizes_on_one_baseline():
    doc = "water water Paris water"
    f = S.Flow(doc, S.salient_mask(doc))
    pages = S.layout(f, 20.0, 8.0)
    img = S.draw(f, pages, 20.0, 8.0)[0]
    a = np.asarray(img) < 128
    cols = np.where(a.any(0))[0]
    ink_rows = lambda x0, x1: np.where(a[:, x0:x1].any(1))[0]      # noqa: E731
    _, ems, sp = f.widths(20.0, 8.0)
    ww = f.widths(20.0, 8.0)[0]
    x_paris = S.MARGIN + ww[0] + sp[0] + ww[1] + sp[1]
    small = ink_rows(S.MARGIN, int(S.MARGIN + ww[0]))
    big = ink_rows(int(x_paris), int(x_paris + ww[2]))
    assert big.max() - big.min() > 1.8 * (small.max() - small.min())  # the salient word is drawn larger
    assert abs(int(big.max()) - int(small.max())) <= 1                # and on the same baseline
    assert len(cols)


# --------------------------------------------------------------------------- scoring
def spec_for(qs, vt=1000):
    return {"qs": qs, "arm": "salient_large", "d": 0, "em_sal": 13.76, "em_fill": 4.0, "doc": "x" * 10,
            "mask": np.zeros(10, bool), "layout": [{"H": 1932, "lines": []}], "vt": vt, "target": vt}


def qrow(i, kind, ans, sub=None):
    return {"id": i, "kind": kind, "sub": sub, "pos": 0.5, "answer": ans, "anchor_cov": 0.5,
            "num_cov": None if sub == "count" else 1.0}


def test_make_row_scores_and_flags():
    qs = [qrow("N01", "needle", "516"), qrow("S01", "number", "2000"), qrow("C01", "complex", "7", "count")]
    rec = {"text": "N01: 516\nS01: 1,999\nC01: 7", "input_tokens": 1200, "output_tokens": 10, "cost_usd": 0.5,
           "latency_s": 3.0, "cached": False, "error": None, "stop_reason": "end_turn"}
    r = S.make_row(spec_for(qs), 1, rec)
    assert [p["correct"] for p in r["per_question"]] == [True, False, True]
    assert r["cache_salt"] == "salient1" and not r["incomplete"] and r["unanswered"] == 0
    r2 = S.make_row(spec_for(qs, vt=2000), 0, {**rec, "input_tokens": 1500})
    assert r2["incomplete"]                               # pages lost: input tokens < 0.97 x the formula


def fake_row(arm, doc, rep, correct, kinds, subs=None, tokens=1000):
    subs = subs or {}
    per = [{"id": q, "kind": kinds[q], "sub": subs.get(q), "correct": c, "given": "1", "anchor_cov": 0.5,
            "num_cov": None if subs.get(q) == "count" else (1.0 if q.startswith("N") else 0.0)}
           for q, c in correct.items()]
    return {"arm": arm, "doc": doc, "repeat": rep, "per_question": per, "input_tokens": tokens, "cost_usd": 0.5,
            "acc_all": sum(correct.values()) / len(correct), "image_tokens_formula": tokens, "target_tokens": tokens,
            "output_tokens": 10, "latency_s": 1.0, "raw": "", "em_salient": 13.76, "em_filler": 4.0}


def test_analyse_predictions_and_tests():
    kinds = {**{f"N{i}": "needle" for i in range(10)}, **{f"C{i}": "complex" for i in range(10)}}
    subs = {f"C{i}": "count" for i in range(10)}
    allright = {q: True for q in kinds}
    u59 = {**allright, **{f"C{i}": False for i in range(4)}}               # 80%
    sl = {**allright, **{f"C{i}": False for i in range(2)}}                # 90%
    ss = {**allright, **{f"N{i}": False for i in range(8)}}                # 60%
    v2 = {0: {"text": fake_row("text", 0, 0, allright, kinds, subs), 117.6: fake_row("image", 0, 0, allright, kinds, subs),
              88.2: fake_row("image", 0, 0, allright, kinds, subs), 58.8: fake_row("image", 0, 0, u59, kinds, subs)}}
    rows = [fake_row("salient_large", 0, 0, sl, kinds, subs), fake_row("salient_small", 0, 0, ss, kinds, subs),
            {**fake_row("salient_small", 0, 1, allright, kinds, subs), "unanswered": 3}]
    res = S.analyse(rows, v2)
    assert res["invalid"] == [(0, "salient_small", 1)]
    p = res["predictions"]
    assert p["M1"]["diff"] == pytest.approx(0.10) and p["M1"]["pass"]
    assert p["M2"]["lookups"] == 1.0 and p["M2"]["multi_step_diff"] == pytest.approx(-0.2) and p["M2"]["pass"]
    assert p["M3"]["diff"] == pytest.approx(0.30) and p["M3"]["pass"]
    t = res["tests"]["salient_large vs uniform 58.8 (first_call)"]
    assert (t["n"], t["ref_only"], t["arm_only"]) == (20, 0, 2) and t["arm_minus_ref"] == pytest.approx(0.10)
    t = res["tests"]["salient_large vs salient_small (first_call)"]
    assert (t["arm_only"], t["ref_only"]) == (8, 2)
    w = res["arms"]["salient_small"]["wrong"]["needle"]
    assert w["wrong"] == 8 and w["number_salient"] == 8
    md = S.summary_md(res)
    assert "M1 PASS" in md and "M4: not run" in md


def test_analyse_small_word_arm():
    kinds = {"W01": "small", "W02": "small", "N01": "needle"}
    subs = {"W01": "passive", "W02": "negation"}
    base = {"page118": [(0, fake_row("page118", 0, 0, {"W01": True, "W02": True, "N01": True}, kinds, subs))],
            "page59": [(0, fake_row("page59", 0, 0, {"W01": False, "W02": True, "N01": True}, kinds, subs))]}
    for _, r in base["page118"] + base["page59"]:
        for p in r["per_question"]:
            p["lure"] = "9"
    sw = fake_row(S.SW_ARM, 0, 0, {"W01": False, "W02": True, "N01": True}, kinds, subs)
    v2 = {0: {}}
    res = S.analyse([sw], v2, base)
    x = res["small_words"]
    assert x["salient_large"]["small"]["acc"] == 0.5 and x["page118"]["small"]["acc"] == 1.0
    assert res["predictions"]["M4"]["pass"]
    t = res["tests"][f"{S.SW_ARM} vs Test A page118 (small-word questions)"]
    assert t["n"] == 2 and t["ref_only"] == 1


def test_jobs_and_salt():
    assert S.salt(0) is None and S.salt(1) == "salient1"
    jobs = S.todo_jobs([{"doc": 0, "arm": "salient_large", "repeat": 0}], [0, 1, 2])
    assert len(jobs) == 11 and jobs[0] == (0, "salient_small", 0)
    assert S.todo_jobs([], [0, 1, 2], small_words=True) == [(0, S.SW_ARM, 0), (1, S.SW_ARM, 0), (2, S.SW_ARM, 0)]
