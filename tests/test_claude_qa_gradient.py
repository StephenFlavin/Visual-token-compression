"""claude_qa_gradient: tier split, page order, token totals, scoring by tier. No network, no Claude calls."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "claude_qa"))
import claude_qa_gradient as G  # noqa: E402
from vbpe import metrics  # noqa: E402


def words(n: int) -> str:
    return " ".join(f"w{i:05d}" for i in range(n))


def test_split_thirds_equal_and_on_spaces():
    doc = words(3000)
    b = G.split_thirds(doc, [])
    assert b[0] == 0 and b[3] == len(doc) and b[0] < b[1] < b[2] < b[3]
    for k, c in ((1, b[1]), (2, b[2])):
        assert doc[c - 1] == " "                         # each third starts at a word
        assert abs(c - len(doc) * k / 3) <= 7            # within one word of the exact third


def test_split_never_cuts_a_fact():
    doc = words(3000)
    third = len(doc) // 3
    span = (third - 20, third + 50)                      # a fact straddling the first cut
    b = G.split_thirds(doc, [span])
    assert b[1] >= span[1]
    assert G.tier_of(span[0], b) == G.tier_of(span[1] - 1, b) == 0


def test_tier_of():
    b = [0, 10, 20, 30]
    assert [G.tier_of(i, b) for i in (0, 9, 10, 19, 20, 29)] == [0, 0, 1, 1, 2, 2]


def fake_render(seg: str, budget: float):
    """Pages of 1932 px whose count and height scale with chars x budget, like render_pages."""
    per_page = int(1932 * 1932 / budget)
    pages, info = [], []
    for i in range(0, len(seg), per_page):
        n = len(seg[i:i + per_page])
        H = min(1932, max(56, -(-int(n * budget / 1932) // 28) * 28))
        pages.append(Image.new("L", (1932, H), int(budget)))
        info.append({"W": 1932, "H": H, "chars": n, "vision_tokens": metrics.vision_tokens(1932, H),
                     "px_per_char": 1932 * H / n, "text": seg[i:i + per_page]})
    return pages, info


def test_pages_in_document_order_one_density_each():
    doc = words(60000)
    b = G.split_thirds(doc, [])
    for arm, budgets in G.ARMS.items():
        pages, info = G.render_tiered(doc, b, budgets, render=fake_render)
        assert [x["tier"] for x in info] == sorted(x["tier"] for x in info)          # old to new
        assert "".join(x["text"] for x in info) == doc                              # nothing lost or reordered
        for im, x in zip(pages, info):
            assert x["budget"] == budgets[x["tier"]] and im.getpixel((0, 0)) == int(x["budget"])


def test_mixed_tokens_match_uniform_88():
    doc = words(60000)
    b = G.split_thirds(doc, [])
    uni = sum(x["vision_tokens"] for x in fake_render(doc, 88.2)[1])
    for budgets in G.ARMS.values():
        tok = sum(x["vision_tokens"] for x in G.render_tiered(doc, b, budgets, render=fake_render)[1])
        assert abs(tok / uni - 1) < 0.03
    # the two mixed arms spend the same tokens per tier, mirrored
    t = {a: [sum(x["vision_tokens"] for x in G.render_tiered(doc, b, bs, render=fake_render)[1] if x["tier"] == k)
             for k in range(3)] for a, bs in G.ARMS.items()}
    assert t["recent_sharp"][0] < t["recent_sharp"][1] < t["recent_sharp"][2]
    assert t["old_sharp"][0] > t["old_sharp"][2]


def test_anchor_spans_number_question_covers_the_number():
    doc = "Alpha beta. The quick brown fox jumped over 1,234 lazy dogs in the yard today. Next one."
    q = {"id": "S01", "kind": "number", "answer": "1234", "anchors": ["The quick brown fox jumped over"]}
    (s, e), = G.anchor_spans(doc, q)
    assert doc[s:e].strip().endswith("today.") and "1,234" in doc[s:e]
    n = {"id": "N01", "kind": "needle", "fact": "Next one."}
    assert G.anchor_spans(doc, n) == [(doc.find("Next one."), len(doc))]


def row(arm, doc, rep, correct: dict, kinds: dict, tokens=1000, cost=0.5):
    per = [{"id": q, "kind": kinds[q], "sub": None, "correct": c, "given": "1"} for q, c in correct.items()]
    return {"arm": arm, "doc": doc, "repeat": rep, "per_question": per, "input_tokens": tokens, "cost_usd": cost,
            "acc_all": sum(correct.values()) / len(correct), "image_tokens_formula": tokens, "output_tokens": 10,
            "latency_s": 1.0, "raw": ""}


def test_analyse_scores_by_tier_and_predictions():
    # 6 lookups (two per tier) and 2 multi-step questions: C1 all in the last third, C2 touches the first third
    kinds = {**{f"N{i}": "needle" for i in range(6)}, "C1": "complex", "C2": "complex"}
    qt = {0: {**{f"N{i}": [i // 2] for i in range(6)}, "C1": [2, 2], "C2": [0, 2]}}
    allright = {q: True for q in kinds}
    first_wrong = {**allright, "N0": False, "N1": False, "C2": False}     # the first third read badly
    last_wrong = {**allright, "N4": False, "N5": False, "C1": False, "C2": False}
    v2 = {0: {"text": row("text", 0, 0, allright, kinds), 117.6: row("image", 0, 0, allright, kinds),
              88.2: row("image", 0, 0, allright, kinds), 58.8: row("image", 0, 0, {**allright, "N0": False}, kinds)}}
    rows = [row("recent_sharp", 0, 0, first_wrong, kinds), row("old_sharp", 0, 0, last_wrong, kinds),
            {**row("old_sharp", 0, 1, allright, kinds), "incomplete": True}]
    res = G.analyse(rows, v2, qt)
    ar = res["arms"]
    assert res["invalid"] == [(0, "old_sharp", 1)]
    assert ar["recent_sharp"]["lookup_by_tier"][0]["acc"] == 0.0 and ar["recent_sharp"]["lookup_by_tier"][2]["acc"] == 1.0
    assert ar["old_sharp"]["lookup_by_tier"][2]["acc"] == 0.0
    assert ar["uniform 58.8"]["lookup_by_tier"][0]["acc"] == 0.5
    # multi-step split: for recent_sharp, C2 touches its 58.8 third (tier 0) and C1 does not
    dense = ar["recent_sharp"]["complex_by_recent_sharp_tiers"]["dense"]
    assert (dense["k"], dense["n"]) == (0, 1)
    assert ar["recent_sharp"]["complex_by_recent_sharp_tiers"]["sharp"]["acc"] == 1.0
    # for old_sharp, the 58.8 third is tier 2: both questions touch it
    assert ar["old_sharp"]["complex_by_old_sharp_tiers"]["dense"]["n"] == 2
    t = res["tests"]["recent_sharp vs uniform 117.6 (first_call)"]
    assert t["n"] == 8 and t["ref_only"] == 3 and t["arm_only"] == 0 and t["arm_minus_ref"] == pytest.approx(-3 / 8)
    p = res["predictions"]
    cell = next(c for c in p["N1"]["cells"] if c["arm"] == "recent_sharp" and c["tier"] == 0)
    assert cell["density"] == 58.8 and cell["uniform"] == 0.5 and cell["mixed"] == 0.0 and not p["N1"]["pass"]
    assert p["N2"]["recent_minus_old"] == pytest.approx(5 / 8 - 4 / 8)
    md = G.summary_md(res)
    assert "N1 FAIL" in md and "first third" in md


def test_analyse_without_new_rows():
    kinds = {"N0": "needle"}
    r = row("image", 0, 0, {"N0": True}, kinds)
    res = G.analyse([], {0: {"text": r, 117.6: r, 88.2: r, 58.8: r}}, {0: {"N0": [1]}})
    assert res["predictions"] == {} and "Not judged yet" in G.summary_md(res)


def test_salt_and_jobs():
    assert G.salt(0) is None and G.salt(1) == "gradient1"
    jobs = G.todo_jobs([{"doc": 0, "arm": "recent_sharp", "repeat": 0}], [0, 1, 2])
    assert len(jobs) == 11 and (0, "recent_sharp", 0) not in jobs
    assert jobs[:5] == [(0, "old_sharp", 0), (1, "recent_sharp", 0), (1, "old_sharp", 0), (2, "recent_sharp", 0),
                        (2, "old_sharp", 0)]
