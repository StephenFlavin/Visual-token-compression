"""Tests for vtc/analysis.py and experiments/phases/phase1_sweep.py (fake transcriber; never calls Claude)."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from vtc import analysis, config

ROOT = Path(__file__).resolve().parent.parent
GRID = [2, 4, 8, 16, 32, 64, 128, 256]


def _sweep_module():
    spec = importlib.util.spec_from_file_location("phase1_sweep", ROOT / "experiments" / "phases" / "phase1_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lin_log_rows(scheme, k=1.0, passages=("p0", "p1", "p2"), offsets=(-0.01, 0.0, 0.01), variant="natural",
                 grid=GRID, a=0.5, b=0.1):
    """CER(x) = a - b*ln(x/k) (clipped at 0), plus a per-passage offset with zero mean.

    Linear in log x, so log-x interpolation is exact: crossing of threshold t at x = k*exp((a-t)/b).
    """
    rows = []
    for pid, off in zip(passages, offsets):
        for x in grid:
            c = max(0.0, a - b * math.log(x / k)) + off
            rows.append({"run": "t", "phase": 1, "model": "fake", "variant": variant, "passage_id": pid,
                         "scheme": scheme, "budget": x, "px_per_char": float(x), "cer": c,
                         "n_chars": 1000, "text_tokens_est": 250.0})
    return rows


def expected_cross(t, k=1.0, a=0.5, b=0.1):
    return k * math.exp((a - t) / b)


# --------------------------------------------------------------------------------- crossing / p_at
@pytest.mark.parametrize("thr", [0.10, 0.03])
def test_p_at_known_crossing(thr):
    rows = lin_log_rows("latin")
    assert analysis.p_at(rows, "latin", thr) == pytest.approx(expected_cross(thr), rel=1e-6)


def test_p_at_scaled_scheme():
    rows = lin_log_rows("latin") + lin_log_rows("subst300", k=0.5)
    assert analysis.p_at(rows, "subst300", 0.10) == pytest.approx(expected_cross(0.10, k=0.5), rel=1e-6)


def test_p_at_never_and_left_censored():
    flat = [{"scheme": "s", "budget": x, "px_per_char": x, "cer": 0.5, "passage_id": "a"} for x in GRID]
    assert analysis.p_at(flat, "s", 0.1) is None
    low = [{"scheme": "s", "budget": x, "px_per_char": x, "cer": 0.01, "passage_id": "a"} for x in GRID]
    p, cens = analysis.crossing(np.array(GRID, float), np.full(len(GRID), 0.01), 0.1)
    assert p == GRID[0] and cens == "left"
    assert analysis.p_at(low, "s", 0.1) == GRID[0]
    assert analysis.p_at([], "s", 0.1) is None


def test_crossing_uses_monotone_smoothing():
    # a non-monotone blip below threshold at x=4 must not count as the crossing
    x = np.array([2, 4, 8, 16, 32], float)
    y = np.array([0.30, 0.05, 0.25, 0.08, 0.02])
    p_raw, _ = analysis.crossing(x, y, 0.10, smooth=False)
    p_iso, _ = analysis.crossing(x, y, 0.10)
    assert p_raw < 4.01
    assert p_iso > 4.0


def test_isotonic_decreasing():
    out = analysis.isotonic_decreasing(np.array([0.5, 0.3, 0.4, 0.1]))
    np.testing.assert_allclose(out, [0.5, 0.35, 0.35, 0.1])
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.random(12)
        w = rng.random(12) + 0.1
        s = analysis.isotonic_decreasing(y, w)
        assert np.all(np.diff(s) <= 1e-12)
        assert np.sum(s * w) == pytest.approx(np.sum(y * w))       # PAVA preserves the weighted mean
    already = np.array([0.9, 0.5, 0.5, 0.2])
    np.testing.assert_allclose(analysis.isotonic_decreasing(already), already)
    nan = analysis.isotonic_decreasing(np.array([0.5, np.nan, 0.6]))
    assert np.isnan(nan[1]) and nan[0] == nan[2] == pytest.approx(0.55)


# ---------------------------------------------------------------------------------------- bootstrap
def test_bootstrap_shape_and_ci():
    rows = lin_log_rows("latin", passages=[f"p{i}" for i in range(5)], offsets=[-0.02, -0.01, 0, 0.01, 0.02])
    t = analysis.CurveTable(rows, "latin")
    b = t.bootstrap(n_boot=200, seed=1)
    assert b.shape == (200, len(GRID))
    cv = analysis.curve(rows, "latin", n_boot=300)
    assert len(cv["mean"]) == len(GRID) == len(cv["lo"]) == len(cv["hi"])
    assert all(lo <= m + 1e-12 <= hi + 2e-12 for lo, m, hi in zip(cv["lo"], cv["mean"], cv["hi"]))
    assert cv["budget"] == GRID
    ci = analysis.p_at_ci(rows, "latin", 0.10, n_boot=300)
    assert ci["lo"] <= ci["p"] <= ci["hi"]
    assert ci["frac_crossed"] == 1.0


def test_unpaired_phase2_rows():
    """Phase 2 rows: no budget / passage key; grouped by em, x = px_per_char."""
    rows = []
    for em, x in [(5, 3.0), (6, 4.5), (8, 7.0), (10, 11.0), (12, 16.0)]:
        for rep, off in enumerate((-0.01, 0.01)):
            rows.append({"scheme": "latin", "em": em, "px_per_char": x, "cer": 0.5 - 0.1 * math.log(x) * 2 + off})
    t = analysis.CurveTable(rows, "latin")
    assert t.units is None and t.bootstrap(50).shape == (50, 5)
    p = analysis.p_at(rows, "latin", 0.10)
    assert p == pytest.approx(math.exp(2.0), rel=1e-6)
    h = analysis.headline(rows, n_boot=50)
    assert h["phase"] == 2 and "K3" in h["verdicts"]


# ----------------------------------------------------------------------------------------- headline
def test_headline_ratio_math_and_k2():
    rows = (lin_log_rows("latin", k=1.0) + lin_log_rows("latin_condensed", k=0.8)
            + lin_log_rows("subst300", k=0.6) + lin_log_rows("quad3", k=1.5))
    h = analysis.headline(rows, n_boot=200)
    assert h["phase"] == 1
    t10 = h["conditions"]["natural"]["thresholds"]["0.1"]
    assert t10["best_latin"] == "latin_condensed"
    best = expected_cross(0.10, k=0.8)
    assert t10["best_latin_p"] == pytest.approx(best, rel=1e-6)
    sub = t10["schemes"]["subst300"]
    assert sub["ratio"] == pytest.approx(0.6 / 0.8, rel=1e-6)
    assert sub["saving"] == pytest.approx(0.25, rel=1e-6)
    assert sub["compression"] == pytest.approx(0.25 / (sub["p"] / config.CLAUDE_PX_PER_TOKEN), rel=1e-6)
    assert sub["ratio_lo"] <= sub["ratio"] <= sub["ratio_hi"]
    assert t10["schemes"]["quad3"]["ratio"] == pytest.approx(1.5 / 0.8, rel=1e-6)
    assert "NOT TRIGGERED" in h["verdicts"]["K2"]
    assert "CONFIRMED" in h["verdicts"]["P1"]        # quad3 needs 1.875x best Latin
    # now subst only 10% better -> K2 triggered
    rows2 = [r for r in rows if r["scheme"] != "subst300"] + lin_log_rows("subst300", k=0.72)
    assert "TRIGGERED - zero-shot negative" in analysis.headline(rows2, n_boot=100)["verdicts"]["K2"]


def test_headline_variants_prior_credit_and_refusals():
    rows = (lin_log_rows("latin") + lin_log_rows("subst300", k=0.9)
            + lin_log_rows("latin", variant="scrambled", passages=("s0", "s1", "s2"), a=0.6)
            + lin_log_rows("subst300", variant="scrambled", passages=("s0", "s1", "s2"), k=0.5, a=0.6))
    for r in rows:
        r["refusal"] = r["cer"] > 0.4
    h = analysis.headline(rows, n_boot=100)
    assert set(h["conditions"]) == {"natural", "scrambled"}
    assert "P3" in h["verdicts"]
    pc = analysis.prior_credit(rows)
    d = pc["all"]["latin"]["16"]
    assert d["diff"] == pytest.approx(-0.1) and d["n_natural"] == 3
    rr = analysis.refusal_rates(rows)
    assert 0 < rr["natural"]["latin"]["rate"] < 1


def test_load_runs_dedupe(tmp_path):
    base = {"run": "r", "model": "m", "variant": "natural", "passage_id": "p", "scheme": "latin",
            "budget": 8, "px_per_char": 8.0}
    lines = [dict(base, cer=0.5, error=None), dict(base, cer=None, error="boom"),
             dict(base, budget=11, cer=None, error="boom"), dict(base, budget=11, cer=0.2, error=None)]
    f = tmp_path / "runs.jsonl"
    f.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rows = analysis.load_runs(f)
    assert len(rows) == 2
    assert {r["budget"]: r["cer"] for r in rows} == {8: 0.5, 11: 0.2}
    assert len(analysis.load_runs(tmp_path, dedupe=False)) == 4


def test_write_summary(tmp_path):
    rows = lin_log_rows("latin") + lin_log_rows("latin_condensed", k=0.8) + lin_log_rows("subst300", k=0.7)
    s = analysis.write_summary(rows, tmp_path, n_boot=100)
    for name in ("summary.json", "summary.md", "cer_curves.png", "p_at_bars.png"):
        assert (tmp_path / name).stat().st_size > 0
    js = json.loads((tmp_path / "summary.json").read_text())
    assert "K2" in js["headline"]["verdicts"] and s["counts"]["usable"] == len(rows)
    assert "| subst300 |" in (tmp_path / "summary.md").read_text()


# --------------------------------------------------------------------------------- phase1_sweep
def test_sweep_helpers():
    m = _sweep_module()
    for b in m.DEFAULT_BUDGETS + m.PILOT_BUDGETS:
        W, H = m.canvas_size(b, 3000)
        assert W % 28 == 0 and H % 28 == 0 and W >= H >= 56 and W - H <= 28
        assert abs(math.log(W * H / (b * 3000))) < 0.25
    sizes = [m.canvas_size(b, 3000) for b in m.DEFAULT_BUDGETS]
    assert len(set(sizes)) == len(sizes)
    assert m.canvas_size(0.01, 10) == (56, 56)
    assert m.looks_like_refusal("I'm unable to read this image.", "x" * 30)
    assert m.looks_like_refusal("short", "a much longer target transcription")
    assert not m.looks_like_refusal("the cat sat on the mat", "the cat sat on the mat")
    t = m.truncate_text("One two three. Four five six seven. Eight nine ten eleven twelve.", 33)
    assert t == "One two three. Four five six seven."
    assert m.truncate_text("aaa bbb ccc ddd", 9) == "aaa bbb"


def test_phase1_dry_run_end_to_end(tmp_path):
    m = _sweep_module()
    args = ["--dry-run", "--schemes", "latin", "latin_condensed", "--budgets", "10", "30",
            "--n-passages", "2", "--passage-chars", "400", "--out-dir", str(tmp_path), "--run-name", "t"]
    assert m.main(args) == 0
    run = tmp_path / "t"
    rows = [json.loads(x) for x in (run / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 2 * 2
    need = {"run", "phase", "model", "effort", "variant", "passage_id", "scheme", "budget", "px_per_char",
            "W", "H", "em", "n_lines", "fit", "vision_tokens", "key_image_tokens", "text_tokens_est", "cer",
            "symbol_cer", "output_text", "cost_usd", "input_tokens", "output_tokens", "latency_s", "cached",
            "error", "refusal", "skipped"}
    for r in rows:
        assert need <= set(r)
        assert r["W"] % 28 == 0 and r["H"] % 28 == 0
        assert r["px_per_char"] == pytest.approx(r["W"] * r["H"] / r["n_chars"])
        assert r["n_chars"] <= 500
        assert r["cost_usd"] == 0 and r["model"] == "fake"
        assert 0 <= r["cer"] <= 1.5
        png = run / "images" / r["scheme"] / f"{r['passage_id']}_{r['budget']:g}.png"
        assert png.exists()
    assert (run / "summary.md").exists() and (run / "cer_curves.png").exists()
    # resumable: a second invocation adds nothing
    assert m.main(args) == 0
    assert len((run / "runs.jsonl").read_text().splitlines()) == 8


def test_real_run_requires_yes_and_respects_max_cost(tmp_path, monkeypatch):
    """Without --yes nothing is submitted; over --max-cost aborts. transcribe_many must never run."""
    m = _sweep_module()
    pytest.importorskip("vtc.claude_eval")
    from vtc import claude_eval

    def boom(*a, **k):
        raise AssertionError("Claude must not be called in tests")
    monkeypatch.setattr(claude_eval, "transcribe_many", boom)
    monkeypatch.setattr(claude_eval, "transcribe", boom)
    base = ["--schemes", "latin", "--budgets", "10", "--n-passages", "1", "--passage-chars", "300",
            "--out-dir", str(tmp_path), "--run-name", "real", "--no-analysis"]
    try:
        assert m.main(base) == 1                                   # estimate printed, needs --yes
        assert m.main(base + ["--max-cost", "0.0001", "--yes"]) == 3   # over budget -> abort
    except SystemExit as e:      # corpus unavailable: real runs refuse to use the built-in sample
        assert "corpus" in str(e)
    assert not (tmp_path / "real" / "runs.jsonl").exists() or \
        (tmp_path / "real" / "runs.jsonl").read_text().strip() == ""
