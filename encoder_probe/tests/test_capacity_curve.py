"""capacity_curve.py (docs/PLAN_3090.md Run 0 / 0b): text sets, windows, shifts, rows, resume, alignment stats.
TINY Qwen on the CPU, no network."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "encoder_probe" / "experiments"))

import capacity_curve as C  # noqa: E402


def test_shuffled_pages_keep_words_and_density():
    nat, shu = C.PageSet("latin_p24", "natural", 3), C.PageSet("latin_p24", "shuffled", 3)
    assert C.shuffle_words("a b c d e f", 0, 1) == C.shuffle_words("a b c d e f", 0, 1)
    assert sorted(C.shuffle_words("a b c d e f", 0, 1).split()) == list("abcdef")
    for a, b in zip(nat.pages, shu.pages):
        assert a.refs != b.refs and len(a.refs) == len(b.refs)
        wa, wb = " ".join(a.refs).split(), " ".join(b.refs).split()
        shared = sum(min(wa.count(w), wb.count(w)) for w in set(wb))
        assert shared >= 0.9 * len(wb)                       # same passage's words (page ends may differ)
        assert abs(a.n_chars - b.n_chars) < 0.1 * a.n_chars


def test_shift_pastes_the_same_text_with_no_cut_and_upscale():
    ps = C.PageSet("latin_p24", "natural", 2, margin=21)
    im0, im = ps.image(0, (0, 0)), ps.image(0, (14, 7))
    assert np.array_equal(im[7:7 + ps.inner, 14:14 + ps.inner], im0[:ps.inner, :ps.inner])
    assert (im[:7] == 255).all() and (im[:, :14] == 255).all()
    up = C.PageSet("latin_p12_x2", "natural", 2)
    assert up.f == 2 and up.inner == 224 and up.image(0).shape == (448, 448, 3)
    assert np.array_equal(up.image(0)[::2, ::2], up.pages[0].img)


def test_window_is_whole_lines_within_budget():
    pg = C.PageSet("latin_p12", "natural", 2).pages[0]
    a, b = C.window(pg, 600)
    assert a == 0 and sum(len(r) for r in pg.refs[a:b]) + (b - a - 1) <= 600 < len("\n".join(pg.refs[:b + 1]))
    m = C.window(pg, 600, "mid")
    assert m[0] > 0 and m[1] < len(pg.refs)


def test_curve_rows_resume_and_report(tmp_path):
    common = ["--tiny", "--device", "cpu", "--threads", "1", "--pages", "3", "--window-chars", "200",
              "--cer-pages", "1", "--cer-max-new", "8", "--out-root", str(tmp_path), "--run-name", "r"]
    enc = ["--encodings", "latin_p24", "latin_p16", "--texts", "natural", "shuffled"]
    assert C.main([*enc, *common, "--max-points", "1"]) == 0
    assert C.main([*enc, *common]) == 0
    rows = [json.loads(x) for x in (tmp_path / "r" / "runs.jsonl").read_text().splitlines()]
    samples = [json.loads(x) for x in (tmp_path / "r" / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and len(samples) == 12                    # resume scored nothing twice
    for s in samples:
        st = C.sample_stats(s)
        assert st["ref"] == min(s["loss_swap"], s["loss_blank"]) and s["window_chars"] <= 200
        assert np.isclose(st["bits_vtok"], st["gap"] / np.log(2) * s["page_chars"] / s["window_chars"] / 256)
    r = rows[0]
    assert r["cer_pages"] == 1 and r["bits_per_vtok_ci"][0] <= r["bits_per_vtok"] <= r["bits_per_vtok_ci"][1]
    assert C.main([*enc, *common]) == 0                              # all done: no model load, report only
    md = (tmp_path / "r" / "summary.md").read_text()
    assert "latin_p16" in md and (tmp_path / "r" / "capacity.png").exists()
    assert "Spearman" in md and "shuffled recovered fraction below natural at" in md


def test_spearman():
    assert C.spearman([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1])[0] == -1.0
    rho, p = C.spearman([1, 2, 2, 4], [1, 3, 3, 9], reps=500)
    assert np.isclose(rho, 1.0) and p > 0.01             # ties averaged; 4 points cannot be significant


def test_shift_grid_scores_blank_once_per_page(tmp_path):
    common = ["--tiny", "--device", "cpu", "--threads", "1", "--pages", "2", "--window-chars", "150",
              "--cer-pages", "0", "--out-root", str(tmp_path), "--run-name", "b"]
    assert C.main(["--encodings", "latin_p24", "--texts", "natural", "--shifts", "0", "14",
                   "--arms", "real", "blank", *common]) == 0
    samples = [json.loads(x) for x in (tmp_path / "b" / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 8 and all(s["loss_swap"] is None for s in samples)
    for p in (0, 1):
        assert len({s["loss_blank"] for s in samples if s["page"] == p}) == 1
        assert len({s["loss_real"] for s in samples if s["page"] == p}) > 1
    assert "Grid alignment" in (tmp_path / "b" / "summary.md").read_text()


def test_alignment_stats_null_vs_effect():
    rng = np.random.default_rng(0)
    shifts = [(dx, dy) for dy in (0, 7) for dx in (0, 7)]
    page = rng.normal(5, 1, (16, 1))
    null = C.alignment_stats(page + rng.normal(0, 0.1, (16, 4)), shifts, reps=2000)
    eff = C.alignment_stats(page + rng.normal(0, 0.1, (16, 4)) + np.array([0.5, 0, 0, 0]), shifts, reps=2000)
    assert null["perm_p"] > 0.05 and eff["perm_p"] < 0.01
    assert eff["best_shift"] == [0, 0] and eff["aligned_vs_mean_ci"][0] > 0.3
    assert eff["crossval_ci"][0] > 0 and null["crossval_ci"][0] < 0 < null["crossval_ci"][1]
