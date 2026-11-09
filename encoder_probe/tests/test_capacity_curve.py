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
    assert all(s["target"] for s in samples) and rows[0].get("cer_trunc") is not None
    assert C.cer_trunc("ab cd", "ab cd ef gh") == 0 and C.cer_trunc("ab cd", "ab") == 0.6


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
    assert eff["aligned_shift"] == [0, 0]
    off = [(7, 7), (28, 7), (28, 28), (7, 28)]                  # no (0, 0): the on-grid shift is (28, 28)
    assert C.alignment_stats(page + rng.normal(0, 0.1, (16, 4)), off, reps=200)["aligned_shift"] == [28, 28]
    assert eff["crossval_ci"][0] > 0 and null["crossval_ci"][0] < 0 < null["crossval_ci"][1]


def test_run1_recipe_frozen_llm_shuffled_margin_then_adapter_curve(tmp_path):
    """PLAN_3090 Runs 1-2: LoRA off (LLM frozen), ViT blocks + merger trained, shuffled text, top margin; the saved
    adapter loads into capacity_curve.py."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from finetune_qwen import main as ft
    from encoder_probe.ep import qwen_ft as Q
    FT = ["--tiny", "--device", "cpu", "--steps", "2", "--train-pages", "3", "--test-pages", "2", "--batch-size", "2",
          "--max-target-chars", "80", "--max-new-tokens", "16", "--eval-windows", "1", "--eval-cloze", "1",
          "--eval-mc", "2", "--val-samples", "3", "--eval-every", "2", "--ckpt-every", "0", "--log-every", "100",
          "--eval-conds", "clean", "--lora-r", "0", "--vision-blocks", "1", "--task-mix", "1.0",
          "--text", "shuffled", "--top-margin", "7", "--run-name", "r1",
          "--out-root", str(tmp_path / "res"), "--cache-root", str(tmp_path / "cache")]
    assert ft(["--encoding", "latin_p24", *FT]) == 0
    assert ft(["--encoding", "latin_p24", *FT, "--blind-images"]) == 0
    row = json.loads((tmp_path / "res/r1/runs.jsonl").read_text().splitlines()[0])
    blind = json.loads((tmp_path / "res/r1_blind/runs.jsonl").read_text().splitlines()[0])
    assert row["fair_hash"] == blind["fair_hash"] and row["lora_modules"] == 0
    st = torch.load(row["adapter"], map_location="cpu", weights_only=False)
    assert st["lora"]["r"] == 0 and not any("lora_" in n for n in st["trainable"])
    assert any(".blocks." in n for n in st["trainable"]) and any("merger" in n for n in st["trainable"])
    m, _ = Q.load_model(torch.device("cpu"), tiny=True)
    pm = Q.make_trainable(m, 0, 32, 0.0, 1, True, True)
    names = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert any("patch_embed" in n for n in names) and not any("language_model" in n for n in names)
    common = ["--tiny", "--device", "cpu", "--threads", "1", "--pages", "2", "--window-chars", "200",
              "--cer-pages", "0", "--out-root", str(tmp_path), "--run-name", "cap", "--adapter", row["adapter"]]
    assert C.main(["--encodings", "latin_p24", "--texts", "shuffled", *common]) == 0
    r = json.loads((tmp_path / "cap" / "runs.jsonl").read_text().splitlines()[0])
    assert r["adapter"] == row["adapter"]
    # the Runs 1-2 placement (one --shifts value = the 7-px margin) is reported in the main table
    assert C.main(["--encodings", "latin_p24", "--texts", "shuffled", *common[:-4], "--run-name", "cap7",
                   "--shifts", "7"]) == 0
    assert "| latin_p24 | shuffled |" in (tmp_path / "cap7" / "summary.md").read_text()


def test_analyse_run12_paired_and_matched(tmp_path):
    import analyse_run12 as A

    def write(d, enc_bits):
        d.mkdir(parents=True)
        with open(d / "samples.jsonl", "w") as f:
            for (enc, text), bits in enc_bits.items():
                for p, b in enumerate(bits):   # loss_real chosen so bits_vtok == b exactly
                    gap = b * np.log(2) * 256
                    f.write(json.dumps({"encoding": enc, "text": text, "page": p, "page_chars": 100,
                                        "window_chars": 100, "vision_tokens": 256, "n_tokens": 50,
                                        "loss_real": 1000.0 - gap, "loss_swap": 1000.0, "loss_blank": 2000.0}) + "\n")
    rng = np.random.default_rng(0)
    base = rng.normal(10, 1, 40)
    write(tmp_path / "f", {("latin_p6", "shuffled"): base, ("lg_n0_pb_c2_h7", "shuffled"): base * 0,
                           ("latin_p8", "shuffled"): base, ("lg_n0_pb_c4_h7", "shuffled"): base * 0})
    write(tmp_path / "t", {("latin_p6", "shuffled"): base + 2, ("lg_n0_pb_c2_h7", "shuffled"): base + 5,
                           ("latin_p8", "shuffled"): base + 2, ("lg_n0_pb_c4_h7", "shuffled"): base + 2.5})
    res = A.analyse(A.load(tmp_path / "f"), A.load(tmp_path / "t"), {})
    r1 = next(r for r in res["run1"] if r["encoding"] == "latin_p6")
    assert abs(r1["trained_minus_frozen"][0] - 2) < 1e-6 and r1["moves_right"] and res["verdict_run1"] == "PASS"
    r2 = next(r for r in res["run2"] if r["glyph"] == "lg_n0_pb_c2_h7")
    assert abs(r2["glyph_minus_latin"][0] - 3) < 1e-6 and r2["glyph_wins"]
    assert res["gate_A2"] == "PASS" and res["gate_A1"] == "n/a" and res["verdict_run2"] == "n/a"   # no md banks yet
    assert "PASS" in A.write_md(res, tmp_path / "s.md")


def test_shift_dx_sweeps_dy_only(tmp_path):
    common = ["--tiny", "--device", "cpu", "--threads", "1", "--pages", "2", "--window-chars", "150",
              "--cer-pages", "0", "--out-root", str(tmp_path), "--run-name", "dy"]
    assert C.main(["--encodings", "latin_p24", "--texts", "natural", "--shifts", "7", "21", "--shift-dx", "7",
                   "--shift-margin", "28", "--arms", "real", "blank", *common]) == 0
    samples = [json.loads(x) for x in (tmp_path / "dy" / "samples.jsonl").read_text().splitlines()]
    assert sorted({tuple(s["shift"]) for s in samples}) == [(7, 7), (7, 21)] and len(samples) == 4
    import analyse_run12 as A
    sw = A.load_sweep(tmp_path / "dy")
    assert set(sw[("latin_p24", "natural")]) == {7, 21}


def test_bar_status_placement_sensitive():
    import analyse_run12 as A
    assert A.bar_status(21.0, None)["status"] == "PASS"
    assert A.bar_status(19.0, None)["status"] == "below bar"
    b = A.bar_status(19.0, {7: 10.0, 14: 9.0, 21: 11.0, 28: 10.5})
    assert b["best_dy"] == 21 and abs(b["adjusted"] - 20.9) < 1e-9 and b["status"] == "placement-sensitive pass"
    assert A.bar_status(17.0, {7: 10.0, 21: 13.0})["status"] == "below bar"        # below 18 at the trained placement
    assert A.bar_status(19.0, {7: 10.0, 21: 10.4})["status"] == "below bar"        # adjusted 19.76


def test_pageset_allow_fewer(monkeypatch):
    """Amendment 7: with allow_fewer, a set takes every page the held-out text supplies (>= 4), else it raises."""
    import pytest
    full = C.text_stream_s("test", 10 ** 5)
    monkeypatch.setattr(C, "text_stream_s", lambda split, need: full[:30000])
    with pytest.raises(ValueError):
        C.PageSet("latin_p24", "natural", 50)
    ps = C.PageSet("latin_p24", "shuffled", 50, allow_fewer=True)
    assert 4 <= len(ps.pages) < 50
