"""Colour-gist phase A (docs/COLOUR_GIST.md): stream encodings, per-stream tasks, per-stream eval rows and the
gate analysis. TINY Qwen on the CPU, ~1 min."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "encoder_probe" / "experiments"))

from encoder_probe.ep import r3_tasks as T  # noqa: E402
from encoder_probe.ep.geom import get_encoding_s, make_pages_s  # noqa: E402


@pytest.fixture(scope="module")
def pages():
    return {k: make_pages_s(get_encoding_s(f"rgb{k}s_p24", 448), "test", 3) for k in (1, 2, 3)}


def test_streams_own_their_channels(pages):
    lat = make_pages_s(get_encoding_s("latin_p24", 448), "test", 2)
    p1, p2, p3 = pages[1][0], pages[2][0], pages[3][0]
    assert np.array_equal(p1.img, lat[0].img) and p1.refs == lat[0].refs      # 1 stream == latin_p24
    assert np.array_equal(p2.img[:, :, 0], lat[0].img[:, :, 0])              # stream 1 = the first page-worth
    assert (p2.img[:, :, 2] == 255).all() and p2.colours == ["cyan", "magenta"]
    assert p3.colours == ["cyan", "magenta", "yellow"] and len(p3.streams) == 3
    assert p2.refs == p2.streams[0] + p2.streams[1]
    assert 1.8 < p2.n_chars / p1.n_chars < 2.2 and 2.7 < p3.n_chars / p1.n_chars < 3.3
    assert p1.after and p1.after.split()[0] not in " ".join(p1.refs)[-5:]   # text that follows the page


def test_stream_questions_name_the_stream_and_lure_the_other(pages):
    pg = pages[2][0]
    others = [" ".join(p.refs) for p in pages[2][1:]]
    r = np.random.default_rng(0)
    for _ in range(20):
        q, ans, m = T.stream_mc_sample(pg, 0, others, r)
        opts = m["options"]
        assert "cyan" in q and ans != m["lure"]
        assert opts["ABCD".index(ans)] in T.stream_flat(pg, 0)
        lure = opts["ABCD".index(m["lure"])]
        assert lure in T.stream_flat(pg, 1) and lure not in T.stream_flat(pg, 0)
        c = T.stream_cloze_sample(pg, 1, r)
        assert c is None or ("magenta" in c[0] and T._all_flat(pg).count(c[2]["context"]) == 1)
    q1, a1, m1 = T.stream_mc_sample(pages[1][0], 0, others, np.random.default_rng(1))
    assert "black" in q1 and m1["options"]["ABCD".index(m1["lure"])] in pages[1][0].after
    seen = {T.page_sample(pg, "", others, r, 2000, 0.6)[2]["stream"] for _ in range(40)}
    assert seen == {0, 1}


FT = ["--tiny", "--device", "cpu", "--steps", "2", "--train-pages", "3", "--test-pages", "2", "--batch-size", "2",
      "--max-target-chars", "80", "--max-new-tokens", "16", "--eval-windows", "1", "--eval-cloze", "1",
      "--eval-mc", "2", "--val-samples", "3", "--eval-every", "2", "--ckpt-every", "0", "--log-every", "100",
      "--eval-conds", "clean"]


def test_finetune_rows_carry_per_stream_metrics_and_paired_val(tmp_path):
    from finetune_qwen import main
    common = [*FT, "--run-name", "c", "--out-root", str(tmp_path / "res"), "--cache-root", str(tmp_path / "cache")]
    for enc in ("rgb1s_p24", "rgb2s_p24"):
        assert main(["--encoding", enc, *common]) == 0
        assert main(["--encoding", enc, *common, "--blind-images"]) == 0
    rows = {json.loads(l)["encoding"]: json.loads(l) for l in (tmp_path / "res/c/runs.jsonl").read_text().splitlines()}
    blind = {json.loads(l)["encoding"]: json.loads(l)
             for l in (tmp_path / "res/c_blind/runs.jsonl").read_text().splitlines()}
    r2 = rows["rgb2s_p24"]
    assert [s["colour"] for s in r2["per_stream"]] == ["cyan", "magenta"]
    assert all(s["n_mc"] >= 1 and s["lure_rate"] is not None for s in r2["per_stream"])
    assert len(r2["val_per_sample"]) == 3 and r2["fair_hash"] == rows["rgb1s_p24"]["fair_hash"]
    assert [n for _, n in r2["val_per_sample"]] == [n for _, n in blind["rgb2s_p24"]["val_per_sample"]]
    assert rows["rgb2s_p24"]["chars_per_token"] > 1.8 * rows["rgb1s_p24"]["chars_per_token"]
    import analyse_colour as AC
    AC.RES = tmp_path / "res"
    out, verdict = AC.analyse("c", "c_blind", 24)
    assert [r["k"] for r in out] == [1, 2] and "R" in out[1] and "headline_R2_gt_1" in verdict


def test_ratio_and_ci_on_synthetic_arms():
    import analyse_colour as AC
    n = np.full(50, 100.0)
    one = (np.full(50, 50.0), n)                  # 0.5 nats per text token
    two_equal = (np.full(50, 50.0), n)            # same gap per token, twice the chars per vision token
    r, lo, hi = AC.ratio_ci(two_equal, one, 14.0, 7.0)
    assert abs(r - 2) < 1e-9 and lo <= r <= hi
    half = (np.full(50, 25.0), n)                 # half the gap per token: no gain per vision token
    assert abs(AC.ratio_ci(half, one, 14.0, 7.0)[0] - 1) < 1e-9
