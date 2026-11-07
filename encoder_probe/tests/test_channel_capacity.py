"""channel_capacity.py (PLAN_3090 amendment 6): the block code, its geometry, a tiny run and exact resume."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import channel_capacity as CH  # noqa: E402


def test_geometry_offers_96_294_1176_bits_with_equal_bits_per_image():
    got = {c: CH.geometry(c, 4, 4704) for c in (7, 4, 2)}
    assert [got[c]["bits_per_token"] for c in (7, 4, 2)] == [96, 294, 1176]
    assert [got[c]["side"] for c in (7, 4, 2)] == [7, 4, 2]
    assert [got[c]["S"] for c in (7, 4, 2)] == [196, 112, 56]
    assert all(g["n_digits"] == 1568 for g in got.values())


def test_blocks_round_trip_and_tokens_own_contiguous_digits():
    for cell, L in ((7, 4), (4, 4), (2, 4), (4, 2)):
        g = CH.geometry(cell, L, 4704 if L == 4 else CH.geometry(cell, 2, 147 * 4)["bits_per_image"])
        img, d = CH.make_sample(g, np.random.default_rng(1))
        assert img.shape == (g["S"], g["S"], 3) and img.dtype == np.uint8
        assert len(d) == g["n_digits"] and set(d) <= set("01234567")
        assert CH.decode_image(g, img) == d
        # repaint token (0, 1) only: only that token's digit range changes
        img2 = img.copy()
        img2[:28, 28:56] = 255 - img2[:28, 28:56]
        d2 = CH.decode_image(g, img2)
        per = g["bits_per_token"] // 3
        diff = [i for i in range(len(d)) if d[i] != d2[i]]
        assert diff and min(diff) >= per and max(diff) < 2 * per


def _run(tmp, extra):
    common = ["--tiny", "--device", "cpu", "--threads", "1", "--cell", "7", "--levels", "2", "--bits-per-image",
              "192", "--steps", "2", "--batch-size", "2", "--eval-samples", "4", "--val-samples", "2",
              "--eval-every", "2", "--vision-blocks", "1", "--out-root", str(tmp / "res"),
              "--cache-root", str(tmp / "cache")]
    return CH.main(common + extra)


def test_tiny_run_writes_a_row_and_resume_is_exact(tmp_path):
    assert _run(tmp_path / "a", ["--ckpt-every", "0", "--run-name", "c"]) == 0
    row = json.loads((tmp_path / "a/res/c/runs.jsonl").read_text().splitlines()[0])
    assert row["bits_per_token"] == 48 and row["n_tokens"] == 4 and row["n_digits"] == 64
    for k in ("recovered_bits_per_token", "swap_recovered_bits_per_token", "digit_acc"):
        assert k in row["trained"] and k in row["frozen"]
    assert row["config"]["train_patch_embed"] is False and row["fair"]["vision_blocks"] == 1
    assert _run(tmp_path / "a", ["--ckpt-every", "0", "--run-name", "c"]) == 0       # skip: done
    assert len((tmp_path / "a/res/c/runs.jsonl").read_text().splitlines()) == 1
    assert _run(tmp_path / "b", ["--ckpt-every", "1", "--stop-after", "1", "--run-name", "c"]) == 0
    assert _run(tmp_path / "b", ["--ckpt-every", "1", "--run-name", "c"]) == 0
    rb = json.loads((tmp_path / "b/res/c/runs.jsonl").read_text().splitlines()[0])
    assert np.allclose(rb["trained"]["recovered_bits_per_token"], row["trained"]["recovered_bits_per_token"])
    assert rb["curve"] == row["curve"]


def test_analyse_run12_reads_the_channel_rows(tmp_path):
    import analyse_run12 as A
    assert _run(tmp_path, ["--ckpt-every", "0", "--run-name", "c"]) == 0
    chan = A.load_chan(tmp_path / "res" / "c")
    assert len(chan) == 1
    pts = [{"encoding": "latin_p8", "text": "shuffled", "trained": [20.0, 18.0, 22.0]}]
    ro = A.chan_readout(chan, pts)
    assert ro["latin"] == "latin_p8" and ro["headroom"] is False and "swap_ok" in ro
    res = {"points": [], "run1": [], "run2": [], "verdict_run1": "n/a", "verdict_run2": "n/a", "gate_A1": "n/a",
           "gate_A2": "n/a", "bar_recovered_cpt": {}, "chan": chan, "chan_readout": ro}
    assert "Channel capacity" in A.write_md(res, tmp_path / "s.md")
