"""G2-lite checks (ep/g2lite.py, experiments/run_g2lite.py) on the tiny random-init Qwen2.5-VL tower, CPU only."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "encoder_probe" / "experiments"))

from encoder_probe.ep import g2lite as G  # noqa: E402
from encoder_probe.ep.probe_grid import GridProbe  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def tower():
    from encoder_probe.ep import qwen_ft
    return qwen_ft.load_model(torch.device("cpu"), tiny=True)[0].model.visual.eval()


@pytest.fixture(scope="module")
def vision(tower):
    from run_g2lite import TinyVision
    return TinyVision(tower)


def _imgs(seed, n=3):
    return np.random.default_rng(seed).integers(0, 255, (n, 448, 448, 3), dtype=np.uint8)


def test_window_reverse_matches_tower(tower):
    from encoder_probe.ep import qwen_split as QS
    meta = QS.vit_meta(tower, torch.tensor([[1, 32, 32]]))
    assert torch.equal(G.window_reverse(448, 14, 2, tower.window_size), meta["reverse"])
    assert G.window_reverse(448, 14, 2, 112).shape == (256,)


def test_cached_states_through_frozen_merger_reproduce_vlm_tokens(vision, tower):
    """The cached path (final-block states, window order, as token_cache.build stores them) through the frozen
    merger front = the qwen25vl tokens vlm_encoders.QwenVision gives for the same pages (pooler_output)."""
    imgs = _imgs(1)
    k = len(tower.blocks)
    ref = vision(imgs).float()                                   # (3, 256, out) as the token cache stores them
    d = vision(imgs, split=k)
    h = d[f"h{k}"]                                                # (3, 1024, 32) bf16, window order
    assert h.dtype == torch.bfloat16 and h.shape == (3, 1024, 32)
    rev = G.window_reverse(448, 14, 2, tower.window_size)
    f = G.Front("frozen", copy.deepcopy(tower.merger), rev, 32)
    with torch.no_grad():
        got = f.raw(h)
        # bf16 storage of the states is the only difference
        rel = ((got - ref).norm() / ref.norm()).item()
        assert got.shape == ref.shape and rel < 2e-2, rel
        assert np.corrcoef(got.flatten(), ref.flatten())[0, 1] > 0.9995
        # with the fp32 states the reconstruction is exact (ordering / grouping is right)
        from encoder_probe.ep import qwen_split as QS
        from encoder_probe.ep.vlm_encoders import qwen_pixel_values
        thw = torch.tensor([[1, 32, 32]] * 3)
        h32 = QS.head(tower, torch.from_numpy(qwen_pixel_values(imgs)), thw, k)
        assert torch.allclose(f.raw(h32.reshape(3, 1024, 32)), tower(hidden_states=torch.from_numpy(
            qwen_pixel_values(imgs)), grid_thw=thw).pooler_output.reshape(3, 256, -1), atol=1e-5, rtol=1e-5)
        # direct arm: the 4 patch states of each merge group, groups row-major (same group order as the merger)
        dd = G.Front("direct", None, rev, 32).raw(h32.reshape(3, 1024, 32))
        assert dd.shape == (3, 256, 128)
        grp = h32.reshape(3, 256, 4, 32)[:, rev]
        assert torch.equal(dd, grp.reshape(3, 256, 128))


def test_standardiser_streams_mean_and_std(tower):
    rev = G.window_reverse(448, 14, 2, tower.window_size)
    h = torch.randn(5, 1024, 32).to(torch.bfloat16)
    for arm in G.ARMS:
        f = G.Front(arm, None if arm == "direct" else copy.deepcopy(tower.merger), rev, 32)
        raw = f.fit_standardiser(h, batch=2, keep=True)
        with torch.no_grad():
            x = f.raw(h).reshape(-1, f.d)
            assert torch.allclose(raw.reshape(-1, f.d), x)
            assert torch.allclose(f.mu, x.mean(0), atol=1e-5) and torch.allclose(f.sd, x.std(0), rtol=1e-4)
            z = f(h).reshape(-1, f.d)
            assert z.mean(0).abs().max() < 1e-4 and (z.std(0) - 1).abs().max() < 1e-3


@pytest.mark.parametrize("arm", G.ARMS)
def test_only_intended_parameters_train(tower, arm):
    rev = G.window_reverse(448, 14, 2, tower.window_size)
    merger = copy.deepcopy(tower.merger)
    before = {k: v.clone() for k, v in merger.state_dict().items()}
    front = G.Front(arm, None if arm == "direct" else merger, rev, 32)
    h = torch.randn(2, 1024, 32).to(torch.bfloat16)
    front.fit_standardiser(h)
    probe = GridProbe(8, 5, d=front.d, G=16, P=28)
    model = G.G2Reader(None if arm == "frozen" else front, probe)
    x = G.precompute(front, h) if arm == "frozen" else h
    names = G.trainable_names(model)
    assert all(n.startswith("probe.") for n in names if arm != "trained")
    assert any(n.startswith("front.merger.") for n in names) == (arm == "trained")
    if arm == "trained":   # the whole merger: RMSNorm weight and both MLP layers
        assert {n for n in names if n.startswith("front.")} == {f"front.merger.{k}" for k in G.MERGER_KEYS}
    p0 = {n: p.detach().clone() for n, p in model.named_parameters()}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)
    feat = model.features(x)
    lp = model.read(feat, torch.tensor([0, 1]), torch.tensor([0, 3]))
    loss = lp.mean() + model.recon_loss(feat, torch.zeros_like(feat))
    loss.backward()
    opt.step()
    for n, p in model.named_parameters():
        moved = not torch.equal(p0[n], p.detach())
        assert moved == p.requires_grad, n
    after = merger.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) == (arm != "trained")
    assert torch.equal(front.mu, front.mu) and not front.mu.requires_grad


# ------------------------------------------------------------------------------ script end to end (tiny)

RUN = ["--tiny", "--steps", "4", "--train-pages", "3", "--test-pages", "4", "--eval-every", "2",
       "--ckpt-every", "2", "--eval-pages", "2", "--poll", "0.01"]


def _run(out, cache, arm, *extra, enc="latin_p24"):
    from run_g2lite import main
    return main(["--arm", arm, "--encoding", enc, *RUN, "--run-name", "r", "--out-root", str(out),
                 "--cache-root", str(cache), *extra])


def _rows(out):
    return [json.loads(l) for l in (out / "r" / "runs.jsonl").read_text().splitlines()]


def test_resume_is_exact_and_row_schema(tmp_path):
    a, b, cache = tmp_path / "a", tmp_path / "b", tmp_path / "cache"   # one cache (pages + states) for both
    assert _run(a, cache, "trained") == 0                                  # builds the hidden-state cache
    assert (cache / "tokens" / "qwen25vl_h6" / "clean" / "latin_p24_tr3_te4.pt").exists()
    assert _run(b, cache, "trained", "--stop-after", "2") == 0             # "crash" after the step-2 checkpoint
    ck = cache / "ckpt" / "g2lite__r__trained__latin_p24__4__tp3__s0.pt"
    assert ck.exists() and not (b / "r" / "runs.jsonl").exists()
    assert _run(b, cache, "trained") == 0                                  # resumes from step 2
    assert not ck.exists()
    ra, rb = _rows(a)[-1], _rows(b)[-1]
    for k in ("cer", "curve", "train_curve", "train_ctc", "train_ctc_last", "samples"):
        assert ra[k] == rb[k], k
    p = "bf16" if ra["bf16"] else "fp32"
    assert ra["precision_segments"] == [[1, 4, p]] and rb["precision_segments"] == [[1, 2, p], [3, 4, p]]
    r = ra
    for k in ("encoding", "arm", "steps", "train_pages", "seed", "phase", "round", "gate", "encoder", "cond",
              "scheme", "family", "pitch", "S", "P", "G", "n_tokens", "d", "px_per_char", "budget",
              "chars_per_token", "chars_per_vlm_token", "cer", "score_v", "train_ctc", "train_ctc_last",
              "train_curve", "bf16", "device", "test_pages", "lr", "n_test_chars", "digest_test", "curve",
              "trainable", "params", "merger_check", "config", "s_per_step", "samples"):
        assert k in r, k
    assert r["arm"] == "trained" and r["encoder"] == "qwen25vl_h6" and r["round"] == 3 and r["gate"] == "G2-lite"
    assert r["n_tokens"] == 256 and r["test_pages"] == 4 and r["train_pages"] == 3 and r["d"] == 64
    assert abs(r["chars_per_token"] * r["px_per_char"] - 784) < 1e-6
    assert r["budget"] == round(r["px_per_char"], 3) and 0 <= r["cer"]
    assert [s for s, _ in r["curve"]] == [2, 4] and [s for s, _ in r["train_curve"]] == [2, 4]
    assert r["config"]["arm"] == "trained" and r["config"]["lr"] == 2e-3 and r["config"]["steps"] == 4
    assert "front.merger" in r["trainable"] and r["params"]["merger_trainable"] > 0
    assert r["merger_check"]["rel_l2"] < 2e-2 and r["merger_check"]["corr"] > 0.999
    assert _run(a, cache, "trained") == 0 and len(_rows(a)) == 1                  # done -> skipped
    # the other arms: separate keys, same probe recipe and seed, only the arm's parameters
    assert _run(a, cache, "frozen") == 0 and _run(a, cache, "direct") == 0 and _run(a, cache, "blind") == 0
    rows = {x["arm"]: x for x in _rows(a)}
    assert set(rows) == set(G.ARMS) | {"blind"}
    for x in rows.values():   # the op split reproduces the row's CER
        assert abs(x["ops"]["sub"] + x["ops"]["ins"] + x["ops"]["del"] - x["ops"]["cer"]) < 1e-9
        assert abs(x["ops"]["cer"] - x["cer"]) < 1e-9
    b = rows["blind"]
    assert b["blind"]["source"] == "frozen" and sorted(b["blind"]["perm_test"]) == list(range(4))
    assert all(i != j for i, j in enumerate(b["blind"]["perm_test"]))
    assert b["trainable"] == rows["frozen"]["trainable"]
    assert rows["frozen"]["trainable"] == ["probe.blocks", "probe.grid", "probe.inp", "probe.out", "probe.skip"]
    assert rows["direct"]["trainable"] == rows["frozen"]["trainable"] and rows["direct"]["d"] == 128
    assert rows["frozen"]["params"]["merger_trainable"] == 0 and rows["frozen"]["d"] == 64
    for arm in ("frozen", "direct"):
        cfg = {k: v for k, v in rows[arm]["config"].items() if k != "arm"}
        assert cfg == {k: v for k, v in rows["trained"]["config"].items() if k != "arm"}
        assert rows[arm]["digest_test"] == rows["trained"]["digest_test"]


def test_print_queue_order():
    import contextlib
    import io
    from run_g2lite import CODES, ORDER, main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["--print-queue"])
    lines = buf.getvalue().splitlines()
    runs = [l for l in lines if "--arm " in l]
    assert len(lines) == len(ORDER) + len(CODES) and all(l.endswith("--threads 1") for l in lines)
    assert [tuple(l.split("--arm ")[1].split(" --steps")[0].split(" --encoding ")) for l in runs] == ORDER
    assert ORDER[:4] == [("trained", "rgb3_lg_pb_h5"), ("trained", "latin_p8"), ("trained", "latin_p5"),
                         ("frozen", "rgb3_lg_pb_h5")]
    for i, l in enumerate(lines):                 # each cache is built right before its first run
        if "--build-only" in l:
            e = l.split("--encoding ")[1].split()[0]
            assert f"--encoding {e} " in lines[i + 1] and not any(f"--encoding {e} " in x for x in lines[:i])


def test_build_only_and_run_lock(tmp_path):
    from encoder_probe.ep import flock
    from run_g2lite import LeanVision, main
    cache = tmp_path / "cache"
    args = ["--tiny", "--train-pages", "3", "--test-pages", "4", "--encoding", "latin_p24",
            "--cache-root", str(cache), "--poll", "0.01"]
    assert main(["--build-only", *args]) == 0
    hp = cache / "tokens" / "qwen25vl_h6" / "clean" / "latin_p24_tr3_te4.pt"
    assert hp.exists() and (cache / "tokens" / "qwen25vl" / "clean" / "latin_p24_tr3_te4.pt").exists()
    assert main(["--build-only", *args]) == 0                              # ready: nothing to do
    # a run whose lock is held elsewhere is skipped (no row, no work)
    ck = cache / "ckpt" / "g2lite__r__frozen__latin_p24__4__tp3__s0.lock"
    ck.parent.mkdir(parents=True, exist_ok=True)
    f = open(ck, "a")
    assert flock.try_lock(f)
    assert main(["--arm", "frozen", *args, "--steps", "4", "--run-name", "r", "--out-root", str(tmp_path / "res")]) == 0
    assert not (tmp_path / "res" / "r" / "runs.jsonl").exists()
    flock.unlock(f)

    class Inner:
        compute = "x"

        def __call__(self, imgs, **kw):
            return imgs.sum()
    last = np.ones((2, 4, 4, 3), np.uint8)
    lv = LeanVision(Inner(), last)
    lv(np.zeros_like(last))
    assert lv.inner is not None and lv.compute == "x"
    lv(last.copy())
    assert lv.inner is None
