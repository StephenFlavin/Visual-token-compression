"""WP-LG2 tests: alphabet codec, class mapping, a tiny joint-training loop, row schema.

If the LG1 modules (lg/glyphs.py, lg/compose.py, lg/channel.py) are not importable yet, minimal
stand-ins defined in this file are installed in ``sys.modules`` (tests only).
"""
from __future__ import annotations

import importlib
import itertools
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch.set_num_threads(1)


# ============================================================================ LG1 stand-ins
class StubGlyphBank(torch.nn.Module):
    STUB = True

    def __init__(self, n_symbols, height_px, widths_px, ss=4, init="random", learnable=True):
        super().__init__()
        self.ss, self.height_ss = ss, height_px * ss
        self.register_buffer("widths_ss", torch.as_tensor(widths_px, dtype=torch.long) * ss)
        wmax = int(self.widths_ss.max())
        cols = torch.arange(wmax)[None, None, :] < self.widths_ss[:, None, None]
        self.register_buffer("mask", cols.expand(n_symbols, self.height_ss, wmax).float())
        logits = torch.randn(n_symbols, self.height_ss, wmax) * 0.5
        self.logits = torch.nn.Parameter(logits, requires_grad=learnable)

    def forward(self, tau: float = 1.0, hard: bool = False):
        p = torch.sigmoid(self.logits / tau)
        if hard:
            p = p + ((p > 0.5).float() - p).detach()
        return p * self.mask

    @classmethod
    def from_font(cls, symbols, font_path, height_px, ss=4, cjk_map=None):
        g = torch.Generator().manual_seed(0)
        widths = [max(1, round(0.5 * height_px)) if len(s) == 1 else height_px for s in symbols]
        b = cls(len(symbols), height_px, widths, ss=ss, learnable=False)
        b.logits.data = torch.randn(b.logits.shape, generator=g) * 4
        return b

    def save(self, path):
        torch.save(self.state_dict(), path)

    def atlas_png(self, path, symbols):
        Path(path).write_bytes(b"")


def stub_compose(ink, ids_batch, widths_ss):
    H = ink.shape[1]
    lines, ws = [], []
    for ids in ids_batch:
        parts = [ink[i, :, : int(widths_ss[i])] for i in ids]
        lines.append(torch.cat(parts, dim=1))
        ws.append(lines[-1].shape[1])
    W = max(ws)
    canvas = torch.zeros(len(ids_batch), 1, H, W)
    for b, ln in enumerate(lines):
        canvas[b, 0, :, : ln.shape[1]] = ln
    return canvas, torch.tensor(ws, dtype=torch.long)


class StubChannel:
    def __init__(self, preset: str = "default", ss: int = 4):
        self.preset, self.ss = preset, ss

    def __call__(self, canvas_ss, widths_ss, generator=None, train=True):
        ss, m = self.ss, 2 * self.ss
        x = F.pad(canvas_ss, (m, m + ss, m, m))
        x = F.avg_pool2d(x, ss)
        if train:
            x = x + 0.03 * torch.randn(x.shape, generator=generator)
        return x.clamp(0, 1), (widths_ss + ss - 1) // ss + 4


def _install_stubs():
    pkg = "learned_glyphs.lg"
    for name, attrs in (("glyphs", {"GlyphBank": StubGlyphBank}),
                        ("compose", {"compose": stub_compose}),
                        ("channel", {"Channel": StubChannel})):
        try:
            importlib.import_module(f"{pkg}.{name}")
        except ImportError:
            mod = types.ModuleType(f"{pkg}.{name}")
            mod.__dict__.update(attrs)
            mod.STUB = True
            sys.modules[f"{pkg}.{name}"] = mod


_install_stubs()

from learned_glyphs.lg.alphabet import Alphabet  # noqa: E402
from learned_glyphs.lg import train as T  # noqa: E402


@pytest.fixture(scope="module")
def lines():
    from vtc.corpus import iter_lines
    return list(itertools.islice(iter_lines("val", seed=3), 400))


# ============================================================================ alphabet
@pytest.mark.parametrize("n", [0, 300])
def test_alphabet_roundtrip(lines, n):
    a = Alphabet.build(n)
    assert len(a) == 95 + n and a.n_chars == 95 and a.n_classes == len(a) + 1
    assert a.symbols[:95] == [chr(c) for c in range(32, 127)]
    for t in lines[:200] + ["", "a", " the and was ", "x~{|}"]:
        ids = a.encode(t)
        assert a.decode(ids) == t
        assert all(0 <= i < len(a) for i in ids)
    if n == 0:
        assert all(not a.is_token(i) for t in lines[:50] for i in a.encode(t))
        assert len(a.encode(lines[0])) == len(lines[0])


def test_alphabet_matches_phase2_segmentation(lines):
    from vtc.dictionary import Dictionary
    a = Alphabet.build(300)
    d = Dictionary.load(300)
    assert a.tokens == d.tokens
    for t in lines[:100]:
        assert [a.symbols[i] for i in a.encode(t)] == d.segment(t)
    assert a.is_token(95) and not a.is_token(94)
    assert any(a.is_token(i) for i in a.encode(lines[0] + " of the "))


def test_alphabet_drops_non_ascii_and_stats(lines):
    a = Alphabet.build(0)
    assert a.decode(a.encode("café 一x")) == "caf x"
    st = Alphabet.build(300).stats(lines[:200])
    assert 0.4 < st["seg_ratio"] < 0.9 and 0.2 < st["token_coverage"] < 0.8
    assert len(st["counts"]) == 395


def test_class_mapping():
    assert Alphabet.to_classes([0, 5, 94]) == [1, 6, 95]
    assert Alphabet.from_classes([1, 0, 6, 95]) == [0, 5, 94]
    tg, tl = T._ctc_targets([[0, 3], [7]])
    assert tg.tolist() == [1, 4, 8] and tl.tolist() == [2, 1]


def test_schedules_and_names():
    cfg = T.JointConfig(steps=100, tau_start=1.0, tau_end=0.2, hard_from=0.6, height_px=7)
    assert T.tau_at(cfg, 1) == pytest.approx(1.0) and T.tau_at(cfg, 100) == pytest.approx(0.2)
    assert not T.hard_at(cfg, 60) and T.hard_at(cfg, 61)
    assert cfg.char_w_px == 3 and cfg.token_w_px == 6 and cfg.em == pytest.approx(7 / 1.2)
    assert cfg.scheme == "lg_learned_n0" and cfg.tag == "lg_learned_n0_h7"
    f = T.JointConfig(learn_glyphs=False, n_tokens=300, height_px=5, channel="harsh")
    assert f.scheme == "lg_fixed_n300" and f.tag == "lg_fixed_n300_h5_harsh"
    assert T.JointConfig.from_dict({**f.__dict__, "junk": 1}) == f


# ============================================================================ training loop
SMALL = dict(channels=(8, 16, 24), d_model=48, hidden=32, layers=1)


def _tiny(tmp_path, **kw):
    base = dict(n_tokens=0, height_px=7, steps=60, batch=8, eval_every=20, n_val=16, n_test=16,
                n_expand_probe=50, bucket_mult=4, lr_reader=3e-3, lr_glyph=0.05, hard_from=0.5,
                run_name="t", out_root=str(tmp_path), model=SMALL, n_samples=3)
    base.update(kw)
    return T.JointConfig(**base)


def test_tiny_training_reduces_loss_and_row_schema(tmp_path, lines):
    cfg = _tiny(tmp_path)
    train = lines[40:]
    rows = T.train_joint(cfg, log=lambda m: None, train_lines=train * 3, eval_lines=(lines[:16], lines[16:32]))
    assert len(rows) == 1
    row = rows[0]
    for k in T.ROW_KEYS:
        assert k in row, k
    assert row["phase"] == "lg" and row["train_mode"] == "joint" and row["scheme"] == "lg_learned_n0"
    assert row["em"] == pytest.approx(7 / 1.2) and row["height_px"] == 7
    assert row["px_per_char"] == pytest.approx(7 * 3, rel=1e-6)    # fixed 3-px cells, 1 symbol/char
    assert row["learn_glyphs"] is True and row["n_tokens"] == 0 and row["steps"] == 60
    curve = [json.loads(x) for x in (tmp_path / "t" / "curves" / f"{cfg.tag}.jsonl").read_text()
             .splitlines()]
    assert [p["step"] for p in curve] == [20, 40, 60]
    assert curve[-1]["train_loss"] < curve[0]["train_loss"]
    assert curve[-1]["hard"] and not curve[0]["hard"]
    out = tmp_path / "t"
    assert (out / "banks" / f"{cfg.tag}.pt").exists() and (out / "readers" / f"{cfg.tag}.pt").exists()
    assert json.loads((out / "runs.jsonl").read_text().splitlines()[0])["tag"] == cfg.tag
    assert (out / "samples" / f"{cfg.tag}.txt").read_text().startswith("REF ")


def test_token_and_fixed_code_paths(tmp_path, lines):
    for kw in (dict(n_tokens=300, oversample_alpha=0.5), dict(n_tokens=300, learn_glyphs=False)):
        cfg = _tiny(tmp_path, steps=6, eval_every=3, **kw)
        row = T.train_joint(cfg, log=lambda m: None, train_lines=lines[40:] * 2,
                            eval_lines=(lines[:8], lines[8:16]), save=False)[0]
        assert row["n_symbols"] == 395 and row["seg_ratio"] < 0.9
        assert (row["glyph_params"] > 0) == cfg.learn_glyphs
        assert row["bits_per_char"] > 0 and row["bits_per_px2"] >= 0
        assert row["min_hamming_ss"] is not None and "cer_heldout" in row
        import inspect
        if "test_only" in inspect.signature(T.lg1()[2].__init__).parameters:
            assert row["cer_heldout"] is not None


def test_readers_resets_and_guard(tmp_path, lines):
    msgs = []
    cfg = _tiny(tmp_path, steps=8, eval_every=4, n_readers=2, reader_resets=2, reset_until=0.5,
                guard_ss=2)
    assert cfg.scheme == "lg_learned_n0_rr2_nr2" and T.reset_steps(cfg) == [2, 4]
    row = T.train_joint(cfg, log=msgs.append, train_lines=lines[40:] * 2,
                        eval_lines=(lines[:8], lines[8:16]))[0]
    assert row["n_readers"] == 2 and row["reader_resets"] == 2
    assert sum("re-initialised" in m for m in msgs) == 2
    from learned_glyphs.lg.glyphs import GlyphBank
    bank = GlyphBank.load(tmp_path / "t" / "banks" / f"{cfg.tag}.pt")
    with torch.no_grad():
        ink = bank(tau=0.2, hard=True)
    assert float(ink[:, :, :2].abs().sum()) == 0.0              # left guard band baked in
    w = bank.widths_ss
    assert all(float(ink[i, :, w[i] - 2: w[i]].sum()) == 0.0 for i in range(len(w)))
    rep = json.loads((tmp_path / "t" / "glyph_report" / f"{cfg.tag}.json").read_text())
    assert "nearest_pairs" in rep and "top_confusions" in rep


def test_penalties():
    k = 4
    big = torch.zeros(1, 16, 16)
    big[0, 4:12, 4:12] = 1
    thin = torch.zeros(1, 16, 16)
    thin[0, :, 7] = 1
    m = torch.ones(1, 16, 16)
    assert float(T.min_feature_penalty(big, m, k)) == 0.0
    assert float(T.min_feature_penalty(thin, m, k)) > 0.05
    assert float(T.min_feature_penalty(1 - thin, m, k)) > 0.05     # thin gap
    ink = torch.stack([torch.zeros(16, 16), big[0], torch.ones(16, 16)])
    mm = torch.ones(3, 16, 16)
    # glyph 0 (space) skipped; glyph 1 coverage .25 inside band; glyph 2 coverage 1 -> .55 over
    assert float(T.coverage_penalty(ink, mm, 0.10, 0.45)) == pytest.approx(0.55 / 2)


def _block_glyph(bits, ss=4):
    """(H, W) 0/1 at 1x -> (H*ss, W*ss) whole-pixel ink at ss resolution."""
    t = torch.as_tensor(bits, dtype=torch.float32)
    return t.repeat_interleave(ss, 0).repeat_interleave(ss, 1)


def test_min_distance_penalty_and_stats():
    ss = 4
    a = _block_glyph([[1, 0], [0, 0], [1, 1]])
    b = _block_glyph([[1, 0], [0, 0], [1, 1]])            # identical to a
    c = _block_glyph([[1, 1], [0, 0], [1, 1]])            # 1 px from a and b
    d = _block_glyph([[0, 1], [1, 1], [0, 0]])            # 5 px from a/b, 4 from c
    wide = torch.zeros(3 * ss, 4 * ss)
    wide[:, : 2 * ss] = a                                 # same pixels as a, but a 4-px cell
    ink = torch.zeros(5, 3 * ss, 4 * ss)
    for i, g in enumerate((a, b, c, d)):
        ink[i, :, : 2 * ss] = g
    ink[4] = wide
    wss = torch.tensor([2 * ss] * 4 + [4 * ss])
    # pairs of the 2-px group: (a,b)=0 (a,c)=1 (b,c)=1, others >= 4; the lone 4-px cell is never compared
    assert float(T.min_distance_penalty(ink, wss, ss, 2.0)) == pytest.approx((2 + 1 + 1) / 4)
    assert float(T.min_distance_penalty(ink, wss, ss, 1.0)) == pytest.approx(1 / 4)
    assert float(T.min_distance_penalty(ink[2:4], wss[2:4], ss, 2.0)) == 0.0
    st = T.distance_stats(ink, wss, ss, list("abcdw"))
    g2 = st["by_width"]["w2px"]
    assert st["identical_pairs"] == 1 and st["min_l1_1x"] == 0.0
    assert (g2["n"], g2["distinct"], g2["pairs_lt_1px"], g2["pairs_lt_2px"]) == (4, 3, 1, 3)
    assert g2["closest"][0][:2] == ["a", "b"] and st["by_width"]["w4px"]["n"] == 1
    # soft relaxation: equals the L1 for binary ink, and coincident glyphs still get a gradient
    x = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    assert torch.equal(T.soft_l1_matrix(x), torch.cdist(x, x, p=1))
    z = (6 * torch.stack([a, a]) - 3).requires_grad_(True)  # two identical, nearly binary glyphs
    pen = T.min_distance_penalty(torch.sigmoid(z), torch.tensor([2 * ss, 2 * ss]), ss, 2.0)
    pen.backward()
    assert float(pen.detach()) > 0.5 and float(z.grad.abs().sum()) > 0


def test_min_dist_option_tag_suffix_and_resume(tmp_path, lines):
    assert T.JointConfig().min_dist_px == 0.0 and T.JointConfig().ckpt_every == 0   # default off
    kw = dict(steps=8, eval_every=2, min_dist_px=2.0, min_dist_weight=1.0, tag_suffix="md", ckpt_every=4)
    data = dict(train_lines=lines[40:] * 2, eval_lines=(lines[:8], lines[8:16]))
    cfg = _tiny(tmp_path / "a", **kw)
    assert cfg.tag == "lg_learned_n0_h7md"
    full = T.train_joint(cfg, log=lambda m: None, **data)[0]
    assert full["scheme"] == "lg_learned_n0md" and "identical_pairs_1x" in full
    assert full["min_l1_1x_same_width"] is not None
    assert not T.ckpt_path(cfg).exists()                       # removed after the row
    cfg2 = _tiny(tmp_path / "b", **kw)
    assert T.train_joint(cfg2, log=lambda m: None, stop_at=5, **data) == []
    assert T.ckpt_path(cfg2).exists()                          # step-4 checkpoint
    msgs = []
    res = T.train_joint(cfg2, log=msgs.append, **data)[0]
    assert any("resumed from step 4" in m for m in msgs)
    assert res["cer"] == full["cer"] and res["best_step"] == full["best_step"]
    b1 = torch.load(tmp_path / "a" / "t" / "banks" / f"{cfg.tag}.pt", weights_only=False)
    b2 = torch.load(tmp_path / "b" / "t" / "banks" / f"{cfg.tag}.pt", weights_only=False)
    k = "logits" if isinstance(b1, dict) and "logits" in b1 else None
    if k:
        assert torch.equal(b1[k], b2[k])
    else:
        for name in b1:
            assert torch.equal(b1[name], b2[name]), name
    curve = (tmp_path / "b" / "t" / "curves" / f"{cfg.tag}.jsonl").read_text().splitlines()
    assert [json.loads(x)["step"] for x in curve] == [2, 4, 6, 8]


def test_channel_ramp_levels():
    Channel = T.lg1()[2]
    cfg = T.JointConfig(channel="default", channel_ramp=0.5, steps=100)
    ramp = T.make_channel_ramp(cfg, Channel, log=lambda m: None)
    if len(ramp) == 1:
        pytest.skip("Channel has no params dataclass (stub)")
    assert len(ramp) == T.RAMP_LEVELS
    assert T.channel_at(cfg, ramp, 1) is ramp[0] and T.channel_at(cfg, ramp, 100) is ramp[-1]
    assert ramp[0].params.noise == (0.0, 0.0) and ramp[-1].params == Channel("default").params


PH2 = ROOT / "results" / "phase2" / "phase2_main" / "checkpoints" / "latin_em7.pt"


@pytest.mark.skipif(not PH2.exists(), reason="no Phase-2 latin checkpoint")
def test_frozen_reader_mode(tmp_path, lines):
    cfg = _tiny(tmp_path, steps=4, eval_every=2, height_px=8, frozen_reader=str(PH2), model={})
    row = T.train_joint(cfg, log=lambda m: None, train_lines=lines[40:],
                        eval_lines=(lines[:8], lines[8:16]), save=False)[0]
    assert row["train_mode"] == "frozen_reader" and row["scheme"] == "lg_learned_n0_frozenreader"
    assert row["glyph_params"] > 0 and row["reader_resets"] == 0
    a = Alphabet.build(0)
    import torch as _t
    vocab = _t.load(PH2, weights_only=False)["vocab"]
    cm = T.ClassMap(a, vocab)
    assert cm.n_classes == len(vocab) + 2
    assert a.decode(cm.decode(cm.targets([a.encode("The cat")])[0].tolist())) == "The cat"
