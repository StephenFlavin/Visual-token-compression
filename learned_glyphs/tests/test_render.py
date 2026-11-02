"""WP-LG1 tests: GlyphBank, compose, Channel (+ the fixed DejaVu bank vs Phase-2 PIL rendering)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_glyphs.lg.channel import Channel, jpeg_approx, pil_jpeg  # noqa: E402
from learned_glyphs.lg.compose import compose, line_widths_ss  # noqa: E402
from learned_glyphs.lg.glyphs import GlyphBank, coverage, min_feature_penalty  # noqa: E402
from vbpe import config  # noqa: E402

torch.set_num_threads(1)
SYMS = [chr(c) for c in range(32, 127)]
S2I = {s: i for i, s in enumerate(SYMS)}
LINES = ["At the start of the 1960s, Prime Minister", "quick brown fox jumps over",
         "Hello, world! (42)"]
HAVE_FONT = Path(config.LATIN_FONT).exists()


def _ids(lines=LINES):
    return [[S2I[c] for c in t] for t in lines]


# ------------------------------------------------------------------ GlyphBank
def test_bank_shapes_mask_and_range():
    bank = GlyphBank(5, 6, [2.0, 3.0, 2.5, 4.0, 1.0], ss=4, guard_ss=0)
    assert bank.widths_ss.tolist() == [8, 12, 10, 16, 4]
    ink = bank(tau=0.5).detach()
    assert ink.shape == (5, 24, 16)
    assert float(ink.min()) >= 0 and float(ink.max()) <= 1
    for i, w in enumerate(bank.widths_ss.tolist()):
        assert float(ink[i, :, w:].abs().sum()) == 0.0


def test_guard_band_zero_ink():
    bank = GlyphBank(3, 5, 3.0, ss=4, init_bias=3.0, init_std=0.0)   # default guard (1, 1)
    ink = bank().detach()
    assert bank.guard_ss == (1, 1)
    assert float(ink[:, 0].sum()) == 0 and float(ink[:, -1].sum()) == 0
    assert float(ink[:, :, 0].sum()) == 0 and float(ink[:, :, 11].sum()) == 0
    assert float(ink[:, 1:-1, 1:11].min()) > 0.9


def test_hard_straight_through():
    bank = GlyphBank(4, 5, 2.0, guard_ss=0, init_std=1.0)
    soft = bank(tau=1.0)
    hard = bank(tau=1.0, hard=True)
    assert set(torch.unique(hard).tolist()) <= {0.0, 1.0}
    assert torch.equal(hard, (soft > 0.5).float() * bank.mask)
    w = torch.randn_like(hard)
    (hard * w).sum().backward()
    g_hard = bank.logits.grad.clone()
    bank.logits.grad = None
    (bank(tau=1.0) * w).sum().backward()
    assert torch.allclose(g_hard, bank.logits.grad)            # sigmoid gradient passes through
    assert float(g_hard.abs().sum()) > 0


def test_save_load_roundtrip(tmp_path):
    bank = GlyphBank(6, 5, 2.25, symbols=list("abcdef"), seed=3)
    bank.save(tmp_path / "b.pt")
    b2 = GlyphBank.load(tmp_path / "b.pt")
    assert torch.allclose(bank(0.7, True), b2(0.7, True)) and b2.symbols == list("abcdef")
    frozen = GlyphBank.load(tmp_path / "b.pt", learnable=False)
    assert not any(p.requires_grad for p in frozen.parameters())
    bank.atlas_png(tmp_path / "atlas.png")
    assert (tmp_path / "atlas.png").stat().st_size > 0


def test_min_feature_penalty_and_coverage():
    x = torch.zeros(2, 12, 12)
    x[0, 2:6, 2:6] = 1                     # one output pixel (4x4 ss) block: no sub-pixel detail
    x[1, 2:6, 2:6] = 1
    x[1, 8, 8] = 1                         # single ss dot: sub-pixel detail
    p = min_feature_penalty(x, 4, reduction="none")
    assert float(p[0]) == 0.0 and float(p[1]) > 0
    xg = x.clone().requires_grad_()
    min_feature_penalty(xg, 4).backward()
    assert float(xg.grad.abs().sum()) > 0
    c = coverage(x, torch.tensor([12, 6]))
    assert torch.allclose(c, torch.tensor([16 / 144, 17 / 72]))


def test_random_binary_bank():
    bank = GlyphBank.random_binary(40, 6, 3.0, ss=4, density=0.3, seed=1)
    ink = bank()
    assert not any(p.requires_grad for p in bank.parameters())
    assert set(torch.unique(ink).tolist()) <= {0.0, 1.0}
    assert float(bank.min_feature_penalty()) == 0.0                  # >= 1 output px features
    assert float(ink[0].sum()) == 0                                   # space is blank
    flat = {ink[i].numpy().tobytes() for i in range(1, 40)}
    assert len(flat) == 39                                            # all distinct
    assert 0.1 < float(bank.coverage()[1:].mean()) < 0.45


# ------------------------------------------------------------------ compose
def test_compose_offsets_and_widths():
    n, Hs = 4, 8
    widths = torch.tensor([3, 5, 2, 4])
    ink = torch.zeros(n, Hs, 5)
    for i in range(n):
        ink[i, :, : int(widths[i])] = i + 1        # value = symbol id + 1 (fills its cell)
    ids = [[0, 1, 2], [3], [], [2, 2, 1, 0]]
    cv, w = compose(ink, ids, widths)
    assert w.tolist() == [10, 4, 0, 12]
    assert cv.shape == (4, 1, Hs, 12)
    row = cv[0, 0, 0].tolist()
    assert row == [1] * 3 + [2] * 5 + [3] * 2 + [0, 0]
    assert cv[2].abs().sum() == 0
    assert cv[3, 0, 3].tolist() == [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1]
    assert line_widths_ss(ids, widths).tolist() == [10, 4, 0, 12]
    cvm, wm = compose(ink, ids, widths, margin_ss=2)
    assert cvm.shape == (4, 1, Hs + 4, 16) and wm.tolist() == [14, 8, 4, 16]
    assert cvm[0, 0, 2, 2:14].tolist() == row[:10] + [0, 0] and cvm[0, 0, 0].abs().sum() == 0


def test_compose_differentiable():
    ink = torch.rand(3, 4, 6, requires_grad=True)
    cv, _ = compose(ink, [[0, 2, 0], [1]], torch.tensor([6, 3, 2]))
    cv.sum().backward()
    g = ink.grad
    assert torch.all(g[0] == 2) and torch.all(g[2, :, :2] == 1) and torch.all(g[2, :, 2:] == 0)
    assert torch.all(g[1, :, :3] == 1) and torch.all(g[1, :, 3:] == 0)


# ------------------------------------------------------------------ channel
@pytest.mark.parametrize("preset", ["clean", "default", "harsh"])
def test_gradient_flows_bank_to_output(preset):
    bank = GlyphBank(95, 6, 2.5, ss=4)
    cv, w = compose(bank(tau=0.5), _ids(), bank.widths_ss)
    x, wpx = Channel(preset)(cv, w, generator=torch.Generator().manual_seed(0), train=True)
    (x * torch.randn_like(x)).sum().backward()
    g = bank.logits.grad
    used = sorted({i for t in _ids() for i in t})
    assert float(g[used].abs().sum()) > 0
    unused = [i for i in range(95) if i not in used]
    assert float(g[unused].abs().sum()) == 0


@pytest.mark.parametrize("preset", ["clean", "default", "harsh", "harsh+heldout"])
def test_output_convention_and_eval_determinism(preset):
    bank = GlyphBank(95, 6, 2.5, ss=4, init_bias=-4.0, init_std=0.0)    # ~blank glyphs
    cv, w = compose(bank().detach(), _ids(), bank.widths_ss)
    ch = Channel(preset)
    x1, w1 = ch(cv, w, train=False)
    x2, w2 = ch(cv, w, train=False)
    assert torch.equal(x1, x2) and torch.equal(w1, w2)
    assert x1.dtype == torch.float32 and x1.shape[:2] == (3, 1)
    assert x1.shape[2] == 6 + 2 * 2                                     # height_px + 2 margin
    assert w1.tolist() == [int(-(-int(v) // 4)) + 4 for v in w]
    assert x1.shape[3] == max(w1.tolist())
    assert float(x1.min()) >= 0 and float(x1.max()) <= 1
    assert torch.allclose(x1 * 255, torch.round(x1 * 255), atol=1e-3)   # 8-bit
    if preset == "clean":                                            # blank page -> all background
        assert float(x1.max()) < 0.05
    # train mode is random (except test-only channels, which refuse)
    if ch.test_only:
        with pytest.raises(ValueError):
            ch(cv, w, train=True)
    else:
        a, _ = ch(cv, w, generator=torch.Generator().manual_seed(1), train=True)
        b, _ = ch(cv, w, generator=torch.Generator().manual_seed(2), train=True)
        assert a.shape == x1.shape
        if preset != "clean":
            assert not torch.equal(a, b)


def test_ink_is_one_background_zero():
    ink = torch.ones(1, 8, 8)
    cv, w = compose(ink, [[0, 0]], torch.tensor([8]))
    x, _ = Channel("clean")(cv, w, train=False, phase=0)
    assert float(x[0, 0, 2:4, 2:6].min()) == 1.0         # solid ink -> 1 (margin 2 px)
    assert float(x[0, 0, 0].max()) == 0.0 and float(x[0, 0, :, 0].max()) == 0.0


def test_phase_shift_moves_ink():
    ink = torch.zeros(1, 8, 4)
    ink[0, :, 3] = 1                     # last ss column of the cell: output px 2 (margin 2 px)
    cv, w = compose(ink, [[0]], torch.tensor([4]))
    ch = Channel("clean", quantize=False)
    x0, _ = ch(cv, w, train=False, phase=0)
    x3, _ = ch(cv, w, train=False, phase=torch.tensor([[1, 0]]))     # +1 ss px -> next px
    assert float(x0[0, 0, 2, 2]) == pytest.approx(0.25) and float(x0[0, 0, 2, 3]) == 0
    assert float(x3[0, 0, 2, 2]) == 0 and float(x3[0, 0, 2, 3]) == pytest.approx(0.25)


def test_jpeg_approx_close_to_pil():
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(2, 1, 16, 40, generator=g) > 0.8).float()
    for q in (50, 90):
        a = jpeg_approx(x, torch.full((2,), float(q))).clamp(0, 1)
        b = pil_jpeg(x, q)
        assert float((a - b).abs().mean()) < 0.5 * float((b - x).abs().mean())


# ------------------------------------------------------------------ calibration vs Phase 2
@pytest.mark.skipif(not HAVE_FONT, reason="DejaVu font missing")
def test_fixed_dejavu_bank_matches_phase2_pil_em8():
    """Fixed DejaVu bank + clean channel (phase 0) vs vbpe Phase-2 PIL rendering (phase 0) at em 8.

    Threshold: relative MAD (sum|a-b| / sum a) over the first 12 chars must be < 0.25. Rationale:
    shifting the SAME PIL rendering by a quarter pixel (the Phase-2 phase resolution, which the
    reader must be invariant to) already gives ~0.33; the bank's residual (~0.19) comes from glyph
    origins snapped to the ss grid (<= 1/8 px) and cell-clipped side bearings. Only the prefix is
    compared because rounding each advance to whole ss pixels accumulates drift along the line
    (random walk, ~0.5 px after 50 chars). Mean ink must agree within 3%.
    """
    from vbpe.crnn import data as D
    from vbpe.layout import render_line

    em = 8
    bank = GlyphBank.from_font(SYMS, config.LATIN_FONT, 10, ss=4, em=em)
    assert bank.height_px == 10 and not any(p.requires_grad for p in bank.parameters())
    cv, w = compose(bank(), _ids(), bank.widths_ss)
    x, wpx = Channel("clean")(cv, w, train=False, phase=0)
    scheme = D.resolve_scheme("latin")
    rp = D.RenderParams(phase=False)
    for i, t in enumerate(LINES):
        a = (255 - np.asarray(render_line(scheme.encode(t), em, rp.layout()), np.float32)) / 255
        b = x[i, 0].numpy()
        assert a.shape[0] == b.shape[0]                              # round(1.2 em) + 2 margin
        assert abs(a.shape[1] - int(wpx[i])) <= 2
        Wp = 2 + int(np.ceil(sum(int(bank.widths_ss[S2I[c]]) for c in t[:12]) / 4))
        rel = np.abs(a[:, :Wp] - b[:, :Wp]).sum() / a[:, :Wp].sum()
        assert rel < 0.25, (t, rel)
        W = min(a.shape[1], int(wpx[i]))
        assert abs(a[:, :W].mean() - b[:, :W].mean()) / a[:, :W].mean() < 0.03
