"""WP-LG3 tests: LearnedGlyphScheme (bridge), transfer loop, report on synthetic rows.

Fast (< ~30 s, 1 torch thread). Tests needing LG1 (lg/glyphs.py, compose.py, channel.py) or LG2
(lg/alphabet.py, lg/train.py) are skipped if those modules are not importable.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(f"learned_glyphs.lg.{mod}") is not None
    except ModuleNotFoundError:
        return False


HAVE_LG1 = all(_have(m) for m in ("glyphs", "compose", "channel"))
HAVE_LG2 = _have("alphabet") and _have("train")
need_lg1 = pytest.mark.skipif(not HAVE_LG1, reason="LG1 modules (glyphs/compose/channel) missing")
need_lg2 = pytest.mark.skipif(not HAVE_LG2, reason="LG2 modules (alphabet/train) missing")

from learned_glyphs.lg import report  # noqa: E402
from learned_glyphs.lg.scheme import (PUA_BASE, LearnedGlyphScheme, label_of,  # noqa: E402
                                      resample_cell)
from vbpe.layout import LayoutParams, render_line, render_page  # noqa: E402

TEXTS = ["The quick brown fox jumps over the lazy dog.",
         "In 1984, the band's album (and the tour) was released; it sold 3,000 copies!",
         " leading and trailing  spaces ", "a"]


# ---------------------------------------------------------------------------- helpers
class _StubAlphabet:
    """Per-char alphabet (95 printable ASCII) + a few tokens, greedy longest-match."""

    def __init__(self, tokens=(" the ", "and ", "ing")):
        self.symbols = [chr(c) for c in range(32, 127)] + list(tokens)
        self.n_tokens = len(tokens)
        self.index = {s: i for i, s in enumerate(self.symbols)}
        self._toks = sorted(tokens, key=len, reverse=True)

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            for t in self._toks:
                if text.startswith(t, i):
                    out.append(self.index[t])
                    i += len(t)
                    break
            else:
                if 32 <= ord(text[i]) < 127:
                    out.append(self.index[text[i]])
                i += 1
        return out


def _alphabet(n_tokens: int = 0):
    if HAVE_LG2:
        from learned_glyphs.lg.alphabet import Alphabet
        return Alphabet.build(n_tokens)
    return _StubAlphabet(())


def _random_bitmaps(alpha, h_px=6, ss=4, seed=0):
    rng = np.random.default_rng(seed)
    hs = h_px * ss
    maps = []
    for s in alpha.symbols:
        w = max(1, round((0.45 if len(s) == 1 else 0.9) * h_px)) * ss
        maps.append((rng.random((hs, w)) > 0.6).astype(float))
    return maps, hs, ss


@pytest.fixture(scope="module")
def stub_scheme():
    alpha = _StubAlphabet()
    maps, hs, ss = _random_bitmaps(alpha)
    return LearnedGlyphScheme.from_bitmaps(maps, hs, ss, alphabet=alpha, name="lg:test")


# ---------------------------------------------------------------------------- scheme
def test_labels_are_one_pua_codepoint_per_symbol(stub_scheme):
    units = stub_scheme.encode("and the thing")
    assert all(len(u.labels) == 1 and len(u.labels[0]) == 1 for u in units)
    assert all(ord(u.labels[0]) >= PUA_BASE for u in units)
    assert "".join(u.source for u in units) == "and the thing"
    assert any(len(u.source) > 1 for u in units)          # tokens are single units
    assert label_of(0) == ""


@pytest.mark.parametrize("text", TEXTS)
def test_round_trip_stub(stub_scheme, text):
    assert stub_scheme.decode(stub_scheme.transcription_target(text)) == text


@need_lg2
@pytest.mark.parametrize("n_tokens", [0, 300])
def test_round_trip_real_alphabet(n_tokens):
    alpha = _alphabet(n_tokens)
    maps, hs, ss = _random_bitmaps(alpha)
    sch = LearnedGlyphScheme.from_bitmaps(maps, hs, ss, alphabet=alpha)
    for t in TEXTS:
        assert sch.decode(sch.transcription_target(t)) == t
    if n_tokens:
        assert len(sch.encode(TEXTS[0])) < len(TEXTS[0])


def test_space_unit_flags(stub_scheme):
    u = [x for x in stub_scheme.encode("a b") if x.source == " "][0]
    assert u.is_space


def test_advance_proportional_to_em(stub_scheme):
    units = stub_scheme.encode(TEXTS[0])
    a5 = np.array([u.advance(5.0) for u in units])
    a8 = np.array([u.advance(8.0) for u in units])
    np.testing.assert_allclose(a8, a5 * 8 / 5)
    # advance = width_ss / height_ss * 1.2 em for learned banks (em_ref = h / 1.2)
    u = units[0]
    w = stub_scheme.bitmaps[u.sid].shape[1]
    assert math.isclose(u.advance(5.0), w / stub_scheme.height_ss * 1.2 * 5.0)


@pytest.mark.parametrize("em", [3.3, 4.0, 5.0, 6.7, 9.0])
def test_ink_within_line_box(stub_scheme, em):
    """Ink stays inside the cell = one line pitch [baseline - 0.9 em, baseline + 0.3 em]."""
    p = LayoutParams(margin=2, supersample=4)
    img = render_line(stub_scheme.encode(TEXTS[1]), em, p)
    ink = 255 - np.asarray(img, dtype=float)
    rows = np.nonzero(ink.max(axis=1) > 0)[0]
    base = p.first_baseline(em)
    top, bot = stub_scheme.ink_box_em()
    assert rows.min() >= math.floor(base + top * em) and rows.max() + 1 <= math.ceil(base + bot * em)
    assert img.size[1] == int(round(1.2 * em)) + 4          # Phase-2 line height unchanged
    cols = np.nonzero(ink.max(axis=0) > 0)[0]
    width = sum(u.advance(em) for u in stub_scheme.encode(TEXTS[1]))
    assert cols.max() + 1 <= math.ceil(2 + width) + 1


def test_page_render_fits(stub_scheme):
    img, info = render_page(stub_scheme.encode(" ".join(TEXTS * 4)), 280, 140)
    assert info["fit"] and info["n_lines"] >= 2


def test_resample_exact_and_mass_preserving():
    rng = np.random.default_rng(1)
    b = (rng.random((24, 13)) > 0.5).astype(float)
    np.testing.assert_array_equal(resample_cell(b, 1.0), b)
    for s, fx, fy in [(0.25, 0.0, 0.0), (0.37, 0.5, 0.25), (1.6, 0.75, 0.0)]:
        r = resample_cell(b, s, fx, fy)
        assert r.min() >= 0 and r.max() <= 1 + 1e-9
        assert math.isclose(r.sum(), b.sum() * s * s, rel_tol=1e-9)
    # 4x box downsample == mean pooling
    r = resample_cell(b[:, :12], 0.25)
    np.testing.assert_allclose(r, b[:, :12].reshape(6, 4, 3, 4).mean(axis=(1, 3)))


# ---------------------------------------------------------------------------- with LG1
@need_lg1
@need_lg2
@pytest.mark.parametrize("kind", ["learned", "fixed"])
def test_native_em_matches_bank_rendering(kind):
    """At the bank's native em the scheme reproduces lg.compose (ss res) and, through the Phase-2
    supersample-4 render, the clean Channel at phase 0 (up to 8-bit rounding)."""
    from learned_glyphs.lg.channel import Channel
    from learned_glyphs.lg.compose import compose
    from learned_glyphs.lg.glyphs import GlyphBank

    alpha = _alphabet(300 if kind == "learned" else 0)
    h = 6
    if kind == "learned":
        bank = GlyphBank(len(alpha), h, [round(0.45 * h)] * 95 + [round(0.9 * h)] * alpha.n_tokens,
                         init="random", symbols=alpha.symbols, seed=3)
    else:
        bank = GlyphBank.from_font(alpha.symbols, height_px=h)
    sch = LearnedGlyphScheme(bank, alpha)
    assert math.isclose(sch.native_em, h / 1.2)
    for t in TEXTS[:2]:
        ids = alpha.encode(t)
        with torch.no_grad():
            can, wss = compose(bank(hard=True), [ids], bank.widths_ss)
        ref = can[0, 0].numpy()
        img = render_line(sch.encode(t), sch.native_em * bank.ss, LayoutParams(margin=0, supersample=1))
        got = 1 - np.asarray(img, dtype=float) / 255
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() <= 0.5 / 255 + 1e-6
        x, _ = Channel("clean", ss=bank.ss)(can, wss, train=False, phase=0)
        img2 = render_line(sch.encode(t), sch.native_em, LayoutParams(margin=2, supersample=4))
        g2 = 1 - np.asarray(img2, dtype=float) / 255
        assert np.abs(x[0, 0].numpy()[:, :g2.shape[1]] - g2).max() <= 1.5 / 255


@need_lg1
@need_lg2
def test_bank_path_and_registry_name(tmp_path, monkeypatch):
    from learned_glyphs.lg import scheme as S
    from learned_glyphs.lg.glyphs import GlyphBank

    monkeypatch.setattr(S, "RESULTS", tmp_path)
    alpha = _alphabet(0)
    bank = GlyphBank(len(alpha), 5, 2, init="random", symbols=alpha.symbols)
    p = tmp_path / "runA" / "banks" / "lg_learned_n0_h5.pt"
    bank.save(p)
    name = S.scheme_name_for(p)
    assert "/" not in name and name.startswith("lg:runA+banks+lg_learned_n0_h5.pt@")
    S.make_lg_scheme.cache_clear()
    sch = S.make_lg_scheme(name)
    assert sch.symbols == alpha.symbols and sch.decode(sch.transcription_target("hi there")) == "hi there"
    with pytest.raises(ValueError):
        S.make_lg_scheme(name.split("@")[0] + "@00000000")
    # runtime Phase-2 hook resolves the name through vbpe.crnn.data
    from vbpe.crnn import data as D
    monkeypatch.setattr(D, "resolve_scheme", D.resolve_scheme)
    S.install_phase2_hook()
    assert D.resolve_scheme(name) is sch
    assert D.resolve_scheme("latin").name == "latin"
    img, labels, ppc, e = D.render_sample(sch, "hello world", 4.0, D.RenderParams(),
                                          D.line_rng(0, "train", 0), train=True)
    assert img.shape[0] == int(round(1.2 * e)) + 4 and len(labels) == len("hello world")
    assert sch.decode("".join(labels)) == "hello world"


@need_lg1
@need_lg2
def test_transfer_loop_tiny(tmp_path):
    from learned_glyphs.lg.glyphs import GlyphBank
    from learned_glyphs.lg.transfer import TransferConfig, train_reader_on_bank

    alpha = _alphabet(0)
    bank = GlyphBank.from_font(alpha.symbols, height_px=6, learnable=True)
    p = tmp_path / "joint" / "banks" / "lg_learned_n0_h6.pt"
    bank.save(p)
    lines = [t.strip() for t in TEXTS[:2]] * 20
    cfg = TransferConfig(steps=4, eval_every=2, batch=4, run_name="tr", out_root=str(tmp_path),
                         n_expand_probe=20, bucket_mult=2)
    rows = train_reader_on_bank(p, cfg, train_lines=lines, eval_lines=(lines[:4], lines[:6]))
    r = rows[0]
    for k in ("phase", "scheme", "em", "height_px", "px_per_char", "cer", "symbol_cer", "steps",
              "train_mode", "channel", "n_tokens", "learn_glyphs", "train_time_s"):
        assert k in r
    assert r["phase"] == "lg" and r["train_mode"] == "transfer" and r["scheme"] == "lg_learned_n0"
    assert math.isclose(r["em"], 5.0) and r["height_px"] == 6
    saved = [json.loads(x) for x in (tmp_path / "tr" / "runs.jsonl").read_text().splitlines()]
    assert saved[0]["bank_sha8"] == r["bank_sha8"]
    assert (tmp_path / "tr" / "samples" / "transfer_lg_learned_n0_h6.txt").exists()


# ---------------------------------------------------------------------------- report
def _curve_rows(scheme, mode, p3, channel="default", ems=tuple(np.arange(2.0, 9.01, 0.25)),
                phase="lg"):
    """Synthetic rows whose CER crosses 3 % at px_per_char = p3 (CER = 0.03 * (p3/x)^3)."""
    rows = []
    for em in ems:
        x = 0.75 * em * em                     # any increasing px/char grid (3 .. 61)
        rows.append({"phase": phase, "run": f"syn_{mode}", "train_mode": mode, "scheme": scheme,
                     "em": float(em), "height_px": 1.2 * em, "px_per_char": x,
                     "cer": min(1.0, 0.03 * (p3 / x) ** 3), "symbol_cer": 0.0, "steps": 10,
                     "channel": channel, "n_tokens": 0, "learn_glyphs": True, "train_time_s": 1.0})
    return rows


def _syn_rows(go: bool):
    k = 0.6 if go else 0.9
    rows = []
    rows += _curve_rows("latin", "per_em", 14.0, phase=2)
    rows += _curve_rows("lg_fixed_n0", "joint", 15.0)
    rows += _curve_rows("lg_fixed_n0", "joint", 25.0, channel="harsh")
    for mode in ("joint", "transfer", "bridge"):
        rows += _curve_rows("lg_learned_n0", mode, 12.0, channel="phase2" if mode == "bridge" else "default")
        rows += _curve_rows("lg_learned_n300", mode, 14.0 * k, channel="phase2" if mode == "bridge" else "default")
    rows += _curve_rows("lg_learned_n0", "transfer", 22.0, channel="harsh")
    rows += _curve_rows("lg_learned_n300", "transfer", 25.0 * k, channel="harsh")
    for r in rows:
        if r["phase"] == 2:
            for k2 in ("height_px", "channel", "n_tokens", "learn_glyphs"):
                r.pop(k2)
    return rows


def test_display_names():
    assert report.display_name({"phase": 2, "scheme": "latin"}) == "latin"
    assert report.display_name({"phase": "lg", "scheme": "lg_learned_n0", "train_mode": "transfer",
                                "channel": "harsh"}) == "lg_learned_n0/transfer/harsh"
    assert report.display_name({"phase": "lg", "scheme": "lg_learned_n0", "train_mode": "bridge",
                                "channel": "phase2"}) == "lg_learned_n0/bridge"


@pytest.mark.parametrize("go", [True, False])
def test_decision_on_synthetic_rows(go):
    rows = report.analysis_rows(_syn_rows(go))
    dec = report.decision(rows)
    t = dec["modes"]["transfer"]
    assert math.isclose(t["P_learned_n0"][0], 12.0, rel_tol=0.05)
    assert math.isclose(t["S1"][0], 12.0 / 15.0, rel_tol=0.05)
    assert math.isclose(t["S2"][0], 14.0 * (0.6 if go else 0.9) / 12.0, rel_tol=0.05)
    assert math.isclose(dec["best_phase2_latin"][0], 14.0, rel_tol=0.05)
    assert dec["GO"] is go
    md = "\n".join(report.decision_md(dec))
    assert ("Verdict: GO" in md) is go


def test_write_report_synthetic(tmp_path):
    res = report.write_report(tmp_path / "summary", rows=_syn_rows(True), n_boot=50, samples=False)
    md = (tmp_path / "summary" / "summary.md").read_text()
    assert "## Decision (PLAN criteria)" in md and "lg_learned_n300/transfer" in md
    assert (tmp_path / "summary" / "cer_curves.png").exists()
    assert res["decision"]["GO"] is True


def test_confusions():
    pairs = [("hello world", "hellu warld"), ("abc", "abd"), ("the cat", "tha cat")]
    c = report.confusions(pairs)
    assert c["n_ref"] == 21 and c["n_edits"] == 4
    assert dict(c["subs"])[("o", "u")] == 1 and dict(c["subs"])[("e", "a")] == 1


def test_read_samples(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("REF hello there\nHYP helo there\n\nREF abc\nHYP \n\n")
    assert report.read_samples(f) == [("hello there", "helo there"), ("abc", "")]


@need_lg1
@need_lg2
def test_atlas_comparison(tmp_path):
    from learned_glyphs.lg.glyphs import GlyphBank
    alpha = _alphabet(0)
    learned = GlyphBank(len(alpha), 6, 3, init="random", symbols=alpha.symbols)
    p = report.atlas_comparison([("learned", learned)], tmp_path / "atlas.png", symbols=list("abcxyz"))
    from PIL import Image
    im = Image.open(p)
    assert im.size[0] > 100 and im.size[1] > 100
