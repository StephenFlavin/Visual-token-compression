"""Fast tests for the Phase-2 CRNN harness (vbpe/crnn)."""
import json
import string

import numpy as np
import pytest
import torch

from vbpe.crnn import data as D
from vbpe.crnn.model import CRNN, greedy_decode
from vbpe.crnn.train import TrainConfig, evaluate, train_run

torch.set_num_threads(1)

LINES = [
    "The quick brown fox jumps over the lazy dog again.",
    "Pack my box with five dozen liquor jugs, said Ann.",
    "In 1998 the river flooded the old mill (twice).",
    "Sphinx of black quartz, judge my vow; it is late.",
]


def _lineset(scheme, em, lines=LINES, train=False, rp=D.RenderParams()):
    sch = D.resolve_scheme(scheme)
    samples = [D.render_sample(sch, ln, em, rp, D.line_rng(0, "train", i), train)
               for i, ln in enumerate(lines)]
    H = max(rp.max_height(em), max(s[0].shape[0] for s in samples))
    widths = np.array([s[0].shape[1] for s in samples])
    off = np.concatenate([[0], np.cumsum(widths)])
    img = np.full((H, off[-1]), 255, np.uint8)
    for s, o in zip(samples, off[:-1]):
        img[: s[0].shape[0], o:o + s[0].shape[1]] = s[0]
    return D.LineSet(img, off, [s[1] for s in samples], list(lines),
                     np.array([s[2] for s in samples]), np.array([s[3] for s in samples]), {})


def test_render_sample_shapes_and_labels():
    sch = D.resolve_scheme("latin")
    rp = D.RenderParams()
    img, labels, ppc, e = D.render_sample(sch, LINES[0], 6, rp, D.line_rng(0, "train", 0), True)
    assert img.dtype == np.uint8 and img.ndim == 2
    assert img.shape[0] == round(1.2 * e) + 2 * rp.margin
    assert "".join(labels) == sch.transcription_target(LINES[0])
    assert abs(e - 6) <= 6 * rp.em_jitter
    content = sum(u.advance(e) for u in sch.encode(LINES[0]))
    assert ppc == pytest.approx(1.2 * e * content / len(LINES[0]))
    # eval rendering uses the exact em
    _, _, _, e2 = D.render_sample(sch, LINES[0], 6, rp, D.line_rng(0, "test", 0), False)
    assert e2 == 6


def test_subpixel_phase_shifts_ink():
    sch = D.resolve_scheme("latin")
    units = sch.encode("ill ill")
    from vbpe.layout import render_line
    cents = []
    for ph in (0.0, 0.5):
        im = 255 - np.asarray(render_line([D.PhaseOffset(ph, 8)] + units, 8, D.RenderParams().layout()),
                              dtype=float)
        cents.append((im.sum(0) * np.arange(im.shape[1])).sum() / im.sum())
    assert 0.3 < cents[1] - cents[0] < 0.7


def test_lineset_roundtrip_and_cache(tmp_path):
    ls = D.build_lineset("latin", 5, "val", 12, workers=1, cache_dir=tmp_path, allow_fallback=True,
                         verbose=False)
    assert len(ls) == 12 and ls.images.shape[0] == ls.height
    assert ls.offsets[-1] == ls.images.shape[1] and (ls.widths > 0).all()
    ls2 = D.build_lineset("latin", 5, "val", 12, workers=1, cache_dir=tmp_path, allow_fallback=True)
    assert ls2.texts == ls.texts and ls2.labels == ls.labels
    np.testing.assert_array_equal(ls2.images, ls.images)
    x, w, tg, tl = D.collate(ls, np.arange(4), D.Vocab.from_labels(ls.labels))
    assert x.shape == (4, 1, ls.height, int(w.max())) and float(x.max()) <= 1.0
    assert int(tl.sum()) == len(tg)


def test_same_lines_same_jitter_across_schemes():
    a = D.render_sample(D.resolve_scheme("latin"), LINES[1], 7, D.RenderParams(),
                        D.line_rng(0, "train", 3), True)
    b = D.render_sample(D.resolve_scheme("stub_quad"), LINES[1], 7, D.RenderParams(),
                        D.line_rng(0, "train", 3), True)
    assert a[3] == b[3]


@pytest.mark.parametrize("scheme,em", [("latin", 5), ("latin_condensed", 5), ("stub_quad", 5),
                                       ("stub_quad", 8), ("latin", 12)])
def test_frame_rule_T_ge_2L_plus_1(scheme, em):
    ls = _lineset(scheme, em)
    s = D.col_stride_for_em(em)
    e = D.choose_expand(ls.widths, ls.label_lens, s)
    T = D.n_frames(ls.widths, s, e)
    assert (T >= 2 * ls.label_lens + 1).all()
    if e > 1:  # minimal
        assert not (D.n_frames(ls.widths, s, e - 1) >= 2 * ls.label_lens + 1).all()
    m = CRNN(10, col_stride=s, expand=e).eval()
    x, w = D.collate(ls, np.arange(len(ls)))
    with torch.no_grad():
        lp, lens = m(x, w)
    assert lens.tolist() == T.tolist()
    assert lp.shape[0] == int(lens.max()) and lp.shape[1] == len(ls)


def test_vocab_multilabel_units():
    sch = D.resolve_scheme("stub_quad")
    units = sch.encode("abcdef gh")
    assert units[0].labels == ("a", "b", "c", "d") and len(units[0].labels) > 1
    ls = _lineset("stub_quad", 6)
    v = D.Vocab.from_labels(ls.labels)
    assert v.itos[:2] == ["<blank>", "<unk>"] and " " in v.stoi
    ids = v.encode(ls.labels[0])
    assert v.decode(ids) == list(ls.labels[0])
    assert sch.decode("".join(v.decode(ids))) == LINES[0]
    assert v.encode(["☃"]) == [D.Vocab.UNK]


def test_large_vocab_stub():
    sch = D.resolve_scheme("stub_cjk")
    text = " ".join(a + b for a in string.ascii_letters for b in string.ascii_letters)
    labels = [tuple(lab for u in sch.encode(text) for lab in u.labels)]
    v = D.Vocab.from_labels(labels)
    assert len(v) >= 2700
    assert sch.decode("".join(labels[0])) == text


def test_greedy_decode():
    lp = torch.full((6, 1, 4), -10.0)
    for t, k in enumerate([2, 2, 0, 2, 3, 0]):
        lp[t, 0, k] = 0
    assert greedy_decode(lp, torch.tensor([6])) == [[2, 2, 3]]


def test_short_training_reduces_loss():
    torch.manual_seed(0)
    ls = _lineset("latin", 8, LINES * 2, train=True)
    v = D.Vocab.from_labels(ls.labels)
    s = D.col_stride_for_em(8)
    m = CRNN(len(v), col_stride=s, expand=D.choose_expand(ls.widths, ls.label_lens, s))
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    ctc = torch.nn.CTCLoss(zero_infinity=True)
    x, w, tg, tl = D.collate(ls, np.arange(len(ls)), v)
    losses = []
    for _ in range(20):
        lp, lens = m(x, w)
        loss = ctc(lp, tg, lens, tl)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-3:]) < 0.7 * np.mean(losses[:3])


def test_train_run_writes_row(tmp_path):
    tr = _lineset("latin", 6, LINES * 2, train=True)
    va = _lineset("latin", 6, LINES[:2])
    te = _lineset("latin", 6, LINES[2:])
    cfg = TrainConfig(scheme="latin", em=6, steps=4, batch=4, eval_every=2)
    row = train_run(cfg, tmp_path, data=(tr, va, te), log=lambda *_: None)
    for k in ("phase", "scheme", "em", "px_per_char", "cer", "symbol_cer", "steps", "train_time_s",
              "n_params"):
        assert k in row
    assert row["phase"] == 2 and row["n_test"] == 2 and row["px_per_char"] > 0
    rows = [json.loads(x) for x in open(tmp_path / "runs.jsonl")]
    assert rows[0]["scheme"] == "latin"
    assert (tmp_path / "curves" / "latin_em6.jsonl").exists()
    v = D.Vocab.from_labels(tr.labels)
    m = CRNN(len(v), col_stride=2, expand=D.choose_expand(te.widths, te.label_lens, 2))
    ev = evaluate(m, te, v, D.resolve_scheme("latin"))
    assert ev["cer"] >= 0 and ev["n"] == 2
