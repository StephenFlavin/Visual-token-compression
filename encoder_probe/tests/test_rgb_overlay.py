"""Fast checks for the RGB overlay encodings (no model downloads)."""
import numpy as np
import pytest

from encoder_probe.ep.pages import H, W, Encoding, get_encoding, make_pages
from encoder_probe.ep.probe import edit_distance
from encoder_probe.ep.rgb_overlay import (T_FRAMES, RGB3Encoding, ctc_ok, get_rgb_encoding, mono_cells,
                                          mono_ink, split_streams)

PITCHES = (5, 6, 8, 10)
NEW_NAMES = ([f"mono_p{p}" for p in PITCHES] + [f"rgb3_p{p}" for p in PITCHES] + ["rgb3a_p6"]
             + [f"rgbw_p{p}_{m}" for p in (6, 8) for m in ("chan", "paint")] + ["rgbw_p8", "rgbw_p8_paint_s2"])


def _check_page(enc, pg):
    assert pg.img.shape == (H, W, 3) and pg.img.dtype == np.uint8
    assert len(pg.labels) == len(pg.refs) == enc.n_lines
    cls = set(enc.classes)
    assert all(isinstance(lab, tuple) and set(lab) <= cls for lab in pg.labels)
    for lab, r in zip(pg.labels, pg.refs):
        assert enc.clean_hyp(enc.decode("".join(lab))) == r     # a perfect read scores 0 CER
        assert ctc_ok(len(lab))


@pytest.mark.parametrize("p", PITCHES)
def test_rgb3_channels_are_independent_mono_streams(p):
    enc = get_rgb_encoding(f"rgb3_p{p}")
    pg = make_pages(enc, "test", 1)[0]
    n = enc.n_cells
    streams = [split_streams(r, n) for r in pg.refs]
    for c in range(3):
        alone = mono_ink([s[c] for s in streams], p)            # this stream alone, as a mono page
        assert (alone == pg.img[:, :, c]).all()
    r_text = "".join(s[0] for s in streams)                     # all lines are full here
    mono, used = get_rgb_encoding(f"mono_p{p}").render(r_text, 0)
    assert used == len(r_text) and (mono.img[:, :, 0] == pg.img[:, :, 0]).all()   # R = mono page of stream R


@pytest.mark.parametrize("p", PITCHES)
def test_rgb3_labels_decode_and_fit_ctc(p):
    enc = get_rgb_encoding(f"rgb3_p{p}")
    assert 2 * 3 * enc.n_cells + 1 <= T_FRAMES
    pg = make_pages(enc, "test", 2)
    for page in pg:
        _check_page(enc, page)
        for lab, r in zip(page.labels, page.refs):
            n = enc.n_cells
            assert len(lab) == len(r) <= 3 * n               # the last lines can be short if the text runs out
            if len(r) == 3 * n:
                assert [x[1] for x in lab[:6]] == [r[0], r[n], r[2 * n], r[1], r[n + 1], r[2 * n + 1]]


def test_rgb3_tagged_decode_is_robust_to_one_dropped_label():
    enc = get_rgb_encoding("rgb3_p6")
    lab, ref = (lambda pg: (pg.labels[0], pg.refs[0]))(make_pages(enc, "test", 1)[0])
    hyp = enc.decode("".join(lab[:10] + lab[11:]))
    assert edit_distance(ref, hyp) == 1


def test_rgb3_plain_ascii_variant_roundtrip():
    enc = get_rgb_encoding("rgb3a_p6")
    assert enc.classes == get_rgb_encoding("mono_p6").classes
    for page in make_pages(enc, "test", 1):
        _check_page(enc, page)
    n = enc.n_cells
    for L in (5, n + 3, 2 * n + 7):                                # short last lines decode exactly too
        chunk = "".join(chr(33 + i % 90) for i in range(L))
        st = split_streams(chunk, n)
        il = "".join(st[c][k] for k in range(n) for c in range(3) if k < len(st[c]))
        assert enc.decode(il) == chunk


def test_rgb3_unsupported_pitch():
    assert (mono_cells(3), mono_cells(4)) == (148, 111)
    RGB3Encoding(4)                                    # 2*333+1 = 667 <= 672 frames
    with pytest.raises(ValueError):
        RGB3Encoding(3)                                # 2*444+1 = 889 > 672


@pytest.mark.parametrize("p", PITCHES)
def test_rgb3_holds_three_times_mono(p):
    a = sum(pg.n_chars for pg in make_pages(get_rgb_encoding(f"rgb3_p{p}"), "test", 2))
    b = sum(pg.n_chars for pg in make_pages(get_rgb_encoding(f"mono_p{p}"), "test", 2))
    assert abs(a / b - 3) < 0.15


@pytest.mark.parametrize("p", PITCHES)
def test_mono_pages(p):
    enc = get_rgb_encoding(f"mono_p{p}")
    pg = make_pages(enc, "test", 1)[0]
    _check_page(enc, pg)
    assert (pg.img[..., 0] == pg.img[..., 1]).all() and (pg.img[..., 0] == pg.img[..., 2]).all()
    assert all(len(r) == enc.n_cells for r in pg.refs)


@pytest.mark.parametrize("name", ["rgbw_p6_chan", "rgbw_p8_paint", "rgbw_p10_chan"])
def test_rgbw_pages(name):
    enc = get_rgb_encoding(name)
    pg = make_pages(enc, "test", 2)
    for page in pg:
        _check_page(enc, page)
        assert all(r == r.strip() and "  " not in r for r in page.refs)
    joined = " ".join(r for page in pg for r in page.refs)
    assert joined.startswith("Robert Boulter is an English film")          # source order, nothing skipped


def test_rgbw_slots_and_modes():
    chan, paint = get_rgb_encoding("rgbw_p8_chan"), get_rgb_encoding("rgbw_p8_paint")
    text = "hello my name is jimmy and so on " * 50
    placed, refs, _ = chan.layout(text, 0)
    (l0, x0, c0, w0), (_, x1, c1, w1), (_, x2, c2, w2), (_, x3, c3, w3) = placed[:4]
    assert (w0, w1, w2, w3) == ("hello", "my", "name", "is") and (c0, c1, c2, c3) == (0, 1, 2, 0)
    assert x1 - x0 == x2 - x1 == 4 and x3 > x2                    # 1 px shift at 4x supersampling
    assert refs[0].startswith("hello my name is jimmy")
    a, b = chan.render(text, 0)[0].img, paint.render(text, 0)[0].img
    assert not (a == b).all()
    px = b.reshape(-1, 3).astype(int)
    assert ((px[:, 0] > px[:, 1] + 80) & (px[:, 0] > px[:, 2] + 80)).any()   # some red ink stays visible
    s2 = get_rgb_encoding("rgbw_p8_paint_s2")
    assert s2.shift == 2 and s2.layout(text, 0)[0][1][1] - s2.layout(text, 0)[0][0][1] == 8


def test_get_encoding_resolves_new_names():
    for name in NEW_NAMES:
        enc = get_encoding(name)
        assert isinstance(enc, Encoding) and enc.pitch == int(name.split("_p")[1].split("_")[0])
    with pytest.raises(KeyError):
        get_encoding("rgbx_p6")
