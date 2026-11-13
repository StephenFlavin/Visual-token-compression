from vtc import metrics
from vtc.layout import LayoutParams, fit_em, render_line, render_page
from vtc.schemes import get_scheme

TEXT = ("The quick brown fox jumps over the lazy dog. " * 40).strip()


def test_latin_fit_and_render():
    s = get_scheme("latin")
    units = s.encode(TEXT)
    img, info = render_page(units, 300, 300)
    assert info["fit"] and info["em"] > 4
    assert img.size == (300, 300)
    assert min(img.getdata()) < 128  # some ink
    # a slightly bigger em must not fit
    assert fit_em(units, 300, 300) == info["em"]


def test_condensed_fits_larger_em():
    a = fit_em(get_scheme("latin").encode(TEXT), 300, 300)
    b = fit_em(get_scheme("latin_condensed").encode(TEXT), 300, 300)
    assert b > a


def test_transcription_target_roundtrip():
    s = get_scheme("latin")
    assert s.decode(s.transcription_target(TEXT)) == TEXT


def test_render_line():
    img = render_line(get_scheme("latin").encode("hello world"), 12.0, LayoutParams(margin=2))
    assert img.size[1] == round(1.2 * 12) + 4


def test_metrics():
    assert metrics.cer("a  b c", "a b c") == 0
    assert abs(metrics.cer("abcd", "abce") - 0.25) < 1e-9
    assert metrics.px_per_char(10, 10, "abcd") == 25
