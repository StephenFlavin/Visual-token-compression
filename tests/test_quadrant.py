import time

import numpy as np
import pytest

from vbpe.layout import LayoutParams, render_line, render_page
from vbpe.schemes import get_scheme
from vbpe.schemes.quadrant import QuadCell, QuadrantScheme

TEXT = ("The quick brown fox (aged 7) jumps over the lazy dog; a I, x... "
        "Internationalization is 20-character-long! ") * 12
NAMES = ["quad2", "quad3", "quad4"]


@pytest.mark.parametrize("name", NAMES)
def test_transcription_target_is_identity(name):
    s = get_scheme(name)
    assert s.transcription_target(TEXT) == TEXT
    assert s.decode(s.transcription_target(TEXT)) == TEXT
    for u in s.encode(TEXT):
        assert "".join(u.labels) == u.source


@pytest.mark.parametrize("k", [2, 3, 4])
def test_chunking(k):
    s = QuadrantScheme(k=k)
    assert s.chunks("a") == ["a"]
    assert s.chunks("I, x.") == ["I,", " ", "x."]
    long = "internationalization"
    ch = s.chunks(long)
    assert "".join(ch) == long
    assert all(len(c) == k for c in ch[:-1]) and 1 <= len(ch[-1]) <= k
    # punctuation and digits are packed like letters; spaces are separate units
    assert s.chunks("(1984).") == ["(1984)."[i:i + k] for i in range(0, 7, k)]
    assert s.chunks("a  b") == ["a", " ", " ", "b"]
    units = s.encode("ab cd")
    assert [u.is_space for u in units].count(True) == 1
    assert all(isinstance(u, QuadCell) for u in units if not u.is_space)


def test_layout_reading_order():
    s = QuadrantScheme(k=4)
    (tl, tr, bl, br), _ = s.cell_geometry("abcd")
    assert tl[1] == tr[1] < bl[1] == br[1]          # rows
    assert tl[0] == bl[0] < tr[0] == br[0]          # columns
    (a, b, c), _ = QuadrantScheme(k=3).cell_geometry("abc")
    assert a[1] == b[1] < c[1] and a[0] < b[0] and c[2] >= a[2] + b[2]
    (a, b), _ = QuadrantScheme(k=2).cell_geometry("ab")
    assert a[0] == b[0] and a[1] < b[1]


@pytest.mark.parametrize("name", NAMES)
def test_render_page_fits(name):
    s = get_scheme(name)
    units = s.encode(TEXT)
    img, info = render_page(units, 400, 400)
    assert info["fit"] and info["em"] > 4 and img.size == (400, 400)
    assert np.asarray(img).min() < 100
    # supersampled rendering works too
    img2, info2 = render_page(units, 400, 400, LayoutParams(supersample=3))
    assert info2["fit"] and np.asarray(img2).min() < 100


@pytest.mark.parametrize("name", NAMES + ["quad3_dejavu", "quad4_mono"])
@pytest.mark.parametrize("em", [9.3, 24.0, 47.7])
def test_ink_stays_in_em_box(name, em):
    s = get_scheme(name)
    p = LayoutParams(margin=3, line_spacing=1.6)  # extra leading so a leak would be visible
    text = "Wg(|)_`j Qy{}[]@ qWMm, 4.7% 'tl' gyp! " + TEXT[:60]
    img = render_line(s.encode(text), em, p)
    a = np.asarray(img)
    base = p.first_baseline(em)
    top, bot = base - 0.85 * em, base + 0.25 * em
    ink_rows = np.where((a < 250).any(axis=1))[0]
    assert len(ink_rows) > 0
    assert ink_rows.min() >= np.floor(top) and ink_rows.max() <= np.ceil(bot) - 1 + 1e-9


def test_fixed_width_cells():
    s = QuadrantScheme(k=4, cell_width_em=0.5)
    _, adv4 = s.cell_geometry("abcd")
    _, adv4b = s.cell_geometry("mWiW")
    assert adv4 == adv4b
    assert s.transcription_target(TEXT) == TEXT


def test_prompts_and_legend():
    for name in NAMES:
        s = get_scheme(name)
        assert "top-left" in s.vlm_system_prompt() or name == "quad2"
        assert "top" in s.vlm_system_prompt()
        imgs = s.vlm_key_images()
        assert len(imgs) == 1
        w, h = imgs[0].size
        assert w <= 800 and h <= 600


def test_render_speed():
    s = get_scheme("quad3")
    units = s.encode(TEXT * 2)  # ~2.5k chars
    render_page(units, 400, 400)  # warm caches
    t = time.time()
    render_page(s.encode(TEXT * 2), 500, 500)
    assert time.time() - t < 2.0


def test_bad_names():
    with pytest.raises(KeyError):
        get_scheme("quad5")
    with pytest.raises(ValueError):
        QuadrantScheme(k=1)
