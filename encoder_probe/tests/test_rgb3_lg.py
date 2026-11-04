"""Fast checks for the round-3 moonshot encodings rgb3_lg_* (RGB overlay of learned-glyph streams; no downloads)."""
import numpy as np
import pytest

from encoder_probe.ep.geom import get_encoding_s, make_pages_s
from encoder_probe.ep.pages import LP, get_encoding
from encoder_probe.ep.probe import edit_distance
from encoder_probe.ep.rgb_overlay import TAGS, RGB3BankEncoding, ctc_ok
from vbpe.layout import _draw_lines

NAMES = ("rgb3_lg_pb_h5", "rgb3_lg_free_h4")
SIZES = (448, 512)                           # identity28/qwen25vl and identity32/deepencoder page sides
G = 16                                       # tokens per side at both sizes


@pytest.fixture(scope="module")
def pages():
    return {(n, S): (get_encoding_s(n, S), make_pages_s(get_encoding_s(n, S), "test", 2))
            for n in NAMES for S in SIZES}


def _channel_units(enc, lab, c):
    """Units of channel c rebuilt from its tagged labels alone (bank symbol ids)."""
    from learned_glyphs.lg.scheme import PUA_BASE, BankGlyphUnit
    return [BankGlyphUnit(ord(x[1]) - PUA_BASE, enc.scheme) for x in lab if x[0] == TAGS[c]]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("S", SIZES)
def test_shapes_classes_and_roundtrip(pages, name, S):
    enc, pg = pages[(name, S)]
    assert isinstance(enc, RGB3BankEncoding) and enc.name == name and enc.S == S
    single = get_encoding_s(name[len("rgb3_"):], S)
    assert enc.pitch == single.pitch and enc.n_lines == S // enc.pitch
    assert enc.classes == [t + c for t in TAGS for c in single.classes]           # 3 x 395 tagged classes
    cls = set(enc.classes)
    for page in pg:
        assert page.img.shape == (S, S, 3) and page.img.dtype == np.uint8
        assert len(page.labels) == len(page.refs) == enc.n_lines
        for lab, r in zip(page.labels, page.refs):
            assert set(lab) <= cls and r == r.strip() and "  " not in r
            assert enc.clean_hyp(enc.decode("".join(lab))) == r                    # perfect read = 0 CER


def test_refs_are_the_source_text_in_order(pages):
    enc, pg = pages[("rgb3_lg_pb_h5", 448)]
    joined = " ".join(r for page in pg for r in page.refs)
    assert joined.startswith("Robert Boulter is an English film")
    single = make_pages_s(get_encoding_s("lg_pb_h5", 448), "test", 1)[0]
    # the overlay page holds the same text stream, three wrapped lines per page line
    assert joined.startswith(" ".join(single.refs[:20]).rsplit(" ", 3)[0])


@pytest.mark.parametrize("name", NAMES)
def test_channels_are_independent_bank_streams(pages, name):
    enc, pg = pages[(name, 448)]
    page = pg[0]
    for c in range(3):
        lines = [_channel_units(enc, lab, c) for lab in page.labels]
        alone = np.asarray(_draw_lines(lines, enc.em, (448, 448), LP, LP.supersample), np.uint8)
        assert (alone == page.img[:, :, c]).all()            # channel c = the bank page of its own labels only
        dec = [enc.scheme.decode("".join(x[1] for x in lab if x[0] == TAGS[c])) for lab in page.labels]
        assert sum(map(len, dec)) > 0.3 * page.n_chars       # each channel carries ~1/3 of the text
    a, b, c = (page.img[:, :, k] for k in range(3))
    assert not (a == b).all() and not (b == c).all()


def test_tagged_decode_drop_costs_one_glyph_of_one_stream(pages):
    enc, pg = pages[("rgb3_lg_free_h4", 448)]
    lab, ref = pg[0].labels[0], pg[0].refs[0]
    k = 31
    hyp = enc.clean_hyp(enc.decode("".join(lab[:k] + lab[k + 1:])))
    lost = enc.scheme.decode(lab[k][1])
    assert edit_distance(ref, hyp) <= len(lost)


@pytest.mark.parametrize("name", NAMES)
def test_density_is_three_times_the_single_bank(pages, name):
    enc, pg = pages[(name, 448)]
    single = make_pages_s(get_encoding_s(name[len("rgb3_"):], 448), "test", 2)
    cpt = sum(p.n_chars for p in pg) / (len(pg) * G * G)
    cpt1 = sum(p.n_chars for p in single) / (len(single) * G * G)
    assert abs(cpt / cpt1 - 3) < 0.1, (cpt, cpt1)
    assert cpt > {"rgb3_lg_pb_h5": 310, "rgb3_lg_free_h4": 390}[name]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("S", (224, 448, 512))
def test_ctc_budget_holds_worst_case(name, S):
    enc = get_encoding(name) if S == 224 else get_encoding_s(name, S)
    assert enc.S == S and enc.min_adv_ss == pytest.approx(8.0)            # narrowest glyph: 2 px
    # full width would allow S/2 glyphs per stream: 2 * 3S/2 + 1 = 3S+1 > 3S frames, one frame short
    assert not ctc_ok(3 * (S // 2), S)
    assert enc.line_px == S - 1 and enc.max_glyphs == (S - 1) // 2 and ctc_ok(3 * enc.max_glyphs, S)
    worst = "~" * (3 * enc.n_lines * enc.max_glyphs + 1000)              # every glyph a 2 px single char
    page, used = enc.render(worst, 0)
    assert used == 3 * enc.n_lines * enc.max_glyphs
    for lab in page.labels:
        assert len(lab) == 3 * enc.max_glyphs and ctc_ok(len(lab), S)
        assert [x[0] for x in lab[:6]] == list("RGBRGB")                  # equal advances: rgb3's R[k], G[k], B[k]


@pytest.mark.parametrize("name", NAMES)
def test_real_lines_fit_ctc_and_interleave_by_x(pages, name):
    enc, pg = pages[(name, 448)]
    sem = enc.em * LP.supersample
    for page in pg:
        for lab in page.labels:
            assert ctc_ok(len(lab), 448) and len(lab) <= 3 * enc.max_glyphs
        lab = page.labels[3]
        xs = {c: [] for c in range(3)}
        for c in range(3):                     # centre x of each channel's glyphs, in stream order
            x = 0.0
            for u in _channel_units(enc, lab, c):
                a = u.advance(sem)
                xs[c].append(x + a / 2)
                x += a
        merged = sorted((x, c) for c in range(3) for x in xs[c])
        assert [TAGS[c] for _, c in merged] == [t[0] for t in lab]


def test_registry_names():
    for n in NAMES:
        assert get_encoding(n).S == 224 and get_encoding_s(n, 448).family.startswith("rgb3_lg_")
    with pytest.raises(KeyError):
        get_encoding_s("rgb3_lg_nope", 448)
