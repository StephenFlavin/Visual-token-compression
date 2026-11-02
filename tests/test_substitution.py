import numpy as np
import pytest

from vbpe import config
from vbpe.layout import GlyphUnit, LayoutParams, render_line, render_page
from vbpe.schemes import get_scheme
from vbpe.schemes.substitution import (CJKUnit, SubstitutionScheme, ink_extent, load_symbols,
                                       make_substitution_scheme, normalize_fullwidth)

STUB_TOKENS = [
    "the ", " of the ", "tion", "and ", "ing ", " of ", "ed ", "in ", "to ", "is ", "was ", "that ",
    "The ", "for ", "with ", "as ", "his ", "on ", "er ", "es ", "ation", "ment", "an ", "by ",
    "which ", "from ", "were ", "it ", "he ", "at ", "ly ", "this ", "had ", "are ", "been ",
    "ther", "ould", "ight", "ever", "ence", "ous ", ", ", ". ",
]


class StubDictionary:
    """Greedy longest-match segmenter (stand-in for vbpe.dictionary.Dictionary)."""

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self._maxlen = max(map(len, self.tokens))
        self._set = set(self.tokens)

    def segment(self, text):
        out, i = [], 0
        while i < len(text):
            for L in range(min(self._maxlen, len(text) - i), 1, -1):
                if text[i:i + L] in self._set:
                    out.append(text[i:i + L])
                    i += L
                    break
            else:
                out.append(text[i])
                i += 1
        return out


TEXTS = [
    "The history of the nation was shaped by the formation of trade routes, and the people "
    "who lived there were known for their ingenuity.",
    "In 1997, it was reported that 42% of the population (about 3.1 million) had moved; "
    "\"nobody expected this,\" he said -- and yet it happened!",
    "x",
    "   leading and trailing spaces   ",
    "Mixed: ATION, tion; THE the The tHe [brackets] {braces} <angle> a/b\\c @#$^&*_+=~`|",
]


def scheme(**kw):
    kw.setdefault("dictionary", StubDictionary(STUB_TOKENS))
    return SubstitutionScheme(n_tokens=len(STUB_TOKENS), **kw)


@pytest.mark.parametrize("text", TEXTS)
def test_roundtrip(text):
    s = scheme()
    assert s.decode(s.transcription_target(text)) == text
    assert "".join(u.source for u in s.encode(text)) == text


def test_compresses_and_uses_symbols():
    s = scheme()
    t = TEXTS[0]
    target = s.transcription_target(t)
    assert len(target) < 0.75 * len(t)
    assert any(ch in s.sym2tok for ch in target)


def test_advances():
    s = scheme()
    units = s.encode(TEXTS[0])
    cjk = [u for u in units if isinstance(u, CJKUnit)]
    lat = [u for u in units if not isinstance(u, CJKUnit)]
    assert cjk and lat
    for em in (7.3, 10.0, 23.17):
        assert all(u.advance(em) == em for u in cjk)
    assert all(isinstance(u, GlyphUnit) and u.font_path == str(config.CONDENSED_FONT) for u in lat)
    s2 = scheme(cjk_scale=0.9)
    assert all(u.advance(10.0) == pytest.approx(9.0) for u in s2.encode("the ") if isinstance(u, CJKUnit))


def test_ink_inside_em_box():
    top, bot = ink_extent(str(config.CJK_FONT), tuple(load_symbols()))
    assert top < -0.84 and bot > 0.05          # Noto metrics as measured (sanity)
    s = scheme()
    assert top + s.cjk_dy >= -0.85 and bot + s.cjk_dy <= 0.25
    s2 = scheme(cjk_scale=1.1)
    assert 1.1 * top + s2.cjk_dy >= -0.85 and 1.1 * bot + s2.cjk_dy <= 0.25
    with pytest.raises(ValueError):
        scheme(cjk_scale=1.3)._setup()


def test_render_page_fits_and_inks():
    s = scheme()
    units = s.encode(" ".join(TEXTS[:2]) * 6)
    img, info = render_page(units, 300, 300)
    assert info["fit"] and info["em"] > 5
    a = np.asarray(img)
    assert a.min() < 128
    # ink of the first line must not reach above the em box (top margin rows stay blank)
    p = LayoutParams()
    top_px = int(p.first_baseline(info["em"]) - 0.86 * info["em"])
    assert top_px >= 2 and a[: top_px - 1].min() > 200


def test_render_line():
    s = scheme()
    img = render_line(s.encode("the nation of the people"), 16.0, LayoutParams(margin=2))
    assert img.size[1] == round(1.2 * 16) + 4


def test_condensed_vs_subst_width():
    s = scheme()
    lat = get_scheme("latin_condensed")
    t = TEXTS[0]
    w_s = sum(u.advance(10) for u in s.encode(t))
    w_l = sum(u.advance(10) for u in lat.encode(t))
    assert w_s < w_l


def test_legend_prompt_contains_all_tokens():
    s = scheme(legend=True)
    sp = s.vlm_system_prompt()
    for tok, sym in zip(s.tokens, s.symbols):
        assert f"{sym}\t{tok.replace(' ', chr(0x2423))}" in sp
    assert s.name.endswith("_legend")
    assert "Chinese" in scheme().vlm_system_prompt()
    assert "LEGEND" not in scheme().vlm_system_prompt()


def test_decode_normalisation():
    s = scheme()
    sym_the = s.tok2sym["the "]
    assert s.decode("ＡＢＣ　１２３！") == "ABC 123!"
    assert s.decode("a，b。c：d；“e”‘f’（g）！？") == "a,b.c:d;\"e\"'f'(g)!?"
    assert s.decode(sym_the + "cat") == "the cat"
    assert s.decode("爨cat") == "?cat"        # unknown ideograph -> error marker
    # legend mode: plain English passes through, stray symbols are still mapped
    sl = scheme(legend=True)
    assert sl.decode("the cat " + sym_the + "dog") == "the cat the dog"


def test_symbol_pool():
    syms = load_symbols()
    assert len(syms) >= 3500
    assert len(set(syms)) == len(syms)
    assert all(len(c) == 1 and 0x4E00 <= ord(c) <= 0x9FFF for c in syms)
    for bad in "一二三十口日曰人入八丿丨乙":
        assert bad not in syms
    assert syms[0] == "的"


def test_factory_names():
    d = StubDictionary(STUB_TOKENS)
    s = make_substitution_scheme("subst30", dictionary=d)
    assert s.n_tokens == 30 and not s.legend and s.name == "subst30"
    s = get_scheme("subst30_legend", dictionary=d)
    assert s.legend and s.name == "subst30_legend" and len(s.legend_text().splitlines()) == 30
    with pytest.raises(KeyError):
        make_substitution_scheme("substx")


def test_real_dictionary_if_available():
    try:
        s = get_scheme("subst300")
        s._setup()
    except (ImportError, FileNotFoundError, OSError) as e:
        pytest.skip(f"dictionary not available: {e}")
    from vbpe.corpus import load_passages
    for t in TEXTS + [p["text"] for p in load_passages("natural")[:3] + load_passages("scrambled")[:2]]:
        assert s.decode(s.transcription_target(t)) == t
    assert abs(s.cjk_dy - 0.0211) < 0.003
