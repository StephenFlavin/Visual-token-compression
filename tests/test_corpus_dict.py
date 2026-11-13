"""WP-A tests: corpus cleaning, passages, line stream, Visual-BPE dictionary segmentation.

Tests needing cached data (data/*.txt, passages, dict_*.json) skip if it is absent; build it with
``python3 scripts/build_data.py``. No test touches the network.
"""
from __future__ import annotations

import itertools
import random
import string
from functools import lru_cache

import pytest

from vtc import config, corpus
from vtc.dictionary import Dictionary, dict_path, text_width_em


def _need(*names: str) -> None:
    missing = [n for n in names if not (config.DATA / n).exists()]
    if missing:
        pytest.skip(f"cached data missing: {missing} (run scripts/build_data.py)")


# ----------------------------------------------------------------------------- cleaning

def test_clean_paragraph_detokenises():
    raw = (' In 2000 Boulter had a guest @-@ starring role on The Bill ; he portrayed " Scott Parry " '
           'in the episode , " In Safe Hands " . It cost $ 1 @,@ 000 ( 3 @.@ 5 % ) , the players \' '
           'union said it wasn \'t Boulter \'s fault . \n')
    out = corpus.clean_paragraph(raw)
    assert out == ('In 2000 Boulter had a guest-starring role on The Bill; he portrayed "Scott Parry" '
                   'in the episode, "In Safe Hands". It cost $1,000 (3.5%), the players\' '
                   "union said it wasn't Boulter's fault.")


def test_headings_and_blanks_dropped():
    assert corpus.clean_paragraph(" = Robert Boulter = \n") == ""
    assert corpus.clean_paragraph(" = = = 2000 – 2005 = = = \n") == ""
    assert corpus.clean_paragraph("   \n") == ""


def test_sentence_split_and_filter():
    par = 'Mr. Smith went to St. Louis in 1990. He met J. R. R. Tolkien there! "Really?" she asked.'
    assert corpus.split_sentences(par) == [
        "Mr. Smith went to St. Louis in 1990.", "He met J. R. R. Tolkien there!", '"Really?" she asked.']
    assert not corpus.keep_sentence("Too short.")
    assert not corpus.keep_sentence("Café society was popular in the city at that time.")
    assert corpus.keep_sentence("This sentence is plain ASCII and long enough.")


def test_passage_builder_deterministic_and_in_range():
    rng = random.Random(0)
    sents = [" ".join(rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(rng.randint(4, 40)))
             .capitalize() + "." for _ in range(3000)]
    a = corpus.make_natural_passages(sents, n=5, seed=7)
    b = corpus.make_natural_passages(sents, n=5, seed=7)
    assert a == b and len(a) == 5
    assert all(corpus.PASSAGE_MIN <= len(p["text"]) <= corpus.PASSAGE_MAX for p in a)
    assert corpus.make_natural_passages(sents, n=5, seed=8) != a


# ----------------------------------------------------------------------------- cached corpus

def test_passages_cached():
    _need("passages_natural.jsonl", "passages_scrambled.jsonl")
    nat = corpus.load_passages("natural")
    scr = corpus.load_passages("scrambled")
    assert len(nat) == corpus.N_NATURAL and len(scr) == corpus.N_SCRAMBLED
    printable = set(map(chr, range(32, 127)))
    for p in nat + scr:
        assert set(p["text"]) <= printable
        assert corpus.PASSAGE_MIN <= len(p["text"]) <= corpus.PASSAGE_MAX
        assert "  " not in p["text"] and p["text"] == p["text"].strip()
    assert len({p["id"] for p in nat + scr}) == len(nat) + len(scr)
    for n, s in zip(nat, scr):
        assert sorted(n["text"].split(" ")) == sorted(s["text"].split(" "))
        assert n["text"] != s["text"]
        assert corpus.scramble(n["text"], corpus.PASSAGE_SEED + int(n["id"][3:])) == s["text"]


def test_iter_lines_deterministic():
    _need("val.txt")
    a = list(itertools.islice(corpus.iter_lines("val", seed=3), 300))
    b = list(itertools.islice(corpus.iter_lines("val", seed=3), 300))
    c = list(itertools.islice(corpus.iter_lines("val", seed=4), 300))
    assert a == b and a != c
    assert all(30 <= len(x) <= 70 and x == x.strip() and x.isascii() for x in a)
    short = list(itertools.islice(corpus.iter_lines("val", min_chars=10, max_chars=20), 100))
    assert all(10 <= len(x) <= 20 for x in short)


# ----------------------------------------------------------------------------- dictionary

def _toy() -> Dictionary:
    return Dictionary([" the ", "the", "ing ", " of the ", "mm", "tion", "ation"])


def _brute_min(d: Dictionary, s: str) -> float:
    @lru_cache(None)
    def f(i: int) -> float:
        if i == len(s):
            return 0.0
        best = d.char_width(s[i]) + f(i + 1)
        for t in d.tokens:
            if s.startswith(t, i):
                best = min(best, d.symbol_em + f(i + len(t)))
        return best
    return f(0)


@pytest.mark.parametrize("seed", range(5))
def test_segment_roundtrip_and_optimal(seed):
    d = _toy()
    rng = random.Random(seed)
    pieces = [" the ", "the", "ing ", " of the ", "mm", "ation", "tion", "a", " ", "x", "e", "MM"]
    for _ in range(40):
        s = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 12)))
        seg = d.segment(s)
        assert "".join(seg) == s
        assert all(len(x) == 1 or x in d for x in seg)
        w = d.width(seg)
        assert w <= d.latin_width(s) + 1e-9
        assert abs(w - _brute_min(d, s)) < 1e-9


def test_token_only_used_when_it_saves():
    d = Dictionary(["ab", "the "])
    assert d.token_saving("ab") < 0          # 'ab' is narrower than 1 em -> never used
    assert d.segment("ab ab") == ["a", "b", " ", "a", "b"]
    assert d.segment("the cat") == ["the ", "c", "a", "t"]
    assert d.segment("") == []
    assert "".join(d.segment("naïve café")) == "naïve café"


def test_save_load_roundtrip(tmp_path):
    d = _toy()
    p = tmp_path / "d.json"
    d.save(p)
    e = Dictionary.from_file(p)
    assert e.tokens == d.tokens and e.symbol_em == d.symbol_em
    assert e.segment("of the nation") == d.segment("of the nation")


def test_condensed_narrower_than_dejavu():
    s = "The quick brown fox jumps over the lazy dog."
    assert text_width_em(s) < 0.8 * text_width_em(s, config.LATIN_FONT)


def test_cached_dictionaries_nested_and_useful():
    if not dict_path(3000).exists():
        pytest.skip("data/dict_3000.json missing (run scripts/build_data.py)")
    big = Dictionary.load(3000)
    assert len(big) == 3000 and len(set(big.tokens)) == 3000
    assert all(len(t) >= 2 and t.isascii() and big.token_saving(t) > 0 for t in big.tokens)
    prev = None
    for n in (100, 300, 1000):
        d = Dictionary.load(n)
        assert d.tokens == big.tokens[:n]
    _need("passages_natural.jsonl")
    text = corpus.load_passages("natural")[0]["text"]
    for n in (100, 300, 1000, 3000):
        d = Dictionary.load(n)
        seg = d.segment(text)
        assert "".join(seg) == text
        w = d.width(seg) / d.latin_width(text)
        assert w < 1.0 and (prev is None or w <= prev + 1e-9)
        prev = w


def test_segment_random_ascii_roundtrip():
    if not dict_path(1000).exists():
        pytest.skip("data/dict_1000.json missing")
    d = Dictionary.load(1000)
    rng = random.Random(1)
    alphabet = string.ascii_letters + string.digits + " .,;'\"()-"
    for _ in range(50):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 200)))
        seg = d.segment(s)
        assert "".join(seg) == s and d.width(seg) <= d.latin_width(s) + 1e-9
