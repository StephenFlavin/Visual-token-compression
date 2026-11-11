from text_stats import top_bigrams


def test_counts_and_ties():
    t = "a b c a b d c a b"
    assert top_bigrams(t, 2) == [(("a", "b"), 3), (("c", "a"), 2)]


def test_empty():
    assert top_bigrams("", 3) == []
