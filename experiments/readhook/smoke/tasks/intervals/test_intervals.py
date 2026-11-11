from intervals import merge_intervals


def test_touching_and_reversed():
    assert merge_intervals([(4, 6), (3, 1), (10, 8)]) == [(1, 6), (8, 10)]


def test_nested():
    assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]
