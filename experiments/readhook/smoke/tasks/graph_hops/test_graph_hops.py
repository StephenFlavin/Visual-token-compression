from graph_hops import hops_from


def test_basic():
    e = [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")]
    assert hops_from(e, "a") == {"a": 0, "b": 1, "c": 1, "d": 2}


def test_max_hops():
    e = [("a", "b"), ("b", "c")]
    assert hops_from(e, "a", max_hops=1) == {"a": 0, "b": 1}
