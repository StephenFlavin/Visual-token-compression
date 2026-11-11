import pytest
from moving import moving_average


def test_window():
    assert moving_average([1, 2, 3, 4], 2) == [None, 1.5, 2.5, 3.5]


def test_bad_window():
    with pytest.raises(ValueError):
        moving_average([1], 0)
