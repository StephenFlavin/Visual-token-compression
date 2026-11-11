"""Rolling statistics for integer sensor readings."""
from __future__ import annotations


def clamp(xs: list[int], lo: int, hi: int) -> list[int]:
    return [min(hi, max(lo, x)) for x in xs]


def diffs(xs: list[int]) -> list[int]:
    """Successive differences."""
    return [b - a for a, b in zip(xs, xs[1:])]


def moving_average(xs: list[int], w: int) -> list[float | None]:
    """Trailing mean over windows of w readings, rounded to 6 decimals.

    The first w - 1 positions have no full window and are None. w must be >= 1.
    """
    if w < 1:
        raise ValueError("window must be >= 1")
    out: list[float | None] = []
    for i in range(len(xs)):
        if i < w - 1:
            out.append(None)
        else:
            out.append(round(sum(xs[i - w + 1:i + 1]) / w, 6))
    return out


def moving_max(xs: list[int], w: int) -> list[int | None]:
    """Trailing maximum over windows of w readings (None until the first full window)."""
    return [None if i < w - 1 else max(xs[i - w + 1:i + 1]) for i in range(len(xs))]


def spikes(xs: list[int], w: int, factor: float = 2.0) -> list[int]:
    """Positions where a reading exceeds factor times the trailing mean of the window ending before it."""
    avg = moving_average(xs, w)
    return [i for i in range(w, len(xs)) if avg[i - 1] is not None and xs[i] > factor * avg[i - 1]]
