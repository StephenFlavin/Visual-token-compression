"""Interval utilities for the booking calendar. Intervals are (start, end) integer pairs, both ends included."""
from __future__ import annotations


def normalise(iv: tuple[int, int]) -> tuple[int, int]:
    """Swap reversed intervals so start <= end."""
    a, b = iv
    return (a, b) if a <= b else (b, a)


def length(iv: tuple[int, int]) -> int:
    a, b = normalise(iv)
    return b - a + 1


def overlaps(x: tuple[int, int], y: tuple[int, int]) -> bool:
    (a, b), (c, d) = normalise(x), normalise(y)
    return a <= d and c <= b


def merge_intervals(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent intervals.

    Reversed intervals are normalised first. Two intervals merge when they overlap or touch
    (e.g. (1, 3) and (4, 6) become (1, 6), because the ends are integers). The result is sorted.
    """
    cur = [normalise(iv) for iv in ivs]
    changed = True
    while changed:
        changed = False
        out: list[tuple[int, int]] = []
        for iv in cur:
            for j, ov in enumerate(out):
                if iv[0] <= ov[1] + 1 and ov[0] <= iv[1] + 1:
                    out[j] = (min(iv[0], ov[0]), max(iv[1], ov[1]))
                    changed = True
                    break
            else:
                out.append(iv)
        cur = out
    return sorted(cur)


def total_covered(ivs: list[tuple[int, int]]) -> int:
    """Number of integers covered by at least one interval."""
    return sum(length(iv) for iv in merge_intervals(ivs))


def gaps(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Uncovered stretches between the merged intervals."""
    m = merge_intervals(ivs)
    return [(m[i][1] + 1, m[i + 1][0] - 1) for i in range(len(m) - 1)]
