"""Greedy assignment of jobs to machine slots for the nightly batch."""
from __future__ import annotations


def total_duration(jobs: list[tuple[int, int, int]]) -> int:
    """Jobs are (job_id, priority, duration)."""
    return sum(d for _, _, d in jobs)


def by_priority(jobs: list[tuple[int, int, int]]) -> list[int]:
    """Job ids, highest priority first, ties by ascending id."""
    return [j for j, _, _ in sorted(jobs, key=lambda t: (-t[1], t[0]))]


def assign_slots(jobs: list[tuple[int, int, int]], capacities: list[int]) -> dict[int, int | None]:
    """Assign each job to a slot.

    Jobs are taken highest priority first, ties by ascending job id. Each job goes to the slot with the most
    remaining capacity, ties to the lowest slot index, provided its duration fits there; otherwise it is
    unassigned (None). Returns {job_id: slot index or None} with keys in ascending order.
    """
    remaining = list(capacities)
    todo = list(jobs)
    result: dict[int, int | None] = {}
    while todo:
        todo.sort(key=lambda t: (-t[1], t[0]))
        job_id, _, dur = todo.pop(0)
        best = None
        for i in range(len(remaining)):
            if best is None or remaining[i] > remaining[best]:
                best = i
        if best is not None and remaining[best] >= dur:
            remaining[best] -= dur
            result[job_id] = best
        else:
            result[job_id] = None
    return {k: result[k] for k in sorted(result)}


def utilisation(capacities: list[int], assignment: dict[int, int | None], jobs: list[tuple[int, int, int]]) -> list[float]:
    """Fraction of each slot's capacity used by an assignment, rounded to 4 decimals."""
    used = [0] * len(capacities)
    dur = {j: d for j, _, d in jobs}
    for j, s in assignment.items():
        if s is not None:
            used[s] += dur[j]
    return [round(u / c, 4) if c else 0.0 for u, c in zip(used, capacities)]
