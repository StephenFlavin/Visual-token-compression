from scheduler import assign_slots


def test_greedy():
    jobs = [(1, 1, 5), (2, 3, 4), (3, 3, 6)]
    assert assign_slots(jobs, [10, 6]) == {1: 1, 2: 0, 3: 0}
