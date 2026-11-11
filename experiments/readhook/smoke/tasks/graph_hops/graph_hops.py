"""Graph helpers for the dependency checker. Graphs are lists of directed edges (u, v) between string nodes."""
from __future__ import annotations


def nodes(edges: list[tuple[str, str]]) -> list[str]:
    """All nodes, sorted."""
    return sorted({u for u, _ in edges} | {v for _, v in edges})


def out_degree(edges: list[tuple[str, str]]) -> dict[str, int]:
    """Number of distinct successors of each node that has any."""
    succ: dict[str, set] = {}
    for u, v in edges:
        succ.setdefault(u, set()).add(v)
    return {u: len(s) for u, s in sorted(succ.items())}


def has_self_loop(edges: list[tuple[str, str]]) -> bool:
    return any(u == v for u, v in edges)


def hops_from(edges: list[tuple[str, str]], src: str, max_hops: int | None = None) -> dict[str, int]:
    """Fewest edges from `src` to every node reachable from it, `src` itself at 0.

    Nodes further than `max_hops` (if given) are left out. The result's keys are sorted.
    """
    dist = {src: 0}
    queue = [src]
    while queue:
        u = queue.pop(0)
        if max_hops is not None and dist[u] >= max_hops:
            continue
        succ = []
        for a, b in edges:
            if a == u and b not in succ:
                succ.append(b)
        for v in succ:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return {k: dist[k] for k in sorted(dist)}


def reverse(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(v, u) for u, v in edges]
