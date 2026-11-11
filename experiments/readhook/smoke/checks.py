"""Hidden checks for the smoke tasks (never copied into the agent's directory).

equivalence(): every public function of the refactored module against the original on random inputs
(results, or exception types, must match). bench(): the target function on a large input, original vs refactored.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import random
import time
from pathlib import Path

TASKS = {"orders": "dedupe_orders", "text_stats": "top_bigrams", "graph_hops": "hops_from",
         "intervals": "merge_intervals", "moving": "moving_average", "scheduler": "assign_slots"}
HERE = Path(__file__).resolve().parent


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_{abs(hash(str(path)))}", path)
    m = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = m                    # dataclasses look the module up while it loads
    spec.loader.exec_module(m)
    return m


def cases(task: str, m, rng: random.Random):
    """Yield (function name, args) for random small inputs; `m` supplies classes such as Order."""
    if task == "orders":
        skus = ["ab-1", "AB1", " ab1 ", "x-y", "XY", "q", "Q-", "zz"]
        days = ["2026-01-30", "2026-01-31", "2026-02-01", "2026-02-02", "2026-03-15"]
        def order(i):
            return m.Order(i, rng.choice(skus), rng.randint(0, 6), rng.choice([0.5, 1.0, 2.25, 9.99]),
                           rng.choice(list(m.VAT_RATES)), rng.choice(days),
                           rng.sample(["Red", "red", "bulk", "BULK", "gift"], rng.randint(0, 3)))
        for _ in range(400):
            os_ = [order(i) for i in range(rng.randint(0, 14))]
            yield "dedupe_orders", (os_,)
            yield "totals_by_rate", (os_,)
            yield "tag_index", (os_,)
            yield "monthly_summary", (os_,)
            yield "orders_between", (os_, rng.choice(days), rng.choice(days))
            if os_:
                yield "line_total", (os_[0],)
            yield "normalise_sku", (rng.choice(skus),)
    elif task == "text_stats":
        vocab = ["a", "b", "the", "cat's", "dog", "and", "run", "an", "of", "x"]
        for _ in range(500):
            t = " ".join(rng.choice(vocab) + rng.choice(["", "", ".", "!", ",", "?"]) for _ in range(rng.randint(0, 30)))
            t = rng.choice([t, t.upper(), t + " 42 x"])
            k = rng.randint(0, 6)
            for f, a in [("tokenize", (t,)), ("word_count", (t, rng.random() < .5)), ("longest_words", (t, k)),
                         ("sentence_lengths", (t,)), ("top_bigrams", (t, k)), ("type_token_ratio", (t,))]:
                yield f, a
    elif task == "graph_hops":
        ns = list("abcdefgh")
        for _ in range(500):
            e = [(rng.choice(ns), rng.choice(ns)) for _ in range(rng.randint(0, 16))]
            yield "hops_from", (e, rng.choice(ns), rng.choice([None, None, 0, 1, 2, 3]))
            yield "nodes", (e,)
            yield "out_degree", (e,)
            yield "has_self_loop", (e,)
            yield "reverse", (e,)
    elif task == "intervals":
        for _ in range(600):
            ivs = [(rng.randint(-5, 25), rng.randint(-5, 25)) for _ in range(rng.randint(0, 10))]
            yield "merge_intervals", (ivs,)
            yield "total_covered", (ivs,)
            yield "gaps", (ivs,)
            if len(ivs) >= 2:
                yield "overlaps", (ivs[0], ivs[1])
                yield "normalise", (ivs[0],)
                yield "length", (ivs[1],)
    elif task == "moving":
        for _ in range(600):
            xs = [rng.randint(-20, 60) for _ in range(rng.randint(0, 15))]
            w = rng.randint(0, 7)
            yield "moving_average", (xs, w)
            if w >= 1:
                yield "moving_max", (xs, w)
                yield "spikes", (xs, w, rng.choice([1.5, 2.0, 3.0]))
            yield "clamp", (xs, 0, 30)
            yield "diffs", (xs,)
    elif task == "scheduler":
        for _ in range(600):
            ids = rng.sample(range(1, 40), rng.randint(0, 12))
            jobs = [(j, rng.randint(0, 3), rng.randint(1, 9)) for j in ids]
            caps = [rng.randint(0, 15) for _ in range(rng.randint(0, 4))]
            yield "assign_slots", (jobs, caps)
            yield "total_duration", (jobs,)
            yield "by_priority", (jobs,)
            yield "utilisation", (caps, {j: (rng.randrange(len(caps)) if caps and rng.random() < .7 else None)
                                         for j, _, _ in jobs}, jobs)


def call(fn, args):
    try:
        return ("ok", fn(*copy.deepcopy(args)))
    except Exception as e:  # noqa: BLE001
        return ("raise", type(e).__name__)


def equivalence(task: str, new_path: Path, seed: int = 0) -> dict:
    orig = load(HERE / "tasks" / task / f"{task}.py", task + "_orig")
    try:
        new = load(new_path, task + "_new")
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "import_error": f"{type(e).__name__}: {e}", "fails": {}}
    fails, n = {}, 0
    for f, args in cases(task, orig, random.Random(seed)):
        nargs = args
        if task == "orders":     # the new module has its own Order class: rebuild the objects
            def conv(x):
                if isinstance(x, list):
                    return [conv(y) for y in x]
                if isinstance(x, orig.Order):
                    return new.Order(**{k: copy.deepcopy(v) for k, v in x.__dict__.items()})
                return x
            nargs = tuple(conv(a) for a in args)
        a, b = call(getattr(orig, f), args), call(getattr(new, f, None) or (lambda *x: None), nargs)
        if task == "orders":
            norm = lambda r: (r[0], [o.__dict__ for o in r[1]]) if r[0] == "ok" and isinstance(r[1], list) and r[1] and hasattr(r[1][0], "__dict__") else r  # noqa: E731
            a, b = norm(a), norm(b)
        n += 1
        if a != b:
            fails.setdefault(f, 0)
            fails[f] += 1
    public = {k for k, v in vars(orig).items() if inspect.isfunction(v) and not k.startswith("_") and v.__module__ == orig.__name__}
    missing = sorted(k for k in public if not inspect.isfunction(getattr(new, k, None)))
    return {"ok": not fails and not missing, "cases": n, "fails": fails, "missing": missing}


def big_input(task: str, m, rng: random.Random):
    if task == "orders":
        return ([m.Order(i, f"s{rng.randint(0, 3000)}", rng.randint(0, 9), 1.0, placed=f"2026-01-{rng.randint(1, 28):02d}")
                 for i in range(6000)],)
    if task == "text_stats":
        vocab = ["".join(rng.choice("abcdefghij") for _ in range(3)) for _ in range(120)]
        return (" ".join(rng.choice(vocab) for _ in range(6000)), 10)
    if task == "graph_hops":
        return ([(f"n{rng.randint(0, 3000)}", f"n{rng.randint(0, 3000)}") for _ in range(15000)], "n0", None)
    if task == "intervals":
        return ([(a, a + rng.randint(0, 3)) for a in (rng.randint(0, 60000) for _ in range(4000))],)
    if task == "moving":
        return ([rng.randint(0, 1000) for _ in range(60000)], 3000)
    if task == "scheduler":
        return ([(j, rng.randint(0, 50), rng.randint(1, 20)) for j in range(6000)], [rng.randint(50, 900) for _ in range(40)])


def bench(task: str, new_path: Path) -> dict:
    orig = load(HERE / "tasks" / task / f"{task}.py", task + "_orig")
    new = load(new_path, task + "_new")
    args = big_input(task, orig, random.Random(1))
    nargs = args
    if task == "orders":
        nargs = ([new.Order(**o.__dict__) for o in args[0]],)
    fn = TASKS[task]
    t0 = time.perf_counter(); r0 = getattr(orig, fn)(*copy.deepcopy(args)); t1 = time.perf_counter()
    r1 = getattr(new, fn)(*copy.deepcopy(nargs)); t2 = time.perf_counter()
    if task == "orders":
        r0, r1 = [o.__dict__ for o in r0], [o.__dict__ for o in r1]
    return {"orig_s": round(t1 - t0, 4), "new_s": round(t2 - t1, 4), "speedup": round((t1 - t0) / max(t2 - t1, 1e-6), 1),
            "same_result": r0 == r1}


if __name__ == "__main__":     # self-check: the originals are equivalent to themselves; time the originals
    for t in TASKS:
        p = HERE / "tasks" / t / f"{t}.py"
        print(t, equivalence(t, p)["ok"], bench(t, p))
