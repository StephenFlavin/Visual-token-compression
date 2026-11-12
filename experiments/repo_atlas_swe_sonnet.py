"""Sonnet with the repo atlas on the hard coding tasks (REPO_ATLAS_PLAN.md Q3): plain vs atlas, two fresh runs per
task and arm, Claude Code on Sonnet 5.5 at high effort.

A thin runner over Q1's (``experiments/repo_atlas_swe_rerun.py``, imported, not edited): same 16 tasks, prompt,
tools, flags, atlas pages, environments, grading, memory-off, index blocking and leak scan. What this file adds: the
model as a parameter, its own work dir and results dir, its own run-directory hash and order seed, its own cost
bound, a legibility gate (20 lookups on two atlases, Sonnet against Opus) and an analysis that shows Q1's Opus rows
and the plan-quality test's Sonnet rows beside its own (never pooled).

    python3 experiments/repo_atlas_swe_sonnet.py gate-build                    # the 20 lookup questions (free)
    python3 experiments/repo_atlas_swe_sonnet.py gate --models claude-sonnet-5-5 claude-opus-5-5 [--cap 1]
    python3 experiments/repo_atlas_swe_sonnet.py validate [--workers 2]
    python3 experiments/repo_atlas_swe_sonnet.py order
    python3 experiments/repo_atlas_swe_sonnet.py run --cap 20 [--max-runs N] [--workers 2]
    python3 experiments/repo_atlas_swe_sonnet.py analyse                       # standalone, on whatever rows exist

Rows: results/repo_atlas_swe_sonnet/{gate_questions.json,gate.jsonl,validate,runs,spend}.jsonl. Work dir in the
scratchpad (``swe_sonnet``): links to the first test's atlas builds and dataset and to today's git stores (read only).
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

import repo_atlas_swe as sw  # noqa: E402
import repo_atlas_swe_rerun as rr  # noqa: E402

OUT = ROOT / "results" / "repo_atlas_swe_sonnet"
Q1 = ROOT / "results" / "repo_atlas_swe_rerun"               # read only
PQ = ROOT / "results" / "plan_quality" / "work.jsonl"        # read only
SCRATCH = rr.SCRATCH
DEFAULT_WORK = Path(os.environ.get("RA_SWE3_WORK", str(SCRATCH / "swe_sonnet")))
MODEL = "claude-sonnet-5-5"
EFFORT = "high"
ARMS = rr.ARMS
REPS = rr.REPS
SEED = 20261113
BOUND_FLOOR = {"plain": 1.0, "atlas": 2.0}
GATE_TASKS = ("scientific-python__docstub-123", "pytest-dev__pyfakefs-1286")
GATE_PER_TASK = 10
GATE_PASS = 0.80
GATE_SYSTEM = ("You answer lookup questions about a Python repository from page images of its source. Use only the "
               "images; do not guess from memory.")
ACK = "\n\nReply only OK."


# --------------------------------------------------------------------------- pure functions (tested)
def run_dir_tag(iid: str, arm: str, rep: int) -> str:
    """Neutral directory name (its own salt: never equal to Q1's tag for the same job)."""
    import hashlib
    return hashlib.sha1(f"sonnet|{iid}|{arm}|{rep}".encode()).hexdigest()[:10]


def run_order(ids: list[str], seed: int = SEED) -> list[tuple[str, str, int]]:
    return rr.run_order(ids, seed)


def bound(arm: str, own_costs: list[float]) -> float:
    """Reserved before a run: the larger of the arm's floor and 1.5x the dearest run of that arm so far here."""
    return max([BOUND_FLOOR[arm]] + [1.5 * c for c in own_costs])


def tagged_lines(flow: str) -> list[int]:
    return [int(x) for x in re.findall(r"§(\d+) ", flow)]


def gate_candidates(path: str, src: str, flow: str) -> list[dict]:
    """Lookup questions with answers from the source (ast): the definition right after a tagged line, a module
    constant's value, a function's parameter names."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    defs = sorted((n.lineno, n.name) for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    tags = set(tagged_lines(flow))
    for n_line in sorted(tags):
        nxt = [(ln, nm) for ln, nm in defs if ln >= n_line]
        if nxt and nxt[0][0] - n_line < 10:
            out.append({"kind": "after_line", "path": path, "line": n_line, "answer": nxt[0][1], "anchor": f"§{n_line} ",
                        "q": f"In {path}: what is the name of the first function or class whose `def`/`class` keyword "
                             f"comes at or after the line tagged §{n_line}? Give the name only."})
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", node.targets[0].id) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (str, int, float)) and not isinstance(node.value.value, bool)
                and 0 < len(repr(node.value.value)) <= 40):
            out.append({"kind": "constant", "path": path, "name": node.targets[0].id, "anchor": node.targets[0].id,
                        "answer": ast.get_source_segment(src, node.value) or repr(node.value.value),
                        "value": node.value.value,
                        "q": f"In {path}: what value is the module-level constant {node.targets[0].id} assigned? "
                             "Give the value only."})
    names = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and names.count(n.name) == 1:
            a = n.args
            ps = [x.arg for x in a.posonlyargs + a.args] + ([a.vararg.arg] if a.vararg else []) + \
                 [x.arg for x in a.kwonlyargs] + ([a.kwarg.arg] if a.kwarg else [])
            if 2 <= len(ps) <= 6:
                out.append({"kind": "params", "path": path, "name": n.name, "answer": ", ".join(ps),
                            "anchor": f"def {n.name}(",
                            "q": f"In {path}: list the parameter names of `{n.name}` in order, comma-separated (names "
                                 "only: no defaults, annotations or stars; include self/cls if present)."})
    return out


def pick_questions(cands: list[dict], n_pages: int, k: int, seed: int) -> list[dict]:
    """k questions spread over early / middle / late thirds of the atlas and over the three kinds, one per file
    where possible, deterministic."""
    rng = random.Random(seed)
    third = lambda p: min(2, 3 * (p - 1) // max(1, n_pages))  # noqa: E731
    pools = {}
    for c in cands:
        if c.get("page"):
            pools.setdefault((third(c["page"]), c["kind"]), []).append(c)
    for v in pools.values():
        rng.shuffle(v)
    keys = sorted(pools)
    picked, used = [], set()
    while len(picked) < k and any(pools.values()):
        for key in keys:
            if len(picked) >= k:
                break
            pool = pools[key]
            j = next((i for i, c in enumerate(pool) if (c["path"], c["kind"]) not in used), 0 if pool else None)
            if j is None:
                continue
            c = pool.pop(j)
            used.add((c["path"], c["kind"]))
            picked.append({**c, "third": ["early", "middle", "late"][third(c["page"])]})
    return picked


def page_of_anchor(flow: str, chunks: list[str], path: str, anchor: str) -> int | None:
    """1-based page holding the first ``anchor`` after the file's header (pages = chunks of the flow, in order)."""
    h = flow.find(f"▶{path}◀")
    if h < 0:
        return None
    nxt = flow.find("▶", h + 1)
    off = flow.find(anchor, h)
    if off < 0 or (nxt >= 0 and off > nxt):
        return None
    pos = 0
    for i, c in enumerate(chunks):
        start = flow.find(c, pos)
        if start <= off < start + len(c):
            return i + 1
        pos = start + len(c)
    return None


def norm(s: str) -> str:
    s = s.strip().strip("`").strip()
    s = re.sub(r"^['\"]|['\"]$", "", s)
    return re.sub(r"\s+", "", s.lower()).rstrip("()")


def score(q: dict, answer: str) -> bool:
    if q["kind"] == "params":
        got = [norm(x).lstrip("*") for x in re.split(r"[,\s]+", answer.strip().strip("`()")) if x.strip()]
        return got == [x.lstrip("*").lower() for x in q["answer"].split(", ")]
    if q["kind"] == "constant" and "value" in q:
        try:
            if ast.literal_eval(answer.strip().strip("`")) == q["value"]:
                return True
        except (ValueError, SyntaxError):
            pass
    return norm(answer) == norm(q["answer"])


def parse_answers(text: str, n: int) -> dict[int, str]:
    out = {}
    for m in re.finditer(r"^\s*\**Q(\d+)\**\s*[:.)-]\s*(.*?)\s*$", text, re.M):
        i = int(m.group(1))
        if 1 <= i <= n and i not in out:
            out[i] = m.group(2)
    return out


# --------------------------------------------------------------------------- io
_alock = threading.Lock()
_rlock = threading.Lock()
_reserved = [0.0]


def jl(path: Path) -> list[dict]:
    return sw.jl(path)


def append(name: str, row: dict) -> None:
    with _alock:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / name, "a") as f:
            f.write(json.dumps(row) + "\n")


def spent() -> float:
    return sum(r.get("cost_usd") or 0 for r in jl(OUT / "spend.jsonl"))


def setup_work(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for name, src in (("ctx", SCRATCH / "ra_swe" / "ctx"), ("ds", SCRATCH / "ra_swe" / "ds"),
                      ("git", SCRATCH / "pq" / "git")):
        d = work / name
        if not d.exists() and src.exists():
            d.symlink_to(src)


def patch_rr() -> None:
    """Point Q1's module at this test: model, results dir, run-dir hash. Only this process's copy changes."""
    sw.MODEL = MODEL
    sw.EFFORT = EFFORT
    rr.OUT = OUT
    rr.run_dir_tag = run_dir_tag


# --------------------------------------------------------------------------- gate
def gate_build(work: Path) -> list[dict]:
    import repo_atlas as RA
    tm = sw.task_map(work)
    qs = []
    for gi, iid in enumerate(GATE_TASKS):
        t = tm[iid]
        meta = sw.contexts(work, t)
        co = sw.fresh_repo(work, t, work / "gate" / "co" / iid / t["repo"].split("/")[1])
        files, _ = RA.collect(co, False, 10)
        from vbhook import render
        flow = RA.atlas_flow(files)
        chunks = render.chunk(flow, render.page_chars(88.2))
        assert len(chunks) == meta["pages"], (iid, len(chunks), meta["pages"])
        cands = []
        for f in files:
            if f["path"].endswith(".py"):
                for c in gate_candidates(f["path"], (co / f["path"]).read_text(encoding="utf-8"), f["flow"]):
                    c["page"] = page_of_anchor(flow, chunks, c["path"], c["anchor"])
                    cands.append(c)
        for q in pick_questions(cands, meta["pages"], GATE_PER_TASK, SEED + gi):
            qs.append({"task": iid, **q})
        import shutil
        shutil.rmtree(work / "gate" / "co" / iid, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_questions.json").write_text(json.dumps(qs, indent=1))
    return qs


def gate_call(model: str, pngs: list[bytes], context: str, questions: str, timeout: float = 1200.0) -> dict:
    """Message 1: the atlas pages + their index + "Reply only OK."; message 2: the questions (as the QA test)."""
    from vbpe import claude_eval as CE
    c1 = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                       "data": base64.b64encode(b).decode("ascii")}} for b in pngs]
    c1.append({"type": "text", "text": context + ACK})
    stdin = "".join(json.dumps({"type": "user", "message": {"role": "user", "content": c}}) + "\n"
                    for c in (c1, [{"type": "text", "text": questions}])).encode()
    cmd = CE.build_command(GATE_SYSTEM, model, EFFORT)
    t0 = time.monotonic()
    rc, out, err = CE._run_cli(cmd, stdin, CE.build_env(), timeout)
    ev = [json.loads(l) for l in out.splitlines() if l.strip().startswith("{")]
    res = [e for e in ev if e.get("type") == "result"]
    asst = [e for e in ev if e.get("type") == "assistant"]
    u0 = (asst[0]["message"].get("usage") or {}) if asst else {}
    first_in = sum(u0.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    mu = (res[-1].get("modelUsage") or {}) if res else {}
    return {"text": (res[-1].get("result") or "") if len(res) == 2 else "", "rc": rc,
            "ok": len(res) == 2 and all(r.get("subtype") == "success" and not r.get("is_error") for r in res),
            "cost_usd": float((res[-1].get("total_cost_usd") if res else 0) or 0), "first_turn_input": first_in,
            "served": max(mu, key=lambda m: mu[m].get("outputTokens", 0)) if mu else None,
            "seconds": round(time.monotonic() - t0, 1), "error": None if res else (err[-300:] or out[-300:]),
            "final": (res[-1].get("result") or "")[-300:] if res else ""}


def gate(work: Path, models: list[str], cap: float, atlas_root: Path | None = None, label: str = "88.2") -> None:
    qs = json.loads((OUT / "gate_questions.json").read_text())
    tm = sw.task_map(work)
    done = {(r["model"], r["task"], r["label"]) for r in jl(OUT / "gate.jsonl") if r.get("ok")}
    for model in models:
        for iid in GATE_TASKS:
            if (model, iid, label) in done:
                continue
            meta = sw.contexts(work, tm[iid])
            adir = (atlas_root / iid) if atlas_root else Path(meta["atlas_dir"])
            pages = sorted(adir.glob("page_*.png"))
            from PIL import Image
            sizes = [Image.open(p).size for p in pages]
            formula = sum(-(-w // 28) * -(-h // 28) + 3 for w, h in sizes)
            if spent() + formula * 6e-6 + 0.05 > cap:     # Opus input at the cache-write rate ($5/M) + margin
                print(f"[gate] SKIP {model} {iid}: cap ${cap}", flush=True)
                return
            mine = [q for q in qs if q["task"] == iid]
            text = "\n".join(f"Q{i + 1}: {q['q']}" for i, q in enumerate(mine))
            questions = ("Answer each question from the page images only. Reply with one line per question, exactly "
                         f"`Q<n>: <answer>`, and nothing else.\n\n{text}")
            ctx = (f"The images above are the source files of the repository {tm[iid]['repo']} (tests, docs and data "
                   "files left out), in this order, each starting with a header ▶path◀. The code is minified: §<n> "
                   "marks the start of original line n (shown on every 10th line, a bare § starts the others); each » "
                   "before a line is one level of indentation where indentation matters; blank lines and other "
                   "indentation are not shown. Index (file: page):\n\n" + (adir / "index.txt").read_text())
            r = gate_call(model, [p.read_bytes() for p in pages], ctx, questions)
            append("spend.jsonl", {"what": f"gate {model} {iid} {label}", "cost_usd": r["cost_usd"],
                                   "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
            if sw.USAGE_LIMIT.search(r["final"] or "") and not r["ok"]:
                print(f"[gate] USAGE LIMIT: {r['final']}", flush=True)
                return
            ans = parse_answers(r["text"], len(mine))
            items = [{"q": i + 1, "kind": q["kind"], "third": q["third"], "page": q["page"], "answer": q["answer"],
                      "got": ans.get(i + 1, ""), "correct": score(q, ans.get(i + 1, ""))} for i, q in enumerate(mine)]
            row = {"model": model, "task": iid, "label": label, **{k: r[k] for k in r if k != "text"},
                   "pages": len(pages), "max_side": max(max(s) for s in sizes), "image_tokens_formula": formula,
                   "pages_arrived": r["first_turn_input"] >= sw.PAGE_ARRIVAL * formula,
                   "n": len(items), "correct": sum(x["correct"] for x in items), "items": items, "raw": r["text"][-3000:]}
            append("gate.jsonl", row)
            print(f"[gate] {model:20} {iid:35} {row['correct']}/{row['n']} first_in={r['first_turn_input']:,} "
                  f"formula={formula:,} arrived={row['pages_arrived']} ${r['cost_usd']:.3f} served={r['served']}",
                  flush=True)


def gate_summary(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        if not r.get("ok"):
            continue
        k = (r["model"], r["label"])
        d = out.setdefault(k, {"correct": 0, "n": 0, "arrived": True, "tasks": 0})
        d["correct"] += r["correct"]
        d["n"] += r["n"]
        d["tasks"] += 1
        d["arrived"] &= bool(r["pages_arrived"])
    for d in out.values():
        d["share"] = d["correct"] / d["n"] if d["n"] else 0.0
        d["passes"] = d["tasks"] == len(GATE_TASKS) and d["share"] >= GATE_PASS
    return out


# --------------------------------------------------------------------------- validation and runs
def validate(work: Path, workers: int) -> None:
    patch_rr()
    rr.validate(work, workers)


def run(work: Path, cap: float, max_runs: int | None, workers: int) -> None:
    patch_rr()
    tm = sw.task_map(work)
    vrows = {v["id"]: v for v in jl(OUT / "validate.jsonl") if v.get("valid")}
    order = [j for j in rr.smoke_first(run_order(rr.task_ids())) if j[0] in vrows]
    done = {(r["id"], r["arm"], r["rep"]) for r in jl(OUT / "runs.jsonl")}
    todo = [j for j in order if j not in done]
    if max_runs is not None:
        todo = todo[:max_runs]
    print(f"[run] {len(order)} jobs, {len(done)} done, {len(todo)} to run now; spent ${spent():.2f} (cap ${cap}) "
          f"model {sw.MODEL}", flush=True)
    rep1_left = {j for j in order if j[2] == 1 and j not in done}
    lock = threading.Lock()
    stop = threading.Event()

    def one(job):
        iid, arm, rep = job
        if stop.is_set():
            return
        if rep == 2:
            while True:
                with lock:
                    if not rep1_left:
                        break
                if stop.is_set():
                    return
                time.sleep(10)
        b = bound(arm, [r["cost_usd"] for r in jl(OUT / "runs.jsonl") if r["arm"] == arm])
        with _rlock:
            ok = spent() + _reserved[0] + b <= cap
            if ok:
                _reserved[0] += b
        if not ok:
            print(f"[run] SKIP {iid} {arm} r{rep}: ${spent():.2f} spent + reserved + bound ${b:.2f} > cap ${cap}",
                  flush=True)
            stop.set()
            return
        try:
            for attempt in range(2):
                try:
                    row = rr.run_one(work, tm[iid], arm, rep, set(vrows[iid].get("env_broken_p2p") or []))
                except Exception as ex:  # noqa: BLE001
                    print(f"[run] {iid} {arm} r{rep} ERROR {str(ex)[-300:]}", flush=True)
                    continue
                if sw.USAGE_LIMIT.search(row["final"] or "") and (row["is_error"] or row["rc"]):
                    stop.set()
                    append("spend.jsonl", {"what": f"{iid} {arm} r{rep}: usage-limit stop, not recorded", "cost_usd": 0})
                    print(f"[run] {iid} {arm} r{rep}: USAGE LIMIT, not recorded: {row['final'][-200:]}", flush=True)
                    return
                if row["subtype"] is None and row["cost_usd"] == 0 and attempt == 0:
                    print(f"[run] {iid} {arm} r{rep}: no result event (rc={row['rc']}); retry", flush=True)
                    continue
                append("runs.jsonl", row)
                print(f"[run] {iid:45} {arm:5} r{rep} resolved={row['resolved']!s:5} {row['grade'].get('f2p')} "
                      f"{row['grade'].get('p2p')} turns={row['turns']} ${row['cost_usd']:.2f} {row['seconds']}s "
                      f"first_in={row['first_turn_input']:,}{'' if row['pages_arrived'] else ' PAGES DROPPED'} "
                      f"leaks={len(row['leak_suspects'])} mem={row['memory_before']['memory_files']}/"
                      f"{row['memory_after']['memory_files']} model={list(row['model_usage'])} spent=${spent():.2f}",
                      flush=True)
                return
        finally:
            with _rlock:
                _reserved[0] -= b
            with lock:
                rep1_left.discard(job)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"[run] finished; spent ${spent():.2f}{' (STOPPED)' if stop.is_set() else ''}", flush=True)


# --------------------------------------------------------------------------- analysis
def arm_table(rows: list[dict], ids: list[str], label: str) -> list[str]:
    pt = rr.per_task(rows, ids)
    L = []
    for a in ARMS:
        rs = [r for r in rows if r["arm"] == a and r["id"] in ids]
        if not rs:
            continue
        full = [i for i in ids if (i, a) in pt]
        k = [sum(pt[(i, a)]) for i in full]
        r1 = [r for r in rs if r["rep"] == 1]
        r2 = [r for r in rs if r["rep"] == 2]
        L.append(f"| {label} {a} | {sum(r['resolved'] for r in rs)} of {len(rs)} | {k.count(2)} / {k.count(1)} / "
                 f"{k.count(0)} ({len(full)} tasks) | {sum(r['resolved'] for r in r1)} of {len(r1)} | "
                 f"{sum(r['resolved'] for r in r2)} of {len(r2)} | {statistics.median(r['turns'] or 0 for r in rs):g} | "
                 f"${statistics.mean(r['cost_usd'] for r in rs):.2f} | ${statistics.median(r['cost_usd'] for r in rs):.2f} | "
                 f"{statistics.median(r['seconds'] for r in rs):.0f} |")
    return L


def analyse() -> str:
    rows = jl(OUT / "runs.jsonl")
    ids = [i for i in rr.task_ids() if any(v["id"] == i and v.get("valid") for v in jl(OUT / "validate.jsonl"))]
    q1 = jl(Q1 / "runs.jsonl")
    pq = [r for r in jl(PQ) if r.get("arm") == "sonnet_atlas"]
    L = ["## Gate", ""]
    gs = gate_summary(jl(OUT / "gate.jsonl"))
    for (m, lab), d in sorted(gs.items()):
        L.append(f"* {m} at {lab} px²/char: {d['correct']} of {d['n']} ({100 * d['share']:.0f}%), {d['tasks']} atlases, "
                 f"all pages arrived: {d['arrived']}")
    L += ["", f"## Runs (Sonnet 5.5 high): tasks validated {len(ids)}; rows {len(rows)}", "",
          "| arm | task-runs solved | tasks both / one / neither | run 1 | run 2 | median turns | mean cost per run | "
          "median cost | median seconds |", "|---|---|---|---|---|---|---|---|---|"]
    L += arm_table(rows, ids, "Sonnet")
    L += arm_table(q1, ids, "Q1 Opus")
    if pq:
        L.append(f"| plan-quality Sonnet medium + atlas brief (1 run) | {sum(r['resolved'] for r in pq)} of {len(pq)} | - | "
                 f"- | - | {statistics.median(r['turns'] for r in pq):g} | ${statistics.mean(r['cost_usd'] for r in pq):.2f} | "
                 f"${statistics.median(r['cost_usd'] for r in pq):.2f} | {statistics.median(r['seconds'] for r in pq):.0f} |")
    by = {(r["id"], r["arm"], r["rep"]): r for r in rows}
    q1by = {(r["id"], r["arm"], r["rep"]): r for r in q1}
    pqby = {r["id"]: r for r in pq}
    yn = lambda r: ("yes" if r["resolved"] else "no") if r else "-"  # noqa: E731
    L += ["", "| task | Sonnet plain r1 / r2 (turns, $) | Sonnet atlas r1 / r2 (turns, $) | Q1 Opus plain r1/r2 | "
          "Q1 Opus atlas r1/r2 | PQ Sonnet brief |", "|---|---|---|---|---|---|"]
    for i in ids:
        cells = []
        for a in ARMS:
            c = []
            for k in REPS:
                r = by.get((i, a, k))
                c.append(f"{yn(r)} ({r['turns']}, ${r['cost_usd']:.2f})" if r else "-")
            cells.append(" / ".join(c))
        q = [" / ".join(yn(q1by.get((i, a, k))) for k in REPS) for a in ARMS]
        L.append(f"| {i} | {cells[0]} | {cells[1]} | {q[0]} | {q[1]} | {yn(pqby.get(i))} |")
    pt = rr.per_task(rows, ids)
    full = [i for i in ids if all((i, a) in pt for a in ARMS)]
    if full:
        kp = [sum(pt[(i, "plain")]) for i in full]
        ka = [sum(pt[(i, "atlas")]) for i in full]
        mt = lambda a: [statistics.mean(by[(i, a, k)]["turns"] for k in REPS) for i in full]  # noqa: E731
        mc = lambda a: [statistics.mean(by[(i, a, k)]["cost_usd"] for k in REPS) for i in full]  # noqa: E731
        ms = lambda a: [statistics.mean(by[(i, a, k)]["seconds"] for k in REPS) for i in full]  # noqa: E731
        s_res = sw.paired_sign(kp, ka)
        both = sw.mcnemar([x == 2 for x in ka], [x == 2 for x in kp])
        s_turn = sw.paired_sign(mt("atlas"), mt("plain"))
        s_cost = sw.paired_sign(mc("atlas"), mc("plain"))
        s_sec = sw.paired_sign(ms("atlas"), ms("plain"))
        lo, hi = rr.bootstrap_diff(ka, kp, seed=SEED)
        diff = sum(ka) - sum(kp)
        n_runs = 2 * len(full)
        ratio = statistics.mean(mc("atlas")) / max(1e-9, statistics.mean(mc("plain")))
        L += ["", f"## Paired tests over {len(full)} tasks with both runs of both arms (exact)",
              f"* resolved runs per task (0-2): atlas higher on {s_res['a_lower']}, plain higher on {s_res['b_lower']}, "
              f"tied {len(full) - s_res['n']}; sign test p = {s_res['p']:.3f}",
              f"* resolved in both runs: atlas only {both['only_a']}, plain only {both['only_b']}; exact McNemar "
              f"p = {both['p']:.3f}",
              f"* atlas - plain resolved task-runs: {diff:+d} of {n_runs}; bootstrap 95% interval over tasks "
              f"{lo:+.0f} .. {hi:+.0f}",
              f"* mean turns per task: atlas lower on {s_turn['a_lower']} of {s_turn['n']} untied, sign test p = "
              f"{s_turn['p']:.4f}; medians {statistics.median(mt('atlas')):g} vs {statistics.median(mt('plain')):g}",
              f"* mean cost per task: atlas lower on {s_cost['a_lower']} of {s_cost['n']}, p = {s_cost['p']:.4f}; "
              f"mean per run ${statistics.mean(mc('atlas')):.2f} vs ${statistics.mean(mc('plain')):.2f} (ratio {ratio:.2f})",
              f"* mean seconds per task: atlas lower on {s_sec['a_lower']} of {s_sec['n']}, p = {s_sec['p']:.3f}",
              "", "Flips between run 1 and run 2:"]
        for a in ARMS:
            fl = [i for i in full if pt[(i, a)][0] != pt[(i, a)][1]]
            L.append(f"* {a}: {len(fl)} of {len(full)} ({', '.join(fl) or 'none'})")
        L.append("* atlas vs plain within each run: " + ", ".join(
            f"run {k}: atlas {sum(by[(i, 'atlas', k)]['resolved'] for i in full)} vs plain "
            f"{sum(by[(i, 'plain', k)]['resolved'] for i in full)}" for k in REPS))
        # S3: Sonnet atlas vs Opus plain (Q1), task-runs on the tasks both have complete
        q1pt = rr.per_task(q1, ids)
        both_ids = [i for i in full if (i, "plain") in q1pt]
        s_at = sum(sum(pt[(i, "atlas")]) for i in both_ids)
        o_pl = sum(sum(q1pt[(i, "plain")]) for i in both_ids)
        sg = gate_summary(jl(OUT / "gate.jsonl"))
        s1 = sg.get((MODEL, "88.2"), {}).get("passes")
        L += ["", "## Predictions",
              f"* S1 the gate passes without re-rendering: {'PASS' if s1 else 'FAIL' if s1 is False else 'n/a'}",
              f"* S2 Sonnet atlas solves no more than plain (difference under 3 of 32): {'PASS' if diff < 3 else 'FAIL'} "
              f"({diff:+d} of {n_runs})",
              f"* S3 Sonnet atlas stays below Opus plain (Q1) on the {len(both_ids)} tasks complete in both: "
              f"{'PASS' if s_at < o_pl else 'FAIL'} ({s_at} vs {o_pl} of {2 * len(both_ids)})",
              f"* S4 the atlas multiplies Sonnet's cost per run by at least 3: {'PASS' if ratio >= 3 else 'FAIL'} "
              f"({ratio:.2f}x)"]
    for r in rows:                                            # re-scan with the current pattern
        b = DEFAULT_WORK / "runs" / r["dir_tag"]
        r["leak_suspects"] = rr.leak_hits(r.get("leak_suspects") or [], r["id"].split("__")[1].split("-")[0],
                                          rr.own_prefixes(b, b / r["repo_dir"]))
    leaks = [(r["id"], r["arm"], r["rep"], r["leak_suspects"]) for r in rows if r.get("leak_suspects")]
    mem = [(r["id"], r["arm"], r["rep"]) for r in rows
           if r["memory_before"]["memory_files"] or r["memory_after"]["memory_files"]]
    L += ["", f"## Checks", f"Leak suspects: {len(leaks)} runs"]
    L += [f"  * {x[0]} {x[1]} r{x[2]}: {x[3][:3]}" for x in leaks]
    L.append(f"Memory: runs with a memory file before or after: {len(mem)} of {len(rows)}; init events naming an "
             f"auto-memory path: {sum(bool(r.get('memory_path_init')) for r in rows)}")
    L.append(f"Pages dropped (atlas runs): {sum(not r['pages_arrived'] for r in rows)}; models served: "
             f"{sorted({m for r in rows for m in r.get('model_usage') or {}})}")
    L.append(f"Runs with more than one result event: {sum((r['results'] or 0) > 1 for r in rows)}; timeouts: "
             f"{sum(r['rc'] == -9 for r in rows)}; errors: {sum(bool(r['is_error']) for r in rows)}")
    L.append(f"\nSpent (list price, gate + agent runs): ${spent():.2f}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["gate-build", "gate", "validate", "order", "run", "analyse"])
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--max-runs", type=int)
    ap.add_argument("--models", nargs="+", default=[MODEL, "claude-opus-5-5"])
    ap.add_argument("--atlas-root", type=Path, help="gate: re-rendered atlases (<root>/<task>/page_*.png, index.txt)")
    ap.add_argument("--label", default="88.2")
    a = ap.parse_args(argv)
    setup_work(a.work)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "gate-build":
        for q in gate_build(a.work):
            print(q["task"][:20], q["third"], q["page"], q["kind"], q["path"], "->", q["answer"])
    elif a.cmd == "gate":
        gate(a.work, a.models, min(a.cap, 1.0) if a.label == "88.2" else a.cap, a.atlas_root, a.label)
        print(json.dumps({f"{m} {lab}": d for (m, lab), d in gate_summary(jl(OUT / "gate.jsonl")).items()}, indent=1))
    elif a.cmd == "validate":
        validate(a.work, a.workers)
    elif a.cmd == "order":
        for j in rr.smoke_first(run_order(rr.task_ids())):
            print(*j)
    elif a.cmd == "run":
        run(a.work, a.cap, a.max_runs, min(a.workers, 2))
    else:
        txt = analyse()
        (OUT / "summary.md").write_text(txt + "\n")
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
