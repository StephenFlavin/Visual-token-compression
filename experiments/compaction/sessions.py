"""Session material for the scaled compaction test (results/compaction_scale/README.md).

Each builder writes the material into a project directory and returns
    {"kind", "read_prompt", "remove": [paths to delete after reading], "questions": [40], "task": [5]}
Questions: {"id", "kind", "question", "answer"} with exact numeric answers.

* research(k): 14 Wikipedia articles with planted facts (claude_qa_research.build, seed 100 + k), read with Read.
* code(k): ~22 Python files of this repository (~280k chars), read with Read; constants, def/class lines, which file.
* logs(k): 15 synthetic service logs (5 services x 3 days, ~20k chars each) printed with Bash (`cat`); request
  latencies, deploy builds, incident sizes, p99 differences, the day with the highest p99.
"""
from __future__ import annotations

import random
import re
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PART_CHARS = 50_000                                     # Read refuses files above ~25k tokens

READ_FILES = """You are {role}. Read each of these files in full with the Read tool, one after another, in this order:
{files}
After each file, note in one short line what it covers. When you have read all of them, reply DONE."""

RUN_CMDS = """You are on call and reviewing yesterday's service logs before a postmortem. Run each of these commands with the
Bash tool, one after another, in this order, and look at the full output:
{files}
After each command, note in one short line what the log shows. When you have run all of them, reply DONE."""


def _wrap_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(textwrap.wrap(text, 1000, break_long_words=False, break_on_hyphens=False)) + "\n")


def research(k: int, proj: Path) -> dict:
    import sys
    sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
    import claude_qa_research as R
    arts = R.articles()[k * 14:(k + 1) * 14]
    _, srcs, qs = R.build(arts, seed=100 + k)
    names = []
    for s, txt in enumerate(srcs, 1):
        parts = [txt[i:i + PART_CHARS] for i in range(0, len(txt), PART_CHARS)]
        for j, part in enumerate(parts, 1):
            n = f"source_{s:02d}.txt" if len(parts) == 1 else f"source_{s:02d}_part{j}.txt"
            _wrap_write(proj / "sources" / n, part)
            names.append(f"sources/{n}")
    rng = random.Random(k)
    look = [q for q in qs if q["kind"] in R.LOOKUP]
    syn = [q for q in qs if q["kind"] in R.SYNTH]
    rng.shuffle(look)
    by = {}
    for q in syn:
        by.setdefault(q["kind"], []).append(q)
    syn_pick = [q for i in range(10) for kind in sorted(by) if i < len(by[kind]) for q in [by[kind][i]]][:20]
    pick = look[:20] + syn_pick
    rng.shuffle(pick)
    strip = lambda q: {x: q[x] for x in ("id", "kind", "question", "answer")}  # noqa: E731
    return {"kind": "research", "files": names, "remove": ["sources"],
            "read_prompt": READ_FILES.format(role="a research assistant collecting sources for a report",
                                             files="\n".join(names)),
            "questions": [strip(q) for q in pick], "task": [strip(q) for q in look[20:25]]}


CONST = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*=\s*(-?\d+(?:\.\d+)?)\s*(?:#.*)?$", re.M)
DEFN = re.compile(r"^(def|class) (\w+)", re.M)


def _code_files() -> list[Path]:
    fs = []
    for p in sorted(ROOT.glob("**/*.py")):
        rel = p.relative_to(ROOT).as_posix()
        if any(x in rel for x in ("/tests/", "test_", "data/", "__pycache__", "/cache/", "experiments/compaction/")):
            continue
        t = p.read_text(errors="ignore")
        if 2000 < len(t) < 50000 and len(DEFN.findall(t)) >= 3:
            fs.append(p)
    random.Random(7).shuffle(fs)
    return fs


def code(k: int, proj: Path) -> dict:
    fs, groups, cur, size = _code_files(), [], [], 0
    for p in fs:                                                  # 3 groups of ~280k characters
        cur.append(p)
        size += p.stat().st_size
        if size >= 280_000:
            groups.append(cur)
            cur, size = [], 0
    files = groups[k]
    names = []
    for p in files:
        rel = "code/" + p.relative_to(ROOT).as_posix()
        dst = proj / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst)
        names.append(rel)
    rng = random.Random(100 + k)
    consts, defs, owner = [], [], {}
    for i, (p, rel) in enumerate(zip(files, names), 1):
        t = p.read_text(errors="ignore")
        seen = {}
        for m in CONST.finditer(t):
            seen[m.group(1)] = seen.get(m.group(1), 0) + 1
        for m in CONST.finditer(t):
            if seen[m.group(1)] == 1:
                consts.append((rel, m.group(1), m.group(2)))
        lines = t.splitlines()
        names_here = [m.group(2) for m in DEFN.finditer(t)]
        for ln, line in enumerate(lines, 1):
            m = DEFN.match(line)
            if m and names_here.count(m.group(2)) == 1:
                defs.append((rel, m.group(1), m.group(2), ln))
                owner.setdefault(m.group(2), set()).add(i)
    rng.shuffle(consts)
    rng.shuffle(defs)
    q, n = [], 0

    def add(kind, text, ans, into):
        nonlocal n
        n += 1
        into.append({"id": f"{'KLW'['const def file'.split().index(kind)]}{n:02d}", "kind": kind, "question": text,
                     "answer": str(ans)})
    pool_c = [("const", f"In {r}, what number is the module-level constant {c} set to?", v) for r, c, v in consts]
    pool_d = [("def", f"On which line of {r} does the top-level `{d} {name}` statement start?", ln)
              for r, d, name, ln in defs]
    pool_f = [("file", f"Which file defines the top-level function or class `{name}`? Answer with the file's number "
                       f"in the reading order (1 = {names[0]}, {len(names)} = {names[-1]}).", next(iter(owner[name])))
              for r, d, name, ln in defs if len(owner[name]) == 1]
    seen_names = set()
    pool_f = [x for x in pool_f if not (x[1] in seen_names or seen_names.add(x[1]))]
    nc = min(12, len(pool_c) - 3)
    chosen = pool_c[:nc] + pool_d[:28 - nc] + pool_f[-12:]
    task_src = pool_c[nc:nc + 2] + pool_d[28 - nc:31 - nc]
    rng.shuffle(chosen)
    for kind, text, ans in chosen:
        add(kind, text, ans, q)
    task = []
    for kind, text, ans in task_src:
        add(kind, text, ans, task)
    return {"kind": "code", "files": names, "remove": ["code"],
            "read_prompt": READ_FILES.format(role="a developer getting to know this codebase before changing it",
                                             files="\n".join(names)),
            "questions": q, "task": task}


SERVICES = ["payments", "search", "auth", "ingest", "billing"]
ROUTES = {"payments": ["/v1/charge", "/v1/refund", "/v1/capture"], "search": ["/q", "/suggest", "/facet"],
          "auth": ["/login", "/token", "/refresh"], "ingest": ["/batch", "/stream", "/ack"],
          "billing": ["/invoice", "/usage", "/plan"]}


def _log(rng: random.Random, svc: str, day: int, used: set) -> tuple[str, dict]:
    def rid():
        while True:
            r = f"r-{rng.randint(10000, 99999)}"
            if r not in used:
                used.add(r)
                return r
    lines, reqs, t = [], [], 8 * 3600
    deploy_at, inc_at = rng.randint(20, 80), rng.randint(90, 150)
    build, inc, affected = rng.randint(40000, 69999), f"INC-{rng.randint(1000, 9999)}", rng.randint(50, 900)
    version = f"v{rng.randint(2, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 9)}"
    for i in range(175):
        t += rng.randint(20, 400)
        ts = f"2026-10-{day:02d}T{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}.{rng.randint(0, 999):03d}Z"
        if i == deploy_at:
            lines.append(f"{ts} INFO {svc} DEPLOY version={version} build={build} by=ci-runner-{rng.randint(1, 9)}")
            continue
        if i == inc_at:
            lines.append(f"{ts} WARN {svc} INCIDENT {inc} opened sev={rng.randint(1, 3)} affected_requests={affected}")
            continue
        r, lat = rid(), int(rng.lognormvariate(4.6, 0.7))
        status = rng.choices([200, 201, 404, 500, 503], [80, 8, 6, 4, 2])[0]
        lvl = "ERROR" if status >= 500 else "INFO"
        extra = f" err=E{rng.randint(100, 999)}" if status >= 500 else ""
        lines.append(f"{ts} {lvl} {svc} req={r} route={rng.choice(ROUTES[svc])} status={status} "
                     f"latency_ms={lat} user=u-{rng.randint(1000, 9999)}{extra}")
        reqs.append((r, lat))
    p99, p50 = rng.randint(600, 2500), rng.randint(60, 140)
    lines.append(f"2026-10-{day:02d}T23:59:59.999Z INFO {svc} SUMMARY day={day} requests={rng.randint(9000, 40000)} "
                 f"errors={rng.randint(20, 400)} p50_ms={p50} p99_ms={p99}")
    return "\n".join(lines) + "\n", {"reqs": reqs, "build": build, "inc": inc, "affected": affected, "p99": p99}


def logs(k: int, proj: Path) -> dict:
    rng, used, facts, names = random.Random(500 + k), set(), {}, []
    for svc in SERVICES:
        for day in (1, 2, 3):
            text, f = _log(rng, svc, day, used)
            p = proj / "logs" / f"{svc}_day{day}.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            facts[(svc, day)] = f
            names.append(f"cat logs/{svc}_day{day}.log")
    pool = {"latency": [], "build": [], "incident": [], "p99diff": [], "p99max": []}
    for (svc, day), f in facts.items():
        r, lat = rng.choice(f["reqs"])
        pool["latency"].append((f"In the {svc} log for day {day}, what latency_ms did request {r} have?", lat))
        pool["build"].append((f"Which build number was deployed to {svc} on day {day}?", f["build"]))
        pool["incident"].append((f"How many affected requests did incident {f['inc']} ({svc}, day {day}) report?",
                                 f["affected"]))
    for svc in SERVICES:
        p = [facts[(svc, d)]["p99"] for d in (1, 2, 3)]
        for a, b in ((1, 2), (2, 3), (1, 3)):
            pool["p99diff"].append((f"By how many milliseconds did {svc}'s daily-summary p99 differ between day {a} "
                                    f"and day {b}? (absolute difference)", abs(p[b - 1] - p[a - 1])))
        pool["p99max"].append((f"On which day (1, 2 or 3) was {svc}'s daily-summary p99 highest?", 1 + p.index(max(p))))
    for v in pool.values():
        rng.shuffle(v)
    want = {"latency": 12, "build": 8, "incident": 7, "p99diff": 8, "p99max": 5}
    qs, task, n = [], [], 0
    for kind, m in want.items():
        for text, ans in pool[kind][:m]:
            n += 1
            qs.append({"id": f"G{n:02d}", "kind": kind, "question": text, "answer": str(ans)})
    for kind, m in (("latency", 2), ("build", 2), ("incident", 1)):
        for text, ans in pool[kind][want[kind]:want[kind] + m]:
            n += 1
            task.append({"id": f"G{n:02d}", "kind": kind, "question": text, "answer": str(ans)})
    rng.shuffle(qs)
    return {"kind": "logs", "files": names, "remove": ["logs"], "read_prompt": RUN_CMDS.format(files="\n".join(names)),
            "questions": qs, "task": task}


BUILDERS = {"research": research, "code": code, "logs": logs}
