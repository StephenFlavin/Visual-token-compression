"""Repo atlas, Q2: does a whole-repository port (Go to Rust) go better when Claude Code has the source as page images?

Benchmark: RepoMod-Bench (github.com/Modelcode-ai/mcode-benchmark, commit 73093fc): whole-repository ports graded by
pytest suites that drive the built CLI. Project: ``gitleaks`` (Go -> Rust, v8.30.0 at ca20267, 35 tests). Design,
pre-registration and results: results/repo_atlas_port/README.md; plan: docs/REPO_ATLAS_PLAN.md (Q2).

    python3 experiments/repo_atlas_port.py prepare             # atlas (pages + index) and its size (no Claude calls)
    python3 experiments/repo_atlas_port.py reference           # build the Go original, run the hidden tests on it
    python3 experiments/repo_atlas_port.py smoke               # cheap harness check ($0.60 guard)
    python3 experiments/repo_atlas_port.py run --arms plain atlas    # agent runs (parallel); resumable after a restart
    python3 experiments/repo_atlas_port.py finish              # copy ports/transcripts, audit, memory check (after runs)
    python3 experiments/repo_atlas_port.py grade               # cargo build --release + the hidden pytest suite
    python3 experiments/repo_atlas_port.py analyse             # summary.md + B1-B3

Arms (Claude Code headless, Opus 5.5, --effort high, tools Bash/Read/Edit/Write, $15 guard, 2.5 h wall clock):
``plain`` (the benchmark's prompt; the Go repository in ``workspace/src``), ``atlas`` (first message: the Go source as
page images + index, answered "OK"; second message: the same prompt).

Scratch (work dirs, streams, cargo targets, the hidden tests) lives under PORT_WORK, never in the repo. The hidden
tests are stored there only as one XOR-scrambled tar (``stash/bench.bin``) and unpacked only while grading, after
every agent run has ended.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
WORK = Path(os.environ.get("PORT_WORK", "/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/port"))
OUT = ROOT / "results" / "repo_atlas_port"
PROJECT = "gitleaks"
SRC_REPO = "https://github.com/gitleaks/gitleaks"
SRC_COMMIT = "ca20267a84aa1fa2c2a9c1a13cdb50cafb48eeb0"      # metadata.yml source_commit (v8.30.0 tag + 2 fixes)
BENCH_COMMIT = "73093fccb69b31cdada5caad87127810c5b9ee28"
MODEL = "claude-opus-5-5"
EFFORT = "high"
ARMS = ("plain", "atlas")
CAP = 50.0                    # hard cap, list price, every Claude call of this test
GUARD = 15.0                  # per agent run (kill when the live cost reaches it)
WALL_S = 2.5 * 3600           # per agent run
MAX_TURNS = 500               # effectively unlimited: the guard and the wall clock bind
PAGE_ARRIVAL = 0.97
XOR = 0x5A
# Live guard prices, $ per million tokens (Opus 5.5 list: in 4, out 20, cache write 1.25x, cache read 0.1x)
PRICE = {"input_tokens": 4.0, "cache_creation_input_tokens": 5.0, "cache_read_input_tokens": 0.4, "output_tokens": 20.0}
USAGE_KEYS = tuple(PRICE)

# RepoMod-Bench dataset/gitleaks/benchmark.yml, rendered into templates/v3/prompt_cli.md.j2 (generate_prompts.py)
BENCH = {"source_lang": "Go", "source_desc": "Go-based gitleaks implementation", "target_lang": "Rust",
         "target_desc": "Rust-based gitleaks implementation", "build_instructions": "- Build: `cargo build --release`",
         "run_cmd": "./target/release/gitleaks"}
RUN_CMD = "./target/release/gitleaks"

ATLAS_TEXT = ("The {n_pages} page images above show the Go repository that is in `{ws}/src/`: its {n_files} source, "
              "config and documentation files (the Go unit tests and test data are left out), joined into one flow "
              "with a ▶path◀ header per file. The text is minified: §<n> marks the start of original line n (the "
              "number is shown on every 10th line, a bare § starts the others), » marks indentation where it "
              "matters, and other whitespace is not shown. They show the source at the start of the task: use them "
              "for orientation and read the files when exact text matters. Which page holds which file:\n\n{index}\n"
              "The task follows in the next message. Reply only OK.")
ACK = "Reply only OK."
RESUME_MSG = ("The session was interrupted by a machine restart. Continue the task from where you left off; the "
              "files you wrote are still on disk.")


# ----------------------------------------------------------------------------------------------- pure helpers


def render_prompt(template: str, ws: str) -> str:
    """The benchmark's v3 CLI prompt for gitleaks, as generate_prompts.py renders it (jinja2, trim_blocks,
    lstrip_blocks), with its container path /workspace replaced by the run's workspace."""
    import jinja2
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    text = env.from_string(template).render(**BENCH)
    return re.sub(r"(?<![\w.-])/workspace\b", lambda m: ws.rstrip("/"), text)


def user_msg(content) -> dict:
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    return {"type": "user", "message": {"role": "user", "content": content}}


def messages(arm: str, prompt: str, pngs: list[bytes] | None = None, atlas_text: str = "") -> list[dict]:
    """stream-json user messages: plain = the prompt; atlas = (pages + atlas text + "Reply only OK."), then the
    prompt (a context is read from the prompt cache only when it is its own first message)."""
    import base64
    if arm == "plain":
        if pngs:
            raise ValueError("the plain arm gets no pages")
        return [user_msg(prompt)]
    if not pngs or ACK not in atlas_text:
        raise ValueError(f"the {arm} arm needs its pages and the acknowledgement request")
    imgs = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(b).decode("ascii")}} for b in pngs]
    return [user_msg(imgs + [{"type": "text", "text": atlas_text}]), user_msg(prompt)]


def image_tokens(sizes) -> int:
    return sum(math.ceil(w / 28) * math.ceil(h / 28) for w, h in sizes)


def parse_events(lines) -> list[dict]:
    ev = []
    for l in lines:
        l = l.strip()
        if l.startswith("{"):
            try:
                ev.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return ev


def msg_usages(ev: list[dict]) -> list[tuple[str, dict, int]]:
    """Distinct assistant API messages in stream order: (message id, usage with each field's max over the message's
    events, index of the message's first event)."""
    out: dict[str, list] = {}
    current = None
    for i, e in enumerate(ev):
        if e.get("type") == "stream_event":               # --include-partial-messages: message_delta has the
            se = e.get("event") or {}                     # message's final output_tokens (assistant events do not)
            if se.get("type") == "message_start":
                m = se.get("message") or {}
                current, u = m.get("id"), m.get("usage") or {}
            elif se.get("type") == "message_delta":
                u = se.get("usage") or {}
            else:
                continue
            mid = current
        elif e.get("type") == "assistant":
            m = e.get("message") or {}
            mid, u = m.get("id"), m.get("usage") or {}
        else:
            continue
        if mid is None:
            continue
        if mid not in out:
            out[mid] = [mid, {k: 0 for k in USAGE_KEYS}, i]
        for k in USAGE_KEYS:
            out[mid][1][k] = max(out[mid][1][k], int(u.get(k) or 0))
    return [tuple(v) for v in out.values()]


def usage_cost(u: dict, price: dict = PRICE) -> float:
    return sum(u.get(k, 0) * p for k, p in price.items()) / 1e6


def segment_cost(ev: list[dict]) -> dict:
    """Cost of one CLI process: the last result's total_cost_usd (cumulative within a process) plus the live
    estimate of every API message after that result (a killed process writes no result for its last stretch)."""
    res_idx = [i for i, e in enumerate(ev) if e.get("type") == "result"]
    reported = float(ev[res_idx[-1]].get("total_cost_usd") or 0.0) if res_idx else 0.0
    last = res_idx[-1] if res_idx else -1
    msgs = msg_usages(ev)
    tail = sum(usage_cost(u) for _id, u, i in msgs if i > last)
    est_all = sum(usage_cost(u) for _id, u, _i in msgs)
    return {"reported": reported, "tail_est": tail, "cost": reported + tail, "estimated": tail > 0,
            "est_all": est_all, "n_results": len(res_idx)}


def guard_hit(cost_so_far: float, elapsed_s: float, guard: float = GUARD, wall_s: float = WALL_S) -> str | None:
    if cost_so_far >= guard:
        return "guard"
    if elapsed_s >= wall_s:
        return "wall"
    return None


def task_turns(ev: list[dict], arm: str) -> dict:
    """Turns of the task, the atlas arm's acknowledgement excluded. ``turns``: tool calls + closing answers (what the
    CLI's num_turns counts, measured; also defined for a run killed before its result); ``cli_turns``: the sum of
    num_turns over the task's result events; ``api_calls``: distinct assistant API messages (parallel tool calls
    share one)."""
    msgs = msg_usages(ev)
    res = [(i, e) for i, e in enumerate(ev) if e.get("type") == "result"]
    skip_before = -1
    if arm != "plain" and res:
        skip_before = res[0][0]                          # the first result closes the acknowledgement
    calls = [m for m in msgs if m[2] > skip_before]
    task_res = [e for i, e in res if i > skip_before]
    ack_calls = len(msgs) - len(calls)
    tools = {c.get("id") for i, e in enumerate(ev) if i > skip_before and e.get("type") == "assistant"
             for c in (e.get("message") or {}).get("content") or [] if isinstance(c, dict) and c.get("type") == "tool_use"}
    return {"turns": len(tools) + len(task_res), "api_calls": len(calls), "ack_calls": ack_calls,
            "cli_turns": sum(int(e.get("num_turns") or 0) for e in task_res) if task_res else None,
            "agent_s": round(sum(int(e.get("duration_ms") or 0) for _i, e in res) / 1000),
            "n_results": len(res),
            "usage": {k: sum(u[k] for _m, u, _i in msgs) for k in USAGE_KEYS}}


def tool_calls(ev: list[dict]) -> list[dict]:
    return [c for e in ev if e.get("type") == "assistant" for c in (e.get("message") or {}).get("content") or []
            if isinstance(c, dict) and c.get("type") == "tool_use"]


LIMIT = re.compile(r"usage limit|limit reached|hit your limit|resets? at|rate.?limit|overloaded|\b429\b|\b529\b", re.I)


def limit_hit(ev: list[dict], stderr: str) -> bool:
    """The process ended on a usage limit or API error rather than on the agent's own terms."""
    res = [e for e in ev if e.get("type") == "result"]
    if not res:
        return bool(LIMIT.search(stderr)) or not msg_usages(ev)
    r = res[-1]
    if r.get("subtype") in ("error_max_turns", "error_max_budget_usd"):
        return False
    return bool(r.get("is_error")) and bool(LIMIT.search(f"{r.get('result')} {stderr} {r.get('api_error_status')}"))


# Download / network commands that could fetch an existing port or the project's releases (hits read by hand).
LEAK = re.compile(r"\b(cargo\s+(?:install|add|search|fetch|vendor|info)\b|git\s+(?:clone|fetch|pull|submodule|remote)\b|"
                  r"curl\b|wget\b|go\s+(?:install|get)\b|pip3?\s+(?:install|download)\b|npm\s+(?:install|i)\b)|"
                  r"https?://|crates\.io|docs\.rs|lib\.rs|github\.com", re.I)
LEAK_DEP = re.compile(r"leak|secret|truffle|ripsecret|gitguardian|noseyparker|detect[-_]secrets", re.I)


def leak_hits(commands: list[str]) -> list[str]:
    return [c[:300] for c in commands if LEAK.search(c)]


def cargo_deps(cargo_toml: str) -> list[str]:
    """Dependency names from the [dependencies]-like tables of a Cargo.toml."""
    try:
        import tomllib
        d = tomllib.loads(cargo_toml)
    except Exception:  # noqa: BLE001
        return []
    names = []
    for key in ("dependencies", "dev-dependencies", "build-dependencies"):
        names += list((d.get(key) or {}).keys())
    for t in (d.get("target") or {}).values():
        for key in ("dependencies", "dev-dependencies", "build-dependencies"):
            names += list((t.get(key) or {}).keys())
    return sorted(set(names))


def lock_packages(lock: str) -> list[str]:
    return sorted(set(re.findall(r'(?m)^name = "([^"]+)"', lock)))


def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r'"(?:\\.|[^"\\])*"', '""', src)
    return re.sub(r"//[^\n]*", "", src)


def count_unsafe(src: str) -> int:
    return len(re.findall(r"\bunsafe\s*(?:\{|fn\b|impl\b|trait\b|extern\b)", strip_comments(src)))


CONC = {
    "thread_spawn": r"\bthread::(?:spawn|scope|Builder)\b|\.spawn\s*\(",
    "rayon": r"\brayon\b|\.par_iter\(|\.into_par_iter\(|\.par_bridge\(",
    "channels": r"\bmpsc\b|\bcrossbeam\w*|\bchannel\s*(?:::<[^>]*>)?\s*\(|\b(?:un)?bounded\s*(?:::<[^>]*>)?\s*\(",
    "mutex": r"\bMutex\b",
    "rwlock": r"\bRwLock\b",
    "arc": r"\bArc\b",
    "atomic": r"\bAtomic[A-Z]\w*",
    "async": r"\basync\s+(?:fn|move|\{)|\.await\b|\btokio\b",
}


def concurrency(src: str) -> dict:
    s = strip_comments(src)
    c = {k: len(re.findall(p, s)) for k, p in CONC.items()}
    c["concurrent"] = bool(c["thread_spawn"] or c["rayon"] or c["async"])
    return c


def rust_lines(files: dict[str, str]) -> int:
    return sum(sum(1 for l in t.splitlines() if l.strip()) for p, t in files.items() if p.endswith(".rs"))


def parse_junit(xml_text: str) -> dict[str, str]:
    """pytest junit xml -> {test id: passed|failed|error|skipped}."""
    out = {}
    root = ET.fromstring(xml_text)
    for tc in root.iter("testcase"):
        tid = f"{tc.get('classname')}::{tc.get('name')}"
        if tc.find("failure") is not None:
            out[tid] = "failed"
        elif tc.find("error") is not None:
            out[tid] = "error"
        elif tc.find("skipped") is not None:
            out[tid] = "skipped"
        else:
            out[tid] = "passed"
    return out


def score(outcomes: dict[str, str], reference: dict[str, str], built: bool) -> dict:
    """Tests that pass on the Go original here are scored; the others are excluded for every arm."""
    scored = sorted(t for t, o in reference.items() if o == "passed")
    passed = [t for t in scored if built and outcomes.get(t) == "passed"]
    n = len(scored)
    return {"built": built, "n_tests": n, "n_passed": len(passed), "share": len(passed) / n if n else 0.0,
            "all_pass": bool(built and n and len(passed) == n),
            "excluded": sorted(t for t, o in reference.items() if o != "passed"),
            "failed_tests": [t for t in scored if t not in passed]}


def predictions(rows: dict[str, dict]) -> dict:
    """B1 neither arm passes every test; B2 |share difference| < 15 points; B3 atlas uses fewer turns (api calls)."""
    p, a = rows["plain"], rows["atlas"]
    diff = abs(p["share"] - a["share"])
    return {"B1": {"pass": not p["all_pass"] and not a["all_pass"], "plain": p["n_passed"], "atlas": a["n_passed"],
                   "n": p["n_tests"]},
            "B2": {"pass": diff < 0.15, "diff_points": round(100 * diff, 1)},
            "B3": {"pass": a["turns"] < p["turns"], "plain": p["turns"], "atlas": a["turns"]}}


# ----------------------------------------------------------------------------------------------- scratch helpers


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


_lock = threading.Lock()


def append(name: str, row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with _lock, open(OUT / name, "a") as f:
        f.write(json.dumps(row) + "\n")


def spent() -> float:
    return sum(r.get("cost_usd") or 0 for r in jl(OUT / "spend.jsonl"))


def cli_store(cwd: Path) -> Path:
    return Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def xor(b: bytes) -> bytes:
    return b.translate(bytes(i ^ XOR for i in range(256)))


def unpack_tests(dest: Path) -> Path:
    """The hidden pytest suite, unpacked from the scrambled stash (only while grading)."""
    data = xor((WORK / "stash" / "bench.bin").read_bytes())
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(dest, filter="data")
    return dest / PROJECT / "tests"


def src_tree(d: Path, skip=(".git", "target")) -> dict[str, str]:
    out = {}
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d)
        if p.is_file() and not any(s in rel.parts for s in skip):
            try:
                out[str(rel)] = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                pass
    return out


def run_group(cmd, cwd: Path, env: dict, timeout: float) -> tuple[str, bool, int]:
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return out.decode("utf-8", "replace"), False, p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return out.decode("utf-8", "replace"), True, -9


def pytest_suite(tests: Path, work_dir: Path, run_cmd: str, junit: Path) -> dict[str, str]:
    cmd = [sys.executable, "-m", "pytest", f"--work-dir={work_dir}", f"--run-cmd={run_cmd}", "-q", "-p",
           "no:cacheprovider", "--timeout=120", f"--junitxml={junit}"]
    out, timed_out, rc = run_group(cmd, tests, dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), 1800)
    if not junit.exists():
        raise RuntimeError(f"pytest wrote no junit xml (rc {rc}, timed out {timed_out}): {out[-800:]}")
    return parse_junit(junit.read_text())


# ----------------------------------------------------------------------------------------------- prepare / reference


def source_checkout() -> Path:
    """The Go repository at the benchmark's commit (a git checkout, for the atlas's file list)."""
    d = WORK / "src_clone" / PROJECT
    if not (d / ".git").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", SRC_REPO, str(d)], check=True)
    head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != SRC_COMMIT:
        subprocess.run(["git", "-C", str(d), "fetch", "-q", "--depth", "1", "origin", SRC_COMMIT], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", SRC_COMMIT], check=True)
    return d


def prepare() -> dict:
    import repo_atlas as RA
    src = source_checkout()
    ad = WORK / "atlas" / PROJECT
    if not (ad / "index.json").exists():
        RA.build(src, ad, 88.2, False, 10)
    meta = json.loads((ad / "index.json").read_text())
    info = {"project": PROJECT, "rule": "repo_atlas.py build with default settings (88.2 px2/char, tests/testdata/"
            "vendored dirs excluded, files > 200 kB skipped, line number on every 10th line)",
            "files": meta["files"], "pages": len(meta["pages"]), "image_tokens": meta["image_tokens"],
            "image_tokens_formula": image_tokens([p["size"] for p in meta["pages"]]),
            "text_tokens_est": meta["text_tokens_est"], "uncovered": meta["uncovered"],
            "fits": meta["image_tokens"] <= RA.FIT_TOKENS, "index": meta["index"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "atlas.json").write_text(json.dumps(info, indent=1))
    print(f"[prepare] atlas {info['pages']} pages, {info['image_tokens']:,} image tokens (fits: {info['fits']}), "
          f"{info['files']} files, text ~{info['text_tokens_est']:,} tokens")
    return info


def reference() -> dict:
    """Build the Go original as the benchmark does and run the hidden suite against it."""
    src = source_checkout()
    d = WORK / "ref" / "src"
    shutil.rmtree(d.parent, ignore_errors=True)
    shutil.copytree(src, d, ignore=shutil.ignore_patterns(".git"))
    out, to, rc = run_group(["go", "build", "-buildvcs=false", "-o", "gitleaks"], d, dict(os.environ), 900)
    if rc:
        raise RuntimeError(f"go build failed: {out[-800:]}")
    tests = unpack_tests(WORK / "ref" / "bench")
    outcomes = pytest_suite(tests, d, "./gitleaks", WORK / "ref" / "junit.xml")
    shutil.rmtree(d.parent, ignore_errors=True)
    row = {"project": PROJECT, "commit": SRC_COMMIT, "bench_commit": BENCH_COMMIT,
           "go": subprocess.run(["go", "version"], capture_output=True, text=True).stdout.strip(),
           "n": len(outcomes), "passed": sum(o == "passed" for o in outcomes.values()), "outcomes": outcomes}
    (OUT / "reference.json").write_text(json.dumps(row, indent=1))
    print(f"[reference] Go original: {row['passed']}/{row['n']} tests pass")
    return row


# ----------------------------------------------------------------------------------------------- agent runs


def workspace(arm: str) -> Path:
    return WORK / "runs" / arm / "workspace"


def make_workspace(arm: str) -> Path:
    base = WORK / "runs" / arm
    shutil.rmtree(base, ignore_errors=True)
    ws = base / "workspace"
    shutil.copytree(source_checkout(), ws / "src", ignore=shutil.ignore_patterns(".git"))
    (ws / "dst").mkdir()
    (base / "tmp").mkdir()
    template = (WORK / "stash" / "prompt_cli_v3.j2.txt").read_text()
    (ws / "prompt.md").write_text(render_prompt(template, str(ws)))
    return ws


def atlas_message_parts() -> tuple[list[bytes], str, list]:
    ad = WORK / "atlas" / PROJECT
    meta = json.loads((ad / "index.json").read_text())
    pngs = [(ad / f"page_{p['page']:03d}.png").read_bytes() for p in meta["pages"]]
    return pngs, (ad / "index.txt").read_text(), [p["size"] for p in meta["pages"]]


def state_path(arm: str) -> Path:
    return WORK / "runs" / arm / "state.json"


def load_state(arm: str) -> dict | None:
    p = state_path(arm)
    return json.loads(p.read_text()) if p.exists() else None


def save_state(arm: str, st: dict) -> None:
    p = state_path(arm)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, p)


def agent_env(arm_dir: Path) -> dict:
    from vbpe import claude_eval as CE
    env = CE.build_env()
    env.update({k: os.environ[k] for k in ("CARGO_HOME", "RUSTUP_HOME", "GOPATH", "GOROOT") if k in os.environ})
    env.update(CARGO_BUILD_JOBS="2", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", TMPDIR=str(arm_dir / "tmp"))
    return env


def cli_command(session: str, resume: bool, guard_left: float) -> list[str]:
    from vbpe import claude_eval as CE
    cmd = [CE.CLAUDE_BIN, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", MODEL, "--effort", EFFORT, "--tools", "Bash,Read,Edit,Write",
           "--allowedTools", "Bash,Read,Edit,Write",
           "--disallowedTools", "Bash(curl:*)", "Bash(wget:*)", "Bash(git clone:*)", "Bash(cargo install:*)",
           "WebFetch", "WebSearch",
           "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
           "--include-partial-messages",              # message_delta events carry each message's real output tokens
           "--max-turns", str(MAX_TURNS), "--max-budget-usd", f"{max(0.01, guard_left):.2f}"]
    cmd += ["--resume", session] if resume else ["--session-id", session]
    return cmd


def segment(arm: str, st: dict, msgs: list[dict], resume: bool, guard: float, wall_s: float) -> dict:
    """Run one CLI process; the stream goes to disk line by line; kill the process group when the live cost
    (earlier segments + this one) reaches the guard or the wall clock runs out."""
    arm_dir = WORK / "runs" / arm
    n = len(st["segments"])
    stream = arm_dir / f"stream_{n}.jsonl"
    prior_cost = sum(s.get("cost", 0.0) for s in st["segments"])
    prior_s = sum(s.get("elapsed_s", 0.0) for s in st["segments"])
    cmd = cli_command(st["session_id"], resume, guard - prior_cost)
    p = subprocess.Popen(cmd, cwd=workspace(arm), env=agent_env(arm_dir), stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    seg = {"n": n, "resume": resume, "pid": p.pid, "started": time.time(), "stream": stream.name, "killed": None}
    st["segments"].append(seg)
    st["status"] = "running"
    save_state(arm, st)
    events: list[dict] = []
    err_chunks: list[bytes] = []

    def reader():
        with open(stream, "a") as f:
            for raw in p.stdout:
                line = raw.decode("utf-8", "replace")
                f.write(line)
                f.flush()
                events.extend(parse_events([line]))

    def err_reader():
        for raw in p.stderr:
            err_chunks.append(raw)

    th = [threading.Thread(target=reader, daemon=True), threading.Thread(target=err_reader, daemon=True)]
    for t in th:
        t.start()
    p.stdin.write("".join(json.dumps(m) + "\n" for m in msgs).encode())
    p.stdin.close()
    last_save = 0.0
    while p.poll() is None:
        time.sleep(3)
        el = time.time() - seg["started"]
        c = segment_cost(list(events))
        why = guard_hit(prior_cost + c["cost"], prior_s + el, guard, wall_s)
        if time.time() - last_save > 30:
            seg.update(elapsed_s=el, cost=c["cost"], cost_est_live=True)
            save_state(arm, st)
            last_save = time.time()
        if why:
            seg["killed"] = why
            print(f"[port] {arm}: {why} at ${prior_cost + c['cost']:.2f}, {(prior_s + el) / 3600:.2f} h: killing", flush=True)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
    p.wait()
    for t in th:
        t.join(timeout=30)
    el = time.time() - seg["started"]
    ev = parse_events(stream.read_text().splitlines())
    c = segment_cost(ev)
    stderr = b"".join(err_chunks).decode("utf-8", "replace")
    sid = next((e.get("session_id") for e in ev if e.get("type") == "system" and e.get("session_id")), None)
    seg.update(elapsed_s=el, cost=c["cost"], cost_reported=c["reported"], cost_tail_est=c["tail_est"],
               cost_est_all=c["est_all"], cost_estimated=c["estimated"], exit=p.returncode, cli_session=sid,
               stderr_tail=stderr[-600:], ended=time.time(), cost_est_live=False)
    limit = seg["killed"] is None and limit_hit(ev, stderr)
    seg["usage_limit"] = limit
    st["status"] = "usage_limit" if limit else "ended"
    save_state(arm, st)
    append("spend.jsonl", {"what": f"{PROJECT} {arm} segment {n}", "cost_usd": round(c["cost"], 4),
                           "reported": c["reported"], "tail_estimated": round(c["tail_est"], 4),
                           "killed": seg["killed"], "usage_limit": limit, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return seg


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        return "claude" in Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return False


def run_arm(arm: str, guard: float = GUARD, wall_s: float = WALL_S) -> dict:
    """Start the arm, or carry it on after a restart: a run whose process died without ending (container restart)
    is resumed once with --resume in the same directory; a second death leaves it cut."""
    st = load_state(arm)
    if st and st["status"] in ("ended", "finished"):
        return st
    if st and st["status"] == "usage_limit":
        raise RuntimeError(f"{arm}: stopped on a usage limit; resume by hand when the limit resets")
    if st and st["status"] == "running":
        seg = st["segments"][-1]
        if pid_alive(seg.get("pid")):
            raise RuntimeError(f"{arm}: its CLI process {seg['pid']} is still running")
        ev = parse_events((WORK / "runs" / arm / seg["stream"]).read_text().splitlines())
        c = segment_cost(ev)
        seg.update(killed="restart", cost=c["cost"], cost_reported=c["reported"], cost_tail_est=c["tail_est"],
                   cost_estimated=c["estimated"], elapsed_s=seg.get("elapsed_s", 0.0), cost_est_live=False)
        append("spend.jsonl", {"what": f"{PROJECT} {arm} segment {seg['n']} (died in a restart)",
                               "cost_usd": round(c["cost"], 4), "reported": c["reported"],
                               "tail_estimated": round(c["tail_est"], 4), "killed": "restart"})
        if st.get("resumed"):
            st["status"] = "ended"
            save_state(arm, st)
            return st
        st["resumed"] = True
        save_state(arm, st)
        prior = sum(s.get("cost", 0.0) for s in st["segments"])
        if spent() + (guard - prior) > CAP:
            raise RuntimeError(f"{arm}: resuming could cross the cap")
        print(f"[port] {arm}: resuming session {st['session_id']} after a restart (spent ${prior:.2f} so far)", flush=True)
        segment(arm, st, [user_msg(RESUME_MSG)], True, guard, wall_s)
        return load_state(arm)
    # fresh start
    if spent() + guard > CAP:
        raise RuntimeError(f"{arm}: a ${guard} run could cross the ${CAP} cap (spent ${spent():.2f})")
    ws = make_workspace(arm)
    store = cli_store(ws)
    pre_store = store.exists()
    shutil.rmtree(store, ignore_errors=True)
    prompt = (ws / "prompt.md").read_text()
    sizes = []
    if arm == "plain":
        msgs = messages("plain", prompt)
    else:
        pngs, index, sizes = atlas_message_parts()
        text = ATLAS_TEXT.format(n_pages=len(pngs), ws=str(ws), index=index,
                                 n_files=json.loads((WORK / "atlas" / PROJECT / "index.json").read_text())["files"])
        msgs = messages(arm, prompt, pngs, text)
    st = {"arm": arm, "project": PROJECT, "session_id": str(uuid.uuid4()), "workspace": str(ws), "segments": [],
          "status": "new", "resumed": False, "guard": guard, "wall_s": wall_s, "store_existed_before": pre_store,
          "image_tokens_formula": image_tokens(sizes), "started": time.time()}
    save_state(arm, st)
    print(f"[port] start {arm} (session {st['session_id']}, guard ${guard}, wall {wall_s / 3600:.1f} h)", flush=True)
    segment(arm, st, msgs, False, guard, wall_s)
    return load_state(arm)


def run(arms: list[str]) -> None:
    errs = {}

    def go(a):
        try:
            st = run_arm(a)
            segs = st["segments"]
            print(f"[port] {a}: {st['status']}, ${sum(s.get('cost', 0) for s in segs):.2f}, "
                  f"{sum(s.get('elapsed_s', 0) for s in segs) / 3600:.2f} h, killed={[s.get('killed') for s in segs]}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            errs[a] = repr(e)
            print(f"[port] {a}: ERROR {e!r}", flush=True)

    ts = [threading.Thread(target=go, args=(a,)) for a in arms]
    for t in ts:
        t.start()
        time.sleep(10)
    for t in ts:
        t.join()
    print(f"[port] runs over; spent ${spent():.2f} of ${CAP}; errors: {errs or 'none'}", flush=True)


# ----------------------------------------------------------------------------------------------- finish / grade


ALLOWED_OUTSIDE = ("/root/.cargo", "/root/.rustup", "/root/go/", "/usr/", "/bin/", "/lib", "/etc/", "/dev/", "/proc/",
                   "/opt/", "/sys/")


def audit(calls: list[dict], arm_dir: Path) -> dict:
    """Paths outside the run's own directory (toolchains and system dirs allowed), filesystem-wide searches, reads of
    Claude Code's stores, and download commands. Hits are read by hand."""
    own = str(arm_dir)
    store = str(cli_store(arm_dir / "workspace"))
    outside, searches, claude_store = [], [], []
    for c in calls:
        inp = c.get("input") or {}
        text = " ".join(str(v) for v in inp.values() if isinstance(v, (str, int)))
        for p in re.findall(r"(?<![\w.~/:-])(/[A-Za-z_][\w./-]*)", text):
            if p.startswith(own) or p.startswith(store) or p.startswith(ALLOWED_OUTSIDE) or p == "/":
                continue
            if p.startswith("/tmp/") and not p.startswith("/tmp/claude"):
                continue
            outside.append(p[:160])
        if re.search(r"\.claude/", text):
            claude_store.append(text[:200])
        if c.get("name") == "Bash" and re.search(r"\b(find|locate|grep\s+-r\w*|rg|ls)\s+(-\S+\s+)*(/|~|\$HOME|/tmp|/root|/home)(\s|$)",
                                                  str(inp.get("command", ""))):
            searches.append(str(inp.get("command"))[:200])
    return {"outside_paths": sorted(set(outside))[:40], "fs_searches": searches[:10], "claude_store": claude_store[:10],
            "suspect": bool(outside or searches or claude_store)}


def transcript_line(line: str) -> str | None:
    """Transcript for results/: token-delta events dropped (message_start/message_delta kept for usage), base64
    images removed."""
    if '"type":"stream_event"' in line.replace(" ", "") and not re.search(r'"type": ?"message_(start|delta)"', line):
        return None
    return strip_images(line)


def strip_images(line: str) -> str:
    return re.sub(r'"data": ?"[A-Za-z0-9+/=]{200,}"', '"data": "<base64 removed>"', line)


def finish() -> None:
    """After every arm's agent has ended: copy each port's source and transcript into results/, audit the commands,
    check Claude Code's store for memory, delete the store, write runs.jsonl rows."""
    states = {a: load_state(a) for a in ARMS}
    if any(s is None or s["status"] not in ("ended", "finished") for s in states.values()):
        raise SystemExit(f"not every arm has ended: { {a: (s or {}).get('status') for a, s in states.items()} }")
    done = {r["arm"] for r in jl(OUT / "runs.jsonl")}
    for arm, st in states.items():
        if arm in done:
            continue
        arm_dir = WORK / "runs" / arm
        ws = Path(st["workspace"])
        ev = []
        lines = []
        for s in st["segments"]:
            ls = (arm_dir / s["stream"]).read_text().splitlines()
            lines += ls
            ev += parse_events(ls)
        tdir = OUT / "transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        with gzip.open(tdir / f"{arm}.jsonl.gz", "wt") as f:
            f.write("\n".join(t for t in (transcript_line(l) for l in lines) if t is not None) + "\n")
        calls = tool_calls(ev)
        cmds = [str((c.get("input") or {}).get("command", "")) for c in calls if c.get("name") == "Bash"]
        (OUT / "commands").mkdir(exist_ok=True)
        (OUT / "commands" / f"{arm}.txt").write_text("\n".join(f"$ {c}" for c in cmds) + "\n")
        dst = ws / "dst"
        port_dir = OUT / "ports" / arm
        shutil.rmtree(port_dir, ignore_errors=True)
        shutil.copytree(dst, port_dir, ignore=shutil.ignore_patterns("target", ".git", "*.o", "*.rlib"))
        files = src_tree(port_dir)
        rs = "\n".join(t for p, t in files.items() if p.endswith(".rs"))
        deps = cargo_deps(files.get("Cargo.toml", ""))
        lock = lock_packages(files.get("Cargo.lock", ""))
        store = cli_store(ws)
        store_files = sorted(str(p.relative_to(store)) for p in store.rglob("*") if p.is_file()) if store.exists() else []
        tt = task_turns(ev, arm)
        first = msg_usages(ev)[0][1] if msg_usages(ev) else {}
        first_in = sum(first.get(k, 0) for k in USAGE_KEYS[:3])
        res = [e for e in ev if e.get("type") == "result"]
        row = {"project": PROJECT, "arm": arm, "model": MODEL, "effort": EFFORT, "session_id": st["session_id"],
               "turns": tt["turns"], "api_calls": tt["api_calls"], "cli_turns": tt["cli_turns"], "ack_calls": tt["ack_calls"],
               "agent_s": tt["agent_s"], "n_results": tt["n_results"],
               "wall_s": round(sum(s.get("elapsed_s", 0) for s in st["segments"])),
               "hours": round(sum(s.get("elapsed_s", 0) for s in st["segments"]) / 3600, 2),
               "cost_usd": round(sum(s.get("cost", 0) for s in st["segments"]), 4),
               "cost_estimated": any(s.get("cost_estimated") for s in st["segments"]),
               "usage": tt["usage"], "output_tokens": tt["usage"]["output_tokens"],
               "input_tokens": sum(tt["usage"][k] for k in USAGE_KEYS[:3]),
               "segments": [{k: s.get(k) for k in ("n", "resume", "killed", "cost", "cost_reported", "elapsed_s", "exit",
                                                   "usage_limit")} for s in st["segments"]],
               "cut": any(s.get("killed") for s in st["segments"]),
               "cut_reason": [s.get("killed") for s in st["segments"] if s.get("killed")],
               "resumed": st.get("resumed", False),
               "subtypes": [r.get("subtype") for r in res], "final": (res[-1].get("result") or "")[-600:] if res else "",
               "first_call_input": first_in, "image_tokens_formula": st.get("image_tokens_formula", 0),
               "pages_arrived": arm == "plain" or first_in >= PAGE_ARRIVAL * st.get("image_tokens_formula", 0),
               "rust_lines": rust_lines(files), "rust_files": sum(p.endswith(".rs") for p in files),
               "unsafe": count_unsafe(rs), "concurrency": concurrency(rs), "deps": deps,
               "dep_flags": [d for d in deps + lock if LEAK_DEP.search(d)],
               "leak_hits": leak_hits(cmds), "n_bash": len(cmds),
               "tools": {k: sum(c.get("name") == k for c in calls) for k in sorted({c.get("name") for c in calls})},
               "audit": audit(calls, arm_dir),
               "memory": {"store_existed_before": st.get("store_existed_before"), "store_files": store_files[:50],
                          "memory_dir": any(f.startswith("memory") for f in store_files)}}
        append("runs.jsonl", row)
        shutil.rmtree(store, ignore_errors=True)
        st["status"] = "finished"
        save_state(arm, st)
        print(f"[finish] {arm}: turns {row['turns']} ${row['cost_usd']:.2f} {row['hours']} h cut={row['cut']} "
              f"rust lines {row['rust_lines']} concurrent={row['concurrency']['concurrent']} unsafe={row['unsafe']} "
              f"pages_arrived={row['pages_arrived']} leak_hits={len(row['leak_hits'])} dep_flags={row['dep_flags']} "
              f"suspect={row['audit']['suspect']} memory={row['memory']['memory_dir']}", flush=True)


def grade() -> None:
    """Clean release build of each port's source (results/repo_atlas_port/ports/<arm>), then the hidden suite."""
    ref = json.loads((OUT / "reference.json").read_text())
    if any((load_state(a) or {}).get("status") != "finished" for a in ARMS):
        raise SystemExit("finish every arm before grading (the tests must stay hidden while any agent runs)")
    done = {r["arm"] for r in jl(OUT / "grades.jsonl")}
    for arm in ARMS:
        if arm in done:
            continue
        g = WORK / "grade" / arm
        shutil.rmtree(g, ignore_errors=True)
        shutil.copytree(OUT / "ports" / arm, g / "dst")
        env = dict(os.environ, CARGO_BUILD_JOBS="4", CARGO_TERM_COLOR="never")
        env.pop("CARGO_TARGET_DIR", None)
        t0 = time.time()
        log, to, rc = run_group(["cargo", "build", "--release"], g / "dst", env, 2400)
        built = rc == 0 and (g / "dst" / "target" / "release" / "gitleaks").exists()
        outcomes = {}
        if built:
            tests = unpack_tests(g / "bench")
            outcomes = pytest_suite(tests, g / "dst", RUN_CMD, g / "junit.xml")
        row = {"project": PROJECT, "arm": arm, **score(outcomes, ref["outcomes"], built), "outcomes": outcomes,
               "build_s": round(time.time() - t0), "build_timeout": to, "build_log_tail": log[-1500:]}
        append("grades.jsonl", row)
        shutil.rmtree(g, ignore_errors=True)
        print(f"[grade] {arm}: built={built} {row['n_passed']}/{row['n_tests']} ({100 * row['share']:.1f}%)", flush=True)


def analyse() -> str:
    runs = {r["arm"]: r for r in jl(OUT / "runs.jsonl")}
    grades = {r["arm"]: r for r in jl(OUT / "grades.jsonl")}
    rows = {a: {**runs[a], **grades[a]} for a in ARMS if a in runs and a in grades}
    lines = ["# Repo atlas, Q2 (gitleaks Go -> Rust): summary", "",
             "*Written by `experiments/repo_atlas_port.py analyse`. One run per arm: an anecdote, not a measurement.*", "",
             "| Arm | Tests passed | Builds | Concurrent | thread/rayon/async | Mutex / RwLock / Arc / channels | `unsafe` | "
             "Rust lines | Turns | CLI turns | API calls | Cost | Hours | Output tokens | Cut |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in rows.items():
        c = r["concurrency"]
        lines.append(f"| {a} | {r['n_passed']} / {r['n_tests']} ({100 * r['share']:.1f}%) | {'yes' if r['built'] else 'no'} | "
                     f"{'yes' if c['concurrent'] else 'no'} | {c['thread_spawn']} / {c['rayon']} / {c['async']} | "
                     f"{c['mutex']} / {c['rwlock']} / {c['arc']} / {c['channels']} | {r['unsafe']} | {r['rust_lines']:,} | "
                     f"{r['turns']} | {r['cli_turns']} | {r['api_calls']} | ${r['cost_usd']:.2f} | {r['hours']:.2f} | {r['output_tokens']:,} | "
                     f"{', '.join(r['cut_reason']) if r['cut'] else 'no'} |")
    if len(rows) == 2:
        p = predictions(rows)
        lines += ["", "**Predictions:**",
                  f"* B1 (neither arm passes every test): **{'PASS' if p['B1']['pass'] else 'FAIL'}** "
                  f"(plain {p['B1']['plain']}, atlas {p['B1']['atlas']} of {p['B1']['n']})",
                  f"* B2 (test shares differ by < 15 points): **{'PASS' if p['B2']['pass'] else 'FAIL'}** "
                  f"({p['B2']['diff_points']} points)",
                  f"* B3 (atlas uses fewer turns): **{'PASS' if p['B3']['pass'] else 'FAIL'}** "
                  f"(plain {p['B3']['plain']}, atlas {p['B3']['atlas']} turns)"]
        failed = sorted(set(rows["plain"]["failed_tests"]) | set(rows["atlas"]["failed_tests"]))
        if failed:
            lines += ["", "| Failing test | plain | atlas |", "|---|---|---|"]
            for t in failed:
                lines.append(f"| `{t.split('::', 1)[-1]}` | {rows['plain']['outcomes'].get(t, 'not run')} | "
                             f"{rows['atlas']['outcomes'].get(t, 'not run')} |")
    lines += ["", f"Spend: ${spent():.2f} of ${CAP} (spend.jsonl)."]
    text = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(text)
    return text


# ----------------------------------------------------------------------------------------------- smoke


SMOKE_PROMPT = ("Translate the Go program in `{ws}/src/main.go` to Rust: a cargo package in `{ws}/dst/` whose binary is "
                "named `hello`. Build it with `cargo build --release`, run it once, and reply DONE.")


SMOKE_GUARD_PROMPT = ("In `{ws}/dst/`, create ten files a1.txt ... a10.txt, one per Write tool call, each holding a "
                      "four-line poem about a different Rust keyword. Then reply DONE.")


def smoke(guard: float = 0.60, no_atlas: bool = False) -> None:
    """Harness check on a toy task with the atlas arm's two-message input: stream to disk, live accounting against
    the CLI's reported cost, page arrival, the guard (set low enough that it may fire), memory store."""
    arm = "smoke_guard" if no_atlas else "smoke"
    base = WORK / "runs" / arm
    shutil.rmtree(base, ignore_errors=True)
    ws = base / "workspace"
    (ws / "src").mkdir(parents=True)
    (ws / "dst").mkdir()
    (base / "tmp").mkdir()
    (ws / "src" / "main.go").write_text('package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("hello")\n}\n')
    if no_atlas:                                 # guard check: a cheap task that runs past a tiny guard
        sizes = []
        msgs = messages("plain", SMOKE_GUARD_PROMPT.format(ws=ws))
    else:
        pngs, index, sizes = atlas_message_parts()
        text = ATLAS_TEXT.format(n_pages=len(pngs), ws=str(ws), index=index, n_files=0)
        msgs = messages("atlas", SMOKE_PROMPT.format(ws=ws), pngs, text)
    st = {"arm": arm, "session_id": str(uuid.uuid4()), "workspace": str(ws), "segments": [], "status": "new",
          "store_existed_before": cli_store(ws).exists()}
    save_state(arm, st)
    seg = segment(arm, st, msgs, False, guard, 900)
    ev = parse_events((base / seg["stream"]).read_text().splitlines())
    msgs_u = msg_usages(ev)
    first_in = sum(msgs_u[0][1].get(k, 0) for k in USAGE_KEYS[:3]) if msgs_u else 0
    res = [e for e in ev if e.get("type") == "result"]
    store = cli_store(ws)
    store_files = sorted(str(p.relative_to(store)) for p in store.rglob("*") if p.is_file()) if store.exists() else []
    row = {"what": "smoke", "killed": seg["killed"], "cost": seg["cost"], "reported": seg["cost_reported"],
           "est_all": seg["cost_est_all"], "results": [(r.get("subtype"), r.get("num_turns"), r.get("total_cost_usd"))
                                                     for r in res],
           "est_at_results": [round(sum(usage_cost(u) for _m, u, i in msgs_u if i < j), 4)
                              for j, e in enumerate(ev) if e.get("type") == "result"],
           "first_call_input": first_in, "image_tokens_formula": image_tokens(sizes),
           "pages_arrived": first_in >= PAGE_ARRIVAL * image_tokens(sizes),
           "turns": task_turns(ev, "plain" if no_atlas else "atlas"),
           "store_files": store_files, "built": (ws / "dst" / "target" / "release" / "hello").exists(),
           "session": seg.get("cli_session") == st["session_id"]}
    (OUT / f"{arm}.json").write_text(json.dumps(row, indent=1))
    shutil.rmtree(store, ignore_errors=True)
    print(json.dumps(row, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "reference", "finish", "grade", "analyse"):
        sub.add_parser(name)
    s = sub.add_parser("smoke")
    s.add_argument("--guard", type=float, default=0.60)
    s.add_argument("--no-atlas", action="store_true", help="cheap guard check without the pages")
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "reference":
        reference()
    elif a.cmd == "smoke":
        smoke(a.guard, a.no_atlas)
    elif a.cmd == "run":
        run(a.arms)
    elif a.cmd == "finish":
        finish()
    elif a.cmd == "grade":
        grade()
    else:
        print(analyse())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
