"""Repo atlas, migration test: does the C project as page images help Claude Code migrate it to safe Rust?

Benchmark: CRUST-Bench (github.com/anirudhkhatry/CRUST-bench): 100 C repositories, each with hand-written Rust
interfaces (bodies ``unimplemented!()``) and Rust tests, graded by ``cargo test``. Pre-registration, arms, scores and
results: results/repo_atlas_migrate/README.md.

    python3 experiments/repo_atlas_migrate.py prepare      # clone, unpack, select, validate, atlases (no Claude calls)
    python3 experiments/repo_atlas_migrate.py run [--projects ...] [--workers 2]   # agent runs, resumable
    python3 experiments/repo_atlas_migrate.py report       # tables + paired tests -> summary.md

Arms (Claude Code headless, Opus 5.5, --effort high, tools Bash/Read/Edit/Write, 60 turns, 30 min, $5 per run):
``plain`` (C source + Rust skeleton), ``atlas`` (the same plus the C project as pages in the first message),
``atlas_only`` (Rust skeleton only; the pages are the only view of the C code).

Scratch (clones, work dirs, cargo targets, pages, transcripts) lives under MIG_WORK, never in the repo. The C sources
are kept there only as one XOR-scrambled tar (``store/cbench.bin``), so that an atlas_only agent cannot come across
them with find/grep; every transcript is audited for reads outside the work dir and for network fetches.
"""
from __future__ import annotations

import argparse
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "experiments" / "readhook")]
WORK = Path(os.environ.get("MIG_WORK", "/tmp/claude-0/-home-user-visual-bpe/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/mig"))
OUT = ROOT / "results" / "repo_atlas_migrate"
BENCH_URL = "https://github.com/anirudhkhatry/CRUST-bench.git"

MODEL, EFFORT = "claude-opus-5-5", "high"
ARMS = ("plain", "atlas", "atlas_only")
MAX_TURNS, WALL_S, RUN_BUDGET = 60, 1800, 5.0
CAP = 30.0                      # hard cap, list price, all Claude calls of this test
CAP_MARGIN = 0.5                # kept free: the CLI checks its budget between turns
LINES_MIN, LINES_MAX, FILES_MIN, N_PROJECTS = 800, 4000, 3, 10
PAGE_ARRIVAL = 0.97
CARGO_ENV = ("CARGO_HOME", "RUSTUP_HOME", "CARGO_HTTP_CAINFO")
XOR = 0x5A

# ----------------------------------------------------------------------------------------------- pure functions


def norm_name(name: str) -> str:
    """CBench and RBench name the same project differently (``c-aces`` / ``c_aces``, ``2DPartInt`` / ``proj_2DPartInt``)."""
    n = name.lower()
    n = n[5:] if n.startswith("proj_") else n
    return re.sub(r"[^a-z0-9]", "", n)


def c_stats(files: dict[str, str]) -> dict:
    """{relative path: text} of a C project -> lines and file count over .c and .h files (physical lines)."""
    c = {p: t for p, t in files.items() if p.endswith((".c", ".h"))}
    return {"c_files": len(c), "c_lines": sum(len(t.splitlines()) for t in c.values())}


def in_range(s: dict, lo: int = LINES_MIN, hi: int = LINES_MAX, min_files: int = FILES_MIN) -> bool:
    return lo <= s["c_lines"] <= hi and s["c_files"] >= min_files


def order_candidates(stats: list[dict], largest_first: bool = True) -> list[dict]:
    """The selection rule's order: candidates in range, by C lines (largest first unless the smaller-end amendment
    applies), ties by name."""
    c = [s for s in stats if in_range(s)]
    return sorted(c, key=lambda s: ((-s["c_lines"] if largest_first else s["c_lines"]), s["project"]))


TEST_LINE = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored)\b", re.M)
RUNNING = re.compile(r"Running (?:unittests )?\S*?src/bin/(\S+?)\.rs", re.M)


def parse_tests(log: str) -> dict:
    """``cargo test --no-fail-fast`` output -> per-test outcome, keyed ``<bin>::<test>`` (the same test name can appear
    in two test binaries). Also the benchmark's own counts (``... ok`` and ``... FAILED`` substrings)."""
    out, cur = {}, "?"
    events = sorted([(m.start(), "bin", m.group(1)) for m in RUNNING.finditer(log)]
                    + [(m.start(), "test", (m.group(1), m.group(2))) for m in TEST_LINE.finditer(log)])
    for _, kind, val in events:
        if kind == "bin":
            cur = val
        else:
            out[f"{cur}::{val[0]}"] = val[1]
    return {"tests": out, "bench_ok": log.count("... ok"), "bench_failed": log.count("... FAILED")}


def score_tests(outcomes: dict[str, str], baseline: list[str], built: bool) -> dict:
    """Score against the baseline list (tests that exist in the untouched skeleton, ignored ones left out). A test
    missing from the output (a crashed or timed-out binary) counts as failing."""
    passed = [t for t in baseline if outcomes.get(t) == "ok"]
    n = len(baseline)
    return {"built": built, "n_tests": n, "n_passed": len(passed) if built else 0,
            "share": (len(passed) / n if n else 0.0) if built else 0.0,
            "all_pass": bool(built and n and len(passed) == n)}


def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r'"(?:\\.|[^"\\])*"', '""', src)
    return re.sub(r"//[^\n]*", "", src)


def count_unsafe(src: str) -> int:
    """``unsafe`` blocks, functions, impls and traits in Rust source (comments and string literals ignored)."""
    return len(re.findall(r"\bunsafe\s*(?:\{|fn\b|impl\b|trait\b|extern\b)", strip_comments(src)))


def pub_signatures(src: str) -> list[str]:
    """Public items of an interface file, whitespace-normalised: ``pub fn`` headers up to the body or ``;``, and the
    names of public structs, enums, traits, types and consts. Used to check that the agent kept the interface."""
    s = strip_comments(src)
    sigs = []
    for m in re.finditer(r"\bpub(?:\([^)]*\))?\s+(?:const\s+|async\s+)?fn\s+[^{;]*", s):
        sigs.append("fn " + " ".join(m.group(0).split()).split(" fn ", 1)[-1].rstrip())
    for m in re.finditer(r"\bpub(?:\([^)]*\))?\s+(struct|enum|trait|type|const|static|mod)\s+([A-Za-z_][A-Za-z0-9_]*)", s):
        sigs.append(f"{m.group(1)} {m.group(2)}")
    return sorted(set(re.sub(r"\s*([(),:<>&\[\]])\s*", r"\1", x) for x in sigs))


def integrity(pristine: dict[str, bytes], final: dict[str, bytes]) -> dict:
    """{relative path: bytes} of the untouched skeleton and of the agent's crate -> what changed that must not.
    Tests (src/bin/) must be byte-identical and no test file may be added; every public signature of every
    pristine interface file must still be present in the same file."""
    tests_changed = sorted(p for p in pristine if p.startswith("src/bin/") and final.get(p) != pristine[p])
    tests_added = sorted(p for p in final if p.startswith("src/bin/") and p not in pristine)
    missing = {}
    for p, b in pristine.items():
        if p.startswith("src/") and not p.startswith("src/bin/") and p.endswith(".rs"):
            want = set(pub_signatures(b.decode("utf-8", "replace")))
            have = set(pub_signatures(final.get(p, b"").decode("utf-8", "replace")))
            if want - have:
                missing[p] = sorted(want - have)[:10]
    return {"tests_changed": tests_changed, "tests_added": tests_added, "signatures_missing": missing,
            "cargo_toml_changed": final.get("Cargo.toml") != pristine.get("Cargo.toml"),
            "tests_ok": not tests_changed and not tests_added, "interfaces_ok": not missing}


COMMON = """You are migrating a C library to safe Rust.

{source}

The Rust crate in `rust/` is the target. Its interface files (the `.rs` files under `rust/src/` outside `rust/src/bin/`) \
declare the types and functions to implement; bodies that are `unimplemented!()` are yours to write. The tests are in \
`rust/src/bin/`.

Implement the Rust interfaces as a faithful migration of the C library in safe Rust so that the provided tests pass \
(run `cargo test` in `rust/`). Do not edit the tests in `rust/src/bin/` and do not change the signatures of the \
interface functions and types; you may add private fields, helper functions and modules. Do not use `unsafe`. Do \
not add dependencies beyond those already in `rust/Cargo.toml`. cargo and rustc are installed. Work only with what is in this directory: do not look \
for this project, its C source or other ports of it elsewhere on the machine or online.

When all tests pass, or you can do no more, reply DONE."""

SOURCE = {
    "plain": "The C source is in `c_src/`.",
    "atlas": ("The C source is in `c_src/`. The page images above show the same C source: every .c and .h file of "
              "`c_src/` joined into one flow with a ▶path◀ header per file, whitespace minified, a line number (§n) on every "
              "10th line. Use them for orientation and read the files when exact text matters. Which page holds which "
              "file:\n\n{index}"),
    "atlas_only": ("The C source is not in this directory, and you cannot reach it: the page images above are your only "
                   "view of it. They show every .c and .h file of the C project joined into one flow with a ▶path◀ header per "
                   "file, whitespace minified, a line number (§n) on every 10th line. Which page holds which file:"
                   "\n\n{index}"),
}


def instruction(arm: str, index: str = "") -> str:
    return COMMON.format(source=SOURCE[arm].format(index=index.strip()))


def first_message(arm: str, pngs: list[bytes], index: str = "") -> dict:
    """The stream-json user message: the atlas pages (atlas arms only) followed by the instruction."""
    import base64
    if arm == "plain" and pngs:
        raise ValueError("the plain arm gets no pages")
    if arm != "plain" and not pngs:
        raise ValueError(f"the {arm} arm needs its pages")
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(b).decode("ascii")}} for b in pngs]
    content.append({"type": "text", "text": instruction(arm, index)})
    return {"type": "user", "message": {"role": "user", "content": content}}


def image_tokens(sizes: list[tuple[int, int]]) -> int:
    return sum(math.ceil(w / 28) * math.ceil(h / 28) for w, h in sizes)


NET = re.compile(r"\b(curl|wget|git\s+clone|git\s+fetch|pip\s+(?:install|download)|cargo\s+(?:install|search|add))\b|"
                 r"https?://|github\.com|crates\.io", re.I)
ALLOWED_OUTSIDE = ("/root/.cargo", "/root/.rustup", "/usr/", "/bin/", "/lib", "/etc/", "/dev/null", "/proc/self",
                   "/tmp/")


def audit_calls(calls: list[dict], cwd: str) -> dict:
    """Tool calls of one run -> anything that reaches past the work dir: absolute paths outside it (toolchain and
    system dirs and /tmp scratch files allowed, but not the scratch tree under /tmp/claude*), filesystem-wide searches
    and network commands. Any of these marks the run ``suspect``, to be read by hand."""
    outside, net, searches = [], [], []
    for c in calls:
        inp = c.get("input") or {}
        text = " ".join(str(v) for v in inp.values() if isinstance(v, (str, int)))
        if NET.search(text):
            net.append(text[:200])
        own = str(cli_store(Path(cwd)))
        for p in re.findall(r"(?<![\w.~/:-])(/[A-Za-z_][\w./-]*)", text):
            if p.startswith(cwd) or p.startswith(own):
                continue
            if (not p.startswith(ALLOWED_OUTSIDE) or p.startswith("/tmp/claude")) and p != "/":
                outside.append(p[:120])
        if c.get("name") == "Bash" and re.search(r"\b(find|locate|grep\s+-r|rg|ls)\s+(-\S+\s+)*(/|~|\$HOME|/tmp|/root|/home)(\s|$)",
                                                  str(inp.get("command", ""))):
            searches.append(str(inp.get("command"))[:200])
    return {"outside_paths": sorted(set(outside))[:20], "network": net[:10], "fs_searches": searches[:10],
            "suspect": bool(net or searches or outside)}


def run_cost(usages: list[dict], price: dict | None = None) -> float:
    """List price of a run from its per-message usage (used only when a run is killed before its result event).
    Opus 5.5 prices fitted exactly on earlier rows ($ per million: input 4, cache write 5, cache read 0.2, output 20)."""
    pr = price or {"input_tokens": 4.0, "cache_creation_input_tokens": 5.0, "cache_read_input_tokens": 0.2,
                   "output_tokens": 20.0}
    return sum(u.get(k, 0) * v for u in usages for k, v in pr.items()) / 1e6


def exact_binom_two_sided(k: int, n: int) -> float:
    """Two-sided exact binomial test at p = 0.5 (McNemar on the discordant pairs; the sign test)."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def sign_test(diffs: list[float]) -> dict:
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return {"pos": pos, "neg": neg, "ties": len(diffs) - pos - neg, "p": exact_binom_two_sided(pos, pos + neg)}


def median(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


# ----------------------------------------------------------------------------------------------- scratch helpers


def sh(cmd, cwd=None, timeout=1800, env=None, check=False):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env,
                       shell=isinstance(cmd, str), stdin=subprocess.DEVNULL)
    if check and p.returncode:
        raise RuntimeError(f"{cmd}: {p.stderr[-800:]}")
    return p


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


_lock = threading.Lock()


def append(name: str, row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with _lock, open(OUT / name, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_tree(d: Path, skip=(".git", "target")) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*"))
            if p.is_file() and not any(s in p.relative_to(d).parts for s in skip)}


_XOR_TABLE = bytes(i ^ XOR for i in range(256))


def xor(b: bytes) -> bytes:
    return b.translate(_XOR_TABLE)


def store_cbench(cbench: Path, dest: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(cbench.iterdir()):
            tf.add(p, arcname=p.name, filter=lambda ti: None if "/.git" in "/" + ti.name or ti.name.endswith("/.git") else ti)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(xor(buf.getvalue()))


def extract_c(project: str, dest: Path) -> Path:
    """Unpack one C project from the scrambled store into ``dest``."""
    data = xor((WORK / "store" / "cbench.bin").read_bytes())
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name == project or m.name.startswith(project + "/")]
        for m in members:
            m.name = m.name[len(project):].lstrip("/") or "."
        tf.extractall(dest, members=[m for m in members if m.name != "."], filter="data")
    return dest


def text_files(d: Path) -> dict[str, str]:
    out = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and ".git" not in p.relative_to(d).parts:
            try:
                out[str(p.relative_to(d))] = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                pass
    return out


def format_skeleton(rproj: Path, dest: Path) -> None:
    """The benchmark's own layout step (src/check_benchmarks/check_build.py): interface files move from
    src/interfaces/ to src/. .git and target/ are dropped."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(rproj, dest, ignore=shutil.ignore_patterns(".git", "target"))
    inter = dest / "src" / "interfaces"
    if inter.exists():
        for f in inter.iterdir():
            if f.is_file() and f.suffix == ".rs":
                shutil.move(str(f), dest / "src" / f.name)
        shutil.rmtree(inter)


def cargo_env(target: Path) -> dict:
    env = dict(os.environ, CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="2", RUSTFLAGS="-Awarnings",
               CARGO_TERM_COLOR="never", LANG="C.UTF-8")
    return env


BIN_TIMEOUT = 300           # per test binary (amendment 1: one slow binary no longer hides the others)


def run_group(cmd: list[str], cwd: Path, env: dict, timeout: int) -> tuple[str, str, bool, int]:
    """Run a command in its own process group; on timeout kill the whole group (cargo and the test binary) and
    return what it printed so far."""
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), False, p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), True, -9


def parse_bin(stdout: str, name: str) -> dict[str, str]:
    """One test binary's stdout -> {"<bin>::<test>": ok | FAILED | ignored}."""
    return {f"{name}::{m.group(1)}": m.group(2) for m in TEST_LINE.finditer(stdout)}


def cargo_grade(crate: Path, target: Path, bin_timeout: int = BIN_TIMEOUT) -> dict:
    """Build the tests, then run each test target on its own (the library's unit tests as ``lib``, then every
    src/bin/*.rs) with its own time limit. Returns built, outcomes keyed ``<bin>::<test>`` and the timed-out targets."""
    env = cargo_env(target)
    t0 = time.time()
    out, err, to, rc = run_group(["cargo", "test", "--no-run"], crate, env, 1200)
    built = rc == 0
    res = {"built": built, "build_s": round(time.time() - t0), "build_log_tail": "" if built else (err[-3000:] or "BUILD TIMEOUT")}
    if not built:
        return {**res, "outcomes": {}, "bench_ok": 0, "bench_failed": 0, "timeout": False, "timed_out_bins": []}
    t1 = time.time()
    outcomes, ok, failed, tbins = {}, 0, 0, []
    targets = [("lib", ["--lib"])] + [(b.stem, ["--bin", b.stem]) for b in sorted((crate / "src" / "bin").glob("*.rs"))]
    for name, sel in targets:
        out, _err, to, _rc = run_group(["cargo", "test", "--no-fail-fast", *sel], crate, env, bin_timeout)
        outcomes.update(parse_bin(out, name))
        ok += out.count("... ok")
        failed += out.count("... FAILED")
        if to:
            tbins.append(name)
    return {**res, "outcomes": outcomes, "bench_ok": ok, "bench_failed": failed, "timeout": bool(tbins),
            "timed_out_bins": tbins, "test_s": round(time.time() - t1)}


# ----------------------------------------------------------------------------------------------- prepare


def prepare() -> None:
    bench = WORK / "bench"
    if not (WORK / "store" / "cbench.bin").exists():
        if not (bench / "CRUST-bench").exists():
            bench.mkdir(parents=True, exist_ok=True)
            sh(["git", "clone", "-q", BENCH_URL, str(bench / "CRUST-bench")], check=True, timeout=3600)
        head = sh(["git", "-C", str(bench / "CRUST-bench"), "rev-parse", "HEAD"]).stdout.strip()
        data = bench / "data"
        if not (data / "CBench").exists():
            data.mkdir(parents=True, exist_ok=True)
            sh(["unzip", "-q", str(bench / "CRUST-bench" / "datasets" / "CRUST_bench.zip"), "-d", str(data)], check=True)
        rb = WORK / "rbench"
        if not rb.exists():
            rb.mkdir(parents=True)
            for p in sorted((data / "RBench").iterdir()):
                if p.is_dir():
                    shutil.copytree(p, rb / p.name, ignore=shutil.ignore_patterns(".git", "target"))
        stats = []
        cb = {norm_name(p.name): p for p in (data / "CBench").iterdir() if p.is_dir()}
        for p in sorted(rb.iterdir()):
            c = cb.get(norm_name(p.name))
            if c is None:
                stats.append({"project": p.name, "c_project": None})
                continue
            s = c_stats(text_files(c))
            tests = sum(len(re.findall(r"#\[test\]", f.read_text(errors="replace"))) for f in (p / "src" / "bin").glob("*.rs"))
            stats.append({"project": p.name, "c_project": c.name, **s, "test_fns": tests})
        (OUT / "dataset.json").write_text(json.dumps({"source": BENCH_URL, "commit": head, "projects": stats}, indent=1))
        store_cbench(data / "CBench", WORK / "store" / "cbench.bin")
    stats = json.loads((OUT / "dataset.json").read_text())["projects"]
    matched = [s for s in stats if s.get("c_project")]
    cand = order_candidates(matched)
    vdone = {r["project"] for r in jl(OUT / "validate.jsonl")}
    for s in cand:
        if s["project"] in vdone:
            continue
        r = validate(s)
        append("validate.jsonl", r)
        print(f"[validate] {s['project']:40} lines={s['c_lines']:5} files={s['c_files']:3} valid={r['valid']} "
              f"tests={r.get('n_tests')} {r.get('reason', '')}", flush=True)
    vrows = {r["project"]: r for r in jl(OUT / "validate.jsonl")}
    for s in cand:
        if vrows[s["project"]]["valid"] and not (WORK / "atlas" / s["project"] / "index.json").exists():
            m = build_atlas(s["project"], s["c_project"])
            print(f"[atlas] {s['project']:40} {len(m['pages'])} pages {m['image_tokens']:,} image tokens "
                  f"uncovered={m['uncovered']}", flush=True)
    write_selection(stats, cand, vrows)
    if bench.exists():
        shutil.rmtree(bench)                 # the plain C sources go: only the scrambled store remains


def validate(s: dict) -> dict:
    """The harness works locally: the untouched skeleton's tests compile and they fail (not all pass)."""
    sk = WORK / "pristine" / s["project"]
    if not (sk / "Cargo.toml").exists():
        format_skeleton(WORK / "rbench" / s["project"], sk)
    sk_copy = WORK / "tmp_validate" / s["project"]           # cargo may write into the crate (Cargo.lock)
    shutil.rmtree(sk_copy, ignore_errors=True)
    shutil.copytree(sk, sk_copy)
    sk = sk_copy
    tgt = WORK / "targets" / f"validate_{s['project']}"
    g = cargo_grade(sk, tgt)
    shutil.rmtree(tgt, ignore_errors=True)
    shutil.rmtree(sk, ignore_errors=True)
    base = sorted(t for t, o in g["outcomes"].items() if o != "ignored")
    r = {"project": s["project"], "built": g["built"], "n_tests": len(base), "baseline": base,
         "baseline_passing": sorted(t for t in base if g["outcomes"][t] == "ok"),
         "ignored": sorted(t for t, o in g["outcomes"].items() if o == "ignored"), "build_s": g["build_s"]}
    if not g["built"]:
        r.update(valid=False, reason="tests do not compile", log=g["build_log_tail"][-600:])
    elif not base:
        r.update(valid=False, reason="no tests run")
    elif len(r["baseline_passing"]) == len(base):
        r.update(valid=False, reason="all tests pass before implementation")
    elif g["timeout"]:
        r.update(valid=False, reason="test run timed out")
    else:
        r["valid"] = True
    return r


def build_atlas(project: str, c_project: str) -> dict:
    """The C source as pages, with repo_atlas's minifiers and renderer: every .c and .h file of the project, whatever
    its directory (tests and examples included: the plain arm sees them too). Other files (Makefiles, READMEs, data)
    are listed in ``skipped``."""
    import repo_atlas as RA
    from vbhook import render
    from vbhook.minify import minify
    tmp = WORK / "tmp_c" / project
    shutil.rmtree(tmp, ignore_errors=True)
    extract_c(c_project, tmp)
    files, uncovered, skipped = [], [], []
    for path, text in text_files(tmp).items():
        if not path.endswith((".c", ".h")) or not text.strip():
            skipped.append(path)
            continue
        got = minify(text, path)
        if got is None:
            uncovered.append(path)
            continue
        flow, m = got
        files.append({"path": path, "chars": len(text), "flow": RA.thin_tags(flow, 10), "kind": m.kind,
                      "text_tokens_est": len(text) / m.cpt})
    shutil.rmtree(tmp, ignore_errors=True)
    out = WORK / "atlas" / project
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    flow = RA.atlas_flow(files)
    pages, n = [], 0
    for chunk in render.chunk(flow, render.page_chars(88.2)):
        starts = re.findall(r"▶([^◀]+)◀", chunk)
        for pg in render.render(chunk, 88.2):
            n += 1
            pg.image.save(out / f"page_{n:03d}.png")
            pages.append({"page": n, "tokens": pg.tokens, "size": list(pg.image.size), "starts": starts})
    index = []
    for p in pages:
        for path in p["starts"]:
            if index:
                index[-1]["pages"][1] = max(index[-1]["pages"][1], p["page"])
            index.append({"path": path, "pages": [p["page"], p["page"]]})
    if index:
        index[-1]["pages"][1] = pages[-1]["page"]
    meta = {"project": project, "files": [f["path"] for f in files], "uncovered": uncovered, "skipped": skipped,
            "source_chars": sum(f["chars"] for f in files), "flow_chars": len(flow),
            "image_tokens": sum(p["tokens"] for p in pages), "formula_tokens": image_tokens([tuple(p["size"]) for p in pages]),
            "text_tokens_est": round(sum(f["text_tokens_est"] for f in files)), "pages": pages, "index": index}
    (out / "index.json").write_text(json.dumps(meta, indent=1))
    (out / "index.txt").write_text("\n".join(
        f"{e['path']}: page {e['pages'][0]}" + (f"-{e['pages'][1]}" if e["pages"][1] != e["pages"][0] else "")
        for e in index) + "\n")
    return meta


def write_selection(stats: list[dict], cand: list[dict], vrows: dict) -> dict:
    matched = [s for s in stats if s.get("c_project")]
    valid = [s["project"] for s in cand if vrows.get(s["project"], {}).get("valid")]
    atl = {}
    for p in valid:
        f = WORK / "atlas" / p / "index.json"
        if f.exists():
            m = json.loads(f.read_text())
            atl[p] = {"pages": len(m["pages"]), "image_tokens": m["image_tokens"], "text_tokens_est": m["text_tokens_est"],
                      "atlas_files": len(m["files"]), "uncovered": m["uncovered"], "skipped": m["skipped"]}
    funnel = {"projects": len(stats), "c_and_rust_matched": len(matched),
              "lines_800_4000": sum(LINES_MIN <= s["c_lines"] <= LINES_MAX for s in matched),
              "and_3plus_c_files": len(cand), "validated": len(valid),
              "invalid": {s["project"]: vrows[s["project"]].get("reason") for s in cand
                          if s["project"] in vrows and not vrows[s["project"]]["valid"]}}
    sel = {"rule": f"C lines (.c+.h, physical) in [{LINES_MIN}, {LINES_MAX}], at least {FILES_MIN} .c/.h files, the "
                   "untouched skeleton's tests compile and fail; ordered by C lines, largest first; first "
                   f"{N_PROJECTS} that validate",
           "funnel": funnel, "selected": valid[:N_PROJECTS],
           "smaller_end_order": [s["project"] for s in order_candidates(matched, largest_first=False)
                                 if vrows.get(s["project"], {}).get("valid")],
           "meta": {s["project"]: {**{k: s[k] for k in ("c_project", "c_lines", "c_files", "test_fns")},
                                   "n_tests": vrows[s["project"]]["n_tests"], **atl.get(s["project"], {})}
                    for s in cand if vrows.get(s["project"], {}).get("valid")}}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel


# ----------------------------------------------------------------------------------------------- agent runs


def cli_store(cwd: Path) -> Path:
    """Where the CLI keeps a cwd's auto-memory and spilled tool outputs (they can hold C source text): deleted
    after every run so that no later run (an atlas_only run above all) can come across them."""
    return Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def baselines() -> dict:
    """Per-project skeleton test lists: validate_v2.jsonl (tests keyed by binary, amendment 1) if present."""
    v2 = OUT / "validate_v2.jsonl"
    return {r["project"]: r for r in jl(v2 if v2.exists() else OUT / "validate.jsonl")}


def rebaseline(projects: list[str]) -> None:
    done = {r["project"] for r in jl(OUT / "validate_v2.jsonl")}
    for p in projects:
        if p in done:
            continue
        r = validate({"project": p})
        append("validate_v2.jsonl", r)
        print(f"[rebaseline] {p:40} valid={r['valid']} tests={r['n_tests']} {r.get('reason', '')}", flush=True)


def regrade() -> None:
    """Amendment 1: re-grade every row graded with grade v1 from the crate the agent left (finals/), with the
    per-binary grader and the per-binary baselines; also re-run the audit with the refined path rule."""
    rows = jl(OUT / "runs.jsonl")
    base = baselines()
    for r in rows:
        if r.get("grade_version") == 2:
            continue
        crate = WORK / "regrade" / r["project"] / r["arm"]
        shutil.rmtree(crate, ignore_errors=True)
        shutil.copytree(WORK / "finals" / r["project"] / r["arm"], crate)
        pristine = read_tree(WORK / "pristine" / r["project"])
        integ = integrity(pristine, read_tree(crate))
        for rel in integ["tests_changed"]:
            (crate / rel).write_bytes(pristine[rel])
        for rel in integ["tests_added"]:
            (crate / rel).unlink()
        tgt = WORK / "targets" / f"regrade_{r['project']}_{r['arm']}"
        g = cargo_grade(crate, tgt)
        shutil.rmtree(tgt, ignore_errors=True)
        shutil.rmtree(crate, ignore_errors=True)
        r["grade_v1"] = {k: r[k] for k in ("built", "n_tests", "n_passed", "share", "all_pass", "bench_ok",
                                           "bench_failed", "test_timeout")}
        r.update(score_tests(g["outcomes"], base[r["project"]]["baseline"], g["built"]))
        r.update(bench_ok=g["bench_ok"], bench_failed=g["bench_failed"], test_timeout=g["timeout"],
                 timed_out_bins=g.get("timed_out_bins", []), grade_version=2)
        print(f"[regrade] {r['project']:28} {r['arm']:10} v1 {r['grade_v1']['n_passed']}/{r['grade_v1']['n_tests']} -> "
              f"{r['n_passed']}/{r['n_tests']} all={r['all_pass']} timed out: {r['timed_out_bins']}", flush=True)
    for r in rows:
        stream = WORK / "streams" / f"{r['project']}__{r['arm']}.jsonl"
        if stream.exists():
            ev = [json.loads(l) for l in stream.read_text().splitlines() if l.startswith("{")]
            calls = [c for e in ev if e.get("type") == "assistant" for c in e["message"]["content"] if c.get("type") == "tool_use"]
            r["audit"] = audit_calls(calls, str(WORK / "runs" / r["project"] / r["arm"]))
            rsum = result_summary(ev)
            if rsum and "agent_s" not in r:
                u = rsum["usage"]
                r.update(turns=rsum["turns"], agent_s=rsum["agent_s"], n_results=rsum["n_results"], wall_s=r["seconds"],
                         final=rsum["final"], subtype=rsum["subtype"], output_tokens=u["output_tokens"],
                         input_tokens=sum(u[k] for k in USAGE_KEYS[:3]), usage=u)
    tmp = OUT / "runs.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, OUT / "runs.jsonl")


def spent() -> float:
    return sum(r.get("cost_usd") or 0 for r in jl(OUT / "spend.jsonl"))


class UsageLimit(RuntimeError):
    """The subscription's usage limit (or an API outage) ended the run: not a result, not retried in a loop."""


LIMIT = re.compile(r"usage limit|limit reached|hit your limit|resets? at|rate.?limit|overloaded|api_error|\b429\b|\b529\b", re.I)


def limit_hit(res: dict | None, stderr: str, n_messages: int, timed_out: bool) -> bool:
    """A run that ended on a usage limit or an API error rather than on the agent's own terms."""
    if timed_out:
        return False
    if res is None:
        return n_messages == 0 or bool(LIMIT.search(stderr))
    if res.get("subtype") in ("error_max_turns", "error_max_budget_usd"):
        return False
    return bool(res.get("is_error")) and bool(LIMIT.search(str(res.get("result")) + " " + stderr
                                                           + " " + str(res.get("api_error_status"))))


def result_summary(ev: list[dict]) -> dict:
    """A stream can hold several result events: background tasks the agent started wake it again after its answer,
    and each wake ends with its own result. total_cost_usd is cumulative (the last one is the run's), turns and
    durations are per segment (summed); the answer is the first result's text."""
    rs = [e for e in ev if e.get("type") == "result"]
    if not rs:
        return {}
    return {"turns": sum(r.get("num_turns") or 0 for r in rs), "cost_usd": float(rs[-1].get("total_cost_usd") or 0.0),
            "agent_s": round(sum(r.get("duration_ms") or 0 for r in rs) / 1000), "n_results": len(rs),
            "subtype": rs[0].get("subtype"), "final": (rs[0].get("result") or "")[-400:],
            "usage": {k: sum((r.get("usage") or {}).get(k, 0) for r in rs) for k in USAGE_KEYS}}


USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def run_one(project: str, arm: str, budget: float) -> dict:
    from vbpe import claude_eval as CE
    sel = json.loads((OUT / "selection.json").read_text())
    meta = sel["meta"][project]
    vrow = baselines()[project]
    base = WORK / "runs" / project / arm
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    crate = base / "rust"
    shutil.copytree(WORK / "pristine" / project, crate)
    if arm != "atlas_only":
        extract_c(meta["c_project"], base / "c_src")
    pngs, index, sizes = [], "", []
    if arm != "plain":
        ad = WORK / "atlas" / project
        am = json.loads((ad / "index.json").read_text())
        pngs = [(ad / f"page_{p['page']:03d}.png").read_bytes() for p in am["pages"]]
        sizes = [tuple(p["size"]) for p in am["pages"]]
        index = (ad / "index.txt").read_text()
    msg = first_message(arm, pngs, index)
    cmd = [CE.CLAUDE_BIN, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", MODEL, "--effort", EFFORT, "--tools", "Bash,Read,Edit,Write",
           "--allowedTools", "Bash,Read,Edit,Write",
           "--disallowedTools", "Bash(curl:*)", "Bash(wget:*)", "Bash(git clone:*)", "WebFetch", "WebSearch",
           "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
           "--max-turns", str(MAX_TURNS), "--max-budget-usd", f"{budget:.2f}"]
    env = CE.build_env()
    env.update({k: os.environ[k] for k in CARGO_ENV if k in os.environ})
    env.update(CARGO_BUILD_JOBS="2")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=base, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = p.communicate((json.dumps(msg) + "\n").encode(), timeout=WALL_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(p.pid, signal.SIGKILL)
        stdout, stderr = p.communicate()
    secs = time.time() - t0
    stdout = stdout.decode("utf-8", "replace")
    sdir = WORK / "streams"
    sdir.mkdir(exist_ok=True)
    (sdir / f"{project}__{arm}.jsonl").write_text(stdout)
    ev = []
    for l in stdout.splitlines():
        if l.strip().startswith("{"):
            try:
                ev.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    res = next((e for e in reversed(ev) if e.get("type") == "result"), None)
    asst = [e for e in ev if e.get("type") == "assistant"]
    usages, seen = [], set()
    for e in asst:
        mid = e["message"].get("id")
        if mid not in seen and e["message"].get("usage"):
            seen.add(mid)
            usages.append(e["message"]["usage"])
    first = usages[0] if usages else {}
    first_in = sum(first.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    calls = [c for e in asst for c in e["message"]["content"] if c.get("type") == "tool_use"]
    names = [c["name"] for c in calls]
    if res is not None:
        cost, cost_est = float(res.get("total_cost_usd") or 0.0), False
        u = result_summary(ev)["usage"]
    else:
        cost, cost_est = run_cost(usages), True
        u = {k: sum(x.get(k, 0) for x in usages) for k in ("input_tokens", "cache_creation_input_tokens",
                                                           "cache_read_input_tokens", "output_tokens")}
    append("spend.jsonl", {"what": f"{project} {arm}", "cost_usd": cost, "estimated": cost_est})
    err = stderr.decode("utf-8", "replace")
    if limit_hit(res, err, len(usages), timed_out):
        shutil.rmtree(base, ignore_errors=True)
        raise UsageLimit(f"{project} {arm}: {str((res or {}).get('result'))[:300]} {err[-300:]}")
    # ---- integrity, then grading on the work dir as the agent left it (tests restored if touched)
    pristine = read_tree(WORK / "pristine" / project)
    final = read_tree(crate)
    integ = integrity(pristine, final)
    keep = WORK / "finals" / project / arm
    shutil.rmtree(keep, ignore_errors=True)
    shutil.copytree(crate, keep, ignore=shutil.ignore_patterns("target", ".git"))
    if not integ["tests_ok"]:
        for rel in integ["tests_changed"]:
            (crate / rel).parent.mkdir(parents=True, exist_ok=True)
            (crate / rel).write_bytes(pristine[rel])
        for rel in integ["tests_added"]:
            (crate / rel).unlink()
    shutil.rmtree(crate / "target", ignore_errors=True)
    shutil.rmtree(base / "c_src", ignore_errors=True)
    tgt = WORK / "targets" / f"grade_{project}_{arm}"
    g = cargo_grade(crate, tgt)
    shutil.rmtree(tgt, ignore_errors=True)
    sc = score_tests(g["outcomes"], vrow["baseline"], g["built"])
    lib_src = "\n".join(v.decode("utf-8", "replace") for k, v in read_tree(crate).items()
                        if k.startswith("src/") and not k.startswith("src/bin/") and k.endswith(".rs"))
    audit = audit_calls(calls, str(base))
    rsum = result_summary(ev)
    row = {"project": project, "arm": arm, "model": MODEL, "effort": EFFORT, **sc,
           "bench_ok": g["bench_ok"], "bench_failed": g["bench_failed"], "test_timeout": g["timeout"],
           "build_log_tail": g["build_log_tail"][-800:],
           "unsafe": count_unsafe(lib_src), "unimplemented_left": len(re.findall(r"unimplemented!\(", strip_comments(lib_src))),
           "turns": rsum.get("turns"), "subtype": rsum.get("subtype", "killed_wall_clock" if timed_out else "no_result"),
           "agent_s": rsum.get("agent_s", round(secs)), "n_results": rsum.get("n_results", 0), "wall_s": round(secs),
           "is_error": (res or {}).get("is_error"), "cost_usd": cost, "cost_estimated": cost_est, "budget_usd": budget,
           "seconds": round(secs), "timed_out": timed_out, "output_tokens": u.get("output_tokens"),
           "input_tokens": sum(u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
           "usage": {k: u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")},
           "assistant_messages": len(usages), "tools": {k: names.count(k) for k in sorted(set(names))},
           "first_turn_input": first_in, "image_tokens_formula": image_tokens(sizes),
           "pages_arrived": (arm == "plain") or first_in >= PAGE_ARRIVAL * image_tokens(sizes),
           "integrity": integ, "audit": audit, "permission_denials": [str(d)[:200] for d in (res or {}).get("permission_denials") or []][:10],
           "final": rsum.get("final", ""), "stderr_tail": stderr.decode("utf-8", "replace")[-400:]}
    row["grade_version"] = 2
    row["timed_out_bins"] = g.get("timed_out_bins", [])
    shutil.rmtree(base / "rust", ignore_errors=True)
    shutil.rmtree(cli_store(base), ignore_errors=True)
    return row


def run(projects: list[str] | None, workers: int) -> None:
    sel = json.loads((OUT / "selection.json").read_text())
    order = projects or sel["selected"]
    done = {(r["project"], r["arm"]) for r in jl(OUT / "runs.jsonl")}
    jobs = [(p, a) for p in order for a in ARMS if (p, a) not in done]
    print(f"[mig] {len(jobs)} runs to do; spent ${spent():.2f} of ${CAP}", flush=True)
    state = {"running": {}, "lock": threading.Condition()}

    def started(p):
        return any(q == p for (q, _a) in done) or any(q == p for (q, _a) in state["running"])

    def est_run() -> float:
        costs = [r["cost_usd"] for r in jl(OUT / "runs.jsonl") if r.get("cost_usd")]
        return max(1.0, sum(costs) / len(costs) * 1.2) if costs else RUN_BUDGET

    def committed() -> float:
        return spent() + sum(state["running"].values())

    def pick():
        """Next eligible job in order: an atlas_only run waits until the same project's plain and atlas runs are
        over (their work dirs hold the C source); a new project starts only if its three runs fit the cap at the
        current estimate, after the runs of already started projects."""
        pending_started = [(p, a) for (p, a) in jobs if started(p) and (p, a) not in state["running"]]
        for (p, a) in jobs:
            if (p, a) in state["running"]:
                continue
            if a == "atlas_only" and any((p, b) in state["running"] or (p, b) in jobs for b in ("plain", "atlas")):
                continue
            if not started(p):
                need = 3 * est_run() + sum(min(RUN_BUDGET, est_run()) for _ in pending_started)
                if committed() + need > CAP - CAP_MARGIN:
                    continue
                return (p, a)
            return (p, a)
        return None

    def worker():
        while True:
            with state["lock"]:
                while True:
                    if not jobs:
                        return
                    j = pick()
                    if j:
                        budget = min(RUN_BUDGET, CAP - CAP_MARGIN - committed())
                        if budget < 0.5:
                            j = None
                    if j:
                        break
                    if not state["running"]:
                        print("[mig] nothing more fits the cap; stopping", flush=True)
                        jobs.clear()
                        return
                    state["lock"].wait(30)
                state["running"][j] = budget
            print(f"[mig] start {j[0]} {j[1]} (budget ${budget:.2f}, committed ${committed():.2f})", flush=True)
            try:
                row = run_one(j[0], j[1], budget)
                append("runs.jsonl", row)
                print(f"[mig] done  {j[0]:28} {j[1]:10} all={row['all_pass']!s:5} {row['n_passed']}/{row['n_tests']} "
                      f"built={row['built']} turns={row['turns']} ${row['cost_usd']:.2f} {row['seconds']}s "
                      f"pages_ok={row['pages_arrived']} tests_ok={row['integrity']['tests_ok']} "
                      f"iface_ok={row['integrity']['interfaces_ok']} suspect={row['audit']['suspect']} {row['subtype']}",
                      flush=True)
                ok = True
            except UsageLimit as e:
                print(f"[mig] USAGE LIMIT, not recorded, stopping: {e}", flush=True)
                ok = False
                with state["lock"]:
                    jobs.clear()
            except Exception as e:  # noqa: BLE001
                print(f"[mig] ERROR {j}: {e!r}", flush=True)
                ok = False
            with state["lock"]:
                state["running"].pop(j, None)
                if ok:
                    done.add(j)
                if j in jobs and ok:
                    jobs.remove(j)
                elif j in jobs:
                    jobs.remove(j)              # not recorded: retried on the next start
                state["lock"].notify_all()

    ts = [threading.Thread(target=worker) for _ in range(workers)]
    for t in ts:
        t.start()
        time.sleep(5)
    for t in ts:
        t.join()
    print(f"[mig] finished; spent ${spent():.2f}", flush=True)


# ----------------------------------------------------------------------------------------------- report


def report() -> str:
    rows = {(r["project"], r["arm"]): r for r in jl(OUT / "runs.jsonl")}
    projects = [p for p in dict.fromkeys(r[0] for r in rows) if all((p, a) in rows for a in ARMS)]
    sel = json.loads((OUT / "selection.json").read_text())
    L = [f"*{len(projects)} projects with all three arms; spend ${spent():.2f} at list price (cap ${CAP}).*", ""]
    L += ["| Arm | Projects | All tests pass | Mean share passing | Builds | Median turns | Mean cost | Median seconds | "
          "Median output tokens | unsafe (total) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        rs = [rows[(p, a)] for p in projects]
        if not rs:
            continue
        L.append(f"| {a} | {len(rs)} | {sum(r['all_pass'] for r in rs)} | {100 * sum(r['share'] for r in rs) / len(rs):.1f}% | "
                 f"{sum(r['built'] for r in rs)} | {median([r['turns'] for r in rs])} | "
                 f"${sum(r['cost_usd'] for r in rs) / len(rs):.2f} | {median([r.get('agent_s', r['seconds']) for r in rs])} | "
                 f"{median([r['output_tokens'] for r in rs])} | {sum(r['unsafe'] for r in rs)} |")
    L += ["", "| Project | C lines | Tests | " + " | ".join(f"{a}: pass / turns / $ / s" for a in ARMS) + " |",
          "|---|---|---|" + "---|" * len(ARMS)]
    for p in projects:
        m = sel["meta"][p]
        cells = []
        for a in ARMS:
            r = rows[(p, a)]
            flag = "" if r["built"] else " (no build)"
            flag += "" if r["integrity"]["tests_ok"] else " (tests restored)"
            flag += "" if r["integrity"]["interfaces_ok"] else " (signatures changed)"
            flag += " (SUSPECT)" if r["audit"]["suspect"] else ""
            flag += "" if r["pages_arrived"] else " (PAGES DROPPED)"
            cells.append(f"{r['n_passed']}/{r['n_tests']}{' ALL' if r['all_pass'] else ''}{flag} / {r['turns']} / "
                         f"{r['cost_usd']:.2f} / {r.get('agent_s', r['seconds'])}")
        L.append(f"| {p} | {m['c_lines']} | {m['n_tests']} | " + " | ".join(cells) + " |")
    L += ["", "Paired tests (exact; McNemar on all-tests-pass, sign tests on the rest):", ""]
    for a, b in (("atlas", "plain"), ("atlas_only", "plain"), ("atlas_only", "atlas")):
        if not projects:
            break
        A, B = [rows[(p, a)] for p in projects], [rows[(p, b)] for p in projects]
        ab = sum(x["all_pass"] and not y["all_pass"] for x, y in zip(A, B))
        ba = sum(y["all_pass"] and not x["all_pass"] for x, y in zip(A, B))
        parts = [f"all-pass {a} only {ab}, {b} only {ba}, p = {exact_binom_two_sided(ab, ab + ba):.3f}"]
        for k in ("share", "turns", "cost_usd", "agent_s", "output_tokens"):
            d = [(x.get(k, x.get("seconds")) or 0) - (y.get(k, y.get("seconds")) or 0) for x, y in zip(A, B)]
            st = sign_test(d)
            parts.append(f"{k}: median diff {median(d):+.3g} ({st['pos']}+ / {st['neg']}- / {st['ties']}=, p = {st['p']:.3f})")
        L.append(f"* **{a} vs {b}:** " + "; ".join(parts))
    s = "\n".join(L) + "\n"
    (OUT / "summary.md").write_text(s)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["prepare", "run", "report", "rebaseline", "regrade"])
    ap.add_argument("--projects", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "rebaseline":
        rebaseline(a.projects or json.loads((OUT / "selection.json").read_text())["selected"])
    elif a.cmd == "regrade":
        regrade()
    elif a.cmd == "run":
        run(a.projects, min(2, a.workers))
    else:
        print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
