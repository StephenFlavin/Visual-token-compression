"""Repo atlas against SWE-QA: does the whole repository as page images help answer questions about it?

Benchmark: SWE-QA (Peng et al., arXiv 2509.14635; data github.com/peng-weihan/SWE-QA-Bench, HF swe-qa/SWE-QA-Benchmark):
48 questions per Python repository with reference answers, pinned commits in ``repo_commit.txt``, an LLM-judge
rubric (5 criteria x 1-20). Design, predictions and results: results/atlas/qa/README.md.

Arms (Opus 5.5, high effort): closed (no context), map (repo map), atlas (pages + index), text (source as text),
agent (Claude Code, read-only tools, one question per run). Arms 1-4 answer 8 questions per call; every call is two
user messages: the context (answered "OK"), then the questions, so later calls on one repository read the context
from the prompt cache (a single message is never reused: measured, see the README).

    python3 experiments/atlas/repo_atlas_qa.py prepare  --work DIR --bench DIR   # clones, contexts, batches (no model calls)
    python3 experiments/atlas/repo_atlas_qa.py estimate --work DIR
    python3 experiments/atlas/repo_atlas_qa.py run      --work DIR --arms closed map atlas text --cap 22
    python3 experiments/atlas/repo_atlas_qa.py run      --work DIR --arms agent --cap 22 [--agent-n N]
    python3 experiments/atlas/repo_atlas_qa.py judge    --work DIR --cap 22
    python3 experiments/atlas/repo_atlas_qa.py analyse                           # prints the tables (kept as summary.md), writes summary.json

Clones, page images and agent transcripts stay in --work; rows, selection, sizes, indexes and the summary go to
results/atlas/qa/. Rows are appended; finished (question, arm) pairs and judged answers are skipped (resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "atlas")]
import repo_atlas as RA  # noqa: E402
import repo_atlas_eval as RE  # noqa: E402

OUT = ROOT / "results" / "atlas" / "qa"
CACHE = ROOT / "results" / "cache" / "claude"
MODEL = "claude-opus-5-5"
EFFORT = "high"
JUDGE_EFFORT = "low"
BATCH = 8
SEED = 20261112
TEXT_MAX_TOKENS = 130_000
AGENT_TURNS = 40
PAGE_ARRIVAL = 0.97
WORDS = 120
CLOSED1_N = 36          # exploratory control (amendment 2): closed-book, one question per call, agent_subset(36)
ARMS = ("closed", "map", "atlas", "text", "agent", "closed1")   # closed1: exploratory, one question per call
CATS = ("what", "why", "where", "how")
PRICE_IN, PRICE_WRITE, PRICE_READ, PRICE_OUT = 4e-6, 5e-6, 0.2e-6, 20e-6

SYSTEM = ("You are an expert software engineer answering questions about a Python repository. "
          "Answer only in the format asked for.")
INSTRUCTION = ("Answer each question below about the {repo} repository (commit {sha}). For each one, start a new line "
               "with its id exactly as given (for example `Q7:`), then one paragraph of at most {words} words: the "
               "answer, naming the specific files, classes and functions involved. No code blocks, no headings, no "
               "other text. If you are not sure, give your best answer.")
AGENT_INTRO = ("The current directory is a checkout of the {repo} repository (commit {sha}). Use your tools to look at "
               "the code as much as you need; do not change anything.")
ACK = "\n\nReply only OK."

# The benchmark's judge prompt (Benchmark construction/score/llm-as-a-judge.py, verbatim), split at INPUT so the
# fixed part can be the (cached) system prompt and the INPUT block the user message.
JUDGE_HEAD = """You are a STRICT and RIGOROUS evaluator. You must rate the candidate answer STRICTLY against the reference answer. Be CONSERVATIVE with high scores - only award high scores (16-20) when the candidate answer is truly excellent and closely matches the reference answer in quality and content.

CRITICAL EVALUATION PRINCIPLES:
1. Compare the candidate answer DIRECTLY with the reference answer point by point
2. Any deviation, omission, or inaccuracy should result in score reduction
3. High scores (16-20) should be RARE - reserve them only for answers that are nearly perfect
4. Be strict about factual accuracy - even minor errors should lower the correctness score
5. Missing key points from the reference answer should significantly reduce completeness score
6. Vague or imprecise language should lower clarity scores
7. When in doubt between two score ranges, choose the LOWER one

Evaluation Criteria and Scoring Guidelines (each scored 1 to 20, total score 100):
        1. Correctness (STRICT - penalize any inaccuracies):
            20 — ONLY if completely correct with ALL core points and details accurate, matching reference answer precisely
            16-19 — Mostly correct but must have only TRIVIAL inaccuracies; any noticeable error reduces to 15 or below
            12-15 — Partially correct; has some errors or omissions that affect understanding; main points may be accurate but details are wrong
            8-11 — Several errors or ambiguities that significantly affect understanding of core information
            4-7 — Many errors; misleading or fails to convey key information correctly
            1-3 — Serious errors; completely wrong or misleading
        2. Completeness (STRICT - penalize missing information):
            20 — ONLY if covers ALL key points from reference answer without ANY omission; must match reference in depth
            16-19 — Covers most key points but missing some non-trivial information; minor omissions are acceptable
            12-15 — Missing several important key points; content is noticeably incomplete compared to reference
            8-11 — Important information largely missing; content is one-sided or superficial
            4-7 — Covers very little relevant information; seriously incomplete
            1-3 — Covers almost no relevant information; completely incomplete
        3. Relevance (STRICT - penalize off-topic content):
            20 — ONLY if content is fully focused on question topic with NO irrelevant information whatsoever
            16-19 — Mostly focused but may have minor peripheral information; any significant off-topic content reduces score
            12-15 — Generally on topic but contains some off-topic content that detracts from answer
            8-11 — Topic not sufficiently focused; contains considerable off-topic or tangential content
            4-7 — Content deviates from topic; includes excessive irrelevant information
            1-3 — Majority of content irrelevant to the question
        4. Clarity (STRICT - penalize unclear expression):
            20 — ONLY if language is exceptionally fluent, clear, and precise; very easy to understand without any ambiguity
            16-19 — Mostly fluent and clear but may have minor unclear points; any significant ambiguity reduces score
            12-15 — Generally clear but some expressions are unclear or not concise; may require effort to understand
            8-11 — Expression somewhat awkward; has ambiguity or lacks fluency that hinders understanding
            4-7 — Language obscure; sentences are not smooth; significantly hinders understanding
            1-3 — Expression confusing; very difficult to understand
        5. Reasoning (STRICT - penalize weak logic):
            20 — ONLY if reasoning is exceptionally clear, logical, and well-structured; argumentation is excellent and matches reference quality
            16-19 — Reasoning is clear and logical with solid argumentation; minor logical gaps may exist
            12-15 — Reasoning generally reasonable but has noticeable logical jumps or organization issues
            8-11 — Reasoning is average; has logical jumps or organization problems that affect understanding
            4-7 — Reasoning unclear; lacks logical order; difficult to follow
            1-3 — No clear reasoning; logic is chaotic
"""
JUDGE_TAIL = """
OUTPUT:
    Please output ONLY a JSON object with 5 integer fields in the range [1,20], corresponding
    to the evaluation scores:
        {
        "correctness": <1-20>,
        "completeness": <1-20>,
        "relevance": <1-20>,
        "clarity": <1-20>,
        "reasoning": <1-20>
        }

SCORING INSTRUCTIONS:
- Read the reference answer carefully and identify ALL key points, details, and structure
- Compare the candidate answer systematically against the reference answer
- For each criterion, start with a conservative score and only increase if the candidate truly deserves it
- If the candidate answer is significantly shorter, less detailed, or less precise than the reference, reduce scores accordingly
- If the candidate answer contains information not in the reference (unless it's clearly relevant and accurate), consider reducing relevance score
- When scoring, ask yourself: "Does this candidate answer match the quality and completeness of the reference answer?" If not, reduce scores
- Average or mediocre answers should receive scores in the 8-15 range, not higher
- Only truly excellent answers that closely match the reference should receive 16-20 scores

REQUIREMENT:
    No explanation, no extra text, no formatting other than valid JSON. Be strict and conservative with your scores."""
JUDGE_SYSTEM = JUDGE_HEAD + "\nThe INPUT (question, reference answer, candidate answer) is in the user message.\n" + JUDGE_TAIL
JUDGE_INPUT = "INPUT:\n    Question:{question}\n    Reference Answer:{reference}\n    Candidate Answer:{candidate}"
JUDGE_KEYS = ("correctness", "completeness", "relevance", "clarity", "reasoning")


# --------------------------------------------------------------------------- pure functions (tested)
def category(question: str) -> str:
    """The paper's top-level type (What / Why / Where / How): the first of those words in the question."""
    m = re.search(r"\b(what|why|where|how)\b", question, re.I)
    return m.group(1).lower() if m else "other"


def make_batches(qids_by_cat: dict[str, list[str]], size: int = BATCH, seed: int = SEED) -> list[list[str]]:
    """Seeded, category-balanced batches: shuffle each category, deal them round-robin into one sequence (so each run
    of len(cats) holds one of each), then cut into batches of ``size``; the order inside a batch is shuffled too."""
    rng = random.Random(seed)
    lists = {c: sorted(v) for c, v in sorted(qids_by_cat.items())}
    for c in lists:
        rng.shuffle(lists[c])
    seq, i = [], 0
    while any(lists.values()):
        for c in sorted(lists):
            if lists[c]:
                seq.append(lists[c].pop(0))
        i += 1
    batches = [seq[k:k + size] for k in range(0, len(seq), size)]
    for b in batches:
        rng.shuffle(b)
    return batches


def questions_block(items: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"{qid}: {q}" for qid, q in items)


def parse_answers(text: str, qids: list[str]) -> dict[str, str]:
    """Split a batched answer into {qid: paragraph}. An answer starts at a line beginning with its id and a colon
    (markdown bold or a list marker around it allowed) and runs to the next id line. Missing ids are absent."""
    pat = re.compile(r"^[ \t]*(?:[-*][ \t]*)?\**(" + "|".join(re.escape(q) for q in qids) + r")\**[ \t]*[:.)]\**[ \t]*",
                     re.M) if qids else None
    out: dict[str, str] = {}
    if not text or pat is None:
        return out
    ms = list(pat.finditer(text))
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(text)
        body = text[m.end():end].strip()
        if m.group(1) not in out and body:
            out[m.group(1)] = body
    return out


_PATH = re.compile(r"(?<![\w./-])((?:[\w.-]+/)*[\w-]+\.(?:py|pyi|cfg|toml|ini|txt|rst|md|json|yaml|yml))(?![\w/])")
_DOTTED = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?:\(\))?(?![\w(])")
_BACKTICK = re.compile(r"`([^`\n]{2,80})`")
_CAMEL = re.compile(r"(?<![\w./])([A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)(?![\w/]|\.\w)")
_SNAKE = re.compile(r"(?<![\w./])(_*[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+_*)(?![\w/]|\.\w)")
_EXT = re.compile(r"\.(py|pyi|cfg|toml|ini|txt|rst|md|json|yaml|yml)$")
_STOP_DOTTED = re.compile(r"^(e\.g|i\.e|etc|vs)$", re.I)


def _clean_entity(e: str) -> str | None:
    e = e.strip().strip("()[]{}:;,'\"").rstrip(".")
    e = re.sub(r"\(.*$", "", e)                       # foo(bar) -> foo
    e = re.sub(r"^\./", "", e)                        # ./src/x.py -> src/x.py
    e = re.sub(r"^(self|cls)\.", "", e)               # self.attr -> attr
    if len(e) < 3 or " " in e or _STOP_DOTTED.match(e):
        return None
    return e


def extract_entities(text: str) -> list[str]:
    """Code entities in a (reference) answer: file paths, dotted names, identifiers in backticks, CamelCase names and
    snake_case names with an underscore. Plain words without an underscore or inner capital are not entities."""
    found: list[str] = []

    def add(e):
        e = _clean_entity(e)
        if e and e not in found:
            found.append(e)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # code blocks: their names are counted via the prose
    for m in _BACKTICK.finditer(text):
        inner = m.group(1).strip()
        if re.fullmatch(r"[\w./-]+(?:\(\))?", inner):
            add(inner)
    for rx in (_PATH, _DOTTED, _CAMEL, _SNAKE):
        for m in rx.finditer(text):
            add(m.group(1))
    # keep code-like items only (bare backticked words such as `True` or `json` are not entities)
    kept = [e for e in found if _EXT.search(e) or "." in e or "_" in e or "/" in e
            or re.search(r"[a-z][A-Z]|^[A-Z][a-z0-9]+[A-Z]", e)]
    # one entity per thing: drop an item that is a shorter form of another (``utils.py`` of ``src/requests/utils.py``,
    # ``ensure_ascii`` of ``app.json.ensure_ascii``); the longer one accepts those forms (entity_variants)
    return [e for e in kept if not any(f != e and (f.endswith("/" + e) or f.endswith("." + e)) for f in kept)]


def entity_variants(e: str) -> list[str]:
    """Forms of an entity that count as naming it: the entity itself; for a path, its last two components and its
    file name (the ``src/`` layout or a package prefix may be left out); for a dotted name, its last two components
    and its last component when that has an underscore or an inner capital (a distinctive identifier)."""
    v = [e]
    if "/" in e:
        v += ["/".join(e.split("/")[-2:]), e.rstrip("/").split("/")[-1]]
    elif "." in e and not _EXT.search(e):
        parts = e.split(".")
        if len(parts) > 2:
            v.append(".".join(parts[-2:]))
        if "_" in parts[-1] or re.search(r"[a-z][A-Z]|^[A-Z][a-z0-9]+[A-Z]", parts[-1]):
            v.append(parts[-1])
    return list(dict.fromkeys(x for x in v if x))


def entity_recall(entities: list[str], answer: str) -> float | None:
    """Share of entities named in the answer (any variant, as a whole token, case-sensitive). None if no entities."""
    if not entities:
        return None
    hit = 0
    for e in entities:
        for v in entity_variants(e):
            if re.search(r"(?<![\w])" + re.escape(v) + r"(?![\w])", answer or ""):
                hit += 1
                break
    return hit / len(entities)


def cross_file(reference: str) -> bool:
    """The pre-registered stand-in for the paper's cross-file label (the released data has none): the reference
    answer names at least two distinct source file paths (compared by basename)."""
    names = {p.split("/")[-1] for p in _PATH.findall(reference) if p.endswith((".py", ".pyi"))}
    return len(names) >= 2


def is_test_entity(e: str) -> bool:
    """A test file, test function or fixture file (the atlas, map and text contexts leave tests out)."""
    return bool(re.search(r"(^|/)tests?/|(^|[./])test_|conftest|Test[A-Z]", e))


def parse_judge(text: str) -> dict | None:
    """The judge's JSON -> {criterion: int, total}; None if any criterion is missing or outside 1-20."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in JUDGE_KEYS:
        v = d.get(k)
        if not isinstance(v, (int, float)) or not 1 <= v <= 20:
            return None
        out[k] = int(v)
    out["total"] = sum(out[k] for k in JUDGE_KEYS)
    return out


def bootstrap_ci(values: list[float], n: int = 10000, seed: int = 1) -> tuple[float, float, float]:
    """Mean and 95% percentile interval over resamples of the items."""
    import numpy as np
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    means = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def image_tokens(sizes) -> int:
    return RE.image_tokens([tuple(s) for s in sizes])


# --------------------------------------------------------------------------- dataset, clones, contexts
def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)


def read_commits(bench: Path) -> dict[str, tuple[str, str]]:
    """repo_commit.txt -> {name: (owner/repo, short sha)} (the last line has no newline: read it too)."""
    out = {}
    for line in (bench / "repo_commit.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            url, sha = parts
            full = url.rstrip("/").removesuffix(".git").split("github.com/")[1]
            out[full.split("/")[1]] = (full, sha)
    return out


def clone(work: Path, full: str, sha: str) -> Path:
    d = work / "clones" / full.split("/")[1]
    if not d.exists():
        sh(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", f"https://github.com/{full}.git", str(d)])
    sh(["git", "-C", str(d), "checkout", "-q", "--detach", sha])
    return d


def export(work: Path, name: str, sha: str) -> Path:
    """A plain copy of the pinned commit without .git (the agent's cwd: no history to read)."""
    d = work / "export" / name
    with _export_lock:                                  # parallel agent runs share one export per repository
        if not d.exists():
            tmp = d.with_name(name + ".tmp")
            if tmp.exists():
                subprocess.run(["rm", "-rf", str(tmp)], check=True)
            tmp.mkdir(parents=True)
            arch = subprocess.run(["git", "-C", str(work / "clones" / name), "archive", sha], capture_output=True,
                                  check=True).stdout
            subprocess.run(["tar", "-x", "-C", str(tmp)], input=arch, check=True)
            tmp.rename(d)                               # appears only once complete
    return d


_export_lock = threading.Lock()
_stop = threading.Event()                               # set on a usage-limit error: launch nothing more
_LIMIT = re.compile(r"hit your .{0,20}limit|usage limit|rate limit reached", re.I)


def limit_hit(err: str | None) -> bool:
    if err and _LIMIT.search(err):
        if not _stop.is_set():
            print(f"[qa] USAGE LIMIT: {err[:200]} -- stopping, nothing more is launched", flush=True)
        _stop.set()
        return True
    return False


def prepare(work: Path, bench: Path) -> dict:
    """Measure every repository's atlas at its pinned commit, keep those at most RA.FIT_TOKENS image tokens, build the
    contexts, number the questions and fix the batches."""
    commits = read_commits(bench)
    sizes, chosen = {}, []
    for name, (full, sha) in sorted(commits.items()):
        if name in ("django", "sympy", "scikit-learn", "matplotlib", "astropy"):
            sizes[name] = {"repo": full, "commit": sha, "fits": False, "why": "not measured: far over (>= 2x pytest)"}
            continue
        co = clone(work, full, sha)
        st = RA.stats(co, 88.2, False, 10)
        sizes[name] = {"repo": full, "commit": sh(["git", "-C", str(co), "rev-parse", "HEAD"]).stdout.strip(),
                       **{k: st[k] for k in ("files", "source_chars", "pages", "image_tokens", "text_tokens_est", "fits")}}
        if st["fits"]:
            chosen.append(name)
    questions, batches, ctx_sizes = [], {}, {}
    for name in chosen:
        full, sha = commits[name]
        rows = [json.loads(l) for l in (bench / "Benchmark" / f"{name}.jsonl").read_text().splitlines() if l.strip()]
        by_cat: dict[str, list[str]] = {}
        for i, r in enumerate(rows):
            qid = f"Q{i + 1}"
            c = category(r["question"])
            by_cat.setdefault(c, []).append(qid)
            questions.append({"key": f"{name}/{qid}", "repo": name, "qid": qid, "category": c,
                              "question": r["question"], "reference": r["answer"],
                              "cross_file": cross_file(r["answer"]), "entities": extract_entities(r["answer"])})
        batches[name] = make_batches(by_cat)
        ctx_sizes[name] = contexts(work, name, full, sizes[name]["commit"])
    text_ok = [n for n in chosen if ctx_sizes[n]["text_tokens_est"] <= TEXT_MAX_TOKENS]
    sel = {"source": "github.com/peng-weihan/SWE-QA-Bench Benchmark/*.jsonl (identical to HF swe-qa/SWE-QA-Benchmark)",
           "rule": f"every repository whose atlas (repo_atlas.collect defaults) is at most {RA.FIT_TOKENS} image tokens at "
                   "its pinned commit; all its questions", "repos": chosen, "text_repos": text_ok, "sizes": sizes,
           "batches": batches, "seed": SEED, "questions": questions}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    (OUT / "contexts.json").write_text(json.dumps(ctx_sizes, indent=1))
    return sel


def contexts(work: Path, name: str, full: str, sha: str) -> dict:
    t_dir = work / "ctx" / name
    meta_p = t_dir / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())
    co = work / "clones" / name
    files, uncovered = RA.collect(co, False, 10)
    paths = [f["path"] for f in files]
    texts = [(p, (co / p).read_text(encoding="utf-8")) for p in paths]
    t_dir.mkdir(parents=True, exist_ok=True)
    head = f"Repository {full} at commit {sha[:7]}"
    rmap = (f"{head}: a repo map of its {len(paths)} source files (tests, docs and data files left out). Per file: its "
            "path, then the signature line of every class, function and method in it, indented.\n\n" + RE.build_map(texts))
    text = (f"{head}: the full text of its {len(paths)} source files (tests, docs and data files left out), each after "
            "a line `=== <path> ===`.\n\n" + "".join(f"=== {p} ===\n{x}\n" for p, x in texts))
    am = RA.build(co, t_dir / "atlas", 88.2, False, 10)
    index = (t_dir / "atlas" / "index.txt").read_text()
    atlas = (f"The images above are the {len(paths)} source files of the repository {full} at commit {sha[:7]} (tests, "
             "docs and data files left out), in this order, each starting with a header ▶path◀. The code is minified: "
             "§<n> marks the start of original line n (shown on every 10th line, a bare § starts the others); each » "
             "before a line is one level of indentation where indentation matters; blank lines and other indentation "
             "are not shown. Index (file: page):\n\n" + index)
    closed = f"The questions that follow are about the {full} repository at commit {sha[:7]}. No code from it is shown."
    for k, v in (("closed", closed), ("map", rmap), ("text", text), ("atlas", atlas)):
        (t_dir / f"{k}.txt").write_text(v)
    meta = {"repo": full, "commit": sha, "files": len(paths), "uncovered": len(uncovered),
            "chars": {"map": len(rmap), "text": len(text), "atlas_index": len(atlas)},
            "text_tokens_est": round(sum(f["text_tokens_est"] for f in files)),
            "pages": len(am["pages"]), "image_tokens": am["image_tokens"], "page_sizes": [p["size"] for p in am["pages"]],
            "atlas_dir": str(t_dir / "atlas")}
    meta_p.write_text(json.dumps(meta, indent=1))
    (OUT / "index").mkdir(parents=True, exist_ok=True)
    (OUT / "index" / f"{name}.txt").write_text(index)
    return meta


def sel_load() -> dict:
    return json.loads((OUT / "selection.json").read_text())


# --------------------------------------------------------------------------- spend, rows
_lock, _alock = threading.Lock(), threading.Lock()
_reserved = [0.0]


def jl(name: str) -> list[dict]:
    p = OUT / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def append(name: str, row: dict) -> None:
    with _alock:
        with open(OUT / name, "a") as f:
            f.write(json.dumps(row) + "\n")


def spent() -> float:
    return sum(r["cost_usd"] for r in jl("spend.jsonl"))


def reserve(cost: float, cap: float) -> bool:
    with _lock:
        if spent() + _reserved[0] + cost > cap:
            return False
        _reserved[0] += cost
        return True


def release(cost: float) -> None:
    with _lock:
        _reserved[0] -= cost


def book(what: str, rec: dict) -> None:
    if not rec.get("cached") and rec.get("cost_usd"):
        append("spend.jsonl", {"what": what, "cost_usd": rec["cost_usd"], "input_tokens": rec.get("input_tokens"),
                               "output_tokens": rec.get("output_tokens"), "error": rec.get("error"),
                               "t": time.strftime("%Y-%m-%dT%H:%M:%S")})


# --------------------------------------------------------------------------- the two-message call
def call_two(images, system: str, context: str, questions: str, effort: str, timeout: float = 1200.0,
             max_retries: int = 4, salt: str | None = None) -> dict:
    """Message 1: images + context + "Reply only OK."; message 2: the questions. Cached on disk like
    claude_eval.transcribe (errors never cached). Usage = the second request's (its prompt is the whole context plus
    the questions); cost = the CLI's cumulative total."""
    from vbpe import claude_eval as CE
    for img in images:
        CE.check_image_bounds(img, hires=True)
    pngs = [CE.png_bytes(i) for i in images]
    key = CE.cache_key(MODEL, system, json.dumps(["two-message", context + ACK, questions]), pngs, effort, salt)
    path = CACHE / f"{key}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        rec.update(cached=True)
        return rec
    import base64
    c1 = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                       "data": base64.b64encode(b).decode("ascii")}} for b in pngs]
    c1.append({"type": "text", "text": context + ACK})
    stdin = "".join(json.dumps({"type": "user", "message": {"role": "user", "content": c}}) + "\n"
                    for c in (c1, [{"type": "text", "text": questions}])).encode()
    cmd = CE.build_command(system, MODEL, effort)
    err = "no attempt"
    for attempt in range(max_retries + 1):
        if attempt:
            time.sleep(min(60, 4 * 2 ** attempt))
        t0 = time.monotonic()
        rc, out, stderr = CE._run_cli(cmd, stdin, CE.build_env(), timeout)
        secs = time.monotonic() - t0
        evs = []
        for line in out.splitlines():
            if line.strip().startswith("{"):
                try:
                    evs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        res = [e for e in evs if e.get("type") == "result"]
        if len(res) == 2 and all(r.get("subtype") == "success" and not r.get("is_error") for r in res) and rc == 0:
            u = res[1].get("usage") or {}
            u1 = res[0].get("usage") or {}
            rec = {"text": res[1].get("result") or "", "ack": res[0].get("result") or "",
                   "input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                   + u.get("cache_read_input_tokens", 0), "output_tokens": u.get("output_tokens", 0) + u1.get("output_tokens", 0),
                   "usage": {k: u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                                                      "output_tokens")},
                   "usage_ack": {k: u1.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                           "cache_read_input_tokens", "output_tokens")},
                   "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
                   "cost_usd": float(res[1].get("total_cost_usd") or 0.0), "seconds": round(secs, 1),
                   "stop_reason": res[1].get("stop_reason"), "model": RE_served(res[1]), "cached": False, "error": None,
                   "attempts": attempt + 1, "cache_key": key}
            CACHE.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp{os.getpid()}_{threading.get_ident()}")
            tmp.write_text(json.dumps(rec, indent=1))
            os.replace(tmp, path)
            return rec
        cost = max([float(r.get("total_cost_usd") or 0.0) for r in res] or [0.0])
        err = f"rc={rc} results={[(r.get('subtype'), str(r.get('result'))[:200]) for r in res]} {stderr[-300:]}"
        if cost:
            book("error (retried)", {"cost_usd": cost, "error": err})
    return {"text": "", "error": err, "cost_usd": 0.0, "cached": False}


def RE_served(res: dict) -> str:
    mu = res.get("modelUsage") or {}
    return max(mu, key=lambda m: mu[m].get("outputTokens", 0)) if mu else MODEL


# --------------------------------------------------------------------------- arms 1-4
def batch_jobs(work: Path, arms: list[str]) -> list[dict]:
    from PIL import Image
    sel = sel_load()
    qs = {q["key"]: q for q in sel["questions"]}
    done = {(r["key"], r["arm"]) for r in jl("answers.jsonl")}
    jobs = []
    for name in sel["repos"]:
        meta = json.loads((work / "ctx" / name / "meta.json").read_text())
        for arm in arms:
            if arm == "text" and name not in sel["text_repos"]:
                continue
            ctx = (work / "ctx" / name / f"{arm.rstrip('1')}.txt").read_text()
            groups = sel["batches"][name] if arm != "closed1" else \
                [[k.split("/")[1]] for k in agent_subset(CLOSED1_N) if k.startswith(name + "/")]
            for b, qids in enumerate(groups):
                keys = [f"{name}/{q}" for q in qids]
                if all((k, arm) in done for k in keys):
                    continue
                imgs = [Image.open(Path(meta["atlas_dir"]) / f"page_{i + 1:03d}.png") for i in range(meta["pages"])] \
                    if arm == "atlas" else []
                qtext = INSTRUCTION.format(repo=meta["repo"], sha=meta["commit"][:7], words=WORDS) + "\n\n" + \
                    questions_block([(q, qs[f"{name}/{q}"]["question"]) for q in qids])
                ctx_tok = (meta["image_tokens"] if arm == "atlas" else 0) + len(ctx) / 3.2
                jobs.append({"repo": name, "arm": arm, "batch": b, "qids": qids, "images": imgs, "context": ctx,
                             "questions": qtext, "image_tokens": meta["image_tokens"] if arm == "atlas" else 0,
                             "bound": ctx_tok * PRICE_WRITE * 1.2 + 2 * ctx_tok * PRICE_READ + 12000 * PRICE_OUT})
    return jobs


def run_batches(work: Path, arms: list[str], cap: float, workers: int) -> None:
    jobs = batch_jobs(work, arms)
    print(f"[qa] {len(jobs)} batched calls to make; spent ${spent():.2f} (cap ${cap})", flush=True)

    def run(job):
        if not reserve(job["bound"], cap):
            print(f"[qa] SKIP {job['repo']} {job['arm']} b{job['batch']}: cap", flush=True)
            return
        try:
            rec = call_two(job["images"], SYSTEM, job["context"], job["questions"], EFFORT)
        finally:
            release(job["bound"])
        book(f"{job['repo']} {job['arm']} b{job['batch']}", rec)
        if rec.get("error"):
            print(f"[qa] ERROR {job['repo']} {job['arm']} b{job['batch']}: {rec['error'][:300]}", flush=True)
            return
        arrived = job["arm"] != "atlas" or rec["input_tokens"] >= PAGE_ARRIVAL * job["image_tokens"]
        ans = parse_answers(rec["text"], job["qids"])
        base = {k: rec.get(k) for k in ("input_tokens", "output_tokens", "thinking_tokens", "cost_usd", "seconds",
                                        "cached", "usage", "usage_ack", "stop_reason", "model", "cache_key")}
        n = len(job["qids"])
        call_row = {"repo": job["repo"], "arm": job["arm"], "batch": job["batch"], "qids": job["qids"], **base,
                    "image_tokens_formula": job["image_tokens"], "pages_arrived": arrived, "answered": len(ans),
                    "ack": rec.get("ack", "")[:200], "text": rec["text"]}
        append("calls.jsonl", call_row)
        for q in job["qids"]:
            append("answers.jsonl", {"key": f"{job['repo']}/{q}", "repo": job["repo"], "qid": q, "arm": job["arm"],
                                     "batch": job["batch"], "answer": ans.get(q, ""), "missing": q not in ans,
                                     "valid": arrived, "cost_usd": (rec.get("cost_usd") or 0) / n,
                                     "input_tokens": rec["input_tokens"], "seconds": (rec.get("seconds") or 0) / n,
                                     "turns": 1})
        print(f"[qa] {job['repo']:9} {job['arm']:6} b{job['batch']} answered {len(ans)}/{n} in={rec['input_tokens']:,} "
              f"(read {rec['usage'].get('cache_read_input_tokens', 0):,}) ${rec.get('cost_usd', 0):.3f} {rec.get('seconds', 0):.0f}s"
              f"{'' if arrived else ' PAGES DROPPED'}", flush=True)

    # per (repo, arm): batch 0 first so that it writes the cache, then the rest in parallel
    firsts = [j for j in jobs if j["batch"] == min(x["batch"] for x in jobs if (x["repo"], x["arm"]) == (j["repo"], j["arm"]))]
    rest = [j for j in jobs if j not in firsts]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, firsts))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, rest))


# --------------------------------------------------------------------------- arm 5: agent
def agent_subset(n: int | None) -> list[str]:
    """All question keys, or a seeded subset of n, balanced over repositories and categories (round-robin)."""
    sel = sel_load()
    keys = [q["key"] for q in sel["questions"]]
    if not n or n >= len(keys):
        return keys
    rng = random.Random(SEED + 5)
    groups: dict[tuple, list[str]] = {}
    for q in sel["questions"]:
        groups.setdefault((q["repo"], q["category"]), []).append(q["key"])
    for g in groups.values():
        rng.shuffle(g)
    order, gs = [], [groups[k] for k in sorted(groups)]
    while any(gs):
        for g in gs:
            if g:
                order.append(g.pop(0))
    return order[:n]


def agent_once(work: Path, q: dict, meta: dict) -> dict:
    from vbpe import claude_eval as CE
    cwd = export(work, q["repo"], meta["commit"])
    prompt = (AGENT_INTRO.format(repo=meta["repo"], sha=meta["commit"][:7]) + "\n\n"
              + INSTRUCTION.format(repo=meta["repo"], sha=meta["commit"][:7], words=WORDS) + "\n\n"
              + questions_block([(q["qid"], q["question"])]))
    cmd = [CE.CLAUDE_BIN, "-p", prompt, "--output-format", "stream-json", "--verbose", "--model", MODEL, "--effort",
           EFFORT, "--tools", "Read,Grep,Glob", "--setting-sources", "", "--strict-mcp-config",
           "--disable-slash-commands", "--no-session-persistence", "--max-turns", str(AGENT_TURNS)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=CE.build_env(), cwd=cwd, timeout=1800,
                           stdin=subprocess.DEVNULL)
        stdout, rc, err = p.stdout, p.returncode, p.stderr
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc, err = -9, "timeout"
    secs = time.time() - t0
    tdir = work / "agent_streams"
    tdir.mkdir(exist_ok=True)
    (tdir / f"{q['key'].replace('/', '_')}.jsonl").write_text(stdout)
    ev = []
    for l in stdout.splitlines():
        if l.strip().startswith("{"):
            try:
                ev.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    res = next((e for e in reversed(ev) if e.get("type") == "result"), None)
    calls = [c for e in ev if e.get("type") == "assistant" for c in e["message"]["content"] if c.get("type") == "tool_use"]
    if res is None or (res.get("is_error") and res.get("subtype") != "error_max_turns"):
        return {"error": f"rc={rc} {str((res or {}).get('result'))[:300]} {err[-300:]}",
                "cost_usd": float((res or {}).get("total_cost_usd") or 0.0)}
    u = res.get("usage") or {}
    text = res.get("result") or ""
    if not text:
        texts = [c["text"] for e in ev if e.get("type") == "assistant" for c in e["message"]["content"] if c.get("type") == "text"]
        text = texts[-1] if texts else ""
    names = [c["name"] for c in calls]
    outside = [c["input"] for c in calls if any(isinstance(v, str) and v.startswith("/") and not v.startswith(str(cwd))
                                                 for v in (c.get("input") or {}).values())]
    return {"text": text, "cost_usd": float(res.get("total_cost_usd") or 0.0), "seconds": round(secs, 1),
            "turns": res.get("num_turns"), "subtype": res.get("subtype"),
            "input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0), "tools": {k: names.count(k) for k in sorted(set(names))},
            "tool_calls": len(calls), "outside_cwd": outside[:10],
            "permission_denials": [str(d)[:200] for d in res.get("permission_denials") or []][:10], "model": RE_served(res)}


def run_agent(work: Path, cap: float, workers: int, n: int | None) -> None:
    sel = sel_load()
    qs = {q["key"]: q for q in sel["questions"]}
    done = {r["key"] for r in jl("answers.jsonl") if r["arm"] == "agent"}
    todo = [k for k in agent_subset(n) if k not in done]
    print(f"[qa] {len(todo)} agent runs to make; spent ${spent():.2f} (cap ${cap})", flush=True)
    worst = [0.25]                                       # measured agent runs: $0.045-0.082

    def run(key):
        if _stop.is_set():
            return
        bound = worst[0]
        if not reserve(bound, cap):
            print(f"[qa] SKIP agent {key}: cap", flush=True)
            return
        q = qs[key]
        meta = json.loads((work / "ctx" / q["repo"] / "meta.json").read_text())
        try:
            rec = agent_once(work, q, meta)
        finally:
            release(bound)
        book(f"{key} agent", rec)
        if rec.get("cost_usd"):
            worst[0] = max(0.25, rec["cost_usd"] * 3)
        if rec.get("error"):
            limit_hit(rec["error"])
            print(f"[qa] ERROR agent {key}: {rec['error'][:300]}", flush=True)
            return
        ans = parse_answers(rec["text"], [q["qid"]])
        text = ans.get(q["qid"]) or rec["text"].strip()      # one question: an answer without the id still counts
        row = {"key": key, "repo": q["repo"], "qid": q["qid"], "arm": "agent", "answer": text,
               "missing": not text, "id_format": q["qid"] in ans, "valid": True,
               **{k: rec.get(k) for k in ("cost_usd", "input_tokens", "output_tokens", "seconds", "turns", "subtype",
                                         "tools", "tool_calls", "outside_cwd", "permission_denials", "model")}}
        append("answers.jsonl", row)
        print(f"[qa] {key:16} agent turns={row['turns']} in={row['input_tokens']:,} ${row['cost_usd']:.3f} "
              f"{row['seconds']:.0f}s tools={row['tools']}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))


# --------------------------------------------------------------------------- judge
def judge(cap: float, workers: int) -> None:
    """One judge call per answer, the benchmark's rubric, blind: the judge sees question, reference and the answer
    text only; calls go in a seeded random order. Empty answers (missing) are not judged: they score the minimum (5)."""
    from vbpe import claude_eval as CE
    sel = sel_load()
    qs = {q["key"]: q for q in sel["questions"]}
    done = {(r["key"], r["arm"]) for r in jl("judged.jsonl")}
    todo = [r for r in latest_answers().values() if (r["key"], r["arm"]) not in done]
    random.Random(SEED + 7).shuffle(todo)
    print(f"[qa] {len(todo)} answers to judge; spent ${spent():.2f} (cap ${cap})", flush=True)

    def run(r):
        if _stop.is_set():
            return
        q = qs[r["key"]]
        if r["missing"] or not r["answer"].strip():
            append("judged.jsonl", {"key": r["key"], "arm": r["arm"], "scores": None, "total": 5, "unjudged": "empty"})
            return
        bound = 0.06
        if not reserve(bound, cap):
            print(f"[qa] SKIP judge {r['key']} {r['arm']}: cap", flush=True)
            return
        try:
            rec = CE.transcribe([], JUDGE_SYSTEM, JUDGE_INPUT.format(question=q["question"], reference=q["reference"],
                                                                    candidate=r["answer"]),
                                model=MODEL, effort=JUDGE_EFFORT, timeout=300.0)
        finally:
            release(bound)
        book(f"judge {r['key']} {r['arm']}", rec)
        if rec.get("error"):
            limit_hit(rec["error"])
            print(f"[qa] ERROR judge {r['key']} {r['arm']}: {rec['error'][:200]}", flush=True)
            return
        sc = parse_judge(rec["text"])
        append("judged.jsonl", {"key": r["key"], "arm": r["arm"], "scores": sc, "total": sc["total"] if sc else None,
                                "unjudged": None if sc else "unparsable", "raw": rec["text"][:500],
                                "cost_usd": rec.get("cost_usd"), "cached": rec.get("cached")})

    if todo:                                    # the first call alone writes the rubric (system prompt) to the cache
        run(todo[0])
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo[1:]))


def latest_answers() -> dict[tuple, dict]:
    out = {}
    for r in jl("answers.jsonl"):
        out[(r["key"], r["arm"])] = r
    return out


# --------------------------------------------------------------------------- estimate and analysis
def estimate(work: Path, agent_cost: float, agent_n: int | None, out_tokens: int = 4000,
             judge_cost: float = 0.009) -> dict:
    sel = sel_load()
    est = {a: 0.0 for a in ARMS}
    n_answers = 0
    for name in sel["repos"]:
        meta = json.loads((work / "ctx" / name / "meta.json").read_text())
        nb = len(sel["batches"][name])
        nq = sum(len(b) for b in sel["batches"][name])
        for arm in ("closed", "map", "atlas", "text"):
            if arm == "text" and name not in sel["text_repos"]:
                continue
            ctx = (meta["image_tokens"] if arm == "atlas" else 0) + len((work / "ctx" / name / f"{arm}.txt").read_text()) / 3.2
            # first call writes the context; every call reads it twice (ack turn + question turn) except the write
            est[arm] += ctx * PRICE_WRITE + (2 * nb - 1) * ctx * PRICE_READ + nb * (out_tokens * PRICE_OUT + 600 * PRICE_WRITE)
            n_answers += nq
    na = len(agent_subset(agent_n))
    est["agent"] = na * agent_cost
    est["judge"] = (n_answers + na) * judge_cost
    est["total"] = sum(est.values())
    return {k: round(v, 2) for k, v in est.items()}


def analyse() -> str:
    import numpy as np
    sel = sel_load()
    qs = {q["key"]: q for q in sel["questions"]}
    ans = latest_answers()
    jd = {}
    for r in jl("judged.jsonl"):
        jd[(r["key"], r["arm"])] = r
    rows = {}
    for (k, arm), a in ans.items():
        j = jd.get((k, arm))
        if j is None or j.get("total") is None:
            continue
        rows[(k, arm)] = {"score": j["total"], "valid": a.get("valid", True), "missing": a["missing"],
                          "recall": entity_recall(qs[k]["entities"], a["answer"]),
                          "recall_src": entity_recall([e for e in qs[k]["entities"] if not is_test_entity(e)], a["answer"]),
                          "cost": a.get("cost_usd") or 0,
                          "in": a.get("input_tokens") or 0, "secs": a.get("seconds") or 0, "turns": a.get("turns") or 1,
                          "cat": qs[k]["category"], "cross": qs[k]["cross_file"], "repo": qs[k]["repo"]}
    arms = [a for a in ARMS if any(x == a for _, x in rows)]
    L = ["### Per arm (judge total 5-100; entity recall 0-1; 95% bootstrap intervals over questions)", "",
         "| arm | questions | judge score | entity recall | recall, non-test entities | missing answers | mean input tokens | "
         "cost per question | seconds per question | turns |", "|---|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for arm in arms:
        rs = [v for (k, a), v in rows.items() if a == arm and v["valid"]]
        s = bootstrap_ci([r["score"] for r in rs])
        rec = [r["recall"] for r in rs if r["recall"] is not None]
        e = bootstrap_ci(rec)
        es = bootstrap_ci([r["recall_src"] for r in rs if r["recall_src"] is not None])
        summ[arm] = {"n": len(rs), "score": s, "recall": e, "recall_src": es,
                     "cost_per_q": float(np.mean([r["cost"] for r in rs]))}
        L.append(f"| {arm} | {len(rs)} | {s[0]:.1f} ({s[1]:.1f}-{s[2]:.1f}) | {e[0]:.2f} ({e[1]:.2f}-{e[2]:.2f}) | "
                 f"{es[0]:.2f} ({es[1]:.2f}-{es[2]:.2f}) | "
                 f"{sum(r['missing'] for r in rs)} | {np.mean([r['in'] for r in rs]):,.0f} | ${np.mean([r['cost'] for r in rs]):.3f} | "
                 f"{np.mean([r['secs'] for r in rs]):.0f} | {np.mean([r['turns'] for r in rs]):.1f} |")
    inval = [(k, a) for (k, a), v in rows.items() if not v["valid"]]
    L += ["", f"Invalid (pages dropped) answers left out: {len(inval)}.", ""]

    def paired(a, b, filt=lambda v: True, key="score"):
        ks = [k for (k, x) in rows if x == a and (k, b) in rows and rows[(k, a)]["valid"] and rows[(k, b)]["valid"]
              and filt(rows[(k, a)])]
        d = [rows[(k, a)][key] - rows[(k, b)][key] for k in ks
             if rows[(k, a)][key] is not None and rows[(k, b)][key] is not None]
        return len(d), bootstrap_ci(d)

    pairs = [("atlas", "map"), ("atlas", "agent"), ("atlas", "text"), ("atlas", "closed"), ("map", "closed"),
             ("text", "closed"), ("agent", "closed"), ("agent", "closed1"), ("closed1", "closed"), ("atlas", "closed1")]
    L += ["### Paired differences (same questions; 95% bootstrap intervals)", "",
          "| difference | questions | judge score | entity recall |", "|---|---|---|---|"]
    diffs = {}
    for a, b in pairs:
        if a in arms and b in arms:
            n, s = paired(a, b)
            n2, e = paired(a, b, key="recall")
            diffs[f"{a}-{b}"] = {"n": n, "score": s, "recall": e}
            L.append(f"| {a} − {b} | {n} | {s[0]:+.1f} ({s[1]:+.1f} to {s[2]:+.1f}) | {e[0]:+.3f} ({e[1]:+.3f} to {e[2]:+.3f}) |")
    L += ["", "### Per category (judge score; n per cell in brackets)", "",
          "| category | " + " | ".join(arms) + " | atlas − map | atlas − agent |", "|---|" + "---|" * (len(arms) + 2)]
    groups = [(c, lambda v, c=c: v["cat"] == c) for c in CATS] + [
        ("cross-file (reference names >= 2 .py files)", lambda v: v["cross"]),
        ("single-file or none", lambda v: not v["cross"])]
    percat = {}
    for name, f in groups:
        cells = []
        for arm in arms:
            rs = [v["score"] for (k, a), v in rows.items() if a == arm and v["valid"] and f(v)]
            cells.append(f"{np.mean(rs):.1f} ({len(rs)})" if rs else "")
        dm = paired("atlas", "map", f) if "map" in arms and "atlas" in arms else (0, (np.nan,) * 3)
        dg = paired("atlas", "agent", f) if "agent" in arms and "atlas" in arms else (0, (np.nan,) * 3)
        percat[name] = {"atlas-map": dm, "atlas-agent": dg}
        L.append(f"| {name} | " + " | ".join(cells) + f" | {dm[1][0]:+.1f} ({dm[1][1]:+.1f} to {dm[1][2]:+.1f}) | "
                 f"{dg[1][0]:+.1f} ({dg[1][1]:+.1f} to {dg[1][2]:+.1f}) |")
    L += ["", "### Per repository (judge score)", "", "| repository | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for repo in sel["repos"]:
        cells = []
        for arm in arms:
            rs = [v["score"] for (k, a), v in rows.items() if a == arm and v["valid"] and v["repo"] == repo]
            cells.append(f"{np.mean(rs):.1f} ({len(rs)})" if rs else "")
        L.append(f"| {repo} | " + " | ".join(cells) + " |")
    # spread / ceiling check
    allsc = [v["score"] for v in rows.values()]
    if allsc:
        L += ["", f"Score spread over all judged answers: min {min(allsc)}, quartiles "
              f"{np.percentile(allsc, 25):.0f} / {np.percentile(allsc, 50):.0f} / {np.percentile(allsc, 75):.0f}, max {max(allsc)}."]
    sp = jl("spend.jsonl")
    by = {}
    for r in sp:
        w = r["what"]
        k = ("harness checks" if w.startswith("harness") else "judge" if w.startswith("judge") else "agent" if w.endswith(" agent") else
             w.split()[1] if len(w.split()) == 3 and w.split()[1] in ARMS else "other")
        by[k] = by.get(k, 0) + r["cost_usd"]
    L += ["", "### Spend (list price, every uncached call)", "", "| part | spent |", "|---|---|"]
    L += [f"| {k} | ${v:.2f} |" for k, v in sorted(by.items())] + [f"| **total** | **${sum(by.values()):.2f}** |"]
    (OUT / "summary.json").write_text(json.dumps({"arms": summ, "diffs": diffs, "percat": percat, "spend": by}, indent=1))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["prepare", "estimate", "run", "judge", "analyse"])
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("REPO_ATLAS_QA_WORK", "/tmp/repo_atlas_qa_work")))
    ap.add_argument("--bench", type=Path, help="a clone of github.com/peng-weihan/SWE-QA-Bench")
    ap.add_argument("--arms", nargs="+", default=["closed", "map", "atlas", "text"], choices=ARMS)
    ap.add_argument("--cap", type=float, default=22.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--agent-n", type=int, default=None, help="seeded subset of questions for the agent arm")
    ap.add_argument("--agent-cost", type=float, default=0.07)
    ap.add_argument("--judge-cost", type=float, default=0.008)
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare":
        sel = prepare(a.work, a.bench)
        print(json.dumps(sel["sizes"], indent=1))
        print("repos", sel["repos"], "text", sel["text_repos"], "questions", len(sel["questions"]))
    elif a.cmd == "estimate":
        print(json.dumps(estimate(a.work, a.agent_cost, a.agent_n, judge_cost=a.judge_cost), indent=1))
    elif a.cmd == "run":
        one = [x for x in a.arms if x != "agent"]
        if one:
            run_batches(a.work, one, a.cap, a.workers)
        if "agent" in a.arms:
            run_agent(a.work, a.cap, a.workers, a.agent_n)
        print(f"[qa] spent ${spent():.2f} list price", flush=True)
    elif a.cmd == "judge":
        judge(a.cap, a.workers)
        print(f"[qa] spent ${spent():.2f} list price", flush=True)
    else:
        print(analyse())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
