"""Repo atlas against security flaws: does the whole repository as page images help FIND a published, fixed flaw?

Tasks come from the GitHub Advisory Database (reviewed advisories published from 2026-07-01, after the model's
training data): one fix commit on GitHub, the vulnerable version = its parent, a small library whose atlas fits one
request. Every arm is told only that the repository at this commit has at least one security vulnerability that was
fixed later, and reports up to 5 findings (file | function | weakness class | why). Locations and classes only.

Arms (Opus 5.5, high effort): closed (file paths only), atlas (pages + index, one call), agent (Claude Code, read-only
tools in an export of the vulnerable commit), agent_atlas (the same agent with the pages in its first message).
Design, predictions and results: results/atlas/sec/README.md.

    python3 experiments/atlas/repo_atlas_sec.py select   [--work DIR]     # advisory funnel, clones, atlas sizes (no model calls)
    python3 experiments/atlas/repo_atlas_sec.py prepare  [--work DIR]     # exports, atlases, contexts, gold (no model calls)
    python3 experiments/atlas/repo_atlas_sec.py run      [--work DIR] --tasks 3 --cap 25    # first n tasks, all arms
    python3 experiments/atlas/repo_atlas_sec.py analyse                   # prints the tables, writes summary.json

Clones, exports, page images and agent transcripts stay in --work; rows, the selection and summary.json go to
results/atlas/sec/. Rows are appended per (task, arm); finished pairs are skipped (resumable). Errors are
retried on the next start, never recorded as results; a usage-limit error stops the run.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "atlas"), str(ROOT / "experiments" / "readhook")]
import repo_atlas as RA  # noqa: E402
import repo_atlas_eval as RE  # noqa: E402
from vtchook.minify.base import REGISTRY, detect  # noqa: E402

OUT = ROOT / "results" / "atlas" / "sec"
DEFAULT_WORK = Path(os.environ.get("REPO_ATLAS_SEC_WORK",
                                   "/tmp/claude-0/-home-user-visual-token-compression/588efa4c-1231-5348-b908-e0ab20b8aad7/scratchpad/sec"))
MODEL = "claude-opus-5-5"
EFFORT = "high"
ARMS = ("closed", "atlas", "agent", "agent_atlas")
AGENT_TURNS = 40
AGENT_BUDGET_USD = 4.0          # a run-away guard per agent run (the CLI's --max-budget-usd); recorded if hit
PAGE_ARRIVAL = 0.97

SINCE = "2026-07-01"            # advisories published from this date; the fix commit must be from it too
MAX_EXAMINED = 300
MAX_TASKS = 14
PER_REPO = 2
MIN_TOKENS, MAX_TOKENS = 15_000, RA.FIT_TOKENS
MAX_GOLD = 5
MAX_PATHS_PREFILTER = 2500      # more atlas-candidate paths than this at the parent: too large without measuring
CODE_KINDS = {"python", "shell", "javascript", "typescript", "tsx", "java", "kotlin", "scala", "c", "cpp", "csharp",
              "go", "rust", "swift", "php", "dart"}
# programming languages repo_atlas has no minifier for: a fix touching one of these fails rule 3
UNCOVERED_CODE = re.compile(r"\.(rb|erb|rake|ex|exs|heex|eex|erl|hrl|lua|pl|pm|r|cr|clj|cljs|hs|ml|mli|fs|fsx|vb|"
                            r"groovy|gradle|m|mm|nim|zig|vue|svelte|astro|twig|jinja2?|j2|njk|hbs|handlebars|mustache|"
                            r"liquid|cfm|jsp|asp|aspx|cshtml|razor|tpl|pas|f90|jl|v|sol|move|cairo|elm|purs|re|res|"
                            r"coffee|ps1|psm1|bat|cmd|tcl|awk|sed|d|ada|adb|ads|cob|cbl)$", re.I)

SYSTEM = ("You are an expert security code auditor. You review source code defensively and report where its "
          "vulnerabilities are. Answer only in the format asked for.")
INSTRUCTION = ("This repository, at this commit, contains at least one security vulnerability that was fixed later. "
               "Audit it and report up to 5 findings, most likely first, each on one line as "
               "`FINDING: <file path> | <function or line range> | <weakness class, CWE-style> | <one sentence why>`, "
               "with file paths exactly as they appear in the repository. Report locations and classes only: no "
               "exploit code, no proof of concept.")
CLOSED_HEAD = ("Repository {repo} at commit {sha}. Its code is not shown: only the paths of its {n} source files "
               "(tests, docs and data files left out).\n\n{paths}")
ATLAS_HEAD = ("The images above are the {n} source files of the repository {repo} at commit {sha} (tests, docs and data "
              "files left out), in this order, each starting with a header ▶path◀. The code is minified: §<n> marks the "
              "start of original line n (shown on every 10th line, a bare § starts the others); each » before a line "
              "is one level of indentation where indentation matters; blank lines and other indentation are not "
              "shown. Index (file: page):\n\n{index}")
AGENT_HEAD = ("The current directory is a checkout of the {repo} repository at commit {sha}. Use your tools to look at "
              "the code as much as you need; do not change anything.")
AGENT_ATLAS_NOTE = ("The page images show the repository's source; use them for orientation and read the files when "
                    "exact text matters.")

# --------------------------------------------------------------------------- CWE families (fixed before scoring)
FAMILY_CWES = {
    "injection": [74, 75, 77, 78, 88, 89, 90, 91, 93, 94, 95, 96, 97, 113, 116, 117, 134, 643, 652, 917, 943, 1236,
                  1336, 99, 470, 1427],
    "xss": [79, 80, 81, 83, 84, 85, 86, 87],
    "path": [22, 23, 24, 25, 26, 27, 28, 29, 32, 35, 36, 37, 38, 39, 40, 41, 59, 61, 62, 64, 65, 66, 73, 434, 641,
             706, 1386],
    "ssrf": [918, 441],
    "access": [250, 266, 269, 270, 271, 272, 276, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291,
               294, 302, 303, 304, 305, 306, 307, 308, 309, 425, 501, 520, 521, 522, 523, 549, 551, 552, 566, 603,
               620, 639, 640, 648, 708, 732, 749, 862, 863, 939, 1188, 1220, 1230, 1390, 1391],
    "memory": [118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 135, 170, 190, 191, 192, 193,
               194, 195, 196, 197, 415, 416, 457, 466, 467, 468, 469, 476, 562, 587, 590, 680, 681, 682, 761, 762,
               763, 786, 787, 788, 805, 806, 822, 823, 824, 825, 908, 909, 1284, 1335, 1341],
    "dos": [400, 401, 404, 405, 407, 409, 674, 689, 690, 703, 754, 755, 770, 771, 772, 773, 774, 775, 776, 789, 834,
            835, 1050, 1325, 1333],
    "crypto": [295, 296, 297, 298, 299, 310, 311, 312, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330,
               331, 334, 335, 336, 337, 338, 340, 345, 346, 347, 348, 349, 351, 353, 354, 358, 757, 759, 760, 780,
               916, 924, 1204, 1240],
    "deserialization": [502, 915, 1321],
    "info": [200, 201, 202, 203, 204, 208, 209, 210, 211, 212, 213, 214, 215, 359, 497, 532, 538, 540, 548, 598,
             615, 1295],
    "redirect": [601],
    "csrf": [352, 1385],
    "xxe": [611, 827],
    "race": [362, 364, 366, 367, 368, 412, 413, 421, 662, 667, 821, 833, 1223],
    "session": [384, 565, 613, 614, 1004, 1275, 1021, 346],
    "validation": [20, 112, 115, 138, 141, 150, 159, 172, 176, 178, 179, 180, 182, 184, 185, 186, 436, 444, 451, 472,
                   473, 625, 641, 646, 692, 697, 704, 707, 838, 843, 1173, 1285, 1286, 1287, 1288, 1289, 1389],
}
CWE_FAMILY = {}
for _fam, _ids in FAMILY_CWES.items():
    for _i in _ids:
        CWE_FAMILY.setdefault(_i, _fam)
# words in a finding's class field when it gives no CWE id (first match wins, in this order)
FAMILY_WORDS = [
    ("xss", r"cross[- ]site scripting|\bxss\b|html injection|script injection into (?:html|the page)"),
    ("ssrf", r"\bssrf\b|server[- ]side request forgery"),
    ("csrf", r"\bcsrf\b|cross[- ]site request forgery"),
    ("xxe", r"\bxxe\b|xml external entit"),
    ("redirect", r"open redirect"),
    ("deserialization", r"deserializ|prototype pollution|mass assignment|unsafe (?:yaml|pickle) load"),
    ("path", r"path traversal|directory traversal|zip ?slip|symlink|arbitrary file (?:read|write)|file upload"),
    ("injection", r"injection|command exec|code exec|\beval\b|template injection|\bssti\b|format string|"
                  r"crlf|header injection|log injection"),
    ("memory", r"buffer|overflow|out[- ]of[- ]bounds|use[- ]after[- ]free|double free|null (?:pointer )?deref|"
               r"memory (?:safety|corruption)|integer (?:overflow|underflow|wrap)|uninitiali"),
    ("dos", r"denial[- ]of[- ]service|\bdos\b|redos|regular expression|resource exhaustion|uncontrolled "
            r"(?:resource|recursion|allocation)|unbounded|infinite loop|decompression bomb|zip bomb|algorithmic "
            r"complexity|stack exhaustion|excessive (?:allocation|memory)"),
    ("crypto", r"certificate|\btls\b|\bssl\b|crypto|signature|\bhmac\b|random|nonce|weak (?:hash|cipher)|timing"),
    ("race", r"race condition|\btoctou\b|time[- ]of[- ]check"),
    ("session", r"session|cookie"),
    ("info", r"information (?:disclosure|exposure|leak)|sensitive (?:data|information)|secret|credential leak|"
             r"exposure of"),
    ("access", r"auth|access control|permission|privilege|bypass|idor|insecure direct object"),
    ("validation", r"validation|sanitiz|request smuggling|type confusion|improper (?:input|check)"),
]


def cwe_family(cwe: int | str | None) -> str:
    """CWE id (``79``, ``'CWE-79'``) -> its family in FAMILY_CWES, or 'other'."""
    if cwe is None:
        return "other"
    m = re.search(r"(\d+)", str(cwe))
    return CWE_FAMILY.get(int(m.group(1)), "other") if m else "other"


def class_families(text: str) -> set[str]:
    """A finding's weakness-class field -> families: from every CWE id in it; if it has none, from the first matching
    word pattern in FAMILY_WORDS (none: empty set)."""
    ids = re.findall(r"CWE[- ]?(\d+)", text or "", re.I)
    fams = {cwe_family(i) for i in ids} - {"other"}
    if ids:
        return fams
    for fam, rx in FAMILY_WORDS:
        if re.search(rx, text or "", re.I):
            return {fam}
    return set()


# --------------------------------------------------------------------------- advisories (pure)
COMMIT_URL = re.compile(r"^https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/(?:pull/\d+/)?commits?/([0-9a-f]{7,40})"
                        r"(?:[/?#.].*)?$", re.I)


def fix_commits(refs: list[dict]) -> list[tuple[str, str]]:
    """GitHub commit URLs among an advisory's references -> distinct (owner/repo lowercased, sha), in order. A short
    sha that is a prefix of a longer one in the same repository is the same commit."""
    out: list[tuple[str, str]] = []
    for r in refs or []:
        m = COMMIT_URL.match((r.get("url") or "").strip())
        if not m:
            continue
        repo, sha = f"{m.group(1)}/{m.group(2)}".lower(), m.group(3).lower()
        same = [i for i, (rp, s) in enumerate(out) if rp == repo and (s.startswith(sha) or sha.startswith(s))]
        if same:
            i = same[0]
            if len(sha) > len(out[i][1]):
                out[i] = (repo, sha)
            continue
        out.append((repo, sha))
    return out


def parse_advisory(d: dict) -> dict:
    db = d.get("database_specific") or {}
    return {"id": d.get("id"), "published": d.get("published", ""), "withdrawn": d.get("withdrawn"),
            "cwes": [c for c in db.get("cwe_ids") or []], "severity": db.get("severity"),
            "ecosystems": sorted({(a.get("package") or {}).get("ecosystem") for a in d.get("affected") or []} - {None}),
            "commits": fix_commits(d.get("references") or [])}


def order_advisories(advs: list[dict], since: str = SINCE) -> list[dict]:
    """Rule 1: reviewed advisories published on or after ``since``, newest first (ties: id)."""
    keep = [a for a in advs if a["published"] >= since]
    return sorted(keep, key=lambda a: (a["published"], a["id"]), reverse=True)


def code_kind(path: str) -> str | None:
    m = detect(path)
    return m.kind if m and m.kind in CODE_KINDS else None


def counted_source(path: str) -> bool:
    """A source file for rule 3: not a test, doc, changelog, lock or vendored file (repo_atlas_eval.is_gold) and in a
    programming language (with or without a minifier). Config and data files (json, yaml, toml, ...) are not."""
    return RE.is_gold(path) and (code_kind(path) is not None or bool(UNCOVERED_CODE.search(path)))


def classify_fix(changed: list[str]) -> tuple[str | None, list[str]]:
    """Rule 3 on the fix's changed paths -> (reason it fails or None, gold = the counted source files)."""
    src = [p for p in changed if counted_source(p)]
    if not src:
        return "no source file changed", []
    if any(code_kind(p) is None for p in src):
        return "a changed source file has no minifier", src
    if len(src) > MAX_GOLD:
        return f"more than {MAX_GOLD} source files changed", src
    return None, src


def patch_sections(patch: str, keep: set[str]) -> str:
    """Only the ``diff --git`` sections of the files in ``keep``."""
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    out = []
    for p in parts:
        m = re.match(r"diff --git a/(\S+) b/(\S+)", p)
        if m and (m.group(2) in keep or m.group(1) in keep):
            out.append(p)
    return "".join(out)


_KW = {"if", "for", "while", "switch", "catch", "return", "function", "sizeof", "new", "elif", "def", "fn", "func",
       "else", "case", "when", "match", "with", "typeof", "await", "async", "super", "this", "self", "assert", "throw",
       "import", "from", "class", "struct", "enum", "interface", "trait", "type", "public", "private", "protected",
       "static", "final", "const", "let", "var", "try", "finally", "foreach", "elseif", "synchronized", "require",
       "constructor", "init", "__init__", "get", "set", "and", "not", "the", "to", "of", "in", "is", "or"}
# one declaration per line: Python def/class; JS/TS function, assignment of a function or arrow, class method; Go
# func; Rust fn; Java/Kotlin/C#/PHP/C methods (modifiers, a return type, a name, then '(' and a line ending in '{' or
# an open parameter list; a bare call is not a declaration)
DECL = [re.compile(p) for p in (
    r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", r"^\s*class\s+([A-Za-z_]\w*)",
    r"\bfunction\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(",
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|[\w$]+\s*=>)",
    r"^\s*(?:[\w$]+\.)*([A-Za-z_$][\w$]*)\s*[:=]\s*(?:async\s+)?function\b",
    r"^\s+(?:(?:async|static|get|set|public|private|protected)\s+)*([A-Za-z_$][\w$]*)\s*\([^()]*\)\s*(?::\s*[^{;=]+)?\{\s*$",
    r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)",
    r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|synchronized|override|open|suspend|inline|"
    r"async|virtual|readonly|export|default|get|set)\s+)*(?:(?:fun|function)\s+(?:<[^>]*>\s*)?|(?:<[^>]*>\s*)?"
    r"[\w$<>\[\],.?]+\s+)([A-Za-z_$][\w$]*)\s*\((?:[^;]*(?:\{|,|\())?\s*$",
    r"^\s*(?:(?:public|private|protected|internal|abstract|final|open|data|sealed|export|default)\s+)*"
    r"(?:class|interface|object|trait|struct|enum)\s+([A-Za-z_]\w*)")]


def decl_name(line: str) -> str | None:
    """The name a line declares (a function, method or class), or None. Control-flow and comment lines are not
    declarations."""
    if re.match(r"^\s*(?://|#|\*|/\*|--)", line):
        return None
    if re.match(r"^\s*(?:\}\s*)?(?:if|for|while|switch|catch|return|else|elif|try|with|case|foreach|elseif|do|"
                r"throw|new|await|yield|assert|raise|print|echo)\b", line):
        return None
    for rx in DECL:
        m = rx.search(line)
        if m and m.group(1) not in _KW and len(m.group(1)) >= 3:
            return m.group(1)
    return None


def _indent(s: str) -> int:
    return len(s) - len(s.lstrip(" \t"))


def changed_old_lines(section: str) -> list[int]:
    """1-based line numbers in the old file that a diff section changes: removed lines, and for added lines the old
    line just above the insertion."""
    out, old = [], 0
    for line in section.splitlines():
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if m:
            old = int(m.group(1)) - 1
            continue
        if old < 0 or line.startswith(("diff --git", "index ", "--- ", "+++ ", "\\")):
            continue
        if line.startswith("-"):
            old += 1
            out.append(old)
        elif line.startswith("+"):
            out.append(max(1, old))
        elif line.startswith(" ") or line == "":
            old += 1
    return sorted(set(out))


def enclosing(old_lines: list[str], n: int) -> str | None:
    """The declaration enclosing old line n (1-based): the line itself if it declares something, else the nearest
    line above it that declares something and is indented less than the nearest non-blank line at or above n."""
    if not 1 <= n <= len(old_lines):
        return None
    own = decl_name(old_lines[n - 1])
    if own:
        return own
    ref = next((old_lines[i] for i in range(n - 1, -1, -1) if old_lines[i].strip()), "")
    lim = _indent(ref) + (1 if ref.strip()[:1] in ")]}" else 0)   # a closing line belongs to the block it closes
    for i in range(n - 2, -1, -1):
        ln = old_lines[i]
        if ln.strip() and _indent(ln) < lim:
            nm = decl_name(ln)
            if nm:
                return nm
            if re.search(r"\w", ln):                                  # a line of braces alone does not open a scope
                lim = _indent(ln) + (1 if ln.strip()[:1] in ")]}" else 0)
    return None


def gold_functions(patch: str, gold: list[str], old_texts: dict[str, str]) -> list[str]:
    """Functions, methods or classes the fix changes in its gold files: for every changed old line, its enclosing
    declaration in the parent's file (``enclosing``), plus names declared on added lines. Names shorter than 3
    characters and keywords are dropped. Approximate by design (the README says so)."""
    names: list[str] = []
    for sec in re.split(r"(?m)^(?=diff --git )", patch_sections(patch, set(gold))):
        m = re.match(r"diff --git a/(\S+) b/(\S+)", sec)
        if not m:
            continue
        old = (old_texts.get(m.group(1)) or "").splitlines()
        found = [enclosing(old, n) for n in changed_old_lines(sec)]
        found += [decl_name(l[1:]) for l in sec.splitlines() if l.startswith("+") and not l.startswith("+++")]
        for x in found:
            if x and x not in names:
                names.append(x)
    return names


# --------------------------------------------------------------------------- answers and scores (pure)
FINDING = re.compile(r"^\s*(?:[-*]\s*|\d+[.)]\s*)?\**\s*FINDING\s*\**\s*\d*\s*:\s*\**\s*(.+?)\s*$", re.I)


def norm_file(p: str, cwd: str | None, repo_paths: set[str] | None) -> str:
    p = RE.norm_path(p.strip().strip("`"), cwd)
    p = re.sub(r"[#@].*$", "", p).strip()
    if repo_paths is not None and p not in repo_paths and "/" in p and p.split("/", 1)[1] in repo_paths:
        p = p.split("/", 1)[1]                       # a leading repository-name directory
    return p


def parse_findings(text: str, cwd: str | None = None, repo_paths: set[str] | None = None) -> list[dict]:
    """``FINDING: file | where | class | why`` lines -> up to 5 findings in order (file normalised; a line with fewer
    fields keeps what it has). Duplicate lines count once."""
    out = []
    for line in (text or "").splitlines():
        m = FINDING.match(line)
        if not m:
            continue
        f = [x.strip().strip("`*").strip() for x in m.group(1).split("|")]
        f += [""] * (4 - len(f))
        d = {"file": norm_file(f[0], cwd, repo_paths), "where": f[1], "cls": f[2], "why": " | ".join(f[3:]).strip(" |")}
        if any(d == o for o in out):
            continue
        out.append(d)
        if len(out) == 5:
            break
    return out


def score(findings: list[dict], gold: list[str], funcs: list[str], families: set[str],
          repo_paths: set[str] | None = None) -> dict:
    """Primary: a gold file named in the top 5 findings. Also: the first finding's file is gold; a gold function named
    (whole token) in a finding that names a gold file; class match = the first finding naming a gold file has a
    weakness class in the advisory's CWE families; paths not in the repository."""
    top = findings[:5]
    hits = [f for f in top if f["file"] in gold]
    named = []
    for f in hits:
        text = f"{f['where']} {f['why']}"
        named += [fn for fn in funcs if fn not in named and re.search(rf"(?<![\w$]){re.escape(fn)}(?![\w$])", text)]
    first_hit = hits[0] if hits else None
    return {"gold_top5": bool(hits), "gold_first": bool(top) and top[0]["file"] in gold,
            "gold_rank": next((i + 1 for i, f in enumerate(top) if f["file"] in gold), None),
            "func_named": bool(named), "funcs_named": named,
            "class_match": bool(first_hit) and bool(class_families(first_hit["cls"]) & families),
            "n_findings": len(top),
            "not_in_repo": [f["file"] for f in top if repo_paths is not None and f["file"] not in repo_paths]}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar test on paired booleans: b01 = a false and b true, b10 = a true and b false."""
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    b01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return {"a_only": b10, "b_only": b01, "p": 1.0}
    k = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": b10, "b_only": b01, "p": p}


def image_tokens(sizes) -> int:
    return sum(-(-w // 28) * -(-h // 28) for w, h in sizes)


def prompt_text(arm: str, *, repo: str, sha: str, paths: list[str] | None = None, index: str = "") -> str:
    """The text part of each arm's (first) message; the atlas arms put the page images before it."""
    n = len(paths or [])
    if arm == "closed":
        return CLOSED_HEAD.format(repo=repo, sha=sha, n=n, paths="\n".join(paths or [])) + "\n\n" + INSTRUCTION
    if arm == "atlas":
        return ATLAS_HEAD.format(repo=repo, sha=sha, n=n, index=index.strip()) + "\n\n" + INSTRUCTION
    if arm == "agent":
        return AGENT_HEAD.format(repo=repo, sha=sha) + "\n\n" + INSTRUCTION
    if arm == "agent_atlas":
        return (ATLAS_HEAD.format(repo=repo, sha=sha, n=n, index=index.strip()) + "\n\n" + AGENT_HEAD.format(repo=repo, sha=sha)
                + " " + AGENT_ATLAS_NOTE + "\n\n" + INSTRUCTION)
    raise ValueError(arm)


# --------------------------------------------------------------------------- git helpers
def git(*args, cwd=None, check=True, timeout=600, binary=False):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=not binary, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args[:3])}: {(p.stderr if not binary else p.stderr.decode())[-300:]}")
    return p.stdout


_repo_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def repo_lock(repo: str) -> threading.Lock:
    with _locks_lock:
        return _repo_locks.setdefault(repo, threading.Lock())


def fetch_fix(work: Path, repo: str, sha: str) -> dict:
    """Trees (no blobs) of the fix commit and its parents; -> {sha, parent, date, changed}."""
    d = work / "git" / repo.replace("/", "__")
    with repo_lock(repo):
        if not (d / ".git").exists():
            d.mkdir(parents=True, exist_ok=True)
            git("init", "-q", cwd=d)
            git("remote", "add", "origin", f"https://github.com/{repo}.git", cwd=d)
        if len(sha) < 40:
            raise RuntimeError("fix commit given as a short sha")
        git("fetch", "-q", "--depth", "2", "--filter=blob:none", "origin", sha, cwd=d, timeout=900)
    parents = git("rev-list", "--parents", "-n", "1", sha, cwd=d).split()[1:]
    if not parents:
        raise RuntimeError("fix commit has no parent")
    changed = git("diff", "--name-only", "--no-renames", parents[0], sha, cwd=d).split("\n")
    date = git("log", "-1", "--format=%cI", sha, cwd=d).strip()
    return {"sha": sha, "parent": parents[0], "date": date, "changed": [c for c in changed if c]}


def atlas_paths_at(work: Path, repo: str, parent: str) -> list[str]:
    d = work / "git" / repo.replace("/", "__")
    paths = git("ls-tree", "-r", "--name-only", parent, cwd=d).splitlines()
    return [p for p in paths if RA.wanted(p, False) and detect(p) is not None]


def snapshot(work: Path, repo: str, parent: str) -> Path:
    """A full checkout of the parent commit (depth 1), for measuring and building the atlas."""
    d = work / "snap" / f"{repo.replace('/', '__')}@{parent[:12]}"
    if (d / ".done").exists():
        return d
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    git("init", "-q", cwd=d)
    git("fetch", "-q", "--depth", "1", f"https://github.com/{repo}.git", parent, cwd=d, timeout=1800)
    git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD", cwd=d, timeout=900)
    (d / ".done").write_text("")
    return d


# --------------------------------------------------------------------------- select (no model calls)
def load_advisories(adb: Path) -> list[dict]:
    advs = []
    for f in sorted((adb / "advisories" / "github-reviewed" / "2026").glob("*/*/*.json")):
        advs.append(parse_advisory(json.loads(f.read_text())))
    return advs


def select(work: Path, adb: Path, workers: int = 6) -> dict:
    """Rules 1-5 in order (see the README): published from SINCE, newest first; not withdrawn; exactly one fix commit
    on GitHub, fetchable, committed on or after SINCE; the fix changes 1-5 counted source files, all with a minifier;
    at most PER_REPO per repository; at least one gold file exists at the parent and every one that does is in the
    atlas's file set (repo_atlas.collect); the atlas at the parent is MIN_TOKENS-MAX_TOKENS image tokens. Stop at MAX_TASKS tasks or MAX_EXAMINED advisories examined."""
    advs = order_advisories(load_advisories(adb))[:MAX_EXAMINED]
    pre: dict[str, dict] = {}

    def stage1(a):
        if a.get("withdrawn"):
            return {"why": "withdrawn"}
        if len(a["commits"]) != 1:
            return {"why": f"{'no' if not a['commits'] else 'more than one'} fix commit on GitHub"}
        repo, sha = a["commits"][0]
        try:
            fx = fetch_fix(work, repo, sha)
        except Exception as e:  # noqa: BLE001
            return {"why": "fix commit not fetchable", "detail": str(e)[:200], "repo": repo}
        if fx["date"][:10] < SINCE:
            return {"why": f"fix commit before {SINCE}", "repo": repo, **fx}
        why, gold = classify_fix(fx["changed"])
        if why:
            return {"why": why, "repo": repo, "gold": gold, **fx}
        return {"why": None, "repo": repo, "gold": gold, **fx}

    with ThreadPoolExecutor(workers) as ex:
        for a, r in zip(advs, ex.map(stage1, advs)):
            pre[a["id"]] = r
            print(f"[sel] {a['published'][:10]} {a['id']} {r.get('repo', '')}: {r['why'] or 'pass 1-3'}", flush=True)
    tasks, funnel, per_repo, examined = [], [], {}, 0
    for a in advs:
        if len(tasks) >= MAX_TASKS:
            break
        examined += 1
        r = pre[a["id"]]
        ent = {"advisory": a["id"], "published": a["published"], "cwes": a["cwes"], "ecosystems": a["ecosystems"],
               "repo": r.get("repo"), "fix": r.get("sha"), "parent": r.get("parent"), "fix_date": r.get("date"),
               "changed": r.get("changed"), "gold": r.get("gold")}
        why = r["why"]
        if why is None and per_repo.get(r["repo"], 0) >= PER_REPO:
            why = f"repository already has {PER_REPO} tasks"
        if why is None:
            cand = atlas_paths_at(work, r["repo"], r["parent"])
            if not [g for g in r["gold"] if g in cand]:
                why = "a gold file is outside the atlas's file set"
            elif len(cand) > MAX_PATHS_PREFILTER:
                why, ent["atlas_paths"] = "atlas too large (path count)", len(cand)
        if why is None:
            try:
                snap = snapshot(work, r["repo"], r["parent"])
                files, _ = RA.collect(snap, False, 10)
                st = RA.stats(snap, 88.2, False, 10)
            except Exception as e:  # noqa: BLE001
                why, ent["detail"] = "parent commit not checkable", str(e)[:200]
            else:
                ent["atlas_image_tokens"] = st["image_tokens"]
                ent["atlas_files"] = st["files"]
                in_atlas = {f["path"] for f in files}
                at_parent = [g for g in r["gold"] if (snap / g).is_file()]
                if not at_parent or any(g not in in_atlas for g in at_parent):
                    why = "a gold file is outside the atlas's file set"
                elif st["image_tokens"] < MIN_TOKENS:
                    why = "atlas too small"
                elif st["image_tokens"] > MAX_TOKENS:
                    why = "atlas too large"
                if why:
                    shutil.rmtree(snap, ignore_errors=True)
        ent["why"] = why
        funnel.append(ent)
        if why is None:
            per_repo[r["repo"]] = per_repo.get(r["repo"], 0) + 1
            tasks.append({**ent, "id": f"{r['repo'].replace('/', '__')}@{r['parent'][:8]}"})
            print(f"[sel] TASK {len(tasks)}: {a['id']} {r['repo']} gold={r['gold']} atlas={ent['atlas_image_tokens']:,}",
                  flush=True)
    counts: dict[str, int] = {}
    for f in funnel:
        counts[f["why"] or "selected"] = counts.get(f["why"] or "selected", 0) + 1
    sel = {"rule": select.__doc__.strip(), "since": SINCE, "examined": examined,
           "date_range": [funnel[-1]["published"] if funnel else None, funnel[0]["published"] if funnel else None],
           "counts": counts, "tasks": tasks, "funnel": funnel}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel


def selection() -> list[dict]:
    return json.loads((OUT / "selection.json").read_text())["tasks"]


# --------------------------------------------------------------------------- prepare (no model calls)
def private_dir(work: Path) -> Path:
    """Fix patches live next to the work dir, not in it (the agents' exports are inside the work dir)."""
    return work.parent / (work.name + "_private")


def refresh_funcs(work: Path, meta: dict) -> dict:
    """Gold functions from the saved fix patch and the parent's files in the export (re-run after a rule fix)."""
    patch = (private_dir(work) / f"{meta['id']}.diff").read_text()
    exp = Path(meta["export"])
    old = {g: (exp / g).read_text(encoding="utf-8", errors="replace") for g in meta["gold"] if (exp / g).is_file()}
    meta["funcs"] = gold_functions(patch, meta["gold"], old)
    (work / "ctx" / meta["id"] / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def prepare_task(work: Path, t: dict) -> dict:
    """Export (git archive of the parent, no .git), atlas, the closed list, gold functions, repository paths."""
    t_dir = work / "ctx" / t["id"]
    meta_p = t_dir / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())
    snap = snapshot(work, t["repo"], t["parent"])
    files, uncovered = RA.collect(snap, False, 10)
    paths = [f["path"] for f in files]
    t_dir.mkdir(parents=True, exist_ok=True)
    am = json.loads((t_dir / "atlas" / "index.json").read_text()) if (t_dir / "atlas" / "index.json").exists() \
        else RA.build(snap, t_dir / "atlas", 88.2, False, 10)
    exp = work / "export" / t["id"] / t["repo"].split("/")[1]
    if not exp.exists():
        tmp = exp.with_name(exp.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        arch = git("archive", t["parent"], cwd=snap, binary=True)
        subprocess.run(["tar", "-x", "-C", str(tmp)], input=arch, check=True)
        tmp.rename(exp)
    all_paths = git("ls-files", cwd=snap).splitlines()
    d = work / "git" / t["repo"].replace("/", "__")
    patch = git("diff", "--no-renames", t["parent"], t["fix"], "--", *t["gold"], cwd=d, timeout=900)
    old = {g: (snap / g).read_text(encoding="utf-8", errors="replace") for g in t["gold"] if (snap / g).is_file()}
    funcs = gold_functions(patch, t["gold"], old)
    meta = {"id": t["id"], "repo": t["repo"], "parent": t["parent"], "files": len(paths), "paths": paths,
            "uncovered": len(uncovered), "pages": len(am["pages"]), "image_tokens": am["image_tokens"],
            "page_sizes": [p["size"] for p in am["pages"]], "atlas_dir": str(t_dir / "atlas"), "export": str(exp),
            "gold": [g for g in t["gold"] if g in all_paths], "gold_new": [g for g in t["gold"] if g not in all_paths],
            "gold_in_atlas": [g for g in t["gold"] if g in paths], "funcs": funcs,
            "families": sorted({cwe_family(c) for c in t["cwes"]} - {"other"}) or ["other"],
            "patch_lines": sum(1 for l in patch.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---")))}
    (t_dir / "repo_paths.json").write_text(json.dumps(all_paths))
    private_dir(work).mkdir(parents=True, exist_ok=True)
    (private_dir(work) / f"{t['id']}.diff").write_text(patch)
    meta_p.write_text(json.dumps(meta, indent=1))
    (OUT / "index").mkdir(parents=True, exist_ok=True)
    shutil.copy(t_dir / "atlas" / "index.txt", OUT / "index" / f"{t['id']}.txt")
    return meta


def move_secrets(work: Path) -> None:
    """After prepare: the fix commits (partial clones) and gold patches leave the tree the agents can reach."""
    shutil.rmtree(work / "git", ignore_errors=True)
    shutil.rmtree(work / "snap", ignore_errors=True)


# --------------------------------------------------------------------------- spend, rows
_lock, _alock = threading.Lock(), threading.Lock()
_stop = threading.Event()
_LIMIT = re.compile(r"hit your .{0,20}limit|usage limit|limit reached|resets? at|out of (?:extra )?usage", re.I)


def jl(name: str) -> list[dict]:
    p = OUT / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def append(name: str, row: dict) -> None:
    with _alock:
        with open(OUT / name, "a") as f:
            f.write(json.dumps(row) + "\n")


def spent() -> float:
    return sum(r.get("cost_usd") or 0 for r in jl("spend.jsonl"))


def book(what: str, rec: dict) -> None:
    if not rec.get("cached") and rec.get("cost_usd"):
        append("spend.jsonl", {"what": what, "cost_usd": rec["cost_usd"], "input_tokens": rec.get("input_tokens"),
                               "error": (rec.get("error") or None) and str(rec["error"])[:300],
                               "t": time.strftime("%Y-%m-%dT%H:%M:%S")})


def limit_hit(err: str | None) -> bool:
    if err and _LIMIT.search(err):
        if not _stop.is_set():
            print(f"[sec] USAGE LIMIT: {err[:300]} -- stopping, nothing more is launched", flush=True)
        _stop.set()
        return True
    return False


def done_pairs() -> set:
    return {(r["id"], r["arm"]) for r in jl("runs.jsonl")}


# --------------------------------------------------------------------------- arms
def pngs_of(meta: dict) -> list[bytes]:
    return [(Path(meta["atlas_dir"]) / f"page_{i + 1:03d}.png").read_bytes() for i in range(meta["pages"])]


def one_shot(arm: str, t: dict, meta: dict) -> dict:
    from PIL import Image
    from vtc import claude_eval as CE
    index = (Path(meta["atlas_dir"]) / "index.txt").read_text()
    prompt = prompt_text(arm, repo=t["repo"], sha=t["parent"][:7], paths=meta["paths"], index=index)
    imgs = [Image.open(Path(meta["atlas_dir"]) / f"page_{i + 1:03d}.png") for i in range(meta["pages"])] \
        if arm == "atlas" else []
    rec = CE.transcribe(imgs, SYSTEM, prompt, model=MODEL, effort=EFFORT, hires=True, timeout=1500.0, max_retries=2)
    rec["seconds"] = rec.get("latency_s")
    rec["turns"] = 1
    rec["first_turn_input"] = rec.get("input_tokens")
    return rec


def agent_run(arm: str, t: dict, meta: dict, work: Path) -> dict:
    from vtc import claude_eval as CE
    index = (Path(meta["atlas_dir"]) / "index.txt").read_text()
    text = prompt_text(arm, repo=t["repo"], sha=t["parent"][:7], paths=meta["paths"], index=index)
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(b).decode("ascii")}}
               for b in (pngs_of(meta) if arm == "agent_atlas" else [])]
    content.append({"type": "text", "text": text})
    msg = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
    cwd = meta["export"]
    cmd = [CE.CLAUDE_BIN, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", MODEL, "--effort", EFFORT, "--tools", "Read,Grep,Glob",
           "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
           "--max-turns", str(AGENT_TURNS), "--max-budget-usd", f"{AGENT_BUDGET_USD:.2f}"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, input=msg, capture_output=True, text=True, env=CE.build_env(), cwd=cwd, timeout=2400)
        stdout, rc, err = p.stdout, p.returncode, p.stderr
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc, err = -9, "timeout"
    secs = time.time() - t0
    sdir = work / "streams"
    sdir.mkdir(exist_ok=True)
    (sdir / f"{t['id']}__{arm}.jsonl").write_text(stdout)
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
    outside = [c["input"] for c in calls if any(isinstance(v, str) and ((v.startswith("/") and not v.startswith(cwd))
                                                                     or re.search(r"(^|/)\.\.(/|$)", v))
                                                 for v in (c.get("input") or {}).values())]
    cost = float((res or {}).get("total_cost_usd") or 0.0)
    if res is None or (res.get("is_error") and res.get("subtype") not in ("error_max_turns", "error_max_budget_usd")):
        return {"error": f"rc={rc} {str((res or {}).get('result'))[:300]} {err[-300:]}", "cost_usd": cost}
    u = res.get("usage") or {}
    out = res.get("result") or ""
    if not out or not FINDING.search(out):                 # max turns / budget: the last text that has findings
        texts = [c["text"] for e in asst for c in e["message"]["content"] if c.get("type") == "text"]
        withf = [x for x in texts if any(FINDING.match(l) for l in x.splitlines())]
        out = withf[-1] if withf else (out or (texts[-1] if texts else ""))
    mu = res.get("modelUsage") or {}
    return {"text": out, "cost_usd": cost, "seconds": round(secs, 1), "turns": res.get("num_turns"),
            "subtype": res.get("subtype"), "stop_reason": res.get("stop_reason"),
            "input_tokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0), "first_turn_input": first_in,
            "usage": {k: u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                                               "output_tokens")},
            "tools": {k: names.count(k) for k in sorted(set(names))}, "tool_calls": len(calls),
            "files_read": sorted({str((c.get("input") or {}).get("file_path", "")).replace(cwd + "/", "")
                                  for c in calls if c["name"] == "Read"}),
            "outside_cwd": outside[:10], "permission_denials": [str(d)[:200] for d in res.get("permission_denials") or []][:10],
            "model": max(mu, key=lambda m: mu[m].get("outputTokens", 0)) if mu else MODEL, "cached": False}


def run_pair(work: Path, t: dict, arm: str) -> None:
    if _stop.is_set():
        return
    meta = prepare_task(work, t)
    rec = one_shot(arm, t, meta) if arm in ("closed", "atlas") else agent_run(arm, t, meta, work)
    book(f"{t['id']} {arm}", rec)
    if rec.get("error"):
        limit_hit(rec["error"])
        print(f"[sec] ERROR {t['id']} {arm}: {str(rec['error'])[:300]}", flush=True)
        return
    repo_paths = set(json.loads((work / "ctx" / t["id"] / "repo_paths.json").read_text()))
    cwd = meta["export"] if arm.startswith("agent") else None
    fnd = parse_findings(rec.get("text", ""), cwd, repo_paths)
    sc = score(fnd, meta["gold"], meta["funcs"], set(meta["families"]), repo_paths)
    tok = image_tokens(meta["page_sizes"]) if arm in ("atlas", "agent_atlas") else 0
    arrived = tok == 0 or (rec.get("first_turn_input") or 0) >= PAGE_ARRIVAL * tok
    row = {"id": t["id"], "repo": t["repo"], "arm": arm, "gold": meta["gold"], "funcs": meta["funcs"],
           "families": meta["families"], "findings": fnd, **sc, "no_findings": not fnd,
           "image_tokens_formula": tok, "pages_arrived": arrived,
           **{k: rec.get(k) for k in ("model", "input_tokens", "output_tokens", "first_turn_input", "cost_usd", "seconds",
                                      "turns", "subtype", "stop_reason", "tools", "tool_calls", "files_read", "outside_cwd",
                                      "permission_denials", "cached", "cache_key", "usage", "thinking_tokens")},
           "answer": rec.get("text", "")}
    append("runs.jsonl", row)
    print(f"[sec] {t['id'][:40]:40} {arm:11} top5={row['gold_top5']!s:5} first={row['gold_first']!s:5} "
          f"func={row['func_named']!s:5} cls={row['class_match']!s:5} n={row['n_findings']} turns={row['turns']} "
          f"in={row['input_tokens'] or 0:,} first_in={row['first_turn_input'] or 0:,} ${row['cost_usd'] or 0:.3f} "
          f"{row['seconds'] or 0:.0f}s{'' if arrived else ' PAGES DROPPED'}", flush=True)


def task_bound(meta: dict, rows: list[dict]) -> float:
    """A conservative estimate of one whole task (four arms) at list price, from the rows so far: the closed and agent
    arms at their largest cost so far, the atlas arms scaled per image token at their largest rate so far (before
    any rows: $0.15 + $1.5 + $8/M image tokens x 2)."""
    tok = meta["image_tokens"]
    def worst(arm, per_token):
        rs = [r for r in rows if r["arm"] == arm and r.get("cost_usd")]
        if not rs:
            return None
        if per_token:
            return max(r["cost_usd"] / max(1, r["image_tokens_formula"]) for r in rs) * tok
        return max(r["cost_usd"] for r in rs)
    c = worst("closed", False) or 0.15
    a = worst("atlas", True) or tok * 8e-6
    g = worst("agent", False) or 1.5
    rs = [r for r in rows if r["arm"] == "agent_atlas" and r.get("cost_usd")]
    ga = max((r["cost_usd"] - 0.0) for r in rs) * max(1.0, tok / max(r["image_tokens_formula"] for r in rs)) if rs \
        else g + tok * 8e-6
    return round(1.15 * (c + a + g + ga), 2)


def run_order(tasks: list[dict]) -> list[dict]:
    """The order tasks are run in (fixed before any call): round-robin over repositories, so a capped budget covers as
    many repositories as it can; each round in ascending atlas size (ties: selection order), a repository's tasks in
    ascending atlas size."""
    rank = {t["id"]: i for i, t in enumerate(tasks)}
    key = lambda t: (t["atlas_image_tokens"], rank[t["id"]])  # noqa: E731
    by: dict[str, list[dict]] = {}
    for t in sorted(tasks, key=key):
        by.setdefault(t["repo"], []).append(t)
    out, k = [], 0
    while any(len(v) > k for v in by.values()):
        out += sorted((v[k] for v in by.values() if len(v) > k), key=key)
        k += 1
    return out


def run(work: Path, n_tasks: int, cap: float, workers: int) -> None:
    tasks = run_order(selection())[:n_tasks]
    for t in tasks:
        if _stop.is_set():
            break
        todo = [a for a in ARMS if (t["id"], a) not in done_pairs()]
        if not todo:
            continue
        meta = prepare_task(work, t)
        bound = task_bound(meta, jl("runs.jsonl"))
        if spent() + bound > cap:
            print(f"[sec] STOP before {t['id']}: spent ${spent():.2f} + task bound ${bound:.2f} > cap ${cap}", flush=True)
            break
        print(f"[sec] task {t['id']} ({meta['image_tokens']:,} image tokens): {todo}; spent ${spent():.2f}, bound "
              f"${bound:.2f}", flush=True)
        # the long agent runs first, the two one-shot calls fill in
        order = [a for a in ("agent_atlas", "agent", "atlas", "closed") if a in todo]
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda a: run_pair(work, t, a), order))
    print(f"[sec] spent ${spent():.2f} list price", flush=True)


# --------------------------------------------------------------------------- analysis
def analyse() -> str:
    import statistics as st
    sel = json.loads((OUT / "selection.json").read_text())
    tasks = sel["tasks"]
    rows = {}
    for r in jl("runs.jsonl"):
        rows[(r["id"], r["arm"])] = r
    done = [t["id"] for t in tasks if all((t["id"], a) in rows for a in ARMS)]
    L = [f"Tasks with all four arms: {len(done)}.", "",
         "| arm | tasks | gold file in top 5 | gold file first | gold function named | class match | findings (mean) | "
         "mean input tokens | mean cost | median turns | median seconds | wrong paths | pages dropped |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for arm in ARMS:
        rs = [rows[(i, arm)] for i in done]
        if not rs:
            continue
        n = len(rs)
        summ[arm] = {k: sum(bool(r[k]) for r in rs) for k in ("gold_top5", "gold_first", "func_named", "class_match")}
        L.append(f"| {arm} | {n} | {summ[arm]['gold_top5']} | {summ[arm]['gold_first']} | {summ[arm]['func_named']} | "
                 f"{summ[arm]['class_match']} | {st.mean(r['n_findings'] for r in rs):.1f} | "
                 f"{st.mean(r['input_tokens'] or 0 for r in rs):,.0f} | ${st.mean(r['cost_usd'] or 0 for r in rs):.3f} | "
                 f"{st.median(r['turns'] or 1 for r in rs):g} | {st.median(r['seconds'] or 0 for r in rs):.0f} | "
                 f"{sum(len(r['not_in_repo']) for r in rs)} | {sum(not r['pages_arrived'] for r in rs)} |")
    L += ["", "| pair (a vs b) | a only | b only | exact McNemar p |", "|---|---|---|---|"]
    tests = {}
    for a, b in (("atlas", "closed"), ("agent_atlas", "agent"), ("agent", "atlas"), ("agent", "closed"),
                 ("agent_atlas", "closed")):
        if done:
            m = mcnemar([rows[(i, a)]["gold_top5"] for i in done], [rows[(i, b)]["gold_top5"] for i in done])
            tests[f"{a}-{b}"] = m
            L.append(f"| {a} vs {b} | {m['a_only']} | {m['b_only']} | {m['p']:.3f} |")
    if done:
        d = [(rows[(i, "agent_atlas")]["turns"] or 0) - (rows[(i, "agent")]["turns"] or 0) for i in done]
        tests["turns_agent_atlas_minus_agent"] = {"median": st.median(d), "fewer": sum(x < 0 for x in d),
                                                  "more": sum(x > 0 for x in d)}
        L += ["", f"Turns, agent_atlas − agent (paired): median {st.median(d):+g}; fewer on {sum(x < 0 for x in d)}, "
              f"more on {sum(x > 0 for x in d)}, equal on {sum(x == 0 for x in d)} tasks."]
    L += ["", "| task | repository | CWE (family) | gold files | atlas tokens | closed | atlas | agent | agent_atlas |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in tasks:
        cells = []
        for arm in ARMS:
            r = rows.get((t["id"], arm))
            if r is None:
                cells.append("not run")
                continue
            c = (f"#{r['gold_rank']}" if r["gold_top5"] else "miss") + (" fn" if r["func_named"] else "") + \
                (" cls" if r["class_match"] else "")
            if r["no_findings"]:
                c = "no findings" + (" (turn limit)" if r.get("subtype") == "error_max_turns" else "")
            cells.append(c)
        fams = ",".join(sorted({cwe_family(c) for c in t["cwes"]}))
        L.append(f"| {t['id']} | {t['repo']} | {', '.join(t['cwes']) or 'none'} ({fams}) | {len(t['gold'])} | "
                 f"{t['atlas_image_tokens']:,} | " + " | ".join(cells) + " |")
    sp = jl("spend.jsonl")
    by = {}
    for r in sp:
        k = r["what"].split()[-1] if r["what"].split()[-1] in ARMS else "other"
        by[k] = by.get(k, 0) + r["cost_usd"]
    L += ["", "| part | spent (list price) |", "|---|---|"] + [f"| {k} | ${v:.2f} |" for k, v in sorted(by.items())] + \
         [f"| **total** | **${sum(by.values()):.2f}** |"]
    (OUT / "summary.json").write_text(json.dumps({"done": done, "arms": summ, "tests": tests, "spend": by}, indent=1))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["select", "prepare", "run", "analyse"])
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--adb", type=Path, default=None, help="advisory-database checkout (default: WORK/adb)")
    ap.add_argument("--tasks", type=int, default=MAX_TASKS, help="run the first n selected tasks")
    ap.add_argument("--cap", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "select":
        sel = select(a.work, a.adb or a.work / "adb")
        print(json.dumps(sel["counts"], indent=1), "examined", sel["examined"])
    elif a.cmd == "prepare":
        sizes = {}
        for t in selection():
            m = refresh_funcs(a.work, prepare_task(a.work, t))
            sizes[t["id"]] = {k: m[k] for k in ("files", "pages", "image_tokens", "gold", "gold_new", "gold_in_atlas",
                                                "funcs", "families", "patch_lines")}
            print(t["id"], json.dumps(sizes[t["id"]]), flush=True)
        (OUT / "contexts.json").write_text(json.dumps(sizes, indent=1))
        move_secrets(a.work)
    elif a.cmd == "run":
        run(a.work, a.tasks, a.cap, a.workers)
    else:
        print(analyse())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
