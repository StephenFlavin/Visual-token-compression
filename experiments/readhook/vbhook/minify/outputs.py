"""Command and tool output minifiers: pytest, Python tracebacks, git (diff/show/log -p, log, status), grep/rg,
compiler and linter diagnostics, generic logs and directory listings (ls -l, ls, tree, du; find/fd/rg --files).

The hook knows the command line: ``kind_for_command(cmd)`` maps it to a kind (None: sniff the text). Every kind
handles any text (unknown lines take the generic path), so a wrong guess costs density, never correctness.

Conventions on top of base.py (``LEGENDS[kind]`` says them in one sentence for the hook's note):

* Generic line layout (pytest, traceback, diagnostics, log, listing): an unindented line is a record and gets
  ``§<n> `` (n = its line number in the output); an indented line is joined untagged with one ``»`` per indentation
  level (an indent stack, as Python's tokenizer does), so every line still starts with a visible marker. A gutter
  line (``  12 |  code``, ``   |``) is indented to its ``|``. Whitespace runs inside a line become one space, except
  in pytest ``E`` lines (assertion messages and diffs: kept as is).
* Caret lines (Python 3.11+ tracebacks, pytest, gcc/clang, rustc/ruff, mypy, tsc --pretty, javac): a line of
  ``^``/``~`` under a source line marks columns, which a reflowed proportional page cannot show. The marked span is
  bracketed inside the source line instead: ``⟦x⟧`` = carets ``^``, ``⟨x⟩`` = tildes ``~``, and
  ``⟦a ⟦+⟧ b⟧`` = tildes with carets under the inner part (``~~^~~``). A caret line's gutter ``|`` stays after the
  source line and its label (rustc ``^^^ expected i32``) becomes the next, indented item. ``expand()`` rebuilds the
  caret line. Caret lines that do not fit this (several runs, past the line end, tabs) are kept as ordinary lines.
* pytest / log / diagnostics / traceback ``normalise``: separator lines (``===== FAILURES =====``, ``_____ test_x
  _____``, ``-----``, ``_ _ _ _``) keep 3 characters per run; ANSI/OSC codes and progress text overwritten by ``\\r``
  are dropped (every kind).
* gitdiff: hunk lines are tagged with their NEW file line number (removed lines: their OLD line number), the
  ``+``/``-`` marker stays first, code indentation becomes ``»`` per level after it; hunk headers are kept exactly;
  file/commit headers are untagged; commit message and --stat lines are ``»`` lines.
* gitlog: one tag per entry (``commit`` line, or each line of --oneline/--format output); Author/Date join the entry,
  message lines are ``»`` lines. gitstatus: one tag per entry; short-format status codes show a blank as ``.``
  (git's porcelain v2 notation), restored by ``expand()``.
* grep: ``»path §<line> content`` (path once per run of matches; ``§<line>- content`` = a -A/-B/-C context line;
  ``»--`` = group separator; other lines are their own ``»`` group). ``expand()`` restores ``path:line:content``.
* find: ``§<n> dir/ name name ...`` for consecutive entries of one directory (``expand()`` restores the paths);
  disabled when any path contains whitespace.
* listing: ``ls -l`` entries and ``tree`` rows are self-delimiting (permission string, tree connector) and untagged;
  tree indentation units become ``│`` / ``»``.

Measured 2026-11-11 on 101 real outputs (505k chars; most are embedded in tests/test_outputs.py, each capped at
30k chars): text tokens counted with the token-counting call of SPEC.md, page tokens = ``render.page_tokens(flow)``
at 88.2 px²/char. save = text / page tokens summed over the samples as single tool results (small samples pay the
~138-token minimum image); large = the same for outputs of many pages; BE = raw length (chars) below which text is
cheaper (image <= text), BE 0.8 = with the hook's 0.8 margin; tags = share of the flow that is ``§n`` tags; every /
none = savings with a tag on every line / with no tags at all (line starts then rest on » and self-delimiting
prefixes only; for gitdiff no tags = no line numbers).

    kind         n  cpt   flow/raw  save   large  BE   BE 0.8  tags   every  none
    pytest       13 2.16  0.76      4.79x  5.3x   300  380     6.0%   4.63   5.02
    log          10 1.69  0.99      4.86x  5.1x   240  300     4.1%   4.84   5.10
    traceback    14 1.90  0.88      2.03x  5.2x   270  330     1.4%   2.03   2.03
    diagnostics  17 2.10  0.97      3.51x  4.3x   290  370     2.8%   3.49   3.60
    gitdiff      16 2.20  1.02      3.37x  3.8x   310  390     5.5%   3.37   3.52
    gitlog        9 2.22  1.00      3.51x  3.9x   310  390     1.7%   3.44   3.51
    gitstatus     4 1.95  1.10      0.92x  4.0x   270  340    12.3%   (all samples < 600 chars)
    grep          9 2.12  0.81      4.10x  5.0x   300  370     -      (tags are the line numbers)
    find          3 2.00  0.54      3.93x  8.0x   280  350     -
    listing       6 1.65  0.95      4.30x  5.5x   230  290     1.8%   3.83   4.30

Tags on records cost 1.4-6% of the page tokens (dropping them all would save that much and lose the line
structure); tagging every line instead of using » / self-delimiting starts costs 1-11% more, so it does not pay.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Callable

from .base import ANSI_RE, LEVEL, register, strip_marks, tag

# ----------------------------------------------------------------------------------------------- normalisation
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_SEP = re.compile(r"^(?P<i>[ \t]*)(?P<l>([=_\-!*#])\3{3,})(?:(?P<t>[ \t]+\S.*?[ \t]+)(?P<r>\3{4,}))?(?P<e>[ \t]*)$")
_USEP = re.compile(r"^[ \t]*_(?: _){3,}[ \t]*$")


def _cr(line: str) -> str:
    line = line.rstrip("\r")
    if "\r" not in line:
        return line
    parts = [p for p in line.split("\r") if p.strip()]     # a terminal shows only the last overwrite
    return parts[-1] if parts else ""


def norm_term(s: str) -> str:
    """ANSI/OSC escape codes removed; text overwritten by a bare carriage return (progress bars) dropped."""
    s = ANSI_RE.sub("", _OSC.sub("", s))
    if "\r" in s:
        s = "\n".join(_cr(x) for x in s.split("\n"))
    return s


def _sep_line(x: str) -> str:
    if _USEP.match(x):
        return "_ _ _"
    m = _SEP.match(x)
    if not m:
        return x
    c = m.group(3)
    return c * 3 + (f" {m['t'].strip()} " + c * 3 if m["t"] else "")


def norm_sep(s: str) -> str:
    """norm_term + separator lines collapsed to 3 characters per run (they carry no other information)."""
    return "\n".join(_sep_line(x) for x in norm_term(s).split("\n"))


def _lines(text: str, norm: Callable[[str], str]) -> list[tuple[int, str]]:
    return list(enumerate(norm(text).split("\n"), 1))


# ------------------------------------------------------------------------------------------- caret encoding
BRACKETS = "⟦⟧⟨⟩"
_CARET = re.compile(r"^(?P<g>[ ]*\|)?(?P<pad>[ ]*)(?P<run>[~^]+)(?:[ ]+(?P<label>[^\s~^].*?))?[ ]*$")
_RUN_OK = re.compile(r"^(?:~*\^+~*|~+)$")
_GUTTER = re.compile(r"^[ \t]*\d*[ \t]*\|")


def _bracket(src: str, a: int, b: int, run: str) -> str:
    seg = src[a:b]
    if "^" in run and "~" in run:
        i, j = run.index("^"), run.rindex("^") + 1
        enc = "⟦" + seg[:i] + "⟦" + seg[i:j] + "⟧" + seg[j:] + "⟧"
    elif "~" in run:
        enc = "⟨" + seg + "⟩"
    else:
        enc = "⟦" + seg + "⟧"
    return src[:a] + enc + src[b:]


def _caret(src: str, line: str):
    """-> (bracketed src, gutter residue, label) when ``line`` marks columns of ``src``, else None."""
    if "\t" in src or "\t" in line or any(c in src for c in BRACKETS):
        return None
    m = _CARET.match(line)
    if not m or not _RUN_OK.match(m["run"]):
        return None
    g = m["g"] or ""
    lo = 0
    if g:
        gc = len(g) - 1
        if len(src) <= gc or src[gc] != "|":
            return None
        lo = gc + 1
    a = len(g) + len(m["pad"])
    b = a + len(m["run"])
    body = src[lo:]
    first = lo + len(body) - len(body.lstrip(" "))
    if a < first or b > len(src.rstrip()):
        return None
    return _bracket(src, a, b, m["run"]), ("|" if g else ""), m["label"]


_GROUP_START = re.compile("[⟦⟨]")
_NEXT_LINE = re.compile(r"§\d+ |" + LEVEL)


def caret_expand(flow: str) -> str:
    """Undo the bracket encoding: brackets removed, the caret line re-inserted where the next line starts."""
    out, pos = [], 0
    while True:
        m = _GROUP_START.search(flow, pos)
        if not m:
            out.append(flow[pos:])
            return "".join(out)
        k = m.start()
        if flow[k] == "⟨":
            j = flow.index("⟩", k)
            plain, carets, end = flow[k + 1:j], "~" * (j - k - 1), j + 1
        else:
            depth, plain, kinds, i, nested = 0, [], [], k, False
            while True:
                c = flow[i]
                if c == "⟦":
                    depth += 1
                    nested = nested or depth == 2
                elif c == "⟧":
                    depth -= 1
                    if depth == 0:
                        break
                else:
                    plain.append(c)
                    kinds.append(depth)
                i += 1
            end = i + 1
            plain = "".join(plain)
            carets = "".join(("^" if d == 2 or not nested else "~") for d in kinds)
        nl = _NEXT_LINE.search(flow, end)
        ins = nl.start() if nl else len(flow)
        out.append(flow[pos:k] + plain + flow[end:ins] + " " + carets + " ")
        pos = ins


def _expand_caret_marks(flow: str) -> str:
    try:
        return strip_marks(caret_expand(flow))
    except (ValueError, IndexError):        # malformed brackets: the invariant check then rejects the flow
        return strip_marks(flow)


# ------------------------------------------------------------------------------------- generic line engine
def squash(s: str) -> str:
    """Whitespace runs -> one space, ends stripped; a bracketed caret span is kept verbatim (its width counts)."""
    starts = [i for i in (s.find("⟦"), s.find("⟨")) if i >= 0]
    if not starts:
        return " ".join(s.split())
    i = min(starts)
    j = max(s.rfind("⟧"), s.rfind("⟩")) + 1
    pre, mid, post = s[:i], s[i:j], s[j:]
    a, b = " ".join(pre.split()), " ".join(post.split())
    if a and pre[-1].isspace():
        a += " "
    if b and post[0].isspace():
        b = " " + b
    return a + mid + b


def _width(raw: str) -> int:
    g = _GUTTER.match(raw)
    if g:
        return len(g.group(0).expandtabs(8)) - 1
    e = raw.expandtabs(8)
    return len(e) - len(e.lstrip())


@dataclass
class Item:
    n: int
    raw: str
    level: int = 0
    tagged: bool = True
    fmt: Callable[[str], str] = squash
    suffix: str = ""
    caret_ok: bool = True


TAG_MODE = "default"        # measurement hook: "all" tags every item, "none" drops every tag (levels kept)


def emit(items: list[Item]) -> str:
    parts = []
    for it in items:
        t = it.fmt(it.raw)
        if it.suffix:
            t = (t + " " + it.suffix).strip()
        tagged = it.tagged if TAG_MODE == "default" else TAG_MODE == "all"
        parts.append((tag(it.n) if tagged else "") + LEVEL * it.level + t)
    return " ".join(p for p in parts if p)


class Stack:
    """Indentation widths -> levels (push on deeper, pop on shallower), like Python's INDENT/DEDENT."""

    def __init__(self):
        self.s = [0]

    def level(self, w: int) -> int:
        while w < self.s[-1]:
            self.s.pop()
        if w > self.s[-1]:
            self.s.append(w)
        return len(self.s) - 1


def line_items(lines, width=_width, special=None, carets: bool = True) -> list[Item]:
    """Policy for line-oriented output: unindented lines are tagged records; indented lines get one » per level
    and no tag; a caret line is folded into the line above. ``special(n, raw, stack)`` may return an Item."""
    items: list[Item] = []
    st = Stack()
    for n, raw in lines:
        if not raw.strip():
            continue
        prev = items[-1] if items else None
        if carets and prev is not None and prev.n == n - 1 and prev.caret_ok and not prev.suffix:
            c = _caret(prev.raw, raw)
            if c:
                prev.raw, prev.suffix, label = c
                prev.caret_ok = False
                if label:
                    items.append(Item(n, label, prev.level + 1, False, caret_ok=False))
                continue
        it = special(n, raw, st) if special else None
        if it is None:
            lv = st.level(width(raw))
            it = Item(n, raw, lv, lv == 0)
        items.append(it)
    return items


def _flow_lines(text: str, norm=norm_sep, width=_width, special=None) -> str:
    if any(c in text for c in BRACKETS):
        raise ValueError("text contains the caret bracket characters")
    return emit(line_items(_lines(text, norm), width, special))


# ----------------------------------------------------------------------------------------------- sniffers
def _nonblank(s: str, limit: int = 400) -> list[str]:
    out = []
    for x in _head(s).split("\n"):
        if x.strip():
            out.append(x)
            if len(out) >= limit:
                break
    return out


def _head(s: str, n: int = 60000) -> str:
    return norm_term(s[:n]) if ("\x1b" in s[:n] or "\r" in s[:n]) else s[:n]


def _share(lines: list[str], rx: re.Pattern) -> float:
    return sum(1 for x in lines if rx.search(x)) / max(1, len(lines))


_PYTEST_RX = re.compile(r"^=+ (?:test session starts|FAILURES|ERRORS|short test summary info|warnings summary) =+$"
                        r"|^=+ .*\b(?:passed|failed|errors?|skipped|deselected|no tests ran)\b.* in [\d.]+s"
                        r"|^(?:FAILED|ERROR|PASSED) \S+\.py(?:::|\b)"
                        r"|^\S+\.py::\S+ (?:PASSED|FAILED|ERROR|SKIPPED|XFAIL)", re.M)
_DIFF_RX = re.compile(r"^diff --git |^diff --cc |^@@@? -\d+(?:,\d+)? (?:-\d+(?:,\d+)? )?\+\d+(?:,\d+)? @@", re.M)
_COMMIT_RX = re.compile(r"^[*|/\\ ]*commit [0-9a-f]{40}\b", re.M)
_ONELINE_RX = re.compile(r"^[*|/\\ ]*[0-9a-f]{7,12} \S")
_STATUS_RX = re.compile(r"^(?:On branch |HEAD detached |Changes to be committed:|Changes not staged for commit:"
                        r"|Untracked files:|nothing to commit|Your branch is )", re.M)
_SHORT_STATUS = re.compile(r"^(?:[ MADRCUT?!]{2} \S|## \S)")
_TB_RX = re.compile(r"^[ |+]*(?:Exception Group )?Traceback \(most recent call last\):|^[ |]*File \"[^\"]+\", line \d+",
                    re.M)
_DIAG_RX = re.compile(r"^\S.*?:\d+(?::\d+)?:? (?:fatal error|error|warning|note|info|hint|remark)\b"
                      r"|^\S+\(\d+,\d+\): (?:error|warning)|^\S+:\d+:\d+ - (?:error|warning)"
                      r"|^\S+:\d+:\d+: [A-Z]+\d+ |^\S+\.go:\d+:\d+: |: \[(?:Error|Warning)/\S+\]$"
                      r"|^\s+\d+:\d+\s+(?:error|warning)\s"
                      r"|^(?:error|warning)(?:\[\w+\])?: |^\s*--> \S+:\d+:\d+|^\d+ (?:errors?|warnings?) generated\."
                      r"|^Found \d+ errors?|^error: could not compile")
_DIAG_STRONG = re.compile(r"^\s*--> \S+:\d+:\d+|^(?:error|warning)\[\w+\]: |^\d+ errors? generated\.|^Found \d+ errors?"
                          r"|^\S+:\d+:\d+ - error TS\d+|^error: could not compile")
_GREP_RX = re.compile(r"^[^\s:][^:]*:\d+[:-]|^\d+[:-]|^--$|^[^\s:][^:]*?\.\w+-\d+-")
_LS_RX = re.compile(r"^[-dlcbpsD][-rwxsStT]{9}[.+@]?\s+\d")
_DU_RX = re.compile(r"^\d+(?:[.,]\d+)?[KMGTPkB]?\t\S")
_STAT_RX = re.compile(r"^ \S.* \| +(?:\d+ ?[+-]*|Bin .*)$|^ \d+ files? changed")
_TEST_RX = re.compile(r"^(?:test \S+ \.\.\. |=== RUN|--- (?:FAIL|PASS|SKIP):|(?:not )?ok \d+ - |ok  \t|FAIL\t"
                      r"|PASS$|FAIL$|running \d+ tests?|test result: )")
_TREE_RX = re.compile(r"^(?:[│ ]   |\|   )*(?:[├└]── |[|`]-- )")
_TS_RX = re.compile(r"^\s*\[?(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|\d{2}:\d{2}:\d{2}|[A-Z][a-z]{2} [ \d]\d \d{2}:\d{2}"
                    r"|[IWEF]\d{4} \d{2}:\d{2}|(?:TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|CRITICAL|FATAL)\b"
                    r"|\d{2}/[A-Z][a-z]{2}/\d{4}:)|^\[[\w.:/-]+\] \S")


def sniff_pytest(s: str) -> bool:
    s = _head(s)
    return bool(_PYTEST_RX.search(s))


def sniff_gitdiff(s: str) -> bool:
    s = _head(s)
    if _DIFF_RX.search(s) or re.search(r"^--- \S.*\n\+\+\+ \S.*\n@@ ", s, re.M):
        return True
    ls = _nonblank(s)
    return len(ls) >= 2 and _share(ls, _STAT_RX) >= 0.8


def sniff_gitlog(s: str) -> bool:
    s = _head(s)
    if _COMMIT_RX.search(s):
        return True
    ls = _nonblank(s)
    return len(ls) >= 3 and _share(ls, _ONELINE_RX) >= 0.6


def sniff_gitstatus(s: str) -> bool:
    s = _head(s)
    if _STATUS_RX.search(s):
        return True
    ls = _nonblank(s)
    return len(ls) >= 2 and _share(ls, _SHORT_STATUS) >= 0.9 and bool(re.search(r"^(?:\?\? | M |M  |A  |D  )", s, re.M))


def sniff_traceback(s: str) -> bool:
    s = _head(s)
    return bool(_TB_RX.search(s))


def sniff_diagnostics(s: str) -> bool:
    ls = _nonblank(s)
    sh = _share(ls, _DIAG_RX)
    return sh >= 0.15 or (sh >= 0.03 and any(_DIAG_STRONG.search(x) for x in ls))


def sniff_grep(s: str) -> bool:
    ls = _nonblank(s)
    return len(ls) >= 2 and _share(ls, _GREP_RX) >= 0.6


def sniff_listing(s: str) -> bool:
    ls = _nonblank(s)
    return len(ls) >= 2 and (_share(ls, _LS_RX) >= 0.5 or _share(ls, _TREE_RX) >= 0.5 or _share(ls, _DU_RX) >= 0.8)


def sniff_find(s: str) -> bool:
    ls = _nonblank(s)
    return (len(ls) >= 3 and all(len(x.split()) == 1 for x in ls) and _share(ls, re.compile("/")) >= 0.7
            and not any(x.startswith(("http:", "https:")) for x in ls[:20]))


def sniff_log(s: str) -> bool:
    ls = _nonblank(s)
    return len(ls) >= 2 and (_share(ls, _TS_RX) >= 0.4 or _share(ls, _TEST_RX) >= 0.2
                              or sum(1 for x in ls if _TEST_RX.search(x)) >= 3)


# ------------------------------------------------------------------------------------------------- pytest
_E_LINE = re.compile(r"^E(?P<sp>[ ]*)(?P<rest>.*)$")


def _pytest_special():
    e_stack = Stack()
    e_base = [None]

    def special(n, raw, st):
        m = _E_LINE.match(raw)
        if m and (raw == "E" or raw.startswith("E ")):
            st.level(0)
            w = len(m["sp"])
            rest = m["rest"].rstrip()
            if e_base[0] is None or w < e_base[0]:
                e_base[0], e_stack.s = w, [w]
            sub = e_stack.level(w) if rest else 0
            pre = "E " + LEVEL * sub if rest else "E"
            k = len(m["sp"]) + 1
            return Item(n, raw, 0, True, fmt=lambda r, pre=pre, k=k: pre + r[k:].rstrip(), caret_ok=False)
        e_base[0] = None
        if raw.startswith(">") and (len(raw) == 1 or raw[1] == " "):
            w = len(raw) - len(raw[1:].lstrip(" "))
            lv = st.level(w)
            return Item(n, raw, lv, lv == 0, fmt=lambda r: ("> " + squash(r[1:])).strip())
        return None

    return special


def pytest_flow(text: str) -> str:
    return _flow_lines(text, special=_pytest_special())


# -------------------------------------------------------------------------- tracebacks, diagnostics, logs
def traceback_flow(text: str) -> str:
    return _flow_lines(text)


def diagnostics_flow(text: str) -> str:
    return _flow_lines(text)


def log_flow(text: str) -> str:
    return _flow_lines(text)


# ------------------------------------------------------------------------------------------------ git diff
_HUNK = re.compile(r"^(?P<at>@{2,}) (?P<ranges>[-+\d, ]+?) (?P=at)(?P<rest>.*)$")
_DIFF_HDR = re.compile(r"^(?:diff |index |--- |\+\+\+ |new file mode|deleted file mode|old mode|new mode"
                       r"|similarity index|dissimilarity index|rename from|rename to|copy from|copy to|Binary files"
                       r"|GIT binary patch|commit [0-9a-f]{7,}|Author:|AuthorDate:|Commit:|CommitDate:|Date:"
                       r"|Merge:|From [0-9a-f]{40}|From:|Subject:)")


def _indent_unit(bodies: list[str]) -> int:
    ws = [len(b) - len(b.lstrip(" ")) for b in bodies if b.strip() and not b.startswith("\t")]
    ws = [w for w in ws if w]
    if not ws:
        return 4
    for u in (4, 2):
        if sum(1 for w in ws if w % u == 0) >= 0.8 * len(ws):
            return u
    return 1


def _code_level(body: str, unit: int) -> tuple[int, str]:
    tabs = len(body) - len(body.lstrip("\t"))
    rest = body[tabs:]
    sp = len(rest) - len(rest.lstrip(" "))
    return tabs + sp // unit, rest.strip()


def gitdiff_flow(text: str) -> str:
    lines = _lines(text, norm_term)
    # pass 1: hunk extents (counts from the hunk headers), so '--- x' inside a hunk is a removed line
    kinds: dict[int, tuple] = {}
    i, N = 0, len(lines)
    bodies = []
    while i < N:
        n, raw = lines[i]
        h = _HUNK.match(raw)
        if not h:
            i += 1
            continue
        mw = len(h["at"]) - 1
        rng = re.findall(r"([-+])(\d+)(?:,(\d+))?", h["ranges"])
        old = [(int(a), int(b) if b else 1) for s, a, b in rng if s == "-"]
        new = [(int(a), int(b) if b else 1) for s, a, b in rng if s == "+"]
        if not old or not new:
            i += 1
            continue
        on, rem_o = old[0][0], sum(c for _, c in old)
        nn, rem_n = new[0][0], new[0][1]
        kinds[n] = ("hunk",)
        i += 1
        while i < N and (rem_n > 0 or (mw == 1 and rem_o > 0)):
            n, raw = lines[i]
            mk = raw[:mw]
            if raw.strip() == "" and mw == 1:              # blank context line (trailing space stripped)
                kinds[n] = ("blank",)
                on, nn, rem_o, rem_n = on + 1, nn + 1, rem_o - 1, rem_n - 1
            elif len(mk) == mw and all(c in " +-" for c in mk):
                body = raw[mw:]
                bodies.append(body)
                if "-" in mk and mw == 1:
                    kinds[n] = ("-", on, mk, body)
                    on, rem_o = on + 1, rem_o - 1
                elif "-" in mk:                             # combined diff: not in the result
                    kinds[n] = ("-", nn, mk, body)
                else:
                    kinds[n] = ("+" if "+" in mk else " ", nn, mk, body)
                    nn, rem_n = nn + 1, rem_n - 1
                    if mw == 1 and mk == " ":
                        on, rem_o = on + 1, rem_o - 1
            elif raw.startswith("\\"):
                kinds[n] = ("note",)
            else:
                break
            i += 1
        if i < N and lines[i][1].startswith("\\"):
            kinds[lines[i][0]] = ("note",)
            i += 1
    unit = _indent_unit(bodies)
    parts = []
    for n, raw in lines:
        k = kinds.get(n)
        if k is None:
            s = raw.strip()
            if not s:
                continue
            if _DIFF_HDR.match(raw):
                parts.append(squash(raw))
            else:                                           # commit message, --stat lines, other text
                parts.append(LEVEL * max(1, _width(raw) // 4) + squash(raw))
        elif k[0] == "hunk":
            parts.append(raw.strip())
        elif k[0] == "blank":
            continue
        elif k[0] == "note":
            parts.append(raw.strip())
        else:
            _, num, mk, body = k
            lv, code = _code_level(body, unit)
            t = mk.replace(" ", "") + LEVEL * lv + code
            tagged = TAG_MODE != "none"
            parts.append((tag(num) if tagged else "") + t)
    return " ".join(parts)


# ------------------------------------------------------------------------------------------------- git log
_LOG_HDR = re.compile(r"^[*|/\\ ]*(?:Author|AuthorDate|Commit|CommitDate|Date|Merge|Reflog|Reflog message):")


def gitlog_flow(text: str) -> str:
    items = []
    for n, raw in _lines(text, norm_term):
        if not raw.strip():
            continue
        if _COMMIT_RX.match(raw) or re.match(r"^[*|/\\ ]*commit [0-9a-f]{7,}", raw):
            items.append(Item(n, raw, 0, True))
        elif _LOG_HDR.match(raw):
            items.append(Item(n, raw, 0, False))
        elif raw[0] in " \t":
            items.append(Item(n, raw, 1, False))
        else:
            items.append(Item(n, raw, 0, True))
    return emit(items)


# ---------------------------------------------------------------------------------------------- git status
_SHORT = re.compile(r"^([ MADRCUT?!])([ MADRCUT?!]) (?=\S)")
_SHORT_FLOW = re.compile(r"(§\d+ )([.MADRCUT?!]{2}) ")


def _status_expand(flow: str) -> str:
    return strip_marks(_SHORT_FLOW.sub(lambda m: m.group(1) + m.group(2).replace(".", " ") + " ", flow))


def gitstatus_flow(text: str) -> str:
    items, short = [], set()
    for n, raw in _lines(text, norm_term):
        if not raw.strip():
            continue
        m = _SHORT.match(raw)
        if m and (m.group(1) + m.group(2)).strip():
            code = (m.group(1) + m.group(2)).replace(" ", ".")
            items.append(Item(n, code + " " + raw[3:], 0, True))
            short.add(n)
        elif raw.startswith("\t") or re.match(r"^\s+(?:modified|new file|deleted|renamed|copied|typechange|"
                                              r"both modified|both added|both deleted|added by|deleted by):", raw):
            items.append(Item(n, raw, 0, True))                  # an entry
        elif raw[0] in " \t":
            items.append(Item(n, raw, 1, False))                 # (use "git ..." to ...) hints
        else:
            items.append(Item(n, raw, 0, True))
    for it in items:                         # a long-format line that expand() would read as a short-format code
        if it.n not in short and it.tagged and _SHORT_FLOW.match(tag(it.n) + it.fmt(it.raw)):
            raise ValueError("ambiguous status line")
    return emit(items)


# ----------------------------------------------------------------------------------------------- grep / rg
_G_PATH = re.compile(r"^(.+?):(\d+):")
_G_NOPATH = re.compile(r"^(\d+)([:-])")


def _grep_items(lines):
    paths: dict[str, int] = {}
    nonblank = [raw for _, raw in lines if raw.strip()]
    for raw in nonblank:
        m = _G_PATH.match(raw)
        if m and not m.group(1).isdigit():
            paths[m.group(1)] = paths.get(m.group(1), 0) + 1
    single = not paths and nonblank and all(_G_NOPATH.match(r) or r == "--" for r in nonblank)
    by_len = sorted(paths, key=len, reverse=True)
    out = []                                                # (path or None for passthrough, n, sep, content)
    for _, raw in lines:
        if not raw.strip():
            continue
        if single:
            m = _G_NOPATH.match(raw)
            if m:
                out.append(("", m.group(1), m.group(2), raw[m.end():]))
                continue
        hit = None
        for p in by_len:
            if raw.startswith(p) and len(raw) > len(p) and raw[len(p)] in ":-":
                s = raw[len(p)]
                m = re.match(r"(\d+)" + re.escape(s), raw[len(p) + 1:])
                if m:
                    hit = (p, m.group(1), s, raw[len(p) + 1 + m.end():])
                    break
        out.append(hit if hit else (None, "", "", raw))
    return out


def _grep_expand(flow: str) -> str:
    out = []
    for grp in flow.split(LEVEL):
        if not grp.strip():
            continue
        ms = list(re.finditer(r"§(\d+)(-?) ?", grp))
        if not ms:
            out.append(grp.strip())
            continue
        head = grp[:ms[0].start()].strip()
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(grp)
            c = grp[m.end():end].strip()
            s = "-" if m.group(2) else ":"
            out.append(f"{head}{s}{m.group(1)}{s}{c}" if head else f"{m.group(1)}{s}{c}")
    return "\n".join(out)


def grep_flow(text: str) -> str:
    parts, cur = [], None
    for path, n, sep, content in _grep_items(_lines(text, norm_term)):
        c = " ".join(content.split())
        if path is None:
            parts.append(LEVEL + c)
            cur = None
            continue
        t = f"§{n}" + ("-" if sep == "-" else "")
        t = f"{t} {c}" if c else t
        if path != cur:
            parts.append(LEVEL + (path + " " + t if path else t))   # single-file grep -n: no path
            cur = path
        else:
            parts.append(t)
    return " ".join(parts)


# ------------------------------------------------------------------------------------------------- find
def _find_expand(flow: str) -> str:
    out = []
    for seg in re.split(r"§\d+ ", flow):
        toks = seg.split()
        if not toks:
            continue
        if toks[0].endswith("/") and len(toks) > 1:
            out.extend(toks[0] + t for t in toks[1:])
        else:
            out.extend(toks)
    return "\n".join(out)


def find_flow(text: str) -> str:
    entries = [(n, raw.strip()) for n, raw in _lines(text, norm_term) if raw.strip()]
    if any(len(p.split()) != 1 for _, p in entries):          # whitespace in a path: no grouping
        return emit([Item(n, p) for n, p in entries])
    parts, i = [], 0
    while i < len(entries):
        n, p = entries[i]
        d = p.rsplit("/", 1)[0] + "/" if "/" in p.rstrip("/") else None
        j = i + 1
        if d is not None and not p.endswith("/"):
            while (j < len(entries) and entries[j][1].startswith(d) and "/" not in entries[j][1][len(d):]
                   and entries[j][1] != d and not entries[j][1].endswith("/")):
                j += 1
        if j - i >= 2:
            parts.append(tag(n) + d + " " + " ".join(e[len(d):] for _, e in entries[i:j]))
        else:
            parts.append(tag(n) + p)
            j = i + 1
        i = j
    return " ".join(parts)


# ---------------------------------------------------------------------------------------------- listings
_TREE_ROW = re.compile(r"^(?P<pre>(?:│   |\|   |    )*)(?P<conn>[├└]── |[|`]-- )(?P<name>.*)$")


def _listing_special(n, raw, st):
    t = _TREE_ROW.match(raw)
    if t:
        pre = t["pre"]
        units = [pre[k:k + 4] for k in range(0, len(pre), 4)]
        p = "".join(LEVEL if u == "    " else u[0] for u in units)
        conn, name = t["conn"].strip(), " ".join(t["name"].split())
        return Item(n, raw, 0, False, fmt=lambda r, s=p + conn + " " + name: s, caret_ok=False)
    if _LS_RX.match(raw):
        return Item(n, raw, 0, False, caret_ok=False)
    return None


def listing_flow(text: str) -> str:
    items = line_items(_lines(text, norm_term), special=_listing_special, carets=False)

    def bare(it):                             # a plain `ls` name: one token, not a header, not an ls -l/tree row
        return it.tagged and len(it.raw.split()) == 1 and not it.raw.rstrip().endswith(":")

    for a, b in zip(items, items[1:], strict=False):  # names after a "dir:" header or another name are joined untagged
        if bare(b) and b.n == a.n + 1 and (bare(a) or not a.tagged and len(a.raw.split()) == 1
                                           or a.raw.rstrip().endswith(":")):
            b.tagged = False
    return emit(items)


# ------------------------------------------------------------------------------------------ command lines
_FILTERS = {"head", "tail", "less", "more", "tee", "sort", "uniq", "cat", "column", "cut", "nl", "fold"}
_WRAPPERS = {"sudo", "time", "nohup", "nice", "env", "command", "exec", "stdbuf", "unbuffer", "xargs"}
_SKIP = {"cd", "export", "source", ".", "set", "mkdir", "echo", "printf", "true", "pushd", "popd", "sleep",
         "clear", "unset", "alias", "ulimit", "trap"}
_DIAG = {"gcc", "g++", "cc", "c++", "clang", "clang++", "make", "cmake", "ninja", "rustc", "javac", "tsc", "eslint",
         "mypy", "ruff", "flake8", "pylint", "pyright", "golangci-lint", "swiftc", "kotlinc", "mvn", "gradle",
         "gradlew", "./gradlew", "biome", "oxlint", "shellcheck", "hadolint", "stylelint", "clippy-driver", "dmypy",
         "basedpyright", "pyflakes", "pycodestyle", "bandit", "vue-tsc", "svelte-check", "ty"}
_TESTS = {"jest", "vitest", "mocha", "ava", "tap", "karma", "phpunit", "rspec", "ctest", "tox", "nox", "bats",
          "playwright", "cypress"}
_LOGS = {"journalctl", "dmesg", "docker-compose"}


def _words(seg: str) -> list[str]:
    try:
        w = shlex.split(seg, comments=False)
    except ValueError:
        w = seg.split()
    while w and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w[0]) or w[0] in _WRAPPERS
                 or (w[0] == "timeout" and len(w) > 1)):
        if w[0] == "timeout":
            w = w[1:]
            while w and (w[0].startswith("-") or re.match(r"^\d+[smhd]?$", w[0])):
                w = w[1:]
            continue
        w = w[1:]
        while w and w[0].startswith("-"):     # wrapper options (env -i, xargs -0, sudo -E)
            w = w[1:]
    return w


def _classify(w: list[str]) -> str | None:
    if not w:
        return None
    c = w[0].rsplit("/", 1)[-1]
    rest = w[1:]
    if c in ("uv", "poetry", "pipenv", "pdm", "hatch", "rye", "pixi", "conda", "mamba") and rest[:1] == ["run"]:
        return _classify(rest[1:])
    if c in ("npx", "bunx", "pnpx") or (c in ("pnpm", "yarn") and rest[:1] == ["exec"]):
        return _classify(rest[1:] if c in ("pnpm", "yarn") else rest)
    if re.match(r"^python[\d.]*$", c) or c == "py":
        if "-m" in rest:
            k = rest.index("-m")
            mod = rest[k + 1] if k + 1 < len(rest) else ""
            return {"pytest": "pytest", "py.test": "pytest", "unittest": "log", "mypy": "diagnostics",
                    "ruff": "diagnostics", "flake8": "diagnostics", "pylint": "diagnostics",
                    "pyright": "diagnostics", "compileall": "diagnostics"}.get(mod)
        return None
    if c in ("pytest", "py.test"):
        return "pytest"
    if c == "git":
        k = 0
        while k < len(rest) and rest[k].startswith("-"):
            k += 2 if rest[k] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace") else 1
        sub, args = (rest[k], rest[k + 1:]) if k < len(rest) else ("", [])
        if sub in ("diff", "show") and any(x in ("--name-only", "--name-status") for x in args):
            return "find" if "--name-only" in args else None
        if sub in ("diff", "show", "format-patch", "stash") and not (sub == "stash" and args[:1] != ["show"]):
            return "gitdiff"
        if sub in ("log", "whatchanged"):
            return "gitdiff" if any(a in ("-p", "-u", "--patch") or a.startswith("--patch") for a in args) \
                else "gitlog"
        if sub in ("reflog", "shortlog", "cherry"):
            return "gitlog"
        if sub == "status":
            return "gitstatus"
        if sub == "grep":
            return "find" if any(a in ("-l", "-L", "--name-only", "--files-with-matches") for a in args) else "grep"
        if sub == "ls-files":
            return "find"
        if sub == "blame":
            return "log"
        return None
    if c in ("diff", "colordiff") and any(a.startswith("-u") or a.startswith("--unified") or a == "-ru"
                                          or a.startswith("-r") for a in rest):
        return "gitdiff"
    if c in ("grep", "egrep", "fgrep", "rg", "ag", "ack", "ugrep", "zgrep"):
        if any(a in ("-l", "-L", "--files-with-matches", "--files-without-match", "--files") for a in rest):
            return "find"
        return "grep"
    if c in ("find", "fd", "fdfind", "locate"):
        return "find"
    if c in ("ls", "tree", "du", "exa", "eza", "lsd", "dir"):
        return "listing"
    if c == "cargo":
        sub = next((a for a in rest if not a.startswith("-") and not a.startswith("+")), "")
        return {"test": "log", "nextest": "log", "build": "diagnostics", "check": "diagnostics",
                "clippy": "diagnostics", "b": "diagnostics", "c": "diagnostics", "t": "log",
                "doc": "diagnostics"}.get(sub)
    if c == "go":
        sub = rest[0] if rest else ""
        return {"test": "log", "build": "diagnostics", "vet": "diagnostics", "install": "diagnostics"}.get(sub)
    if c in ("npm", "pnpm", "yarn", "bun"):
        sub = rest[0] if rest else ""
        if sub == "run" and len(rest) > 1:
            sub = rest[1]
        if sub in ("test", "t", "jest", "vitest") or sub.startswith("test"):
            return "log"
        if sub in ("lint", "typecheck", "type-check", "tsc", "build", "check", "eslint"):
            return "diagnostics"
        return None
    if c in ("node", "deno") and ("--test" in rest or rest[:1] == ["test"]):
        return "log"
    if c == "dotnet" and rest[:1] in (["build"], ["test"]):
        return "diagnostics" if rest[0] == "build" else "log"
    if c in ("docker", "kubectl", "podman") and "logs" in rest:
        return "log"
    if c in _DIAG:
        if c == "ruff" and rest[:1] == ["format"]:
            return None
        return "diagnostics"
    if c in _TESTS:
        return "log"
    if c in _LOGS:
        return "log"
    return None


_REDIR = re.compile(r"\d?>&\d|&>>?\s*\S+|\d?>>?\s*\S+|<\s*\S+")


def _pipeline_kind(pipe: list[list[str]]) -> str | None:
    producer = _classify(pipe[0])
    for w in reversed(pipe[1:]):               # the last command that reshapes the lines decides
        c = w[0].rsplit("/", 1)[-1]
        if c in _FILTERS:
            continue
        if c in ("grep", "egrep", "fgrep", "rg", "ag"):
            flags = [a for a in w[1:] if a.startswith("-")]
            if any(a in ("-c", "--count", "-l", "-L", "--files-with-matches") for a in flags):
                return None
            if any(a in ("-n", "--line-number") or (not a.startswith("--") and "n" in a) for a in flags):
                return "grep"                  # `... | grep -n x` prints n:line
            return producer                    # a filter keeps the producer's line format
        return None                            # sed, awk, jq, wc ... reshape the output
    return producer


def kind_for_command(cmd: str) -> str | None:
    """Kind of the output of a shell command line, or None (then the text is sniffed). A pipeline is classified by
    its producer unless a later stage reshapes the lines (head/tail/sort... and plain grep filters keep the format);
    a chain (&&, ||, ;) by its commands that print (cd/export/echo... are ignored); disagreeing commands give None."""
    if not cmd or not cmd.strip():
        return None
    kinds = []
    for seg in re.split(r"&&|\|\||;|\n", _REDIR.sub(" ", cmd)):
        pipe = [w for w in (_words(p) for p in seg.split("|")) if w]
        if not pipe or (len(pipe) == 1 and pipe[0][0].rsplit("/", 1)[-1] in _SKIP):
            continue
        kinds.append(_pipeline_kind(pipe))
    if not kinds:
        return None
    return kinds[0] if all(k == kinds[0] for k in kinds) else None


# ------------------------------------------------------------------------------------------------- legends
LEGENDS = {
    "pytest": "Each unindented output line starts with §<n> (its line number in the output); » marks an indented "
              "line, one per level; in source excerpts ⟦x⟧ is the span the ^ carets marked (⟦a ⟦b⟧ c⟧: ~ under a "
              "and c, ^ under b; ⟨x⟩: ~); separator lines are shortened.",
    "traceback": "Each unindented line starts with §<n> (output line); » marks indented lines, one per level; ⟦x⟧ is "
                 "the span the ^ carets marked under the source line (⟦a ⟦b⟧ c⟧: ~ under a and c, ^ under b).",
    "diagnostics": "Each unindented line starts with §<n> (output line); » marks indented lines (code excerpts, "
                   "notes), one per level; ⟦x⟧ / ⟨x⟩ is the span the ^ / ~ markers pointed at.",
    "log": "Each unindented line starts with §<n> (its line number in the output); » marks indented continuation "
           "lines, one per level; progress-bar redraws are dropped.",
    "gitdiff": "Hunk lines carry §<n> = their line number in the NEW file (removed lines: in the OLD file), then the "
               "+/- marker (none for context), then one » per indentation level; headers are untagged.",
    "gitlog": "§<n> starts each commit (n = output line); » marks message and stat lines.",
    "gitstatus": "§<n> starts each entry; in short format a '.' in the two-letter status code stands for a blank.",
    "grep": "»path starts the matches in one file; §<n> content = a match on line n; §<n>- content = a context "
            "line; »-- separates context groups.",
    "find": "§<n> dir/ a b c lists the entries dir/a, dir/b, dir/c (n = output line of the first).",
    "listing": "ls -l rows start at their permission string; tree rows at their connector, with │ or » per "
               "indentation level; other lines start with §<n>.",
}
# tags that are NOT the output's own line numbers (do not use them as Read offsets into the saved output)
FILE_LINE_TAGS = {"gitdiff", "grep"}


# ------------------------------------------------------------------------------------------------ registry
# Order matters for sniffing (first match wins): specific formats first, the generic log/traceback after them.
# cpt = chars per Claude token of the ORIGINAL output, measured with the token-counting API on the real samples.
CPT = {"pytest": 2.16, "gitdiff": 2.20, "gitstatus": 1.95, "gitlog": 2.22, "log": 1.69, "traceback": 1.90,
       "diagnostics": 2.10, "grep": 2.12, "listing": 1.65, "find": 2.00}
register("pytest", cpt=CPT["pytest"], sniff=sniff_pytest, expand=_expand_caret_marks, normalise=norm_sep)(pytest_flow)
register("gitdiff", cpt=CPT["gitdiff"], sniff=sniff_gitdiff, normalise=norm_term)(gitdiff_flow)
register("gitstatus", cpt=CPT["gitstatus"], sniff=sniff_gitstatus, expand=_status_expand,
         normalise=norm_term)(gitstatus_flow)
register("gitlog", cpt=CPT["gitlog"], sniff=sniff_gitlog, normalise=norm_term)(gitlog_flow)
register("log", exts=(".log",), cpt=CPT["log"], sniff=sniff_log, expand=_expand_caret_marks,
         normalise=norm_sep)(log_flow)
register("traceback", cpt=CPT["traceback"], sniff=sniff_traceback, expand=_expand_caret_marks,
         normalise=norm_sep)(traceback_flow)
register("diagnostics", cpt=CPT["diagnostics"], sniff=sniff_diagnostics, expand=_expand_caret_marks,
         normalise=norm_sep)(diagnostics_flow)
register("grep", cpt=CPT["grep"], sniff=sniff_grep, expand=_grep_expand, normalise=norm_term)(grep_flow)
register("listing", cpt=CPT["listing"], sniff=sniff_listing, normalise=norm_term)(listing_flow)
register("find", cpt=CPT["find"], sniff=sniff_find, expand=_find_expand, normalise=norm_term)(find_flow)
