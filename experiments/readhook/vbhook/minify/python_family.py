"""Minifiers for whitespace- and line-sensitive code: Python, shell scripts, Makefiles, Dockerfiles.

Read ``SPEC.md`` and ``base.py`` first. Every flow keeps all non-whitespace characters (backslash continuations
included) and adds only ``§<n> `` tags and ``»`` markers.

Python (``python``)
    * One tag per logical line, ``»`` per block level (from the tokenizer's INDENT/DEDENT). Continuation lines inside
      brackets are joined untagged (measured on 15 library files: tagging them costs 4.5% more page tokens, for
      line numbers that Read offsets and tracebacks rarely need inside one statement); a tag is still forced after
      a line that ends in a comment (else the comment would swallow the joined code) and after a backslash
      continuation (the ``\\`` is kept, the next line is tagged).
    * Comment-only lines get their own tag and the level of the statement after them (+1.1% page tokens).
    * Spaces between tokens are removed (saves 3% of page tokens) except where SPEC rule 5 needs one: between two
      non-operator tokens (names, numbers, strings, keywords), around hard keywords next to operators
      (``return -1``, ``x) if y``, ``in (a, b)``), between operators whose touching characters would read as
      another operator (``a - -b``, ``x < -1``, ``a * *b``) and before ``.`` after a
      number (``1 .real``). One space before a comment.
    * Single-line strings and f-strings are copied verbatim (an f-string's span is taken from the source, so
      nothing inside ``{...}`` or a format spec changes).
    * Multi-line strings: docstrings (a string that is a whole statement) are joined into paragraphs: a new tag
      only after a blank line, where the indentation changes, at list/doctest/field markers (``>>>``, ``-``,
      ``:param``...), at rst headings and after a line ending in ``\\``; deeper lines carry one ``»`` per indent
      step relative to the docstring's base. Other multi-line strings (SQL, templates, embedded code) keep every
      line, tagged, with ``»`` for their relative indentation, so embedded code keeps its structure. Docstring
      paragraphs cost 2.4% more page tokens than collapsing each docstring into one paragraph (``doc_mode``).
      Overall about the same page tokens as ``pymin.reflow`` (+0.7%), with comments placed, docstring structure
      kept and a reconstruction that stays runnable with comments.
    * ``reconstruct()`` rebuilds runnable code (4 spaces per level); ``same_ast()`` compares ASTs with the
      whitespace inside string constants collapsed.

Shell (``shell``): every logical line is tagged; backslash continuations are joined (``\\`` kept); lines inside a
multi-line quote and heredoc bodies are kept verbatim, one tag per line; comments kept; indentation dropped and runs
of blanks outside quotes collapsed.

Makefile (``makefile``): one tag per logical line; a leading TAB (the recipe prefix, semantic) becomes ``»``; other
indentation dropped; backslash continuations joined; runs of blanks outside quotes collapsed.

Dockerfile (``dockerfile``): one tag per instruction; continuation lines (escape char from the ``# escape=``
directive) joined; comment lines inside a continued instruction are tagged, and so is the line after them;
heredoc bodies kept verbatim, one tag per line.
"""
from __future__ import annotations

import ast
import io
import keyword
import os
import re
import tokenize
from typing import NamedTuple

from .base import LEVEL, register, tag

# --------------------------------------------------------------------------------------------------------- Python

_HARD_KW = frozenset(keyword.kwlist) - {"None", "True", "False"}
# two characters that, touching across a removed space, would form (or read as) another operator
_GLUE = frozenset("** // << >> <= >= == != -> := += -= *= /= %= &= |= ^= @= .. -- ++ <- <> ~~".split())
_SOFT_KW = frozenset(("match", "case", "type"))
_KW_AFTER_OP_OK = frozenset("( [ { , = : ;".split())        # `(not x)`, `=lambda`, `,await f()`
_OP_AFTER_KW_OK = frozenset(") ] } : , ; . =".split())       # `else:`, `yield)`, `pass;`
_FS_START = {getattr(tokenize, n) for n in ("FSTRING_START", "TSTRING_START") if hasattr(tokenize, n)}
_FS_END = {getattr(tokenize, n) for n in ("FSTRING_END", "TSTRING_END") if hasattr(tokenize, n)}
_DOC_MARK = re.compile(r"(>>>|\.\.\.|[-*+•] |:[A-Za-z]|@[A-Za-z]|\.\. |\d+[.)] |#|\||[A-Za-z_][\w.]* : )")
_UNDERLINE = re.compile(r"""^([-=~^*+#"'`])\1{2,}$""")


class _Tok(NamedTuple):
    type: int
    string: str
    start: tuple
    end: tuple


def _slice(lines: list[str], a: tuple, b: tuple) -> str:
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        return lines[r1 - 1][c1:c2]
    return lines[r1 - 1][c1:] + "".join(lines[r1:r2 - 1]) + lines[r2 - 1][:c2]


def _tokens(src: str) -> tuple[list[_Tok], list[str]]:
    """Tokens with each f-string (3.12+: FSTRING_START ... FSTRING_END) merged into one STRING copied from the
    source."""
    lines = io.StringIO(src).readlines()
    out, depth, start = [], 0, None
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.ERRORTOKEN and t.string.strip():
            raise ValueError(f"tokenize error at {t.start}")
        if t.type in _FS_START:
            if depth == 0:
                start = t.start
            depth += 1
            continue
        if depth:
            if t.type in _FS_END:
                depth -= 1
                if depth == 0:
                    out.append(_Tok(tokenize.STRING, _slice(lines, start, t.end), start, t.end))
            continue
        out.append(_Tok(t.type, t.string, t.start, t.end))
    return out, lines


def _sep(a: _Tok, b: _Tok) -> str:
    """Separator between two tokens that had whitespace (or a line break) between them."""
    if b.type == tokenize.COMMENT:
        return " "
    a_op, b_op = a.type == tokenize.OP, b.type == tokenize.OP
    if not a_op and not b_op:
        return " "
    if a_op and b_op:
        return " " if a.string[-1] + b.string[0] in _GLUE else ""
    if a_op:                                   # operator, then a word
        if b.type == tokenize.NAME and b.string in _HARD_KW and a.string not in _KW_AFTER_OP_OK:
            return " "
        if b.string.startswith(".") and a.string.endswith("."):
            return " "
        return ""
    if a.type == tokenize.NAME and a.string in _HARD_KW and b.string not in _OP_AFTER_KW_OK:
        return " "
    if a.type == tokenize.NUMBER and b.string.startswith("."):
        return " "
    return ""


def _indent(s: str) -> int:
    return len(s.expandtabs(8)) - len(s.expandtabs(8).lstrip())


def _ws1(s: str) -> str:
    return re.sub(r"[ \t\f\v]+", " ", s)


def _string_flow(text: str, row: int, mode: str) -> str:
    """A multi-line string token -> flow text. Its first line stays where it is; later lines are joined with a
    space or start with a tag. ``mode``: "doc" (paragraphs, see module doc), "lines" (a tag per line),
    "collapse" (one paragraph). A line after one ending in ``\\`` is always tagged with its blanks kept."""
    parts = text.split("\n")
    body = [p for p in parts[1:] if p.strip()]
    widths = sorted({_indent(p) for p in body}) or [0]
    rank = {w: i for i, w in enumerate(widths)}
    out = [parts[0].rstrip()]
    prev_blank, prev_w, prev_bs, prev_under = False, widths[0], parts[0].rstrip().endswith("\\"), False
    prev_dt = False                            # previous line was a doctest prompt: its output starts a line
    for k, raw in enumerate(parts[1:], 1):
        s = raw.strip()
        if prev_bs:                            # after `\`+newline the leading blanks are part of the value
            out.append(" " + tag(row + k) + _ws1(raw.rstrip()))
            prev_bs, prev_blank = raw.rstrip().endswith("\\"), False
            continue
        if not s:
            prev_blank = True
            continue
        w = _indent(raw)
        if mode == "lines":
            new = True
        elif mode == "collapse":
            new = False
        else:
            nxt = parts[k + 1].strip() if k + 1 < len(parts) else ""
            under = bool(_UNDERLINE.match(s))
            heading = bool(_UNDERLINE.match(nxt)) and not under
            new = (prev_blank or w != prev_w or prev_under or prev_dt or heading or bool(_DOC_MARK.match(s))) \
                and not under
        out.append(" " + (tag(row + k) + LEVEL * rank[w] if new else "") + _ws1(s))
        prev_blank, prev_w, prev_bs = False, w, s.endswith("\\")
        prev_under = bool(_UNDERLINE.match(s))
        prev_dt = s.startswith((">>>", "..."))
    return "".join(out)


def python_flow(src: str, cont_tags: bool = False, doc_mode: str = "smart", keep_spaces: bool = False) -> str:
    """Python source -> flow. ``cont_tags``: tag every continuation line inside brackets too. ``doc_mode``:
    "smart" (default, see module doc), "lines" (every docstring line tagged) or "collapse" (one paragraph)."""
    toks, lines = _tokens(src)
    out: list[str] = []
    depth, logical_start, prev, prev_first = 0, True, None, False
    n = len(toks)
    for i, t in enumerate(toks):
        tt = t.type
        if tt == tokenize.INDENT:
            depth += 1
            continue
        if tt == tokenize.DEDENT:
            depth -= 1
            continue
        if tt in (tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL):
            continue
        if tt == tokenize.NEWLINE:
            logical_start, prev = True, None
            continue
        row = t.start[0]
        lead = " " if out else ""
        first = logical_start and tt != tokenize.COMMENT       # first token of a logical line
        if logical_start and tt == tokenize.COMMENT:          # comment-only line: the next statement's level
            j, d = i + 1, depth
            while j < n and toks[j].type in (tokenize.COMMENT, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                d += {tokenize.INDENT: 1, tokenize.DEDENT: -1}.get(toks[j].type, 0)
                j += 1
            out.append(lead + tag(row) + LEVEL * max(d, 0))
        elif logical_start:
            out.append(lead + tag(row) + LEVEL * depth)
            logical_start = False
        elif row != prev.end[0]:                              # a new physical line inside a statement
            gap = _slice(lines, prev.end, t.start)
            if "\\" in gap:
                out.append(" " + " ".join("\\" * gap.count("\\")) + " " + tag(row))
            elif prev.type == tokenize.COMMENT or cont_tags or tt == tokenize.COMMENT:
                out.append(" " + tag(row))
            else:
                out.append(_sep(prev, t))
        elif t.start[1] > prev.end[1]:                        # `match (x):`, `case [a]:` keep their space
            out.append(" " if keep_spaces or (prev_first and prev.string in _SOFT_KW) else _sep(prev, t))
        if tt == tokenize.STRING and "\n" in t.string:
            j = i + 1
            while j < n and toks[j].type == tokenize.COMMENT:
                j += 1
            is_doc = first and (j >= n or toks[j].type == tokenize.NEWLINE)
            out.append(_string_flow(t.string, row, (doc_mode if is_doc else "lines").replace("smart", "doc")))
        elif tt == tokenize.COMMENT:
            out.append(_ws1(t.string.rstrip()))
        else:
            out.append(t.string)
        prev, prev_first = (None, False) if logical_start else (t, first)
    tail = _slice(lines, prev.end, (len(lines), len(lines[-1]))) if prev and lines else ""
    if "\\" in tail:                                          # a stray continuation at EOF
        out.append(" " + "\\" * tail.count("\\"))
    return "".join(out)


def reconstruct(flow: str) -> str:
    """Python flow -> runnable source (4 spaces per level; every tag starts a line)."""
    out = []
    for m in re.finditer(r"§(\d+) (»*)(.*?)(?= §\d+ |$)", flow, re.S):
        out.append("    " * len(m.group(2)) + m.group(3))
    return "\n".join(out) + "\n"


def same_ast(a: str, b: str) -> bool:
    """AST equality with the whitespace inside string constants collapsed (joining lines changes it)."""
    class N(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = re.sub(r"\s+", " ", node.value).strip()
            elif isinstance(node.value, bytes):
                node.value = re.sub(rb"\s+", b" ", node.value).strip()
            return node

    def d(s):
        return ast.dump(N().visit(ast.parse(s)))
    return d(a) == d(b)


_PY_SHEBANG = re.compile(r"#![^\n]*\bpython[\d.]*\b")


@register("python", exts=(".py", ".pyi", ".pyw"), cpt=2.9, sniff=lambda s: bool(_PY_SHEBANG.match(s)))
def minify_python(src: str) -> str:
    # Edit-safe by default (VBHOOK_TIGHT=1 for the denser variant): Claude Code checks an Edit's old_string before any
    # hook runs, so text copied from the image must match the file: keep one space where the source had whitespace
    # inside a line and tag every physical line (+~7% page tokens; the smoke runs showed edits failing without it).
    safe = os.environ.get("VBHOOK_TIGHT", "0") != "1"
    return python_flow(src, cont_tags=safe, keep_spaces=safe)


# ---------------------------------------------------------------------------------------------- shell scanner

_HEREDOC = re.compile(r"<<(-?)[ \t]*(?:'([^'\n]*)'|\"([^\"\n]*)\"|\\?([A-Za-z_][\w.-]*))")


class _Line(NamedTuple):
    text: str            # the line with blank runs outside quotes collapsed (ends untouched if inside a quote)
    cont: bool           # ends with an unquoted `\` (a continuation)
    stack: tuple         # quote context open at the end of the line
    heredocs: tuple      # (delimiter, strip_tabs) started on this line


def _scan(line: str, stack: tuple = (), comments: bool = True, heredocs: bool = True) -> _Line:
    """Shell-ish scan of one line from the quote context ``stack``: collapse blank runs outside quotes and
    comments-or-not, find heredoc starts, a trailing continuation and the context left open."""
    st = list(stack)
    out: list[str] = []
    docs: list = []
    i, n, cont = 0, len(line), False
    while i < n:
        c = line[i]
        top = st[-1] if st else None
        if top == "'":
            out.append(c)
            if c == "'":
                st.pop()
            i += 1
            continue
        if c == "\\":
            if i + 1 >= n:
                cont = True
                out.append(c)
                i += 1
                continue
            out.append(line[i:i + 2])
            i += 2
            continue
        if top in ('"', "$'", "`"):
            if (top == '"' and c == '"') or (top == "$'" and c == "'") or (top == "`" and c == "`"):
                st.pop()
            elif top in ('"', "`") and line.startswith("$(", i):
                st.append("(")
                out.append("$(")
                i += 2
                continue
            elif top in ('"', "`") and line.startswith("${", i):
                st.append("{")
                out.append("${")
                i += 2
                continue
            elif top == '"' and c == "`":
                st.append("`")
            out.append(c)
            i += 1
            continue
        # unquoted context: top level, inside $( ), ${ } or ( )
        if c in " \t" and top != "{":
            j = i
            while j < n and line[j] in " \t":
                j += 1
            out.append(" ")
            i = j
            continue
        if c == "#" and comments and (i == 0 or line[i - 1] in " \t;|&(") and top != "{":
            out.append(_ws1(line[i:]))
            break
        if c == "'":
            st.append("'")
        elif c == '"':
            st.append('"')
        elif c == "`":
            st.append("`")
        elif line.startswith("$'", i):
            st.append("$'")
            out.append("$'")
            i += 2
            continue
        elif line.startswith("$(", i) or line.startswith("${", i):
            st.append(line[i + 1])
            out.append(line[i:i + 2])
            i += 2
            continue
        elif c == "(" and top == "(":
            st.append("(")
        elif (c == ")" and top == "(") or (c == "}" and top == "{"):
            st.pop()
        elif c == "<" and heredocs and line.startswith("<<", i) and not line.startswith("<<<", i) \
                and (i == 0 or line[i - 1] != "<") and "((" not in line[:i]:
            m = _HEREDOC.match(line, i)
            if m:
                delim = next(g for g in m.groups()[1:] if g is not None)
                docs.append((delim, m.group(1) == "-"))
                out.append(m.group(0))
                i = m.end()
                continue
        out.append(c)
        i += 1
    return _Line("".join(out), cont, tuple(st), tuple(docs))


def _shell_like(src: str, mode: str = "sh", escape: str = "\\") -> str:
    """Shared driver for shell (``sh``), Makefile (``make``) and Dockerfile (``docker``) flows."""
    out: list[str] = []
    stack: tuple = ()            # quote context open at the end of the previous line (shell only)
    pending: list = []           # heredocs waiting for their bodies
    doc = None                   # (delimiter, strip_tabs) of the heredoc body being copied
    cont = False                 # previous line ended with a continuation
    after_comment = False        # Dockerfile: previous line was a comment inside a continued instruction
    carried: tuple = ()          # Dockerfile: quote context carried into a continuation line
    for no, raw in enumerate(src.split("\n"), 1):
        raw = raw.rstrip("\r")
        lead = " " if out else ""
        if doc is not None:                                   # heredoc body: verbatim, one tag per line
            out.append(lead + tag(no) + raw)
            if (raw.lstrip("\t") if doc[1] else raw) == doc[0]:
                doc = pending.pop(0) if pending else None
            continue
        if stack:                                             # inside a multi-line quote: verbatim start
            ln = _scan(raw, stack)
            out.append(lead + tag(no) + (ln.text if ln.stack else ln.text.rstrip()))
        else:
            body = raw.strip()
            if not body:
                if mode != "docker":                          # a blank line ends a shell/make continuation
                    cont = False
                continue
            if mode == "docker" and body.startswith("#"):
                out.append(lead + tag(no) + _ws1(body))
                after_comment = cont
                continue                                      # comments never continue or end an instruction
            ctx = carried if cont else ()
            if escape != "\\" and body.endswith(escape):    # Dockerfile with `# escape=``
                ln = _scan(body[:-1].rstrip(), ctx)
                ln = _Line(ln.text + " " + escape, True, ln.stack, ln.heredocs)
            else:
                ln = _scan(body, ctx, heredocs=mode != "make")
            if cont and not after_comment:
                out.append(" " + ln.text.strip())
            else:
                mark = LEVEL if mode == "make" and raw.startswith("\t") and not cont else ""
                out.append(lead + tag(no) + mark + ln.text.strip())
            after_comment = False
        if mode == "make":                                    # make: any odd trailing `\` continues (comments too)
            cont = (len(raw) - len(raw.rstrip("\\"))) % 2 == 1
            stack = ()
        elif mode == "docker":                                # Docker joins on the escape char, quoted or not
            cont = body.endswith(escape)
            carried = ln.stack if cont else ()
        elif ln.stack:
            stack, cont = ln.stack, False                     # the next line starts inside a quote
        else:
            stack, cont = (), ln.cont
        pending.extend(ln.heredocs)
        if pending and not cont and not stack:
            doc = pending.pop(0)
    return "".join(out)


_SH_SHEBANG = re.compile(r"#!\s*/(usr/)?(local/)?bin/(env\s+)?(ba|z|da|k)?sh\b")


@register("shell", exts=(".sh", ".bash", ".zsh", ".ksh"),
          names=(".bashrc", ".bash_profile", ".profile", ".zshrc", ".zprofile", ".zshenv"),
          cpt=2.1, sniff=lambda s: bool(_SH_SHEBANG.match(s)))
def minify_shell(src: str) -> str:
    return _shell_like(src)


@register("makefile", exts=(".mk", ".mak"), names=("Makefile", "makefile", "GNUmakefile"), cpt=1.65)
def minify_makefile(src: str) -> str:
    return _shell_like(src, "make")


_ESCAPE_DIRECTIVE = re.compile(r"\A(?:[ \t]*#[ \t]*\w+[ \t]*=[^\n]*\n)*?[ \t]*#[ \t]*escape[ \t]*=[ \t]*([\\`])",
                               re.I)
_FROM_FIRST = re.compile(r"\A(?:[ \t]*(?:#[^\n]*)?\n)*[ \t]*(?:ARG[ \t][^\n]*\n(?:[ \t]*(?:#[^\n]*)?\n)*[ \t]*)*"
                         r"FROM[ \t]", re.I)


@register("dockerfile", exts=(".dockerfile",), names=("Dockerfile", "Containerfile", "dockerfile"), cpt=1.95,
          sniff=lambda s: bool(_FROM_FIRST.match(s)))
def minify_dockerfile(src: str) -> str:
    m = _ESCAPE_DIRECTIVE.match(src)
    return _shell_like(src, "docker", escape=m.group(1) if m else "\\")


# ------------------------------------------------------------------------------- reconstruction (checks only)

def reconstruct_lines(flow: str, recipe_tab: bool = False) -> str:
    """Shell/Makefile/Dockerfile flow -> text, one line per tag (``»`` -> TAB for Makefiles). A joined
    continuation (`` \\ `` after a blank) is split again: an escaped blank after a blank is rare in real scripts,
    so this is a heuristic, good enough for syntax checks."""
    out = []
    for m in re.finditer(r"§(\d+) ?(»?)(.*?)(?= §\d+ |$)", flow, re.S):
        body = re.sub(r"(?<=[ \t])\\ ", "\\\n", m.group(3))
        out.append(("\t" if recipe_tab and m.group(2) else m.group(2)) + body)
    return "\n".join(out) + "\n"
