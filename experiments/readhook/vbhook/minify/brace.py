"""Minifiers for brace/semicolon languages, where whitespace is not semantic: JavaScript, TypeScript (and TSX),
Java, Kotlin, Scala, C, C++, C#, Go, Rust, Swift, PHP, Dart.

Read ``SPEC.md`` and ``base.py`` first. One lexer, parameterised per language (``Lang``), splits the source into
tokens; the flow is the tokens joined by tags or by the shortest whitespace that keeps them apart.

Lexer
    * Comments: ``//`` and ``/* */`` everywhere (nested block comments in Rust, Kotlin, Swift, Scala and Dart);
      ``#`` in PHP (not ``#[``, an attribute); a ``#!`` line at offset 0.
    * Strings, copied byte for byte (newlines and indentation inside them included): JS/TS template literals with
      ``${}`` (nested strings, templates and braces inside), Java text blocks, Kotlin/Scala/Dart/Swift triple
      quotes, Kotlin/Dart/Scala ``${}`` and Swift ``\\()`` interpolation, Rust raw strings ``r#"..."#`` (and
      ``b``/``c``/``br``), Go raw backtick strings, C# verbatim ``@""``, interpolated ``$""`` and raw ``\"\"\"``
      strings, C++ raw strings ``R"d(...)d"``, Swift extended delimiters ``#"..."#``, Dart raw ``r''``, PHP
      single/double/backtick strings and heredoc/nowdoc. Char literals where ``'`` is not a string quote (strictly
      one character or escape, so Rust lifetimes and Scala symbols stay words); C/C++ digit separators
      (``1'000``). Backtick identifiers (Kotlin, Scala, Swift) are single tokens.
    * JS regex literals, by the previous token (after an operator, ``(``, ``,``, ``return``... a ``/`` starts a
      regex if it closes on the same line). JSX (in .js/.jsx/.mjs/.cjs/.tsx) is parsed as elements, attributes,
      ``{}`` expressions and text; a ``<`` that does not parse as an element (``<T,>()``, comparisons) backtracks
      to an operator.
    * Preprocessor directives (C/C++ and C# ``#`` at line start, Swift ``#if/#else/#elseif/#endif``) run to the
      end of the line, ``\\``-newline continuing them; inside them every space is kept (``#define F (x)`` is not
      ``#define F(x)``) and unmatched quotes are tolerated (``#error don't``).
    * PHP files start in HTML mode: text outside ``<?php ... ?>`` becomes words (runs of blanks collapsed).

Flow
    * Tag policy: ``§<n> `` before each line that starts a statement; continuation lines are joined untagged.
      Measured on 56 popular files (2.2M chars): flow = 0.90 of the source's chars with this policy, 0.94 with a
      tag on every line, 0.84 with only the forced tags below (no usable line numbers); so statement tags cost
      ~7% over the minimum and save ~4% over tagging every line. A line continues the previous one when that line
      ends in ``( [ ,`` or a binary/assignment operator, or when it starts with ``.``/``?.``/``&&``/``||``/``??``/
      ``?:``, a closing ``) ] }``, or (newline-insensitive languages only) a binary operator, or an Allman ``{``
      after a header. A tag is forced after a line comment (unless the next line is also a line comment: the run
      reads as one comment), around each directive, and after ``MAX_UNTAGGED`` untagged lines in a row, so a line
      number is never far from a tag. Tags always sit on real line starts, never inside a token, so in
      newline-sensitive languages (Go, Kotlin, Swift, Scala, JS ASI) a tag also stands for the newline.
    * Spaces between tokens are dropped except: between two word-like tokens (identifiers, keywords, numbers,
      ``@Ann``, ``'a``, ``$x``, ``\\Ns``); between a literal and a word or another literal (``"a" "b"``, string
      prefixes and suffixes); between two operator characters (``a - -b``, ``x < -1``, ``> >``, ``/ /``, ``& &``);
      before ``.`` after a word or number (``foo .bar``, ``1 .x``) and after ``.`` before a digit; between a
      keyword and an operator (``return -1``); next to comments and JSX/HTML text; inside directives. Swift needs
      balanced whitespace around operators: an operator keeps its spaces unless both can go, and ``?``/``!``
      always keep theirs.
    * Comments are kept; runs of whitespace inside them become one space (the renderer collapses them anyway, code
      examples in doc comments included).
"""
from __future__ import annotations

import bisect
import os
import re
from dataclasses import dataclass

from .base import LEVEL, register, tag

MAX_UNTAGGED = 8            # force a tag after this many untagged line joins in a row


class LexError(ValueError):
    pass


class _JSXFail(Exception):
    pass


@dataclass(frozen=True)
class Lang:
    name: str
    strings: str                  # string scanner key
    nested: bool = False          # nested block comments
    hash_comment: bool = False    # PHP: '#' starts a line comment
    pp: str = ""                  # "c": '#' at line start is a directive; "swift": #if/#elseif/#else/#endif
    semi: bool = True             # newline never ends a statement by itself: leading binary operators continue
    allman: bool = True           # a line starting with '{' after a header continues it
    regex: bool = False
    jsx: bool = False
    php: bool = False
    swift_ops: bool = False       # balanced whitespace around operators
    digit_sep: bool = False


LANGS = {l.name: l for l in (
    Lang("javascript", "js", regex=True, jsx=True),
    Lang("typescript", "js", regex=True),
    Lang("tsx", "js", regex=True, jsx=True),
    Lang("java", "java"),
    Lang("kotlin", "kotlin", nested=True, semi=False, allman=False),
    Lang("scala", "scala", nested=True, semi=False, allman=False),
    Lang("c", "c", pp="c", digit_sep=True),
    Lang("cpp", "c", pp="c", digit_sep=True),
    Lang("csharp", "cs", pp="c"),
    Lang("go", "go", semi=False, allman=False),
    Lang("rust", "rust", nested=True),
    Lang("swift", "swift", nested=True, pp="swift", semi=False, allman=False, swift_ops=True),
    Lang("php", "php", hash_comment=True, php=True),
    Lang("dart", "dart", nested=True),
)}


class Tok:
    __slots__ = ("s", "e", "kind", "pp", "lc")

    def __init__(self, s, e, kind, pp=0, lc=False):
        # kind: w word, n number, o operator, p punctuation, s literal, c comment, t text (JSX/HTML), x JSX/PHP tag
        self.s, self.e, self.kind, self.pp, self.lc = s, e, kind, pp, lc

    def __repr__(self):
        return f"Tok({self.s},{self.e},{self.kind!r})"


_OPS = sorted(set("""
>>>= ... ..< ..= <<= >>= >>> === !== **= &&= ||= ??= <=> ?-> ->* &^=
:: -> => ++ -- && || == != <= >= << >> += -= *= /= %= &= |= ^= ?. ?? ?: .. !! ** .* := <- &^ |>
+ - * / % = & | ^ ! < > ? : . ~
""".split()), key=len, reverse=True)
_OP = re.compile("|".join(map(re.escape, _OPS)))
OPCH = frozenset("+-*/%=&|^!<>?:.~")
_WS = re.compile(r"\s+")
_NUM = re.compile(r"(?:\d|\.\d)(?:[eEpP][+-](?=\d)|\w|\.(?=\d))*")
_NUM_SEP = re.compile(r"(?:\d|\.\d)(?:[eEpP][+-](?=\d)|\w|\.(?=\d)|'(?=\w))*")
_WORD = re.compile(r"[@#$]*[^\W\d][\w$]*|\$[\w$]*")
_CHAR = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F_]{1,8}\}|u+[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{1,8}|[0-7]{1,3}"
                   r"|[^\n])|[^'\\\n])'")
_CHAR_C = re.compile(r"'(?:\\(?:x[0-9a-fA-F]+|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|[0-7]{1,3}|[^\n])|[^'\\\n]){1,4}'")
_KT_NOT = re.compile(r"!(?:is|in)(?![\w$])")
_LIFETIME = re.compile(r"'[^\W\d]\w*")
_BACKTICK_ID = re.compile(r"`[^`\n]+`")
_CPP_RAW = re.compile(r"(?:u8|u|U|L)?R\"([^ ()\\\t\n\v\f\r]{0,16})\(")
_RUST_RAW = re.compile(r"(?:b|c)?r(#*)\"")
_SWIFT_RAW = re.compile(r"(#+)(\"\"\"|\")")
_CS_RAW = re.compile(r"(\$*)(\"{3,})")
_SWIFT_PP = re.compile(r"#(?:if|elseif|else|endif|sourceLocation)\b")
_HEREDOC = re.compile(r"<<<[ \t]*(['\"]?)([A-Za-z_]\w*)\1\r?\n")
_PHP_OPEN = re.compile(r"<\?(?:php\b|=|(?=\s))")
_JSX_OPEN = re.compile(r"<\s*([A-Za-z_$][\w$.:-]*)?")
_JSX_ATTR = re.compile(r"[A-Za-z_$][\w$:.-]*")
_JSX_TEXT_END = re.compile(r"[<{]")
_JSX_CLOSE = re.compile(r"</\s*([A-Za-z_$][\w$.:-]*)?\s*>")
_REGEX_KW = frozenset("return typeof instanceof in of new delete void throw case do else yield await".split())
_JSX_KW = frozenset("return yield default case await else".split())
_KEYWORDS = frozenset("""return throw case yield await typeof delete void new in of instanceof else do sizeof go
defer is as not and or echo print include require""".split())
_NO_ALLMAN = frozenset("return break continue throw yield".split())

OPEN_END = frozenset("""( [ , . ?. = + - * / % && || & | ^ == != === !== <= >= -> => += -= *= /= %= &= |= ^= <<= >>=
>>>= **= ??= &&= ||= ?? ?: :: << := <- ** |>""".split())
LEAD_ALL = frozenset(". ?. && || ?? ?:".split())
LEAD_SEMI = LEAD_ALL | frozenset("""+ - / % = == != === !== < > <= >= | ^ ? : -> => += -= *= /= %= &= |= ^= << >>
>>> ** <=> |> ?->""".split())


def _wordish(ch: str) -> bool:
    return ch.isalnum() or ch in "_$@#'`\\" or ord(ch) > 127


class Lexer:
    def __init__(self, lang: Lang, src: str):
        self.L, self.src, self.n = lang, src, len(src)
        self.toks: list[Tok] = []
        self.pp = 0
        self.pp_seq = 0

    # ------------------------------------------------------------------ driver
    def run(self) -> list[Tok]:
        if self.L.php:
            self._html(0)
        else:
            self.code(0, None)
        return self.toks

    def emit(self, s, e, kind, lc=False) -> Tok:
        t = Tok(s, e, kind, self.pp, lc)
        self.toks.append(t)
        return t

    def _html(self, i):
        src, n = self.src, self.n
        while i < n:
            m = _PHP_OPEN.search(src, i)
            k = m.start() if m else n
            for w in re.finditer(r"\S+", src[i:k]):
                self.emit(i + w.start(), i + w.end(), "t")
            if not m:
                return
            self.emit(m.start(), m.end(), "x")
            i = self.code(m.end(), None)

    # ------------------------------------------------------------------ code
    def code(self, i, stop):
        """Lex code from i. With ``stop`` (``}`` or ``)``), return after the unmatched closer (emitted)."""
        src, n, L = self.src, self.n, self.L
        depth = 0
        prev = None
        open_c = "(" if stop == ")" else "{"
        while True:
            m = _WS.match(src, i)
            if m:
                if self.pp and "\n" in m.group() and not self._pp_continues(i, m.group()):
                    self.pp = 0
                i = m.end()
            if i >= n:
                if stop:
                    raise LexError("unclosed interpolation or expression")
                return i
            c = src[i]
            if stop and c == stop and depth == 0:
                self.emit(i, i + 1, "p")
                return i + 1
            if L.php and src.startswith("?>", i):
                if stop:
                    raise LexError("?> inside an expression")
                self.pp = 0
                self.emit(i, i + 2, "x")
                return i + 2
            # comments
            if (src.startswith("//", i) or (L.hash_comment and c == "#" and not src.startswith("#[", i))
                    or (i == 0 and src.startswith("#!") and not src.startswith("#!["))):
                e = src.find("\n", i)
                e = n if e < 0 else e
                if L.php:
                    q = src.find("?>", i, e)
                    e = q if q >= 0 else e
                e2 = e
                while e2 > i and src[e2 - 1].isspace():
                    e2 -= 1
                self.emit(i, e2, "c", lc=True)
                i = e
                continue
            if src.startswith("/*", i):
                e = self._block_comment(i)
                self.emit(i, e, "c")
                i = e
                continue
            # directives
            if (L.pp and c == "#" and not self.pp and self._at_line_start(i)
                    and (L.pp == "c" or _SWIFT_PP.match(src, i))):
                self.pp_seq += 1
                self.pp = self.pp_seq
            # literals
            r = self._string(i)
            if r:
                e, kind = r
                prev = self.emit(i, e, kind)
                i = e
                continue
            if L.regex and c == "/" and (self._regex_ok(prev) or (src[prev.s:prev.e] == "}" and self._at_line_start(i))):
                e = self._regex(i)
                if e:
                    prev = self.emit(i, e, "s")
                    i = e
                    continue
            if (L.jsx and c == "<" and i + 1 < n and (src[i + 1].isalpha() or src[i + 1] in ">_$")
                    and self._jsx_ok(prev)):
                mark = len(self.toks)
                try:
                    i = self._jsx(i)
                    prev = self.toks[-1]
                    continue
                except (_JSXFail, LexError, RecursionError):
                    del self.toks[mark:]
            m = (_NUM_SEP if L.digit_sep else _NUM).match(src, i)
            if m:
                prev = self.emit(i, m.end(), "n")
                i = m.end()
                continue
            m = _WORD.match(src, i) or (L.name == "kotlin" and _KT_NOT.match(src, i))
            if m:
                prev = self.emit(i, m.end(), "w")
                i = m.end()
                continue
            m = _OP.match(src, i)
            if m:
                prev = self.emit(i, m.end(), "o")
                i = m.end()
                continue
            if stop:
                if c == open_c:
                    depth += 1
                elif c == stop:
                    depth -= 1
            prev = self.emit(i, i + 1, "p")
            i += 1

    def _pp_continues(self, i, ws):
        last = self.toks[-1] if self.toks else None
        return (last is not None and last.e == i and self.src[last.s:last.e] == "\\" and ws.count("\n") == 1)

    def _at_line_start(self, i):
        ls = self.src.rfind("\n", 0, i) + 1
        return not self.src[ls:i].strip(" \t\f\v\r")

    def _block_comment(self, i):
        src = self.src
        if not self.L.nested:
            e = src.find("*/", i + 2)
            if e < 0:
                raise LexError("unterminated block comment")
            return e + 2
        depth, j = 0, i
        while True:
            a, b = src.find("/*", j), src.find("*/", j)
            if b < 0:
                raise LexError("unterminated block comment")
            if 0 <= a < b:
                depth += 1
                j = a + 2
            else:
                depth -= 1
                j = b + 2
                if depth == 0:
                    return j

    # ------------------------------------------------------------------ JS regex and JSX
    def _text(self, t):
        return self.src[t.s:t.e]

    def _regex_ok(self, prev):
        if prev is None:
            return True
        t = self._text(prev)
        if prev.kind == "p":
            return t not in ")]}"
        if prev.kind == "o":
            return t not in ("++", "--")
        if prev.kind == "w":
            return t in _REGEX_KW
        return False

    def _regex(self, i):
        src, n = self.src, self.n
        j, cls = i + 1, False
        if j < n and src[j] in "/*":
            return None
        while j < n:
            c = src[j]
            if c == "\n":
                return None
            if c == "\\":
                j += 2
                continue
            if cls:
                cls = c != "]"
            elif c == "[":
                cls = True
            elif c == "/":
                j += 1
                while j < n and (src[j].isalnum() or src[j] == "_"):
                    j += 1
                return j
            j += 1
        return None

    def _jsx_ok(self, prev):
        if prev is None:
            return True
        t = self._text(prev)
        if prev.kind == "p":
            return t in "([{,;"
        if prev.kind == "o":
            return t not in ("++", "--")
        if prev.kind == "w":
            return t in _JSX_KW
        return False

    def _skip_ws(self, j):
        m = _WS.match(self.src, j)
        return m.end() if m else j

    def _jsx(self, i):
        src, n = self.src, self.n
        m = _JSX_OPEN.match(src, i)
        name = m.group(1) or ""
        if not name and not src.startswith(">", m.end()):
            raise _JSXFail
        self.emit(i, m.end(), "x")
        j = m.end()
        while True:                                   # attributes
            j = self._skip_ws(j)
            if j >= n:
                raise _JSXFail
            if src.startswith("/>", j):
                self.emit(j, j + 2, "x")
                return j + 2
            c = src[j]
            if c == ">":
                if src.find("</" + name, j) < 0:      # no closing tag anywhere: a generic, not JSX (fail fast)
                    raise _JSXFail
                self.emit(j, j + 1, "x")
                j += 1
                break
            if c == "{":
                self.emit(j, j + 1, "p")
                j = self.code(j + 1, "}")
                continue
            m = _JSX_ATTR.match(src, j)
            if not m:
                raise _JSXFail
            self.emit(j, m.end(), "w")
            j = m.end()
            j2 = self._skip_ws(j)
            if src.startswith("=", j2) and not src.startswith("==", j2):
                self.emit(j2, j2 + 1, "o")
                j = self._skip_ws(j2 + 1)
                c = src[j] if j < n else ""
                if c in ("\"", "'"):
                    e = src.find(c, j + 1)
                    if e < 0:
                        raise _JSXFail
                    self.emit(j, e + 1, "s")
                    j = e + 1
                elif c == "{":
                    self.emit(j, j + 1, "p")
                    j = self.code(j + 1, "}")
                elif c == "<":
                    j = self._jsx(j)
                else:
                    raise _JSXFail
        while True:                                   # children
            mk = _JSX_TEXT_END.search(src, j)
            k = mk.start() if mk else n
            for w in re.finditer(r"\S+", src[j:k]):
                self.emit(j + w.start(), j + w.end(), "t")
            if k >= n:
                raise _JSXFail
            if src[k] == "{":
                self.emit(k, k + 1, "p")
                j = self.code(k + 1, "}")
                continue
            if src.startswith("</", k):
                m = _JSX_CLOSE.match(src, k)
                if not m or (m.group(1) or "") != name:
                    raise _JSXFail
                self.emit(k, m.end(), "x")
                return m.end()
            j = self._jsx(k)

    # ------------------------------------------------------------------ strings
    def _fail(self, what):
        if self.pp:
            return None                               # tolerated inside directives (#error don't)
        raise LexError(what)

    def _interp(self, j, mode):
        src = self.src
        if mode == "${" and src.startswith("${", j):
            start = j + 2
        elif mode == "swift" and src.startswith("\\(", j):
            mark = len(self.toks)
            e = self.code(j + 2, ")")
            del self.toks[mark:]
            return e
        elif mode == "php" and (src.startswith("{$", j) or src.startswith("${", j)):
            start = j + 1 if src[j] == "{" else j + 2
        elif mode == "cs" and src.startswith("{", j):
            if src.startswith("{{", j):
                return j + 2
            start = j + 1
        else:
            return None
        mark = len(self.toks)
        e = self.code(start, "}")
        del self.toks[mark:]
        return e

    def _q(self, i, q, esc=True, multi=False, interp=None, dbl=False):
        """i at the opening quote q; return the index after the closing quote."""
        src, n = self.src, self.n
        j = i + 1
        while j < n:
            if interp:
                k = self._interp(j, interp)
                if k:
                    j = k
                    continue
            c = src[j]
            if esc and c == "\\":
                j += 2
                continue
            if c == q:
                if dbl and src.startswith(q, j + 1):
                    j += 2
                    continue
                return j + 1
            if c == "\n" and not multi:
                return self._fail("unterminated string")
            j += 1
        return self._fail("unterminated string")

    def _tq(self, i, qq, esc=True, interp=None, run=False):
        """i at the first quote of an opening triple (or longer) quote qq."""
        src, n = self.src, self.n
        j = i + len(qq)
        while j < n:
            if interp:
                k = self._interp(j, interp)
                if k:
                    j = k
                    continue
            if esc and src[j] == "\\":
                j += 2
                continue
            if src.startswith(qq, j):
                e = j + len(qq)
                if run:
                    while e < n and src[e] == qq[0]:
                        e += 1
                return e
            j += 1
        raise LexError("unterminated multi-line string")

    def _closing(self, i, start, closer):
        e = self.src.find(closer, start)
        if e < 0:
            raise LexError("unterminated raw string")
        return e + len(closer)

    def _string(self, i):
        """-> (end, kind) if a literal starts at i, else None."""
        src, key = self.src, self.L.strings
        c = src[i]
        free = i == 0 or not (src[i - 1].isalnum() or src[i - 1] in "_$")   # not inside a word
        e = None
        if key == "js":
            if c in "'\"":
                e = self._q(i, c)
            elif c == "`":
                e = self._q(i, "`", multi=True, interp="${")
        elif key == "java":
            if src.startswith('"""', i):
                e = self._tq(i, '"""')
            elif c == '"':
                e = self._q(i, '"')
            elif c == "'":
                return self._char(i, _CHAR)
        elif key == "kotlin":
            if src.startswith('"""', i):
                e = self._tq(i, '"""', esc=False, interp="${", run=True)
            elif c == '"':
                e = self._q(i, '"', interp="${")
            elif c == "'":
                return self._char(i, _CHAR)
            elif c == "`":
                return self._backtick_id(i)
        elif key == "scala":
            interp = "${" if i > 0 and (src[i - 1].isalnum() or src[i - 1] == "_") else None
            if src.startswith('"""', i):
                e = self._tq(i, '"""', esc=False, interp=interp, run=True)
            elif c == '"':
                e = self._q(i, '"', interp=interp)
            elif c == "'":
                m = _CHAR.match(src, i)
                if m:
                    return m.end(), "s"
                m = _LIFETIME.match(src, i)            # 'symbol
                return (m.end(), "w") if m else None
            elif c == "`":
                return self._backtick_id(i)
        elif key == "c":
            if c in "uULR" and free:
                m = _CPP_RAW.match(src, i)
                if m:
                    return self._closing(i, m.end(), ")" + m.group(1) + '"'), "s"
            if c == '"':
                e = self._q(i, '"')
            elif c == "'":
                return self._char(i, _CHAR_C)
        elif key == "cs":
            if c in "$\"" and (c == '"' or free):
                m = _CS_RAW.match(src, i)
                if m:
                    return self._closing(i, m.end(), m.group(2)), "s"
            if c in "$@" and free:
                m = re.compile(r"(\$@|@\$|@|\$)\"").match(src, i)
                if m:
                    p = m.group(1)
                    if "@" in p:
                        e = self._q(m.end() - 1, '"', esc=False, multi=True, dbl=True,
                                    interp="cs" if "$" in p else None)
                    else:
                        e = self._q(m.end() - 1, '"', interp="cs")
                    return (e, "s") if e else None
            if c == '"':
                e = self._q(i, '"')
            elif c == "'":
                return self._char(i, _CHAR)
        elif key == "go":
            if c == "`":
                e = self._closing(i, i + 1, "`")
            elif c == '"':
                e = self._q(i, '"')
            elif c == "'":
                return self._char(i, _CHAR)
        elif key == "rust":
            if c in "bcr" and free:
                m = _RUST_RAW.match(src, i)
                if m:
                    return self._closing(i, m.end(), '"' + m.group(1)), "s"
                if c in "bc" and src.startswith('"', i + 1):
                    e = self._q(i + 1, '"', multi=True)
                    return e, "s"
                if c == "b" and src.startswith("'", i + 1):
                    m = _CHAR.match(src, i + 1)
                    if m:
                        return m.end(), "s"
            if c == '"':
                e = self._q(i, '"', multi=True)
            elif c == "'":
                m = _CHAR.match(src, i)
                if m:
                    return m.end(), "s"
                m = _LIFETIME.match(src, i)
                return (m.end(), "w") if m else None
        elif key == "swift":
            if c == "#":
                m = _SWIFT_RAW.match(src, i)
                if m:
                    return self._closing(i, m.end(), m.group(2) + m.group(1)), "s"
            if src.startswith('"""', i):
                e = self._tq(i, '"""', interp="swift")
            elif c == '"':
                e = self._q(i, '"', interp="swift")
            elif c == "`":
                return self._backtick_id(i)
        elif key == "php":
            if c == "'":
                e = self._q(i, "'", multi=True)
            elif c in "\"`":
                e = self._q(i, c, multi=True, interp="php")
            elif c == "<" and src.startswith("<<<", i):
                m = _HEREDOC.match(src, i)
                if m:
                    end = re.compile(r"^[ \t]*" + re.escape(m.group(2)) + r"(?![A-Za-z0-9_])", re.M).search(src, m.end())
                    if not end:
                        raise LexError("unterminated heredoc")
                    return end.end(), "s"
        elif key == "dart":
            if c == "r" and free and i + 1 < self.n and src[i + 1] in "'\"":
                q = src[i + 1]
                if src.startswith(q * 3, i + 1):
                    return self._tq(i + 1, q * 3, esc=False), "s"
                e = self._q(i + 1, q, esc=False)
                return (e, "s") if e else None
            if src.startswith("'''", i) or src.startswith('"""', i):
                return self._tq(i, c * 3, interp="${"), "s"
            if c in "'\"":
                e = self._q(i, c, interp="${")
        return (e, "s") if e else None

    def _char(self, i, rx):
        m = rx.match(self.src, i)
        return (m.end(), "s") if m else None

    def _backtick_id(self, i):
        m = _BACKTICK_ID.match(self.src, i)
        return (m.end(), "w") if m else None


# ---------------------------------------------------------------------------------------------------------- flow

def _gap(a: Tok, b: Tok, at: str, bt: str) -> str:
    """Whitespace to keep between two tokens that had whitespace between them in the source."""
    if a.pp and a.pp == b.pp:
        return " "
    if a.kind in "ctx" or b.kind in "ctx":
        return " "
    x = "a" if a.kind == "w" else at[-1]           # word tokens (incl. Kotlin `!is`, `!in`) read as words
    y = "a" if b.kind == "w" else bt[0]
    if _wordish(x) and _wordish(y):
        return " "
    if (a.kind == "s" and (b.kind == "s" or _wordish(y))) or (b.kind == "s" and _wordish(x)):
        return " "
    if x in OPCH and y in OPCH:
        return " "
    if y == "." and _wordish(x):
        return " "
    if x == "." and y.isdigit():
        return " "
    if a.kind == "w" and at in _KEYWORDS and y in OPCH:
        return " "
    return ""


def _continues(L: Lang, src: str, last_code: Tok | None, t: Tok, tt: str) -> bool:
    """True if the line starting with token t continues the previous line (no tag needed)."""
    if last_code is not None:
        lt = src[last_code.s:last_code.e]
        if last_code.kind in "op" and lt in OPEN_END:
            return True
    if t.kind == "p" and tt in ")]}":
        return True
    if t.kind == "o":
        if tt in LEAD_ALL:
            return True
        if L.semi and tt in LEAD_SEMI:
            return True
        if L.swift_ops and t.e < len(src) and src[t.e].isspace() and tt not in ("!", "&", "++", "--", "~"):
            return True                                # binary operator (whitespace on both sides)
    if L.allman and t.kind == "p" and tt == "{" and last_code is not None:
        lt = src[last_code.s:last_code.e]
        if lt == ")" or lt == ">" or (last_code.kind == "w" and lt not in _NO_ALLMAN):
            return True
    return False


def _indent_unit(src: str) -> int:
    """Spaces per indentation level (gcd of the space-indented lines; 4 if unclear)."""
    import math
    g = 0
    for line in src.split("\n"):
        n = len(line) - len(line.lstrip(" "))
        if n and line.strip() and not line.lstrip().startswith("*"):    # skip the ` * ` of block comments
            g = math.gcd(g, n)
    return g if g in (2, 3, 4, 8) else 4


def brace_flow(src: str, lang: Lang | str, policy: str = "stmt", cap: int | None = MAX_UNTAGGED,
               join_comments: bool = True, keep_spaces: bool = False, levels: bool = False) -> str:
    """policy: "stmt" (tag statement starts, see module doc), "all" (tag every line that starts with a token),
    "first" (tag only where needed for correctness: after line comments and around directives)."""
    L = LANGS[lang] if isinstance(lang, str) else lang
    toks = Lexer(L, src).run()
    if not toks:
        return ""
    nls = [k for k, ch in enumerate(src) if ch == "\n"]
    texts = [src[t.s:t.e] for t in toks]
    gaps: list[str | None] = [None] * len(toks)       # separator before token k (None: no whitespace in source)
    unit = _indent_unit(src) if levels else 0

    def lv(pos: int) -> str:
        """`»` per indentation level of the line starting at token position `pos` (edit-safe mode: indentation is
        cosmetic in these languages, but Claude Code's Edit needs it verbatim, so the reader must be able to see it)."""
        if not levels:
            return ""
        i = bisect.bisect_left(nls, pos)
        lead = src[(nls[i - 1] + 1 if i else 0):pos]
        if lead.strip():
            return ""
        return LEVEL * (lead.count("\t") + (len(lead.replace("\t", "")) // unit if unit else 0))

    gaps[0] = tag(bisect.bisect_left(nls, toks[0].s) + 1) + lv(toks[0].s)
    last_code = toks[0] if toks[0].kind != "c" else None
    since = 0
    for k in range(1, len(toks)):
        p, t = toks[k - 1], toks[k]
        if p.e < t.s:
            ws = src[p.e:t.s]
            if "\n" in ws:
                if p.lc:
                    need = not (join_comments and t.lc)
                elif p.pp or t.pp:
                    need = not (p.pp and p.pp == t.pp)
                elif policy == "all":
                    need = True
                elif policy == "first":
                    need = False
                else:
                    need = not _continues(L, src, last_code, t, texts[k])
                if not need and cap is not None and since >= cap and not (p.pp and p.pp == t.pp):
                    need = True
                if need:
                    gaps[k] = " " + tag(bisect.bisect_left(nls, t.s) + 1) + lv(t.s)
                    since = 0
                else:
                    gaps[k] = _gap(p, t, texts[k - 1], texts[k])
                    since += 1
            else:
                gaps[k] = " " if keep_spaces else _gap(p, t, texts[k - 1], texts[k])
        if t.kind != "c":
            last_code = t
    if L.swift_ops:
        for k, t in enumerate(toks):
            if t.kind != "o" or texts[k] in (".", "->", ":"):
                continue
            right = gaps[k + 1] if k + 1 < len(toks) else None
            left = gaps[k]
            always = texts[k] in ("?", "!")
            if left is not None and right is not None and not always and left == "" and right == "":
                continue
            if left == "":
                gaps[k] = " "
            if right == "":
                gaps[k + 1] = " "
    out = []
    for k, t in enumerate(toks):
        if gaps[k]:
            out.append(gaps[k])
        out.append(re.sub(r"\s+", " ", texts[k]) if t.kind == "c" else texts[k])
    return "".join(out)


def token_stream(src: str, lang: Lang | str) -> list[tuple[str, str]]:
    """Non-whitespace token stream for checks: (kind, text), comments with whitespace runs collapsed and runs of
    line comments merged (the flow joins consecutive comment lines)."""
    L = LANGS[lang] if isinstance(lang, str) else lang
    out: list[tuple[str, str]] = []
    for t in Lexer(L, src).run():
        txt = src[t.s:t.e]
        if t.kind == "c":
            txt = re.sub(r"\s+", " ", txt).strip()
            if t.lc and out and out[-1][0] == "lc":
                out[-1] = ("lc", out[-1][1] + " " + txt)
                continue
            out.append(("lc" if t.lc else "c", txt))
        else:
            out.append((t.kind, txt))
    return out


def flow_tokens_match(src: str, flow: str, lang: Lang | str) -> bool:
    """The flow (tags read as newlines) lexes to the same token stream as the source."""
    return token_stream(src, lang) == token_stream(re.sub(r" ?§\d+ ", "\n", flow), lang)


# ------------------------------------------------------------------------------------------------- registration
# cpt: chars per Claude token of the original text, counted as in SPEC.md on 3-5 popular files per language
# (56 files, 2.2M chars; savings at 88.2 px^2/char = text tokens / page tokens: 3.6-4.4x, 3.84x overall).

def _make(name):
    def fn(text: str) -> str:
        # Edit-safe by default (see python_family.minify_python): spaces kept inside lines, every line tagged
        safe = os.environ.get("VBHOOK_TIGHT", "0") != "1"
        return brace_flow(text, name, policy="all" if safe else "stmt", keep_spaces=safe, levels=safe)
    fn.__name__ = f"minify_{name}"
    return fn


_REG = {
    "javascript": ((".js", ".mjs", ".cjs", ".jsx"), 2.53),
    "typescript": ((".ts", ".mts", ".cts"), 2.23),
    "tsx": ((".tsx",), 2.49),
    "java": ((".java",), 2.57),
    "kotlin": ((".kt", ".kts"), 2.51),
    "scala": ((".scala",), 2.42),
    "c": ((".c", ".h"), 2.1),
    "cpp": ((".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++", ".ipp", ".inl"), 2.63),
    "csharp": ((".cs",), 2.78),
    "go": ((".go",), 2.28),
    "rust": ((".rs",), 2.45),
    "swift": ((".swift",), 2.73),
    "php": ((".php", ".phtml"), 2.51),
    "dart": ((".dart",), 2.49),
}
for _name, (_exts, _cpt) in _REG.items():
    register(_name, exts=_exts, cpt=_cpt)(_make(_name))
