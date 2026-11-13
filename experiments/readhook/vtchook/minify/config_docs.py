"""Minifiers for config, data and documentation files (see SPEC.md and base.py for the flow contract).

Kinds: lockfile (always text), yaml, toml, ini, json, jsonl, xml, html, markdown, rst, text, csv, tsv, sql.

Common conventions
* Lines are split on "\\n" only (as Read counts them); a trailing "\\r" is dropped.
* A flow never contains tabs or newlines (the page renderer draws only spaces): any tab left in kept content becomes
  one space.
* ``rebuild(flow, ...)`` puts every ``§<n>`` segment back on line n (blank lines restored from the gaps); the tests
  use it for the parser round-trips (yaml.safe_load, tomllib, configparser, csv).
* Files that are mostly digits or hashes return "" (the caller keeps them as text): exact digits and hashes are the
  riskiest thing to read from an image, and there is no language prior to repair a misread.

Measured on 76 real files (2026-11-11; cpt = chars per token of the original, Haiku count; pages at 88.2 px2/char;
x = text tokens / page tokens, files the guard keeps as text excluded): yaml 2.39 cpt x3.35, toml 2.13 x3.28,
ini 2.53 x2.74, json 2.13 x5.46, jsonl 2.30 x3.72, xml 2.52 x3.79, html 2.30 x3.39, markdown 2.67 x3.06,
rst 2.58 x3.36, text 1.88 x4.11, csv 1.64 x5.12, sql 1.96 x4.25. Against Read's numbered text the savings are
10-25% higher. One page costs at least 138 tokens, so files under ~1.5k characters do not pay.
"""
from __future__ import annotations

import bisect
import json
import re
from collections import Counter

from .base import LEVEL, register

# --------------------------------------------------------------------------------------------- shared helpers

_SEG_RE = re.compile(r"§(\d+) (»*)(.*?)(?= §\d+ |$)", re.S)
_WS = re.compile(r"\s+")


def _lines(text: str) -> list[str]:
    return [ln[:-1] if ln.endswith("\r") else ln for ln in text.split("\n")]


def _seg(n: int, lvl: int, body: str) -> str:
    return f"§{n} " + LEVEL * lvl + body


def _flow(parts) -> str:
    out = " ".join(p for p in parts if p)
    return out.replace("\t", " ").replace("\r", " ").replace("\n", " ").replace("\f", " ").replace("\v", " ")


def _indent(s: str) -> int:
    """Width of the leading whitespace (tabs count 4)."""
    n = 0
    for c in s:
        if c == " ":
            n += 1
        elif c == "\t":
            n += 4 - n % 4
        else:
            break
    return n


def _collapse(s: str) -> str:
    return _WS.sub(" ", s).strip()


def segments(flow: str) -> list[tuple[int, int, str]]:
    """-> [(line, levels, rest)] for every tagged segment of a flow."""
    return [(int(m.group(1)), len(m.group(2)), m.group(3)) for m in _SEG_RE.finditer(flow)]


def rebuild(flow: str, indent=lambda lvl: "    " * lvl) -> str:
    """Put each segment back on its original line (for round-trip checks; not used by the hook)."""
    segs = segments(flow)
    if not segs:
        return ""
    out = [""] * max(n for n, _, _ in segs)
    for n, lvl, rest in segs:
        out[n - 1] = indent(lvl) + rest
    return "\n".join(out) + "\n"


_TOKEN = re.compile(r"[A-Za-z0-9+/=_-]{16,}")


def _is_hash(tok: str) -> bool:
    """Hex digests (16+ hex chars with digits and letters) and base64 blobs (40+ chars, mixed case, many digits)."""
    t = tok.rstrip("=")
    nd = sum(c.isdigit() for c in t)
    if re.fullmatch(r"[0-9a-f]+|[0-9A-F]+", t):
        return 0 < nd < len(t)
    return len(t) >= 40 and nd >= 0.1 * len(t) and any(c.islower() for c in t) and any(c.isupper() for c in t)


def hash_share(text: str) -> float:
    nw = len(re.sub(r"\s+", "", text)) or 1
    return sum(len(m.group(0)) for m in _TOKEN.finditer(text) if _is_hash(m.group(0))) / nw


def digit_share(text: str) -> float:
    nw = re.sub(r"\s+", "", text)
    return sum(c.isdigit() for c in nw) / (len(nw) or 1)


def too_numeric(text: str, digits: float = 0.30, hashes: float = 0.25) -> bool:
    """True when digits or hashes dominate the non-whitespace text (keep such files as text)."""
    if len(text) < 200:
        return False
    return digit_share(text) > digits or hash_share(text) > hashes


class _Levels:
    """Map indentation columns to nesting levels (a stack of open indents, as Python's tokenizer does)."""

    def __init__(self):
        self.stack = [0]

    def push(self, ind: int) -> int:
        while len(self.stack) > 1 and ind < self.stack[-1]:
            self.stack.pop()
        if ind > self.stack[-1]:
            self.stack.append(ind)
        return len(self.stack) - 1

    def peek(self, ind: int) -> int:
        """Level for a line that opens nothing (a comment): never deeper than the innermost open level."""
        return max(0, sum(1 for s in self.stack if s <= ind) - 1)

    def reset(self):
        self.stack = [0]


# ---------------------------------------------------------------------------------------------------- lockfiles
# Lock files are hashes, versions and URLs: pages would save tokens, but nobody reads them for meaning and every
# misread digit is an error. Returning "" keeps them as text.

LOCK_NAMES = ("package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "bun.lock", "Cargo.lock",
              "poetry.lock", "Pipfile.lock", "uv.lock", "pdm.lock", "composer.lock", "Gemfile.lock", "go.sum",
              "flake.lock", "packages.lock.json", "mix.lock", "pubspec.lock", "Podfile.lock", "gradle.lockfile")


@register("lockfile", names=LOCK_NAMES, cpt=2.02)
def lockfile(text: str) -> str:
    return ""


# --------------------------------------------------------------------------------------------------------- YAML

_YBLOCK = re.compile(r"(?:^|[:?-][ \t]+)(?:[&!]\S*[ \t]+)*([|>])([+-]?)(\d?)([+-]?)[ \t]*$")
_YDASH = re.compile(r"^(?:-(?:[ \t]+|$))+")
_YKEYALIGN = re.compile(r"""^((?:- )*(?:"(?:[^"\\]|\\.)*"|'(?:[^']|'')*'|[^\s#'"\[{][^#]*?)):[ \t]{2,}(?=\S)""")


def _ycomment(s: str) -> int:
    """Index where a YAML comment starts in a line's content, or -1. Quotes open only at scalar starts."""
    i, n, start = 0, len(s), True
    while i < n:
        c = s[i]
        if c == "#" and (i == 0 or s[i - 1] in " \t"):
            return i
        if start and c in "\"'":
            q, i = c, i + 1
            while i < n:
                if q == '"' and s[i] == "\\":
                    i += 2
                    continue
                if s[i] == q:
                    if q == "'" and i + 1 < n and s[i + 1] == "'":
                        i += 2
                        continue
                    break
                i += 1
            i += 1
            start = False
            continue
        if c in " \t":
            i += 1
            continue
        if c in "[{,":
            start = True
        elif c in ":-?" and (i + 1 == n or s[i + 1] in " \t"):
            start = True
        else:
            start = False
        i += 1
    return -1


def _ysplit(content: str) -> tuple[str, str]:
    k = _ycomment(content)
    return (content, "") if k < 0 else (content[:k].rstrip(), content[k:])


def _yblock(code: str):
    """-> (indicator, explicit indent digit, n_dashes, direct) when the line opens a block scalar, else None."""
    m = _YBLOCK.search(code)
    if not m:
        return None
    d = _YDASH.match(code)
    ndash = d.group(0).count("-") if d else 0
    rest = code[d.end():] if d else code
    direct = bool(re.match(r"(?:[&!]\S*[ \t]+)*[|>]", rest))
    return m.group(1), m.group(3), ndash, direct


def yaml_unit(text: str) -> int:
    """The file's indent step (most common push from a non-sequence parent), used for block-scalar bodies."""
    return _yaml(text, want_unit=True)


def _yaml(text: str, want_unit: bool = False):
    lines = _lines(text)
    lv = _Levels()
    diffs: Counter = Counter()
    recs = []                      # (line, kind, lvl, payload)
    prev_dash = False
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        s = raw.strip()
        if not s:
            i += 1
            continue
        lead = raw[: len(raw) - len(raw.lstrip())]
        if "\t" in lead:
            raise ValueError("tab in YAML indentation")
        ind = len(lead)
        if s.startswith("#"):
            recs.append((i + 1, "c", lv.peek(ind), s))
            i += 1
            continue
        top = lv.stack[-1]
        lvl = lv.push(ind)
        if ind > top and not prev_dash:
            diffs[ind - top] += 1
        code, comment = _ysplit(s)
        d = _YDASH.match(code)
        prev_dash = bool(d and code[d.end():])
        if prev_dash:
            lv.push(ind + len(d.group(0)))
        recs.append((i + 1, "s", lvl, (code, comment)))
        blk = _yblock(code)
        i += 1
        if blk:
            _, digit, ndash, direct = blk
            if digit:
                raise ValueError("explicit block indentation indicator")
            if prev_dash and d:
                thr = ind + len(d.group(0)) - (2 if direct else 0)
            else:
                thr = ind
            base_lvl = lvl + 1 + (ndash - (1 if direct else 0) if d else 0)
            base = None
            while i < n:
                b = lines[i]
                if not b.strip():
                    i += 1
                    continue
                bind = len(b) - len(b.lstrip(" "))
                if base is None:
                    if bind <= thr:
                        break
                    base = bind
                elif bind < base:
                    break
                recs.append((i + 1, "b", base_lvl, (bind - base, b[bind:].rstrip())))
                i += 1
    unit = diffs.most_common(1)[0][0] if diffs else 2
    if want_unit:
        return unit
    parts = []
    for ln, kind, lvl, p in recs:
        if kind == "c":
            parts.append(_seg(ln, lvl, p))
        elif kind == "b":
            rel, body = p
            parts.append(_seg(ln, lvl + rel // unit, " " * (rel % unit) + body))
        else:
            code, comment = p
            d = _YDASH.match(code)
            if d:
                code = "- " * d.group(0).count("-") + code[d.end():]
            code = _YKEYALIGN.sub(lambda m: m.group(1) + ": ", code)
            parts.append(_seg(ln, lvl, code + (" " + comment if comment and code else comment)))
    return _flow(parts)


@register("yaml", exts=(".yml", ".yaml"), names=(".clang-format", ".clang-tidy"), cpt=2.39)
def yaml_flow(text: str) -> str:
    if too_numeric(text):
        return ""
    return _yaml(text)


def yaml_rebuild(flow: str, unit: int = 2) -> str:
    """Rebuild YAML from a flow: structure at 2 spaces per level, block-scalar bodies at ``unit`` per level."""
    segs = segments(flow)
    out = [""] * (max(n for n, _, _ in segs) if segs else 0)
    base_lvl = None
    for n, lvl, rest in segs:
        if base_lvl is not None and lvl >= base_lvl:
            out[n - 1] = "  " * base_lvl + " " * (unit * (lvl - base_lvl)) + rest
            continue
        base_lvl = None
        out[n - 1] = "  " * lvl + rest
        if rest.startswith("#"):
            continue
        code, _ = _ysplit(rest)
        blk = _yblock(code)
        if blk:
            _, _, ndash, direct = blk
            base_lvl = lvl + 1 + ((ndash - (1 if direct else 0)) if ndash else 0)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------------------- TOML

def _toml_pieces(text: str):
    """Split TOML into per-line lists of (kind, text), kind in code/str/comment, plus the state at each line start
    ('' normal, 'ml' inside a multi-line string, 'arr' inside brackets)."""
    lines = _lines(text)
    out, starts = [], []
    state, depth, q = "", 0, None          # q: open string delimiter
    for raw in lines:
        starts.append("ml" if q in ('"""', "'''") else ("arr" if depth > 0 else ""))
        pieces, buf, kind = [], "", "str" if q else "code"
        i, n = 0, len(raw)

        def flush(k):
            nonlocal buf
            if buf:
                pieces.append((k, buf))
            buf = ""

        while i < n:
            c = raw[i]
            if q:                                        # inside a string
                if q in ('"', '"""') and c == "\\":
                    buf += raw[i:i + 2]
                    i += 2
                    continue
                if raw.startswith(q, i):
                    if len(q) == 3:
                        r = len(raw[i:]) - len(raw[i:].lstrip(q[0]))
                        r = min(r, 5)
                        buf += raw[i:i + r]
                        i += r
                    else:
                        buf += c
                        i += 1
                    flush("str")
                    q = None
                    continue
                buf += c
                i += 1
                continue
            if c == "#":
                flush("code")
                pieces.append(("comment", raw[i:]))
                i = n
                break
            if c in "\"'":
                flush("code")
                q = c * 3 if raw.startswith(c * 3, i) else c
                buf = q
                i += len(q)
                continue
            if c in "[{":
                depth += 1
            elif c in "]}":
                depth = max(0, depth - 1)
            buf += c
            i += 1
        flush("str" if q else "code")
        if q in ('"', "'"):
            raise ValueError("unterminated single-line TOML string")
        out.append(pieces)
    return lines, out, starts


_TOML_TIGHT = re.compile(r"\s*([=,\[\]{}])\s*")


def _toml_line(pieces, lstrip: bool) -> str:
    s = ""
    for k, t in pieces:
        if k == "code":
            t = _TOML_TIGHT.sub(r"\1", _WS.sub(" ", t))
        elif k == "comment":
            s = s.rstrip() + (" " if s.strip() else "")
            t = _collapse(t)
        s += t
    return (s.lstrip() if lstrip else s).rstrip()


@register("toml", exts=(".toml",), names=("Pipfile",), cpt=2.13)
def toml_flow(text: str) -> str:
    if too_numeric(text):
        return ""
    lines, pieces, starts = _toml_pieces(text)
    parts: list[str] = []
    prev_comment = True
    for no, (raw, pcs, st) in enumerate(zip(lines, pieces, starts), 1):
        if st == "ml":
            body = _toml_line(pcs, lstrip=False)
            if body.strip():
                parts.append(_seg(no, 0, body))
            prev_comment = bool(pcs) and pcs[-1][0] == "comment"
            continue
        body = _toml_line(pcs, lstrip=True)
        if not body:
            continue
        has_comment = pcs[-1][0] == "comment"
        if st == "arr" and not prev_comment and not body.startswith("#") and parts:
            sep = "" if parts[-1][-1:] in "[{," or body[:1] in "]}," else " "
            parts[-1] += sep + body
        else:
            parts.append(_seg(no, 0, body))
        prev_comment = has_comment
    return _flow(parts)


def toml_rebuild(flow: str) -> str:
    return rebuild(flow, indent=lambda lvl: "")


# ---------------------------------------------------------------------------------------------------------- INI

_INI_KEY = re.compile(r"^([^=:]*?)\s*([=:])\s*(.*)$")


@register("ini", exts=(".ini", ".cfg", ".conf", ".editorconfig", ".properties"),
          names=(".pylintrc", "pylintrc", ".flake8", ".coveragerc", ".gitconfig", ".gitmodules", ".npmrc",
                 ".pypirc", "setup.cfg", "tox.ini", ".isort.cfg", "mypy.ini", "pytest.ini"), cpt=2.53)
def ini_flow(text: str) -> str:
    lines = _lines(text)
    if any(ln.rstrip().endswith("{") or ln.strip() == "}" for ln in lines
           if not ln.lstrip().startswith(("#", ";"))):
        raise ValueError("brace-structured .conf, not INI")
    parts: list[str] = []
    key_ind = None                       # indent of the key whose value may continue
    for no, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s:
            continue
        ind = _indent(raw)
        if s[0] in "#;":
            parts.append(_seg(no, 1 if ind and key_ind is not None and ind > key_ind else 0, _collapse(s)))
            continue
        if key_ind is not None and ind > key_ind:
            parts.append(_seg(no, 1, s))
            continue
        if s.startswith("["):
            key_ind = None
            parts.append(_seg(no, 0, s))
            continue
        m = _INI_KEY.match(s)
        key_ind = ind
        parts.append(_seg(no, 0, f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else s))
    return _flow(parts)


def ini_rebuild(flow: str) -> str:
    return rebuild(flow, indent=lambda lvl: "    " * lvl)


# --------------------------------------------------------------------------------------------------------- JSON

JSON_TAG_EVERY = 25          # also tag the first token of a line this many lines after the last tag


def _json_min(text: str, first_line: int = 1, every: int | None = JSON_TAG_EVERY, tag_all_top: bool = True) -> str:
    out = []
    i, n, line, depth = 0, len(text), first_line, 0
    last_tag, line_started = None, False
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            line_started = False
            i += 1
            continue
        if c in " \t\r":
            i += 1
            continue
        if not line_started:
            line_started = True
            if c not in "}],:" and (last_tag is None or (tag_all_top and depth <= 1) or
                                    (every and line - last_tag >= every)):
                out.append(("" if not out or out[-1].endswith((",", "[", "{", ":")) else " ") + f"§{line} ")
                last_tag = line
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
            continue
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        out.append(c)
        i += 1
    return "".join(out)


@register("json", exts=(".json", ".geojson", ".webmanifest"), cpt=2.13)
def json_flow(text: str) -> str:
    json.loads(text)                     # strict JSON only (tsconfig-style comments fall back to text)
    if too_numeric(text):
        return ""
    return _flow([_json_min(text)])


@register("jsonl", exts=(".jsonl", ".ndjson"), cpt=2.3)
def jsonl_flow(text: str) -> str:
    if too_numeric(text):
        return ""
    parts = []
    for no, raw in enumerate(_lines(text), 1):
        if not raw.strip():
            continue
        json.loads(raw)
        body = _json_min(raw, no, every=None, tag_all_top=False)
        parts.append(body if body.startswith("§") else f"§{no} " + body)
    return _flow(parts)


# ------------------------------------------------------------------------------------------------ XML and HTML

_VERBATIM = ("pre", "textarea", "script", "style")
_MARKUP_START = re.compile(r"<(?:[A-Za-z_:/!?])")


def _markup_end(text: str, i: int) -> int:
    """End index (exclusive) of the markup token starting at text[i] == '<'."""
    for opener, closer in (("<!--", "-->"), ("<![CDATA[", "]]>"), ("<?", "?>")):
        if text.startswith(opener, i):
            k = text.find(closer, i + len(opener))
            return len(text) if k < 0 else k + len(closer)
    j, n, q = i + 1, len(text), None
    while j < n:
        c = text[j]
        if q:
            if c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c == ">":
            return j + 1
        j += 1
    return n


def _ml_tag_text(tok: str) -> str:
    """Collapse whitespace inside a tag outside quoted values; none around '=' or before '>'."""
    if tok.startswith(("<!--", "<![CDATA[")):
        return tok if tok.startswith("<![CDATA[") else _collapse(tok)
    out, q, ws = [], None, False
    for c in tok:
        if q:
            out.append(" " if c in "\n\r\t" else c)
            if c == q:
                q = None
            continue
        if c.isspace():
            ws = True
            continue
        if ws and out and c not in "=>/" and out[-1] != "=" and out[-1] != "<":
            out.append(" ")
        elif ws and c == "/" and out and out[-1] != "<":
            out.append(" ")
        ws = False
        if c in "\"'":
            q = c
        out.append(c)
    return "".join(out)


def _markup(text: str, html: bool) -> str:
    """Markup tokens and text words; a new tagged segment starts at a line break unless text continues text."""
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    segs: list[list] = []                    # [line, text]
    st = {"ws": False, "nl": True, "last": None}

    def line_of(pos: int) -> int:
        return bisect.bisect_right(starts, pos)

    def emit(pos: int, s: str, kind: str):
        if not segs or (st["nl"] and (kind == "markup" or st["last"] != "text")):
            segs.append([line_of(pos), s])
        else:
            segs[-1][1] += (" " if st["ws"] or st["nl"] else "") + s
        st.update(ws=False, nl=False, last=kind)

    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            st["ws"] = True
            st["nl"] = st["nl"] or "\n" in text[i:j]
            i = j
            continue
        if c == "<" and _MARKUP_START.match(text, i):
            j = _markup_end(text, i)
            tok = text[i:j]
            emit(i, _ml_tag_text(tok), "markup")
            i = j
            m = re.match(r"<([A-Za-z][\w:.-]*)", tok)
            name = m.group(1).lower().split(":")[-1] if m else ""
            if name not in _VERBATIM or tok.endswith("/>"):
                continue
            close = re.compile(r"</\s*" + re.escape(m.group(1)) + r"\s*>", re.I if html else 0)
            cm = close.search(text, i)
            k = cm.start() if cm else n
            keep_lead = name in ("pre", "textarea")
            pos = i
            for bi, bl in enumerate(text[i:k].split("\n")):
                if bl.strip():
                    if bi == 0:                      # same line as the opening tag
                        segs[-1][1] += bl.rstrip() if keep_lead else bl.strip()
                    else:
                        segs.append([line_of(pos), bl.rstrip() if keep_lead else bl.strip()])
                pos += len(bl) + 1
            st.update(ws=False, nl="\n" in text[i:k] and not text[i:k].split("\n")[-1].strip(), last="markup")
            i = k
            if cm:
                emit(k, _ml_tag_text(cm.group(0)), "markup")
                i = cm.end()
            continue
        j = i + 1
        while j < n and not text[j].isspace() and not (text[j] == "<" and _MARKUP_START.match(text, j)):
            j += 1
        emit(i, text[i:j], "text")
        i = j
    return _flow(_seg(ln, 0, s) for ln, s in segs)


@register("xml", exts=(".xml", ".svg", ".xsd", ".xsl", ".xslt", ".plist", ".csproj", ".fsproj", ".vbproj",
                       ".props", ".targets", ".resx", ".xaml", ".kml", ".gpx", ".rss", ".atom", ".wsdl", ".pom"),
          cpt=2.52)
def xml_flow(text: str) -> str:
    if too_numeric(text):
        return ""
    return _markup(text, html=False)


@register("html", exts=(".html", ".htm", ".xhtml"), cpt=2.3)
def html_flow(text: str) -> str:
    return _markup(text, html=True)


# ------------------------------------------------------------------------------------- Markdown, RST, plain text

# Code fences whose language does not care about indentation lose it; every other language (and an unknown or
# missing one) keeps it as one '»' per indent step.
INDENT_FREE = {
    "js", "javascript", "jsx", "mjs", "cjs", "ts", "typescript", "tsx", "json", "jsonc", "json5", "c", "cpp", "c++",
    "cc", "h", "hpp", "cs", "csharp", "java", "kotlin", "kt", "go", "golang", "rust", "rs", "swift", "php", "ruby",
    "rb", "perl", "pl", "lua", "sql", "html", "xml", "svg", "css", "scss", "less", "bash", "sh", "shell", "zsh",
    "fish", "powershell", "ps1", "pwsh", "bat", "cmd", "batch", "dockerfile", "docker", "toml", "ini", "cfg",
    "graphql", "gql", "proto", "protobuf", "scala", "dart", "r", "julia", "jl", "matlab", "objc", "objective-c",
    "groovy", "gradle", "terraform", "hcl", "tf", "nginx", "vim", "elixir", "ex", "erlang", "clojure", "clj",
    "lisp", "scheme", "racket", "ocaml", "ml", "zig", "solidity", "sol", "vue", "svelte", "http", "latex", "tex",
    "cmake", "nix", "verilog", "vhdl", "asm", "nasm", "wasm", "d", "pascal", "delphi", "fortran", "cobol", "apex",
    "csv", "properties", "env", "dotenv", "regex", "mermaid", "plantuml", "dot", "graphviz", "jinja", "django",
    "twig", "handlebars", "hbs", "ejs", "erb", "liquid", "postcss",
}

_MD_FENCE = re.compile(r"^(`{3,}|~{3,})\s*([^\s`{]*)")
_MD_STRUCT = re.compile(
    r"^(?:#{1,6}(?:\s|$)|[-*+]\s|\d{1,9}[.)](?:\s|$)|>|\||<[A-Za-z!/?]|\[[^\]]+\]:\s|[-*_](?:\s*[-*_]){2,}\s*$"
    r"|=+\s*$|-+\s*$|\$\$|\{%|\{\{)")
_MD_LIST = re.compile(r"^([-*+]|\d{1,9}[.)])(\s+|$)")
_MD_TABLE_SEP = re.compile(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")
_RST_ADORN = re.compile(r"^([=\-`:'\"~^_*+#<>.])\1{2,}\s*$")
_RST_STRUCT = re.compile(
    r"^(?:\.\.(?:\s|$)|:[^:\s][^:]*:(?:\s|$)|[-*+•]\s|(?:\d+|#|[a-zA-Z]|[ivxlcdmIVXLCDM]+)[.)]\s"
    r"|\(\w+\)\s|--?\w|>>>|\||\+[-=+]+\+\s*$)")
_RST_CODE_DIRECTIVE = re.compile(
    r"^\.\.\s+(?:code-block|code|sourcecode|doctest|testcode|testoutput|ipython|jupyter-execute|"
    r"parsed-literal|math|productionlist|prompt)::\s*(\S*)", re.I)
_RST_SIMPLE_BORDER = re.compile(r"^=+(?:\s+=+)+\s*$")
_TXT_STRUCT = re.compile(r"^(?:#|[-*+•]\s|\(?\d{1,3}[.)]\s|\(?[a-z][.)]\s|[A-Z][A-Z0-9 ,.'&/-]{3,}$|[=\-_*~]{3,}\s*$)")


def _code_unit(rels: list[int]) -> int:
    pos = [r for r in rels if r > 0]
    return min(min(pos), 4) if pos else 4


def _code_lines(out, block: list[tuple[int, str]], base: int, sensitive: bool):
    rels = [max(0, _indent(r) - base) for _, r in block if r.strip()]
    unit = _code_unit(rels)
    for no, raw in block:
        if not raw.strip():
            continue
        body = raw.strip()
        if sensitive:
            rel = max(0, _indent(raw) - base)
            out.append(_seg(no, rel // unit, " " * (rel % unit) + body))
        else:
            out.append(_seg(no, 0, body))


def _line_structured(par: list[str]) -> bool:
    """Plain-text paragraph whose line breaks look meaningful (records, lists of short lines, tables)."""
    if len(par) < 2:
        return False
    if any("\t" in p for p in par) or any(re.search(r"\S {3,}\S", p.strip()) for p in par):
        return True
    lens = [len(p.strip()) for p in par]
    if len(par) >= 3:
        for d in ";,|":
            counts = {p.count(d) for p in par}
            if len(counts) == 1 and counts.pop() >= 2:
                return True                     # records with a fixed number of delimiters
        short = sum(1 for x in lens[:-1] if x < 0.6 * max(lens))
        return short >= max(1, (len(par) - 1) / 3)
    return max(lens) < 40


def _doc(text: str, flavor: str) -> str:
    lines = _lines(text)
    n = len(lines)
    parts: list[str] = []
    lv = _Levels()
    para_open = False
    prev_blank = True
    list_ind = None                 # md: content column of the open list (for indented code)
    literal = None                  # rst: (indent of the '::' paragraph, language) awaiting a literal block
    md_table = False
    txt_struct_lines: set[int] = set()
    if flavor == "text":
        k = 0
        while k < n:
            if not lines[k].strip():
                k += 1
                continue
            j = k
            while j < n and lines[j].strip():
                j += 1
            if _line_structured(lines[k:j]):
                txt_struct_lines.update(range(k, j))
            k = j
    i = 0
    if flavor == "markdown" and lines and lines[0].strip() == "---":      # YAML front matter
        j = 1
        while j < n and lines[j].strip() not in ("---", "..."):
            j += 1
        if j < n:
            parts.append(_seg(1, 0, "---"))
            _code_lines(parts, [(k + 1, lines[k]) for k in range(1, j)], 0, True)
            parts.append(_seg(j + 1, 0, lines[j].strip()))
            i = j + 1
    while i < n:
        raw = lines[i]
        s = raw.strip()
        no = i + 1
        if not s:
            para_open, prev_blank, md_table = False, True, False
            i += 1
            continue
        ind = _indent(raw)
        # ---- fenced code (markdown; also tolerated in rst/text)
        fm = _MD_FENCE.match(s) if flavor != "rst" else None
        if fm:
            fence, lang = fm.group(1), fm.group(2).lower()
            j = i + 1
            while j < n and not re.match(re.escape(fence[0]) + "{%d,}\\s*$" % len(fence), lines[j].strip()):
                j += 1
            parts.append(_seg(no, lv.push(ind), s))
            _code_lines(parts, [(k + 1, lines[k]) for k in range(i + 1, min(j, n))], ind,
                        lang not in INDENT_FREE)
            if j < n:
                parts.append(_seg(j + 1, lv.push(ind), lines[j].strip()))
            i = j + 1
            para_open, prev_blank = False, False
            continue
        # ---- rst literal block / code directive body, md indented code
        if prev_blank and literal is not None and ind > literal[0]:
            j, block = i, []
            while j < n and (not lines[j].strip() or _indent(lines[j]) >= ind):
                block.append((j + 1, lines[j]))
                j += 1
            _code_lines(parts, block, ind, literal[1] not in INDENT_FREE)
            literal = None
            i = j
            para_open, prev_blank = False, True
            continue
        if flavor == "markdown" and prev_blank and not para_open and ind >= (list_ind or 0) + 4:
            j, block = i, []
            while j < n and (not lines[j].strip() or _indent(lines[j]) >= ind):
                block.append((j + 1, lines[j]))
                j += 1
            while block and not block[-1][1].strip():
                block.pop()
            _code_lines(parts, block, ind, True)
            i = block[-1][0]                    # index of the line after the last code line
            para_open, prev_blank = False, False
            continue
        if literal is not None and prev_blank and ind <= literal[0]:
            literal = None
        # ---- structural line or paragraph
        if flavor == "markdown":
            if "|" in s and i + 1 < n and _MD_TABLE_SEP.match(lines[i + 1].strip()):
                md_table = True
            md_table = md_table and "|" in s
            struct = bool(_MD_STRUCT.match(s)) or md_table
            if s.startswith(">") and para_open and parts and _seg_starts(parts[-1], ">"):
                struct = False
            if _MD_LIST.match(s):
                list_ind = ind + len(_MD_LIST.match(s).group(0))
            elif ind == 0 and not para_open:
                list_ind = None
        elif flavor == "rst":
            struct = bool(_RST_ADORN.match(s) or _RST_STRUCT.match(s) or _RST_SIMPLE_BORDER.match(s))
            if i + 1 < n and _RST_ADORN.match(lines[i + 1].strip() or "x"):
                struct = True               # a section title
        else:
            struct = bool(_TXT_STRUCT.match(s)) or i in txt_struct_lines or (
                i + 1 < n and re.match(r"^[=\-~*_]{3,}\s*$", lines[i + 1].strip()))
        table_like = s.startswith("|") or (flavor == "markdown" and md_table)
        keep_inner = (flavor == "rst" and (s.startswith(("|", "+")) or _RST_SIMPLE_BORDER.match(s))) or \
            (flavor == "text" and i in txt_struct_lines) or s.startswith(">>>") or s.startswith("...")
        if flavor == "markdown" and table_like:
            body = re.sub(r"\s*\|\s*", "|", s)
        elif keep_inner:
            body = s
        else:
            body = _collapse(s)
        if struct or not para_open:
            parts.append(_seg(no, lv.push(ind), body))
        else:
            parts[-1] += " " + body
        # what may follow on the next line
        if flavor == "markdown":
            para_open = not (re.match(r"^(#{1,6}\s|\||<|\[[^\]]+\]:\s|[-*_](\s*[-*_]){2,}\s*$|=+\s*$|-+\s*$)", s)
                             or table_like)
        elif flavor == "rst":
            para_open = not (_RST_ADORN.match(s) or s.startswith(("|", "+", ">>>", "...", ".. ")) or
                             _RST_SIMPLE_BORDER.match(s) or (i + 1 < n and _RST_ADORN.match(lines[i + 1].strip()
                                                                                             or "x")))
            if s.startswith(":") and _RST_STRUCT.match(s):
                para_open = False
            cd = _RST_CODE_DIRECTIVE.match(s)
            if cd:
                literal = (ind, cd.group(1).lower() or "")
            elif s.endswith("::") and not s.startswith(".."):
                start_ind = lv.stack[-1] if lv.stack else 0
                literal = (min(ind, start_ind), "")
        else:
            para_open = i not in txt_struct_lines and not re.match(r"^[=\-~*_]{3,}\s*$", s) and not s.startswith("#")
        prev_blank = False
        i += 1
    return _flow(parts)


def _seg_starts(seg: str, prefix: str) -> bool:
    m = _SEG_RE.match(seg)
    return bool(m and m.group(3).startswith(prefix))


@register("markdown", exts=(".md", ".markdown", ".mdown", ".mkd", ".mdx"), cpt=2.67)
def markdown_flow(text: str) -> str:
    return _doc(text, "markdown")


@register("rst", exts=(".rst", ".rest"), cpt=2.58)
def rst_flow(text: str) -> str:
    return _doc(text, "rst")


@register("text", exts=(".txt", ".text"), names=("README", "LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS",
                                                  "CONTRIBUTORS", "CHANGELOG", "CHANGES", "HISTORY", "NEWS",
                                                  "TODO", "INSTALL", "THANKS", "ACKS"), cpt=1.88)
def text_flow(text: str) -> str:
    if too_numeric(text):
        return ""
    return _doc(text, "text")


# ---------------------------------------------------------------------------------------------------- CSV / TSV

@register("csv", exts=(".csv",), cpt=1.64)
def csv_flow(text: str) -> str:
    """Every physical line keeps its tag (rows stay distinguishable; a line break inside a quoted field shows as a
    tag too); nothing inside a line is touched."""
    if too_numeric(text):
        return ""
    return _flow(_seg(no, 0, raw) for no, raw in enumerate(_lines(text), 1) if raw.strip())


@register("tsv", exts=(".tsv", ".tab"), cpt=1.8)
def tsv_flow(text: str) -> str:
    """Tabs become single spaces, so only files whose fields hold no spaces and are never empty are handled."""
    if too_numeric(text):
        return ""
    parts = []
    for no, raw in enumerate(_lines(text), 1):
        if not raw.strip():
            continue
        fields = raw.split("\t")
        if any(not f or " " in f for f in fields):
            raise ValueError("TSV field with a space or empty field: tabs would be ambiguous")
        parts.append(_seg(no, 0, " ".join(fields)))
    return _flow(parts)


# ---------------------------------------------------------------------------------------------------------- SQL

def _sql_pieces(text: str):
    """Per line: (start_state, [(kind, text)]); kinds code/str/lcomment/bcomment/dollar."""
    mysql = "`" in text or re.search(r"\bENGINE\s*=", text, re.I) is not None
    if re.search(r"\bFROM\s+stdin\s*;", text, re.I):
        raise ValueError("COPY ... FROM stdin data (tab-separated) is kept as text")
    lines = _lines(text)
    state, q, dtag = "", None, None        # state: '', 'str', 'bc', 'dollar'
    res = []
    for raw in lines:
        start = state
        pieces, buf = [], ""
        i, n = 0, len(raw)

        def flush(k):
            nonlocal buf
            if buf:
                pieces.append((k, buf))
            buf = ""

        while i < n:
            c = raw[i]
            if state == "str":
                if mysql and c == "\\":
                    buf += raw[i:i + 2]
                    i += 2
                    continue
                buf += c
                i += 1
                if c == q:
                    if i < n and raw[i] == q:
                        buf += raw[i]
                        i += 1
                        continue
                    flush("str")
                    state = ""
                continue
            if state == "bc":
                k = raw.find("*/", i)
                if k < 0:
                    buf += raw[i:]
                    i = n
                else:
                    buf += raw[i:k + 2]
                    i = k + 2
                    flush("bcomment")
                    state = ""
                continue
            if state == "dollar":
                k = raw.find(dtag, i)
                if k < 0:
                    buf += raw[i:]
                    i = n
                else:
                    buf += raw[i:k + len(dtag)]
                    i = k + len(dtag)
                    flush("dollar")
                    state = ""
                continue
            if raw.startswith("--", i) or (mysql and c == "#"):
                flush("code")
                pieces.append(("lcomment", raw[i:]))
                i = n
                break
            if raw.startswith("/*", i):
                flush("code")
                state, buf = "bc", "/*"
                i += 2
                continue
            if c in "'\"`":
                flush("code")
                state, q, buf = "str", c, c
                i += 1
                continue
            m = re.match(r"\$(?:[A-Za-z_]\w*)?\$", raw[i:])
            if m:
                flush("code")
                state, dtag, buf = "dollar", m.group(0), m.group(0)
                i += len(m.group(0))
                continue
            buf += c
            i += 1
        flush({"str": "str", "bc": "bcomment", "dollar": "dollar"}.get(state, "code"))
        res.append((start, pieces))
    return lines, res


_SQL_TIGHT_AFTER = re.compile(r"([(,])\s+")
_SQL_TIGHT_BEFORE = re.compile(r"\s+([),;])")


def _sql_code(t: str) -> str:
    t = _WS.sub(" ", t)
    return _SQL_TIGHT_BEFORE.sub(r"\1", _SQL_TIGHT_AFTER.sub(r"\1", t))


@register("sql", exts=(".sql", ".psql", ".pgsql", ".ddl", ".dml"), cpt=1.96)
def sql_flow(text: str) -> str:
    if too_numeric(text):
        return ""
    lines, res = _sql_pieces(text)
    parts: list[str] = []
    stmt_start, after_lcomment = True, False
    for no, (raw, (start, pcs)) in enumerate(zip(lines, res), 1):
        if start in ("str", "dollar") and raw.strip():
            keep_lead = start == "str"
            body = ""
            for k, t in pcs:
                body += t if k in ("str", "dollar", "lcomment", "bcomment") else _sql_code(t)
            body = body.rstrip() if keep_lead else body.strip()
            parts.append(_seg(no, 0, body))
            after_lcomment = bool(pcs) and pcs[-1][0] == "lcomment"
            if any(k == "code" and ";" in t for k, t in pcs):
                stmt_start = _ends_stmt(pcs)
            continue
        body = ""
        for k, t in pcs:
            if k == "code":
                body += _sql_code(t)
            elif k == "lcomment":
                body = body.rstrip() + (" " if body.strip() else "") + _collapse(t)
            elif k == "bcomment":
                body += _collapse(t)
            else:
                body += t
        body = body.strip()
        if not body:
            continue
        first = next((k for k, t in pcs if t.strip()), "code")
        meta = raw.lstrip().startswith("\\")             # psql meta-command: one line, no ';'
        if start == "bc":
            new = not parts
        else:
            new = stmt_start or after_lcomment or meta or first in ("lcomment", "bcomment") or not parts
        if new:
            parts.append(_seg(no, 0, body))
        else:
            sep = "" if parts[-1][-1:] in "(," or body[:1] in "),;" else " "
            parts[-1] += sep + body
        after_lcomment = pcs[-1][0] == "lcomment"
        if meta:
            stmt_start = True
        elif any(k in ("code", "str", "dollar") and t.strip() for k, t in pcs):
            stmt_start = _ends_stmt(pcs)
    return _flow(parts)


def _ends_stmt(pcs) -> bool:
    for k, t in reversed(pcs):
        if k in ("lcomment", "bcomment"):
            continue
        if k == "code":
            if t.strip():
                return t.rstrip().endswith(";")
            continue
        return False
    return False
