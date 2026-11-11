"""Java 'reflow' minifier for rendering: drop indentation and line breaks (lines joined, each tagged with its line
number), optionally strip comments. String/char literals and text blocks are left intact."""
import re

def strip_comments(src: str) -> str:
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith('"""', i):                      # text block
            j = src.find('"""', i + 3); j = n if j < 0 else j + 3
            out.append(src[i:j]); i = j
        elif c in "\"'":                                   # string / char literal
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1]); i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i); j = n if j < 0 else j
            i = j                                          # keep the newline (line numbers stay right)
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2); j = n if j < 0 else j + 2
            out.append("\n" * src.count("\n", i, j)); i = j
        else:
            out.append(c); i += 1
    return "".join(out)

def reflow(src: str, comments: bool = True, tags: bool = True) -> str:
    if not comments:
        src = strip_comments(src)
    parts = []
    for no, line in enumerate(src.splitlines(), 1):
        s = re.sub(r"\s+", " ", line.strip()) if '"' not in line else line.strip()
        if s:
            parts.append(f"§{no} {s}" if tags else s)
    return " ".join(parts)

def read_format(src: str) -> str:
    """What Claude Code's Read returns as text: right-aligned line number, tab, line."""
    return "\n".join(f"{i:>6}\t{l}" for i, l in enumerate(src.splitlines(), 1))
