"""Python 'reflow' minifier for rendering. Each logical line becomes "§<line> <depth markers><code>", where the
depth is the block depth from Python's tokenizer, shown as one '»' per level; physical continuation lines (inside
brackets, after a backslash, inside multi-line strings) are joined to their line with a space; blank lines dropped.
Optionally strips comments and docstrings. reconstruct() rebuilds runnable code (4 spaces per level)."""
import ast, io, re, tokenize

def _docstring_lines(src):
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            b = node.body[0]
            if isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant) and isinstance(b.value.value, str):
                out.update(range(b.lineno, b.end_lineno + 1))
    return out

def reflow(src: str, comments: bool = True, docstrings: bool = True) -> str:
    lines = src.splitlines()
    drop = set() if docstrings else _docstring_lines(src)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    depth, start_depth, cont = 0, {}, set()
    comment_at = {}                                 # line -> column where a comment starts
    logical_start = True
    for t in toks:
        if t.type == tokenize.INDENT: depth += 1; continue
        if t.type == tokenize.DEDENT: depth -= 1; continue
        if t.type == tokenize.COMMENT: comment_at[t.start[0]] = t.start[1]; continue
        if t.type in (tokenize.NL, tokenize.ENCODING): continue
        if t.type == tokenize.NEWLINE: logical_start = True; continue
        if logical_start:
            start_depth[t.start[0]] = depth; logical_start = False
        for ln in range(t.start[0] + 1, t.end[0] + 1):
            cont.add(ln)                            # lines inside a multi-line token
    parts, cur = [], None
    for no, raw in enumerate(lines, 1):
        if no in drop: continue
        s = raw
        if not comments and no in comment_at and no not in cont:
            s = s[:comment_at[no]]
        s = s.strip()
        if s.endswith("\\"): s = s[:-1].rstrip()            # line continuation (code or string)
        if not s: continue
        if no in start_depth:
            parts.append(f"§{no} {'»' * start_depth[no]}{s}")
        else:
            parts.append(s)                         # continuation: joined with a space
    return " ".join(parts)

def reconstruct(flow: str) -> str:
    out = []
    for m in re.finditer(r"§(\d+) (»*)(.*?)(?= §\d+ |$)", flow):
        out.append("    " * len(m.group(2)) + m.group(3))
    return "\n".join(out) + "\n"

def same_ast(a: str, b: str) -> bool:
    """AST equality with the whitespace inside string constants collapsed (joining lines changes it)."""
    class N(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value, str): n.value = re.sub(r"\s+", " ", n.value).strip()
            return n
    d = lambda s: ast.dump(N().visit(ast.parse(s)))
    return d(a) == d(b)

def read_format(src: str) -> str:
    return "\n".join(f"{i:>6}\t{l}" for i, l in enumerate(src.splitlines(), 1))
