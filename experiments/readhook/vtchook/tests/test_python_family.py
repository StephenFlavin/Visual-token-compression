"""python_family minifiers: Python, shell, Makefile, Dockerfile (no network, no Claude calls)."""
import glob
import os
import re
import shutil
import subprocess
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(HERE))
from vtchook.minify import REGISTRY, detect, minify, preserves, strip_marks  # noqa: E402
from vtchook.minify import python_family as pf  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = REGISTRY["python"]

EDGE = r'''#!/usr/bin/env python3
"""Module doc.

>>> f(1)
2
"""
import os, sys
x = 1 .real + 0x1F - -3 + (+4) ** -2
y = a if b else -1
z = [i for i in range(3) if not -i]
w = lambda *a, **k: ...
if (n := len(sys.argv)) > 1 and x < -1: pass
s = f"{x!r:>10} {y + 1:{w}} {'a' 'b'}" f'{z}'
t = rf"""multi
   {x}
line"""
u = ('a'
     'b')  # trailing
def g(a, /, b=-1, *, c: int = 2) -> "int":
    # leading comment at depth 1
    total = a + \
        b
    s2 = "abc\
   def"
    return -total if b else (yield)
class C(object):
    QQQDoc

    Args:
        a: first
          continued
    QQQ
    sql = """
        SELECT *
          FROM t
    """
    async def h(self):
        async with x as y: await y
        match self:
            case [1, *rest]: return rest
            case _: return None
call(1,  # first
     2,
     # own line
     3)
del x; x = {**a, 'b': 2}; x[1:-1]; x[::2]
'''.replace("QQQ", "'" * 3)


def repo_files(pattern):
    try:
        out = subprocess.run(["git", "-C", REPO, "ls-files", pattern], capture_output=True, text=True,
                             timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        out = []
    return [os.path.join(REPO, f) for f in out]


# ----------------------------------------------------------------------------------------------------- Python

def py_ok(src, **kw):
    flow = pf.python_flow(src, **kw)
    assert preserves(PY, src, flow), flow
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert pf.same_ast(src, pf.reconstruct(flow)), pf.reconstruct(flow)
    return flow


@pytest.mark.parametrize("kw", [{}, {"cont_tags": True}, {"doc_mode": "lines"}, {"doc_mode": "collapse"}])
def test_python_edge_roundtrip(kw):
    py_ok(EDGE, **kw)


def test_python_spacing():
    f = py_ok(EDGE)
    for frag in ("§7 import os,sys", "x=1 .real+0x1F- -3+(+4)**-2", "§9 y=a if b else -1",
                 "if not -i]", "w=lambda *a,**k:...", "if (n:=len(sys.argv))>1 and x< -1:pass",
                 "def g(a,/,b=-1,*,c:int=2)->\"int\":", "»return -total if b else (yield)",
                 "»»»case [1,*rest]:return rest", "del x;x={**a,'b':2};x[1:-1];x[::2]"):
        assert frag in f, frag
    # f-strings verbatim, including spaces inside replacement fields
    assert "s=f\"{x!r:>10} {y + 1:{w}} {'a' 'b'}\" f'{z}'" in f


def test_python_tags_and_levels():
    f = py_ok(EDGE)
    assert f.startswith("§1 #!/usr/bin/env python3 §2 \"\"\"Module doc. §4 >>> f(1) §5 2 \"\"\"")
    assert "§20 »# leading comment at depth 1" in f          # comment-only line: level of the next statement
    assert "§21 »total=a+ \\ §22 b" in f                      # backslash kept, next line tagged
    assert "§23 »s2=\"abc\\ §24  def\"" in f                  # leading blanks after `\`+newline in a string kept
    assert "u=('a' 'b') # trailing" in f                      # bracket continuation joined untagged
    assert "call(1, # first §43 2, §44 # own line §45 3)" in f   # tag after a comment inside brackets
    assert "»sql=\"\"\" §34 »SELECT * §35 »»FROM t §36 \"\"\"" in f  # other strings: a tag per line
    assert "»'''Doc §29 Args: §30 »a: first §31 »»continued '''" in f  # docstring paragraphs


def test_python_docstring_paragraphs():
    src = '''def f(a):
    """Summary line
    continues here.

    Parameters
    ----------
    a : int
        The thing,
        over two lines.

    Examples
    --------
    >>> f(1)
    2
    - item one
    - item two
    """
'''
    f = py_ok(src)
    assert "»\"\"\"Summary line continues here. §5 Parameters ---------- §7 a : int §8 »The thing, over two lines." in f
    assert "§13 >>> f(1) §14 2 §15 - item one §16 - item two \"\"\"" in f
    collapsed = py_ok(src, doc_mode="collapse")
    assert collapsed.count("§") == 2                         # `def` and the docstring's first line


def test_python_no_glued_tokens():
    """Removing a space never merges two tokens: the token strings of flow and source are identical."""
    import io
    import tokenize
    src = "a = b - -c\nd = e < -f\ng = 1 .real\nh = x if y else z\ni = not -j\nk = 'a' 'b'\nl = 2 if 1 else 3\n"
    f = py_ok(src)

    def toks(s):
        return [t.string for t in tokenize.generate_tokens(io.StringIO(s).readline)
                if t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
                                  tokenize.ENDMARKER)]
    assert toks(src) == toks(pf.reconstruct(f))


def test_python_misc_edges():
    py_ok("")
    py_ok("x = 1")                                             # no trailing newline
    py_ok("if x:\n\ty = 1\n\tif y:\n\t\tz = 2\n")             # tab indentation
    py_ok("x = 1\r\ny = [1,\r\n  2]\r\n")                      # CRLF
    py_ok("@dec(1,\n     2)\nclass A:\n    pass\n    # trailing comment\n# eof comment\n")
    py_ok("s = b'''a\n   b'''\nprint(f'''{s!r}\n  {s}''')\n")
    py_ok("x = [\n    1,\n    2,\n]\ny = {\n    'a': 1,  # c\n}\n")
    py_ok("def f():\n    return f'{x:{\"a\" if y else \"b\"}}'\n")


def test_python_registry_and_detect():
    assert detect("pkg/mod.py").kind == "python" and detect("stubs/a.pyi").kind == "python"
    assert detect(None, "#!/usr/bin/env python3\nprint(1)\n").kind == "python"
    flow, m = minify(EDGE, path="x.py")
    assert m.kind == "python" and flow == pf.python_flow(EDGE)
    assert minify("x = '§1'\n", path="x.py") is None          # marker characters -> caller falls back
    res = minify("def f(:\n  x = (\n", path="x.py")          # broken source: a flow or None, never an exception
    assert res is None or res[1].kind == "python"
    assert 2.0 < PY.cpt < 4.0


def test_python_repo_files():
    files = repo_files("*.py")
    if not files:
        pytest.skip("not a git checkout")
    n = 0
    for path in files:
        src = open(path, encoding="utf-8", errors="replace").read()
        if "§" in src or "»" in src:
            continue
        py_ok(src)
        n += 1
    assert n > 50


# ------------------------------------------------------------------------------------------------------ shell

SH = r'''#!/bin/bash
# setup
set -eu
if [ -n "$X" ]; then
    echo   "a   b"    'c  d'   # spaces kept in quotes
	for f in *.txt; do echo "$f"; done
fi
cat <<EOF > out.txt
  indented line
EOF too long

EOF
cat <<-'END'
	tab stripped $NOT_EXPANDED
	END
x=$((1 << 2)); y=$(cat <<< "here")
cmd --one \
    --two \
    --three
msg="first
  second line"
echo "${x//  /_}" # comment ends with backslash \
echo next
'''


def sh_flow(src):
    flow = REGISTRY["shell"].fn(src)
    assert preserves(REGISTRY["shell"], src, flow), flow
    return flow


def test_shell_flow():
    f = sh_flow(SH)
    assert f.startswith("§1 #!/bin/bash §2 # setup §3 set -eu §4 if [ -n \"$X\" ]; then")
    assert "§5 echo \"a   b\" 'c  d' # spaces kept in quotes §6 for f in *.txt; do echo \"$f\"; done §7 fi" in f
    # heredoc body verbatim, one tag per line, blank lines kept, terminator recognised
    assert "§8 cat <<EOF > out.txt §9   indented line §10 EOF too long §11  §12 EOF §13 cat <<-'END'" in f
    assert "§14 \ttab stripped $NOT_EXPANDED §15 \tEND §16 x=$((1 << 2)); y=$(cat <<< \"here\")" in f
    assert "§17 cmd --one \\ --two \\ --three §20 msg=\"first §21   second line\"" in f   # continuation joined
    assert "§22 echo \"${x//  /_}\" # comment ends with backslash \\ §23 echo next" in f


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_shell_reconstruction_parses(tmp_path):
    files = [p for p in repo_files("*.sh")]
    texts = [SH] + [open(p, encoding="utf-8", errors="replace").read() for p in files]
    for i, src in enumerate(texts):
        a, b = tmp_path / f"a{i}.sh", tmp_path / f"b{i}.sh"
        a.write_text(src)
        b.write_text(pf.reconstruct_lines(sh_flow(src)))
        if subprocess.run(["bash", "-n", str(a)], capture_output=True).returncode == 0:
            r = subprocess.run(["bash", "-n", str(b)], capture_output=True, text=True)
            assert r.returncode == 0, (i, r.stderr)


def test_shell_detect():
    assert detect("x/install.sh").kind == "shell" and detect("a.zsh").kind == "shell"
    assert detect(None, "#!/bin/sh\necho hi\n").kind == "shell"
    assert detect(None, "#!/usr/bin/env bash\necho hi\n").kind == "shell"
    assert detect(None, "#!/bin/zsh\necho hi\n").kind == "shell"


# --------------------------------------------------------------------------------------------------- Makefile

MK = (".PHONY: all clean\n"
      "CFLAGS   :=  -O2 \\\n"
      "            -Wall   # warnings\n"
      "\n"
      "ifeq ($(OS),Linux)\n"
      "  LIBS += -lm\n"
      "endif\n"
      "all: main.o util.o\n"
      "\t$(CC) $(CFLAGS)   -o $@ $^\n"
      "\tif [ -f x ]; then \\\n"
      "\t  echo \"a  b\"; \\\n"
      "\tfi\n"
      "# comment \\\n"
      "  continued comment\n"
      "define HELP\n"
      "Usage:   make\n"
      "endef\n"
      "clean: ; rm -f *.o\n")


def test_makefile_flow():
    m = REGISTRY["makefile"]
    f = m.fn(MK)
    assert preserves(m, MK, f), f
    assert f.startswith("§1 .PHONY: all clean §2 CFLAGS := -O2 \\ -Wall # warnings §5 ifeq ($(OS),Linux) §6 LIBS")
    assert "§8 all: main.o util.o §9 »$(CC) $(CFLAGS) -o $@ $^ §10 »if [ -f x ]; then \\ echo \"a  b\"; \\ fi" in f
    assert "§13 # comment \\ continued comment §15 define HELP §16 Usage: make §17 endef §18 clean: ; rm" in f
    # every recipe line (TAB-led, not a continuation) carries exactly one `»`
    rec = pf.reconstruct_lines(f, recipe_tab=True).splitlines()
    assert [ln.split()[0] for ln in rec if ln.startswith("\t")] == ["$(CC)", "if"]


def test_makefile_detect():
    for name in ("Makefile", "makefile", "GNUmakefile", "rules.mk"):
        assert detect(f"src/{name}").kind == "makefile"


# ------------------------------------------------------------------------------------------------- Dockerfile

DF = ("# syntax=docker/dockerfile:1\n"
      "FROM debian:bookworm AS build\n"
      "ENV A=1 \\\n"
      "    B=2\n"
      "RUN set -eux; \\\n"
      "    apt-get update; \\\n"
      "# a comment inside the instruction\n"
      "    apt-get install -y   curl; \\\n"
      "\n"
      "    rm -rf /var/lib/apt/lists/*\n"
      "RUN <<EOF\n"
      "  echo   heredoc\n"
      "EOF\n"
      "COPY --from=build /a /b\n"
      "CMD [\"run\",   \"--x\"]\n")


def test_dockerfile_flow():
    m = REGISTRY["dockerfile"]
    f = m.fn(DF)
    assert preserves(m, DF, f), f
    assert ("§1 # syntax=docker/dockerfile:1 §2 FROM debian:bookworm AS build §3 ENV A=1 \\ B=2 "
            "§5 RUN set -eux; \\ apt-get update; \\ §7 # a comment inside the instruction "
            "§8 apt-get install -y curl; \\ rm -rf /var/lib/apt/lists/* "
            "§11 RUN <<EOF §12   echo   heredoc §13 EOF §14 COPY --from=build /a /b §15 CMD [\"run\", \"--x\"]") in f


def test_dockerfile_escape_directive():
    src = "# escape=`\nFROM mcr.microsoft.com/windows\nRUN dir c:\\ `\n    && echo done\nCMD x\n"
    m = REGISTRY["dockerfile"]
    f = m.fn(src)
    assert preserves(m, src, f)
    assert "§3 RUN dir c:\\ ` && echo done §5 CMD x" in f


def test_dockerfile_detect():
    assert detect("ctx/Dockerfile").kind == "dockerfile" and detect("a/web.Dockerfile").kind == "dockerfile"
    assert detect(None, "# comment\nARG V=1\nFROM python:$V\nRUN true\n").kind == "dockerfile"


def test_registered_cpts():
    for k in ("python", "shell", "makefile", "dockerfile"):
        assert 1.3 < REGISTRY[k].cpt < 4.0


def test_invariant_rejects_changes():
    """Sanity check of the checker itself on this family's flows."""
    f = REGISTRY["shell"].fn(SH)
    assert not preserves(REGISTRY["shell"], SH, f.replace("set -eu", "set -e"))
    assert re.sub(r"\s+", "", strip_marks(f)) == re.sub(r"\s+", "", SH)


def test_dockerfile_quote_spans_continuation():
    src = "FROM x\nRUN sh -c ' \\\n\t\techo   a; \\\n\t' -- y \\\n  && z\nCMD q\n"
    m = REGISTRY["dockerfile"]
    f = m.fn(src)
    assert preserves(m, src, f)
    assert f == "§1 FROM x §2 RUN sh -c ' \\ echo   a; \\ ' -- y \\ && z §6 CMD q"
