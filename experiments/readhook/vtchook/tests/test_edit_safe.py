"""Edit-safe mode (the default): every source line is visible in the flow as its own tagged segment whose text equals
the line without its indentation (runs of spaces collapsed), so text copied from the image matches the file."""
import os, re, sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vtchook.minify import minify  # noqa: E402

PY = '''import os


class Box:
    """A box."""

    def area(self, w: int, h: int = 2) -> int:
        total = w * h + (w - 1)
        items = [
            1,
            2,
        ]
        return total if total > 0 else -1
'''
JAVA = '''package x;

public class A {
    private final int n = 3;

    public int f(int a, int b) {
        int c = a + b * n;
        if (c > 10 && a != b) {
            return c - 1;
        }
        return foo(a,
                   b);
    }
}
'''


def segments(flow):
    return {int(m.group(1)): m.group(2).replace("»", "").strip()
            for m in re.finditer(r"§(\d+) (.*?)(?= §\d+ |$)", flow)}


@pytest.mark.parametrize("src,path", [(PY, "m.py"), (JAVA, "A.java")])
def test_every_line_copyable(src, path, monkeypatch):
    monkeypatch.delenv("VTCHOOK_TIGHT", raising=False)
    flow, m = minify(src, path)
    segs = segments(flow)
    for no, line in enumerate(src.splitlines(), 1):
        if line.strip() and not line.strip().startswith('"""'):
            assert re.sub(r"\s+", " ", segs.get(no, "")) == re.sub(r"\s+", " ", line.strip()), (no, line)


def test_dense_mode_is_shorter(monkeypatch):
    monkeypatch.delenv("VTCHOOK_TIGHT", raising=False)
    safe = minify(JAVA, "A.java")[0]
    monkeypatch.setenv("VTCHOOK_TIGHT", "1")
    dense = minify(JAVA, "A.java")[0]
    assert len(dense) < len(safe)


def test_brace_levels_show_indentation(monkeypatch):
    monkeypatch.delenv("VTCHOOK_TIGHT", raising=False)
    flow, _ = minify(JAVA, "A.java")
    lv = {int(m.group(1)): len(m.group(2)) for m in re.finditer(r"§(\d+) (»*)", flow)}
    for no, line in enumerate(JAVA.splitlines(), 1):
        if line.strip() and not line.strip().startswith("b);"):            # continuation line: its own indent
            assert lv[no] == (len(line) - len(line.lstrip())) // 4, (no, line)
