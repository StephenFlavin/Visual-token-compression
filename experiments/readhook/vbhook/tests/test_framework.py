"""Hook framework: Read/Edit/Write/Bash hooks end to end as subprocesses (no Claude calls)."""
import base64, io, json, os, subprocess, sys, textwrap
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(HERE))
from vbhook import policy, render  # noqa: E402
from vbhook.minify import REGISTRY  # noqa: E402

needs_python = pytest.mark.skipif("python" not in REGISTRY, reason="python minifier not registered")


def run(script, event, tmp_path):
    env = dict(os.environ, VBHOOK_CACHE=str(tmp_path / "cache"), VBHOOK_LOG=str(tmp_path / "log.jsonl"),
               VBHOOK_BUDGET="88.2")
    p = subprocess.run([sys.executable, os.path.join(HERE, script)], input=json.dumps(event), capture_output=True,
                       text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)["hookSpecificOutput"] if p.stdout.strip() else None


BIG = "\n".join([textwrap.dedent(f"""
    def func_{i}(xs, k={i}):
        \"\"\"Docstring number {i} explains what this does.\"\"\"
        total = 0
        for x in xs:
            if x % (k + 1) == 0:
                total += x * {i}
        return total
    """) for i in range(400)])


def test_policy():
    assert policy.use_image(100, 200)
    assert not policy.use_image(190, 200)


@needs_python
def test_read_big_file_pages(tmp_path):
    f = tmp_path / "big.py"
    f.write_text(BIG)
    out = run("read_hook.py", {"tool_name": "Read", "tool_input": {"file_path": str(f)}}, tmp_path)
    assert out and out["updatedInput"]["file_path"].endswith(".png")
    assert "offset=" in out["additionalContext"]
    nxt = int(out["additionalContext"].split("offset=")[1].split(" ")[0])
    out2 = run("read_hook.py", {"tool_name": "Read", "tool_input": {"file_path": str(f), "offset": nxt}}, tmp_path)
    assert out2["updatedInput"]["file_path"] != out["updatedInput"]["file_path"]
    assert f"lines {nxt}-" in out2["additionalContext"]


@needs_python
def test_read_small_file_stays_text(tmp_path):
    f = tmp_path / "tiny.py"
    f.write_text("x = 1\n")
    assert run("read_hook.py", {"tool_name": "Read", "tool_input": {"file_path": str(f)}}, tmp_path) is None


def test_read_unknown_type_stays_text(tmp_path):
    f = tmp_path / "data.unknownext"
    f.write_text("a b c\n" * 2000)
    assert run("read_hook.py", {"tool_name": "Read", "tool_input": {"file_path": str(f)}}, tmp_path) is None


def test_edit_fuzzy_reindents(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n")
    ev = {"tool_name": "Edit", "tool_input": {"file_path": str(f), "old_string": "for x in xs: total+=x",
                                               "new_string": "total += sum(xs)"}}
    out = run("edit_hook.py", ev, tmp_path)
    assert out["updatedInput"]["old_string"] == "    for x in xs:\n        total += x"
    assert out["updatedInput"]["new_string"] == "    total += sum(xs)"


def test_edit_yaml_unit_and_markers(tmp_path):
    f = tmp_path / "c.yml"
    f.write_text("a:\n  b:\n    c: 1\n")
    ev = {"tool_name": "Edit", "tool_input": {"file_path": str(f), "old_string": "§3 »»c: 1", "new_string": "»»c: 2"}}
    out = run("edit_hook.py", ev, tmp_path)
    assert out["updatedInput"]["old_string"] == "    c: 1" or out["updatedInput"]["old_string"] == "c: 1"
    assert out["updatedInput"]["new_string"].strip() == "c: 2"


def test_edit_exact_untouched(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1\ny = 2\n")
    ev = {"tool_name": "Edit", "tool_input": {"file_path": str(f), "old_string": "y = 2", "new_string": "y = 3"}}
    assert run("edit_hook.py", ev, tmp_path) is None


def test_write_unmarks(tmp_path):
    ev = {"tool_name": "Write", "tool_input": {"file_path": str(tmp_path / "n.py"), "content": "§1 def f():\n»return 1\n"}}
    out = run("edit_hook.py", ev, tmp_path)
    assert out["updatedInput"]["content"] == "def f():\n    return 1\n"


def test_bash_short_output_stays_text(tmp_path):
    ev = {"tool_name": "Bash", "tool_input": {"command": "echo hi"}, "tool_response": {"stdout": "hi", "stderr": ""}}
    assert run("bash_hook.py", ev, tmp_path) is None


@needs_python
def test_bash_cat_of_big_file_becomes_image(tmp_path):
    f = tmp_path / "big.py"
    f.write_text(BIG)
    ev = {"tool_name": "Bash", "tool_input": {"command": f"cat {f}"},
          "tool_response": {"stdout": BIG, "stderr": "", "interrupted": False, "isImage": False}}
    out = run("bash_hook.py", ev, tmp_path)
    o = out["updatedToolOutput"]
    assert o["isImage"] and o["stdout"].startswith("data:image/png;base64,")
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(o["stdout"].split(",", 1)[1])))
    assert im.size[0] == render.W
    assert "saved at" in out["additionalContext"]          # 400 functions do not fit on one page
