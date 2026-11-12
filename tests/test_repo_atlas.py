"""repo_atlas.py on a tiny git repository: no network, no model."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import repo_atlas as RA  # noqa: E402

PY = "import os\n\n\ndef area(w, h):\n    if w < 0:\n        raise ValueError('negative')\n    return w * h\n" * 40
RS = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\nstruct Point { x: i32, y: i32 }\n" * 30


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "demo"
    files = {"src/geometry.py": PY, "src/lib.rs": RS, "tests/test_geometry.py": "def test_area():\n    assert True\n",
             "src/test_inline.py": "x = 1\n", "Cargo.lock": "[[package]]\nname = 'x'\n", "README.md": "# Demo\n\nText.\n",
             "node_modules/dep/index.js": "module.exports = 1;\n", "src/empty.py": "\n"}
    for p, text in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(text)
    (r / "src/blob.bin").write_bytes(bytes(range(256)) * 4)
    (r / "src/untracked.py").parent.mkdir(exist_ok=True)
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    (r / "src/untracked.py").write_text("y = 2\n")
    return r


def test_collect_keeps_source_and_skips_tests_locks_vendored_binary_untracked(repo):
    files, _ = RA.collect(repo)
    assert [f["path"] for f in files] == ["README.md", "src/geometry.py", "src/lib.rs"]
    with_tests, _ = RA.collect(repo, include_tests=True)
    assert {"tests/test_geometry.py", "src/test_inline.py"} <= {f["path"] for f in with_tests}


def test_flow_keeps_every_non_space_character_of_the_source(repo):
    files, _ = RA.collect(repo, tag_every=10)
    for f in files:
        src = re.sub(r"\s+", "", (repo / f["path"]).read_text())
        got = re.sub(r"\s+", "", re.sub(r"§\d* ?", "", f["flow"]).replace("»", ""))
        assert got == src, f["path"]


def test_thin_tags_keeps_one_line_number_in_ten_and_every_separator():
    flow = " ".join(f"§{n} x{n} = {n}" for n in range(1, 31))
    thin = RA.thin_tags(flow, 10)
    assert re.findall(r"§(\d+) ", thin) == ["1", "11", "21"] and thin.count("§") == 30
    assert RA.thin_tags(flow, 1) == flow


def test_stats_and_build_agree_and_the_index_covers_every_file(repo, tmp_path):
    s = RA.stats(repo, 88.2, False, 10)
    assert s["repo"] == "demo" and s["files"] == 3 and s["fits"] and s["fewer_tokens"] > 1.5
    meta = RA.build(repo, tmp_path / "atlas", 88.2, False, 10)
    assert abs(meta["image_tokens"] - s["image_tokens"]) <= 0.1 * s["image_tokens"] + 200
    assert [e["path"] for e in meta["index"]] == ["README.md", "src/geometry.py", "src/lib.rs"]
    n = len(meta["pages"])
    assert all(1 <= e["pages"][0] <= e["pages"][1] <= n for e in meta["index"]) and meta["index"][-1]["pages"][1] == n
    assert sorted(p.name for p in (tmp_path / "atlas").glob("page_*.png")) == [f"page_{i:03d}.png" for i in range(1, n + 1)]
    assert json.loads((tmp_path / "atlas/index.json").read_text())["files"] == 3
    assert (tmp_path / "atlas/index.txt").read_text().splitlines()[0] == "README.md: page 1"
    assert all(p["size"][0] == 1932 and p["size"][1] % 28 == 0 for p in meta["pages"])
