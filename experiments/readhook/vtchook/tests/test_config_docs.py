"""Tests for the config / data / docs minifiers (vtchook/minify/config_docs.py). No network, no Claude calls.

Set VTCHOOK_CORPUS=<dir> to also run the invariant and the parser round-trips over every file below that directory
(sub-directories named after the kind: yaml, toml, ini, json, jsonl, xml, svg, html, md, rst, txt, csv, tsv, sql).
"""
import configparser
import csv
import io
import json
import os
import re
import tomllib
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import pytest

from vtchook.minify import REGISTRY, detect, minify, preserves, strip_marks
from vtchook.minify import config_docs as cd

try:
    import yaml
except ModuleNotFoundError:  # PyYAML missing: the YAML round-trips are skipped
    yaml = None


def run(kind, text, path=None):
    """minify() through the public entry point, plus the invariant and the flow alphabet."""
    res = minify(text, path=path, hint=None if path else kind)
    assert res is not None, f"{kind}: minify declined"
    flow, m = res
    assert m.kind == kind
    assert preserves(m, text, flow)
    assert not re.search(r"[\t\n\r]", flow)
    return flow


def tags(flow):
    return [int(x) for x in re.findall(r"§(\d+) ", flow)]


# ------------------------------------------------------------------------------------------------ registry

def test_registry_and_detect():
    for k in ("lockfile", "yaml", "toml", "ini", "json", "jsonl", "xml", "html", "markdown", "rst", "text", "csv",
              "tsv", "sql"):
        assert k in REGISTRY and 1.0 < REGISTRY[k].cpt < 4.0
    assert detect(".github/workflows/ci.yml").kind == "yaml"
    assert detect("pyproject.toml").kind == "toml"
    assert detect("setup.cfg").kind == "ini"
    assert detect("a/.editorconfig").kind == "ini"
    assert detect("package.json").kind == "json"
    assert detect("data.jsonl").kind == "jsonl"
    assert detect("icon.svg").kind == "xml"
    assert detect("index.html").kind == "html"
    assert detect("README.md").kind == "markdown"
    assert detect("docs/index.rst").kind == "rst"
    assert detect("LICENSE").kind == "text"
    assert detect("x.csv").kind == "csv"
    assert detect("schema.sql").kind == "sql"


def test_lockfiles_stay_text():
    for name in ("package-lock.json", "Cargo.lock", "poetry.lock", "pnpm-lock.yaml", "uv.lock"):
        assert detect(name).kind == "lockfile"
        assert minify('{"a": 1}\n', path=name) is None


def test_numeric_guard():
    rows = "\n".join(f"{i},{i * 3.14159:.5f},{i * 2718}" for i in range(200))
    assert cd.too_numeric(rows)
    assert minify("a,b,c\n" + rows + "\n", path="x.csv") is None
    sha = "\n".join(f'pkg{i} = "sha256:{"%064x" % (i * 7919 + 12345678901234567)}"' for i in range(40))
    assert cd.hash_share(sha) > 0.5 and cd.too_numeric(sha)
    prose = "The quick brown fox jumps over the lazy dog. " * 20
    assert not cd.too_numeric(prose)
    assert cd.hash_share("crates/ruff_linter/resources/test/fixtures/ " * 20) == 0


def test_marker_characters_fall_back():
    assert minify("a: §1\n", path="x.yaml") is None


# ---------------------------------------------------------------------------------------------------- YAML

YAML_CASES = {
    "workflow": """name: Tests

on:
  push:
    branches:
    - 'main'
permissions:
  contents: read     # aligned comment
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 1
      - name: Run
        run: |
          python -m pip install -U pip
          if [ -f x ]; then
            echo "has # hash"
          fi

          pytest -q
      - name: Folded
        run: >-
          one
          two
""",
    "unit4": """jobs:
    build:
        steps:
            -   name: x
                run: |
                    for f in *; do
                        echo $f
                    done
            -   uses: a/b@v1
                with:
                    k:     aligned
""",
    "compact": """a:
- b: 1
  c:
  - 2
  - |
    lit
      more
- - x
  - y
e: "quoted # not comment"   # real
f: 'it''s'
g: [1,
    2, 3]
h: this is a
  multi line plain
""",
    "multidoc": """---
apiVersion: v1
kind: Service
metadata:
  name: redis
  labels: &l
    app: redis
  selector: *l
---
apiVersion: apps/v1
kind: Deployment
spec:
  selector:
    matchLabels: {app: redis}
  template:
    spec:
      containers:
      - name: master
        image: redis
        args: ["--port", "6379"]
...
""",
    "comment_after_block": """a: |
    body
  # comment between
b: 1
c:
  - run: |
      x
    # c2
  - |
     direct
     two
""",
}


@pytest.mark.parametrize("name", sorted(YAML_CASES))
def test_yaml_roundtrip(name):
    text = YAML_CASES[name]
    flow = run("yaml", text, path="ci.yml")
    assert tags(flow) == [i for i, ln in enumerate(text.split("\n"), 1) if ln.strip()]
    if yaml is None:
        pytest.skip("PyYAML not installed")
    rebuilt = cd.yaml_rebuild(flow, cd.yaml_unit(text))
    assert list(yaml.safe_load_all(rebuilt)) == list(yaml.safe_load_all(text))


def test_yaml_levels_and_compaction():
    flow = run("yaml", YAML_CASES["workflow"], path="ci.yml")
    assert "§3 on: §4 »push: §5 »»branches: §6 »»- 'main'" in flow
    assert "§8 »contents: read # aligned comment" in flow
    # block scalar body: one level below the key, relative indentation kept as levels
    assert "§18 »»»»»python -m pip install -U pip" in flow
    assert '§20 »»»»»»echo "has # hash"' in flow
    flow4 = run("yaml", YAML_CASES["unit4"], path="x.yaml")
    assert cd.yaml_unit(YAML_CASES["unit4"]) == 4
    assert "§4 »»»- name: x" in flow4 and "§7 »»»»»»echo $f" in flow4 and "k: aligned" in flow4


def test_yaml_declines_tabs_and_explicit_indent():
    assert minify("a:\n\tb: 1\n", path="x.yaml") is None
    assert minify("a: |2\n   x\n", path="x.yaml") is None


# ---------------------------------------------------------------------------------------------------- TOML

TOML = '''# top comment
[project]
name = "demo"
title = "TOML  Example"   # spaced
dob = 1979-05-27 07:32:00-08:00
dependencies = [
    "requests>=2",
    "rich",   # pretty output
    "click",
]
data = [ ["gamma", "delta"], [1, 2] ]
desc = """
Roses are red
   Violets are blue  \\
   continued"""
lit = \'\'\'
C:\\path\\  keep
\'\'\'
inline = { x = 1, y = "a b" }

[[tool.fruit]]
name = "apple"
[[tool.fruit]]
"quoted key" = 'x # not comment'
'''


def test_toml_roundtrip():
    flow = run("toml", TOML, path="pyproject.toml")
    assert tomllib.loads(cd.toml_rebuild(flow)) == tomllib.loads(TOML)
    assert '§6 dependencies=["requests>=2","rich", # pretty output §9 "click",]' in flow
    assert '§4 title="TOML  Example" # spaced' in flow          # strings untouched
    assert "§14    Violets are blue  \\" in flow                 # multi-line string lines verbatim, tagged
    assert "§24 \"quoted key\"='x # not comment'" in flow


# ----------------------------------------------------------------------------------------------------- INI

INI = """[tox]
envlist =
    py310
    py311
; a comment
[testenv]
deps = pytest
    coverage
commands =
    pytest {posargs}
key: value with   spaces

[flake8]
max-line-length = 120
"""


def _ini(text):
    cp = configparser.ConfigParser(interpolation=None, strict=False, allow_no_value=True)
    cp.read_string(text)
    return {s: dict(cp[s]) for s in cp.sections()}


def test_ini_roundtrip():
    flow = run("ini", INI, path="tox.ini")
    assert _ini(cd.ini_rebuild(flow)) == _ini(INI)
    assert flow.startswith("§1 [tox] §2 envlist= §3 »py310 §4 »py311 §5 ; a comment")
    assert "§11 key:value with   spaces" in flow


def test_ini_gitconfig_keys_are_not_continuations():
    text = '[core]\n\tbare = false\n\tfilemode = true\n[remote "origin"]\n\turl = https://x/y.git\n'
    flow = run("ini", text, path=".gitconfig")
    assert "»" not in flow and tags(flow) == [1, 2, 3, 4, 5]


def test_brace_conf_is_not_ini():
    assert minify("server {\n    listen 80;\n}\n", path="nginx.conf") is None


# ---------------------------------------------------------------------------------------------------- JSON

PKG = """{
  "name": "demo",
  "version": "1.0.0",
  "scripts": {
    "test": "node --test 'a b'",
    "build": "tsc"
  },
  "keywords": [
    "x",
    "y"
  ],
  "nested": {"deep": [1, 2, {"k": "v  with  spaces"}]}
}
"""


def test_json_strip_and_tags():
    flow = run("json", PKG, path="package.json")
    assert json.loads(strip_marks(flow)) == json.loads(PKG)
    assert tags(flow) == [1, 2, 3, 4, 8, 12]                 # first line and top-level members only
    assert '"test":"node --test \'a b\'"' in flow and '"v  with  spaces"' in flow


def test_json_periodic_tags_in_long_members():
    text = json.dumps({"rows": [{"i": i} for i in range(80)]}, indent=2)
    flow = run("json", text, path="big.json")
    t = tags(flow)
    assert t[:2] == [1, 2] and len(t) > 3 and all(b - a >= cd.JSON_TAG_EVERY for a, b in zip(t[1:], t[2:]))
    assert json.loads(strip_marks(flow)) == json.loads(text)


def test_json_with_comments_falls_back():
    assert minify('{\n  // comment\n  "a": 1\n}\n', path="tsconfig.json") is None


def test_jsonl_one_tag_per_record():
    text = '{"a": 1, "b": "x  y"}\n\n{"a": 2}\n'
    flow = run("jsonl", text, path="d.jsonl")
    assert flow == '§1 {"a":1,"b":"x  y"} §3 {"a":2}'


# --------------------------------------------------------------------------------------------- XML / HTML

POM = """<?xml version="1.0" encoding="UTF-8"?>
<!-- Licensed to the
     Apache Software Foundation -->
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <modelVersion>4.0.0</modelVersion>
  <dependencies>
    <dependency>
      <groupId>org.junit</groupId>
      <artifactId  >junit</artifactId>
    </dependency>
  </dependencies>
  <description>Maven is a software build
    management tool.</description>
</project>
"""


def _etree(text):
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip()  # noqa: E731
    return [(e.tag, e.attrib, norm(e.text), norm(e.tail)) for e in ET.fromstring(text.encode()).iter()]


def test_xml_tree_equal():
    flow = run("xml", POM, path="pom.xml")
    assert _etree(strip_marks(flow)) == _etree(POM)
    assert "§4 <project xmlns=" in flow and "§10 <artifactId>junit</artifactId>" in flow
    assert "§13 <description>Maven is a software build management tool.</description>" in flow


HTML = """<!DOCTYPE html>
<html>
<head>
  <style>
    body { margin: 0; }
    pre  { color: red; }
  </style>
  <script>
    // a comment: line structure matters
    var x = 1;
        if (x) { go(); }
  </script>
</head>
<body>
  <p class="a   b">Some text
     wrapped here.</p>
  <pre>
  def f():
      return 1
</pre>
  <textarea>  keep  </textarea>
</body>
</html>
"""


class _Events(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ev = []

    def handle_starttag(self, tag, attrs):
        self.ev.append(("s", tag, [(k, re.sub(r"\s+", " ", v or "")) for k, v in attrs]))

    def handle_endtag(self, tag):
        self.ev.append(("e", tag))

    def handle_data(self, data):
        d = re.sub(r"\s+", " ", data).strip()
        if d:
            self.ev.append(("d", d))


def _html_events(text):
    p = _Events()
    p.feed(text)
    return p.ev


def test_html_events_and_verbatim_regions():
    flow = run("html", HTML, path="index.html")
    assert _html_events(strip_marks(flow)) == _html_events(HTML)
    rebuilt = cd.rebuild(flow, indent=lambda lvl: "").split("\n")
    orig = HTML.split("\n")
    for no in (18, 19):                                 # <pre> lines come back exactly
        assert rebuilt[no - 1] == orig[no - 1]
    assert "§9 // a comment: line structure matters §10 var x = 1; §11 if (x) { go(); }" in flow
    assert "§21 <textarea>  keep</textarea>" in flow
    assert "§15 <p class=\"a   b\">Some text wrapped here.</p>" in flow


def test_svg():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"\n     viewBox="0 0 24 24">\n'
           '  <path d="M3 12h4l3-9\n           l4 18 3-9h4"/>\n</svg>\n')
    flow = run("xml", svg, path="icon.svg")
    assert _etree(strip_marks(flow)) == _etree(svg)
    assert tags(flow) == [1, 3, 5]


# ---------------------------------------------------------------------------------------- Markdown / RST / text

MD = """# Title

Some paragraph text that
wraps over   three
lines.

- item one
  continues here
  - nested item
- item two

| a    | b  |
|------|----|
| 1    | 2  |
x | y

```python
def f():
    if x:
        return 1
```

```js
function f() {
    return 1;
}
```

    indented code
        more

> quote line one
> quote line two

Setext title
============
"""


def test_markdown_structure():
    flow = run("markdown", MD, path="README.md")
    assert "§1 # Title §3 Some paragraph text that wraps over three lines." in flow
    assert "§7 - item one continues here §9 »- nested item §10 - item two" in flow
    assert "§12 |a|b| §13 |------|----| §14 |1|2| §15 x|y" in flow
    assert "§17 ```python §18 def f(): §19 »if x: §20 »»return 1 §21 ```" in flow
    assert "§23 ```js §24 function f() { §25 return 1; §26 } §27 ```" in flow
    assert "§29 indented code §30 »more" in flow
    assert "§32 > quote line one > quote line two" in flow
    assert "§35 Setext title §36 ============" in flow


RST = """.. _intro:

Title
=====

Paragraph one
continues here.

Some examples::

   spam = 1
   if spam:
       print(spam)

.. code-block:: python
   :linenos:

   def f():
       return 1

.. note::

   Indented note text
   wraps.

>>> 2 + 2
4

- bullet
  more
"""


def test_rst_structure():
    flow = run("rst", RST, path="intro.rst")
    assert "§1 .. _intro: §3 Title §4 =====" in flow
    assert "§6 Paragraph one continues here." in flow
    assert "§11 spam = 1 §12 if spam: §13 »print(spam)" in flow
    assert "§15 .. code-block:: python §16 »:linenos: §18 def f(): §19 »return 1" in flow
    assert "§21 .. note:: §23 »Indented note text wraps." in flow
    assert "§26 >>> 2 + 2 §27 4" in flow
    assert "§29 - bullet more" in flow


TXT = """                 GNU LESSER GENERAL PUBLIC LICENSE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.

1. First condition.
2. Second condition.

Zyyy; 1; 0040; ZZ
Latn; 2; 004C; IT
Hani; 3; 5B57; CN
"""


def test_text_reflow_and_records():
    flow = run("text", TXT, path="LICENSE")
    assert "§3 Permission is hereby granted, free of charge, to any person obtaining a copy of this" in flow
    assert "§7 1. First condition. §8 2. Second condition." in flow
    assert "§10 Zyyy; 1; 0040; ZZ §11 Latn; 2; 004C; IT §12 Hani; 3; 5B57; CN" in flow


# ----------------------------------------------------------------------------------------------- CSV / TSV

CSV = 'name,notes,city\n"Smith, J","line one\nline two",  Paris\nDoe,"say ""hi""",Rome\n\nLee,,Oslo\n'


def test_csv_rows_tagged_and_fields_untouched():
    flow = run("csv", CSV, path="people.csv")
    assert flow == ('§1 name,notes,city §2 "Smith, J","line one §3 line two",  Paris §4 Doe,"say ""hi""",Rome '
                    '§6 Lee,,Oslo')
    rebuilt = cd.rebuild(flow, indent=lambda lvl: "")
    assert list(csv.reader(io.StringIO(rebuilt))) == list(csv.reader(io.StringIO(CSV)))


def test_tsv():
    flow = run("tsv", "id\tname\n1\talpha\n2\tbeta\n", path="t.tsv")
    assert flow == "§1 id name §2 1 alpha §3 2 beta"
    assert minify("id\tname\n1\tNew York\n", path="t.tsv") is None      # a space in a field: ambiguous
    assert minify("id\tname\n1\t\n", path="t.tsv") is None              # an empty field: ambiguous


# ---------------------------------------------------------------------------------------------------- SQL

SQL = """-- Schema
-- second comment line

CREATE TABLE users (
    id      integer NOT NULL,   -- primary key
    name    text    DEFAULT 'a  b',
    bio     text
);

INSERT INTO users VALUES (1, 'multi
  line string', NULL);
/* block
   comment */
CREATE FUNCTION f(a integer) RETURNS integer AS $$
BEGIN
    RETURN a + 1;
END;
$$ LANGUAGE plpgsql;
\\connect other
SELECT a - -1 FROM t;
"""


def test_sql_statements_and_literals():
    flow = run("sql", SQL, path="schema.sql")
    assert flow.startswith("§1 -- Schema §2 -- second comment line §4 CREATE TABLE users (id integer NOT NULL, "
                           "-- primary key §6 name text DEFAULT 'a  b',bio text);")
    assert "§10 INSERT INTO users VALUES (1,'multi §11   line string',NULL);" in flow
    assert "§12 /* block comment */" in flow
    assert "§14 CREATE FUNCTION f(a integer) RETURNS integer AS $$ §15 BEGIN §16 RETURN a + 1; §17 END; " \
           "§18 $$ LANGUAGE plpgsql;" in flow
    assert "§19 \\connect other §20 SELECT a - -1 FROM t;" in flow


def test_sql_mysql_backslash_escape():
    text = "INSERT INTO `t` VALUES ('it\\'s  -- not a comment',\n  2);\nSELECT 1;\n"
    flow = run("sql", text, path="dump.sql")
    assert flow == "§1 INSERT INTO `t` VALUES ('it\\'s  -- not a comment',2); §3 SELECT 1;"


def test_sql_copy_stdin_stays_text():
    assert minify("COPY t (a, b) FROM stdin;\n1\tx\n\\.\n", path="dump.sql") is None


# ------------------------------------------------------------------------------- optional: a real corpus

HINT = {"yaml": "yaml", "toml": "toml", "ini": "ini", "json": "json", "jsonl": "jsonl", "xml": "xml", "svg": "xml",
        "html": "html", "md": "markdown", "rst": "rst", "txt": "text", "csv": "csv", "tsv": "tsv", "sql": "sql"}


def _corpus():
    root = os.environ.get("VTCHOOK_CORPUS")
    if not root or not os.path.isdir(root):
        return []
    return [(sub, os.path.join(root, sub, f)) for sub in sorted(os.listdir(root)) if sub in HINT
            for f in sorted(os.listdir(os.path.join(root, sub)))]


@pytest.mark.skipif(not _corpus(), reason="set VTCHOOK_CORPUS to a directory of real files")
@pytest.mark.parametrize("sub,path", _corpus())
def test_real_corpus(sub, path):
    text = open(path, encoding="utf-8", errors="replace").read()[:400000]
    m = REGISTRY[HINT[sub]]
    try:
        flow = m.fn(text)
    except Exception:  # noqa: BLE001 - declining (e.g. JSON with comments) is allowed
        return
    if not flow:
        return
    assert preserves(m, text, flow) and not re.search(r"[\t\n\r]", flow)
    if sub == "yaml" and yaml is not None:
        assert list(yaml.safe_load_all(cd.yaml_rebuild(flow, cd.yaml_unit(text)))) == list(yaml.safe_load_all(text))
    elif sub == "toml":
        assert tomllib.loads(cd.toml_rebuild(flow)) == tomllib.loads(text)
    elif sub == "json":
        assert json.loads(strip_marks(flow)) == json.loads(text)
    elif sub in ("xml", "svg"):
        assert _etree(strip_marks(flow)) == _etree(text)
    elif sub == "html":
        assert _html_events(strip_marks(flow)) == _html_events(text)
    elif sub == "csv":
        rebuilt = cd.rebuild(flow, indent=lambda lvl: "")
        assert list(csv.reader(io.StringIO(rebuilt))) == list(csv.reader(io.StringIO(text)))
