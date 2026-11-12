"""repo_atlas_eval.py: gold files, gold functions, the map, answer parsing and scoring. No network, no model."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import repo_atlas_eval as E  # noqa: E402

PATCH = """diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -386,7 +386,8 @@ def prepare_body(self, data, files):
-        x = 1
+        x = 2
diff --git a/CHANGELOG.md b/CHANGELOG.md
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -1 +1,2 @@
+- fix
diff --git a/tests/test_requests.py b/tests/test_requests.py
--- a/tests/test_requests.py
+++ b/tests/test_requests.py
@@ -10 +10,3 @@ class RequestsTestCase(unittest.TestCase):
+    def test_new(self):
diff --git a/binding/header.go b/binding/header.go
new file mode 100644
--- /dev/null
+++ b/binding/header.go
@@ -0,0 +1,5 @@
+type headerBinding struct{}
+func (headerBinding) Name() string {
+func mapHeader(ptr interface{}, h map[string][]string) error {
diff --git a/docs/usage.rst b/docs/usage.rst
--- a/docs/usage.rst
+++ b/docs/usage.rst
@@ -1 +1 @@
-a
+b
diff --git a/context_test.go b/context_test.go
--- a/context_test.go
+++ b/context_test.go
@@ -1 +1 @@ func TestContext(t *testing.T) {
-a
+b
"""


def test_patch_and_gold_files():
    assert E.patch_files(PATCH) == ["requests/models.py", "CHANGELOG.md", "tests/test_requests.py", "binding/header.go",
                                    "docs/usage.rst", "context_test.go"]
    assert E.gold_files(PATCH) == ["requests/models.py", "binding/header.go"]


def test_gold_functions_from_headers_and_changed_lines_of_gold_files_only():
    f = E.gold_functions(PATCH)
    assert f == ["prepare_body", "headerBinding", "Name", "mapHeader"]
    assert "test_new" not in f and "TestContext" not in f and "RequestsTestCase" not in f


def test_decl_names_languages():
    assert E.decl_names("pub fn strip_suffix(&self, s: &str) -> Option<&str> {") == ["strip_suffix"]
    assert E.decl_names("export function useContext(context) {") == ["useContext"]
    assert E.decl_names("Component.prototype.forceUpdate = function(callback) {") == ["forceUpdate"]
    assert E.decl_names("func (c *Context) ShouldBindHeader(obj any) error {") == ["ShouldBindHeader"]
    assert E.decl_names("        if x:") == []


def test_python_signatures_nested_and_multiline():
    src = ("import os\n\n@deco\nclass A(B):\n    \"\"\"doc\"\"\"\n    def m(self,\n          x=1):\n        pass\n\n"
           "    async def n(self): return 1\n\ndef f(a, b): pass\n\nif True:\n    def g(): pass\n")
    assert E.signatures("pkg/a.py", src) == ["class A(B)", "  def m(self, x=1)", "  async def n(self)", "def f(a, b)",
                                             "def g()"]


def test_python_signatures_fall_back_to_regex_on_py2():
    src = "def f(x):\n    print 'hi'\n\nclass C:\n    pass\n"
    assert E.signatures("a.py", src) == ["def f(x):", "class C:"]


def test_go_rust_js_signatures():
    go = "package gin\n\ntype Context struct {\n\tx int\n}\n\nfunc (c *Context) Bind(obj any) error {\n\treturn nil\n}\n"
    assert E.signatures("context.go", go) == ["type Context struct", "func (c *Context) Bind(obj any) error"]
    rs = "pub struct A {\n}\nimpl A {\n    pub fn new() -> Self {\n        if x {\n        }\n    }\n}\nfn helper(x: u8) {}\n"
    assert E.signatures("src/a.rs", rs) == ["pub struct A", "impl A", "    pub fn new() -> Self", "fn helper(x: u8)"]
    js = ("export function render(vnode, parent) {\n  if (x) {\n  }\n}\nexport const h = (a) => a;\n"
          "Component.prototype.setState = function(update) {\n};\nclass X {\n  componentDidCatch(err) {\n  }\n}\n"
          "export interface Props {\n}\n")
    assert E.signatures("src/index.js", js) == ["export function render(vnode, parent)", "export const h = (a) => a;",
                                                "Component.prototype.setState = function(update)", "class X",
                                                "  componentDidCatch(err)", "export interface Props"]
    assert E.signatures("Cargo.toml", "[package]\n") == []


def test_build_map():
    m = E.build_map([("a.py", "def f():\n    pass\n"), ("README", "x")])
    assert m == "a.py\n  def f()\nREADME\n"


def test_parse_answer_variants():
    text = ("Thinking...\nFILE: `requests/models.py`\n- FILE: ./requests/sessions.py:120\n**FILE:** requests/models.py\n"
            "1. FILE: /work/requests/requests/utils.py (maybe)\nFILE: a\nFILE: b\nFILE: c\nPLAN: change prepare_body.\nPLAN: x")
    files, plan = E.parse_answer(text, cwd="/work/requests")
    assert files == ["requests/models.py", "requests/sessions.py", "requests/utils.py", "a", "b"]
    assert plan == "change prepare_body."
    assert E.parse_answer("I can't help with that.") == ([], "")


def test_score():
    s = E.score(["a.py", "b.py", "x.py"], "Fix prepare_body in A", ["b.py", "a.py"], ["prepare_body", "send", "A"],
                {"a.py", "b.py"})
    assert s["all_gold_top5"] and s["first_right"] and s["gold_share"] == 1.0
    assert s["funcs_named"] == ["prepare_body", "A"] and abs(s["funcs_share"] - 2 / 3) < 1e-9
    assert s["not_in_repo"] == ["x.py"]
    s = E.score(["x.py", "a.py"], "", ["a.py", "b.py"], [], None)
    assert not s["all_gold_top5"] and not s["first_right"] and s["gold_share"] == 0.5 and s["funcs_share"] is None
    s = E.score([], "", ["a.py"], ["f"], set())
    assert not s["all_gold_top5"] and not s["first_right"] and s["funcs_named"] == []


def test_function_names_match_whole_words_only():
    s = E.score(["a.py"], "call requests() and run_all", ["a.py"], ["request", "run"], None)
    assert s["funcs_named"] == []


def test_image_tokens():
    assert E.image_tokens([(1932, 1932), (1932, 100)]) == 69 * 69 + 69 * 4
