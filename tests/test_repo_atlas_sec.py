"""repo_atlas_sec.py: advisory parsing, the selection rule, gold functions, finding parsing, scoring, the CWE mapping
and the exact test. No network, no model."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import repo_atlas_sec as S  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_fix_commits_distinct_and_prefix():
    refs = [{"url": f"https://github.com/Owner/Repo/commit/{SHA}"},
            {"url": f"https://github.com/owner/repo/commit/{SHA[:7]}"},
            {"url": f"https://github.com/owner/repo/pull/12/commits/{SHA}"},
            {"url": "https://github.com/owner/repo/pull/12"},
            {"url": "https://github.com/owner/repo/security/advisories/GHSA-xxxx"},
            {"url": f"https://github.com/owner/repo/commit/{SHA}.patch"}]
    assert S.fix_commits(refs) == [("owner/repo", SHA)]
    two = refs + [{"url": "https://github.com/owner/repo/commit/" + "f" * 40}]
    assert len(S.fix_commits(two)) == 2
    assert S.fix_commits([{"url": "https://gitlab.com/a/b/-/commit/" + SHA}]) == []


def test_parse_and_order_advisories():
    a = S.parse_advisory({"id": "GHSA-1", "published": "2026-08-01T00:00:00Z",
                          "database_specific": {"cwe_ids": ["CWE-22"], "severity": "HIGH"},
                          "affected": [{"package": {"ecosystem": "npm", "name": "x"}}],
                          "references": [{"url": f"https://github.com/o/r/commit/{SHA}"}]})
    assert a["cwes"] == ["CWE-22"] and a["ecosystems"] == ["npm"] and a["commits"] == [("o/r", SHA)]
    b = dict(a, id="GHSA-2", published="2026-09-01T00:00:00Z")
    c = dict(a, id="GHSA-3", published="2026-06-30T23:59:59Z")
    assert [x["id"] for x in S.order_advisories([a, b, c])] == ["GHSA-2", "GHSA-1"]


def test_classify_fix():
    assert S.classify_fix(["CHANGELOG.md", "tests/test_x.py", "src/x.py", "package.json"]) == (None, ["src/x.py"])
    assert S.classify_fix(["README.md", "tests/test_x.py"])[0] == "no source file changed"
    assert S.classify_fix(["lib/a.rb", "lib/b.py"])[0] == "a changed source file has no minifier"
    assert S.classify_fix([f"src/f{i}.go" for i in range(6)])[0] == "more than 5 source files changed"
    assert S.classify_fix(["src/a_test.go", "src/a.go"]) == (None, ["src/a.go"])


PATCH = """diff --git a/src/z.py b/src/z.py
--- a/src/z.py
+++ b/src/z.py
@@ -4,3 +4,4 @@ class Box:
     def open(self, p):
-        return read(p)
+        check(p)
+        return read(p)

diff --git a/tests/test_z.py b/tests/test_z.py
--- a/tests/test_z.py
+++ b/tests/test_z.py
@@ -1 +1,2 @@
+def test_open(): pass
"""
OLD = "import os\n\nclass Box:\n    def open(self, p):\n        return read(p)\n\n    def close(self):\n        pass\n"


def test_patch_sections_and_changed_lines():
    sec = S.patch_sections(PATCH, {"src/z.py"})
    assert "tests/test_z.py" not in sec and "src/z.py" in sec
    assert S.changed_old_lines(sec) == [5]


def test_gold_functions_python_and_java():
    assert S.gold_functions(PATCH, ["src/z.py"], {"src/z.py": OLD}) == ["open"]
    java_old = ("package a;\npublic final class ZipUtil {\n  private ZipUtil() {}\n"
                "  public static void unpack(File zip, File out) {\n    String name = e.getName();\n"
                "    File f = new File(out, name);\n  }\n}\n")
    java_patch = ("diff --git a/src/ZipUtil.java b/src/ZipUtil.java\n--- a/src/ZipUtil.java\n+++ b/src/ZipUtil.java\n"
                  "@@ -5,2 +5,3 @@ public final class ZipUtil {\n     String name = e.getName();\n"
                  "+    checkName(name);\n     File f = new File(out, name);\n")
    assert S.gold_functions(java_patch, ["src/ZipUtil.java"], {"src/ZipUtil.java": java_old}) == ["unpack"]


def test_decl_name():
    assert S.decl_name("func (s *Server) Handle(w http.ResponseWriter) {") == "Handle"
    assert S.decl_name("  private function batchDownloadStart(Request $r)") == "batchDownloadStart"
    assert S.decl_name("export const sanitizeName = (n: string) => {") == "sanitizeName"
    assert S.decl_name("    if (x) {") is None
    assert S.decl_name("    return foo(bar);") is None
    assert S.decl_name("internal fun zipEntry(name: String): ZipEntry {") == "zipEntry"
    assert S.decl_name("tls.parseHelloMessage = function(c, record, length) {") == "parseHelloMessage"
    assert S.decl_name("  ZipEntry(name).apply {") is None
    assert S.decl_name("    foo(a, {") is None
    assert S.decl_name("    // Block SQLite function calls (randomblob, zeroblob),") is None
    assert S.decl_name("  async load(path) {") == "load"
    assert S.decl_name("    public function upload(Request $request)") == "upload"


def test_enclosing_php_brace_line_and_kotlin_params():
    php = ["class C", "{", "    public function upload(Request $r)", "    {", "        if ($a) {", "            $x = 1;",
           "        }", "    }", "}"]
    assert S.enclosing(php, 6) == "upload"
    kt = ["internal inline fun zipEntry(", "  name: String,", "): ZipEntry =", "  ZipEntry(name).apply {", "  }"]
    assert S.enclosing(kt, 3) == "zipEntry" and S.enclosing(kt, 4) == "zipEntry"


def test_parse_findings():
    text = ("Some preamble.\n"
            "FINDING: src/z.py | Box.open (lines 3-5) | CWE-22 Path Traversal | joins user input to a path\n"
            "- **FINDING:** `./src/y.py:10` | load | CWE-502 | unsafe load\n"
            "FINDING: src/z.py | Box.open (lines 3-5) | CWE-22 Path Traversal | joins user input to a path\n"
            "FINDING: z/src/w.py | f | ReDoS | slow regex\n"
            "FINDING: only-a-file.py\n")
    f = S.parse_findings(text, None, {"src/z.py", "src/y.py", "src/w.py"})
    assert [x["file"] for x in f] == ["src/z.py", "src/y.py", "src/w.py", "only-a-file.py"]
    assert f[0]["cls"].startswith("CWE-22") and f[3]["where"] == ""
    agent = S.parse_findings("FINDING: /tmp/x/repo/src/z.py | open | CWE-22 | why", "/tmp/x/repo", {"src/z.py"})
    assert agent[0]["file"] == "src/z.py"
    many = "\n".join(f"FINDING: f{i}.py | g | CWE-20 | w" for i in range(8))
    assert len(S.parse_findings(many)) == 5


def test_score():
    fnd = S.parse_findings("FINDING: a.py | x | CWE-79 | w\nFINDING: src/z.py | open() at 3-5 | CWE-22 path traversal | w\n"
                           "FINDING: nope.py | y | z | w")
    sc = S.score(fnd, ["src/z.py"], ["open"], {"path"}, {"a.py", "src/z.py"})
    assert sc["gold_top5"] and not sc["gold_first"] and sc["gold_rank"] == 2
    assert sc["func_named"] and sc["class_match"] and sc["n_findings"] == 3 and sc["not_in_repo"] == ["nope.py"]
    miss = S.score(fnd[:1], ["src/z.py"], ["open"], {"path"})
    assert not miss["gold_top5"] and not miss["func_named"] and not miss["class_match"]
    # a function named only in a finding about another file does not count
    other = S.parse_findings("FINDING: a.py | open | CWE-22 | w\nFINDING: src/z.py | read | CWE-22 | w")
    assert not S.score(other, ["src/z.py"], ["open"], {"path"})["func_named"]


def test_cwe_families():
    assert S.cwe_family("CWE-22") == "path" and S.cwe_family(79) == "xss" and S.cwe_family("CWE-99999") == "other"
    assert S.cwe_family(None) == "other"
    assert S.class_families("CWE-23 Relative Path Traversal") == {"path"}
    assert S.class_families("Zip Slip (CWE-22) / CWE-59") == {"path"}
    assert S.class_families("Regular expression denial of service") == {"dos"}
    assert S.class_families("Missing authorization") == {"access"}
    assert S.class_families("Improper signature verification") == {"crypto"}
    assert S.class_families("something vague") == set()
    for fam, ids in S.FAMILY_CWES.items():
        assert all(isinstance(i, int) for i in ids), fam


def test_mcnemar():
    assert S.mcnemar([True] * 5, [True] * 5)["p"] == 1.0
    m = S.mcnemar([True] * 6 + [False] * 4, [False] * 6 + [False] * 4)
    assert m["a_only"] == 6 and m["b_only"] == 0 and abs(m["p"] - 2 / 64) < 1e-12
    m = S.mcnemar([True, False, True], [False, True, True])
    assert m["a_only"] == 1 and m["b_only"] == 1 and m["p"] == 1.0


def test_prompts_same_instruction_no_exploit_request():
    common = dict(repo="o/r", sha="abc1234", paths=["src/a.py", "src/b.py"], index="src/a.py: page 1\n")
    texts = {a: S.prompt_text(a, **common) for a in S.ARMS}
    for a, t in texts.items():
        assert t.endswith(S.INSTRUCTION), a
        assert "no exploit code, no proof of concept" in t
    assert "src/b.py" in texts["closed"] and "page 1" in texts["atlas"] and "page 1" in texts["agent_atlas"]
    assert S.AGENT_ATLAS_NOTE in texts["agent_atlas"] and S.AGENT_ATLAS_NOTE not in texts["agent"]
    assert "page" not in texts["agent"] and "src/a.py" not in texts["agent"]


def test_run_order_round_robin_by_size():
    ts = [{"id": "a1", "repo": "a", "atlas_image_tokens": 50}, {"id": "b1", "repo": "b", "atlas_image_tokens": 90},
          {"id": "a2", "repo": "a", "atlas_image_tokens": 20}, {"id": "c1", "repo": "c", "atlas_image_tokens": 70},
          {"id": "b2", "repo": "b", "atlas_image_tokens": 10}]
    assert [t["id"] for t in S.run_order(ts)] == ["b2", "a2", "c1", "a1", "b1"]
