"""Pure functions of experiments/repo_atlas_migrate.py (no cargo, no Claude calls)."""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import repo_atlas_migrate as M  # noqa: E402


def test_norm_name_matches_both_spellings():
    assert M.norm_name("c-aces") == M.norm_name("c_aces")
    assert M.norm_name("proj_2DPartInt") == M.norm_name("2DPartInt")
    assert M.norm_name("Holdem_Odds") == M.norm_name("Holdem-Odds")


def test_c_stats_counts_c_and_h_only():
    s = M.c_stats({"a.c": "x\ny\n", "inc/a.h": "z", "Makefile": "all:\n\tcc\n", "README.md": "hi"})
    assert s == {"c_files": 2, "c_lines": 3}


def test_selection_rule_range_files_and_order():
    st = [{"project": "big", "c_lines": 4001, "c_files": 9},
          {"project": "two_files", "c_lines": 2000, "c_files": 2},
          {"project": "a", "c_lines": 1500, "c_files": 3},
          {"project": "b", "c_lines": 3000, "c_files": 5},
          {"project": "small", "c_lines": 799, "c_files": 4},
          {"project": "edge", "c_lines": 800, "c_files": 3}]
    assert [s["project"] for s in M.order_candidates(st)] == ["b", "a", "edge"]
    assert [s["project"] for s in M.order_candidates(st, largest_first=False)] == ["edge", "a", "b"]


LOG = """   Compiling foo v0.1.0
     Running unittests src/lib.rs (target/debug/deps/foo-1)
running 0 tests
     Running unittests src/bin/test_a.rs (target/debug/deps/test_a-2)
running 2 tests
test test_one ... ok
test test_two ... FAILED
     Running unittests src/bin/test_b.rs (target/debug/deps/test_b-3)
running 2 tests
test test_one ... ok
test test_slow ... ignored
"""


def test_parse_tests_keys_by_binary_and_keeps_bench_counts():
    p = M.parse_tests(LOG)
    assert p["tests"] == {"test_a::test_one": "ok", "test_a::test_two": "FAILED", "test_b::test_one": "ok",
                          "test_b::test_slow": "ignored"}
    assert p["bench_ok"] == 2 and p["bench_failed"] == 1


def test_score_counts_missing_tests_as_failing():
    base = ["test_a::test_one", "test_a::test_two", "test_c::t"]
    s = M.score_tests({"test_a::test_one": "ok", "test_a::test_two": "ok"}, base, built=True)
    assert s["n_passed"] == 2 and not s["all_pass"] and abs(s["share"] - 2 / 3) < 1e-9
    s = M.score_tests({t: "ok" for t in base}, base, built=True)
    assert s["all_pass"] and s["share"] == 1.0
    s = M.score_tests({t: "ok" for t in base}, base, built=False)
    assert not s["all_pass"] and s["share"] == 0.0


def test_count_unsafe_ignores_comments_and_strings():
    src = '''
    // unsafe { not this }
    /* unsafe fn nor this */
    let s = "unsafe { nor this }";
    unsafe { ptr.read() }
    pub unsafe fn f() {}
    unsafe impl Send for X {}
    fn unsafe_name() {}
    '''
    assert M.count_unsafe(src) == 3


IFACE = """
pub struct Buf { data: Vec<u8> }
impl Buf {
    pub fn new(size: usize) -> Self { unimplemented!() }
    pub fn push(&mut self, src: &[u8],
                length: usize) { unimplemented!() }
    fn private(&self) {}
}
pub const MAX: usize = 4;
"""


def test_pub_signatures_normalise_whitespace():
    sig = M.pub_signatures(IFACE)
    assert "fn push(&mut self,&[u8],usize)" in sig
    assert "fn new(usize)->Self" in sig
    # renaming a parameter (e.g. dropping a leading underscore) does not change the interface; a type change does
    assert M.pub_signatures("pub fn f(_a: &mut X) {}") == M.pub_signatures("pub fn f(a: &mut X) {}")
    assert M.pub_signatures("pub fn f(a: u32) {}") != M.pub_signatures("pub fn f(a: u64) {}")
    assert "struct Buf" in sig and "const MAX" in sig
    assert not any("private" in s for s in sig)


def test_integrity_flags_test_edits_additions_and_signature_changes():
    pristine = {"src/buf.rs": IFACE.encode(), "src/bin/test_buf.rs": b"#[test] fn t() {}", "Cargo.toml": b"[package]"}
    impl = IFACE.replace("{ unimplemented!() }", "{ todo_impl() }").replace("    fn private(&self) {}",
                                                                           "    fn private(&self) {}\n    fn helper() {}")
    ok = {**pristine, "src/buf.rs": impl.encode()}
    r = M.integrity(pristine, ok)
    assert r["tests_ok"] and r["interfaces_ok"] and not r["cargo_toml_changed"]
    assert M.integrity(pristine, {**ok, "src/bin/result.txt": b"output"})["tests_ok"]
    bad = {**ok, "src/bin/test_buf.rs": b"#[test] fn t() { assert!(true) }", "src/bin/extra.rs": b"",
           "src/buf.rs": impl.replace("length: usize", "length: u32").encode(), "Cargo.toml": b"[package]\n[deps]"}
    r = M.integrity(pristine, bad)
    assert r["tests_changed"] == ["src/bin/test_buf.rs"] and r["tests_added"] == ["src/bin/extra.rs"]
    assert not r["interfaces_ok"] and "src/buf.rs" in r["signatures_missing"] and r["cargo_toml_changed"]


def test_first_message_per_arm():
    png = b"\x89PNG fake"
    m = M.first_message("plain", [])
    c = m["message"]["content"]
    assert len(c) == 1 and "`c_src/`" in c[0]["text"] and "page images" not in c[0]["text"]
    m = M.first_message("atlas", [png, png], "a.c: page 1\nb.c: page 2\n")
    c = m["message"]["content"]
    assert [x["type"] for x in c] == ["image", "image", "text"]
    assert base64.b64decode(c[0]["source"]["data"]) == png
    assert "`c_src/`" in c[-1]["text"] and "b.c: page 2" in c[-1]["text"] and "read the files" in c[-1]["text"]
    m = M.first_message("atlas_only", [png], "a.c: page 1")
    t = m["message"]["content"][-1]["text"]
    assert "only" in t and "c_src" not in t and "a.c: page 1" in t
    # the shared instruction is the same in every arm
    for arm in M.ARMS:
        t = M.instruction(arm, "x")
        assert "Do not edit the tests" in t and "Do not use `unsafe`" in t and t.endswith("reply DONE.")


def test_first_message_rejects_wrong_page_sets():
    import pytest
    with pytest.raises(ValueError):
        M.first_message("plain", [b"x"])
    with pytest.raises(ValueError):
        M.first_message("atlas_only", [])


def test_image_tokens_formula():
    assert M.image_tokens([(1932, 1932)]) == 69 * 69
    assert M.image_tokens([(28, 29), (1, 1)]) == 2 + 1


def test_audit_flags_outside_reads_network_and_searches():
    cwd = "/tmp/claude-0/x/mig/runs/p/atlas_only"
    calls = [{"name": "Bash", "input": {"command": f"cd {cwd}/rust && cargo test"}},
             {"name": "Read", "input": {"file_path": f"{cwd}/rust/src/lib.rs"}},
             {"name": "Bash", "input": {"command": "ls /root/.cargo/registry"}}]
    assert not M.audit_calls(calls, cwd)["suspect"]
    for bad in ({"name": "Bash", "input": {"command": "find / -name '*.c'"}},
                {"name": "Read", "input": {"file_path": "/tmp/claude-0/x/mig/store/cbench.bin"}},
                {"name": "Bash", "input": {"command": "git clone https://github.com/x/y"}}):
        assert M.audit_calls(calls + [bad], cwd)["suspect"], bad


def test_run_cost_and_exact_tests():
    u = [{"input_tokens": 1_000_000, "output_tokens": 0}, {"output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000}]
    assert abs(M.run_cost(u) - (4 + 20 + 0.2)) < 1e-9
    assert M.exact_binom_two_sided(0, 0) == 1.0
    assert abs(M.exact_binom_two_sided(0, 5) - 0.0625) < 1e-12
    assert abs(M.exact_binom_two_sided(5, 5) - 0.0625) < 1e-12
    assert M.sign_test([1, -1, 0, 2])["pos"] == 2 and M.median([3, 1, 2, None]) == 2


def test_xor_round_trip():
    b = bytes(range(256)) * 3
    assert M.xor(M.xor(b)) == b and M.xor(b) != b and b"int main" not in M.xor(b"int main(void)")


def test_limit_hit_separates_limits_from_agent_endings():
    assert M.limit_hit({"is_error": True, "subtype": "success", "result": "You've hit your limit · resets 9:10am"}, "", 3, False)
    assert M.limit_hit(None, "", 0, False)
    assert not M.limit_hit(None, "", 12, True)                       # wall clock: graded as left
    assert not M.limit_hit({"is_error": True, "subtype": "error_max_turns", "result": ""}, "", 60, False)
    assert not M.limit_hit({"is_error": True, "subtype": "error_max_budget_usd", "result": ""}, "", 40, False)
    assert not M.limit_hit({"is_error": False, "subtype": "success", "result": "DONE"}, "", 30, False)


def test_parse_bin_keys_by_binary():
    out = "running 2 tests\ntest t_one ... ok\ntest t_two ... FAILED\n"
    assert M.parse_bin(out, "test_a") == {"test_a::t_one": "ok", "test_a::t_two": "FAILED"}


def test_audit_ignores_comment_slashes_and_own_cli_store():
    cwd = "/tmp/claude-0/x/mig/runs/p/plain"
    own = str(M.cli_store(M.Path(cwd)))
    calls = [{"name": "Write", "input": {"file_path": f"{cwd}/rust/src/a.rs", "content": "// note\n/// doc\nlet x = a / b;"}},
             {"name": "Read", "input": {"file_path": f"{own}/abc/tool-results/x.txt"}}]
    assert not M.audit_calls(calls, cwd)["suspect"]
    other = str(M.cli_store(M.Path("/tmp/claude-0/x/mig/runs/p/atlas")))
    assert M.audit_calls(calls + [{"name": "Bash", "input": {"command": f"ls {other}/memory"}}], cwd)["suspect"]


def test_result_summary_sums_segments_and_keeps_cumulative_cost():
    ev = [{"type": "result", "num_turns": 29, "total_cost_usd": 2.09, "duration_ms": 741794, "subtype": "success", "result": "DONE"},
          {"type": "assistant"}, {"type": "result", "num_turns": 2, "total_cost_usd": 2.14, "duration_ms": 4335, "result": "x"}]
    s = M.result_summary(ev)
    assert s["usage"]["output_tokens"] == 0 and s["turns"] == 31 and s["cost_usd"] == 2.14 and s["agent_s"] == 746 and s["n_results"] == 2 and s["final"] == "DONE"
    assert M.result_summary([]) == {}


def test_pub_signatures_ignore_trailing_comma():
    assert M.pub_signatures("pub fn w(a: i32,\n b: &str,\n) {}") == M.pub_signatures("pub fn w(a: i32, b: &str) {}")
