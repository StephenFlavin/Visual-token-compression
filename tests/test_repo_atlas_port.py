"""Tests for experiments/repo_atlas_port.py (no network, no Claude calls)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import repo_atlas_port as P  # noqa: E402

TEMPLATE = """# Task

Translate the {{ source_desc }} ({{ source_lang }}) to {{ target_desc }} ({{ target_lang }}).

`/workspace/src/` - source.
`/workspace/dst/` - target.

{{ build_instructions }}

```bash
{{ run_cmd }}
```
"""


def test_render_prompt_replaces_workspace_once():
    ws = "/s/runs/plain/workspace"
    out = P.render_prompt(TEMPLATE, ws)
    assert f"`{ws}/src/`" in out and f"`{ws}/dst/`" in out
    assert out.count("/workspace") == 2                      # only the two paths, no doubled prefix
    assert "Go-based gitleaks implementation (Go)" in out
    assert "- Build: `cargo build --release`" in out and "./target/release/gitleaks" in out
    assert "{{" not in out


def test_messages_plain_and_atlas():
    m = P.messages("plain", "TASK")
    assert len(m) == 1 and m[0]["message"]["content"] == [{"type": "text", "text": "TASK"}]
    with pytest.raises(ValueError):
        P.messages("plain", "TASK", [b"x"])
    with pytest.raises(ValueError):
        P.messages("atlas", "TASK", [], "Reply only OK.")
    m = P.messages("atlas", "TASK", [b"png1", b"png2"], "pages... Reply only OK.")
    assert len(m) == 2
    c0 = m[0]["message"]["content"]
    assert [c["type"] for c in c0] == ["image", "image", "text"] and c0[-1]["text"].endswith("Reply only OK.")
    assert m[1]["message"]["content"][0]["text"] == "TASK"    # the task is the same text as in plain


def test_atlas_text_formats():
    t = P.ATLAS_TEXT.format(n_pages=18, ws="/w", index="a.go: page 1\n", n_files=199)
    assert "`/w/src/`" in t and t.endswith(P.ACK) and "199" in t


def asst(mid, out=10, inp=0, cw=0, cr=0, content=None):
    return {"type": "assistant", "message": {"id": mid, "content": content or [], "usage": {
        "input_tokens": inp, "output_tokens": out, "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr}}}


def result(cost, turns=1, ms=1000, subtype="success", is_error=False, text="OK"):
    return {"type": "result", "subtype": subtype, "total_cost_usd": cost, "num_turns": turns, "duration_ms": ms,
            "is_error": is_error, "result": text}


def test_msg_usages_dedupes_and_takes_max():
    ev = [asst("a", out=5, cw=100), asst("a", out=50, cw=100), asst("b", out=7)]
    mu = P.msg_usages(ev)
    assert [m[0] for m in mu] == ["a", "b"]
    assert mu[0][1]["output_tokens"] == 50 and mu[0][1]["cache_creation_input_tokens"] == 100


def test_usage_cost_prices():
    u = {"input_tokens": 1_000_000, "cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
         "output_tokens": 1_000_000}
    assert P.usage_cost(u) == pytest.approx(4 + 5 + 0.2 + 20)


def test_segment_cost_reported_plus_tail():
    ev = [asst("a", out=1_000_000), result(1.5), asst("b", out=100_000)]
    c = P.segment_cost(ev)
    assert c["reported"] == 1.5 and c["tail_est"] == pytest.approx(2.0) and c["cost"] == pytest.approx(3.5)
    assert c["estimated"]
    c = P.segment_cost([asst("a", out=50_000)])             # killed before any result
    assert c["reported"] == 0 and c["cost"] == pytest.approx(1.0)
    c = P.segment_cost([asst("a"), result(0.7)])
    assert c["cost"] == 0.7 and not c["estimated"]


def test_guard_hit():
    assert P.guard_hit(14.99, 100) is None
    assert P.guard_hit(15.0, 100) == "guard"
    assert P.guard_hit(1.0, 2.5 * 3600) == "wall"
    assert P.guard_hit(0.61, 1, guard=0.6, wall_s=900) == "guard"


def tu(i):
    return {"type": "tool_use", "id": f"tu{i}", "name": "Bash", "input": {"command": "ls"}}


def test_task_turns_excludes_ack():
    ev = [asst("ack"), result(0.4, turns=1), asst("t1", content=[tu(1)]), asst("t1", content=[tu(2)]), asst("t2"),
          result(2.0, turns=3, ms=5000)]
    t = P.task_turns(ev, "atlas")
    assert t["api_calls"] == 2 and t["ack_calls"] == 1 and t["cli_turns"] == 3 and t["agent_s"] == 6
    assert t["turns"] == 3                                    # 2 tool calls + 1 answer = the CLI's num_turns
    t = P.task_turns(ev, "plain")
    assert t["api_calls"] == 3 and t["cli_turns"] == 4 and t["turns"] == 4
    t = P.task_turns([asst("ack"), result(0.4), asst("t1", content=[tu(1)])], "atlas")   # killed mid-task
    assert t["api_calls"] == 1 and t["cli_turns"] is None and t["turns"] == 1


def test_limit_hit():
    assert P.limit_hit([asst("a"), result(1, subtype="error", is_error=True, text="You've hit your limit")], "")
    assert not P.limit_hit([asst("a"), result(1)], "")
    assert not P.limit_hit([asst("a"), result(15, subtype="error_max_budget_usd", is_error=True)], "")
    assert P.limit_hit([], "")                                   # no message at all: not a result
    assert not P.limit_hit([asst("a")], "")                      # killed mid-run by us: a result (cut)


def test_leak_hits_specific():
    cmds = ["cd /tmp/claude-0/x/port/runs/plain/workspace/dst && cargo build --release",
            "grep -rn gitleaks src/ | head", "cargo add regex", "git clone https://github.com/x/gitleaks-rs",
            "go install github.com/zricethezav/gitleaks/v8@latest", "ls ~/.cargo/registry"]
    hits = P.leak_hits(cmds)
    assert "cargo add regex" in hits and any("git clone" in h for h in hits) and any("go install" in h for h in hits)
    assert not any("cargo build" in h for h in hits) and not any("grep -rn" in h for h in hits)


def test_cargo_deps_and_flags():
    toml = """[package]\nname = "gitleaks"\n[dependencies]\nregex = "1"\nclap = { version = "4" }\n
[target.'cfg(unix)'.dependencies]\nlibc = "0.2"\n[dev-dependencies]\ntempfile = "3"\n"""
    assert P.cargo_deps(toml) == ["clap", "libc", "regex", "tempfile"]
    assert P.lock_packages('[[package]]\nname = "regex"\n\n[[package]]\nname = "secrets-scan"\n') == ["regex", "secrets-scan"]
    assert [d for d in ["regex", "secrets-scan", "gitleaks-core"] if P.LEAK_DEP.search(d)] == ["secrets-scan", "gitleaks-core"]


def test_concurrency_and_unsafe():
    src = """
use std::sync::{Arc, Mutex, mpsc};
// thread::spawn in a comment does not count
fn scan() { let (tx, rx) = mpsc::channel(); let m = Arc::new(Mutex::new(0));
  std::thread::scope(|s| { s.spawn(|| {}); }); let x = "unsafe { }"; }
"""
    c = P.concurrency(src)
    assert c["concurrent"] and c["mutex"] == 2 and c["arc"] == 2 and c["channels"] >= 2 and c["thread_spawn"] >= 2
    assert not P.concurrency("fn main() { let v: Vec<u8> = vec![]; }")["concurrent"]
    assert P.concurrency("use rayon::prelude::*; fn f(v: &[u8]) { v.par_iter(); }")["concurrent"]
    assert P.count_unsafe(src) == 0
    assert P.count_unsafe("fn f() { unsafe { x() } }\nunsafe fn g() {}") == 2


def test_rust_lines():
    assert P.rust_lines({"src/main.rs": "fn main() {\n\n}\n", "Cargo.toml": "a\nb\n"}) == 2


def test_parse_junit_and_score():
    xml = """<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="test_a.TestX" name="test_ok"/>
<testcase classname="test_a.TestX" name="test_bad"><failure message="x"/></testcase>
<testcase classname="test_b.TestY" name="test_err"><error message="x"/></testcase>
</testsuite></testsuites>"""
    o = P.parse_junit(xml)
    assert o == {"test_a.TestX::test_ok": "passed", "test_a.TestX::test_bad": "failed", "test_b.TestY::test_err": "error"}
    ref = {"test_a.TestX::test_ok": "passed", "test_a.TestX::test_bad": "passed", "test_b.TestY::test_err": "failed"}
    s = P.score(o, ref, True)
    assert s["n_tests"] == 2 and s["n_passed"] == 1 and s["excluded"] == ["test_b.TestY::test_err"] and not s["all_pass"]
    s = P.score(o, ref, False)
    assert s["n_passed"] == 0 and not s["built"]


def test_predictions():
    rows = {"plain": {"all_pass": False, "n_passed": 20, "n_tests": 35, "share": 20 / 35, "turns": 120},
            "atlas": {"all_pass": False, "n_passed": 23, "n_tests": 35, "share": 23 / 35, "turns": 100}}
    p = P.predictions(rows)
    assert p["B1"]["pass"] and p["B2"]["pass"] and p["B3"]["pass"]
    rows["atlas"].update(all_pass=True, n_passed=35, share=1.0, turns=130)
    p = P.predictions(rows)
    assert not p["B1"]["pass"] and not p["B2"]["pass"] and not p["B3"]["pass"]


def test_audit_flags_outside_and_store(tmp_path):
    arm_dir = tmp_path / "runs" / "plain"
    calls = [{"name": "Bash", "input": {"command": f"ls {arm_dir}/workspace/src && cargo build"}},
             {"name": "Read", "input": {"file_path": "/root/.cargo/registry/src/x/lib.rs"}},
             {"name": "Bash", "input": {"command": "cat ~/.claude/projects/x/memory/notes.md"}},
             {"name": "Bash", "input": {"command": "find / -name '*.py' -path '*gitleaks*'"}},
             {"name": "Read", "input": {"file_path": "/home/user/visual-bpe/README.md"}}]
    a = P.audit(calls, arm_dir)
    assert a["suspect"] and a["claude_store"] and a["fs_searches"]
    assert "/home/user/visual-bpe/README.md" in a["outside_paths"]
    assert not any(p.startswith(str(arm_dir)) or p.startswith("/root/.cargo") for p in a["outside_paths"])
    assert not P.audit(calls[:2], arm_dir)["suspect"]


def test_strip_images():
    line = json.dumps({"source": {"data": "A" * 500}})
    assert "<base64 removed>" in P.strip_images(line) and "AAAA" not in P.strip_images(line)


def test_xor_roundtrip():
    assert P.xor(P.xor(b"hello")) == b"hello" and P.xor(b"a") != b"a"


def test_image_tokens():
    assert P.image_tokens([(1932, 1904), (28, 28)]) == 69 * 68 + 1


def sev(kind, mid=None, out=0, cw=0, cr=0):
    if kind == "message_start":
        return {"type": "stream_event", "event": {"type": "message_start", "message": {"id": mid, "usage": {
            "input_tokens": 2, "output_tokens": out, "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr}}}}
    return {"type": "stream_event", "event": {"type": "message_delta", "usage": {"output_tokens": out}}}


def test_msg_usages_reads_message_delta():
    # assistant events carry the output count at the message's start; message_delta has the final count
    ev = [sev("message_start", "a", out=10, cw=500), asst("a", out=10, cw=500), sev("message_delta", out=900),
          sev("message_start", "b", out=3, cr=600), asst("b", out=3, cr=600), sev("message_delta", out=40)]
    mu = P.msg_usages(ev)
    assert [(m[0], m[1]["output_tokens"]) for m in mu] == [("a", 900), ("b", 40)]
    assert mu[0][1]["cache_creation_input_tokens"] == 500 and mu[1][1]["cache_read_input_tokens"] == 600
    assert mu[0][2] == 0 and mu[1][2] == 3


def test_transcript_line_drops_token_deltas():
    d = json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"text": "x"}}})
    s = json.dumps({"type": "stream_event", "event": {"type": "message_delta", "usage": {"output_tokens": 3}}})
    a = json.dumps({"type": "assistant", "message": {"id": "a"}})
    assert P.transcript_line(d) is None and P.transcript_line(s) == s and P.transcript_line(a) == a


def test_unsafe_after_char_literal():
    src = "fn f(c: char) -> bool { c == '\"' }\nfn g() { unsafe { x() } }\nfn h() { let q = '\\''; unsafe { y() } }"
    assert P.count_unsafe(src) == 2
