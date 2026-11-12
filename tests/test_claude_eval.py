"""Unit tests for vbpe.claude_eval. The CLI subprocess is monkeypatched: no real Claude calls."""
from __future__ import annotations

import json
import threading
import time

import pytest
from PIL import Image

from vbpe import claude_eval as ce
from vbpe import config


def _result_stream(text: str, *, model: str = "claude-opus-5-5", cost: float = 0.01,
                   is_error: bool = False, subtype: str = "success", api_status=None,
                   extra_models: dict | None = None) -> str:
    mu = {model: {"inputTokens": 2, "outputTokens": 7, "cacheReadInputTokens": 10,
                  "cacheCreationInputTokens": 100, "costUSD": cost}}
    mu.update(extra_models or {})
    events = [
        {"type": "system", "subtype": "init", "model": model},
        {"type": "assistant", "message": {"model": model, "content": [{"type": "text", "text": text}]}},
        {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed",
                                                          "unifiedWindows": {"five_hour": {"utilization": 0.1}}}},
        {"type": "result", "subtype": subtype, "is_error": is_error, "result": text,
         "api_error_status": api_status, "total_cost_usd": cost, "stop_reason": "end_turn",
         "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 10,
                   "output_tokens": 7, "output_tokens_details": {"thinking_tokens": 0}},
         "modelUsage": mu},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


class FakeCLI:
    """Scripted replacement for ce._run_cli. ``script`` items: (rc, stdout, stderr) or callable."""

    def __init__(self, script=None, default=None):
        self.script = list(script or [])
        self.default = default
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, cmd, stdin, env, timeout):
        with self.lock:
            self.calls.append({"cmd": cmd, "stdin": stdin, "env": env, "timeout": timeout})
            item = self.script.pop(0) if self.script else self.default
        if callable(item):
            return item(cmd, stdin)
        return item


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "CACHE_DIR", tmp_path / "cache")
    sleeps: list[float] = []
    monkeypatch.setattr(ce, "_sleep", lambda s: sleeps.append(s))

    def install(script=None, default=None):
        f = FakeCLI(script, default if default is not None else (0, _result_stream("hello"), ""))
        f.sleeps = sleeps
        monkeypatch.setattr(ce, "_run_cli", f)
        return f
    return install


def img(w=40, h=20, color=255, mode="L"):
    return Image.new(mode, (w, h), color)


# ---------------------------------------------------------------- basic call + parsing
def test_transcribe_parses_result_and_builds_message(fake):
    f = fake()
    r = ce.transcribe([img(), img(30, 30, 0)], "SYS", "PROMPT")
    assert r["text"] == "hello" and r["error"] is None and r["cached"] is False
    assert r["input_tokens"] == 112 and r["output_tokens"] == 7
    assert r["cost_usd"] == pytest.approx(0.01) and r["model"] == "claude-opus-5-5"
    assert r["rate_limit"] == {"status": "allowed", "five_hour": 0.1}
    call = f.calls[0]
    assert "--system-prompt" in call["cmd"] and call["cmd"][call["cmd"].index("--system-prompt") + 1] == "SYS"
    assert call["cmd"][call["cmd"].index("--tools") + 1] == ""
    assert call["cmd"][call["cmd"].index("--model") + 1] == ce.DEFAULT_MODEL
    msg = json.loads(call["stdin"])
    content = msg["message"]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]  # images first, in order
    assert content[-1]["text"] == "PROMPT"
    assert content[0]["source"]["media_type"] == "image/png"
    # parent-session variables must not leak into the child
    assert not any(k.startswith("CLAUDE_CODE_SESSION") or k == "CLAUDECODE" for k in call["env"])
    assert call["env"]["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "1"


def test_served_model_comes_from_model_usage(fake):
    aux = {"claude-haiku-x": {"inputTokens": 1000, "outputTokens": 3, "costUSD": 0.0001}}
    fake(default=(0, _result_stream("x", model="claude-sonnet-z", extra_models=aux), ""))
    r = ce.transcribe([img()], "s", "p", model="claude-opus-5-5")
    assert r["model"] == "claude-sonnet-z"
    assert set(r["model_usage"]) == {"claude-sonnet-z", "claude-haiku-x"}


# ---------------------------------------------------------------- cache
def test_cache_hit_and_keying(fake):
    f = fake()
    a = ce.transcribe([img()], "s", "p")
    b = ce.transcribe([img()], "s", "p")
    assert len(f.calls) == 1 and b["cached"] is True and b["text"] == a["text"]
    # every key component changes the key
    ce.transcribe([img()], "s2", "p")
    ce.transcribe([img()], "s", "p2")
    ce.transcribe([img()], "s", "p", model="other-model")
    ce.transcribe([img(color=0)], "s", "p")           # different pixels
    ce.transcribe([img(), img()], "s", "p")           # different number of images
    assert len(f.calls) == 6
    # use_cache=False forces a call
    ce.transcribe([img()], "s", "p", use_cache=False)
    assert len(f.calls) == 7


def test_cache_salt_default_key_unchanged_and_salt_only_changes_key(fake):
    import hashlib
    pngs = [ce.png_bytes(img()), ce.png_bytes(img(color=0))]
    # the key as computed before cache_salt existed
    payload = {"model": "m", "system": "s", "prompt": "p", "images": [hashlib.sha256(b).hexdigest() for b in pngs],
               "effort": "high"}
    old = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert ce.cache_key("m", "s", "p", pngs, "high") == old
    assert ce.cache_key("m", "s", "p", pngs, "high", None) == old
    assert ce.cache_key("m", "s", "p", pngs, "high", "1") not in (old, ce.cache_key("m", "s", "p", pngs, "high", "2"))
    # a salted repeat is a new call with a byte-identical request; the unsalted entry stays cached
    f = fake()
    ce.transcribe([img()], "s", "p", effort="high")
    ce.transcribe([img()], "s", "p", effort="high", cache_salt="1")
    ce.transcribe([img()], "s", "p", effort="high", cache_salt="1")
    assert len(f.calls) == 2
    assert f.calls[0]["stdin"] == f.calls[1]["stdin"] and f.calls[0]["cmd"] == f.calls[1]["cmd"]


def test_cache_key_order_sensitive():
    a, b = ce.png_bytes(img(color=0)), ce.png_bytes(img(color=255))
    assert ce.cache_key("m", "s", "p", [a, b]) != ce.cache_key("m", "s", "p", [b, a])
    assert ce.cache_key("m", "s", "p", [a]) == ce.cache_key("m", "s", "p", [ce.png_bytes(img(color=0))])


def test_errors_are_not_cached(fake):
    f = fake(default=(1, "", "boom"))
    r = ce.transcribe([img()], "s", "p", max_retries=0)
    assert r["error"] and r["text"] == ""
    assert not list(ce.CACHE_DIR.glob("*.json")) if ce.CACHE_DIR.exists() else True
    f.default = (0, _result_stream("ok"), "")
    r2 = ce.transcribe([img()], "s", "p")
    assert r2["text"] == "ok" and r2["cached"] is False


# ---------------------------------------------------------------- retries
def test_retry_on_nonzero_exit_then_success(fake):
    f = fake(script=[(1, "", "Error: overloaded"), (1, "", "socket hang up")],
             default=(0, _result_stream("fine"), ""))
    r = ce.transcribe([img()], "s", "p", max_retries=5)
    assert r["text"] == "fine" and r["attempts"] == 3 and len(f.calls) == 3
    assert len(f.sleeps) == 2 and all(s > 0 for s in f.sleeps)


def test_retry_on_missing_result_event(fake):
    f = fake(script=[(0, '{"type":"system","subtype":"init"}\n', "")],
             default=(0, _result_stream("fine"), ""))
    r = ce.transcribe([img()], "s", "p")
    assert r["text"] == "fine" and len(f.calls) == 2


def test_retry_on_rate_limit_error_result(fake):
    f = fake(script=[(1, _result_stream("Rate limit reached", is_error=True, subtype="error_during_execution",
                                        api_status=429), "")],
             default=(0, _result_stream("fine"), ""))
    r = ce.transcribe([img()], "s", "p")
    assert r["text"] == "fine" and len(f.calls) == 2


def test_gives_up_after_max_retries(fake):
    f = fake(default=(1, "", "Error: 529 overloaded"))
    r = ce.transcribe([img()], "s", "p", max_retries=3)
    assert r["error"] is not None and "overloaded" in r["error"]
    assert len(f.calls) == 4 and len(f.sleeps) == 3
    assert f.sleeps[-1] >= f.sleeps[0] * 0.5  # grows (with jitter)


def test_no_retry_on_client_error(fake):
    f = fake(default=(1, _result_stream("Invalid image", is_error=True, subtype="error_during_execution",
                                        api_status=400), ""))
    r = ce.transcribe([img()], "s", "p", max_retries=5)
    assert r["error"] is not None and len(f.calls) == 1


def test_backoff_is_exponential_with_jitter():
    vals = [ce._backoff(a) for a in range(4) for _ in range(50)]
    assert all(0 < v <= 90 for v in vals)
    means = [sum(ce._backoff(a) for _ in range(200)) / 200 for a in range(4)]
    assert means[0] < means[1] < means[2] < means[3]


# ---------------------------------------------------------------- image bounds
def test_image_bound_assertion(fake):
    f = fake()
    with pytest.raises(ValueError):
        ce.transcribe([img(config.CLAUDE_MAX_EDGE + 1, 10)], "s", "p")
    side = int(config.CLAUDE_MAX_AREA ** 0.5) + 1
    with pytest.raises(ValueError):
        ce.transcribe([img(side, side)], "s", "p")
    assert not f.calls
    ce.transcribe([img(config.CLAUDE_MAX_EDGE, config.CLAUDE_MAX_AREA // config.CLAUDE_MAX_EDGE)], "s", "p")
    assert len(f.calls) == 1


def test_png_is_lossless_and_keeps_grayscale():
    import io
    import numpy as np
    a = (np.arange(64 * 32) % 251).astype("uint8").reshape(32, 64)
    im = Image.fromarray(a, "L")
    back = Image.open(io.BytesIO(ce.png_bytes(im)))
    assert back.mode == "L" and (np.asarray(back) == a).all()
    assert ce.png_bytes(im) == ce.png_bytes(im.copy())  # deterministic bytes -> stable cache key
    assert Image.open(io.BytesIO(ce.png_bytes(im.convert("P")))).mode == "RGB"


# ---------------------------------------------------------------- transcribe_many
def test_transcribe_many_order_and_meta(fake):
    def respond(cmd, stdin):
        prompt = json.loads(stdin)["message"]["content"][-1]["text"]
        i = int(prompt.split()[-1])
        time.sleep(0.002 * (10 - i))  # later jobs finish first
        return 0, _result_stream(f"out {i}"), ""
    fake(default=respond)
    jobs = [{"images": [img()], "system": "s", "prompt": f"job {i}", "meta": {"i": i, "tag": ["x", i]}}
            for i in range(10)]
    res = ce.transcribe_many(jobs, workers=4, progress=False)
    assert [r["text"] for r in res] == [f"out {i}" for i in range(10)]
    assert [r["meta"] for r in res] == [{"i": i, "tag": ["x", i]} for i in range(10)]
    assert all("meta" in j for j in jobs)  # caller's jobs not mutated


def test_transcribe_many_isolates_bad_jobs(fake, capsys):
    fake()
    jobs = [{"images": [img()], "system": "s", "prompt": "a", "meta": 1},
            {"images": [img(5000, 10)], "system": "s", "prompt": "b", "meta": 2},
            {"images": [img()], "system": "s", "prompt": "c"}]
    res = ce.transcribe_many(jobs, workers=2, progress=True)
    assert res[0]["error"] is None and res[2]["error"] is None and "meta" not in res[2]
    assert res[1]["error"].startswith("ValueError") and res[1]["meta"] == 2
    assert "3/3" in capsys.readouterr().err


def test_hires_bounds():
    from PIL import Image
    from vbpe import claude_eval as ce
    ce.check_image_bounds(Image.new("L", (1932, 1932)), hires=True)          # 69 x 69 = 4761 tokens
    with pytest.raises(ValueError):
        ce.check_image_bounds(Image.new("L", (1932, 1932)))                  # standard tier
    with pytest.raises(ValueError):
        ce.check_image_bounds(Image.new("L", (1960, 1960)), hires=True)      # 70 x 70 = 4900 tokens
    with pytest.raises(ValueError):
        ce.check_image_bounds(Image.new("L", (2604, 28)), hires=True)        # long edge
