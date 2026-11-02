"""Zero-shot Claude OCR client (WP-E1).

Claude is called through the headless Claude Code CLI (``claude -p``) with stream-json input so
that images can be passed as base64 content blocks. Every call is a single, isolated turn:

  * no tools, no MCP servers, no slash commands/skills, no settings files, no session saved;
  * the subprocess runs in an empty temp dir (no CLAUDE.md / project context);
  * the environment is a small whitelist, so variables of a *parent* Claude Code session
    (session id, remote-session URLs, effort, ...) do not leak in. Inheriting them made the
    child attach to the parent session, added ~800 prompt tokens and triggered an auxiliary
    haiku call; see docs/claude-eval-notes.md.

Public API (PLAN.md section 5)::

    transcribe(images, system, prompt, model=DEFAULT_MODEL, max_retries=5) -> dict
    transcribe_many(jobs, workers=8, progress=True) -> list[dict]

Results are cached on disk in ``results/cache/claude/<sha256>.json`` keyed by
(model, system, prompt, PNG bytes of each image). Errors are never cached.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from vbpe import config

DEFAULT_MODEL = os.environ.get("VBPE_MODEL", "claude-opus-5-5")
# Optional ``--effort`` for the CLI (low/medium/high/xhigh/max). None = CLI default.
DEFAULT_EFFORT = os.environ.get("VBPE_EFFORT") or None
CACHE_DIR = config.RESULTS / "cache" / "claude"
CLAUDE_BIN = os.environ.get("VBPE_CLAUDE_BIN", "claude")
DEFAULT_TIMEOUT = 300.0

# Environment passed to the CLI: only what it needs to find binaries, auth and the network.
_ENV_WHITELIST = (
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "NODE_OPTIONS", "REQUESTS_CA_BUNDLE",
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME",
)
_ENV_FIXED = {
    "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",            # no server-side advisor tool
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",    # no telemetry / auxiliary requests
    "DISABLE_AUTOUPDATER": "1",
    # Default CLI caching writes the whole prompt (incl. the unique image) at the 1-hour TTL
    # rate (2x input price). 5-minute TTL = 1.25x and still lets shared prefixes (system prompt,
    # legend images) be read at 0.1x by concurrent calls. Measured in check_claude_vision.py.
    "FORCE_PROMPT_CACHING_5M": "1",
}

_RETRYABLE = re.compile(
    r"rate.?limit|overload|\b429\b|\b5\d\d\b|timeout|timed out|ECONNRESET|socket hang up|"
    r"network|temporarily|try again|capacity|internal server error|api_error",
    re.I,
)


# --------------------------------------------------------------------------- helpers
def png_bytes(img: Image.Image) -> bytes:
    """Lossless, deterministic PNG encoding. 'L' (grayscale) is accepted by the API as-is."""
    if img.mode not in ("L", "RGB", "RGBA", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def check_image_bounds(img: Image.Image) -> None:
    """Raise ValueError if the API would downsample this image (long edge / area limits)."""
    W, H = img.size
    if max(W, H) > config.CLAUDE_MAX_EDGE or W * H > config.CLAUDE_MAX_AREA:
        raise ValueError(
            f"image {W}x{H} exceeds Claude no-resize limits "
            f"(edge<={config.CLAUDE_MAX_EDGE}, area<={config.CLAUDE_MAX_AREA}); it would be downsampled"
        )


def cache_key(model: str, system: str, prompt: str, pngs: list[bytes], effort: str | None = None) -> str:
    h = hashlib.sha256()
    payload = {"model": model, "system": system, "prompt": prompt,
               "images": [hashlib.sha256(b).hexdigest() for b in pngs]}
    if effort:
        payload["effort"] = effort
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()


def build_message(pngs: list[bytes], prompt: str) -> bytes:
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(b).decode("ascii")}}
               for b in pngs]
    content.append({"type": "text", "text": prompt})
    msg = {"type": "user", "message": {"role": "user", "content": content}}
    return (json.dumps(msg) + "\n").encode()


def build_command(system: str, model: str, effort: str | None = None) -> list[str]:
    cmd = [CLAUDE_BIN, "-p",
           "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", model,
           "--system-prompt", system,
           "--tools", "",
           "--max-turns", "1",
           "--strict-mcp-config",
           "--setting-sources", "",
           "--disable-slash-commands",
           "--no-session-persistence"]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def build_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in _ENV_WHITELIST if k in os.environ}
    env.update(_ENV_FIXED)
    return env


def _run_cli(cmd: list[str], stdin: bytes, env: dict[str, str], timeout: float) -> tuple[int, str, str]:
    """Run one CLI call in a fresh empty directory. Seam for tests (monkeypatch this)."""
    with tempfile.TemporaryDirectory(prefix="vbpe_claude_") as cwd:
        try:
            p = subprocess.run(cmd, input=stdin, capture_output=True, env=env, cwd=cwd, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode("utf-8", "replace")
            return -9, out, f"timeout after {timeout:.0f}s"
        except FileNotFoundError as e:
            return 127, "", f"claude CLI not found: {e}"
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def parse_stream(stdout: str) -> tuple[dict | None, str | None]:
    """Return (final result event, model named in assistant events).

    The last ``rate_limit_event`` (subscription quota status) is attached to the result as
    ``result["_rate_limit"]``."""
    result, asst_model, rate = None, None, None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "result":
            result = ev
        elif t == "assistant":
            asst_model = (ev.get("message") or {}).get("model") or asst_model
        elif t == "rate_limit_event":
            rate = ev.get("rate_limit_info")
    if result is not None and rate:
        result["_rate_limit"] = rate
    return result, asst_model


def _served_model(res: dict, asst_model: str | None, requested: str) -> str:
    mu = res.get("modelUsage") or {}
    if mu:  # the model that produced the most output is the one that answered
        return max(mu, key=lambda m: (mu[m].get("outputTokens", 0), mu[m].get("costUSD", 0)))
    return asst_model or requested


def _record(res: dict, asst_model: str | None, model: str, latency: float, attempts: int) -> dict:
    u = res.get("usage") or {}
    in_tok = (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
              + u.get("cache_read_input_tokens", 0))
    mu = res.get("modelUsage") or {}
    return {
        "text": res.get("result") or "",
        "model": _served_model(res, asst_model, model),
        "input_tokens": in_tok,
        "output_tokens": u.get("output_tokens", 0),
        "cost_usd": float(res.get("total_cost_usd") or 0.0),
        "latency_s": round(latency, 3),
        "cached": False,
        "error": None,
        "requested_model": model,
        "usage": {k: u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                           "cache_read_input_tokens", "output_tokens")},
        "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
        "model_usage": {m: {"in": v.get("inputTokens", 0) + v.get("cacheCreationInputTokens", 0)
                            + v.get("cacheReadInputTokens", 0),
                            "out": v.get("outputTokens", 0), "cost": v.get("costUSD", 0.0)}
                        for m, v in mu.items()},
        "stop_reason": res.get("stop_reason"),
        "attempts": attempts,
        "rate_limit": _rate_summary(res.get("_rate_limit")),
    }


def _rate_summary(info: dict | None) -> dict | None:
    if not info:
        return None
    win = info.get("unifiedWindows") or {}
    return {"status": info.get("status"),
            **{k: v.get("utilization") for k, v in win.items() if isinstance(v, dict)}}


def _backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    return min(cap, base * 2 ** attempt) * (0.5 + random.random())


_sleep = time.sleep  # patched in tests


# --------------------------------------------------------------------------- public API
def transcribe(images: list[Image.Image], system: str, prompt: str, model: str = DEFAULT_MODEL,
               max_retries: int = 5, *, timeout: float = DEFAULT_TIMEOUT, use_cache: bool = True,
               effort: str | None = DEFAULT_EFFORT, cache_dir: Path | None = None) -> dict:
    """Send ``images`` (in order) followed by ``prompt`` to Claude; return text + accounting.

    Keys: text, model (served), input_tokens (incl. cache creation/read), output_tokens,
    cost_usd, latency_s, cached, error (None or str) + diagnostics (usage, model_usage, ...).
    ``use_cache=False`` bypasses the read but still stores a successful result.
    """
    for img in images:
        check_image_bounds(img)
    pngs = [png_bytes(img) for img in images]
    key = cache_key(model, system, prompt, pngs, effort)
    cdir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    path = cdir / f"{key}.json"
    if use_cache and path.exists():
        try:
            rec = json.loads(path.read_text())
            rec.update(cached=True, latency_s=0.0)
            return rec
        except (OSError, json.JSONDecodeError):
            pass

    cmd = build_command(system, model, effort)
    stdin = build_message(pngs, prompt)
    env = build_env()
    last_err = "no attempt made"
    t_all = time.monotonic()
    for attempt in range(max_retries + 1):
        if attempt:
            _sleep(_backoff(attempt - 1))
        t0 = time.monotonic()
        rc, out, err = _run_cli(cmd, stdin, env, timeout)
        latency = time.monotonic() - t0
        res, asst_model = parse_stream(out)
        if res is not None and not res.get("is_error") and res.get("subtype") == "success" and rc == 0:
            rec = _record(res, asst_model, model, latency, attempt + 1)
            rec["cache_key"] = key
            cdir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp{os.getpid()}_{threading.get_ident()}")
            tmp.write_text(json.dumps(rec, indent=1))
            os.replace(tmp, path)
            return rec
        # ---- failure: decide whether to retry
        detail = ""
        if res is not None:
            detail = f"{res.get('subtype')} api_status={res.get('api_error_status')} {str(res.get('result'))[:300]}"
        last_err = f"rc={rc} {detail} {err.strip()[-500:]}".strip()
        status = res.get("api_error_status") if res else None
        fatal = (res is not None and isinstance(status, int) and 400 <= status < 500 and status != 429
                 and not _RETRYABLE.search(detail))
        if fatal:
            break
    return {"text": "", "model": model, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "latency_s": round(time.monotonic() - t_all, 3), "cached": False, "error": last_err,
            "requested_model": model, "attempts": attempt + 1}


def transcribe_many(jobs: list[dict], workers: int = 8, progress: bool = True) -> list[dict]:
    """Run many ``transcribe`` jobs concurrently. Each job = transcribe kwargs (+ optional "meta",
    returned untouched in the result). Results are in input order."""
    n = len(jobs)
    results: list[dict | None] = [None] * n
    lock = threading.Lock()
    stats = {"done": 0, "cost": 0.0, "cached": 0, "errors": 0}
    t0 = time.monotonic()

    def run(i: int) -> None:
        job = dict(jobs[i])
        meta = job.pop("meta", None)
        try:
            rec = transcribe(**job)
        except Exception as e:  # bad job (e.g. oversize image) must not kill the pool
            rec = {"text": "", "model": job.get("model", DEFAULT_MODEL), "input_tokens": 0,
                   "output_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0, "cached": False,
                   "error": f"{type(e).__name__}: {e}"}
        if meta is not None:
            rec["meta"] = meta
        results[i] = rec
        with lock:
            stats["done"] += 1
            stats["cost"] += 0.0 if rec.get("cached") else rec.get("cost_usd", 0.0)
            stats["cached"] += bool(rec.get("cached"))
            stats["errors"] += rec.get("error") is not None
            if progress:
                print(f"[claude] {stats['done']}/{n}  new cost ${stats['cost']:.4f}  "
                      f"cached {stats['cached']}  errors {stats['errors']}  "
                      f"{time.monotonic() - t0:.0f}s", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(run, range(n)))
    return results  # type: ignore[return-value]
