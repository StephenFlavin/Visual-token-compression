"""Fetch only some tensors of a safetensors checkpoint on the HF Hub (HTTP range requests).

A VLM checkpoint shard holds the vision tower and much of the language model. Reading the safetensors header
(8-byte length + JSON) with a range request gives every tensor's byte span, so only the spans of the wanted
tensors are downloaded (merged into contiguous runs) and written to a new local safetensors file.

    fetch_tensors("Qwen/Qwen2.5-VL-3B-Instruct", "model-00001-of-00002.safetensors",
                  lambda k: k.startswith("visual."), out_path, revision=...)
"""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from typing import Callable

import requests

CHUNK = 1 << 24


def _url(repo: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _get(url: str, start: int, end: int, tries: int = 6) -> requests.Response:
    """GET bytes [start, end] (inclusive), streaming; retries with backoff."""
    for k in range(tries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=120)
            if r.status_code == 206:
                return r
            raise IOError(f"HTTP {r.status_code} for range request")
        except (requests.RequestException, IOError):
            if k == tries - 1:
                raise
            time.sleep(2 ** (k + 1))
    raise AssertionError


def read_header(repo: str, filename: str, revision: str = "main") -> tuple[dict, int]:
    """-> (header dict without __metadata__, byte offset where tensor data starts)."""
    url = _url(repo, filename, revision)
    n = struct.unpack("<Q", _get(url, 0, 7).content)[0]
    h = json.loads(_get(url, 8, 8 + n - 1).content)
    h.pop("__metadata__", None)
    return h, 8 + n


def fetch_tensors(repo: str, filename: str, keep: Callable[[str], bool], out: Path,
                  revision: str = "main", rename: Callable[[str], str] = lambda k: k) -> Path:
    """Download the tensors whose name passes ``keep`` into the safetensors file ``out`` (skipped if it exists)."""
    out = Path(out)
    if out.exists():
        return out
    url = _url(repo, filename, revision)
    h, base = read_header(repo, filename, revision)
    items = sorted(((k, v) for k, v in h.items() if keep(k)), key=lambda kv: kv[1]["data_offsets"][0])
    if not items:
        raise KeyError(f"no tensors selected in {repo}/{filename}")
    runs: list[list[int]] = []                       # merged [start, end) spans in the source data section
    for _, v in items:
        a, b = v["data_offsets"]
        if runs and runs[-1][1] == a:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    new_h, pos = {}, 0
    for k, v in items:
        a, b = v["data_offsets"]
        new_h[rename(k)] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [pos, pos + b - a]}
        pos += b - a
    hb = json.dumps(new_h).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = out.with_suffix(".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(b - a for a, b in runs)
    done = 0
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        for a, b in runs:
            got = a
            while got < b:                            # resume a run after a dropped connection
                try:
                    r = _get(url, base + got, base + b - 1)
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        got += len(chunk)
                        done += len(chunk)
                except requests.RequestException:
                    time.sleep(4)
            assert got == b, (got, b)
            print(f"[hf_partial] {done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
    os.replace(tmp, out)
    return out
