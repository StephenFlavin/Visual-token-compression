"""Frozen-VLM token cache: features are extracted ONCE per (encoder, cond, encoding, page counts) and saved.

    encoder_probe/cache/tokens/<encoder>/<cond>/<encoding>_tr<n>_te<m>.pt
        {"train": bf16 (n, G*G, d), "test": bf16 (m', G*G, d), "digest_train", "digest_test", ...}

Only the holder of one global lock (cache/tokens/.vlm.lock, fcntl, released by the kernel if the holder dies)
ever loads a VLM encoder, so two encoders are never in memory at once:
  * the builder (experiments/build_tokens2.py) holds it for its whole run and writes caches in queue order;
  * a probe job whose cache is missing polls: when the lock is free (no builder alive) it takes the lock and
    builds just its own cache; while the builder holds it, it waits for the file.
Files are written to a temp name and renamed, so a reader never sees a partial cache.

Opt-in page cache (EP_PAGE_CACHE=1, or ``split_pages(..., cache=True)``): rendered pages are pickled to
    encoder_probe/cache/pages/<encoding>_S<S>_tr<n>_te<m>.pkl
so that jobs sharing an (encoding, page size, page counts) render them once (rendering is single-threaded CPU work,
~0.3-0.8 s/page). Off by default. Delete the directory after changing any renderer: the file is not re-validated.
"""
from __future__ import annotations

import fcntl
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from encoder_probe.ep.encoders import degrade
from encoder_probe.ep.geom import get_encoding_s, make_pages_s, pages_digest

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "encoder_probe" / "cache" / "tokens"
LOCK = CACHE / ".vlm.lock"
PAGES = ROOT / "encoder_probe" / "cache" / "pages"


def cache_path(encoder: str, cond: str, encoding: str, n_train: int, n_test: int) -> Path:
    return CACHE / encoder / cond / f"{encoding}_tr{n_train}_te{n_test}.pt"


def page_cache_on() -> bool:
    return os.environ.get("EP_PAGE_CACHE", "") not in ("", "0")


def page_cache_path(encoding: str, S: int, n_train: int, n_test: int) -> Path:
    return PAGES / f"{encoding}_S{S}_tr{n_train}_te{n_test}.pkl"


def split_pages(encoding: str, S: int, n_train: int, n_test: int, cache: bool | None = None):
    enc = get_encoding_s(encoding, S)
    cache = page_cache_on() if cache is None else cache
    path = page_cache_path(encoding, S, n_train, n_test)
    if cache and path.exists():
        with open(path, "rb") as f:
            tr, te = pickle.load(f)
        return enc, tr, te
    tr, te = make_pages_s(enc, "train", n_train), make_pages_s(enc, "test", n_test, allow_fewer=True)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            pickle.dump((tr, te), f, protocol=5)
        os.replace(tmp, path)
    return enc, tr, te


def extract(model, pages, cond: str) -> torch.Tensor:
    return model(np.stack([degrade(p.img, cond) for p in pages]))


def build(model, encoder: str, encoding: str, cond: str, S: int, n_train: int, n_test: int,
          tr=None, te=None, log=print) -> Path:
    path = cache_path(encoder, cond, encoding, n_train, n_test)
    if tr is None:
        _, tr, te = split_pages(encoding, S, n_train, n_test)
    t0 = time.time()
    d = {"train": extract(model, tr, cond), "test": extract(model, te, cond),
         "digest_train": pages_digest(tr), "digest_test": pages_digest(te), "encoder": encoder, "cond": cond,
         "encoding": encoding, "n_test_pages": len(te), "extract_s": None,
         "enc_compute": getattr(model, "compute", None)}
    d["extract_s"] = time.time() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(d, tmp)
    os.replace(tmp, path)
    per = d["extract_s"] / (len(tr) + len(te))
    log(f"[cache] wrote {path.relative_to(ROOT)} train {tuple(d['train'].shape)} test {tuple(d['test'].shape)} "
        f"({d['extract_s']:.0f}s, {per:.2f} s/page)", flush=True)
    return path


class VLMLock:
    def __init__(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.f = open(LOCK, "a")

    def try_acquire(self) -> bool:
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def acquire(self) -> None:
        fcntl.flock(self.f, fcntl.LOCK_EX)

    def release(self) -> None:
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _load(path: Path, tr, te) -> dict:
    d = torch.load(path, weights_only=False)
    if d["digest_train"] != pages_digest(tr) or d["digest_test"] != pages_digest(te):
        raise RuntimeError(f"{path}: cached tokens were made from different pages")
    return d


def get_tokens(encoder: str, encoding: str, cond: str, S: int, n_train: int, n_test: int, tr, te,
               poll: float = 30.0, log=print, device="cpu") -> tuple[dict, float]:
    """-> (cache dict, seconds waited). Waits for the builder or builds under the global lock (on ``device``)."""
    from encoder_probe.ep.vlm_encoders import load_vlm
    path = cache_path(encoder, cond, encoding, n_train, n_test)
    t0, last = time.time(), 0.0
    while not path.exists():
        lk = VLMLock()
        if lk.try_acquire():
            try:
                if not path.exists():
                    log(f"[cache] no builder holds the lock: building {path.name} here", flush=True)
                    build(load_vlm(encoder, device=device), encoder, encoding, cond, S, n_train, n_test, tr, te,
                          log=log)
            finally:
                lk.release()
            break
        lk.f.close()
        if time.time() - last > 600:
            log(f"[cache] waiting for {path.relative_to(ROOT)} ({time.time() - t0:.0f}s)", flush=True)
            last = time.time()
        time.sleep(poll)
    return _load(path, tr, te), time.time() - t0
