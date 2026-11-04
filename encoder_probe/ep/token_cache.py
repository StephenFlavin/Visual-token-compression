"""Frozen-VLM token cache: features are extracted ONCE per (encoder, cond, encoding, page counts) and saved.

    encoder_probe/cache/tokens/<encoder>/<cond>/<encoding>_tr<n>_te<m>.pt
        {"train": bf16 (n, G*G, d), "test": bf16 (m', G*G, d), "digest_train", "digest_test", ...}

Only the holder of one global lock (cache/tokens/.vlm.lock, ep/flock.py, released by the kernel if the holder dies)
ever loads a VLM encoder, so two encoders are never in memory at once:
  * the builder (experiments/build_tokens2.py) holds it for its whole run and writes caches in queue order;
  * a probe job whose cache is missing polls: when the lock is free (no builder alive) it takes the lock and
    builds just its own cache; while the builder holds it, it waits for the file.
Files are written to a temp name and renamed, so a reader never sees a partial cache. Builds are resumable in
chunks of 50 pages (``build``: a killed build continues at the first missing chunk); a per-cache lock
(``<cache>.lock``) lets several builders work on different caches at once (build_tokens2.py --no-global-lock).

Opt-in page cache (EP_PAGE_CACHE=1, or ``split_pages(..., cache=True)``): rendered pages are pickled to
    encoder_probe/cache/pages/<encoding>_S<S>_tr<n>_te<m>.pkl
so that jobs sharing an (encoding, page size, page counts) render them once (rendering is single-threaded CPU work,
~0.3-0.8 s/page). Off by default. Delete the directory after changing any renderer: the file is not re-validated.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from encoder_probe.ep import flock
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


def extract(model, pages, cond: str, **kw):
    return model(np.stack([degrade(p.img, cond) for p in pages]), **kw)


def hidden_encoder(encoder: str, split: int) -> str:
    """Cache name of the hidden states entering block ``split`` (R3-0b), e.g. qwen25vl_h28."""
    return f"{encoder}_h{split}"


CHUNK = int(os.environ.get("EP_BUILD_CHUNK", "50"))


def _save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build(model, encoder: str, encoding: str, cond: str, S: int, n_train: int, n_test: int,
          tr=None, te=None, log=print, chunk: int | None = None, split: int | None = None) -> Path:
    """Extract and write the cache; resumable in chunks of ``chunk`` pages (default 50, EP_BUILD_CHUNK).

    Each chunk is saved to ``<cache>.parts/<out>_<split>_<first page>.pt`` with the digest of its pages, so a
    killed build continues at the first missing chunk (a chunk whose pages changed is recomputed). When every chunk
    is there the parts are concatenated into the usual cache file (same format and keys as before) and removed.
    ``split`` = k (Qwen only, R3-0b) also writes the hidden states entering block k to the cache of
    ``hidden_encoder(encoder, k)``. A per-cache lock (``<cache>.lock``) keeps two builders off one cache: the
    second waits for the first to finish."""
    path = cache_path(encoder, cond, encoding, n_train, n_test)
    if tr is None:
        _, tr, te = split_pages(encoding, S, n_train, n_test)
    chunk = chunk or CHUNK
    outs = {"tok": path}
    if split is not None:
        outs[f"h{split}"] = cache_path(hidden_encoder(encoder, split), cond, encoding, n_train, n_test)
    if all(p.exists() for p in outs.values()):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    lk = open(path.with_name(path.name + ".lock"), "a")
    if not flock.try_lock(lk):
        log(f"[cache] another builder holds {path.name}.lock: waiting", flush=True)
        flock.lock(lk)
    try:
        if all(p.exists() for p in outs.values()):
            return path
        parts = path.with_name(path.name + ".parts")
        parts.mkdir(exist_ok=True)
        jobs = [(sp, i, pages[i:i + chunk]) for sp, pages in (("train", tr), ("test", te))
                for i in range(0, len(pages), chunk)]
        t_ex, done, t0 = 0.0, 0, time.time()
        for sp, i, pg in jobs:
            files = {o: parts / f"{o}_{sp}_{i:06d}.pt" for o in outs}
            dg = pages_digest(pg)
            if all(f.exists() for f in files.values()):
                old = [torch.load(f, weights_only=False, mmap=True) for f in files.values()]
                if all(c["digest"] == dg for c in old):
                    t_ex += old[0]["extract_s"]
                    continue
            t1 = time.time()
            r = extract(model, pg, cond, **({} if split is None else {"split": split}))
            r = r if isinstance(r, dict) else {"tok": r}
            dt = time.time() - t1
            t_ex += dt
            for o, f in files.items():
                _save({"x": r[o], "digest": dg, "extract_s": dt}, f)
            done += len(pg)
            log(f"[cache] {path.name}: {sp} pages {i}-{i + len(pg)} ({dt / len(pg):.2f} s/page)", flush=True)
        for o, out in outs.items():
            d = {"digest_train": pages_digest(tr), "digest_test": pages_digest(te),
                 "encoder": encoder if o == "tok" else hidden_encoder(encoder, split), "cond": cond,
                 "encoding": encoding, "n_test_pages": len(te), "extract_s": t_ex,
                 "enc_compute": getattr(model, "compute", None)}
            if o != "tok":
                d.update(split_block=split, order="window")
            for sp in ("train", "test"):
                d[sp] = torch.cat([torch.load(parts / f"{o}_{sp}_{i:06d}.pt", weights_only=False, mmap=True)["x"]
                                   for s2, i, _ in jobs if s2 == sp])
            d = {k: d[k] for k in ("train", "test", *[k for k in d if k not in ("train", "test")])}
            out.parent.mkdir(parents=True, exist_ok=True)
            _save(d, out)
            log(f"[cache] wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} train "
                f"{tuple(d['train'].shape)} test {tuple(d['test'].shape)} ({t_ex:.0f}s extract, "
                f"{t_ex / (len(tr) + len(te)):.2f} s/page, {done} pages this run, {time.time() - t0:.0f}s)",
                flush=True)
            del d
        for f in parts.iterdir():
            f.unlink()
        parts.rmdir()
    finally:
        flock.unlock(lk)
        lk.close()
    return path


def build_locked(path: Path) -> bool:
    """True if a builder currently holds the per-cache lock of ``path``."""
    lp = path.with_name(path.name + ".lock")
    if not lp.exists():
        return False
    with open(lp, "a") as f:
        if flock.try_lock(f):
            flock.unlock(f)
            return False
    return True


class VLMLock:
    def __init__(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.f = open(LOCK, "a")

    def try_acquire(self) -> bool:
        return flock.try_lock(self.f)

    def acquire(self) -> None:
        flock.lock(self.f)

    def release(self) -> None:
        flock.unlock(self.f)
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
        lk = None if build_locked(path) else VLMLock()   # a (parallel) builder is on this cache: wait for it
        if lk is not None and lk.try_acquire():
            try:
                if not path.exists():
                    log(f"[cache] no builder holds the lock: building {path.name} here", flush=True)
                    build(load_vlm(encoder, device=device), encoder, encoding, cond, S, n_train, n_test, tr, te,
                          log=log)
            finally:
                lk.release()
            break
        if lk is not None:
            lk.f.close()
        if time.time() - last > 600:
            log(f"[cache] waiting for {path.relative_to(ROOT)} ({time.time() - t0:.0f}s)", flush=True)
            last = time.time()
        time.sleep(poll)
    return _load(path, tr, te), time.time() - t0
