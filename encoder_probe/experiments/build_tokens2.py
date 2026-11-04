"""Round 2: extract frozen-VLM tokens once per (encoder, cond, encoding) into encoder_probe/cache/tokens/.

    python3 encoder_probe/experiments/build_tokens2.py --encoders qwen25vl deepencoder --conds clean jpeg75

One process, one encoder in memory at a time, holding the global VLM lock for its whole run (see
ep/token_cache.py). Order: for each encoder, all encodings clean, then all jpeg75 (the probe queue's order).
Existing caches are skipped, and each cache is built in chunks of 50 pages that survive a kill
(ep/token_cache.py::build), so it is resumable within a cache too. --split K (qwen25vl, R3-0b) also caches the
hidden states entering ViT block K (cache name qwen25vl_hK). --no-global-lock: only the per-cache lock, so
several builders (different encodings) can run at once (each holds its own ViT copy: ~3 GB in fp32). --device auto (default) = cuda if available (bf16, larger
batches), else CPU. With EP_PAGE_CACHE=1 the rendered pages are cached for the probe jobs (ep/token_cache.py).
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from encoder_probe.ep.encoders import resolve_device  # noqa: E402
from encoder_probe.ep.token_cache import VLMLock, build, cache_path, split_pages  # noqa: E402
from encoder_probe.ep.vlm_encoders import GEOMS, encodings_for, load_vlm  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoders", nargs="+", default=["qwen25vl", "deepencoder"])
    ap.add_argument("--conds", nargs="+", default=["clean", "jpeg75"])
    ap.add_argument("--encodings", nargs="+", help="default: encodings_for(encoder)")
    ap.add_argument("--train-pages", type=int, default=200)
    ap.add_argument("--test-pages", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--split", type=int, help="qwen25vl: also cache the hidden states entering this ViT block")
    ap.add_argument("--no-global-lock", action="store_true", help="skip the global VLM lock (per-cache lock only)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto = cuda if available")
    a = ap.parse_args(argv)
    dev = resolve_device(a.device)
    torch.set_num_threads(a.threads)
    from encoder_probe.ep.token_cache import hidden_encoder
    lk = None if a.no_global_lock else VLMLock()
    t0 = time.time()
    if lk is not None:
        lk.acquire()
        print(f"[build] lock acquired after {time.time() - t0:.0f}s", flush=True)
    try:
        for name in a.encoders:
            todo = [(c, e) for c in a.conds for e in (a.encodings or encodings_for(name))
                    if not cache_path(name, c, e, a.train_pages, a.test_pages).exists()
                    or (a.split is not None and not cache_path(hidden_encoder(name, a.split), c, e, a.train_pages,
                                                               a.test_pages).exists())]
            if not todo:
                continue
            S, _ = GEOMS[name]
            t1 = time.time()
            model = load_vlm(name, device=dev)
            print(f"[build] {name} loaded on {dev.type} ({time.time() - t1:.0f}s); {len(todo)} caches to build",
                  flush=True)
            for cond, enc in todo:
                _, tr, te = split_pages(enc, S, a.train_pages, a.test_pages)
                build(model, name, enc, cond, S, a.train_pages, a.test_pages, tr, te, split=a.split)
                del tr, te
                gc.collect()
            del model
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if lk is not None:
            lk.release()
    print(f"[build] all done ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
