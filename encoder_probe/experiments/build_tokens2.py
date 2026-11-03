"""Round 2: extract frozen-VLM tokens once per (encoder, cond, encoding) into encoder_probe/cache/tokens/.

    python3 encoder_probe/experiments/build_tokens2.py --encoders qwen25vl deepencoder --conds clean jpeg75

One process, one encoder in memory at a time, holding the global VLM lock for its whole run (see
ep/token_cache.py). Order: for each encoder, all encodings clean, then all jpeg75 (the probe queue's order).
Existing caches are skipped, so it is resumable.
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
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    lk = VLMLock()
    t0 = time.time()
    lk.acquire()
    print(f"[build] lock acquired after {time.time() - t0:.0f}s", flush=True)
    try:
        for name in a.encoders:
            todo = [(c, e) for c in a.conds for e in (a.encodings or encodings_for(name))
                    if not cache_path(name, c, e, a.train_pages, a.test_pages).exists()]
            if not todo:
                continue
            S, _ = GEOMS[name]
            t1 = time.time()
            model = load_vlm(name)
            print(f"[build] {name} loaded ({time.time() - t1:.0f}s); {len(todo)} caches to build", flush=True)
            for cond, enc in todo:
                _, tr, te = split_pages(enc, S, a.train_pages, a.test_pages)
                build(model, name, enc, cond, S, a.train_pages, a.test_pages, tr, te)
            del model
            gc.collect()
    finally:
        lk.release()
    print(f"[build] all done ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
