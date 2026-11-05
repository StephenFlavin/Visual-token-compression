"""G2-lite diagnostic: decode held-out lines from a run's checkpoint and split its CER into substitutions,
insertions and deletions (plus hyp/ref length ratio); or split the stored samples of a finished row.

    python3 encoder_probe/experiments/g2lite_decode.py --arm direct --encoding rgb3_lg_pb_h5 --ckpt <path> [--pages 4]
    python3 encoder_probe/experiments/g2lite_decode.py --row encoder_probe/results/g2lite/runs.jsonl --arm frozen \
        --encoding rgb3_lg_pb_h5        # finished rows keep only 3 sample lines (their checkpoint is deleted)
Arms trained/direct only from checkpoints (their state holds the standardiser); frozen needs a fit (slow).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from encoder_probe.ep import token_cache as TC  # noqa: E402
from encoder_probe.ep.g2lite import Front, G2Reader, decode_pairs, edit_ops, load_merger, window_reverse  # noqa: E402
from encoder_probe.ep.probe_grid import GridProbe  # noqa: E402
from encoder_probe.ep.vlm_encoders import WEIGHTS, cpu_has_bf16  # noqa: E402


def show(pairs, n):
    for r, h in pairs[:n]:
        print(f"  ref ({len(r)}): {r[:110]!r}\n  hyp ({len(h)}): {h[:110]!r}")
    hs = [h for _, h in pairs]
    chars = "".join(hs)
    top = sorted({c: chars.count(c) for c in set(chars)}.items(), key=lambda x: -x[1])[:8]
    print(f"  distinct hyps {len(set(hs))}/{len(hs)}; empty {sum(not h.strip() for h in hs)}; "
          f"top hyp chars {top}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--row")
    ap.add_argument("--train-pages", type=int, default=500)
    ap.add_argument("--test-pages", type=int, default=16)
    ap.add_argument("--pages", type=int, default=16, help="test pages to decode")
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args(argv)
    if a.row:
        for l in Path(a.row).read_text().splitlines():
            r = json.loads(l)
            if r["encoding"] == a.encoding and r["arm"] == a.arm:
                print(f"[row] {a.arm} {a.encoding} cer={r['cer']:.4f} (3 stored samples only)")
                pairs = [tuple(x) for x in r["samples"]]
                show(pairs, a.show)
                print("  ops (samples):", {k: round(v, 4) if isinstance(v, float) else v
                                            for k, v in edit_ops(pairs).items()})
        return 0
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    enc, tr, te = TC.split_pages(a.encoding, 448, a.train_pages, a.test_pages, cache=True)
    del tr
    te = te[:a.pages]
    h = torch.load(TC.cache_path(TC.hidden_encoder("qwen25vl", 32), "clean", a.encoding, a.train_pages,
                                 a.test_pages), weights_only=False, mmap=True)["test"][:a.pages]
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    front = Front(a.arm, None if a.arm == "direct" else load_merger(WEIGHTS / "qwen25vl3b_visual.safetensors"),
                  window_reverse(448, 14, 2, 112), 1280)
    probe = GridProbe(enc.pitch, len(enc.classes), d=front.d, G=16, P=28)
    model = G2Reader(front, probe)
    model.load_state_dict(st["model"])          # weights + the standardiser buffers of this run
    bf16 = cpu_has_bf16()
    pairs = decode_pairs(model, h, te, enc, enc.classes, bf16)
    print(f"[ckpt] {a.arm} {a.encoding} step {st['step']} curve {st['curve']} pages {len(te)} lines {len(pairs)}")
    show(pairs, a.show)
    print("  ops:", {k: round(v, 4) if isinstance(v, float) else v for k, v in edit_ops(pairs).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
