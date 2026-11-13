"""Phase 1b: zero-shot sweep with a pretrained CJK+Latin OCR engine (RapidOCR / PP-OCR, CPU).

Why: Claude refuses or safety-blocks substitution / quadrant / scrambled pages (see
results/phase1/pilot), so it cannot score composite schemes. PP-OCR already knows both Latin and
~6k CJK glyphs and has no safety layer, so it answers "are these glyphs legible at this size for a
reader that already knows them?". Caveats: its training is Chinese-heavy (biased towards CJK);
it drops inter-word spaces, so ALL schemes are scored with whitespace-free CER (cer_nospace).
Recognition-only on exact line strips (no detection), see vtc/readers/rapidocr_reader.py.

Rows use the Phase-1 schema so ``python3 -m vtc.analysis`` works on the output.

    python3 experiments/phases/phase1b_ocr.py --variant natural --n-passages 6
    python3 experiments/phases/phase1b_ocr.py --variant scrambled --n-passages 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vtc import config  # noqa: E402
from vtc.metrics import cer, normalize  # noqa: E402

DEFAULT_SCHEMES = ["latin", "latin_condensed", "subst300", "subst1000", "subst3000"]
DEFAULT_BUDGETS = [12, 14, 16, 18, 20, 23, 26, 30, 34, 40, 48, 60, 90]


def _job(args):
    passage, variant, scheme_name, budget, run, img_dir = args
    from experiments.phases.phase1_sweep import canvas_size
    from vtc.layout import render_page
    from vtc.readers.rapidocr_reader import cer_nospace, read_lines
    from vtc.schemes import get_scheme

    scheme = get_scheme(scheme_name)
    text = passage["text"]
    n = len(normalize(text))
    W, H = canvas_size(budget, n, patch=4)
    img, info = render_page(scheme.encode(text), W, H)
    row = dict(run=run, phase=1, model="rapidocr-ppocr-v3", effort=None, variant=variant,
               passage_id=passage["id"], scheme=scheme_name, budget=budget, n_chars=n, W=W, H=H,
               px_per_char=W * H / n, em=info["em"], n_lines=info["n_lines"], fit=info["fit"],
               skipped=not info["fit"], error=None, refusal=False)
    if not info["fit"]:
        row.update(cer=1.0, cer_raw=1.0, symbol_cer=1.0, output_text="")
        return row
    t0 = time.time()
    out = read_lines(img, info["em"], info["n_lines"])
    decoded = scheme.decode(out)
    row.update(latency_s=time.time() - t0, output_text=out,
               cer=cer_nospace(text, decoded),                  # primary metric for this reader
               cer_raw=cer(text, decoded),
               symbol_cer=cer_nospace(scheme.transcription_target(text), out))
    if img_dir:
        p = Path(img_dir) / scheme_name / f"{passage['id']}_{budget:g}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schemes", nargs="+", default=DEFAULT_SCHEMES)
    ap.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    ap.add_argument("--variant", default="natural", choices=["natural", "scrambled"])
    ap.add_argument("--n-passages", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-images", action="store_true")
    a = ap.parse_args(argv)

    from vtc.corpus import load_passages
    run = a.run_name or f"ocr_{a.variant}"
    out_dir = config.RESULTS / "phase1b" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    done = set()
    if runs_path.exists():
        for line in open(runs_path):
            r = json.loads(line)
            done.add((r["passage_id"], r["scheme"], float(r["budget"])))
    passages = load_passages(a.variant)[: a.n_passages]
    jobs = [(p, a.variant, s, b, run, str(out_dir / "images") if a.save_images else None)
            for p in passages for s in a.schemes for b in a.budgets
            if (p["id"], s, float(b)) not in done]
    print(f"run={run} jobs={len(jobs)} (done {len(done)}) workers={a.workers}", flush=True)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    t0 = time.time()
    with Pool(a.workers) as pool, open(runs_path, "a") as f:
        for i, row in enumerate(pool.imap_unordered(_job, jobs), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
