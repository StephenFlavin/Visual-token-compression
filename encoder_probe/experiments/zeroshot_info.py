"""Colour-gist A0b (docs/COLOUR_GIST.md): zero-shot information per vision token of k-stream pages.

The UNTRAINED Qwen2.5-VL-3B-Instruct is asked "Transcribe all the <colour> text on this page." for each stream of
each test page of rgb<k>s_p<p> (the A1 encodings). Teacher-forced loss of the true transcript with the page's own
image vs a blind control (another page's image, the same target) = information the model takes from the image
about that stream, with no training. Per arm: info per vision token = gap per text token x text tokens per vision
token; R(k) = arm k / the 1-stream arm, with a bootstrap CI (analyse_colour.ratio_ci). A zero-shot R(2) is a
starting point for A1, not its result: A1 trains the model to pick the stream it is asked about.

    python3 encoder_probe/experiments/zeroshot_info.py --pitches 16 --pages 12      # CPU bf16 (AMX) or GPU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder_probe.ep import qwen_ft as Q  # noqa: E402
from encoder_probe.ep import r3_tasks as T  # noqa: E402
from encoder_probe.ep.encoders import resolve_device  # noqa: E402
from encoder_probe.ep.geom import get_encoding_s, make_pages_s  # noqa: E402

S = 448


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pitches", type=int, nargs="+", default=[16])
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--variant", default="", help="stream layout variant, e.g. l = brightness-balanced (rgb2ls)")
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--max-target-chars", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run-name", default="z0_info")
    ap.add_argument("--tiny", action="store_true", help="random-init tiny model (CPU tests)")
    ap.add_argument("--out-root", default=str(ROOT / "encoder_probe" / "results"))
    ap.add_argument("--adapter", default=None, help="finetune_qwen.py adapter (.pt): measure the TRAINED model")
    ap.add_argument("--blind-ref", default=None, help="zero-shot runs.jsonl: per sample, blind loss = min(swap, "
                    "zero-shot blind) (conservative: never inflates the gap)")
    a = ap.parse_args(argv)
    dev = resolve_device(a.device)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    model, tok = Q.load_model(dev, tiny=a.tiny, bf16_cpu=dev.type == "cpu" and not a.tiny)
    if a.adapter:
        st = torch.load(a.adapter, map_location="cpu", weights_only=False)
        lo = st["lora"]
        model = Q.make_trainable(model, lo["r"], lo["alpha"], lo["dropout"], lo["vision_blocks"], lo["train_merger"])
        Q.load_trainable_state(model, st["trainable"])
        print(f"[z0i] adapter {Path(a.adapter).name} ({st['encoding']}, {st['steps']} steps)", flush=True)
    model.eval()
    zs = {}
    if a.blind_ref:
        for line in Path(a.blind_ref).read_text().splitlines():
            r = json.loads(line)
            zs[r["encoding"]] = r["val_per_sample_blind"]
    print(f"[z0i] model loaded in {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out_root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in a.pitches:
        for k in a.streams:
            enc = get_encoding_s(f"rgb{k}{a.variant if k > 1 else ''}s_p{p}", S)
            pages = make_pages_s(enc, "test", a.pages)
            cpt = sum(pg.n_chars for pg in pages) / (len(pages) * Q.image_tokens(S))
            per = {"real": [], "blind": []}
            per_stream = [[] for _ in range(k)]
            t1 = time.time()
            for i, pg in enumerate(pages):
                for s in range(k):
                    lines = pg.streams[s]
                    lo, hi = 0, len(lines)
                    if len("\n".join(lines)) > a.max_target_chars:
                        lo, hi = T.window_from(lines, 0, a.max_target_chars)
                    target = "\n".join(lines[lo:hi])
                    prompt = T.stream_transcribe_prompt(pg.colours[s], lo, hi, len(lines))
                    res = {}
                    for arm, img in (("real", pg.img), ("blind", pages[(i + 1) % len(pages)].img)):
                        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16,
                                                             enabled=dev.type == "cuda"):
                            loss, n = Q.lm_loss(model, Q.collate([{"imgs": [img], "prompt": prompt,
                                                                   "answer": target}], tok, dev))
                        res[arm] = (loss.item() * n, n)
                        per[arm].append(res[arm])
                    per_stream[s].append(res["blind"][0] - res["real"][0])
            if enc.name in zs:   # conservative blind: never above the untrained model's no-image loss
                ref = zs[enc.name]
                assert [n for _, n in ref] == [n for _, n in per["blind"]], "zero-shot targets differ"
                per["blind_swap"] = per["blind"]
                per["blind"] = [(min(b, z), n) for (b, n), (z, _) in zip(per["blind"], ref)]
            gap = np.array([b - r for (r, _), (b, _) in zip(per["real"], per["blind"])])
            ntok = np.array([n for _, n in per["real"]], dtype=float)
            row = {"phase": "z0_info", "encoding": enc.name, "pitch": p, "streams": k, "pages": len(pages),
                   "chars_per_token": cpt, "gap_per_text_token": float(gap.sum() / ntok.sum()),
                   "loss_real": float(sum(x for x, _ in per["real"]) / ntok.sum()),
                   "loss_blind": float(sum(x for x, _ in per["blind"]) / ntok.sum()),
                   "per_stream_gap_per_token": [float(np.sum(g) / (ntok.sum() / k)) for g in per_stream],
                   "colours": pages[0].colours, "val_per_sample_real": per["real"],
                   "val_per_sample_blind": per["blind"], "val_per_sample_blind_swap": per.get("blind_swap"),
                   "adapter": a.adapter, "device": str(dev), "time_s": time.time() - t1}
            rows.append(row)
            with open(out / "runs.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[z0i] {enc.name}: {cpt:.1f} chars/token, loss {row['loss_real']:.3f} vs blind "
                  f"{row['loss_blind']:.3f}, gap/token {row['gap_per_text_token']:.3f}, per stream "
                  f"{[round(x, 3) for x in row['per_stream_gap_per_token']]} ({row['time_s']:.0f}s)", flush=True)
    import analyse_colour as AC
    print("\ninfo per vision token vs 1 stream (95% bootstrap CI):")
    base_rows = rows
    if a.blind_ref and not any(r["streams"] == 1 for r in rows):   # reference = the zero-shot 1-stream arm
        base_rows = rows + [json.loads(x) for x in Path(a.blind_ref).read_text().splitlines()
                            if json.loads(x)["streams"] == 1]
    for p in a.pitches:
        base = next((r for r in base_rows if r["pitch"] == p and r["streams"] == 1), None)
        if base is None:
            continue

        def arr(r):
            return (np.array([b - x for (x, _), (b, _) in zip(r["val_per_sample_real"], r["val_per_sample_blind"])]),
                    np.array([n for _, n in r["val_per_sample_real"]], dtype=float))
        for r in rows:
            if r["pitch"] == p and r["streams"] > 1:
                R, lo, hi = AC.ratio_ci(arr(r), arr(base), r["chars_per_token"], base["chars_per_token"])
                print(f"  p{p} {r['streams']} streams: R = {R:.2f} ({lo:.2f}-{hi:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
