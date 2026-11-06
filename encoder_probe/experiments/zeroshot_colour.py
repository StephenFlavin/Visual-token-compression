"""Colour-channel Latin, zero-shot (step Z0 of docs/COLOUR_GIST.md): can pretrained Qwen2.5-VL-3B read two passages
overlaid in different colour channels, with no training at all?

Pages are 448 x 448 (256 vision tokens), DejaVu Sans Mono, word-wrapped, large pitch (default 16 px). Conditions:
* mono      one passage, black on white (the reference);
* solo      one passage alone in a stream colour (colour alone, no overlap);
* rgb2_light  passage A lights the R channel, passage B the G channel, on black: red and green text, overlaps yellow;
* rgb2_dark   passage A darkens G+B (red ink), passage B darkens R (cyan ink), on white; overlaps black.
Each stream owns its own channel(s), so each passage is fully recoverable from the pixels.
Per page and stream: transcription CER ("Transcribe only the red text ..."), and a 4-way MC question ("Which phrase
appears in the red text?") whose distractors include a phrase from the OTHER stream on the same page.

    python3 encoder_probe/experiments/zeroshot_colour.py --pages 6 --transcribe-pages 3     # CPU bf16 (AMX) or GPU
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from encoder_probe.ep import qwen_ft as Q  # noqa: E402
from encoder_probe.ep.encoders import resolve_device  # noqa: E402
from vbpe import config  # noqa: E402

S = 448
LETTERS = "ABCD"


def passages(n: int, min_chars: int = 1200) -> list[str]:
    paras = [p.strip() for p in (ROOT / "data" / "test.txt").read_text(encoding="utf-8").splitlines()
             if len(p.strip()) >= min_chars]
    idx = np.linspace(0, len(paras) - 1, n).astype(int)
    return [paras[i] for i in idx]


def layout(text: str, pitch: int) -> list[str]:
    f = ImageFont.truetype(str(config.LATIN_MONO_FONT), round(pitch / 1.2))
    cells = int((S - 8) // f.getlength("M"))
    return textwrap.wrap(text, cells)[: (S - 8) // pitch]


def ink(lines: list[str], pitch: int) -> np.ndarray:
    """(S, S) float ink coverage in [0, 1] (1 = ink)."""
    f = ImageFont.truetype(str(config.LATIN_MONO_FONT), round(pitch / 1.2))
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        d.text((4, 4 + i * pitch + 0.9 * pitch / 1.2), t, font=f, fill=255, anchor="ls")
    return np.asarray(im, dtype=np.float32) / 255


def page(cond: str, a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    if cond == "mono":
        g = 255 * (1 - a)
        return np.stack([g, g, g], 2).astype(np.uint8)
    if cond == "solo_red_light":    # red text on black
        return np.stack([255 * a, 0 * a, 0 * a], 2).astype(np.uint8)
    if cond == "solo_red_dark":     # red ink on white
        return np.stack([255 + 0 * a, 255 * (1 - a), 255 * (1 - a)], 2).astype(np.uint8)
    if cond == "rgb2_light":        # A lights R, B lights G
        return np.stack([255 * a, 255 * b, 0 * a], 2).astype(np.uint8)
    if cond == "rgb2_dark":         # A darkens G+B (red ink), B darkens R (cyan ink)
        return np.stack([255 * (1 - b), 255 * (1 - a), 255 * (1 - a)], 2).astype(np.uint8)
    raise ValueError(cond)


COLOURS = {"mono": ("", None), "solo_red_light": ("red ", None), "solo_red_dark": ("red ", None),
           "rgb2_light": ("red ", "green "), "rgb2_dark": ("red ", "cyan ")}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def phrase(lines: list[str], rng) -> str:
    words = " ".join(lines).split()
    i = int(rng.integers(0, max(1, len(words) - 4)))
    return " ".join(words[i:i + 4])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pitch", type=int, default=16)
    ap.add_argument("--pages", type=int, default=6, help="pages for MC (and the first --transcribe-pages for CER)")
    ap.add_argument("--transcribe-pages", type=int, default=3)
    ap.add_argument("--conds", nargs="+", default=["mono", "solo_red_light", "solo_red_dark", "rgb2_light",
                                                   "rgb2_dark"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run-name", default="z0_colour")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    dev = resolve_device(a.device)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        Q.QWEN_REPO, revision=Q.QWEN_REV, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    tok = Q.HFTok(AutoTokenizer.from_pretrained(Q.QWEN_REPO, revision=Q.QWEN_REV))
    print(f"[z0] loaded in {time.time() - t0:.0f}s on {dev}", flush=True)

    texts = passages(2 * a.pages + 8)
    others = texts[2 * a.pages:]
    jobs = []   # (cond, page, stream, kind, prompt, ref, image, other-stream MC letter or None)
    for k in range(a.pages):
        la, lb = layout(texts[2 * k], a.pitch), layout(texts[2 * k + 1], a.pitch)
        ia, ib = ink(la, a.pitch), ink(lb, a.pitch)
        for cond in a.conds:
            img = page(cond, ia, ib)
            if k == 0:
                out = ROOT / "encoder_probe" / "results" / a.run_name / f"page0_{cond}_p{a.pitch}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(out)
            ca, cb = COLOURS[cond]
            streams = [("A", la, ca)] + ([("B", lb, cb)] if cb else [])
            for sid, lines, col in streams:
                other = lb if sid == "A" else la
                if k < a.transcribe_pages:
                    p = (f"Transcribe only the {col}text in this image, exactly, line by line. "
                         "Output only the transcription." if col else
                         "Transcribe the text in this image exactly, line by line. Output only the transcription.")
                    jobs.append((cond, k, sid, "transcribe", p, "\n".join(lines), img, None))
                opts = [phrase(lines, rng), phrase(other, rng) if cb else phrase(layout(others[0], a.pitch), rng),
                        phrase(layout(others[(2 * k) % len(others)], a.pitch), rng),
                        phrase(layout(others[(2 * k + 1) % len(others)], a.pitch), rng)]
                order = rng.permutation(4)
                q = (f"Which of these phrases appears in the {col}text in this image? " if col else
                     "Which of these phrases appears in the text in this image? ")
                q += " ".join(f"{LETTERS[j]}) {opts[o]}" for j, o in enumerate(order))
                q += " Answer with the letter only."
                jobs.append((cond, k, sid, "mc", q, LETTERS[int(np.where(order == 0)[0][0])], img,
                             LETTERS[int(np.where(order == 1)[0][0])] if cb else None))

    rows = []
    for kind, mnt in (("mc", 4), ("transcribe", a.max_new_tokens)):
        sel = [j for j in jobs if j[3] == kind]
        for i in range(0, len(sel), a.batch):
            chunk = sel[i:i + a.batch]
            t1 = time.time()
            hyps = Q.generate(model, tok, [{"imgs": [j[6]], "prompt": j[4]} for j in chunk], dev, mnt)
            for j, h in zip(chunk, hyps):
                r = {"cond": j[0], "page": j[1], "stream": j[2], "kind": kind, "pitch": a.pitch, "ref": j[5],
                     "hyp": h}
                if kind == "mc":
                    r["correct"] = h.strip()[:1].upper() == j[5]
                    r["picked_other_stream"] = j[7] is not None and h.strip()[:1].upper() == j[7]
                else:
                    from rapidfuzz.distance import Levenshtein
                    r["cer"] = Levenshtein.distance(norm(j[5]), norm(h)) / max(1, len(norm(j[5])))
                rows.append(r)
            print(f"[z0] {kind} {i + len(chunk)}/{len(sel)} ({time.time() - t1:.0f}s)", flush=True)

    out = ROOT / "encoder_probe" / "results" / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\ncond / stream: MC correct, mean CER")
    for cond in a.conds:
        for sid in "AB":
            mcr = [r for r in rows if r["cond"] == cond and r["stream"] == sid and r["kind"] == "mc"]
            mc = [r["correct"] for r in mcr]
            oth = sum(r.get("picked_other_stream", False) for r in mcr)
            ce = [r["cer"] for r in rows if r["cond"] == cond and r["stream"] == sid and r["kind"] == "transcribe"]
            if mc or ce:
                print(f"  {cond:15s} {sid}: MC {sum(mc)}/{len(mc)} (picked the other stream {oth})  "
                      f"CER {np.mean(ce) if ce else float('nan'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
