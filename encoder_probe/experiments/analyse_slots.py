"""Table, gates and the comparison with the image arms for Test B (text_slots.py, docs/TEXT_SLOTS.md).

    python3 encoder_probe/experiments/analyse_slots.py encoder_probe/results/tb
    python3 encoder_probe/experiments/analyse_slots.py encoder_probe/results/tb --image latin_p8_trained:27.3:14.1:15.2

Each ``--image name:chars_per_token:bits_natural:bits_shuffled`` adds an image arm to compare with (use the upper
end of its 95% interval from analyse_run12.py; ``-`` for a missing text set). The frozen-Qwen Latin points of
Run 0 are built in. Writes <run>/summary.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# name -> (chars per vision token, upper 95% bound of bits per vision token: natural, shuffled). Run 0, frozen
# Qwen2.5-VL-3B, docs/PLAN_3090.md Results.
IMAGE_REF = {"latin_p8 frozen (Run 0)": (28.1, 13.10, 26.40)}
G1_CER, G2_CER = 0.01, 0.03
BUDGET_SAMPLES = 48000   # the Runs 1-2 image arms: 6000 steps x batch 8


def bits_ci(row: dict, text: str = "natural", n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Bits per slot on one eval text set, with a 95% bootstrap interval over the eval windows."""
    w = row["eval"][text]["window_losses"]
    gain = np.minimum(w["blind"], w["prior"]) - np.asarray(w["real"])
    k = (row["seq_tokens"] + 1) / row["mem_slots"] / math.log(2)
    rng = np.random.default_rng(seed)
    boot = gain[rng.integers(0, len(gain), (n_boot, len(gain)))].mean(1) * k
    return float(gain.mean() * k), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def table(rows: list[dict]) -> str:
    out = ["| ratio | trained on | decoder | steps | samples so far | read | chars/slot | CER | blind CER | loss | "
           "blind | prior | bits/slot (95% CI) | recovered | recovered chars/slot | s/step | peak GiB |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for text, e in r["eval"].items():
            b, lo, hi = bits_ci(r, text)
            out.append(f"| {r['ratio']} | {r['text']} | {r['decoder']} | {r['steps']} | {r['cum_samples']} | {text} "
                       f"| {e['chars_per_slot']:.1f} | {e['cer']:.1%} | {e['blind_cer']:.1%} | {e['loss']:.3f} | "
                       f"{e['blind_loss']:.3f} | {e['prior_loss']:.3f} | {b:.1f} ({lo:.1f}-{hi:.1f}) | "
                       f"{e['recovered_frac']:.1%} | {e['recovered_chars_per_slot']:.1f} | {r['s_per_step']:.2f} | "
                       f"{(r.get('peak_gpu_bytes') or 0) / 2**30:.1f} |")
    return "\n".join(out)


def gates(rows: list[dict]) -> list[str]:
    """G1 / G2 read the arm's own training text (shuffled by default), reader with LoRA, last matching row."""
    def pick(ratio):
        c = [r for r in rows if r["ratio"] == ratio and r["decoder"] == "lora" and r["steps"] > 0]
        return c[-1] if c else None
    out = []
    for name, ratio, bar, what in (("G1", 1, G1_CER, "ratio 1 copies the text after a short warm-up"),
                                   ("G2", 2, G2_CER, "ratio 2 copies the text")):
        r = pick(ratio)
        if r is None:
            out.append(f"* **{name}** ({what}, CER <= {bar:.0%}): not run")
        else:
            cer = r["eval"][r["text"]]["cer"]
            out.append(f"* **{name}** ({what}, CER <= {bar:.0%}): {'PASS' if cer <= bar else 'FAIL'} "
                       f"(CER {cer:.2%} on {r['text']} text, {r['steps']} steps)")
    return out


def compare(rows: list[dict], image: dict) -> list[str]:
    """Pre-registered win rule: a text arm beats an image arm on a text set when it carries at least as many
    chars per position and the lower end of its bits-per-slot interval is above the image arm's upper end. A win
    with more than BUDGET_SAMPLES training samples is marked: it is not a same-budget win."""
    out = []
    for name, (cpt, nat, shuf) in image.items():
        for text, ref in (("natural", nat), ("shuffled", shuf)):
            if ref is None:
                continue
            wins = []
            for r in rows:
                e = r["eval"][text]
                if e["chars_per_slot"] >= cpt:
                    b, lo, hi = bits_ci(r, text)
                    if lo > ref:
                        over = "" if r["cum_samples"] <= BUDGET_SAMPLES else ", OVER the image arms' budget"
                        wins.append(f"ratio {r['ratio']} {r['decoder']} ({e['chars_per_slot']:.0f} chars/slot, "
                                    f"{b:.1f} bits, {r['cum_samples']} samples{over})")
            out.append(f"* vs **{name}** ({cpt:.1f} chars/token, {ref:.1f} bits upper bound, {text} text): "
                       + ("TEXT WINS with " + "; ".join(wins) if wins else "no text arm wins"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="encoder_probe/results/<run>")
    ap.add_argument("--image", nargs="*", default=[], help="name:chars_per_token:bits_natural:bits_shuffled")
    a = ap.parse_args(argv)
    rows = [json.loads(x) for x in (Path(a.run) / "runs.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("phase") == "slots" and not r.get("smoke")]
    image = dict(IMAGE_REF)
    for spec in a.image:
        name, cpt, nat, shuf = spec.split(":")
        image[name] = (float(cpt), None if nat == "-" else float(nat), None if shuf == "-" else float(shuf))
    md = "\n\n".join(["# Test B: text slots (analyse_slots.py)", table(rows), "## Gates", "\n".join(gates(rows)),
                      "## Against the image arms", "\n".join(compare(rows, image))]) + "\n"
    (Path(a.run) / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
