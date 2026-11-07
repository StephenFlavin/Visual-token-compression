"""PLAN_3090 Runs 1-2 analysis (amendment 3): trained-ViT Latin vs frozen, and glyphs vs trained Latin.

Inputs are capacity_curve.py run dirs (samples.jsonl), all scored on the same test pages (same encodings, text sets,
--pages, --shifts 7, seed):
* --frozen   the frozen model (no adapter): the Run 0 baseline repeated on the GPU with a 7-px margin;
* --trained  each encoding scored with its own Run 1/2 adapter (one adapter per encoding in this dir);
* --blind    each encoding scored with its blind control's adapter (images deranged during training).
Per (encoding, text): mean bits per vision token with a page-bootstrap 95% CI. Run 1 test: trained minus frozen,
paired by page. Run 2 test: glyph minus the trained Latin arm it is matched to, unpaired page bootstrap (different
encodings consume different text). Writes summary.md / summary.json to --out.

    python encoder_probe/experiments/analyse_run12.py --frozen encoder_probe/results/r12_frozen \\
        --trained encoder_probe/results/r12_trained --blind encoder_probe/results/r12_blind
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity_curve as C  # noqa: E402

# Run 2 matched pairs (amendment 3): glyph arm -> trained Latin arm at the nearest LOWER density
PAIRS = {"lg_n0_pb_c2_h7": "latin_p6", "lg_pb_c2t4_h7": "latin_p5", "lg_n0_pb_h5": "latin_p5"}
LATIN_RUN1 = ["latin_p8", "latin_p6", "latin_p5"]


def load(d: Path | None) -> dict:
    """(encoding, text) -> {page: bits per vision token}, plus chars per token."""
    out = {}
    if d is None or not (Path(d) / "samples.jsonl").exists():
        return out
    for line in (Path(d) / "samples.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        k = (s["encoding"], s["text"])
        e = out.setdefault(k, {"bits": {}, "cpt": []})
        e["bits"][s["page"]] = C.sample_stats(s)["bits_vtok"]
        e["cpt"].append(s["page_chars"] / s["vision_tokens"])
    return out


def boot_mean(x: np.ndarray, reps: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_diff(x: np.ndarray, y: np.ndarray, reps: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
    by = y[rng.integers(0, len(y), (reps, len(y)))].mean(1)
    d = bx - by
    return float(x.mean() - y.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def analyse(frozen: dict, trained: dict, blind: dict) -> dict:
    res = {"points": [], "run1": [], "run2": []}
    for k in sorted(set(frozen) | set(trained)):
        row = {"encoding": k[0], "text": k[1]}
        for name, src in (("frozen", frozen), ("trained", trained), ("blind", blind)):
            if k in src:
                x = np.array(list(src[k]["bits"].values()))
                row[name] = boot_mean(x)
                row["chars_per_token"] = float(np.mean(src[k]["cpt"]))
        res["points"].append(row)
        if k in frozen and k in trained:   # Run 1: paired by page
            pages = sorted(set(frozen[k]["bits"]) & set(trained[k]["bits"]))
            d = np.array([trained[k]["bits"][p] - frozen[k]["bits"][p] for p in pages])
            m, lo, hi = boot_mean(d)
            res["run1"].append({"encoding": k[0], "text": k[1], "n": len(pages), "trained_minus_frozen": [m, lo, hi],
                                "moves_right": lo > 0})
    for g, lat in PAIRS.items():
        for t in ("shuffled", "natural"):
            if (g, t) in trained and (lat, t) in trained:
                x = np.array(list(trained[(g, t)]["bits"].values()))
                y = np.array(list(trained[(lat, t)]["bits"].values()))
                m, lo, hi = boot_diff(x, y)
                res["run2"].append({"glyph": g, "latin": lat, "text": t, "glyph_minus_latin": [m, lo, hi],
                                    "glyph_wins": lo > 0,
                                    "glyph_cpt": float(np.mean(trained[(g, t)]["cpt"])),
                                    "latin_cpt": float(np.mean(trained[(lat, t)]["cpt"]))})
    shuf = [r for r in res["run1"] if r["text"] == "shuffled" and r["encoding"] in ("latin_p6", "latin_p5")]
    res["verdict_run1"] = ("PASS" if any(r["moves_right"] for r in shuf) else "FAIL") if shuf else "n/a"
    g2 = [r for r in res["run2"] if r["text"] == "shuffled" and r["glyph"] in ("lg_n0_pb_c2_h7", "lg_pb_c2t4_h7")]
    res["verdict_run2"] = ("PASS" if any(r["glyph_wins"] for r in g2) else "FAIL") if g2 else "n/a"
    return res


def fmt(t) -> str:
    return "-" if t is None else f"{t[0]:.2f} ({t[1]:.2f} to {t[2]:.2f})"


def write_md(res: dict, path: Path) -> str:
    md = ["# Runs 1-2: bits per vision token (PLAN_3090 amendment 3)", "",
          "| encoding | text | chars/token | frozen | trained | blind (trained on deranged images) |",
          "|---|---|---|---|---|---|"]
    for r in res["points"]:
        md.append(f"| {r['encoding']} | {r['text']} | {r.get('chars_per_token', 0):.1f} | {fmt(r.get('frozen'))} | "
                  f"{fmt(r.get('trained'))} | {fmt(r.get('blind'))} |")
    md += ["", f"**Run 1 (trained minus frozen, paired by page): {res['verdict_run1']}** "
           "(pass = CI above 0 on shuffled text at latin_p6 or latin_p5)", "",
           "| encoding | text | pages | trained − frozen |", "|---|---|---|---|"]
    md += [f"| {r['encoding']} | {r['text']} | {r['n']} | {fmt(r['trained_minus_frozen'])} |" for r in res["run1"]]
    md += ["", f"**Run 2 (glyph minus matched trained Latin): {res['verdict_run2']}** "
           "(pass = CI above 0 on shuffled text for an aligned bank)", "",
           "| glyph (chars/token) | Latin (chars/token) | text | glyph − Latin |", "|---|---|---|---|"]
    md += [f"| {r['glyph']} ({r['glyph_cpt']:.1f}) | {r['latin']} ({r['latin_cpt']:.1f}) | {r['text']} | "
           f"{fmt(r['glyph_minus_latin'])} |" for r in res["run2"]]
    text = "\n".join(md) + "\n"
    path.write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--blind", default=None)
    ap.add_argument("--out", default=str(ROOT / "encoder_probe" / "results" / "analysis_run12"))
    a = ap.parse_args(argv)
    res = analyse(load(Path(a.frozen)), load(Path(a.trained)), load(Path(a.blind) if a.blind else None))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(write_md(res, out / "summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
