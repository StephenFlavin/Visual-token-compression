"""PLAN_3090 Runs 1-2 analysis (amendment 3): trained-ViT Latin vs frozen, and glyphs vs trained Latin.

Inputs are capacity_curve.py run dirs (samples.jsonl), all scored on the same test pages (same encodings, text sets,
--pages, --shifts 7, seed):
* --frozen   the frozen model (no adapter): the Run 0 baseline repeated on the GPU with a 7-px margin;
* --trained  each encoding scored with its own Run 1/2 adapter (one adapter per encoding in this dir);
* --blind    each encoding scored with its blind control's adapter (images deranged during training).
Per (encoding, text): mean bits per vision token with a page-bootstrap 95% CI. Run 1 test: trained minus frozen,
paired by page. Run 2 test: glyph minus the trained Latin arm it is matched to, unpaired page bootstrap (different
encodings consume different text). Optional --sweep: the dy sweep of the trained adapters (``capacity_curve.py
--shifts 7 14 21 28 --shift-dx 7 --arms real blank``, amendment 5), used only for the placement-sensitive bar. Writes summary.md / summary.json to --out.

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

# Run 2 matched pairs (amendments 3-4): glyph arm -> trained Latin arm at about the same density (7-px margin)
PAIRS = {"lg_n0_pb_c4_h7": "latin_p8", "lg_n0_pb_c2_h7md": "latin_p6", "lg_pb_c2t4_h7md": "latin_p5",
         "lg_n0_pb_c2_h7": "latin_p6", "lg_pb_c2t4_h7": "latin_p5"}
STAGE_C = ("lg_n0_pb_c2_h7md", "lg_pb_c2t4_h7md")
BAR_RECOVERED = 20.0   # amendment 4: the write-up bar, recovered chars per vision token
BAR_SENSITIVE_LO = 18.0   # amendment 5: a placement-sensitive pass needs at least this at the trained placement
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
        e = out.setdefault(k, {"bits": {}, "cpt": [], "gap": 0.0, "ref": 0.0})
        st = C.sample_stats(s)
        e["bits"][s["page"]] = st["bits_vtok"]
        e["cpt"].append(s["page_chars"] / s["vision_tokens"])
        e["gap"] += st["gap"]
        e["ref"] += st["ref"]
    return out


def load_sweep(d: Path | None) -> dict:
    """dy sweep (amendment 5): (encoding, text) -> {dy: recovered chars per vision token}, pooled over pages."""
    acc = {}
    if d is None or not (Path(d) / "samples.jsonl").exists():
        return {}
    for line in (Path(d) / "samples.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        st = C.sample_stats(s)
        e = acc.setdefault((s["encoding"], s["text"]), {}).setdefault(int(s["shift"][1]),
                                                                       {"gap": 0.0, "ref": 0.0, "cpt": []})
        e["gap"] += st["gap"]
        e["ref"] += st["ref"]
        e["cpt"].append(s["page_chars"] / s["vision_tokens"])
    return {k: {dy: (v["gap"] / v["ref"]) * float(np.mean(v["cpt"])) for dy, v in sorted(by.items()) if v["ref"]}
            for k, by in acc.items()}


def bar_status(primary: float, sweep_dy: dict | None, train_dy: int = 7) -> dict:
    """Amendment 5: the > 20 bar reads the trained placement (dy = 7, the main curve). The dy sweep (margin 28, less
    text per page) is used only as a ratio: placement-adjusted = primary x best-dy / dy-7 within the sweep. A
    'placement-sensitive pass' = primary in [18, 20] and placement-adjusted > 20 (the best dy is selected post hoc)."""
    out = {"primary": primary, "status": "PASS" if primary > BAR_RECOVERED else "below bar"}
    if sweep_dy and train_dy in sweep_dy and sweep_dy[train_dy] > 0:
        best = max(sweep_dy, key=sweep_dy.get)
        ratio = sweep_dy[best] / sweep_dy[train_dy]
        out.update({"sweep": sweep_dy, "best_dy": best, "best_over_dy7": ratio, "adjusted": primary * ratio})
        if out["status"] != "PASS" and BAR_SENSITIVE_LO <= primary <= BAR_RECOVERED and primary * ratio > BAR_RECOVERED:
            out["status"] = "placement-sensitive pass"
    return out


def load_chan(d: Path | None) -> list[dict]:
    """channel_capacity.py rows (amendment 6), non-smoke, one per (cell, levels, patch-embed arm)."""
    if d is None or not (Path(d) / "runs.jsonl").exists():
        return []
    rows = [json.loads(x) for x in (Path(d) / "runs.jsonl").read_text().splitlines() if x.strip()]
    return [r for r in rows if r.get("phase") == "channel_capacity" and not r.get("smoke")]


def chan_readout(chan: list[dict], points: list[dict]) -> dict:
    """Amendment 6 readout (a diagnostic, not a gate): the best block arm's recovered bits per token vs the best
    trained Latin arm on shuffled text. Blocks are one code at one budget, so their result is a FLOOR on the pipe's
    capacity: headroom above Latin says the pipe is not the wall at that rate; no headroom is inconclusive."""
    if not chan:
        return {}
    best = max(chan, key=lambda r: r["trained"]["recovered_bits_per_token"][0])
    lat = [r for r in points if r["encoding"].startswith("latin_") and r["text"] == "shuffled" and "trained" in r]
    lb = max(lat, key=lambda r: r["trained"][0]) if lat else None
    out = {"best_cell": best["cell"], "best_bits": best["trained"]["recovered_bits_per_token"],
           "swap_ok": all(r["trained"]["swap_recovered_bits_per_token"][2] <= 0.5 for r in chan)}
    if lb:
        out.update({"latin": lb["encoding"], "latin_bits": lb["trained"][0],
                    "headroom": best["trained"]["recovered_bits_per_token"][1] > lb["trained"][0]})
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


def analyse(frozen: dict, trained: dict, blind: dict, sweep: dict | None = None, chan: list | None = None) -> dict:
    res = {"points": [], "run1": [], "run2": []}
    for k in sorted(set(frozen) | set(trained)):
        row = {"encoding": k[0], "text": k[1]}
        for name, src in (("frozen", frozen), ("trained", trained), ("blind", blind)):
            if k in src:
                x = np.array(list(src[k]["bits"].values()))
                row[name] = boot_mean(x)
                row["chars_per_token"] = float(np.mean(src[k]["cpt"]))
                frac = src[k]["gap"] / src[k]["ref"] if src[k]["ref"] else float("nan")
                row[f"{name}_recovered_frac"] = frac
                row[f"{name}_recovered_cpt"] = frac * row["chars_per_token"]
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
    g2 = [r for r in res["run2"] if r["text"] == "shuffled" and r["glyph"] in STAGE_C]
    res["verdict_run2"] = ("PASS" if any(r["glyph_wins"] for r in g2) else "FAIL") if g2 else "n/a"
    # amendment 4, stage A
    pt = {(r["encoding"], r["text"]): r for r in res["points"]}
    c7 = pt.get(("lg_n0_pb_c7_h7", "shuffled"), {})
    res["gate_A1"] = ("n/a" if "trained" not in c7 else
                      ("PASS" if c7["trained_recovered_frac"] >= 0.8 else "FAIL"))
    res["gate_A1_value"] = c7.get("trained_recovered_frac")
    a2 = next((r for r in res["run2"] if r["glyph"] == "lg_n0_pb_c4_h7" and r["text"] == "shuffled"), None)
    if a2 is None:
        res["gate_A2"] = "n/a"
    else:
        lat = pt[("latin_p8", "shuffled")]["trained"][0]
        glyph = pt[("lg_n0_pb_c4_h7", "shuffled")]["trained"][0]
        res["gate_A2"] = "PASS" if glyph >= 0.9 * lat and a2["glyph_minus_latin"][2] >= 0 else "FAIL"
        res["gate_A2_ratio"] = glyph / lat if lat else float("nan")
    res["bar_recovered_cpt"] = {f"{r['encoding']}/{r['text']}": r["trained_recovered_cpt"]
                                for r in res["points"] if "trained_recovered_cpt" in r}
    res["bar"] = {f"{r['encoding']}/{r['text']}": bar_status(r["trained_recovered_cpt"],
                                                             (sweep or {}).get((r["encoding"], r["text"])))
                  for r in res["points"] if "trained_recovered_cpt" in r}
    res["chan"] = chan or []
    res["chan_readout"] = chan_readout(res["chan"], res["points"])
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
    md += ["", f"**Stage A (amendment 4): A1 learnability (lg_n0_pb_c7_h7 recovered fraction on shuffled >= 0.8) "
           f"{res['gate_A1']} ({res.get('gate_A1_value')}); A2 glyph matches Latin at 28 (lg_n0_pb_c4_h7 >= 0.9 x "
           f"latin_p8, CI upper >= 0) {res['gate_A2']} (ratio {res.get('gate_A2_ratio')})**", "",
           f"Write-up bar: recovered chars per vision token > {BAR_RECOVERED:.0f} (trained): "
           + ", ".join(f"{k} {v:.1f}" for k, v in res["bar_recovered_cpt"].items()), "",
           "| encoding/text | recovered chars/token at dy 7 (primary) | dy sweep (recovered chars/token by dy) | "
           "best dy / dy 7 | placement-adjusted | bar (amendment 5) |", "|---|---|---|---|---|---|"]
    for k, b in res.get("bar", {}).items():
        sw = ", ".join(f"{dy}: {v:.1f}" for dy, v in b["sweep"].items()) if "sweep" in b else "-"
        md.append(f"| {k} | {b['primary']:.1f} | {sw} | " + (f"{b['best_over_dy7']:.3f} (dy {b['best_dy']}) | "
                  f"{b['adjusted']:.1f} |" if "sweep" in b else "- | - |") + f" {b['status']} |")
    if res.get("chan"):
        c = res["chan_readout"]
        md += ["", "**Channel capacity (amendment 6, diagnostic): random bits as colour blocks, recovered bits per "
               "vision token = 3 x digits - CE (exact reference, no blind needed)**", "",
               "| cell px | levels | offered bits/token | tokens (px) | frozen | trained (CI) | fraction | digit acc | "
               "swapped images (must be <= 0) |", "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(res["chan"], key=lambda r: r["bits_per_token"]):
            t, f = r["trained"], r["frozen"]
            md.append(f"| {r['cell']}{' +pe' if r['config'].get('train_patch_embed') else ''} | {r['levels']} | "
                      f"{r['bits_per_token']} | {r['n_tokens']} ({r['S']}) | {f['recovered_bits_per_token'][0]:.1f} | "
                      f"{fmt(t['recovered_bits_per_token'])} | {t['recovered_frac']:.3f} | {t['digit_acc']:.3f} | "
                      f"{fmt(t['swap_recovered_bits_per_token'])} |")
        if c:
            md += ["", f"Best block arm: cell {c['best_cell']}, {fmt(c['best_bits'])} bits/token; best trained Latin "
                   f"(shuffled): {c.get('latin', '-')} {c.get('latin_bits', float('nan')):.2f}; headroom (block CI "
                   f"lower bound above Latin): {c.get('headroom', 'n/a')}; swap check (all upper CIs <= 0.5): "
                   f"{'PASS' if c['swap_ok'] else 'FAIL'}"]
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
    ap.add_argument("--sweep", default=None, help="dy-sweep run dir of the trained adapters (amendment 5)")
    ap.add_argument("--chan", default=None, help="channel_capacity.py run dir (amendment 6)")
    ap.add_argument("--out", default=str(ROOT / "encoder_probe" / "results" / "analysis_run12"))
    a = ap.parse_args(argv)
    res = analyse(load(Path(a.frozen)), load(Path(a.trained)), load(Path(a.blind) if a.blind else None),
                  load_sweep(Path(a.sweep) if a.sweep else None), load_chan(Path(a.chan) if a.chan else None))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(write_md(res, out / "summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
