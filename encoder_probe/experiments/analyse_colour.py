"""Colour-gist phase A (docs/gpu/COLOUR_GIST.md): does an overlay carry more information per vision token than one
stream? Applies the pre-registered A1/A2 gates to the finetune_qwen.py rows of the rgb<k>s_p<p> arms.

    python3 encoder_probe/experiments/analyse_colour.py --real colour_a1 --pitch 16
        -> encoder_probe/results/analysis_colour_a1/summary.md (and printed)

Per arm (encoding rgb<k>s_p<p>, its real run and its --blind-images run, clean eval):
* information gap per text token g = (sum blind loss - sum real loss) / target tokens, on the SAME fixed val
  samples (``val_per_sample``: same targets in both runs, images swapped in the blind one);
* information per vision token = g x (text tokens per vision token), text tokens = chars / 4.33. The ratio between
  two arms does not depend on the 4.33;
* ratio R(k) = info per vision token of the k-stream arm / that of the 1-stream arm, with a 95% bootstrap CI
  (val samples resampled, paired real/blind within an arm, independently across arms);
* per stream: MC accuracy, lure rate (MC answers naming the OTHER stream's phrase), cloze, CER.
Gates (pre-registered; amendment 1 = the user's headline bar):
* headline: R(2) > 1 with the CI's lower bound > 1 (overlay carries more per token than one stream);
* strong (A1): R(2) >= 1.5, each stream's MC >= 0.8x the 1-stream MC, lure rate <= 15%;
* A2: R(3) >= 1.3 R(2) and the same per-stream rules;
* equal density ("useful"): the overlay vs one stream at a smaller font with ~ the same chars per token
  (default rgb2s_p16 vs rgb1s_p12), CI above 1.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
EQ = ["rgb2s_p16", "rgb1s_p12"]   # equal-density pair (main() sets it from --equal-density)
CHARS_PER_TEXT_TOKEN = 4.33


def load(run: str) -> dict:
    out = {}
    f = RES / run / "runs.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("cond") == "clean":
                    out[r["encoding"]] = r
    return out


def arm_gap(real: dict, blind: dict) -> tuple[np.ndarray, np.ndarray]:
    """-> (per-sample loss-sum gap blind - real, per-sample target tokens); checks the targets are paired."""
    vr, vb = real["val_per_sample"], blind["val_per_sample"]
    if [n for _, n in vr] != [n for _, n in vb]:
        raise ValueError(f"{real['encoding']}: real and blind val samples differ (not the same targets)")
    return np.array([b - r for (r, _), (b, _) in zip(vr, vb)]), np.array([n for _, n in vr], dtype=float)


def info_per_vision_token(gap: np.ndarray, n: np.ndarray, cpt: float) -> float:
    return gap.sum() / n.sum() * cpt / CHARS_PER_TEXT_TOKEN   # nats per vision token


def ratio_ci(a, b, cpt_a, cpt_b, reps: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """R = info(a) / info(b) and its 95% bootstrap CI. a, b = (gap, n) per-sample arrays."""
    rng = np.random.default_rng(seed)
    point = info_per_vision_token(*a, cpt_a) / info_per_vision_token(*b, cpt_b)
    ia = rng.integers(0, len(a[0]), (reps, len(a[0])))
    ib = rng.integers(0, len(b[0]), (reps, len(b[0])))
    ra = a[0][ia].sum(1) / a[1][ia].sum(1) * cpt_a
    rb = b[0][ib].sum(1) / b[1][ib].sum(1) * cpt_b
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = ra / rb
    lo, hi = np.nanpercentile(rs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def swap_blind(R: dict) -> dict:
    """Interim blind from the real rows' eval-time image swap (small_vlm.py ``val_swap_per_sample``)."""
    return {e: {**r, "val_per_sample": r["val_swap_per_sample"], "per_stream": [], "cer": float("nan")}
            for e, r in R.items() if "val_swap_per_sample" in r}


def analyse(real_run: str, blind_run: str, pitch: int, swap: bool = False) -> tuple[list[dict], dict]:
    R = load(real_run)
    B = swap_blind(R) if swap else load(blind_run)
    arms = {}
    for k in (1, 2, 3):
        enc = f"rgb{k}s_p{pitch}"
        if enc in R and enc in B:
            gap, n = arm_gap(R[enc], B[enc])
            arms[k] = {"enc": enc, "real": R[enc], "blind": B[enc], "gap": gap, "n": n,
                       "cpt": R[enc]["chars_per_token"],
                       "nats_per_vtok": info_per_vision_token(gap, n, R[enc]["chars_per_token"]),
                       "g_per_text_tok": gap.sum() / n.sum()}
    rows, verdict = [], {}
    for k, d in sorted(arms.items()):
        row = {"k": k, "enc": d["enc"], "cpt": d["cpt"], "g": d["g_per_text_tok"],
               "bits_per_vtok": d["nats_per_vtok"] / math.log(2),
               "per_stream": d["real"].get("per_stream", []), "blind_per_stream": d["blind"].get("per_stream", []),
               "mc": d["real"].get("mc_acc"), "cer": d["real"]["cer"], "blind_cer": d["blind"]["cer"]}
        if k > 1 and 1 in arms:
            row["R"], row["R_lo"], row["R_hi"] = ratio_ci((d["gap"], d["n"]), (arms[1]["gap"], arms[1]["n"]),
                                                          d["cpt"], arms[1]["cpt"])
        rows.append(row)
    if 1 in arms and 2 in arms:
        r2 = next(r for r in rows if r["k"] == 2)
        mc1 = arms[1]["real"].get("mc_acc") or 0.0
        ps = r2["per_stream"]
        verdict["headline_R2_gt_1"] = r2["R_lo"] > 1
        verdict["strong_A1"] = bool(r2["R"] >= 1.5 and ps and all(
            (s["mc_acc"] or 0) >= 0.8 * mc1 and (s["lure_rate"] or 0) <= 0.15 for s in ps))
    if 3 in arms and 2 in arms and 1 in arms:
        r2 = next(r for r in rows if r["k"] == 2)
        r3 = next(r for r in rows if r["k"] == 3)
        mc1 = arms[1]["real"].get("mc_acc") or 0.0
        verdict["A2"] = bool(r3["R"] >= 1.3 * r2["R"] and all(
            (s["mc_acc"] or 0) >= 0.8 * mc1 and (s["lure_rate"] or 0) <= 0.15 for s in r3["per_stream"]))
    return rows, verdict


def equal_density(real_run: str, blind_run: str, arm: str, ref: str, swap: bool = False) -> dict | None:
    """The 'useful' bar: info per vision token of ``arm`` (e.g. rgb2s_p16) vs a single stream at a SMALLER font with
    about the same chars per token (e.g. rgb1s_p12). -> {R, R_lo, R_hi, ...} or None if a row is missing."""
    R = load(real_run)
    B = swap_blind(R) if swap else load(blind_run)
    if not all(e in R and e in B for e in (arm, ref)):
        return None
    a, b = arm_gap(R[arm], B[arm]), arm_gap(R[ref], B[ref])
    r, lo, hi = ratio_ci(a, b, R[arm]["chars_per_token"], R[ref]["chars_per_token"])
    return {"arm": arm, "ref": ref, "cpt_arm": R[arm]["chars_per_token"], "cpt_ref": R[ref]["chars_per_token"],
            "R": r, "R_lo": lo, "R_hi": hi, "pass": lo > 1}


# ------------------------------------------------------------------------------------------------- gist
# Word presence ("Does the word w appear in the <colour> text?", small_vlm.py ``gist_pages``): per stream, the AUC
# of the yes-no score separating the stream's content words from other pages' words. Gist per vision token
# G = sum over streams of (content words in the stream) x max(0, 2 (AUC - AUC_blind)) / vision tokens, i.e. the
# number of the page's content words the model "knows", per token. AUC_blind = 0.5 unless a blind row exists.

def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def gist_value(row: dict, pages=None, blind_auc: dict | None = None) -> tuple[float, dict]:
    """-> (G per vision token, {stream: (AUC, separation AUC vs lures, mean content words)}) on ``pages``."""
    gp = row["gist_pages"]
    pages = sorted({e["page"] for e in gp}) if pages is None else pages
    out, G = {}, 0.0
    for st in sorted({e["stream"] for e in gp}):
        es = {e["page"]: e for e in gp if e["stream"] == st}
        sel = [es[p] for p in pages if p in es]
        pos = [x for e in sel for x in e["present"]]
        a = auc(pos, [x for e in sel for x in e["absent"]])
        sep = auc(pos, [x for e in sel for x in e.get("lure", [])])
        ncw = float(np.mean([e["n_cw"] for e in sel])) if sel else 0.0
        base = (blind_auc or {}).get(st, 0.5)
        if not np.isnan(a):
            G += ncw * max(0.0, 2 * (a - base))
        out[st] = (a, sep, ncw)
    return G / row["n_tokens"], out


def gist_ratio(arm: dict, ref: dict, arm_blind: dict | None = None, ref_blind: dict | None = None,
               reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """G(arm) / G(ref) with a 95% bootstrap CI over test pages (resampled independently per arm)."""
    def bauc(b):
        return {st: v[0] for st, v in gist_value(b)[1].items()} if b and "gist_pages" in b else None
    ba, br = bauc(arm_blind), bauc(ref_blind)
    point = gist_value(arm, None, ba)[0] / max(1e-12, gist_value(ref, None, br)[0])
    rng = np.random.default_rng(seed)
    pa = sorted({e["page"] for e in arm["gist_pages"]})
    pr = sorted({e["page"] for e in ref["gist_pages"]})
    rs = []
    for _ in range(reps):
        ga = gist_value(arm, list(rng.choice(pa, len(pa))), ba)[0]
        gr = gist_value(ref, list(rng.choice(pr, len(pr))), br)[0]
        if gr > 0:
            rs.append(ga / gr)
    lo, hi = np.percentile(rs, [2.5, 97.5]) if rs else (float("nan"), float("nan"))
    return float(point), float(lo), float(hi)


def gist_section(real_run: str, blind_run: str, pitch: int, eq: tuple[str, str]) -> list[str]:
    R, B = load(real_run), load(blind_run)
    arms = {k: f"rgb{k}s_p{pitch}" for k in (1, 2, 3) if f"rgb{k}s_p{pitch}" in R and "gist_pages" in R[f"rgb{k}s_p{pitch}"]}
    if 1 not in arms:
        return []
    L = ["", "## Gist (word presence): content words the model knows, per vision token", "",
         "| Arm | gist per vision token | ratio vs 1 stream (95% CI) | per stream: AUC present vs absent (separation "
         "vs the other stream's words) | content words per stream |", "|---|---|---|---|---|"]
    ref = R[arms[1]]
    for k, e in sorted(arms.items()):
        g, per = gist_value(R[e], None, {st: v[0] for st, v in gist_value(B[e])[1].items()}
                            if e in B and "gist_pages" in B[e] else None)
        rat = "1 (reference)"
        if k > 1:
            r, lo, hi = gist_ratio(R[e], ref, B.get(e), B.get(arms[1]))
            rat = f"{r:.2f} ({lo:.2f}-{hi:.2f})"
        ps = "; ".join(f"{a:.2f}" + (f" ({sp:.2f})" if not np.isnan(sp) else "") for a, sp, _ in per.values())
        L.append(f"| {e} | {g:.2f} | {rat} | {ps} | {', '.join(f'{c:.0f}' for _, _, c in per.values())} |")
    a, r = eq
    if a in R and r in R and "gist_pages" in R[a] and "gist_pages" in R[r]:
        x, lo, hi = gist_ratio(R[a], R[r], B.get(a), B.get(r))
        L += ["", f"Gist at equal density: {a} vs {r}: {x:.2f} ({lo:.2f}-{hi:.2f}), "
              f"{'PASS' if lo > 1 else 'not passed'} (needs the CI above 1)"]
    return L


def write_md(rows, verdict, real_run, blind_run, out: Path, eq: dict | None = None) -> str:
    L = [f"# Colour-gist analysis: `{real_run}` vs `{blind_run}` (generated by analyse_colour.py)", "",
         "| Arm | chars/token | info gap (nats/text token) | bits per vision token | ratio vs 1 stream (95% CI) "
         "| MC | CER (blind) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        rat = f"{r['R']:.2f} ({r['R_lo']:.2f}-{r['R_hi']:.2f})" if "R" in r else "1 (reference)"
        mc = f"{r['mc']:.2f}" if r["mc"] is not None else "n/a"
        L.append(f"| {r['enc']} | {r['cpt']:.1f} | {r['g']:.3f} | {r['bits_per_vtok']:.1f} | {rat} | {mc} | "
                 f"{r['cer']:.1%} ({r['blind_cer']:.1%}) |")
    L += ["", "Per stream (real; blind in brackets):", "",
          "| Arm | stream | MC | lure rate | cloze | CER |", "|---|---|---|---|---|---|"]
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"  # noqa: E731
    for r in rows:
        for s, b in zip(r["per_stream"], r["blind_per_stream"] or [{}] * len(r["per_stream"])):
            L.append(f"| {r['enc']} | {s['colour']} | {fmt(s['mc_acc'])} ({fmt(b.get('mc_acc'))}) | "
                     f"{fmt(s['lure_rate'])} ({fmt(b.get('lure_rate'))}) | {fmt(s['cloze_acc'])} "
                     f"({fmt(b.get('cloze_acc'))}) | {s['cer']:.1%} |")
    if eq:
        L += ["", f"Equal density (the 'useful' bar): {eq['arm']} ({eq['cpt_arm']:.1f} chars/token) vs {eq['ref']} "
              f"({eq['cpt_ref']:.1f}): R = {eq['R']:.2f} ({eq['R_lo']:.2f}-{eq['R_hi']:.2f}), "
              f"{'PASS' if eq['pass'] else 'not passed'} (needs the CI above 1)"]
    L += gist_section(real_run, blind_run if not blind_run.startswith("eval-time") else "", rows[0]["enc"].split("_p")[-1]
                      and int(rows[0]["enc"].split("_p")[-1]), tuple(EQ)) if rows else []
    L += ["", "Gates: " + (", ".join(f"{k} = {'PASS' if v else 'fail'}" for k, v in verdict.items()) or
                           "not evaluable (arms missing)")]
    text = "\n".join(L) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real", default="colour_a1")
    ap.add_argument("--blind", default=None, help="default <real>_blind")
    ap.add_argument("--pitch", type=int, default=16)
    ap.add_argument("--swap-blind", action="store_true", help="interim: eval-time image swap of the real runs "
                    "as the blind (small_vlm.py rows), instead of the trained blind runs")
    ap.add_argument("--equal-density", nargs=2, metavar=("ARM", "REF"), default=["rgb2s_p16", "rgb1s_p12"],
                    help="also compare ARM with a single stream REF at about the same chars per token")
    a = ap.parse_args(argv)
    blind = a.blind or f"{a.real}_blind"
    EQ[:] = a.equal_density
    rows, verdict = analyse(a.real, blind, a.pitch, a.swap_blind)
    eq = equal_density(a.real, blind, *a.equal_density, swap=a.swap_blind)
    if a.swap_blind:
        blind = "eval-time image swap (interim)"
    print(write_md(rows, verdict, a.real, blind, eq=eq, out=RES / f"analysis_{re.sub(r'[^A-Za-z0-9_]', '_', a.real)}{'_swap' if a.swap_blind else ''}" /
                   "summary.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
