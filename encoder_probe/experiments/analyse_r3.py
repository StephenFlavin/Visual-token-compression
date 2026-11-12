"""Round-3 R3-1 analysis: how much each code is read, measured against its own blind control.

    python3 encoder_probe/experiments/analyse_r3.py      # writes encoder_probe/results/analysis_r3/summary.md
                                                         # and encoder_probe/img/r31_info.png
    python3 encoder_probe/experiments/analyse_r3.py --real RUN --blind RUN_blind --tag a1   # a later run pair
                                                         # -> results/analysis_a1/summary.md, img/r31_info_a1.png

No torch, no network, no scipy. Reads the committed R3-1 rows (round3_ft_gpu3h, round3_ft_gpu3h_blind). Measures:
* exact one-sided tests: MC / cloze real vs blind (Fisher) and MC vs chance 0.25 (binomial);
* validation-loss information gain: 1 - L_real / L_blind on the same 16 fixed val targets (token-weighted nats);
* bits per vision token: delivered = (L_blind - L_real) * text tokens per vision token / ln 2, needed = L_blind *
  text tokens per vision token / ln 2 (what the image must supply beyond the model's prior to make the loss zero);
* input dependence of the generations: edit distance of each hypothesis to its own reference vs other pages'
  references (4 stored 300-char samples per row: indicative only);
* a crude log-step fit of the val-loss gap (the schedule is cosine to 0 at step 1000, so the late flattening is partly
  the LR decay and the fit understates what a longer schedule gives).
The only typed-in values: the lg_pb_h5 blind val losses at steps 250/500 (job died before writing a row; logged in
docs/background/history/HANDOFF_2026-11.md) and the measured 4.33 chars per Qwen text token (held-out WikiText).
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT_MD = RES / "analysis_r3" / "summary.md"
OUT_IMG = ROOT / "img" / "r31_info.png"
REAL_RUN, BLIND_RUN = "round3_ft_gpu3h", "round3_ft_gpu3h_blind"
CHARS_PER_TEXT_TOKEN = 4.33
LG_BLIND_CURVE = [(250, 2.279), (500, 2.285)]   # lg_pb_h5 blind val loss (no row: disk full at step 500)
LABEL = {"rgb3_lg_pb_h5": "moonshot rgb3_lg_pb_h5"}


def load() -> dict:
    out = {}
    for run, tag in ((REAL_RUN, "real"), (BLIND_RUN, "blind")):
        for line in (RES / run / "runs.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["encoding"], tag, r["cond"])] = r
    return out


def fisher_greater(a: int, n1: int, b: int, n2: int) -> float:
    """One-sided Fisher exact p: P(real successes >= a | margins), i.e. real better than blind."""
    k, n = a + b, n1 + n2
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / math.comb(n, k)


def binom_ge(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, x) * p ** x * (1 - p) ** (n - x) for x in range(k, n + 1))


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def input_dependence(r: dict) -> tuple[float, float]:
    """Mean normalised distance hyp->own ref and hyp->other pages' refs (lower own = reads its own page)."""
    refs, hyps = [s[0] for s in r["samples"]], [s[1] for s in r["samples"]]
    nd = lambda a, b: lev(a, b) / max(len(a), len(b), 1)  # noqa: E731
    own = [nd(x, h) for x, h in zip(refs, hyps)]
    other = [nd(refs[j], hyps[i]) for i, j in itertools.permutations(range(len(refs)), 2)]
    return sum(own) / len(own), sum(other) / len(other)


def logfit(points: list[tuple[int, float]]) -> tuple[float, float]:
    xs, ys = [math.log(s) for s, _ in points], [v for _, v in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def analyse(rows: dict) -> list[dict]:
    res = []
    codes = sorted({k[0] for k in rows if k[1] == "real"}, key=lambda e: rows[(e, "real", "clean")]["chars_per_token"])
    for enc in codes:
        R = rows[(enc, "real", "clean")]
        B = rows.get((enc, "blind", "clean"))
        if B is None and not (enc == "lg_pb_h5" and REAL_RUN == "round3_ft_gpu3h"):
            print(f"skip {enc}: no blind row in {BLIND_RUN}")
            continue
        cpt = R["chars_per_token"]
        tpv = cpt / CHARS_PER_TEXT_TOKEN                      # text tokens per vision token
        real_curve = [(s, v) for s, _, v in R["curve"]]
        blind_curve = [(s, v) for s, _, v in B["curve"]] if B else LG_BLIND_CURVE  # R3-1 lg_pb_h5 only
        bl_by = dict(blind_curve)
        gaps = [(s, bl_by[s] - v) for s, v in real_curve if s in bl_by]
        l_real, (s_b, l_blind) = real_curve[-1][1], blind_curve[-1]
        d = {"enc": enc, "cpt": cpt, "cer": R["cer"],
             "mc": round(R["mc_acc"] * R["n_mc"]), "n_mc": R["n_mc"],
             "cloze": round(R["cloze_acc"] * R["n_cloze"]), "n_cloze": R["n_cloze"],
             "l_real": l_real, "l_blind": l_blind, "blind_step": s_b, "gaps": gaps,
             "real_curve": real_curve, "blind_curve": blind_curve}
        # matched-step gap (the last step both arms have) for the gain and the bit rates
        s_m, g_m = gaps[-1]
        lb_m = bl_by[s_m]
        d.update(gain=g_m / lb_m, gap_step=s_m,
                 bits_delivered=g_m * tpv / math.log(2), bits_needed=lb_m * tpv / math.log(2))
        d["p_mc_chance"] = binom_ge(d["mc"], d["n_mc"], 0.25)
        d["own"], d["other"] = input_dependence(R)
        if B:
            d["b_cer"], d["b_mc"], d["b_cloze"] = B["cer"], round(B["mc_acc"] * B["n_mc"]), round(B["cloze_acc"] * B["n_cloze"])
            d["p_mc"] = fisher_greater(d["mc"], d["n_mc"], d["b_mc"], B["n_mc"])
            d["p_cloze"] = fisher_greater(d["cloze"], d["n_cloze"], d["b_cloze"], B["n_cloze"])
            d["b_own"], d["b_other"] = input_dependence(B)
        if len(gaps) >= 3:
            a, b = logfit(gaps)
            d["gap_slope_per_doubling"] = b * math.log(2)
            d["gap_5000"] = a + b * math.log(5000)
        res.append(d)
    return res


def fmt_p(p: float | None) -> str:
    return "" if p is None else ("< 0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.2f}")


def write_md(res: list[dict]) -> None:
    L = ["# R3-1 analysis (generated by `encoder_probe/experiments/analyse_r3.py`)", "",
         f"Rows: `{REAL_RUN}` and `{BLIND_RUN}` (clean eval). p-values are exact and one-sided (real better than "
         "blind / than chance 0.25).", "",
         "## Is the image used?", "",
         "| Code (chars/token) | CER (blind) | MC real / blind (p vs blind; p vs chance) | Cloze real / blind (p) "
         "| Val loss real / blind (gain) |",
         "|---|---|---|---|---|"]
    for d in res:
        b_mc = f"{d['b_mc']}/{d['n_mc']}" if "b_mc" in d else "n/a"
        b_cl = f"{d['b_cloze']}/{d['n_cloze']}" if "b_cloze" in d else "n/a"
        b_cer = f"{d['b_cer']:.1%}" if "b_cer" in d else "n/a"
        lb = f"{d['l_blind']:.3f}" + ("" if d["blind_step"] == 1000 else f" @ step {d['blind_step']}")
        L.append(f"| {LABEL.get(d['enc'], d['enc'])} ({d['cpt']:.1f}) | {d['cer']:.1%} ({b_cer}) | {d['mc']}/{d['n_mc']} / {b_mc} "
                 f"(p {fmt_p(d.get('p_mc')) or 'n/a'}; {fmt_p(d['p_mc_chance'])}) | {d['cloze']}/{d['n_cloze']} / {b_cl} "
                 f"(p {fmt_p(d.get('p_cloze')) or 'n/a'}) | {d['l_real']:.3f} / {lb} ({d['gain']:+.1%} at step {d['gap_step']}) |")
    L += ["", "## Bits per vision token (from the val-loss gap)", "",
          "delivered = (L_blind - L_real) x text tokens per vision token / ln 2; needed = L_blind x the same (what the "
          f"image must add to the model's prior for a zero loss). Text tokens per vision token = chars/token / "
          f"{CHARS_PER_TEXT_TOKEN}.", "",
          "| Code | chars/token | needed (bits/token) | delivered (bits/token) | share | gap per doubling of steps | "
          "gap at 5000 steps, log fit |", "|---|---|---|---|---|---|---|"]
    for d in res:
        slope = f"{d['gap_slope_per_doubling']:+.3f} nats" if "gap_slope_per_doubling" in d else "n/a"
        g5 = f"{d['gap_5000']:.2f} nats ({d['gap_5000'] / d['l_blind']:.0%})" if "gap_5000" in d else "n/a"
        L.append(f"| {LABEL.get(d['enc'], d['enc'])} | {d['cpt']:.1f} | {d['bits_needed']:.0f} | {d['bits_delivered']:.1f} | "
                 f"{d['bits_delivered'] / d['bits_needed']:.0%} | {slope} | {g5} |")
    L += ["", "## Do generations depend on the page? (4 stored 300-char samples per row: indicative)", "",
          "Normalised edit distance of each hypothesis to its own page's reference vs to the other pages' references. "
          "Equal distances = the output does not depend on which page was shown.", "",
          "| Code | real: own / other | blind: own / other |", "|---|---|---|"]
    for d in res:
        b = f"{d['b_own']:.3f} / {d['b_other']:.3f}" if "b_own" in d else "n/a"
        L.append(f"| {LABEL.get(d['enc'], d['enc'])} | {d['own']:.3f} / {d['other']:.3f} | {b} |")
    L += ["", "## Val-loss gap (blind - real, nats per text token) by step", "",
          "| Code | " + " | ".join(f"step {s}" for s in (250, 500, 750, 1000)) + " |", "|---|---|---|---|---|"]
    for d in res:
        g = dict(d["gaps"])
        L.append(f"| {LABEL.get(d['enc'], d['enc'])} | " + " | ".join(f"{g[s]:+.3f}" if s in g else "" for s in (250, 500, 750, 1000)) + " |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    print(OUT_MD.read_text())


def figure(res: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_figures as F  # shared palette and rcParams

    pal = [F.BLUE, F.AQUA, F.YELLOW, F.ORANGE, F.INK2, F.BLIND_DARK]
    colours = {d["enc"]: pal[i % len(pal)] for i, d in enumerate(res)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for d in res:
        s, g = zip(*d["gaps"])
        a1.plot(s, g, "o-", color=colours[d["enc"]], lw=1.8, ms=4, label=f"{LABEL.get(d['enc'], d['enc'])} ({d['cpt']:.0f})")
    a1.axhline(0, color=F.BLIND_DARK, lw=1, ls="--")
    a1.set_xlabel("training step")
    a1.set_ylabel("val loss: blind minus real (nats / text token)")
    a1.set_title("Information taken from the image, per step")
    a1.set_yscale("symlog", linthresh=0.01, linscale=0.6)
    a1.set_ylim(-0.02, 3)
    a1.set_yticks([-0.01, 0, 0.01, 0.1, 1])
    a1.set_yticklabels(["-0.01", "0", "0.01", "0.1", "1"])
    a1.set_xticks([250, 500, 750, 1000])
    a1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    a1.text(1000, -0.016, "0 = the image adds nothing", color=F.BLIND_DARK, fontsize=8, ha="right")

    xs = [d["cpt"] for d in res]
    a2.plot(xs, [d["bits_needed"] for d in res], "o--", color=F.BLIND_DARK, ms=5, lw=1.2, label="needed for zero loss")
    for d in res:
        v = max(d["bits_delivered"], 0.1)
        a2.plot([d["cpt"]], [v], "o", color=colours[d["enc"]], ms=8,
                mfc=colours[d["enc"]] if d["bits_delivered"] > 0.5 else "none", mew=2)
        lab = f"{d['bits_delivered']:.1f}" if d["bits_delivered"] > 0.5 else "~0"
        a2.annotate(lab, (d["cpt"], v), textcoords="offset points", xytext=(8, -3), fontsize=8, color=F.INK2)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xticks(xs)
    a2.set_xticklabels([f"{x:.0f}" for x in xs])
    a2.minorticks_off()
    a2.set_ylim(0.05, 600)
    a2.set_xlabel("chars per vision token (log)")
    a2.set_ylabel("bits per vision token (log)")
    a2.set_title("Bits per vision token: delivered vs needed")
    a2.plot([], [], "o", color=F.INK2, label="delivered (hollow = ~0)")
    a2.legend(loc="upper left")
    if REAL_RUN == "round3_ft_gpu3h":
        fig.suptitle("R3-1 (1000 steps): latin_p8 delivers about half the needed bits, latin_p5 a tenth, glyph codes "
                     "none", fontsize=10, fontweight="bold")
    else:
        fig.suptitle(f"{REAL_RUN} vs {BLIND_RUN}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_IMG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_IMG}")


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real", default=REAL_RUN, help="run name of the real arms (encoder_probe/results/<run>/)")
    ap.add_argument("--blind", default=None, help="run name of the blind arms (default: <real>_blind)")
    ap.add_argument("--tag", default=None, help="output suffix (default: r3 for R3-1, else the real run name)")
    a = ap.parse_args()
    if a.real != REAL_RUN or a.blind:
        REAL_RUN, BLIND_RUN = a.real, a.blind or f"{a.real}_blind"
        tag = a.tag or REAL_RUN
        OUT_MD, OUT_IMG = RES / f"analysis_{tag}" / "summary.md", ROOT / "img" / f"r31_info_{tag}.png"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    res = analyse(load())
    write_md(res)
    figure(res)
