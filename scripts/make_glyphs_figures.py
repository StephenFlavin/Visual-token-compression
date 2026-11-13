"""Figures for docs/glyphs/RESULTS.md and learned_glyphs/RESULTS.md (the original idea, in pixel space).

1. glyphs_px_at_3pct.png     px² per character needed for 3% character error, per reader and scheme
2. glyphs_cer_vs_density.png character error against px² per character: Latin, substitution, learned banks
3. glyphs_aligned.png        grid-aligned whole-pixel banks: characters per 28x28 vision-token area against error

Every number is read from result rows:
  results/phase1/latin_natural/runs.jsonl            Claude reading natural Latin (Phase 1)
  results/phase1b/ocr_natural/runs.jsonl             PP-OCR zero-shot (Phase 1b)
  results/phase2/merged/runs.jsonl                   CRNN trained per scheme and size (Phase 2, longest run each)
  learned_glyphs/results/e3_bridge_long/runs.jsonl   learned banks, bridge scoring
  learned_glyphs/results/e2_transfer_long/runs.jsonl learned banks, fresh reader (transfer)
  learned_glyphs/results/e3_bridge_align{,_md}/runs.jsonl  grid-aligned banks, bridge scoring
P@3% is computed exactly as vbpe.analysis does (isotonic fit, log-x interpolation).

    python3 scripts/make_glyphs_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_readme_figures as M  # noqa: E402  (house style: palette, chart_style, save_small)

from vbpe.analysis import crossing, load_runs, p_at  # noqa: E402

ROOT = M.ROOT
LG = ROOT / "learned_glyphs" / "results"
TOKEN_AREA = 28 * 28          # px² of one vision token (Claude, Qwen2.5-VL)
BLUE, ORANGE, GREEN, AMBER, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781"


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def pts(rows: list[dict], *schemes: str) -> list[tuple[float, float]]:
    return sorted((r["px_per_char"], r["cer"]) for r in rows if r["scheme"] in schemes)


def cross(points: list[tuple[float, float]], thr: float = 0.03):
    """(px²/char at CER = thr, censored flag) with the analysis module's own fit."""
    x, y = np.array([p[0] for p in points]), np.array([p[1] for p in points])
    return crossing(x, y, thr)


def data() -> dict:
    p2 = jl(ROOT / "results/phase2/merged/runs.jsonl")
    bridge = jl(LG / "e3_bridge_long/runs.jsonl")
    transfer = jl(LG / "e2_transfer_long/runs.jsonl")
    grey = ("lg_learned_n300", "lg_learned_n300_w23")     # same bank family, two cell-width settings
    claude = load_runs([ROOT / "results/phase1/latin_natural/runs.jsonl"])
    ocr = load_runs([ROOT / "results/phase1b/ocr_natural/runs.jsonl"])
    return {"p2": p2, "bridge": bridge, "transfer": transfer, "grey": grey, "claude": claude, "ocr": ocr}


def fig_px_at_3(d: dict) -> None:
    import matplotlib.pyplot as plt
    M.chart_style()
    trained = [
        ("Learned glyphs + chunks (bridge)", cross(pts(d["bridge"], *d["grey"])), GREEN),
        ("Learned glyphs + chunks (fresh reader)", cross(pts(d["transfer"], *d["grey"])), GREEN),
        ("Learned, whole-pixel (bridge)", cross(pts(d["bridge"], "lg_learned_n300_pb")), GREEN),
        ("Plain Latin, DejaVu Sans", cross(pts(d["p2"], "latin")), BLUE),
        ("Plain Latin, condensed font", cross(pts(d["p2"], "latin_condensed")), BLUE),
        ("Substitution, 300 chunks", cross(pts(d["p2"], "subst300")), ORANGE),
        ("Substitution, 1000 chunks", cross(pts(d["p2"], "subst1000")), ORANGE),
        ("Quadrant cells, 3 letters", cross(pts(d["p2"], "quad3")), AMBER),
        ("Quadrant cells, 4 letters", cross(pts(d["p2"], "quad4")), AMBER),
    ]
    zero = [
        ("Claude: plain Latin", (p_at(d["claude"], "latin", 0.03), None), BLUE),
        ("PP-OCR: plain Latin", (p_at(d["ocr"], "latin", 0.03), None), BLUE),
        ("PP-OCR: substitution, 1000 chunks", (p_at(d["ocr"], "subst1000", 0.03), None), ORANGE),
    ]
    rows = [("Reader trained per scheme", None, None)] + trained + \
           [("Zero-shot readers", None, None)] + zero
    H = 0.34 * len(rows) + 1.5
    fig, ax = plt.subplots(figsize=(8.4, H))
    fig.subplots_adjust(left=0.40, right=0.95, top=1 - 0.95 / H, bottom=0.55 / H)
    latin = trained[3][1][0]
    y = 0
    ticks, labels = [], []
    for name, val, col in rows:
        if val is None:
            import matplotlib.transforms as mt
            ax.text(0.02, y, name, transform=mt.blended_transform_factory(fig.transFigure, ax.transData), ha="left",
                    va="center", fontsize=9.5, fontweight="bold", color=M.INK_1)
        else:
            v, cens = val
            ax.barh(y, v, color=col, height=0.62)
            lab = f"{'≤ ' if cens == 'left' else ''}{v:.1f}" if v < 20 else f"{v:.1f}"
            if name.startswith(("Learned", "Plain Latin, c", "Subst", "Quad")):
                lab += f"   ({v / latin:.2f}x Latin)"
            ax.text(v + 0.5, y, lab, va="center", ha="left", fontsize=8.5, color=M.INK_2)
            ticks.append(y)
            labels.append(name)
        y += 1
    ax.set_yticks(ticks, labels, fontsize=9)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, 56)
    ax.set_xlabel("px² per character needed for 3% character error (lower is denser)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=M.GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.text(0.02, 1 - 0.18 / H, "Only the learned alphabet needs fewer pixels than plain Latin", ha="left",
             va="top", fontsize=12, fontweight="bold", color=M.INK_1)
    fig.text(0.02, 1 - 0.50 / H, "Hand-picked glyphs (substitution, quadrant cells) need more. "
             "'≤' = already under 3% at the smallest size trained.", ha="left", va="top", fontsize=8.5, color=M.INK_2)
    M.save_small(fig, "glyphs_px_at_3pct.png")
    plt.close(fig)
    print("glyphs_px_at_3pct:", [(n, v and (round(v[0], 2), v[1])) for n, v, _ in rows])


def fig_curves(d: dict) -> None:
    import matplotlib.pyplot as plt
    M.chart_style()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    fig.subplots_adjust(left=0.10, right=0.80, top=0.84, bottom=0.12)
    series = [
        ("Plain Latin", pts(d["p2"], "latin"), BLUE, "o", "-"),
        ("Substitution,\n300 chunks", pts(d["p2"], "subst300"), ORANGE, "s", "-"),
        ("Learned,\nwhole-pixel", pts(d["bridge"], "lg_learned_n300_pb"), GREEN, "D", "--"),
        ("Learned glyphs\n+ chunks", pts(d["bridge"], *d["grey"]), GREEN, "D", "-"),
    ]
    for name, p, col, mk, ls in series:
        p = [q for q in p if q[0] <= 24]
        x, yv = [q[0] for q in p], [100 * q[1] for q in p]
        ax.plot(x, yv, ls, color=col, marker=mk, ms=5, lw=1.6, mfc=col if ls == "-" else "white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(3, color=M.MUTED_C, lw=0.8, ls=":")
    ax.text(2.45, 3.25, "3% error", fontsize=8.5, color=M.INK_2, va="bottom")
    # direct labels at each line's last point
    ends = {s[0]: [q for q in s[1] if q[0] <= 24][-1] for s in series}
    ax.text(23.5, 0.33, "Plain Latin", color=BLUE, fontsize=9, va="center")
    ax.text(23.6, 1.05, "Substitution,\n300 chunks", color=ORANGE, fontsize=9, va="center")
    e = ends["Learned,\nwhole-pixel"]
    e = pts(d["bridge"], "lg_learned_n300_pb")[0]
    ax.annotate("Learned, whole-pixel", (e[0], 100 * e[1]), xytext=(10, 0), textcoords="offset points", color=GREEN,
                fontsize=9, va="center")
    e = ends["Learned glyphs\n+ chunks"]
    ax.annotate("Learned glyphs\n+ chunks", (e[0], 100 * e[1]), xytext=(-8, -6), textcoords="offset points",
                color=GREEN, fontsize=9, va="top", ha="right")
    ax.set_xticks([2.5, 4, 6, 8, 12, 16, 24], ["2.5", "4", "6", "8", "12", "16", "24"])
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30], ["0.1%", "0.3%", "1%", "3%", "10%", "30%"])
    ax.minorticks_off()
    ax.set_xlim(2.3, 25)
    ax.set_xlabel("px² per character (lower is denser)")
    ax.set_ylabel("character error rate")
    ax.grid(color=M.GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.text(0.02, 0.965, "Learned glyphs keep reading where Latin text has broken down", ha="left", va="top",
             fontsize=12, fontweight="bold", color=M.INK_1)
    fig.text(0.02, 0.905, "Same reader and pipeline for every line (Phase 2 CRNN, trained per point; learned banks "
             "frozen = 'bridge').", ha="left", va="top", fontsize=8.5, color=M.INK_2)
    M.save_small(fig, "glyphs_cer_vs_density.png")
    plt.close(fig)
    print("glyphs_cer_vs_density:", {s[0].replace("\n", " "): [(round(a, 2), round(100 * b, 2)) for a, b in s[1]]
                                     for s in series})


def fig_aligned(d: dict) -> None:
    import matplotlib.pyplot as plt
    M.chart_style()
    al = jl(LG / "e3_bridge_align/runs.jsonl") + jl(LG / "e3_bridge_align_md/runs.jsonl")
    name = {"lg_learned_n0_pb_c7": ("characters 7x7 px", (8, 0), "left"),
            "lg_learned_n0_pb_c4": ("characters 4x7 px", (8, 0), "left"),
            "lg_learned_n0": ("characters 2x7 px", (9, -9), "left"),
            "lg_learned_n300_pb_c2t4": ("characters 2x7 +\nchunks 4x7 px", (8, 6), "left"),
            "lg_learned_n0_pb_c2md": ("2x7, no two glyphs alike", (-8, -2), "right"),
            "lg_learned_n300_pb_c2t4md": ("with chunks,\nno two glyphs alike", (8, -6), "left")}
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.84, bottom=0.12)
    lat = [(TOKEN_AREA / x, 100 * y) for x, y in pts(d["p2"], "latin") if TOKEN_AREA / x >= 14]
    ax.plot([p[0] for p in lat], [p[1] for p in lat], "-o", color=BLUE, ms=5, lw=1.6)
    ax.annotate("Plain Latin", (lat[0][0], lat[0][1]), xytext=(-6, 8), textcoords="offset points", color=BLUE,
                fontsize=9, ha="right")
    out = {}
    for r in al:
        x, y = TOKEN_AREA / r["px_per_char"], 100 * r["cer"]
        lab, off, ha = name[r["scheme"]]
        md = r["scheme"].endswith("md")
        ax.plot([x], [y], "D", color=GREEN, ms=7, mfc="white" if md else GREEN, mew=1.6)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", color=GREEN, fontsize=8.5, ha=ha, va="center")
        out[r["scheme"]] = (round(x, 1), round(y, 3), r["steps"])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([16, 28, 40, 56, 76, 110, 140], ["16", "28", "40", "56", "76", "110", "140"])
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1, 3], ["0.01%", "0.03%", "0.1%", "0.3%", "1%", "3%"])
    ax.minorticks_off()
    ax.set_xlim(13, 160)
    ax.set_ylim(0.007, 7)
    ax.set_xlabel("characters per 28x28-px vision token (784 / px² per character; higher is denser)")
    ax.set_ylabel("character error rate")
    ax.grid(color=M.GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.text(0.02, 0.965, "Glyphs made of whole pixels on a 7-px grid: 76 characters per token at 0.1% error",
             ha="left", va="top", fontsize=12, fontweight="bold", color=M.INK_1)
    fig.text(0.02, 0.905, "Read from raw pixels by the Phase 2 reader (bridge). Hollow marks: banks retrained so "
             "that no two glyphs differ by less than 2 px.", ha="left", va="top", fontsize=8.5, color=M.INK_2)
    M.save_small(fig, "glyphs_aligned.png")
    plt.close(fig)
    print("glyphs_aligned (chars/token, CER %, steps):", out, "latin:", [(round(a, 1), round(b, 2)) for a, b in lat])


if __name__ == "__main__":
    M.OUT.mkdir(parents=True, exist_ok=True)
    D = data()
    fig_px_at_3(D)
    fig_curves(D)
    fig_aligned(D)
