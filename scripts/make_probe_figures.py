"""Figures for the "real model had to read the dense codes" write-ups (encoder_probe/RESULTS.md,
docs/gpu/COLOUR_GIST.md). Written to docs/img/probe_*.png.

    python3 scripts/make_probe_figures.py

1. probe_encoders.png       error per code from raw pixels vs through frozen SigLIP and frozen DeepEncoder
2. probe_colour_layers.png  colour overlay: bits per vision token the untrained Qwen2.5-VL takes from pages
                            with 1, 2 and 3 text layers, at three text sizes

No torch, no network. Every number is read from encoder_probe/results/*/runs.jsonl; nothing is typed in except
4.33 characters per Qwen text token (measured on held-out WikiText, encoder_probe/ROUND3.md).
The other figures of this group are drawn by encoder_probe/experiments/make_figures.py (encoder_probe/img/) and
scripts/make_readme_figures.py --only charts (docs/img/readme_density.png, readme_r3_1.png).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "encoder_probe" / "results"
OUT = ROOT / "docs" / "img"
sys.path.insert(0, str(ROOT / "encoder_probe" / "experiments"))

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
PALE = "#b9b8b2"
SURFACE = "#fcfcfb"
CHARS_PER_TEXT_TOKEN = 4.33

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def rows(run: str) -> list[dict]:
    return [json.loads(line) for line in (RES / run / "runs.jsonl").read_text().splitlines() if line.strip()]


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")


def pick(rs: list[dict], **want) -> dict:
    """The last row that matches (later rows replace earlier ones, as the analysis scripts do)."""
    hit = [r for r in rs if all(r.get(k) == v for k, v in want.items())]
    if not hit:
        raise KeyError(want)
    return hit[-1]


# 1. raw pixels vs frozen encoders ---------------------------------------------------------------------------------
def fig_encoders() -> None:
    r1 = [r for run in ("main", "controls") for r in rows(run)]
    r1 = [r for r in r1 if r.get("score_v") == 2 or r["family"] != "colour"]      # colour rows before the scoring fix
    r2 = rows("round2_gpu_tp200") + rows("round2_gpu_tp1000")

    # (encoding, plain-word label)
    codes1 = [("latin_p16", "Latin\n16-px lines"), ("latin_p8", "Latin\n8-px lines"),
              ("lg_pb_h5", "learned glyphs\nwhole-pixel"), ("lg_free_h4", "learned glyphs\ngrey-level"),
              ("col_b4", "colour blocks\n4x4 px"), ("col_b1", "colour blocks\n1x1 px")]
    codes2 = [("latin_p32", "Latin\n32-px lines"), ("latin_p16", "Latin\n16-px lines"),
              ("lg_pb_h5", "learned glyphs\nwhole-pixel"), ("lg_free_h4", "learned glyphs\ngrey-level"),
              ("rgb3_p8", "RGB overlay\n3 text layers"), ("col_b2", "colour blocks\n2x2 px")]

    def cer1(enc, encoder):
        return 100 * pick(r1, encoding=enc, encoder=encoder, cond="clean")["cer"]

    def cer2(enc, encoder, pages, steps=None):
        want = dict(encoding=enc, encoder=encoder, cond="clean", train_pages=pages)
        if steps:
            want["steps"] = steps
        return 100 * pick(r2, **want)["cer"]

    panels = [
        ("Round 1: SigLIP-B/16 (224-px page, 16-px tokens)", codes1,
         [("raw pixels (no encoder)", BLUE, lambda e: cer1(e, "identity")),
          ("through SigLIP, layer 6 of 12", ORANGE, lambda e: cer1(e, "siglip_l6")),
          ("through SigLIP, last layer", YELLOW, lambda e: cer1(e, "siglip"))],
         lambda e: pick(r1, encoding=e, encoder="identity", cond="clean")["chars_per_token"]),
        ("Round 2: DeepSeek-OCR's DeepEncoder (512-px page, 64-px tokens)", codes2,
         [("raw pixels (no encoder)", BLUE, lambda e: cer2(e, "identity32", 200)),
          ("through DeepEncoder", ORANGE, lambda e: cer2(e, "deepencoder", 1000))],
         lambda e: pick(r2, encoding=e, encoder="deepencoder", cond="clean")["chars_per_vlm_token"]),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.6))
    for ax, (title, codes, series, cpt) in zip(axes, panels):
        n = len(series)
        w = 0.8 / n
        x = np.arange(len(codes))
        ax.axhspan(72, 75, color="#d2d1cb", zorder=0, label="reading nothing: about 72-75%")
        for i, (lab, col, fn) in enumerate(series):
            vals = [fn(e) for e, _ in codes]
            pos = x - 0.4 + w * (i + 0.5)
            ax.bar(pos, vals, w * 0.92, color=col, label=lab)
            for p, v in zip(pos, vals):
                ax.text(p, v + 1.5, f"{v:.0f}%" if v >= 9.5 else f"{v:.1f}%" if v >= 0.95 else f"{v:.2f}%",
                        ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab}\n{cpt(e):.0f} chars/token" if cpt(e) >= 9.5 else f"{lab}\n{cpt(e):.1f} chars/token"
                            for e, lab in codes], fontsize=8)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("character error rate (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", ncol=n + 1, bbox_to_anchor=(0, 1.0), columnspacing=1.2, handlelength=1.2)
    fig.suptitle("Every code reads from raw pixels; behind a frozen encoder only large Latin does",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.012, -0.005,
             "Same small probe and pages in each panel; 16 held-out test pages. Round 1: 200 training pages. Round 2: "
             "200 pages for raw pixels, 1000 for DeepEncoder.\nRows: encoder_probe/results/main, controls, "
             "round2_gpu_tp200, round2_gpu_tp1000.", fontsize=7.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0)
    save(fig, "probe_encoders.png")


# 2. colour overlay: information per vision token for 1, 2, 3 layers ------------------------------------------------
def fig_colour_layers() -> None:
    import analyse_colour as A

    z = rows("z0_info")

    def arr(r):
        real, blind = np.array(r["val_per_sample_real"]), np.array(r["val_per_sample_blind"])
        return blind[:, 0] - real[:, 0], real[:, 1]

    def bits(r):
        return r["gap_per_text_token"] * r["chars_per_token"] / CHARS_PER_TEXT_TOKEN / math.log(2)

    pitches = [16, 12, 8]
    cols = {1: PALE, 2: BLUE, 3: ORANGE}
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    w = 0.26
    for gi, p in enumerate(pitches):
        ref = pick(z, pitch=p, streams=1)
        for k in (1, 2, 3):
            r = pick(z, pitch=p, streams=k)
            v = bits(r)
            xpos = gi + (k - 2) * w
            ax.bar(xpos, v, w * 0.92, color=cols[k], label=f"{k} layer{'s' if k > 1 else ''}" if gi == 0 else None)
            if k == 1:
                ax.text(xpos, v + 0.25, f"{v:.1f} bits", ha="center", va="bottom", fontsize=8.5, color=INK2)
            else:
                ratio, lo, hi = A.ratio_ci(arr(r), arr(ref), r["chars_per_token"], ref["chars_per_token"])
                ax.text(xpos, v + 0.25, f"{v:.1f}\n({ratio:.2f}x)", ha="center", va="bottom", fontsize=8.5,
                        color=INK)
            ax.text(xpos, -0.35, f"{r['chars_per_token']:.0f}", ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.text(-0.42, -0.35, "chars/token:", ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(pitches)))
    ax.set_xticklabels([f"{p}-px lines" for p in pitches])
    ax.tick_params(axis="x", pad=16, length=0)
    ax.set_xlim(-0.75, len(pitches) - 0.5)
    ax.set_ylim(0, 13.5)
    ax.set_ylabel("bits per vision token taken from the page")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3, handlelength=1.2)
    ax.set_title("More text layers on a page deliver less per vision token, at every text size", loc="left")
    fig.text(0.012, -0.02,
             "Untrained Qwen2.5-VL-3B, 12 test pages per bar. Bits = (loss with another page's image - loss with the "
             "page's own image) x text tokens per vision token.\n(0.51x) = share of the 1-layer value at the same "
             "text size, the measure R(k). Rows: encoder_probe/results/z0_info.", fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    save(fig, "probe_colour_layers.png")


if __name__ == "__main__":
    fig_encoders()
    fig_colour_layers()
