"""Figures for encoder_probe/RESULTS.md and ROUND3.md, drawn from the committed rows files.

    python3 encoder_probe/experiments/make_figures.py        # writes encoder_probe/img/*.png

No torch, no network. Every number is read from encoder_probe/results/*/(rows|runs).jsonl; the only values typed in
here are the lg_pb_h5 blind val loss at step 500 (its job died before writing a row; logged in docs/HANDOFF.md) and
the MC chance level (4 options).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "img"

# Reference categorical palette (validated, light surface), then neutrals for controls / floors.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLIND, BLIND_DARK = "#b9b8b2", "#6f6e69"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8,
})


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{p.relative_to(ROOT.parent)}  {p.stat().st_size / 1024:.0f} KB")


# (a) Q1: linear ink readout of frozen Qwen ViT states, by depth --------------------------------------------------
def fig_q1() -> None:
    feats = ["b8", "b16", "b24", "b28", "final", "merger"]
    codes = [  # (encoding, colour, linestyle, marker): glyph codes in colour, Latin controls in grey
        ("lg_pb_h5", BLUE, "-", "o"),
        ("rgb3_lg_pb_h5", ORANGE, "-", "s"),
        ("rgb3_p8", YELLOW, "-", "^"),
        ("rgb3_lg_free_h4", AQUA, "-", "D"),
        ("latin_p16", BLIND_DARK, "--", "o"),
        ("latin_p8", BLIND, "--", "s"),
    ]
    q1 = [r for r in rows(RES / "quick_q1" / "rows.jsonl") if r.get("v") == 2 and r.get("S") == 448]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.axvspan(4.5, 5.5, color=GRID, alpha=0.6, lw=0)
    ax.text(5.0, 1.04, "2x2 merger", ha="center", va="bottom", fontsize=8, color=INK2)
    for enc, col, ls, mk in codes:
        by = {r["feature"]: r for r in q1 if r["encoding"] == enc}
        y = [by[f]["ink_mcc_mean"] for f in feats]
        cpt = by[feats[0]]["chars_per_qwen_token"]
        ax.plot(range(len(feats)), y, ls=ls, marker=mk, ms=5, lw=2, color=col,
                label=f"{enc} ({cpt:.0f} c/tok)")
    ax.set_xticks(range(len(feats)), ["block 8", "block 16", "block 24", "block 28", "final", "after\nmerger"])
    ax.set_ylim(-0.03, 1.1)
    ax.set_ylabel("ink MCC of a linear pixel readout\n(0 = chance, 1 = exact)")
    ax.set_title("Q1: glyph codes stay pixel-like through Qwen's ViT; all collapse at the merger", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="code (chars/Qwen token)", title_fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.text(0.0, -0.06, "Frozen Qwen2.5-VL-3B ViT, ridge readout, 64 train / 16 test pages; rgb3 codes: mean of R/G/B. "
             "Latin (dashed) is read by Qwen\nyet loses its linear readout by block 24, so a low value is not "
             "evidence of loss. Rows: results/quick_q1/rows.jsonl.", fontsize=7, color=INK2, ha="left", va="top")
    save(fig, "q1_ink_mcc.png")


# (b) G2-lite: moonshot through frozen / trained / no merger, against the blind floor -------------------------------
def edit_ops(pairs) -> dict:
    """Same op split as ep/g2lite.py::edit_ops (kept here so this script needs no torch)."""
    from rapidfuzz.distance import Levenshtein
    sub = ins = dele = nr = nh = 0
    for r, h in pairs:
        for tag, *_ in Levenshtein.editops(r, h):
            sub += tag == "replace"
            ins += tag == "insert"
            dele += tag == "delete"
        nr += len(r)
        nh += len(h)
    n = max(1, nr)
    return {"cer": (sub + ins + dele) / n, "sub": sub / n, "ins": ins / n, "del": dele / n, "len_ratio": nh / n}


def correct(ops: dict) -> float:
    """Share of reference chars the alignment matches: (hyp length - sub - ins) / ref chars."""
    return ops["len_ratio"] - ops["sub"] - ops["ins"]


def fig_g2lite() -> None:
    enc = "rgb3_lg_pb_h5"
    base = {r["arm"]: r for r in rows(RES / "g2lite" / "runs.jsonl")
            if r["encoding"] == enc and (r["arm"] != "trained" or r.get("merger_lr") == 1e-4)}
    ops = {r["arm"]: r for r in rows(RES / "g2lite_ops" / "runs.jsonl") if r["encoding"] == enc}
    blind = next(r for r in rows(RES / "g2lite_blind" / "runs.jsonl") if r["encoding"] == enc)
    arms = []  # (label, cer, correct, note, is_blind)
    fr = base["frozen"]  # no ops row: op split from its 3 stored test samples
    arms.append(("frozen merger", fr["cer"], correct(edit_ops(fr["samples"])), "correct: 3 samples", False))
    for arm, lab in [("trained", "trained merger\n(LR 1e-4)"), ("direct", "no merger\n(direct)")]:
        arms.append((lab, ops[arm]["cer"], correct(ops[arm]["ops"]), "", False))
    arms.append(("blind floor\n(1500 steps)", blind["cer"], correct(blind["ops"]), "", True))

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    ylab = [a[0] for a in arms][::-1]
    for ax, idx, title, xlab in [(axes[0], 1, "CER (lower is better)", "CER on 16 test pages"),
                                 (axes[1], 2, "Reference chars matched (higher is better)", "share of ref chars")]:
        vals = [a[idx] * 100 for a in arms][::-1]
        cols = [BLIND if a[4] else BLUE for a in arms][::-1]
        ax.barh(ylab, vals, color=cols, height=0.55, edgecolor=SURFACE, linewidth=2)
        for y, v in enumerate(vals):
            ax.text(v + 1.5, y, f"{v:.1f}%", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 112 if idx == 1 else 40)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel(xlab)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[1].text(arms[0][2] * 100 + 7.5, 3, "(3 samples)", va="center", fontsize=7, color=INK2)
    fig.suptitle("G2-lite, moonshot rgb3_lg_pb_h5 (322 chars/token): only bypassing the merger reads more than "
                 "label statistics", x=0.0, ha="left", fontsize=10, fontweight="bold", y=1.04)
    fig.text(0.0, -0.08, "Frozen Qwen ViT, GridProbe (CTC + pixel reconstruction), 500 train pages, 3000 steps, CPU. "
             "Rows: results/g2lite{,_ops,_blind}/.", fontsize=7, color=INK2, ha="left", va="top")
    fig.tight_layout()
    save(fig, "g2lite_moonshot.png")


# (c) R3-1: each fine-tuned arm against its own blind control -------------------------------------------------------
LG_PB_H5_BLIND_VAL_LOSS_500 = 2.285  # blind job died at step 500 (disk full), no row; docs/HANDOFF.md


def fig_r31() -> None:
    real = {r["encoding"]: r for r in rows(RES / "round3_ft_gpu3h" / "runs.jsonl") if r["cond"] == "clean"}
    blind = {r["encoding"]: r for r in rows(RES / "round3_ft_gpu3h_blind" / "runs.jsonl") if r["cond"] == "clean"}
    arms = ["latin_p8", "latin_p5", "lg_pb_h5", "rgb3_lg_pb_h5"]
    labels = [f"{a}\n{real[a]['chars_per_token']:.0f} c/tok" for a in arms]
    metrics = [("CER (lower is better)", lambda r: r["cer"] * 100, "{:.1f}", (0, 105)),
               ("multiple choice acc. (higher)", lambda r: r["mc_acc"], "{:.2f}", (0, 1.05)),
               ("final val loss (lower)", lambda r: r["curve"][-1][2], "{:.2f}", (0, 2.9))]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    w = 0.38
    for ax, (title, get, fmt, ylim) in zip(axes, metrics):
        for i, a in enumerate(arms):
            vr = get(real[a])
            if a in blind:
                vb, hatch = get(blind[a]), None
            elif title.startswith("final val loss") and a == "lg_pb_h5":
                vb, hatch = LG_PB_H5_BLIND_VAL_LOSS_500, "////"
            else:
                vb, hatch = None, None
            ax.bar(i - w / 2, vr, w, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
                   label="fine-tuned" if i == 0 else None)
            ax.text(i - w / 2, vr + ylim[1] * 0.015, fmt.format(vr), ha="center", va="bottom", fontsize=6.5)
            if vb is None:
                ax.text(i + w / 2, ylim[1] * 0.02, "blind\nnot run", ha="center", va="bottom", fontsize=6,
                        color=INK2)
            else:
                ax.bar(i + w / 2, vb, w, color=BLIND, edgecolor=SURFACE if hatch is None else BLIND_DARK,
                       linewidth=1.5 if hatch is None else 0.0, hatch=hatch,
                       label="blind control" if i == 0 else None)
                ax.text(i + w / 2, vb + ylim[1] * 0.015, fmt.format(vb) + ("*" if hatch else ""), ha="center",
                        va="bottom", fontsize=6.5, color=INK2)
        if title.startswith("multiple"):
            ax.axhline(0.25, color=BLIND_DARK, lw=1, ls=":")
            ax.text(3.65, 0.25, "chance", ha="left", va="center", fontsize=7, color=INK2, clip_on=False)
        ax.set_xticks(range(len(arms)), labels, fontsize=7)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(1.0, 1.05), ncol=2,
               fontsize=8)
    fig.suptitle("R3-1 (LoRA fine-tune, LLM as reader, 1000 steps): only latin_p8 beats its blind control",
                 x=0.0, ha="left", fontsize=10, fontweight="bold", y=1.03)
    fig.text(0.0, -0.04, "Qwen2.5-VL-3B, LoRA r=16 + merger + last 4 ViT blocks, 1000 train / 16 test pages, clean "
             "pages, RTX 4090. Blind control = same recipe, each page's image swapped for another page's.\n"
             "* lg_pb_h5 blind died at step 500 (full disk): val loss at step 500, no CER/MC row. "
             "Rows: results/round3_ft_gpu3h{,_blind}/.", fontsize=7, color=INK2, ha="left", va="top")
    fig.tight_layout()
    save(fig, "r31_vs_blind.png")


# (d) Round 2: Latin through DeepEncoder vs training pages ---------------------------------------------------------
def fig_round2_data() -> None:
    pts: dict[tuple, dict[int, float]] = {}
    for tp in (200, 1000, 3000, 5000):
        for r in rows(RES / f"round2_gpu_tp{tp}" / "runs.jsonl"):
            if r["encoder"] == "deepencoder" and r["cond"] == "clean" and r["encoding"] in ("latin_p32", "latin_p16"):
                pts.setdefault((r["encoding"], r["steps"]), {})[r["train_pages"]] = r["cer"] * 100
    series = [(("latin_p32", 10000), "latin_p32 (8.5 c/tok), 10k steps", BLUE, "-", "o"),
              (("latin_p32", 3000), "latin_p32, 3k steps", BLUE, "--", "o"),
              (("latin_p16", 10000), "latin_p16 (36 c/tok), 10k steps", ORANGE, "-", "s"),
              (("latin_p16", 3000), "latin_p16, 3k steps", ORANGE, "--", "s")]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.axhspan(72, 75, color=GRID, alpha=0.8, lw=0)
    ax.text(5600, 73.5, "unigram floor", va="center", ha="left", fontsize=7, color=INK2)
    for key, lab, col, ls, mk in series:
        xs = sorted(pts[key])
        ys = [pts[key][x] for x in xs]
        ax.plot(xs, ys, ls=ls, marker=mk, ms=5, lw=2, color=col, label=lab, mfc=col if ls == "-" else SURFACE)
    for x in sorted(pts[("latin_p32", 10000)]):
        v = pts[("latin_p32", 10000)][x]
        ax.text(x, v - 6, f"{v:.1f}%", ha="center", va="top", fontsize=7, color=INK)
    v = pts[("latin_p32", 3000)][200]
    ax.text(200 * 1.12, v, f"{v:.1f}%", ha="left", va="center", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([200, 1000, 3000, 5000], ["200", "1000", "3000", "5000"])
    ax.minorticks_off()
    ax.set_xlim(150, 9000)
    ax.set_ylim(0, 95)
    ax.set_xlabel("probe training pages (log scale)")
    ax.set_ylabel("test CER (%)")
    ax.set_title("Round 2: Latin through frozen DeepEncoder is data-limited, never reaches 10%", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    ax.grid(axis="x", visible=False)
    fig.text(0.0, -0.07, "DeepEncoder (Tiny, 512 px, 64 tokens), clean pages, same probe; RTX 3090. "
             "Raw-pixel ceilings (identity32): 0.20% / 0.06%. Rows: results/round2_gpu_tp*/.",
             fontsize=7, color=INK2, ha="left", va="top")
    save(fig, "round2_latin_data.png")


if __name__ == "__main__":
    fig_q1()
    fig_g2lite()
    fig_r31()
    fig_round2_data()
