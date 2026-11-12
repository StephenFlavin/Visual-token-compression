#!/usr/bin/env python3
"""Figures for docs/ (docs/img/ctx_*.png). Run: python3 scripts/make_docs_figures.py

ctx_budget.png: chars that fit in Qwen2.5-VL-3B's 32k-position window per encoding (log x), potential vs measured.
Densities are chars per Qwen vision token at 448 px / 256 tokens (rows in encoder_probe/results/; latin_p6, col_b4
and rgb3_p8 from docs/claude/CONTEXT_BUDGET.md). Status is the R3-1 short-budget result (1000 steps, RTX 4090, 2026-11-05,
encoder_probe/results/round3_ft_gpu3h{,_blind}/).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"

WINDOW = 32768          # Qwen2.5-VL-3B native window (KV positions)
TEXT_CPT = 4.33         # plain-text chars per token, measured on held-out WikiText

# palette (dataviz reference instance, light mode)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE = "#e1e0d9", "#c3c2b7", "#2a78d6"

# (label, chars/token, status, note); status: text | read | floor | notrun
ROWS = [
    ("Plain text tokens", TEXT_CPT, "text", "default"),
    ("latin_p8", 28.1, "read", "read: 55% CER (blind 88.5%), MC 0.88 (0.25)"),
    ("col_b4 (colour blocks)", 49, "notrun", "dead in frozen encoders; not fine-tuned"),
    ("latin_p6", 51, "notrun", "not run yet"),
    ("latin_p5", 72.4, "floor", "at its blind floor (80.9% CER, blind 77.8%)"),
    ("rgb3_p8 (RGB overlay)", 72.8, "notrun", "not run yet"),
    ("lg_pb_h5 (learned glyphs)", 107.3, "floor", "at its blind floor (val loss = blind's)"),
    ("rgb3_lg_pb_h5 (moonshot)", 321.6, "floor", "at its blind floor (loss = blind's)"),
    ("rgb3_lg_free_h4", 405.5, "notrun", "dropped after Q1"),
]


def fmt_chars(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def ctx_budget():
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    text_chars = TEXT_CPT * WINDOW
    ys = list(range(len(ROWS)))[::-1]
    for y, (label, cpt, status, note) in zip(ys, ROWS):
        x = cpt * WINDOW
        if status == "text":
            ax.plot([x], [y], "s", ms=9, color=INK, mec=SURFACE, mew=2, zorder=3)
        else:
            ax.plot([text_chars, x], [y, y], color=GRID, lw=2, solid_capstyle="round", zorder=1)
            face = {"read": BLUE, "floor": SURFACE, "notrun": SURFACE}[status]
            edge = {"read": BLUE, "floor": BLUE, "notrun": MUTED}[status]
            ax.plot([x], [y], "o", ms=10, mfc=face, mec=edge, mew=2, zorder=3)
        mult = "" if status == "text" else f"  {cpt / TEXT_CPT:.1f}x"
        ax.annotate(f"{fmt_chars(x)}{mult}  ·  {note}", (x, y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK if status in ("text", "read") else INK2)
    ax.axvline(text_chars, color=AXIS, lw=1, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]}  ({r[1]:.0f})" if r[2] != "text" else f"{r[0]}  (4.3)" for r in ROWS],
                       fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(8e4, 3e8)
    ticks = [1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(["100k", "300k", "1M", "3M", "10M", "30M", "100M"], fontsize=9, color=INK2)
    ax.set_xlabel("characters that fit in Qwen2.5-VL-3B's 32k-position window (log scale)", fontsize=9, color=INK2)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_ylim(-0.7, len(ROWS) - 0.3)
    fig.suptitle("Context potential per encoding vs what fine-tuned Qwen reads so far", x=0.02, ha="left",
                 fontsize=11.5, color=INK, fontweight="bold")
    ax.set_title("Encoding (chars per vision token). Potential assumes perfect reading. R3-1: LoRA, 1000 steps, "
                 "each arm vs its own blind control", loc="left", fontsize=8.5, color=INK2, pad=8)
    legend = [
        Line2D([], [], ls="", marker="s", ms=8, color=INK, label="plain text (baseline)"),
        Line2D([], [], ls="", marker="o", ms=8, mfc=BLUE, mec=BLUE, mew=2, label="read by fine-tuned Qwen (well below its blind control)"),
        Line2D([], [], ls="", marker="o", ms=8, mfc=SURFACE, mec=BLUE, mew=2, label="fine-tuned, at its blind floor (not read yet)"),
        Line2D([], [], ls="", marker="o", ms=8, mfc=SURFACE, mec=MUTED, mew=2, label="potential only (not fine-tuned)"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(-0.02, -0.16), ncol=2, frameon=False, fontsize=8.5,
              handletextpad=0.4, columnspacing=2.0, labelcolor=INK2)
    fig.tight_layout()
    out = OUT / "ctx_budget.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ctx_budget()
