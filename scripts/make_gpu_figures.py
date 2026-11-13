"""Draw the figures for the GPU-side docs (docs/img/gpu_*.png). Needs only matplotlib.

Every number is read from the capacity-curve rows in encoder_probe/results/*/runs.jsonl (nothing is hard-coded):
1. gpu_capacity_curve.png      bits recovered per vision token against characters per vision token, frozen
                               Qwen2.5-VL-3B, natural and word-shuffled text (Run 0, encoder_probe/results/run0_cpu),
                               with the 3090's 200-page frozen baseline (r12_frozen) as open markers
2. gpu_capacity_recovered.png  the same rows as characters recovered per vision token against characters drawn
3. gpu_phase_p6.png            Run 0b at 6-px text: bits per vision token by vertical offset of the text
                               (encoder_probe/results/run0b_cpu_p6/alignment.json)

    python3 scripts/make_gpu_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "encoder_probe" / "results"
OUT = ROOT / "docs" / "img"

# two categorical slots already validated for the repo's figures (scripts/make_claude_qa_figures.py)
COL = {"natural": "#6250d6", "shuffled": "#eb6834"}
INK, MUTED, GRID = "#111827", "#6b7280", "#e5e7eb"


def rows(run: str, latin_only: bool = True) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"natural": [], "shuffled": []}
    for line in (RES / run / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("phase") != "capacity_curve" or r.get("adapter"):
            continue
        if latin_only and not r["encoding"].startswith("latin_"):
            continue
        out[r["text"]].append(r)
    for v in out.values():
        v.sort(key=lambda r: r["chars_per_token"])
    return out


def style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, color=INK, fontsize=10.5)
    ax.set_ylabel(ylabel, color=INK, fontsize=10.5)
    ax.set_title(title, loc="left", color=INK, fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5)


def curve(key: str, ci: str, fname: str, ylabel: str, title: str, diagonal: bool) -> None:
    cpu, gpu = rows("run0_cpu"), rows("r12_frozen")
    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=160)
    names = {"natural": "natural text", "shuffled": "word-shuffled text (no language prior)"}
    for text in ("shuffled", "natural"):
        rs = cpu[text]
        x = [r["chars_per_token"] for r in rs]
        y = [r[key] for r in rs]
        lo = [r[ci][0] for r in rs]
        hi = [r[ci][1] for r in rs]
        ax.fill_between(x, lo, hi, color=COL[text], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=COL[text], linewidth=2, marker="o", markersize=6, label=names[text])
        g = gpu[text]
        ax.plot([r["chars_per_token"] for r in g], [r[key] for r in g], linestyle="none", marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor=COL[text], markeredgewidth=1.8,
                label="_nolegend_")
        peak = max(rs, key=lambda r: r[key])
        ax.annotate(f"{peak[key]:.1f}", (peak["chars_per_token"], peak[key]), textcoords="offset points",
                    xytext=(10, 8), ha="left", color=INK, fontsize=10, fontweight="bold")
        last = rs[-1]
        ax.annotate(f"{last[key]:.1f}", (last["chars_per_token"], last[key]), textcoords="offset points",
                    xytext=(9, 4 if text == "natural" else -11), ha="left", color=INK, fontsize=10)
    ax.plot([], [], linestyle="none", marker="o", markersize=8, markerfacecolor="white", markeredgecolor=MUTED,
            markeredgewidth=1.8, label="open: 3090 repeat, 113-200 pages, 7-px top margin")
    if diagonal:
        ax.plot([0, 30], [0, 30], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate("every character recovered", (16.6, 16.3), color=MUTED, fontsize=9, ha="left", va="center")
    ymax = max(r[ci][1] for t in cpu.values() for r in t)
    ymax = max(ymax, max(r[key] for t in gpu.values() for r in t))
    ax.set_xlim(0, 80)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xticks([6.8, 12.2, 17.5, 28.1, 49.8, 72.2])
    ax.set_xticklabels(["6.8\n16 px", "12.2\n12 px", "17.5\n10 px", "28.1\n8 px", "49.8\n6 px", "72.2\n5 px"])
    style(ax, "characters drawn per vision token (and the line pitch of the text)", ylabel, title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)
    print("wrote", OUT / fname)


def phase() -> None:
    a = json.loads((RES / "run0b_cpu_p6" / "alignment.json").read_text())
    by = a["mean_by_shift"]                       # keys "dx,dy" -> mean bits per vision token
    dys = [7, 14, 21, 28]
    vals = {dy: [] for dy in dys}
    for k, v in by.items():
        vals[int(k.split(",")[1])].append(v)
    means = [float(np.mean(vals[dy])) for dy in dys]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
    bars = ax.bar([str(d) for d in dys], means, width=0.55, color=COL["natural"])
    for b, m in zip(bars, means):
        ax.annotate(f"{m:.1f}", (b.get_x() + b.get_width() / 2, m), textcoords="offset points", xytext=(0, 4),
                    ha="center", color=INK, fontsize=10)
    ax.set_ylim(0, max(means) * 1.15)
    style(ax, "vertical offset of the text on the page, px (28 px = one vision token)",
          "bits recovered per vision token",
          "6-px text: the vertical offset moves the reading by ~2 bits")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "gpu_phase_p6.png")
    plt.close(fig)
    print("wrote", OUT / "gpu_phase_p6.png")


def main() -> None:
    curve("bits_per_vtok", "bits_per_vtok_ci", "gpu_capacity_curve.png", "bits recovered per vision token",
          "Frozen Qwen2.5-VL-3B: a vision token delivers most at 28 characters, then collapses", False)
    curve("recovered_chars_per_token", "recovered_chars_per_token_ci", "gpu_capacity_recovered.png",
          "characters recovered per vision token",
          "Drawing more characters per token does not deliver more of them", True)
    phase()


if __name__ == "__main__":
    main()
