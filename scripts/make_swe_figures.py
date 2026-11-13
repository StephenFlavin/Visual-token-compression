"""Figures for docs/claude/SWE_IMAGES.md from results/readhook_swe/analysis/summary.json (experiments/readhook/swe/analyse.py).

    python3 scripts/make_swe_figures.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_readme_figures as M  # noqa: E402
from PIL import Image  # noqa: E402

S = {(d["dataset"], d["arm"]): d for d in json.loads((ROOT / "results/readhook_swe/analysis/summary.json").read_text())}
SEG = [("share_output", "output (reasoning, edits)", "#6250d6"),
       ("share_start_context", "starting context, re-read each call", "#1baf7a"),
       ("share_tool_results", "tool results: file contents, command output", "#eb6834"),
       ("share_other_context", "the agent's own messages, re-read", "#3987e5")]


def save(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, format="png")
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(M.OUT / name, optimize=True)


def header(fig, H, title, sub):
    fig.text(0.02, 1 - 0.18 / H, title, ha="left", va="top", fontsize=12, fontweight="bold", color=M.INK_1)
    fig.text(0.02, 1 - 0.50 / H, sub, ha="left", va="top", fontsize=8.5, color=M.INK_2, linespacing=1.45)


def fig_anatomy():
    plt = M.chart_style()
    rows = [("django", "text", "Django, text"), ("django", "vtchook", "Django, page images"),
            ("rust/java", "text", "Rust and Java, text"), ("rust/java", "vtchook", "Rust and Java, page images")]
    rows = [r for r in rows if (r[0], r[1]) in S]
    H = 1.9 + 0.55 * len(rows)
    fig, ax = plt.subplots(figsize=(10.4, H), dpi=150)
    fig.subplots_adjust(left=0.22, right=0.86, top=1 - 1.25 / H, bottom=0.62 / H)
    for y, (ds, arm, lab) in enumerate(rows):
        d, x = S[(ds, arm)], 0.0
        for key, _, col in SEG:
            w = 100 * d[key]
            ax.barh(y, w - 0.4, left=x + 0.2, height=0.62, color=col, edgecolor="none")
            if w >= 6:
                ax.text(x + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=8.5, fontweight="bold",
                        color="#ffffff" if col in ("#6250d6", "#3987e5") else M.INK_1)
            x += w
        ax.text(101.5, y, f"${d['cost']:.2f}, {d['runs']} runs", va="center", fontsize=8.5, color=M.INK_2)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=9)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25%", "50%", "75%", "100%"])
    ax.grid(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SEG]
    fig.legend(handles, [l for _, l, _ in SEG], loc="lower left", bbox_to_anchor=(0.22, 0.0), ncol=2, fontsize=8,
               frameon=False, handlelength=1.2, columnspacing=1.5)
    header(fig, H, "Where a coding agent's money goes: output first; file contents are a small slice",
           "Share of list-price cost, SWE-bench tasks solved by Opus 5.5 (Claude Code, high effort). Only the orange "
           "slice can be shrunk by showing\nfile contents and command output as page images; it fell from "
           f"{100 * S[('django', 'text')]['share_tool_results']:.0f}% to {100 * S[('django', 'vtchook')]['share_tool_results']:.0f}% "
           "(Django), but output and re-read context grew.")
    save(fig, "swe_cost_anatomy.png")
    plt.close(fig)


def fig_bridge():
    plt = M.chart_style()
    t, v = S[("all", "text")], S[("all", "vtchook")]
    comp = lambda d, k: d[k] * d["cost"]  # noqa: E731
    steps = [("tool results\n(page images)", comp(v, "share_tool_results") - comp(t, "share_tool_results")),
             ("output\n(more reasoning)", comp(v, "share_output") - comp(t, "share_output")),
             ("agent's messages\nre-read (more calls)", comp(v, "share_other_context") - comp(t, "share_other_context")),
             ("starting context\nre-read", comp(v, "share_start_context") - comp(t, "share_start_context"))]
    # The slices are built from each call's usage record and do not sum exactly to the CLI's own cost (text arm
    # 100.1%, image arm 102.9%: analyse.py). Show the difference as its own step so that the bridge closes.
    resid = (v["cost"] - t["cost"]) - sum(d for _, d in steps)
    steps.append(("per-call records\nabove the CLI total", resid))
    H = 4.6
    fig, ax = plt.subplots(figsize=(10.4, H), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.97, top=1 - 1.25 / H, bottom=0.75 / H)
    x, level = 0, t["cost"]
    ax.bar(x, level, color="#c9c8c1", width=0.6)
    ax.text(x, level + 0.12, f"${level:.2f}", ha="center", fontsize=9, color=M.INK_1)
    labels = ["text"]
    for lab, d in steps:
        x += 1
        col = "#1baf7a" if d < 0 else "#c2493d"     # not the orange of "tool results" in the other figure
        if lab.startswith("per-call"):
            col = "#898781"
        ax.bar(x, d, bottom=level, color=col, width=0.6)
        ax.text(x, max(level, level + d) + 0.12, f"{'+' if d >= 0 else '−'}${abs(d):.2f}", ha="center", fontsize=9,
                color=M.INK_1)
        level += d
        labels.append(lab)
    x += 1
    ax.bar(x, v["cost"], color="#c9c8c1", width=0.6)
    ax.text(x, v["cost"] + 0.12, f"${v['cost']:.2f}", ha="center", fontsize=9, color=M.INK_1)
    labels.append("page images")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    lo = min(t["cost"], v["cost"]) - 2.2
    ax.set_ylim(lo, max(t["cost"], v["cost"]) + 0.8)
    ax.set_ylabel("list-price cost, all tasks ($)")
    ax.grid(axis="y", color=M.GRID, lw=0.8)
    ax.set_axisbelow(True)
    best = comp(t, "share_tool_results") * (1 - v["image_tokens"] / max(1, v["image_text_tokens"]))
    header(fig, H, "Why the total did not fall: images saved on tool results, then more output and more calls ate it",
           f"All {t['runs']} tasks (Django and Rust/Java), same tasks per arm; y axis starts at ${lo:.0f}. Green: cheaper "
           f"with images; red: dearer; grey: the slices' excess over the CLI total (see text).\nUpper bound if every tool result had been an image at the measured "
           f"{v['image_text_tokens'] / max(1, v['image_tokens']):.1f}x: −${best:.2f} "
           f"({100 * best / t['cost']:.0f}% of the text arm's cost), with everything else unchanged.")
    save(fig, "swe_cost_bridge.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_anatomy()
    fig_bridge()
    print("swe_cost_anatomy.png swe_cost_bridge.png")
