"""Overview figures for docs/claude/CLAUDE_QA.md, drawn from the result rows and summaries (no hard-coded results).

    python3 scripts/make_claude_docs_figures.py

1. claude_uses_overview.png   every tested use of page images on Claude: accuracy change and cost change
2. claude_repeat_calls.png    wording-sensitive questions: accuracy of each repeat call, pages vs text
3. claude_prune_vs_pages.png  pruned text vs pages: accuracy against input tokens
4. claude_gradient.png        graded resolution (old third dense, recent third sharp, and the reverse): lookups by
                              third and overall accuracy against input tokens
5. claude_salient.png         content-chosen mixed resolution (facts large, filler small): overall accuracy against
                              input tokens for every arm, and lookups / multi-step at the token count of uniform 58.8

Sources: results/claude_qa/analysis_v2*/summary.json and the v2 runs (make_claude_qa_figures.costs),
results/claude_qa/{research,prune/analysis,repeat/analysis}/summary.json, results/rag/summary.json,
results/compaction_scale/summary.json, results/readhook_swe/analysis/summary.json,
results/claude_qa/gradient/summary.json, results/claude_qa/salient/summary.json.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_readme_figures as M  # noqa: E402
import make_claude_qa_figures as Q  # noqa: E402
from PIL import Image  # noqa: E402

RES = ROOT / "results"
GOOD, BAD, NEUTRAL, TEXT_C, PRUNE_C = "#1baf7a", "#c2493d", "#898781", "#52514e", "#b7791f"
PAGE_C = "#3987e5"


def load(rel: str):
    return json.loads((RES / rel).read_text())


def save(fig, name: str) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, format="png")
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(M.OUT / name, optimize=True)


def header(fig, H: float, title: str, sub: str) -> None:
    fig.text(0.02, 1 - 0.18 / H, title, ha="left", va="top", fontsize=12, fontweight="bold", color=M.INK_1)
    fig.text(0.02, 1 - 0.50 / H, sub, ha="left", va="top", fontsize=8.5, color=M.INK_2, linespacing=1.45)


# ------------------------------------------------------------------------------------------- 1. overview
def qa_row(model: str, suffix: str, budget: float):
    """(accuracy change in points with its interval, cost change in %) for one model at high effort."""
    s = load(f"claude_qa/analysis_v2{suffix}/summary.json")[f"high/{budget}"]
    d, lo, hi = s["text_minus_image"][:3]
    c = Q.costs(model, "high")
    return (-100 * d, -100 * hi, -100 * lo), 100 * (c[budget] / c["text"] - 1)


def overview_rows():
    rows = []           # (group, label, (diff, lo, hi) or (diff, None, None), cost change %, note)
    for label, model, suf, b in [("Long-document Q&A, Opus, 2.2x fewer tokens", "Opus 5.5", "", 117.6),
                                 ("Long-document Q&A, Opus, 2.9x", "Opus 5.5", "", 88.2),
                                 ("Long-document Q&A, Sonnet, 2.2x", "Sonnet 5.5", "_sonnet", 117.6)]:
        acc, cost = qa_row(model, suf, b)
        rows.append(("works", label, acc, cost, "360 questions"))
    r = load("claude_qa/research/summary.json")
    d = r["pages"]["diff"]["all"]
    rows.append(("works", "Deep-research sources, 2.2x", tuple(100 * d[k] for k in ("arm_minus_text", "lo", "hi")),
                 100 * (r["pages"]["cost"] / r["text"]["cost"] - 1), "360 questions"))
    g = load("rag/summary.json")["multihop"]
    for key, label in (("P5", "RAG: the same 5 passages as pages"), ("Pmatch", "RAG: 14 passages for the price of 5")):
        m = re.match(r"([+-][\d.]+) \(([+-][\d.]+) to ([+-][\d.]+)\)", g[key]["vs_T5"])
        rows.append(("works", label, tuple(float(x) for x in m.groups()),
                     100 * (g[key]["cost_per_q"] / g["T5"]["cost_per_q"] - 1), "MultiHop-RAG, 148 questions"))
    c = load("compaction_scale/summary.json")
    d = c["diffs"]["B-A"]
    rows.append(("works", "Compaction: page archive vs normal compaction", (100 * d["diff"], 100 * d["lo"], 100 * d["hi"]),
                 100 * (c["arms"]["B"]["after_reading_cost"] / c["arms"]["A"]["after_reading_cost"] - 1),
                 "9 sessions, 360 questions"))
    s = {(x["dataset"], x["arm"]): x for x in load("readhook_swe/analysis/summary.json")}
    t, v = s[("all", "text")], s[("all", "vbhook")]
    rows.append(("works", "Coding agent: files and output as pages", (100 * (v["resolved"] - t["resolved"]) / t["runs"], None, None),
                 100 * (v["cost"] / t["cost"] - 1), f"{t['runs']} SWE-bench tasks solved"))
    rp = load("claude_qa/repeat/analysis/summary.json")
    pr = load("claude_qa/prune/analysis/summary.json")["arms"]
    d = rp["page118"]["diff_small"]
    rows.append(("fails", "Wording-sensitive questions, 2.2x", (100 * d["point"], 100 * d["lo"], 100 * d["hi"]),
                 100 * (pr["page118"]["cost_usd"] / pr["text"]["cost_usd"] - 1), "15 calls x 40 questions"))
    acc, cost = qa_row("Opus 5.5", "", 58.8)
    rows.append(("fails", "Denser pages: Opus, 4.2x", acc, cost, "360 questions"))
    acc, cost = qa_row("Haiku 5.5", "_haiku", 117.6)
    rows.append(("fails", "Haiku, 2.2x", acc, cost, "360 questions"))
    d = pr["ptext22"]["diff_all"]
    rows.append(("fails", "Pruned text at the 2.2x pages' token count", (100 * d["arm_minus_text"], 100 * d["lo"], 100 * d["hi"]),
                 100 * (pr["ptext22"]["cost_usd"] / pr["text"]["cost_usd"] - 1), "480 questions"))
    return rows


def fig_overview():
    plt = M.chart_style()
    rows = overview_rows()
    n = len(rows)
    H = 1.75 + 0.42 * n + 0.35
    fig, (axa, axc) = plt.subplots(1, 2, figsize=(10.4, H), dpi=150, sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.375, right=0.975, top=1 - 1.5 / H, bottom=0.55 / H, wspace=0.06)
    ys, y, prev = [], 0.0, None
    for grp, *_ in rows:
        if prev is not None and grp != prev:
            y += 0.7
        ys.append(y)
        y += 1
        prev = grp
    for yy, (grp, label, (d, lo, hi), cost, note) in zip(ys, rows):
        col = GOOD if grp == "works" else BAD
        if lo is not None:
            axa.plot([lo, hi], [yy, yy], color=col, lw=2.2, alpha=0.45, solid_capstyle="round", zorder=2)
        axa.scatter([d], [yy], s=46, color=col, zorder=3, edgecolor=M.SURFACE, lw=1)
        edge = hi if hi is not None else d
        txt = "0.0" if abs(d) < 0.05 else f"{d:+.1f}".replace("-", "−")
        if edge < 20:
            axa.text(edge + 0.9, yy, txt, va="center", ha="left", fontsize=8.5, color=M.INK_1)
        else:
            axa.text((lo if lo is not None else d) - 0.9, yy, txt, va="center", ha="right", fontsize=8.5, color=M.INK_1)
        axc.barh(yy, cost, height=0.55, color=GOOD if cost < -5 else NEUTRAL, zorder=2)
        axc.text(cost - 1.5 if cost < 0 else cost + 1.5, yy, f"{cost:+.0f}%".replace("-", "−"), va="center",
                 ha="right" if cost < 0 else "left", fontsize=8.5, color=M.INK_1)
        axa.text(-0.02, yy - 0.13, label, transform=axa.get_yaxis_transform(), ha="right", va="center", fontsize=8.8,
                 color=M.INK_1)
        axa.text(-0.02, yy + 0.27, note, transform=axa.get_yaxis_transform(), ha="right", va="center", fontsize=7.3,
                 color=M.MUTED_C)
    first_bad = next(i for i, r in enumerate(rows) if r[0] == "fails")
    for ax in (axa, axc):
        ax.axhline((ys[first_bad - 1] + ys[first_bad]) / 2, color=M.AXIS, lw=0.8, ls=(0, (3, 3)))
        ax.axvline(0, color=M.INK_2, lw=1)
        ax.grid(axis="x", color=M.GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelleft=False)
        ax.spines["left"].set_visible(False)
    axa.axvspan(-2, 2, color=GOOD, alpha=0.08, lw=0, zorder=0)
    axa.set_xlim(-19, 24)
    axc.set_xlim(-75, 12)
    axa.set_ylim(ys[-1] + 0.65, -0.65)
    axa.set_xlabel("accuracy change, points (pages − baseline)")
    axc.set_xlabel("list-price cost change")
    axa.set_title("Accuracy", fontsize=9.5, fontweight="bold", loc="left", color=M.INK_1)
    axc.set_title("Cost", fontsize=9.5, fontweight="bold", loc="left", color=M.INK_1)
    header(fig, H, "Page images on Claude: what each tested use gained or lost",
           "Each row compares page images with the same model reading text (the compaction row: with normal compaction), "
           "Opus 5.5 at high effort unless named.\nDot = accuracy change with its 95% interval; the green band is the "
           "2-point margin. Bar = change in list-price cost. Above the dashed line the use works;\nbelow it, it loses "
           "accuracy.")
    save(fig, "claude_uses_overview.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- 2. repeat calls
def fig_repeat():
    plt = M.chart_style()
    s = load("claude_qa/repeat/analysis/summary.json")
    arms = [("text", "text", TEXT_C), ("page118", "pages, 2.2x fewer tokens\n(117.6 px² per character)", PAGE_C),
            ("page88", "pages, 2.9x fewer tokens\n(88.2 px² per character)", PAGE_C)]
    H = 4.3
    fig, ax = plt.subplots(figsize=(10.4, H), dpi=150)
    fig.subplots_adjust(left=0.22, right=0.97, top=1 - 1.3 / H, bottom=0.62 / H)
    flips = pages = 0
    for row, (key, label, col) in enumerate(arms):
        calls = [c for doc in sorted(s[key]["per_call"]) for c in s[key]["per_call"][doc]]
        stack: dict[float, int] = {}
        for c in sorted(calls, key=lambda c: c["passive_right"]):
            x = round(100 * c["small"], 1)
            k = stack.get(x, 0)
            stack[x] = k + 1
            flip = c["passive_right"] == 0
            if key != "text":
                pages += 1
                flips += flip
            off = (k % 4) * 0.42 - 0.63, (k // 4) * 0.19
            ax.scatter([x + off[0]], [row - 0.28 + off[1]], s=58, color=BAD if flip else col, marker="X" if flip else "o",
                       edgecolor=M.SURFACE, lw=0.8, zorder=3)
        m = 100 * s[key]["mean_small"]
        ax.plot([m, m], [row - 0.42, row + 0.42], color=M.INK_1, lw=1.4, zorder=2)
        ax.text(-0.02, row, f"{label}\n{len(calls)} calls, mean {m:.1f}%", transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.8, color=M.INK_1, linespacing=1.35)
    ax.set_ylim(len(arms) - 0.3, -0.65)
    ax.set_xlim(83, 102)
    ax.set_xticks([85, 87.5, 90, 92.5, 95, 97.5, 100], ["85%", "87.5", "90%", "92.5", "95%", "97.5", "100%"])
    ax.set_xlabel("wording-sensitive questions right in one call (40 questions per call)")
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=M.GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.text(88.6, 1.5, f"✕ = a call that read every passive sentence\nbackwards ({flips} of {pages} page calls, "
            f"{s['text']['all_passives_wrong']} of {s['text']['n_calls']} text calls)", fontsize=8.5, color=BAD, va="center",
            ha="left", linespacing=1.4)
    header(fig, H, "Wording-sensitive questions: most page calls are near-perfect, about one in six flips the passives",
           "One mark per call, Opus 5.5 at high effort, the same 3 documents; marks with the same score are spread out "
           "side by side. Questions whose\nanswer turns on one small word (negation, passive voice, \"all but\"). "
           "The black line is the mean over calls.")
    save(fig, "claude_repeat_calls.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- 3. pruned text
def fig_prune():
    plt = M.chart_style()
    a = load("claude_qa/prune/analysis/summary.json")["arms"]
    series = [("pages", ["page118", "page88", "page59"], PAGE_C, "o"),
              ("pruned text", ["ptext15", "ptext22"], PRUNE_C, "s"),
              ("pruned text drawn as pages", ["ppage15"], BAD, "D"),
              ("text", ["text"], TEXT_C, "*")]
    H = 5.1
    fig, axs = plt.subplots(1, 2, figsize=(10.4, H), dpi=150, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=1 - 1.7 / H, bottom=0.62 / H, wspace=0.05)
    panels = [("acc_original", "Original 120 questions per document"),
              ("acc_small", "40 wording-sensitive questions per document")]
    place = {"pages": (10, 4, "left"), "pruned text": (8, -3, "left"),   # label offsets in points (right panel only:
             "pruned text drawn as pages": (8, -3, "left"), "text": (0, -17, "center")}   # the left one is too crowded)
    for ax, (key, title) in zip(axs, panels):
        for name, arms, col, mk in series:
            xs = [a[k]["input_tokens"] / 1000 for k in arms]
            pts = [a[k][key] for k in arms]
            ys = [100 * p[0] for p in pts]
            ax.errorbar(xs, ys, yerr=[[100 * (p[0] - p[1]) for p in pts], [100 * (p[2] - p[0]) for p in pts]], color=col,
                        lw=1.6 if len(arms) > 1 else 0, elinewidth=1, marker=mk, ms=12 if mk == "*" else 7, mec=M.SURFACE,
                        mew=1, zorder=3,
                        capsize=0)
            if key != "acc_small":
                continue
            dx, dy, ha = place[name]
            i = 1 if name in ("pruned text", "pages") else 0
            ax.annotate(name, (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8.5,
                        color=col, fontweight="bold")
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", color=M.INK_1)
        ax.set_xlabel("input tokens per call (thousands); fewer = cheaper")
        ax.set_xlim(40, 265)
        ax.grid(color=M.GRID, lw=0.8)
        ax.set_axisbelow(True)
    axs[0].set_ylim(45, 103)
    axs[0].set_ylabel("answers right (%)")
    t22, p118 = a["ptext22"], a["page118"]
    header(fig, H, "Pruned text does not replace pages: at the same token count it answers fewer questions",
           f"Opus 5.5 at high effort, 3 documents, one call per arm and document; bars are 95% intervals. At about "
           f"{p118['input_tokens'] / 1000:.0f}k tokens, pruned text scores {100 * t22['acc_all'][0]:.1f}%\noverall against "
           f"{100 * p118['acc_all'][0]:.1f}% for pages. Pruning deletes the small words the right-hand questions turn on. "
           "Pruned text was made with LLMLingua-2.\nSeries are labelled in the right-hand panel; the left one uses the same colours and marks.")
    save(fig, "claude_prune_vs_pages.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- 4. graded resolution
DENSITY_C = {117.6: "#1c4f99", 88.2: "#3987e5", 58.8: "#9cc3f2"}     # one hue, darker = sharper (more px² per char)


def fig_gradient():
    plt = M.chart_style()
    g = load("claude_qa/gradient/summary.json")["arms"]
    arms = [("uniform 117.6", "uniform\n117.6", "o"), ("uniform 88.2", "uniform\n88.2", "o"),
            ("uniform 58.8", "uniform\n58.8", "o"), ("recent_sharp", "recent\nsharp", "D"),
            ("old_sharp", "old\nsharp", "s")]
    H = 5.6
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.4, H), dpi=150, gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=0.07, right=0.985, top=1 - 1.75 / H, bottom=0.62 / H, wspace=0.2)
    # left: lookups by third, each mark coloured by the density that third was drawn at
    step = 0.17
    for t in range(3):
        for i, (key, short, mk) in enumerate(arms):
            x = t + (i - 2) * step
            c = g[key]["lookup_by_tier"][str(t)]
            b = g[key]["budgets"][t]
            y, lo, hi = 100 * c["acc"], 100 * c["lo"], 100 * c["hi"]
            axl.plot([x, x], [lo, hi], color=DENSITY_C[b], lw=1.4, alpha=0.6, zorder=2, solid_capstyle="round")
            axl.scatter([x], [y], s=62 if mk == "o" else 58, marker=mk, color=DENSITY_C[b], edgecolor=M.INK_2, lw=0.7,
                        zorder=3)
            axl.text(x, 90.6, f"{b:g}", ha="center", va="top", fontsize=6.8, color=M.INK_2,
                     rotation=90)
        if t:
            axl.axvline(t - 0.5, color=M.GRID, lw=0.8)
    handles = [axl.scatter([], [], s=50, marker=mk, color=M.SURFACE, edgecolor=M.INK_2, lw=0.9,
                           label=short.replace("\n", " ")) for (key, short, mk) in arms if key != "uniform 88.2"
               and key != "uniform 58.8"]
    handles[0].set_label("uniform 117.6, 88.2, 58.8 (first three in each third)")
    handles += [axl.scatter([], [], s=50, marker="o", color=DENSITY_C[b], edgecolor=M.INK_2, lw=0.7,
                            label=f"drawn at {b:g} px²/char") for b in (117.6, 88.2, 58.8)]
    axl.legend(handles=handles, loc="lower left", fontsize=7.4, frameon=True, framealpha=1, edgecolor=M.GRID, ncol=2,
               handletextpad=0.3, columnspacing=1.0, borderpad=0.5, title="number under each mark = its px²/char",
               title_fontsize=7.4)
    axl.set_xticks([0, 1, 2], ["first (oldest) third", "middle third", "last (most recent) third"])
    axl.tick_params(axis="x", length=0)
    axl.set_xlim(-0.55, 2.55)
    axl.set_ylim(80, 101)
    axl.set_yticks([90, 92, 94, 96, 98, 100], ["90", "92", "94", "96", "98", "100%"])
    axl.grid(axis="y", color=M.GRID, lw=0.8)
    axl.set_axisbelow(True)
    axl.set_ylabel("lookups right (%), 95% interval")
    axl.set_title("Lookups by third: each third scores like its density", fontsize=9.5,
                  fontweight="bold", loc="left", color=M.INK_1)
    # right: overall accuracy against input tokens
    pts = [("text", "text", M.INK_2, "*", (0, -16), "center")]
    pts += [(f"uniform {b}", f"uniform {b:g}", DENSITY_C[b], "o", off, ha) for b, off, ha in
            ((117.6, (0, -15), "center"), (88.2, (0, 9), "center"), (58.8, (9, -3), "left"))]
    pts += [("recent_sharp", "recent sharp", "#5f8fc9", "D", (-9, -3), "right"),
            ("old_sharp", "old sharp", "#5f8fc9", "s", (9, -3), "left")]
    ux = [g[f"uniform {b}"]["input_tokens_mean"] / 1000 for b in (58.8, 88.2, 117.6)]
    uy = [100 * g[f"uniform {b}"]["all"]["acc"] for b in (58.8, 88.2, 117.6)]
    axr.plot(ux, uy, color=PAGE_C, lw=1.2, alpha=0.5, zorder=1)
    for key, lab, col, mk, off, ha in pts:
        a = g[key]
        x, y = a["input_tokens_mean"] / 1000, 100 * a["all"]["acc"]
        axr.plot([x, x], [100 * a["all"]["lo"], 100 * a["all"]["hi"]], color=col, lw=1.4, alpha=0.6, zorder=2)
        axr.scatter([x], [y], s=150 if mk == "*" else 58, marker=mk, color=col, edgecolor=M.INK_2, lw=0.7, zorder=3)
        if key in ("recent_sharp", "old_sharp"):
            lab = f"{lab}\n{y:.1f}%"
        else:
            lab = f"{lab} {y:.1f}%"
        axr.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=8,
                     color=M.INK_1, linespacing=1.15)
    axr.set_xlim(40, 265)
    axr.set_ylim(85, 101)
    axr.set_yticks([86, 88, 90, 92, 94, 96, 98, 100], ["86", "88", "90", "92", "94", "96", "98", "100%"])
    axr.grid(color=M.GRID, lw=0.8)
    axr.set_axisbelow(True)
    axr.set_xlabel("input tokens per call (thousands)")
    axr.set_ylabel("all 120 questions right (%)")
    axr.set_title("Overall: uniform 88.2's tokens, fewer right", fontsize=9.5,
                  fontweight="bold", loc="left", color=M.INK_1)
    rs, os_, u88 = g["recent_sharp"], g["old_sharp"], g["uniform 88.2"]
    header(fig, H, "Graded resolution on long-document Q&A: each third reads like its density, so grading is a trade",
           "Opus 5.5, high effort, 3 documents x 120 questions. Mixed arms: the document cut into thirds, each drawn at "
           "its own px² per character\n(recent sharp: 58.8 / 88.2 / 117.6 from oldest to newest; old sharp: the "
           f"reverse), {rs['calls']} calls each; uniform arms and text: the v2 rows, 3 calls each.\nMark colour = the "
           "density that third was drawn at (darker = sharper). Lookups = invented facts and numbers in sentences. "
           f"Both mixed arms use {rs['input_tokens_mean'] / u88['input_tokens_mean']:.2f}x the\ninput tokens of "
           "uniform 88.2; their loss is in multi-step questions with a fact in the 58.8 third, counts above all.")
    save(fig, "claude_gradient.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- 5. content-chosen sizes
SALIENT_C = "#b7791f"


def fig_salient():
    plt = M.chart_style()
    g = load("claude_qa/gradient/summary.json")["arms"]
    s = load("claude_qa/salient/summary.json")["arms"]
    H = 5.6
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.4, H), dpi=150, gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=0.07, right=0.985, top=1 - 1.75 / H, bottom=0.62 / H, wspace=0.22)
    # left: overall accuracy against input tokens, every arm on the uniform curve's axes
    ux = [s[f"uniform {b}"]["input_tokens_mean"] / 1000 for b in (58.8, 88.2, 117.6)]
    uy = [100 * s[f"uniform {b}"]["all"]["acc"] for b in (58.8, 88.2, 117.6)]
    axl.plot(ux, uy, color=PAGE_C, lw=1.2, alpha=0.5, zorder=1)
    pts = [(s["text"], "text", TEXT_C, "*", (0, -15), "center")]
    pts += [(s[f"uniform {b}"], f"uniform {b:g}", DENSITY_C[b], "o", off, ha) for b, off, ha in
            ((117.6, (0, -15), "center"), (88.2, (0, 11), "center"), (58.8, (-9, 0), "right"))]
    pts += [(g["recent_sharp"], "graded by position\n(recent sharp)", "#5f8fc9", "D", (9, -6), "left"),
            (g["old_sharp"], "(old sharp)", "#5f8fc9", "s", (9, 6), "left")]
    pts += [(s[a], lab, SALIENT_C, mk, off, ha) for a, lab, mk, off, ha in
            (("salient_large", "facts large,\nfiller small", "^", (10, 0), "left"),
             ("salient_small", "facts small,\nfiller large", "v", (10, 0), "left")) if a in s]
    for a, lab, col, mk, off, ha in pts:
        x, y = a["input_tokens_mean"] / 1000, 100 * a["all"]["acc"]
        axl.plot([x, x], [100 * a["all"]["lo"], 100 * a["all"]["hi"]], color=col, lw=1.4, alpha=0.6, zorder=2)
        axl.scatter([x], [y], s=150 if mk == "*" else 58, marker=mk, color=col, edgecolor=M.INK_2, lw=0.7, zorder=3)
        axl.annotate(f"{lab} {y:.1f}%" if "\n" not in lab else f"{lab}\n{y:.1f}%", (x, y), xytext=off,
                     textcoords="offset points", ha=ha, va="center", fontsize=8, color=M.INK_1, linespacing=1.15)
    ys = [100 * a["all"]["lo"] for a, *_ in pts]
    lo = max(0, 10 * int(min(ys) // 10) - 5)
    axl.set_xlim(35, 265)
    axl.set_ylim(lo, 102)
    axl.grid(color=M.GRID, lw=0.8)
    axl.set_axisbelow(True)
    axl.set_xlabel("input tokens per call (thousands)")
    axl.set_ylabel("all 120 questions right (%), 95% interval")
    axl.set_title("Overall: at uniform 58.8's tokens, mixing sizes by content falls below it", fontsize=9.5,
                  fontweight="bold", loc="left", color=M.INK_1)
    # right: what each question type gets at the same token count
    cats = [("needle", "invented\nfacts"), ("number", "numbers in\nsentences"), ("complex", "multi-step")]
    arms = [("uniform 58.8", "uniform 58.8", DENSITY_C[58.8]), ("salient_large", "facts large", SALIENT_C),
            ("salient_small", "facts small", "#e3c48f")]
    arms = [x for x in arms if x[0] in s]
    w = 0.26
    for i, (key, lab, col) in enumerate(arms):
        for j, (c, _) in enumerate(cats):
            v = s[key][c]
            x = j + (i - (len(arms) - 1) / 2) * w
            axr.bar(x, 100 * v["acc"], width=w - 0.03, color=col, edgecolor=M.SURFACE, lw=1, zorder=2)
            axr.text(x, 100 * v["acc"] + 1.2, f"{100 * v['acc']:.0f}", ha="center", va="bottom", fontsize=7.6,
                     color=M.INK_1)
    axr.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col, label=lab) for _, lab, col in arms], loc="upper right",
               fontsize=7.6, frameon=True, framealpha=1, edgecolor=M.GRID, bbox_to_anchor=(1.0, 1.0))
    axr.set_xticks(range(len(cats)), [lab for _, lab in cats])
    axr.tick_params(axis="x", length=0)
    axr.set_ylim(0, 118)
    axr.set_yticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100%"])
    axr.grid(axis="y", color=M.GRID, lw=0.8)
    axr.set_axisbelow(True)
    axr.set_ylabel("questions right (%)")
    axr.set_title("By question type, same token count", fontsize=9.5, fontweight="bold", loc="left", color=M.INK_1)
    sl = s.get("salient_large", s["uniform 58.8"])
    header(fig, H, "Facts large, filler small: at a fixed token budget the unreadable filler costs more than the large "
           "facts recover",
           "Opus 5.5, high effort, 3 documents x 120 questions. Mixed arms: every word a rule-based detector marks "
           "(numbers, names, negations) drawn at the em of\n117.6 px²/char pages, the rest at the em that brings the "
           "document to uniform 58.8's image tokens (em 3.5-4.1 px, unreadable); the control swaps them.\n"
           f"Mixed arms: {sl['calls']} calls each; uniform arms and text: the v2 rows, 3 calls each; graded-by-position "
           "arms: yesterday's test (uniform 88.2's tokens).")
    save(fig, "claude_salient.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_overview()
    fig_repeat()
    fig_prune()
    fig_gradient()
    fig_salient()
    print("claude_uses_overview.png claude_repeat_calls.png claude_prune_vs_pages.png claude_gradient.png "
          "claude_salient.png")
