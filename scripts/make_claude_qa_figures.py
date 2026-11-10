"""Figures for docs/CLAUDE_QA.md (Claude answers questions from page images vs from text).

1. claude_qa_verdict.png    accuracy change (image - text, points) per model x effort x token saving, all questions
                            and multi-step questions; worst case (95% bound) and the pre-registered lossless mark
2. claude_qa_pages.png      what the model sees: the same passage at the three densities, actual pixels, with the
                            28-px vision-token grid
3. claude_qa_breakdown.png  accuracy per question type, both models at high effort, shaded by points lost vs text
4. claude_qa_cost.png       list-price cost per call vs accuracy, both models at high effort

Numbers come from results/claude_qa/analysis_v2{,_haiku}/summary.json (experiments/analyse_claude_qa.py) and the
calls' token counts in results/claude_qa/qav2*_*/runs.jsonl.

    python3 scripts/make_claude_qa_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_readme_figures as M  # noqa: E402

ROOT = M.ROOT
QA = ROOT / "results" / "claude_qa"
MODELS = {"Opus 5.5": {"summary": QA / "analysis_v2" / "summary.json", "runs": "qav2_d*", "colour": "#6250d6",
                       "marker": "o", "price": lambda tok: (4e-6, 20e-6)},
          "Haiku 5.5": {"summary": QA / "analysis_v2_haiku" / "summary.json", "runs": "qav2h_d*",
                        "colour": "#1baf7a", "marker": "D",
                        "price": lambda tok: (0.1e-6, 0.5e-6) if tok <= 100_000 else (0.5e-6, 2.5e-6)}}
BUDGETS = [(117.6, "2.2x"), (88.2, "2.9x"), (58.8, "4.2x")]
CACHE_WRITE = 1.25
# "points lost" in 8 discrete bands of the validated sequential blue ramp (steps 100-700, near-surface tint for ~0).
# Discrete bands: every cell matches a legend swatch exactly, and the 64-colour PNG cannot shift them.
LOSS_RAMP = ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LOSS_EDGES = [2, 5, 10, 20, 30, 45, 60]          # band i covers [edge i-1, edge i)
LOSS_LABELS = ["<2", "2-5", "5-10", "10-20", "20-30", "30-45", "45-60", "60+"]
NEUTRAL = "#f0efec"


def save(fig, name: str) -> None:
    """150 dpi PNG quantised to 256 colours (the discrete loss bands keep their exact shade)."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, format="png")
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(M.OUT / name, optimize=True)


def summaries() -> dict:
    return {m: json.loads(v["summary"].read_text()) for m, v in MODELS.items()}


def band(loss: float) -> int:
    return int(np.searchsorted(LOSS_EDGES, max(0.0, loss), side="right"))


def loss_colour(loss: float) -> str:
    return LOSS_RAMP[band(loss)]


def ink_on(loss: float) -> str:
    return "#ffffff" if band(loss) >= 5 else M.INK_1      # white on steps 500-700 only


def fmt_pts(v: float) -> str:
    return ("−" if v < -0.05 else "+" if v > 0.05 else "±") + f"{abs(v):.1f}"


def loss_legend(fig, rect) -> None:
    from matplotlib.patches import Rectangle
    ax = fig.add_axes(rect)
    for i, (c, lab) in enumerate(zip(LOSS_RAMP, LOSS_LABELS)):
        ax.add_patch(Rectangle((i + 0.04, 0), 0.92, 1, facecolor=c, lw=0))
        ax.text(i + 0.5, -0.35, lab, ha="center", va="top", fontsize=7.5, color=M.INK_2)
    ax.set_xlim(0, len(LOSS_RAMP))
    ax.set_ylim(-1.4, 1)
    ax.axis("off")
    ax.text(0, 1.25, "shading: accuracy points lost vs reading text", fontsize=8, color=M.INK_2, va="bottom")


# ------------------------------------------------------------------------------------------- 1. verdict
def fig_verdict():
    plt = M.chart_style()
    from matplotlib.patches import Rectangle
    S = summaries()
    rows = [("Opus 5.5", "high"), ("Opus 5.5", "low"), ("Haiku 5.5", "high"), ("Haiku 5.5", "low")]
    panels = [("All questions", "360 per cell", "text_minus_image", True),
              ("Multi-step questions", "120 per cell: two-hop, three-hop, sums, largest-of-three, counts",
               "complex_text_minus_image", False)]
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.9), dpi=150)
    W, H, G = 1.0, 1.0, 0.035
    for pi, (ax, (title, sub, key, verdict)) in enumerate(zip(axs, panels)):
        for i, (m, e) in enumerate(rows):
            for j, (b, lab) in enumerate(BUDGETS):
                r = S[m][f"{e}/{b}"]
                x0, y0 = j * W + G, i * H + G
                w, h = W - 2 * G, H - 2 * G
                if r.get("disengaged"):
                    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=NEUTRAL, edgecolor="#c3c2b7", hatch="////",
                                           lw=0))
                    if j == 1:
                        ax.text(x0 + w / 2, y0 + h / 2, "not comparable*", ha="center", va="center", fontsize=8.5,
                                color=M.INK_2, bbox=dict(facecolor=NEUTRAL, edgecolor="none", pad=1.5))
                    continue
                d, lo, hi = (100 * v for v in r[key])
                change, worst, loss = -d, -hi, max(0.0, d)
                ax.add_patch(Rectangle((x0, y0), w, h, facecolor=loss_colour(loss), lw=0))
                ink = ink_on(loss)
                ax.text(x0 + w / 2, y0 + h * 0.40, fmt_pts(change), ha="center", va="center", fontsize=12.5,
                        fontweight="bold", color=ink)
                tick = "✓ " if (verdict and r.get("lossless")) else ""
                ax.text(x0 + w / 2, y0 + h * 0.72, f"{tick}worst {fmt_pts(worst)}", ha="center", va="center",
                        fontsize=8, color=ink)
        ax.set_xlim(0, 3 * W)
        ax.set_ylim(4 * H, 0)
        ax.set_xticks([(j + 0.5) * W for j in range(3)], [f"{lab} fewer" for _, lab in BUDGETS], fontsize=9)
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0)
        if pi == 0:
            ax.set_yticks([(i + 0.5) * H for i in range(4)], [f"{m}\n{e} effort" for m, e in rows], fontsize=9)
        else:
            ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=34)
        ax.text(0, 1.115, sub, transform=ax.transAxes, fontsize=8, color=M.INK_2, va="bottom")
    fig.suptitle("Reading page images instead of text: accuracy change in points (image − text)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.885, "Same 3 documents (~245k text tokens each) and questions in every cell; columns: how many "
             "fewer input tokens the pages use. Worst = 95% bound.\n✓ lossless (pre-registered): the worst case is "
             "within 2 points.  * Haiku at low effort barely reasoned over the text (400-800 output tokens), so it "
             "has no text baseline.", fontsize=8.5, color=M.INK_2, linespacing=1.45)
    loss_legend(fig, [0.70, 0.045, 0.25, 0.03])
    fig.subplots_adjust(left=0.11, right=0.985, top=0.70, bottom=0.12, wspace=0.05)
    save(fig, "claude_qa_verdict.png")
    plt.close(fig)
    print("claude_qa_verdict.png")


# ------------------------------------------------------------------------------------------- 2. pages
def fig_pages():
    """The same passage rendered as the QA pages are (DejaVu Sans, 1932-px page, chars = area / budget x 0.97),
    cropped to 31 x 7 vision tokens at actual pixels, with the 28-px token grid."""
    from vbpe import corpus
    from vbpe.layout import render_page
    from vbpe.schemes import get_scheme
    sch = get_scheme("latin")
    text = " ".join(corpus.read_paragraphs("test")[:300])
    P, cw, ch, pad = 28, 31 * 28, 6 * 28, 24
    crops = []
    for b, lab in BUDGETS:
        n = int(1932 * 1932 / b * 0.97)
        img, info = render_page(sch.encode(text[:n]), 1932, 1932)
        crop = img.convert("RGB").crop((0, 0, cw, ch))
        d = ImageDraw.Draw(crop, "RGBA")
        for x in range(0, cw + 1, P):
            d.line([(x, 0), (x, ch)], fill=(57, 135, 229, 70), width=1)
        for y in range(0, ch + 1, P):
            d.line([(0, y), (cw, y)], fill=(57, 135, 229, 70), width=1)
        crops.append((b, lab, info["em"], crop))
    head = 78
    row_h = 46 + ch + 14
    out = Image.new("RGB", (cw + 2 * pad, head + row_h * len(crops) + 8), M.SURFACE)
    d = ImageDraw.Draw(out)
    d.text((pad, 14), "What the model sees: the same passage at the three densities (actual pixels)",
           font=M.ui(19, "Bold"), fill=M.INK_1)
    d.text((pad, 44), "Each blue square is one vision token (28 x 28 px). As text, the same passage costs one token "
           "per ~2.9 characters.", font=M.ui(14), fill=M.INK_2)
    y = head
    for b, lab, em, crop in crops:
        cpt = 784 / b
        d.text((pad, y), f"{lab} fewer tokens than text", font=M.ui(16, "Bold"), fill=M.INK_1)
        d.text((pad + 250, y + 2), f"{b:g} px² per character  ·  ~{cpt:.1f} characters per vision token  ·  "
               f"font size {em:.1f} px", font=M.ui(14), fill=M.INK_2)
        out.paste(crop, (pad, y + 28))
        d.rectangle([pad - 1, y + 27, pad + cw, y + 28 + ch], outline=M.AXIS)
        y += row_h
    out.save(M.OUT / "claude_qa_pages.png", optimize=True)
    print("claude_qa_pages.png", [round(e, 1) for _, _, e, _ in crops])


# ------------------------------------------------------------------------------------------- 3. breakdown
KINDS = [("needle", "Invented fact, looked up"), ("number", "Number in a quoted sentence"),
         ("sum", "Sum of two facts"), ("max", "Largest of three facts"), ("two_hop", "Two-hop chain"),
         ("three_hop", "Three-hop chain"), ("count", "Count across the whole document")]


def fig_breakdown():
    plt = M.chart_style()
    from matplotlib.patches import Rectangle
    S = summaries()
    cols = []   # (model, label, kinds-dict, which: 0 text / 1 image)
    for m in MODELS:
        first = S[m][f"high/{BUDGETS[0][0]}"]["kinds"]
        cols.append((m, "text", first, 0))
        for b, lab in BUDGETS:
            cols.append((m, lab, S[m][f"high/{b}"]["kinds"], 1))
    fig, ax = plt.subplots(figsize=(10.4, 5.4), dpi=150)
    G, gapx = 0.035, 0.45
    xs = []
    x = 0.0
    for ci, (m, lab, kd, w) in enumerate(cols):
        if ci and cols[ci - 1][0] != m:
            x += gapx
        xs.append(x)
        x += 1
    for ri, (k, name) in enumerate(KINDS):
        yb = ri + (0.35 if ri >= 2 else 0)
        for ci, (m, lab, kd, w) in enumerate(cols):
            tc, ic, n = kd[k]
            acc = (tc if w == 0 else ic) / n * 100
            loss = 0 if w == 0 else max(0.0, (tc - ic) / n * 100)
            fill = NEUTRAL if w == 0 else loss_colour(loss)
            ax.add_patch(Rectangle((xs[ci] + G, yb + G), 1 - 2 * G, 1 - 2 * G, facecolor=fill, lw=0))
            ax.text(xs[ci] + 0.5, yb + 0.5, f"{acc:.0f}%", ha="center", va="center", fontsize=9.5,
                    fontweight="bold" if w else "normal", color=M.INK_1 if w == 0 else ink_on(loss))
    n_of = {k: S["Opus 5.5"]["high/117.6"]["kinds"][k][2] for k, _ in KINDS}
    ax.set_yticks([ri + (0.35 if ri >= 2 else 0) + 0.5 for ri in range(len(KINDS))],
                  [f"{name} ({n_of[k]})" for k, name in KINDS], fontsize=9)
    ax.set_xticks([])
    ax.tick_params(axis="both", length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_xlim(0, x)
    ax.set_ylim(len(KINDS) + 0.35, -1.25)
    for ci, c in enumerate(cols):
        ax.text(xs[ci] + 0.5, -0.22, c[1] if c[1] == "text" else f"{c[1]} pages", ha="center", va="bottom",
                fontsize=9, color=M.INK_2)
    for m in MODELS:
        c0 = [i for i, c in enumerate(cols) if c[0] == m]
        ax.text((xs[c0[0]] + xs[c0[-1]] + 1) / 2, -0.78, f"{m}, high effort", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=M.INK_1)
    ax.axhline(2.175, color=M.AXIS, lw=0.8, ls=(0, (3, 3)))
    fig.suptitle("Where it breaks: chains and counts fail first, and Haiku fails at lower densities",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.885, "Accuracy per question type (questions in brackets, pooled over 3 documents). Columns: the "
             "model reading text, then page images at 2.2x, 2.9x and 4.2x fewer tokens.\nShading: points lost vs the "
             "same model reading text. Lookups above the dashed line, multi-step questions below.",
             fontsize=8.5, color=M.INK_2, linespacing=1.45)
    loss_legend(fig, [0.70, 0.04, 0.25, 0.03])
    fig.subplots_adjust(left=0.25, right=0.985, top=0.80, bottom=0.12)
    save(fig, "claude_qa_breakdown.png")
    plt.close(fig)
    print("claude_qa_breakdown.png")


# ------------------------------------------------------------------------------------------- 4. cost
def call_cost(model: str, u: dict) -> float:
    """List price as if uncached (every input token written to the 5-minute cache once, as the first call of a
    document is); reproduces the logged cost of every uncached v2 call of both models."""
    tok = u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"]
    pin, pout = MODELS[model]["price"](tok)
    return tok * pin * CACHE_WRITE + u["output_tokens"] * pout


def costs(model: str, effort: str = "high") -> dict:
    out = {}
    for f in sorted(QA.glob(MODELS[model]["runs"] + "/runs.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("cached") or r["arm"] == "none" or r["effort"] != effort:
                continue
            out.setdefault("text" if r["arm"] == "text" else r["budget"], []).append(call_cost(model, r["usage"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def fig_cost():
    """One row per option (model x input form), two aligned panels: list-price cost per call (bar) and accuracy on
    all 360 questions (dot, 95% Wilson interval). High effort."""
    plt = M.chart_style()
    S = summaries()
    rows_ = []   # (model, label, cost, acc, lo, hi, lossless)
    for m in MODELS:
        c = costs(m)
        first = S[m][f"high/{BUDGETS[0][0]}"]["kinds"]["all"]
        lo, hi = M.wilson(first[0], first[2])
        rows_.append((m, "text", c["text"], first[0] / first[2], lo, hi, None))
        for b, lab in BUDGETS:
            r = S[m][f"high/{b}"]
            a = r["kinds"]["all"]
            lo, hi = M.wilson(a[1], a[2])
            rows_.append((m, f"{lab} fewer tokens (pages)", c[b], a[1] / a[2], lo, hi, bool(r.get("lossless"))))
    fig, (axc, axa) = plt.subplots(1, 2, figsize=(10.4, 5.3), dpi=150, sharey=True,
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})
    ys, y = [], 0.0
    for i, r in enumerate(rows_):
        if i and rows_[i - 1][0] != r[0]:
            y += 0.7
        ys.append(y)
        y += 1
    for yy, (m, lab, cost, acc, lo, hi, ok) in zip(ys, rows_):
        col = MODELS[m]["colour"]
        axc.barh(yy, cost, height=0.62, color=col, lw=0)
        axc.text(cost + 0.02, yy, f"${cost:.2f}" if cost >= 0.1 else f"${cost:.3f}", va="center", fontsize=8.5,
                 color=M.INK_1)
        axa.plot([lo * 100, hi * 100], [yy, yy], color=col, lw=1.4, zorder=2)
        axa.plot(acc * 100, yy, marker=MODELS[m]["marker"], ms=8 if MODELS[m]["marker"] == "o" else 7, color=col,
                 mec=M.SURFACE, mew=1.5, zorder=3)
        note = "  ✓" if ok else ""
        axa.text(hi * 100 + 1.5, yy, f"{acc * 100:.1f}%{note}", va="center", fontsize=8.5, color=M.INK_1)
    labels = [f"{m}: {lab}" for m, lab, *_ in rows_]
    axc.set_yticks(ys, labels, fontsize=9)
    axc.set_ylim(ys[-1] + 0.7, -0.9)
    axc.tick_params(axis="y", length=0)
    axc.set_xlim(0, 1.6)
    axc.set_xticks([0, 0.5, 1.0, 1.5], ["$0", "$0.50", "$1.00", "$1.50"])
    axc.set_xlabel("list-price cost per call")
    axc.set_title("Cost of one call", loc="left", fontsize=10.5, fontweight="bold")
    axa.set_xlim(40, 113)
    axa.set_xticks([40, 60, 80, 100], ["40%", "60%", "80%", "100%"])
    axa.set_xlabel("questions answered correctly (all 360)")
    axa.set_title("What it gets right", loc="left", fontsize=10.5, fontweight="bold")
    axa.tick_params(axis="y", length=0)
    for ax in (axc, axa):
        ax.grid(axis="x", color=M.GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
    axa.spines["bottom"].set_bounds(40, 100)
    fig.suptitle("Opus reading 2.2-2.9x denser pages costs half as much as Opus reading text, for the same answers",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.885, "One call = a ~245k-token document plus 120 questions, high effort, mean of 3 documents. "
             "✓ = no measurable loss vs the same model reading text (pre-registered).\nList prices as if uncached: "
             "Opus \\$4 / \\$20 per million input / output tokens; Haiku \\$0.10 / \\$0.50 up to 100k-token prompts, "
             "\\$0.50 / \\$2.50 above. Not amounts billed.",
             fontsize=8.5, color=M.INK_2, linespacing=1.45)
    fig.subplots_adjust(left=0.235, right=0.985, top=0.76, bottom=0.11, wspace=0.08)
    save(fig, "claude_qa_cost.png")
    plt.close(fig)
    print("claude_qa_cost.png", {m: {k: round(v, 3) for k, v in costs(m).items()} for m in MODELS})


if __name__ == "__main__":
    M.OUT.mkdir(parents=True, exist_ok=True)
    fig_verdict()
    fig_pages()
    fig_breakdown()
    fig_cost()
