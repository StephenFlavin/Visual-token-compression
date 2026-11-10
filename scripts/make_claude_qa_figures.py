"""Figures for docs/CLAUDE_QA.md (Claude answers questions from page images vs from text).

1. claude_qa_verdict.png    accuracy change (image - text, points) per model x effort x token saving, all questions
                            and multi-step questions; worst case (95% bound) and the pre-registered lossless mark
2. claude_qa_pages.png      what the model sees: the same passage at the three densities, actual pixels, with the
                            28-px vision-token grid
3. claude_qa_breakdown.png  accuracy per question type (columns) for each model reading text and pages (row blocks),
                            high effort, shaded by points lost vs the same model reading text
4. claude_qa_cost.png       list-price cost per call and accuracy, one row per model x input form, high effort

Numbers come from results/claude_qa/analysis_v2{,_sonnet,_haiku}/summary.json (experiments/analyse_claude_qa.py) and
the calls' token counts in results/claude_qa/qav2*_*/runs.jsonl. Models missing a summary are skipped.

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
# model colours: validated categorical slots (violet, orange, aqua) - all pairs pass; aqua is < 3:1 on white, so every
# figure names the model next to its marks
ALL_MODELS = {
    "Opus 5.5": {"summary": QA / "analysis_v2" / "summary.json", "runs": "qav2_d*", "colour": "#6250d6",
                 "marker": "o", "price": lambda tok: (4e-6, 20e-6)},
    "Sonnet 5.5": {"summary": QA / "analysis_v2_sonnet" / "summary.json", "runs": "qav2s_d*", "colour": "#eb6834",
                   "marker": "s", "price": lambda tok: (2e-6, 10e-6)},
    "Haiku 5.5": {"summary": QA / "analysis_v2_haiku" / "summary.json", "runs": "qav2h_d*", "colour": "#1baf7a",
                  "marker": "D", "price": lambda tok: (0.1e-6, 0.5e-6) if tok <= 100_000 else (0.5e-6, 2.5e-6)}}
MODELS = {m: v for m, v in ALL_MODELS.items() if v["summary"].exists()}
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


def header(fig, H_in: float, title: str, sub: str) -> None:
    """Title 0.2 in from the top, subtitle (up to two lines) below it; fixed in inches whatever the figure height."""
    fig.text(0.02, 1 - 0.18 / H_in, title, ha="left", va="top", fontsize=12, fontweight="bold", color=M.INK_1)
    fig.text(0.02, 1 - 0.50 / H_in, sub, ha="left", va="top", fontsize=8.5, color=M.INK_2, linespacing=1.45)


def block_label(fig, ax, y_mid: float, text: str) -> None:
    """A bold model name at the figure's left edge, vertically centred on a block of rows (data y)."""
    import matplotlib.transforms as mt
    ax.text(0.02, y_mid, text, transform=mt.blended_transform_factory(fig.transFigure, ax.transData), ha="left",
            va="center", fontsize=9.5, fontweight="bold", color=M.INK_1)


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
    rows = [(m, e) for m in MODELS for e in ("high", "low")]
    ys, y = [], 0.0
    for i, (m, e) in enumerate(rows):
        if i and rows[i - 1][0] != m:
            y += 0.3
        ys.append(y)
        y += 1
    panels = [("All questions", "360 per cell", "text_minus_image", True),
              ("Multi-step questions", "120 per cell: two-hop, three-hop, sums, largest-of-three, counts",
               "complex_text_minus_image", False)]
    TOP, BOT, ROW = 2.05, 0.75, 0.6           # inches: header block, legend strip, one row
    H_in = TOP + BOT + ROW * y
    fig, axs = plt.subplots(1, 2, figsize=(10.4, H_in), dpi=150)
    fig.subplots_adjust(left=0.20, right=0.985, top=1 - TOP / H_in, bottom=BOT / H_in, wspace=0.05)
    G = 0.035
    flagged = []
    for pi, (ax, (title, sub, key, verdict)) in enumerate(zip(axs, panels)):
        for yy, (m, e) in zip(ys, rows):
            for j, (b, lab) in enumerate(BUDGETS):
                r = S[m][f"{e}/{b}"]
                x0, y0, w, h = j + G, yy + G, 1 - 2 * G, 1 - 2 * G
                if r.get("disengaged"):
                    if pi == 0 and j == 0:
                        flagged.append(f"{m} {e}")
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
                ax.text(x0 + w / 2, y0 + h * 0.40, fmt_pts(change), ha="center", va="center", fontsize=12,
                        fontweight="bold", color=ink)
                tick = "✓ " if (verdict and r.get("lossless")) else ""
                ax.text(x0 + w / 2, y0 + h * 0.74, f"{tick}worst {fmt_pts(worst)}", ha="center", va="center",
                        fontsize=7.8, color=ink)
        ax.set_xlim(0, 3)
        ax.set_ylim(y, 0)
        ax.set_xticks([j + 0.5 for j in range(3)], [f"{lab} fewer" for _, lab in BUDGETS], fontsize=9)
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0)
        if pi == 0:
            ax.set_yticks([yy + 0.5 for yy in ys], [f"{e} effort" for _, e in rows], fontsize=8.5)
            for m in MODELS:
                yy = [v for v, r in zip(ys, rows) if r[0] == m]
                block_label(fig, ax, (yy[0] + yy[-1] + 1) / 2, m)
        else:
            ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        x0 = ax.get_position().x0
        fig.text(x0, 1 - 1.18 / H_in, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=M.INK_1)
        fig.text(x0, 1 - 1.42 / H_in, sub, ha="left", va="top", fontsize=8, color=M.INK_2)
    note = ("  * " + ", ".join(sorted(set(flagged))) + " effort: the model barely reasoned over the text, so there "
            "is no text baseline.") if flagged else ""
    header(fig, H_in, "Reading page images instead of text: accuracy change in points (image − text)",
           "Same 3 documents (~245k text tokens each) and questions in every cell; columns: how many fewer input tokens "
           "the pages use. Worst = 95% bound.\n✓ lossless (pre-registered): the worst case is within 2 points." + note)
    loss_legend(fig, [0.70, 0.10 / H_in, 0.25, 0.16 / H_in])
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
KINDS = [("needle", "Invented\nfact"), ("number", "Number in\na sentence"), ("sum", "Sum of\ntwo facts"),
         ("max", "Largest\nof three"), ("two_hop", "Two-hop\nchain"), ("three_hop", "Three-hop\nchain"),
         ("count", "Count across\nthe document")]
N_LOOKUP = 2


def fig_breakdown():
    """Columns: question types (lookups | multi-step). Rows: one block per model (text, then pages at each density),
    high effort. Cell = accuracy; shading = points lost vs the same model reading text on that question type."""
    plt = M.chart_style()
    from matplotlib.patches import Rectangle
    S = summaries()
    rows = []    # (model, label, kinds dict, which: 0 text / 1 image)
    for m in MODELS:
        rows.append((m, "text", S[m][f"high/{BUDGETS[0][0]}"]["kinds"], 0))
        rows += [(m, f"{lab} pages", S[m][f"high/{b}"]["kinds"], 1) for b, lab in BUDGETS]
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and rows[i - 1][0] != r[0]:
            y += 0.35
        ys.append(y)
        y += 1
    xs = [j + (0.3 if j >= N_LOOKUP else 0) for j in range(len(KINDS))]
    TOP, BOT, ROW = 1.15, 0.75, 0.40
    H_in = TOP + BOT + ROW * (y + 1.75)
    fig, ax = plt.subplots(figsize=(10.4, H_in), dpi=150)
    fig.subplots_adjust(left=0.24, right=0.985, top=1 - TOP / H_in, bottom=BOT / H_in)
    G = 0.035
    for yy, (m, lab, kd, w) in zip(ys, rows):
        for xx, (k, _) in zip(xs, KINDS):
            tc, ic, n = kd[k]
            acc = (tc if w == 0 else ic) / n * 100
            loss = 0 if w == 0 else max(0.0, (tc - ic) / n * 100)
            ax.add_patch(Rectangle((xx + G, yy + G), 1 - 2 * G, 1 - 2 * G,
                                   facecolor=NEUTRAL if w == 0 else loss_colour(loss), lw=0))
            ax.text(xx + 0.5, yy + 0.5, f"{acc:.0f}%", ha="center", va="center", fontsize=9,
                    fontweight="normal" if w == 0 else "bold", color=M.INK_1 if w == 0 else ink_on(loss))
    n_of = {k: S[next(iter(MODELS))][f"high/{BUDGETS[0][0]}"]["kinds"][k][2] for k, _ in KINDS}
    for xx, (k, name) in zip(xs, KINDS):
        ax.text(xx + 0.5, -0.12, f"{name}\n({n_of[k]})", ha="center", va="bottom", fontsize=8.3, color=M.INK_2,
                linespacing=1.15)
    ax.text((xs[0] + xs[N_LOOKUP - 1] + 1) / 2, -1.55, "Lookups", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=M.INK_1)
    ax.text((xs[N_LOOKUP] + xs[-1] + 1) / 2, -1.55, "Multi-step questions", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=M.INK_1)
    ax.set_yticks([yy + 0.5 for yy in ys], [r[1] for r in rows], fontsize=8.5)
    for m in MODELS:
        yy = [v for v, r in zip(ys, rows) if r[0] == m]
        block_label(fig, ax, (yy[0] + yy[-1] + 1) / 2, m)
    ax.set_xticks([])
    ax.tick_params(axis="both", length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_xlim(0, xs[-1] + 1)
    ax.set_ylim(y, -1.75)
    header(fig, H_in, "Where it breaks: chains and counts fail first, and smaller models fail at lower densities",
           "Accuracy per question type at high effort (questions per type in brackets, pooled over 3 documents). Grey "
           "rows: the model reading text;\nblue rows: the same model reading pages with 2.2x, 2.9x and 4.2x fewer "
           "tokens, shaded by points lost against its text row.")
    loss_legend(fig, [0.70, 0.10 / H_in, 0.25, 0.16 / H_in])
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
            rows_.append((m, f"{lab} pages", c[b], a[1] / a[2], lo, hi, bool(r.get("lossless"))))
    n_rows = 4 * len(MODELS)
    H_in = 2.45 + 0.36 * n_rows
    fig, (axc, axa) = plt.subplots(1, 2, figsize=(10.4, H_in), dpi=150, sharey=True,
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
    axc.set_yticks(ys, [lab for _, lab, *_ in rows_], fontsize=9)
    for m in MODELS:
        yy = [v for v, r in zip(ys, rows_) if r[0] == m]
        block_label(fig, axc, (yy[0] + yy[-1]) / 2, m)
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
    header(fig, H_in, "Pages at 2.2x fewer tokens halve the cost for Opus and Sonnet, with no measurable loss",
           "One call = a ~245k-token document plus 120 questions, high effort, mean of 3 documents. ✓ = no measurable "
           "loss vs the same model reading text\n(pre-registered). List prices as if uncached: Opus \\$4 / \\$20 and "
           "Sonnet \\$2 / \\$10 per million input / output tokens; Haiku \\$0.10 / \\$0.50 up to 100k-token\n"
           "prompts, \\$0.50 / \\$2.50 above. Calls drew on a subscription quota, so these are not amounts billed.")
    fig.subplots_adjust(left=0.20, right=0.985, top=1 - 1.6 / H_in, bottom=0.6 / H_in, wspace=0.08)
    save(fig, "claude_qa_cost.png")
    plt.close(fig)
    print("claude_qa_cost.png", {m: {k: round(v, 3) for k, v in costs(m).items()} for m in MODELS})


if __name__ == "__main__":
    M.OUT.mkdir(parents=True, exist_ok=True)
    fig_verdict()
    fig_pages()
    fig_breakdown()
    fig_cost()
