"""Figures for docs/CLAUDE_QA.md (Claude answers questions from page images vs from text).

1. claude_qa_verdict.png    accuracy change (image - text, points) per model x effort x token saving, all questions
                            and multi-step questions; worst case (95% bound) and the pre-registered lossless mark
2. claude_qa_pages.png      what the model sees: the same passage at the three densities, actual pixels, with the
                            28-px vision-token grid
3. claude_qa_breakdown.png  accuracy per question type (columns) for each model reading text and pages (row blocks),
                            high effort, shaded by points lost vs the same model reading text
4. claude_qa_cost.png       list-price cost per call and accuracy, one row per model x input form, both efforts
5. claude_qa_curve.png      accuracy change vs token saving, one line per model (high effort), lossless band shaded
6. claude_qa_effort.png     low vs high effort, accuracy change per model and density (models with a valid low row)
7. claude_qa_position.png   single-fact lookups by where the fact sits in the document (does depth matter?)
8. claude_qa_engagement.png output tokens per call, text vs pages, per model and effort (the Haiku exclusion)

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
           "Same 3 documents (~240k tokens of text each) and 120 questions in every cell; columns: how many fewer input "
           "tokens the pages use. Worst = the losing end of the 95% interval.\n✓ lossless (pre-registered): the worst "
           "case is within 2 points." + note)
    loss_legend(fig, [0.70, 0.10 / H_in, 0.25, 0.16 / H_in])
    save(fig, "claude_qa_verdict.png")
    plt.close(fig)
    print("claude_qa_verdict.png")


# ------------------------------------------------------------------------------------------- 2. pages
def fig_pages():
    """The first page of document 1 exactly as the test sent it (the experiment's own document with its inserted
    facts, and its page geometry: 1932 x 1876 px, 4,626 tokens), cropped to 31 x 6 vision tokens at actual pixels,
    with the 28-px token grid."""
    sys.path.insert(0, str(ROOT / "experiments"))
    import claude_qa_render as Q
    from vbpe import metrics
    from vbpe.layout import render_page
    from vbpe.schemes import get_scheme
    sch = get_scheme("latin")
    doc, _ = Q.make_questions(Q.build_document(700000, 0), 40, 40, 0, 40)
    P, cw, ch, pad = 28, 31 * 28, 6 * 28, 24
    crops = []
    for b, lab in BUDGETS:
        first = Q.chunk(doc, int(1932 * 1932 / b * 0.97))[0]
        n = len(metrics.normalize(first))
        H = min(1932, max(2 * P, int(np.ceil(n * b / 1932 / P)) * P))
        img, info = render_page(sch.encode(first), 1932, H)
        crop = img.convert("RGB").crop((0, 0, cw, ch))
        d = ImageDraw.Draw(crop, "RGBA")
        for x in range(0, cw + 1, P):
            d.line([(x, 0), (x, ch)], fill=(57, 135, 229, 70), width=1)
        for y in range(0, ch + 1, P):
            d.line([(0, y), (cw, y)], fill=(57, 135, 229, 70), width=1)
        crops.append((b, lab, info["em"], crop, H))
    head = 98
    row_h = 46 + ch + 14
    out = Image.new("RGB", (cw + 2 * pad, head + row_h * len(crops) + 8), M.SURFACE)
    d = ImageDraw.Draw(out)
    d.text((pad, 14), "What the model sees: the top of the first test page at each density (actual pixels)",
           font=M.ui(19, "Bold"), fill=M.INK_1)
    d.text((pad, 44), "Each blue square is one vision token (28 x 28 px); a full page is 1932 x 1876 px (4,626 tokens).",
           font=M.ui(14), fill=M.INK_2)
    d.text((pad, 64), "As text, Claude's tokenizer takes ~2.9 characters per token. Document 1, as sent to the models.",
           font=M.ui(14), fill=M.INK_2)
    y = head
    for b, lab, em, crop, _ in crops:
        cpt = 784 / b
        d.text((pad, y), f"{lab} fewer tokens than text", font=M.ui(16, "Bold"), fill=M.INK_1)
        d.text((pad + 250, y + 2), f"{b:g} px² per character  ·  ~{cpt:.1f} characters per vision token  ·  "
               f"font size {em:.1f} px", font=M.ui(14), fill=M.INK_2)
        out.paste(crop, (pad, y + 28))
        d.rectangle([pad - 1, y + 27, pad + cw, y + 28 + ch], outline=M.AXIS)
        y += row_h
    out.save(M.OUT / "claude_qa_pages.png", optimize=True)
    print("claude_qa_pages.png", [(round(e, 2), H) for _, _, e, _, H in crops])


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
    document is); reproduces the logged cost of every uncached v2 call of all three models."""
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


def tint(hex_colour: str, amount: float = 0.55) -> tuple:
    """Blend a colour towards the surface (white): the low-effort shade of a model's colour."""
    import matplotlib.colors as mc
    r, g, b_ = mc.to_rgb(hex_colour)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b_ + (1 - b_) * amount)


def fig_cost():
    """One row per option (model x input form), both efforts in each row: list-price cost per call (light bar = low
    effort, solid bar = high) and accuracy on all 360 questions (hollow dot = low, filled = high; 95% Wilson
    interval). Costs as if uncached, mean of 3 documents."""
    plt = M.chart_style()
    S = summaries()
    C = {e: {m: costs(m, e) for m in MODELS} for e in ("low", "high")}
    rows_ = []   # (model, label, {effort: (cost, acc, lo, hi, lossless, flagged)})
    for m in MODELS:
        for key, lab in [("text", "text")] + [(b, f"{l_} pages") for b, l_ in BUDGETS]:
            per = {}
            for e in ("low", "high"):
                r = S[m][f"{e}/{BUDGETS[0][0] if key == 'text' else key}"]
                k, n = (r["kinds"]["all"][0] if key == "text" else r["kinds"]["all"][1]), r["kinds"]["all"][2]
                lo, hi = M.wilson(k, n)
                per[e] = (C[e][m][key], k / n, lo, hi, key != "text" and bool(r.get("lossless")),
                          bool(r.get("disengaged")))
            rows_.append((m, lab, per))
    ys, y = [], 0.0
    for i, r in enumerate(rows_):
        if i and rows_[i - 1][0] != r[0]:
            y += 0.6
        ys.append(y)
        y += 1
    TOP, BOT, ROW = 2.15, 0.65, 0.5
    H_in = TOP + BOT + ROW * (y + 0.6)
    fig, (axc, axa) = plt.subplots(1, 2, figsize=(10.4, H_in), dpi=150, sharey=True)
    fig.subplots_adjust(left=0.20, right=0.985, top=1 - TOP / H_in, bottom=BOT / H_in, wspace=0.08)
    D = 0.21                                       # low effort above, high effort below the row centre
    flagged_any = False
    for yy, (m, lab, per) in zip(ys, rows_):
        col = MODELS[m]["colour"]
        mk = MODELS[m]["marker"]
        for e, dy in (("low", -D), ("high", D)):
            cost, acc, lo, hi, ok, flag = per[e]
            flagged_any |= flag
            axc.barh(yy + dy, cost, height=0.38, color=tint(col) if e == "low" else col, lw=0)
            axc.text(cost + 0.02, yy + dy, f"${cost:.2f}" if cost >= 0.1 else f"${cost:.3f}", va="center",
                     fontsize=7.8, color=M.INK_1)
            axa.plot([lo * 100, hi * 100], [yy + dy] * 2, color=tint(col) if e == "low" else col, lw=1.3, zorder=2)
            axa.plot(acc * 100, yy + dy, marker=mk, ms=6.5, ls="none", zorder=3,
                     **({"mfc": M.SURFACE, "mec": col, "mew": 1.5} if e == "low" else
                        {"color": col, "mec": M.SURFACE, "mew": 1.0}))
            note = ("  ✓" if ok else "") + ("*" if flag else "")
            axa.text(hi * 100 + 1.5, yy + dy, f"{acc * 100:.1f}%{note}", va="center", fontsize=7.8, color=M.INK_1)
    axc.set_yticks(ys, [lab for _, lab, _ in rows_], fontsize=9)
    for m in MODELS:
        yy = [v for v, r in zip(ys, rows_) if r[0] == m]
        block_label(fig, axc, (yy[0] + yy[-1]) / 2, m)
    axc.set_ylim(ys[-1] + 0.7, -0.75)
    axc.tick_params(axis="y", length=0)
    axc.set_xlim(0, 1.6)
    axc.set_xticks([0, 0.5, 1.0, 1.5], ["$0", "$0.50", "$1.00", "$1.50"])
    axc.set_xlabel("list-price cost per call")
    axa.set_xlim(0, 116)
    axa.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axa.set_xlabel("questions answered correctly (all 360)")
    axa.tick_params(axis="y", length=0)
    for ax, title in ((axc, "Cost of one call"), (axa, "What it gets right")):
        ax.grid(axis="x", color=M.GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        fig.text(ax.get_position().x0, 1 - (TOP - 0.18) / H_in, title, ha="left", va="bottom", fontsize=10.5,
                 fontweight="bold", color=M.INK_1)
    axa.spines["bottom"].set_bounds(0, 100)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axc.legend(handles=[Patch(facecolor=tint(M.INK_2), label="low effort"), Patch(facecolor=M.INK_2, label="high effort")],
               loc="lower right", frameon=False, fontsize=8.5)
    axa.legend(handles=[Line2D([], [], marker="o", ms=6.5, mfc=M.SURFACE, mec=M.INK_2, mew=1.5, ls="none",
                               label="low effort"),
                        Line2D([], [], marker="o", ms=6.5, color=M.INK_2, ls="none", label="high effort")],
               loc="lower right", frameon=False, fontsize=8.5)
    star = ("\n* Haiku at low effort answered from text almost without reasoning: what you get at that setting, but "
            "not a fair reading comparison.") if flagged_any else ""
    header(fig, H_in, "Cost per call and accuracy, at low and high effort",
           "One call = a ~240k-token document plus 120 questions, mean of 3 documents. In each row low effort (light bar, "
           "hollow dot) sits above high effort (solid, filled).\n✓ = no measurable loss vs the same model reading text "
           "at that effort (pre-registered). List prices as if uncached: Opus \\$4 / \\$20, Sonnet \\$2 / \\$10 "
           "per million\ninput / output tokens; Haiku \\$0.10 / \\$0.50 up to 100k-token prompts, \\$0.50 / \\$2.50 "
           "above. Input is priced as a cache write (1.25x),\nas on a document's first call. Not amounts billed." + star)
    save(fig, "claude_qa_cost.png")
    plt.close(fig)
    print("claude_qa_cost.png", {e: {m: {k: round(v, 3) for k, v in c.items()} for m, c in C[e].items()} for e in C})


# ------------------------------------------------------------------------------------------- 5. curve
def fig_curve():
    """Accuracy change (image - text, points) vs the measured token saving, one line per model at high effort, for all
    questions and multi-step questions; 95% Newcombe intervals; the pre-registered +-2-point band shaded."""
    plt = M.chart_style()
    S = summaries()
    H_in = 5.3
    fig, axs = plt.subplots(1, 2, figsize=(10.4, H_in), dpi=150, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.95, top=1 - 1.45 / H_in, bottom=0.62 / H_in, wspace=0.08)
    for ax, (title, key, sub) in zip(axs, [("All questions", "text_minus_image", "360 per point"),
                                           ("Multi-step questions", "complex_text_minus_image", "120 per point")]):
        ax.axhspan(-2, 2, color="#ebeae5", lw=0, zorder=0)
        ax.axhline(0, color=M.AXIS, lw=0.8, zorder=1)
        for m, spec in MODELS.items():
            pts = sorted((r["saving"], r) for k, r in S[m].items() if k.startswith("high/") and "saving" in r)
            x = np.array([1.0] + [sv for sv, _ in pts])
            ch = np.array([0.0] + [-100 * r[key][0] for _, r in pts])
            lo = np.array([0.0] + [-100 * r[key][2] for _, r in pts])
            hi = np.array([0.0] + [-100 * r[key][1] for _, r in pts])
            ax.plot(x, ch, color=spec["colour"], lw=2, zorder=3)
            ax.errorbar(x[1:], ch[1:], yerr=[ch[1:] - lo[1:], hi[1:] - ch[1:]], fmt="none", ecolor=spec["colour"],
                        elinewidth=1.2, capsize=0, zorder=3)
            ax.plot(x, ch, ls="none", marker=spec["marker"], ms=7.5 if spec["marker"] != "D" else 6.5,
                    color=spec["colour"], mec=M.SURFACE, mew=1.4, zorder=4)
            ax.text(x[-1] + 0.1, ch[-1], m, fontsize=8.5, color=M.INK_2, va="center")
        ax.set_xticks([1, 2.19, 2.87, 4.15], ["text", "2.2x", "2.9x", "4.2x"])
        ax.set_xlim(0.8, 4.95)
        ax.set_ylim(-90, 6)
        ax.set_yticks([0, -20, -40, -60, -80], ["0", "−20", "−40", "−60", "−80"])
        ax.grid(axis="y", color=M.GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("fewer input tokens than the same document as text")
        x0 = ax.get_position().x0
        fig.text(x0, 1 - 1.12 / H_in, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=M.INK_1)
        fig.text(x0, 1 - 1.34 / H_in, sub, ha="left", va="top", fontsize=8, color=M.INK_2)
    axs[0].set_ylabel("accuracy change vs text (points)")
    axs[0].text(4.9, 3.2, "within 2 points", ha="right", va="bottom", fontsize=7.5, color=M.INK_2)
    header(fig, H_in, "How fast accuracy falls as the pages get denser: Opus holds longest, Haiku drops first",
           "High effort, same 3 documents (~240k tokens of text each) and 120 questions for every model; 95% intervals. "
           "The shaded band is "
           "the pre-registered lossless margin (2 points).\nEach model is compared with itself reading text "
           "(Opus and Sonnet 99.7%, Haiku 98.1%).")
    save(fig, "claude_qa_curve.png")
    plt.close(fig)
    print("claude_qa_curve.png")


# ------------------------------------------------------------------------------------------- 6. effort
def fig_effort():
    """Accuracy change at low vs high effort for each model x density (models whose low-effort rows are valid)."""
    plt = M.chart_style()
    S = summaries()
    ms = [m for m in MODELS if not any(S[m][f"low/{b}"].get("disengaged") for b, _ in BUDGETS)]
    skipped = [m for m in MODELS if m not in ms]
    rows = [(m, b, lab) for m in ms for b, lab in BUDGETS]
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and rows[i - 1][0] != r[0]:
            y += 0.6
        ys.append(y)
        y += 1
    H_in = 2.2 + 0.42 * y
    fig, axs = plt.subplots(1, 2, figsize=(10.4, H_in), dpi=150, sharey=True)
    fig.subplots_adjust(left=0.2, right=0.97, top=1 - 1.6 / H_in, bottom=0.62 / H_in, wspace=0.1)
    for ax, (title, key) in zip(axs, [("All questions (360 per point)", "text_minus_image"),
                                      ("Multi-step questions (120 per point)", "complex_text_minus_image")]):
        ax.axvspan(-2, 2, color="#ebeae5", lw=0, zorder=0)
        ax.axvline(0, color=M.AXIS, lw=0.8, zorder=1)
        for yy, (m, b, lab) in zip(ys, rows):
            col = MODELS[m]["colour"]
            lo_e, hi_e = (-100 * S[m][f"{e}/{b}"][key][0] for e in ("low", "high"))
            ax.plot([lo_e, hi_e], [yy, yy], color=col, lw=2, zorder=2)
            ax.plot(lo_e, yy, marker="o", ms=8, mfc=M.SURFACE, mec=col, mew=1.8, ls="none", zorder=3)
            ax.plot(hi_e, yy, marker="o", ms=8, color=col, mec=M.SURFACE, mew=1.2, ls="none", zorder=3)
        ax.set_xlim(-52, 6)
        ax.set_xticks([0, -10, -20, -30, -40, -50], ["0", "−10", "−20", "−30", "−40", "−50"])
        ax.grid(axis="x", color=M.GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("accuracy change vs text (points)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        x0 = ax.get_position().x0
        fig.text(x0, 1 - 1.22 / H_in, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=M.INK_1)
    axs[0].set_yticks(ys, [f"{lab} pages" for _, _, lab in rows], fontsize=9)
    axs[0].set_ylim(ys[-1] + 0.7, -0.7)
    for m in ms:
        yy = [v for v, r in zip(ys, rows) if r[0] == m]
        block_label(fig, axs[0], (yy[0] + yy[-1]) / 2, m)
    from matplotlib.lines import Line2D
    axs[1].legend(handles=[Line2D([], [], marker="o", ms=7, mfc=M.SURFACE, mec=M.INK_2, mew=1.6, ls="none",
                                  label="low effort"),
                           Line2D([], [], marker="o", ms=7, color=M.INK_2, ls="none", label="high effort")],
                  loc="upper left", frameon=False, fontsize=8.5)
    note = (" " + ", ".join(skipped) + " is left out: at low effort it barely reasoned over the text (appendix A3).") \
        if skipped else ""
    header(fig, H_in, "High effort narrows the loss once the pages are dense",
           "Accuracy change vs the same model reading text, at low (hollow) and high (filled) effort; same documents "
           "and questions. Shaded: within 2 points.\nAt 2.2x both efforts are within 0.3 points of text; the gap opens at "
           "2.9x and 4.2x." + note)
    save(fig, "claude_qa_effort.png")
    plt.close(fig)
    print("claude_qa_effort.png")


# ------------------------------------------------------------------------------------------- 7. position
def per_question(model: str, effort: str, budget: float) -> list[tuple[dict, dict]]:
    """(image answer, text answer) per question, pooled over the 3 documents."""
    out = []
    pre = MODELS[model]["runs"].rstrip("*")
    for f in sorted(QA.glob(f"{pre}*_{effort}_b{int(round(budget))}/runs.jsonl")):
        rows = {json.loads(x)["arm"]: json.loads(x) for x in f.read_text().splitlines() if x.strip()}
        tq = {p["id"]: p for p in rows["text"]["per_question"]}
        out += [(p, tq[p["id"]]) for p in rows["image"]["per_question"]]
    return out


def fig_position():
    """Single-fact lookups (invented facts and numbers in quoted sentences) by the fifth of the document the fact sits
    in, image arm at 2.9x and 4.2x, high effort. Multi-step questions are left out: each needs facts in several
    places, and the latest of them sits late more often, which mixes question type into position."""
    plt = M.chart_style()
    H_in = 4.9
    fig, axs = plt.subplots(1, 2, figsize=(10.4, H_in), dpi=150, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.95, top=1 - 1.5 / H_in, bottom=0.62 / H_in, wspace=0.08)
    edges = [0.2, 0.4, 0.6, 0.8]
    for ax, (b, lab) in zip(axs, BUDGETS[1:]):
        txt = []
        for m, spec in MODELS.items():
            qs = [(i, t) for i, t in per_question(m, "high", b) if i["kind"] in ("needle", "number")]
            fifth = np.digitize([i["pos"] for i, _ in qs], edges)
            ok = np.array([i["correct"] for i, _ in qs])
            txt += [t["correct"] for _, t in qs]
            ys, los, his = [], [], []
            for k in range(5):
                sel = ok[fifth == k]
                lo, hi = M.wilson(int(sel.sum()), len(sel))
                ys.append(sel.mean() * 100), los.append(lo * 100), his.append(hi * 100)
            x = np.arange(5)
            y_ = np.array(ys)
            ax.plot(x, y_, color=spec["colour"], lw=2, zorder=3)
            ax.errorbar(x, y_, yerr=[np.maximum(0, y_ - np.array(los)), np.maximum(0, np.array(his) - y_)], fmt="none",
                        ecolor=spec["colour"], elinewidth=1.1, capsize=0, zorder=3)
            ax.plot(x, y_, ls="none", marker=spec["marker"], ms=7 if spec["marker"] != "D" else 6,
                    color=spec["colour"], mec=M.SURFACE, mew=1.3, zorder=4, label=m)
        ax.axhline(np.mean(txt) * 100, color=M.INK_2, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(0, np.mean(txt) * 100 + 1.2, f"the three models reading text (pooled): {np.mean(txt) * 100:.1f}%", fontsize=7.8,
                color=M.INK_2, va="bottom")
        ax.set_xticks(range(5), ["first", "second", "third", "fourth", "last"])
        ax.set_xlim(-0.3, 4.3)
        ax.set_ylim(40, 104)
        ax.set_yticks([40, 60, 80, 100], ["40%", "60%", "80%", "100%"])
        ax.grid(axis="y", color=M.GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("fifth of the document the fact is in")
        x0 = ax.get_position().x0
        fig.text(x0, 1 - 1.17 / H_in, f"Pages with {lab} fewer tokens", ha="left", va="top", fontsize=10.5,
                 fontweight="bold", color=M.INK_1)
    axs[0].set_ylabel("single-fact lookups answered from pages")
    axs[0].legend(loc="lower right", frameon=False, fontsize=8.5)
    header(fig, H_in, "No depth effect: a fact on the last pages is found about as often as one on the first",
           "Single-fact lookups only (240 per model and panel, 42-52 per fifth, so only large trends would show), high "
           "effort, 3 documents; 95% intervals. 2.2x is left out (every\nmodel at 98-100%). Multi-step questions are left "
           "out: they need facts in several places, so their position mixes with their difficulty.")
    save(fig, "claude_qa_position.png")
    plt.close(fig)
    print("claude_qa_position.png")


# ------------------------------------------------------------------------------------------- 8. engagement
def fig_engagement():
    """Output tokens per call (log scale), text arm vs page arms, per model and effort. Shows why Haiku's low-effort
    rows are not comparable: it answered the text arm almost without reasoning."""
    plt = M.chart_style()
    rows = [(m, e) for m in MODELS for e in ("high", "low")]
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and rows[i - 1][0] != r[0]:
            y += 0.5
        ys.append(y)
        y += 1
    H_in = 2.0 + 0.42 * y
    fig, ax = plt.subplots(figsize=(10.4, H_in), dpi=150)
    fig.subplots_adjust(left=0.2, right=0.97, top=1 - 1.3 / H_in, bottom=0.62 / H_in)
    TEXT_C, PAGE_C = M.INK_2, "#2a78d6"
    for yy, (m, e) in zip(ys, rows):
        pre = MODELS[m]["runs"].rstrip("*")
        t, pg = [], []
        for f in sorted(QA.glob(f"{pre}*_{e}_b*/runs.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get("cached") or r["arm"] == "none":
                    continue
                (t if r["arm"] == "text" else pg).append(r["output_tokens"])
        rng = np.random.default_rng(0)
        ax.plot(pg, yy + rng.uniform(-0.12, 0.12, len(pg)), ls="none", marker="o", ms=6, color=PAGE_C, alpha=0.85,
                mec=M.SURFACE, mew=0.8, zorder=3)
        ax.plot(t, [yy - 0.28] * len(t), ls="none", marker="s", ms=6.5, color=TEXT_C, mec=M.SURFACE, mew=0.8,
                zorder=3)
        if m.startswith("Haiku") and e == "low":
            ax.axhspan(yy - 0.5, yy + 0.4, color="#ebeae5", lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(250, 40000)
    ax.set_xticks([300, 1000, 3000, 10000, 30000], ["300", "1k", "3k", "10k", "30k"])
    ax.set_yticks(ys, [f"{e} effort" for _, e in rows], fontsize=9)
    ax.set_ylim(ys[-1] + 0.7, -0.7)
    for m in MODELS:
        yy = [v for v, r in zip(ys, rows) if r[0] == m]
        block_label(fig, ax, (yy[0] + yy[-1]) / 2, m)
    ax.grid(axis="x", color=M.GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("output tokens per call (reasoning + answers, log scale)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="s", ms=6.5, color=TEXT_C, ls="none", label="reading text (3 calls)"),
                       Line2D([], [], marker="o", ms=6, color=PAGE_C, ls="none",
                              label="reading pages (9 calls: 3 densities x 3 documents)")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=8.5)
    header(fig, H_in, "Haiku at low effort hardly reasons over text, but works through pages: its low row is excluded",
           "Output tokens per call for the same 120 questions (shaded row: excluded from the comparison). Opus and "
           "Sonnet spend thousands of output tokens on both;\nHaiku at low effort answered from text with 424-821 tokens "
           "(one call declined 80 of 120 questions) against 0.8-15k on pages.")
    save(fig, "claude_qa_engagement.png")
    plt.close(fig)
    print("claude_qa_engagement.png")


if __name__ == "__main__":
    M.OUT.mkdir(parents=True, exist_ok=True)
    fig_verdict()
    fig_pages()
    fig_breakdown()
    fig_cost()
    fig_curve()
    fig_effort()
    fig_position()
    fig_engagement()
