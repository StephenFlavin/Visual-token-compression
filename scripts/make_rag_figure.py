"""Figure for results/rag: accuracy against cost per question for every arm, one panel per dataset.

    python3 scripts/make_rag_figure.py
"""
import io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_readme_figures as M  # noqa: E402
from PIL import Image  # noqa: E402

S = json.loads((ROOT / "results/rag/summary.json").read_text())
ARMS = [("closed", "no retrieval", "#9b9a93", "x"), ("T5", "top-5 text", "#3987e5", "s"),
        ("P5", "top-5 pages", "#1baf7a", "o"), ("Pmatch", "budget-matched pages", "#1baf7a", "D"),
        ("T15", "top-15 text", "#6250d6", "s"), ("hybrid", "top-3 text + pages", "#eb6834", "^")]
TITLES = {"planted": "Planted facts (60)", "multihop": "MultiHop-RAG (148)", "hotpot": "HotpotQA (100)"}
plt = M.chart_style()
H = 4.9
fig, axs = plt.subplots(1, 3, figsize=(10.4, H), dpi=150, sharey=False)
fig.subplots_adjust(left=0.07, right=0.98, top=1 - 1.35 / H, bottom=0.95 / H, wspace=0.28)
for ax, ds in zip(axs, ("planted", "multihop", "hotpot")):
    for key, lab, col, mk in ARMS:
        x = S[ds][key]
        ax.scatter([100 * x["cost_per_q"]], [100 * x["em"]], s=70 if key != "closed" else 55, color=col, marker=mk,
                   edgecolor=M.SURFACE if mk not in "x" else None, lw=1.2, zorder=3)
    ys = [100 * S[ds][k]["em"] for k, *_ in ARMS]
    ax.set_ylim(min(ys) - 8, max(ys) + 8)
    ax.set_xlim(0, 100 * max(S[ds][k]["cost_per_q"] for k, *_ in ARMS) * 1.15)
    ax.set_title(TITLES[ds], fontsize=10, fontweight="bold", color=M.INK_1, loc="left")
    ax.set_xlabel("list price per question (cents)")
    ax.grid(color=M.GRID, lw=0.8)
    ax.set_axisbelow(True)
axs[0].set_ylabel("answers right (%)")
handles = [plt.Line2D([], [], ls="none", marker=mk, color=col, markersize=7) for _, _, col, mk in ARMS]
fig.legend(handles, [lab for _, lab, *_ in ARMS], loc="lower center", ncol=6, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2)
fig.text(0.02, 1 - 0.18 / H, "RAG with page images: the same passages cost 27-43% less; the same budget buys more passages",
         ha="left", va="top", fontsize=12, fontweight="bold", color=M.INK_1)
fig.text(0.02, 1 - 0.5 / H, "Opus 5.5, one call per question, BM25 retrieval, pages at 2.9x. Up and to the left is better. "
         "Green circle vs blue square: same top-5\npassages as pages vs text. Green diamond: as many passages as fit in "
         "the top-5's token count as pages (11-14). Cost includes the CLI's fixed prompt.",
         ha="left", va="top", fontsize=8.5, color=M.INK_2, linespacing=1.45)
buf = io.BytesIO(); fig.savefig(buf, dpi=150, format="png"); buf.seek(0)
Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(M.OUT / "rag_cost_accuracy.png", optimize=True)
print("rag_cost_accuracy.png")
