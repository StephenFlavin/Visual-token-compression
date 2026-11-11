"""Figure for results/compaction: accuracy against the cost of answering after compaction, per arm and corpus.

    python3 scripts/make_compaction_figure.py
"""
import io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_readme_figures as M  # noqa: E402
from PIL import Image  # noqa: E402

S = json.loads((ROOT / "results/compaction/summary.json").read_text())["per_arm"]
ARMS = [("A_normal_compaction", "normal compaction\n(summary + raw transcript)", "#eb6834", "s"),
        ("B_page_archive", "+ page archive (2.9x)", "#1baf7a", "o"),
        ("C_text_archive", "+ text archive", "#6250d6", "D"),
        ("D_no_compaction", "no compaction\n(reference)", "#3987e5", "^")]
plt = M.chart_style()
H = 4.8
fig, ax = plt.subplots(figsize=(10.4, H), dpi=150)
fig.subplots_adjust(left=0.08, right=0.97, top=1 - 1.2 / H, bottom=0.7 / H)
for key, lab, col, mk in ARMS:
    pc = S[key]["per_corpus"]
    ax.scatter([p["cost"] for p in pc], [100 * p["acc"] for p in pc], s=46, color=col, marker=mk, alpha=0.45, lw=0)
    mx, my = S[key]["cost"] / 3, 100 * S[key]["acc"]
    ax.scatter([mx], [my], s=120, color=col, marker=mk, edgecolor=M.SURFACE, lw=1.6, zorder=3)
    dy = -11 if key == "C_text_archive" else (6 if key != "A_normal_compaction" else 8)
    ax.annotate(f"{lab}\n{my:.0f}%, ${mx:.2f} per corpus", (mx, my), xytext=(8, dy), textcoords="offset points",
                fontsize=8.5, color=M.INK_1, va="center")
ax.set_xlim(0, 4.2)
ax.set_ylim(0, 108)
ax.set_xlabel("list-price cost of answering 120 questions after compaction ($ per corpus)")
ax.set_ylabel("questions right (%)")
ax.grid(color=M.GRID, lw=0.8)
ax.set_axisbelow(True)
fig.text(0.02, 1 - 0.18 / H, "After compaction, a page archive restores full recall at under half the cost of a text archive",
         ha="left", va="top", fontsize=12, fontweight="bold", color=M.INK_1)
fig.text(0.02, 1 - 0.5 / H, "Opus 5.5 in Claude Code; 3 long sessions (~270k tokens of sources each) compacted with /compact, then "
         "120 exact-answer questions.\nLarge markers: mean over the 3 corpora; small: each corpus. Reading the sources and "
         "compacting (shared by all arms) cost ~$3.5 per corpus and are not included.",
         ha="left", va="top", fontsize=8.5, color=M.INK_2, linespacing=1.45)
buf = io.BytesIO(); fig.savefig(buf, dpi=150, format="png"); buf.seek(0)
Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(M.OUT / "compaction_cost_accuracy.png", optimize=True)
print("compaction_cost_accuracy.png")
