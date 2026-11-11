"""Figure for results/compaction_scale: accuracy by session kind and cost after reading, per arm.

    python3 scripts/make_compaction_scale_figure.py
"""
import io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_readme_figures as M  # noqa: E402
from PIL import Image  # noqa: E402

S = json.loads((ROOT / "results/compaction_scale/summary.json").read_text())
A = S["arms"]
fix = S.get("post_hoc_fix")
ARMS = [("D", "no compaction (reference)", "#3987e5"), ("A", "normal compaction", "#eb6834"),
        ("B", "compaction + page archive (hooks)", "#1baf7a")]
KINDS = ["research", "code", "logs"]
plt = M.chart_style()
H = 4.9
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, H), dpi=150, gridspec_kw={"width_ratios": [1.7, 1]})
fig.subplots_adjust(left=0.065, right=0.985, top=1 - 1.45 / H, bottom=0.62 / H, wspace=0.28)
w = 0.26
for i, (k, lab, col) in enumerate(ARMS):
    xs = [j + (i - 1) * (w + 0.02) for j in range(len(KINDS))]
    ys = [100 * A[k]["by_kind"][kk] for kk in KINDS]
    ax.bar(xs, ys, w, color=col, label=lab, edgecolor=M.SURFACE, lw=1.2)
    for x, y in zip(xs, ys):
        ax.text(x, y + 1.2, f"{y:.0f}", ha="center", va="bottom", fontsize=8, color=M.INK_2)
if fix:
    j = KINDS.index("code")
    x = j + (w + 0.02)
    ax.scatter([x], [100 * fix["code_acc_Bfix"]], marker="_", s=260, color=M.INK_1, lw=2, zorder=4)
    ax.annotate(f"fixed hook\n(post hoc): {100 * fix['code_acc_Bfix']:.0f}", (x, 100 * fix["code_acc_Bfix"]),
                xytext=(0, 7), textcoords="offset points", fontsize=7.5, color=M.INK_1, ha="center", va="bottom")
ax.set_xticks(range(len(KINDS)))
ax.set_xticklabels(KINDS)
ax.set_ylim(0, 135)
ax.set_yticks(range(0, 101, 20))
ax.set_ylabel("questions right after compaction (%)")
ax.grid(axis="y", color=M.GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=False, bbox_to_anchor=(0, 1.0))
costs = [A[k]["after_reading_cost"] / 9 for k, _, _ in ARMS]
ys = range(len(ARMS))
bx.barh(list(ys), costs, 0.55, color=[c for _, _, c in ARMS], edgecolor=M.SURFACE, lw=1.2)
for y, (k, lab, _), c in zip(ys, ARMS, costs):
    bx.text(c + 0.05, y, f"${c:.2f}  ({100 * A[k]['acc']:.0f}% right)", va="center", fontsize=8.5, color=M.INK_1)
bx.set_yticks(list(ys))
bx.set_yticklabels([lab.replace(" (", "\n(").replace(" + ", "\n+ ") for _, lab, _ in ARMS], fontsize=8.5)
bx.invert_yaxis()
bx.set_xlim(0, max(costs) * 1.55)
bx.set_xlabel("$ per session after reading")
bx.grid(axis="x", color=M.GRID, lw=0.8)
bx.set_axisbelow(True)
fig.text(0.015, 1 - 0.18 / H, "At scale, a page archive written by compaction hooks keeps recall near 100% at ~60% of "
         "normal compaction's cost", ha="left", va="top", fontsize=12, fontweight="bold", color=M.INK_1)
fig.text(0.015, 1 - 0.52 / H, "Opus 5.5 in Claude Code; 9 sessions (research reading, codebase exploration, log triage; "
         "~270-410k characters each), /compact, then 4 turns of 10 exact-answer questions\n(120 per session kind) and a 5-item task.\nNormal compaction "
         "re-read the raw transcript (160 reads). Compaction is charged equally to both compacting arms; reading "
         "(shared, ~$1 per session) is not included.", ha="left", va="top", fontsize=8.5, color=M.INK_2,
         linespacing=1.45)
buf = io.BytesIO(); fig.savefig(buf, dpi=150, format="png"); buf.seek(0)
Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(
    M.OUT / "compaction_scale.png", optimize=True)
print("compaction_scale.png")
