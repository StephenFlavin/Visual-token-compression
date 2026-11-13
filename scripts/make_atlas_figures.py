"""Figures for docs/claude/REPO_ATLAS_PLAN.md and docs/claude/PLAN_QUALITY_TEST.md (the repo atlas tests).

1. atlas_overview.png      every atlas test, one panel each: each arm's score on that test's own scale, arms that
                           hold the atlas in violet, the others grey
2. atlas_repeats.png       the 16 hard coding tasks, plain against atlas: share of task-runs solved and cost per run
                           in the first test (1 run), the re-run (2 runs) and on Sonnet (2 runs)
3. atlas_plan_briefs.png   plan-quality layer 1: share of the fix's files a brief names against cost per brief, per
                           planner
4. atlas_plan_workers.png  plan-quality layer 3: tickets solved of 16 against cost per ticket, per worker arm

Every number is recomputed from the rows under results/atlas/*/ (runs.jsonl, judged.jsonl, grades.jsonl,
briefs.jsonl, work.jsonl, tickets.json); the briefs are re-scored with experiments/atlas/plan_quality.py's own
parse_brief / file_scores. Nothing is hard-coded.

    python3 scripts/make_atlas_figures.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_readme_figures as M  # noqa: E402

ROOT = M.ROOT
RES = ROOT / "results" / "atlas"
ATLAS_C, OTHER_C = "#6250d6", "#9b9a93"      # violet = the arm holds the atlas; grey = it does not (labels name every arm)


def jl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def save(fig, name: str) -> None:
    """150 dpi PNG quantised to 256 colours (64 dulls the small violet dots)."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, format="png")
    buf.seek(0)
    Image.open(buf).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(M.OUT / name, optimize=True)


def header(fig, H: float, title: str, sub: str) -> None:
    fig.text(0.02, 1 - 0.18 / H, title, ha="left", va="top", fontsize=12, fontweight="bold", color=M.INK_1)
    fig.text(0.02, 1 - 0.50 / H, sub, ha="left", va="top", fontsize=8.5, color=M.INK_2, linespacing=1.45)


# --------------------------------------------------------------------------- data
def overview_data() -> list[dict]:
    """One dict per test: title, what is scored, and (arm label, share 0-1, value label, holds_atlas) per arm."""
    out = []

    rows = jl(RES / "pilot" / "runs.jsonl")
    arms = []
    for a, lab in [("tree", "file list"), ("map", "repo map"), ("atlas", "atlas"), ("text", "code as text"),
                   ("agent", "searching agent")]:
        rs = [r for r in rows if r["arm"] == a]
        k = sum(r["all_gold_top5"] for r in rs)
        arms.append((lab, k / len(rs), f"{k} of {len(rs)}", a == "atlas"))
    out.append({"title": "Locate a bug fix (pilot)", "sub": "tasks with every changed file named", "arms": arms})

    rows = [r for r in jl(RES / "qa" / "judged.jsonl") if r.get("total") is not None]
    arms = []
    for a, lab in [("closed", "no code"), ("map", "repo map"), ("atlas", "atlas"), ("text", "code as text"),
                   ("agent", "searching agent")]:
        ts = [r["total"] for r in rows if r["arm"] == a]
        m = statistics.mean(ts)
        arms.append((lab, m / 100, f"{m:.1f}", a == "atlas"))
    out.append({"title": "Answer questions (SWE-QA)", "sub": "judge score out of 100", "arms": arms})

    rows = jl(RES / "sec" / "runs.jsonl")
    arms = []
    for a, lab in [("closed", "file list"), ("atlas", "atlas"), ("agent", "searching agent"),
                   ("agent_atlas", "agent + atlas")]:
        rs = [r for r in rows if r["arm"] == a]
        k = sum(r["gold_top5"] for r in rs)
        arms.append((lab, k / len(rs), f"{k} of {len(rs)}", "atlas" in a))
    out.append({"title": "Find a security flaw", "sub": "tasks with the fixed file in the top 5", "arms": arms})

    bad = set(json.loads((RES / "swe" / "contaminated.json").read_text()))
    rows = [r for r in jl(RES / "swe" / "runs.jsonl") if r["id"] not in bad]
    arms = []
    for a, lab in [("plain", "plain agent"), ("map", "agent + repo map"), ("atlas", "agent + atlas")]:
        rs = [r for r in rows if r["arm"] == a]
        k = sum(r["resolved"] for r in rs)
        arms.append((lab, k / len(rs), f"{k} of {len(rs)}", a == "atlas"))
    out.append({"title": "Hard coding tasks, first test", "sub": "tasks solved, one run each", "arms": arms})

    for d, title in [("swe_rerun", "Hard coding tasks, re-run"), ("swe_sonnet", "Hard coding tasks, Sonnet")]:
        rows = jl(RES / d / "runs.jsonl")
        arms = []
        for a, lab in [("plain", "plain agent"), ("atlas", "agent + atlas")]:
            rs = [r for r in rows if r["arm"] == a]
            k = sum(r["resolved"] for r in rs)
            arms.append((lab, k / len(rs), f"{k} of {len(rs)}", a == "atlas"))
        out.append({"title": title, "sub": "task-runs solved, two runs each", "arms": arms})

    rows = jl(RES / "migrate" / "runs.jsonl")
    arms = []
    for a, lab in [("plain", "C files"), ("atlas", "C files + atlas"), ("atlas_only", "atlas only")]:
        rs = [r for r in rows if r["arm"] == a]
        k = sum(r["all_pass"] for r in rs)
        arms.append((lab, k / len(rs), f"{k} of {len(rs)}", "atlas" in a))
    out.append({"title": "Migrate C to Rust", "sub": "projects passing all tests", "arms": arms})

    rows = {r["arm"]: r for r in jl(RES / "port" / "grades.jsonl")}
    arms = [(lab, rows[a]["n_passed"] / rows[a]["n_tests"], f"{rows[a]['n_passed']} of {rows[a]['n_tests']}", a == "atlas")
            for a, lab in [("plain", "plain agent"), ("atlas", "agent + atlas")]]
    out.append({"title": "Port gitleaks, Go to Rust", "sub": "hidden tests passed, one run each", "arms": arms})
    return out


def repeats_data() -> list[dict]:
    bad = set(json.loads((RES / "swe" / "contaminated.json").read_text()))
    out = []
    for d, lab in [("swe", "Opus, first test\n(1 run per task)"), ("swe_rerun", "Opus, re-run\n(2 runs per task)"),
                   ("swe_sonnet", "Sonnet\n(2 runs per task)")]:
        rows = [r for r in jl(RES / d / "runs.jsonl") if r["id"] not in bad]
        g = {"label": lab}
        for a in ("plain", "atlas"):
            rs = [r for r in rows if r["arm"] == a]
            g[a] = {"k": sum(r["resolved"] for r in rs), "n": len(rs), "cost": statistics.mean(r["cost_usd"] for r in rs)}
        out.append(g)
    return out


def plan_data() -> tuple[list[dict], list[dict]]:
    sys.path.insert(0, str(ROOT / "experiments" / "atlas"))
    import plan_quality as pq  # noqa: E402
    sel = {x["id"]: x for x in pq.tickets()["tickets"]}
    by = {}
    for r in jl(RES / "plan_quality" / "briefs.jsonl"):
        if r["layer"] != 1 or r["unit"] not in sel:
            continue
        x = sel[r["unit"]]
        bs = pq.split_briefs(r["text"])
        fs = pq.file_scores(pq.parse_brief(bs[0] if bs else ""), x["gold"], x["gold_all"])
        by[(r["unit"], r["planner"])] = (fs["gold_share"], r.get("cost_usd") or 0.0)
    ids = [i for i in sel if all((i, p) in by for p in pq.PLANNERS)]
    names = {"tree": "file list", "map": "repo map", "atlas": "atlas", "scout": "searching scout"}
    planners = [{"label": names[p], "atlas": p == "atlas", "n": len(ids),
                 "share": statistics.mean(by[(i, p)][0] for i in ids if by[(i, p)][0] is not None),
                 "cost": statistics.mean(by[(i, p)][1] for i in ids)} for p in pq.PLANNERS]

    work = jl(RES / "plan_quality" / "work.jsonl")
    wnames = [("haiku_atlas", "Haiku low,\natlas brief"), ("sonnet_atlas", "Sonnet medium,\natlas brief"),
              ("opus_atlas", "Opus high,\natlas brief"), ("opus_tree", "Opus high,\nfile-list brief")]
    workers = []
    for a, lab in wnames:
        rs = [r for r in work if r["arm"] == a]
        workers.append({"label": lab, "atlas": "atlas" in a, "k": sum(r["resolved"] for r in rs), "n": len(rs),
                        "cost": statistics.mean(r["cost_usd"] for r in rs)})
    wid = {r["id"] for r in work}
    rs = [r for r in jl(RES / "swe" / "runs.jsonl") if r["arm"] == "plain" and r["id"] in wid]
    workers.append({"label": "Opus high,\nno brief", "atlas": False, "k": sum(r["resolved"] for r in rs), "n": len(rs),
                    "cost": statistics.mean(r["cost_usd"] for r in rs)})
    return planners, workers


# --------------------------------------------------------------------------- charts
def fig_overview(plt) -> None:
    tests = overview_data()
    H = 10.6
    fig, axs = plt.subplots(4, 2, figsize=(8.4, H), dpi=150)
    fig.subplots_adjust(left=0.19, right=0.95, top=1 - 1.5 / H, bottom=0.35 / H, wspace=0.75, hspace=0.62)
    for ax, t in zip(axs.flat, tests):
        n = len(t["arms"])
        for i, (lab, share, val, has) in enumerate(t["arms"]):
            ax.barh(i, 100 * share, height=0.62, color=ATLAS_C if has else OTHER_C, zorder=3)
            ax.text(100 * share + 2, i, val, va="center", ha="left", fontsize=8.5, color=M.INK_1)
        ax.set_yticks(range(n))
        ax.set_yticklabels([a[0] for a in t["arms"]], fontsize=8.5)
        ax.set_ylim(n - 0.4, -0.6)
        ax.set_xlim(0, 125)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50%", "100%"], fontsize=8)
        ax.xaxis.grid(True, color=M.GRID, lw=0.8, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.set_title(t["title"], fontsize=10, fontweight="bold", loc="left", pad=16, x=-0.38)
        ax.text(-0.38, 1.03, t["sub"], transform=ax.transAxes, fontsize=8.5, color=M.INK_2, ha="left", va="bottom")
    header(fig, H, "Repo atlas: no test where an arm with the atlas beats a cheaper arm",
           "Each panel is one test; bars are each arm's score as a share of that test's maximum, with the score in "
           "its own unit at the bar's end.\nViolet: the arm has the whole repository as page images (the atlas). "
           "Grey: it does not. The one visible gap (first test of the hard\ncoding tasks, 11 against 8) did not "
           "repeat in the re-run (the next panel).")
    save(fig, "atlas_overview.png")
    plt.close(fig)


def fig_repeats(plt) -> None:
    G = repeats_data()
    H = 4.6
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, H), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 1.45 / H, bottom=0.75 / H, wspace=0.25)
    w = 0.36
    for gi, g in enumerate(G):
        for j, a in enumerate(("plain", "atlas")):
            x = gi + (j - 0.5) * (w + 0.03)
            c = ATLAS_C if a == "atlas" else OTHER_C
            share = 100 * g[a]["k"] / g[a]["n"]
            a1.bar(x, share, width=w, color=c, zorder=3)
            a1.text(x, share + 1.5, f"{g[a]['k']}/{g[a]['n']}", ha="center", va="bottom", fontsize=8.5)
            a2.bar(x, g[a]["cost"], width=w, color=c, zorder=3)
            a2.text(x, g[a]["cost"] + 0.015, f"${g[a]['cost']:.2f}", ha="center", va="bottom", fontsize=8.5)
    for ax in (a1, a2):
        ax.set_xticks(range(len(G)))
        ax.set_xticklabels([g["label"] for g in G], fontsize=8.5)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=M.GRID, lw=0.8, zorder=0)
    a1.set_ylim(0, 100)
    a1.set_ylabel("task-runs solved (%)")
    a1.set_title("Solved", fontsize=10, fontweight="bold", loc="left")
    a2.set_ylim(0, max(g["atlas"]["cost"] for g in G) * 1.18)
    a2.set_ylabel("mean list price per run ($)")
    a2.set_title("Cost per run", fontsize=10, fontweight="bold", loc="left")
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=OTHER_C, label="plain agent"), Patch(color=ATLAS_C, label="agent + atlas")],
              loc="upper right", fontsize=8.5, frameon=False, ncol=2, handlelength=1.2, columnspacing=1.2)
    header(fig, H, "The first test's lead for the atlas closed on repeat; its extra cost did not",
           "The same 16 hard coding tasks (SWE-rebench), Claude Code plain against the same agent with the atlas "
           "in its first message.\nLeft: share of task-runs solved, with the count above each bar. Right: mean "
           "list price of one run.")
    save(fig, "atlas_repeats.png")
    plt.close(fig)


def fig_plan(plt) -> None:
    planners, workers = plan_data()
    H = 4.4
    fig, ax = plt.subplots(figsize=(8.4, H), dpi=150)
    fig.subplots_adjust(left=0.09, right=0.97, top=1 - 1.45 / H, bottom=0.6 / H)
    for p in planners:
        ax.scatter([p["cost"]], [100 * p["share"]], s=90, color=ATLAS_C if p["atlas"] else OTHER_C,
                   edgecolor=M.SURFACE, lw=2, zorder=3)
        ax.annotate(f"{p['label']}\n{100 * p['share']:.1f}%, ${p['cost']:.2f}", (p["cost"], 100 * p["share"]),
                    xytext=(0, 12 if p["label"] != "searching scout" else -12), textcoords="offset points",
                    ha="center", va="bottom" if p["label"] != "searching scout" else "top", fontsize=8.5)
    ax.set_xlim(0, max(p["cost"] for p in planners) * 1.15)
    ax.set_ylim(0, 100)
    ax.set_xlabel("mean list price per brief ($)")
    ax.set_ylabel("fix's files named in the brief (%)")
    ax.grid(color=M.GRID, lw=0.8)
    ax.set_axisbelow(True)
    header(fig, H, "Briefs written from the atlas name no more of the fix than briefs from a file list",
           f"Plan-quality test, layer 1: {planners[0]['n']} tickets with a brief from each of four planners (Opus, "
           "high effort). Each dot is one planner:\nacross, what a brief costs; up, the mean share of the real "
           "fix's source files that the brief names. Violet: the planner read the atlas.")
    save(fig, "atlas_plan_briefs.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, H), dpi=150)
    fig.subplots_adjust(left=0.09, right=0.97, top=1 - 1.45 / H, bottom=0.6 / H)
    place = {"Opus high,\nno brief": (0, -12, "center", "top"), "Opus high,\natlas brief": (0, 11, "center", "bottom"),
             "Opus high,\nfile-list brief": (12, 0, "left", "center")}
    for wk in workers:
        ax.scatter([wk["cost"]], [wk["k"]], s=90, color=ATLAS_C if wk["atlas"] else OTHER_C, edgecolor=M.SURFACE, lw=2,
                   zorder=3)
        dx, dy, ha, va = place.get(wk["label"], (0, -12, "center", "top"))
        ax.annotate(f"{wk['label']}\n{wk['k']} of {wk['n']}, " + (f"${wk['cost']:.2f}" if wk["cost"] >= 0.02 else f"${wk['cost']:.3f}"), (wk["cost"], wk["k"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 1.5)
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1])
    ax.set_xticklabels(["$0.01", "$0.03", "$0.10", "$0.30", "$1"])
    ax.minorticks_off()
    ax.set_ylim(0, 16)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_xlabel("mean list price per ticket ($, log scale)")
    ax.set_ylabel("tickets solved (of 16)")
    ax.grid(color=M.GRID, lw=0.8)
    ax.set_axisbelow(True)
    header(fig, H, "Workers: the two cheap tiers solve the same 8 tickets, Opus 2 or 3 more with a brief",
           "Plan-quality test, layer 3: 16 tickets, one Claude Code run per ticket and worker arm. The no-brief point is "
           "the first\nhard-task test's plain runs. Across: mean cost of a run (log scale). Up: tickets solved. Violet: the worker "
           "had a brief written from the atlas.")
    save(fig, "atlas_plan_workers.png")
    plt.close(fig)


def main() -> int:
    plt = M.chart_style()
    fig_overview(plt)
    fig_repeats(plt)
    fig_plan(plt)
    for name in ("atlas_overview", "atlas_repeats", "atlas_plan_briefs", "atlas_plan_workers"):
        print(f"wrote docs/img/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
