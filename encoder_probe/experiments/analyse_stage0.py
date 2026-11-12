"""Stage 0 (docs/gpu/OWN_ENCODER.md): results table, D3/D10 and val curves from encoder_probe/results/stage0/runs.jsonl.

    python3 encoder_probe/experiments/analyse_stage0.py [runs.jsonl ...] [--out encoder_probe/img/stage0_curves.png]

Prints a markdown table (one line per (ratio, steps) row, later row wins). D3 / D10 are taken per step budget
over the arms that ran at it. Test-window CER is reported, never used to choose anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order


def load(paths) -> list[dict]:
    rows = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("phase") == "stage0" and not r.get("smoke"):
                    rows[(r["ratio"], r["steps"])] = r
    return sorted(rows.values(), key=lambda r: (r["steps"], r["ratio"]))


def table(rows) -> str:
    h = ("| ratio | steps | chars/slot | CER | blind CER | loss | blind loss | gain (nats/tok) | prefix_frac "
         "| exact windows | s/step | peak GiB |")
    out = [h, "|" + "---|" * (h.count("|") - 1)]
    for r in rows:
        out.append(f"| {r['ratio']} | {r['steps']} | {r['chars_per_slot']:.1f} | {100 * r['cer']:.1f}% "
                   f"| {100 * r['blind_cer']:.1f}% | {r['loss']:.3f} | {r['blind_loss']:.3f} "
                   f"| {r['blind_loss'] - r['loss']:.3f} | {r['ops']['prefix_frac']:.3f} "
                   f"| {round(r['ops']['exact_windows'] * r['n_eval_windows'])}/{r['n_eval_windows']} "
                   f"| {r['s_per_step']:.2f} | {r['peak_gpu_bytes'] / 2 ** 30:.1f} |")
    return "\n".join(out)


def density(rows) -> str:
    lines = []
    for steps in sorted({r["steps"] for r in rows}):
        rs = [r for r in rows if r["steps"] == steps]
        for bar in (0.03, 0.10):
            ok = [r["chars_per_slot"] for r in rs if r["cer"] <= bar]
            arms = ", ".join(str(r["ratio"]) for r in rs)
            lines.append(f"* {steps} steps (ratios {arms}): D{round(bar * 100)} = "
                         + (f"{max(ok):.1f} chars/slot" if ok else "none (no arm reaches it)"))
    return "\n".join(lines)


def curves(rows, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    for i, r in enumerate(rows):
        c = SERIES[i % len(SERIES)]
        st = [p[0] for p in r["curve"]]
        lab = f"ratio {r['ratio']}, {r['steps']} steps"
        ax.plot(st, [p[2] for p in r["curve"]], color=c, lw=2, label=f"{lab}: real")
        ax.plot(st, [p[3] for p in r["curve"]], color=c, lw=2, ls="--", label=f"{lab}: blind")
    ax.set_xlabel("training step")
    ax.set_ylabel("val reconstruction loss (nats/token)")
    ax.set_ylim(0, 8)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Stage 0: frozen Qwen decoder rebuilding 480-token windows from m slots", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=[str(ROOT / "encoder_probe/results/stage0/runs.jsonl")])
    ap.add_argument("--out", default=str(ROOT / "encoder_probe/img/stage0_curves.png"))
    a = ap.parse_args(argv)
    rows = load(a.runs)
    print(table(rows))
    print()
    print(density(rows))
    print()
    for r in rows:
        print(f"ratio {r['ratio']}, {r['steps']} steps, val curve (step, real, blind): "
              + ", ".join(f"{s}: {v:.2f}/{b:.2f}" for s, _, v, b in r["curve"]))
    curves(rows, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
