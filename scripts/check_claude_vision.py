"""WP-E1 vision checks: token accounting, legibility sanity curve, latency/cost, concurrency,
determinism, hidden auxiliary calls. Writes results/claude_checks/*.json.

    python scripts/check_claude_vision.py [all|tokens|legibility|determinism|hidden] [--workers 8]

Spends real money (~$0.3 for ``all``). Results of the legibility run are cached
(results/cache/claude), token/determinism runs deliberately bypass the cache.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vbpe import claude_eval as ce  # noqa: E402
from vbpe import config, metrics  # noqa: E402
from vbpe.layout import fit_em, render_page  # noqa: E402
from vbpe.schemes import get_scheme  # noqa: E402

OUT = config.RESULTS / "claude_checks"
IMG = OUT / "img"

# ~1500 chars of original (not memorisable) English prose, ASCII only.
PARAGRAPH = (
    "The harbour office keeps a ledger of every lamp that has ever burned on the eastern cliffs. "
    "Most entries are dull: a date, a keeper's initials, the number of gallons of oil delivered by "
    "the supply boat, and a short remark about the weather. In the winter of the great storm, "
    "however, the handwriting changes. The keeper wrote that the glass in the lantern room cracked "
    "twice in one night, that the gulls sheltered on the gallery rail like a row of grey coats, and "
    "that a fishing schooner called the Margaret Hale drifted past so close he could read the "
    "letters on her stern. He lit a second lamp in the kitchen window as a spare, because the main "
    "light was failing whenever the wind found a gap in the shutters. By morning the schooner had "
    "anchored behind the reef and her crew of seven rowed ashore, cold but unhurt. They stayed for "
    "nine days, eating salted cod and potatoes, mending nets, and teaching the keeper's daughter to "
    "splice rope. When the supply boat returned it carried a letter of thanks from the owners, a "
    "new brass burner, and forty-two pounds in coins collected by the families of the crew. The "
    "keeper recorded the sum to the penny, then added a line that no inspector ever questioned: "
    "spent on paint, glass, and a better kettle. Visitors who climb the tower today can still see "
    "the patched pane, slightly greener than the others, facing the reef where the Margaret Hale "
    "waited out the weather."
)

TOKEN_PROMPT = "Reply with the single word OK."
TOKEN_SYSTEM = "You are a test harness. Reply with exactly one word."


def save(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1))
    print(f"wrote {p}")
    return p


def slim(r: dict) -> dict:
    keys = ("text", "model", "input_tokens", "output_tokens", "cost_usd", "latency_s", "cached",
            "error", "usage", "thinking_tokens", "model_usage", "attempts", "rate_limit")
    return {k: r.get(k) for k in keys}


# --------------------------------------------------------------------------- a. tokens
def raw_call(images: list[Image.Image], system: str, prompt: str) -> dict:
    """Bypass the size assertion (to probe what the API does above the limits)."""
    pngs = [ce.png_bytes(i) for i in images]
    t0 = time.monotonic()
    rc, out, err = ce._run_cli(ce.build_command(system, ce.DEFAULT_MODEL), ce.build_message(pngs, prompt),
                               ce.build_env(), 300)
    res, am = ce.parse_stream(out)
    if res is None:
        return {"error": f"rc={rc} {err[-300:]}"}
    return ce._record(res, am, ce.DEFAULT_MODEL, time.monotonic() - t0, 1)


def check_tokens() -> dict:
    sizes = [(150, 150), (300, 300), (600, 600), (1000, 1000), (1072, 1072), (1568, 733), (500, 1000)]
    base = ce.transcribe([], TOKEN_SYSTEM, TOKEN_PROMPT, use_cache=False)
    rows = []
    for W, H in sizes:
        r = ce.transcribe([Image.new("L", (W, H), 255)], TOKEN_SYSTEM, TOKEN_PROMPT, use_cache=False)
        rows.append({"W": W, "H": H, "area": W * H, "predicted": W * H / 750, **slim(r)})
    # mode check: same size as RGB vs L should cost the same
    r_rgb = ce.transcribe([Image.new("RGB", (600, 600), (255, 255, 255))], TOKEN_SYSTEM, TOKEN_PROMPT,
                          use_cache=False)
    rows.append({"W": 600, "H": 600, "area": 360000, "predicted": 480, "mode": "RGB", **slim(r_rgb)})
    # above the limit (assert bypassed): does the API downscale (tokens saturate)?
    over = []
    for W, H in [(1400, 1400), (2000, 1000)]:
        r = raw_call([Image.new("L", (W, H), 255)], TOKEN_SYSTEM, TOKEN_PROMPT)
        over.append({"W": W, "H": H, "area": W * H, "predicted": W * H / 750, **slim(r)})
    b = base["input_tokens"]
    for row in rows + over:
        if row.get("input_tokens"):
            row["image_tokens"] = row["input_tokens"] - b
            row["ratio_to_predicted"] = round(row["image_tokens"] / row["predicted"], 4)
            row["px_per_token"] = round(row["area"] / max(1, row["image_tokens"]), 1)
    out = {"baseline": slim(base), "rows": rows, "over_limit": over}
    save("tokens", out)
    for row in rows + over:
        print(f"{row['W']:>5}x{row['H']:<5} pred {row['predicted']:7.1f}  img_tok {row.get('image_tokens')}  "
              f"ratio {row.get('ratio_to_predicted')}  ${row.get('cost_usd')}")
    return out


# --------------------------------------------------------------------------- b. legibility
def canvas_for_em(units, target: float) -> int:
    """Smallest square side W with fit_em(W, W) >= target."""
    lo, hi = 20, 1500
    while lo < hi:
        mid = (lo + hi) // 2
        if fit_em(units, mid, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def legibility_jobs(scheme_names=("latin", "latin_condensed"), ems=(4, 5, 6, 7, 8, 10, 12)):
    IMG.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sname in scheme_names:
        if sname == "latin_condensed" and not config.CONDENSED_FONT.exists():
            continue
        s = get_scheme(sname)
        units = s.encode(PARAGRAPH)
        for em in ems:
            W = canvas_for_em(units, em)
            img, info = render_page(units, W, W)
            img.save(IMG / f"{sname}_em{em}.png")
            jobs.append({"images": [img], "system": s.vlm_system_prompt(), "prompt": s.vlm_prompt(),
                         "meta": {"scheme": sname, "target_em": em, "em": info["em"], "W": W, "H": W,
                                  "n_lines": info["n_lines"],
                                  "px_per_char": metrics.px_per_char(W, W, PARAGRAPH),
                                  "vision_tokens": metrics.vision_tokens(W, W)}})
    return jobs


def check_legibility(workers: int = 8) -> dict:
    jobs = legibility_jobs()
    t0 = time.monotonic()
    res = ce.transcribe_many(jobs, workers=workers)
    wall = time.monotonic() - t0
    rows = []
    for r in res:
        m = r["meta"]
        rows.append({**m, "cer": metrics.cer(PARAGRAPH, r["text"]), **slim(r)})
    out = {"n_chars": len(metrics.normalize(PARAGRAPH)), "workers": workers, "wall_s": wall,
           "errors": sum(r["error"] is not None for r in res), "rows": rows}
    save("legibility", out)
    print(f"{'scheme':16} em    W   px2/char  CER    out_tok  lat_s  $")
    for r in rows:
        print(f"{r['scheme']:16} {r['em']:5.2f} {r['W']:4d} {r['px_per_char']:7.2f}  {r['cer']:.3f}  "
              f"{r['output_tokens']:6d}  {r['latency_s']:5.1f}  {r['cost_usd']:.4f}")
    return out


# --------------------------------------------------------------------------- d. determinism
def check_determinism(reps: int = 3, ems=(6, 8)) -> dict:
    jobs = [j for j in legibility_jobs(("latin",), ems)]
    rows = []
    for j in jobs:
        outs = []
        for _ in range(reps):
            jj = {k: v for k, v in j.items() if k != "meta"}
            outs.append(ce.transcribe(**jj, use_cache=False))
        texts = [o["text"] for o in outs]
        pair = [metrics.cer(texts[a], texts[b]) for a in range(reps) for b in range(a + 1, reps)]
        rows.append({**j["meta"], "texts": texts, "cer_vs_ref": [metrics.cer(PARAGRAPH, t) for t in texts],
                     "pairwise_cer": pair, "identical": len(set(texts)) == 1,
                     "cost_usd": sum(o["cost_usd"] for o in outs),
                     "errors": [o["error"] for o in outs]})
        print(f"em {j['meta']['em']}: CER vs ref {rows[-1]['cer_vs_ref']}  pairwise {pair}")
    out = {"reps": reps, "rows": rows}
    save("determinism", out)
    return out


# --------------------------------------------------------------------------- e. hidden calls
def check_hidden() -> dict:
    """Aggregate per-model usage over every call in the other check files."""
    models: dict[str, dict] = {}
    n = 0
    for name in ("tokens", "legibility", "determinism"):
        p = OUT / f"{name}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        recs = d.get("rows", []) + d.get("over_limit", []) + ([d["baseline"]] if "baseline" in d else [])
        for r in recs:
            for m, v in (r.get("model_usage") or {}).items():
                agg = models.setdefault(m, {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
                agg["calls"] += 1
                agg["in"] += v["in"]
                agg["out"] += v["out"]
                agg["cost"] += v["cost"]
            n += 1
    out = {"records": n, "models": models}
    save("hidden_calls", out)
    print(json.dumps(out, indent=1))
    return out


def summarize_cost() -> dict:
    """Least-squares cost model cost = a + b*input_tokens + c*output_tokens over fresh calls."""
    recs = []
    for name in ("tokens", "legibility"):
        p = OUT / f"{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            recs += [r for r in d.get("rows", []) + d.get("over_limit", [])
                     if r.get("cost_usd") and not r.get("cached")]
    if len(recs) < 3:
        return {}
    X = np.array([[1.0, r["input_tokens"], r["output_tokens"]] for r in recs])
    y = np.array([r["cost_usd"] for r in recs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    lat = np.array([[1.0, r["output_tokens"]] for r in recs if r.get("latency_s")])
    ly = np.array([r["latency_s"] for r in recs if r.get("latency_s")])
    lcoef, *_ = np.linalg.lstsq(lat, ly, rcond=None)
    out = {"n": len(recs), "cost_fixed_usd": coef[0], "usd_per_Minput": coef[1] * 1e6,
           "usd_per_Moutput": coef[2] * 1e6, "latency_fixed_s": lcoef[0],
           "latency_per_output_token_s": lcoef[1]}
    save("cost_model", out)
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all",
                    choices=["all", "tokens", "legibility", "determinism", "hidden", "cost"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.what in ("all", "tokens"):
        check_tokens()
    if a.what in ("all", "legibility"):
        check_legibility(a.workers)
    if a.what in ("all", "determinism"):
        check_determinism()
    if a.what in ("all", "hidden"):
        check_hidden()
    if a.what in ("all", "cost"):
        summarize_cost()


if __name__ == "__main__":
    main()
