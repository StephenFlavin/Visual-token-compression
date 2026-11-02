#!/usr/bin/env python3
"""Phase 1: zero-shot Claude transcription sweep over (passage x scheme x pixel budget).

For every passage, scheme and budget B (px^2 per normalised source char) we render the passage on a
square canvas of side round(sqrt(B * n_chars)) at the largest fitting em, ask Claude to transcribe
it, decode the transcription with the scheme and score CER against the source.

Outputs (results/phase1/<run>/):
  runs.jsonl          one row per (passage, scheme, budget); resumable (existing rows are skipped)
  images/<scheme>/    rendered target images (+ _key_<i>.png instruction images, if any)
  config.json         CLI arguments of the latest invocation
  summary.{json,md}, cer_curves.png, p_at_bars.png   (vbpe.analysis, unless --no-analysis)

Examples
  # free, offline smoke test (fake transcriber, built-in sample text if the corpus is missing)
  python3 experiments/phase1_sweep.py --dry-run --schemes latin latin_condensed --budgets 6 18
  # pilot: 1 passage, every registered scheme, wide grid -> prints a cost estimate, needs --yes
  python3 experiments/phase1_sweep.py --pilot --max-cost 10
  python3 experiments/phase1_sweep.py --pilot --max-cost 10 --yes
  # main sweep
  python3 experiments/phase1_sweep.py --variant natural --n-passages 6 --max-cost 40 --yes
  python3 experiments/phase1_sweep.py --variant scrambled --n-passages 3 --max-cost 20 --yes
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vbpe import config, metrics  # noqa: E402
from vbpe.layout import render_page  # noqa: E402

DEFAULT_SCHEMES = ["latin", "latin_condensed", "subst300", "subst1000", "quad3"]
DEFAULT_BUDGETS = [4, 6, 9, 13, 18, 27, 40, 60]
PILOT_BUDGETS = [2, 3, 4, 6, 9, 13, 18, 27, 40, 60, 100]

# Cost heuristic ($ per 1M tokens, input/output), by model-name substring. Used ONLY for the
# pre-flight estimate; real rows record the cost reported by vbpe.claude_eval.
PRICES = [("fable", 10.0, 50.0), ("mythos", 10.0, 50.0), ("opus-5-5", 4.0, 20.0), ("opus", 5.0, 25.0),
          ("sonnet-5", 2.0, 10.0), ("sonnet", 3.0, 15.0), ("haiku-5", 0.10, 0.50), ("haiku", 1.0, 5.0)]
DEFAULT_PRICE = (5.0, 25.0)
CALL_OVERHEAD_USD = 0.013   # measured probe cost of a near-empty `claude -p` image call (PLAN §3)

# Clearly-labelled fallback text for DRY RUNS ONLY when vbpe.corpus is not available.
BUILTIN_SAMPLE_ID = "BUILTIN_SAMPLE"
BUILTIN_SAMPLE = (
    "The river rises in the northern hills and flows south for about two hundred kilometres before "
    "reaching the sea. In the early nineteenth century the valley was known mainly for its wool and "
    "its slate quarries. A railway line was opened in 1868, and within a decade the population of the "
    "largest town had doubled. The old bridge, built of local stone, survived several floods but was "
    "finally replaced in 1931. Local historians disagree about the origin of the name; some connect it "
    "with an old word for a ford, while others argue that it refers to the colour of the water after "
    "heavy rain. During the war the quarries were closed and the workers moved to the coastal "
    "shipyards. After 1950 tourism slowly became the most important industry, and many of the former "
    "mill buildings were converted into hotels, shops and a small museum. The museum holds a "
    "collection of tools, maps and photographs, including the earliest known picture of the bridge. "
    "Each summer a festival is held in the main square, with music, food stalls and a boat race on "
    "the lower part of the river. The race was first recorded in 1894 and has been interrupted only "
    "twice since then. In 2004 the regional council approved a plan to restore the towpath, which "
    "now forms part of a long-distance walking route. Birds such as herons, kingfishers and dippers "
    "are common along the banks, and otters returned to the upper valley in the late 1990s."
)


# -------------------------------------------------------------------------------------- passages
def truncate_text(text: str, n_chars: int | None) -> str:
    """Cut normalised text to ~n_chars, preferring a sentence end within +-25%, else a word boundary."""
    text = metrics.normalize(text)
    if not n_chars or len(text) <= n_chars:
        return text
    ends = [m.end() for m in re.finditer(r"[.!?][\"')\]]*(?=\s)", text)]
    near = [e for e in ends if abs(e - n_chars) <= 0.25 * n_chars]
    if near:
        return text[:min(near, key=lambda e: abs(e - n_chars))].strip()
    cut = text.rfind(" ", 0, n_chars + 1)
    return text[:cut if cut > 0 else n_chars].strip()


def builtin_passages(variant: str, n: int) -> list[dict]:
    sents = re.split(r"(?<=[.!?])\s+", BUILTIN_SAMPLE)
    out = []
    for k in range(max(1, n)):
        rng = random.Random(k)
        s = sents[:]
        if k:
            rng.shuffle(s)
        text = " ".join(s)
        if variant == "scrambled":
            w = text.split()
            rng.shuffle(w)
            text = " ".join(w)
        out.append({"id": f"{BUILTIN_SAMPLE_ID}_{variant}_{k}", "text": text, "builtin": True})
    return out


def get_passages(variant: str, n: int | None, dry_run: bool) -> tuple[list[dict], str]:
    try:
        from vbpe import corpus  # lazy: WP-A
        ps = corpus.load_passages(variant)
        src = "corpus"
    except Exception as e:  # ImportError, missing data, ...
        if not dry_run:
            raise SystemExit(f"vbpe.corpus.load_passages({variant!r}) unavailable ({e!r}); "
                             "real runs need the corpus (scripts/build_data.py).")
        print(f"WARNING: vbpe.corpus unavailable ({type(e).__name__}: {e}); using the BUILT-IN SAMPLE "
              "paragraph (dry run only, not WikiText).", file=sys.stderr)
        ps = builtin_passages(variant, n or 2)
        src = "builtin_sample"
    if n:
        ps = ps[:n]
    return ps, src


# ------------------------------------------------------------------------------- schemes & costs
def load_schemes(names: list[str]) -> dict:
    from vbpe.schemes import get_scheme
    out = {}
    for n in names:
        try:
            out[n] = get_scheme(n)
        except Exception as e:
            print(f"WARNING: scheme {n!r} unavailable, skipped ({type(e).__name__}: {e})", file=sys.stderr)
    return out


def price_for(model: str | None, price_in: float | None, price_out: float | None) -> tuple[float, float]:
    pin, pout = DEFAULT_PRICE
    m = (model or "").lower()
    for sub, a, b in PRICES:
        if sub in m:
            pin, pout = a, b
            break
    return (price_in if price_in is not None else pin), (price_out if price_out is not None else pout)


def est_output_tokens(target: str) -> float:
    ascii_n = sum(1 for c in target if ord(c) < 128)
    return ascii_n / 4.0 + 1.5 * (len(target) - ascii_n) + 20


def est_call_cost(job_info: dict, prices: tuple[float, float], overhead: float) -> float:
    tin = job_info["vision_tokens"] + job_info["key_image_tokens"] + job_info["prompt_tokens"]
    return overhead + tin * prices[0] / 1e6 + job_info["out_tokens_est"] * prices[1] / 1e6


# ----------------------------------------------------------------------------------- fake reader
def _corrupt(target: str, rate: float, rng: random.Random) -> str:
    if rate <= 0 or not target:
        return target
    alphabet = sorted(set(target)) or list("abc")
    out = []
    for ch in target:
        u = rng.random()
        if u < rate * 0.6:
            out.append(rng.choice(alphabet))
        elif u < rate * 0.8:
            continue
        elif u < rate:
            out.append(ch)
            out.append(rng.choice(alphabet))
        else:
            out.append(ch)
    return "".join(out)


def make_fake_transcriber(noise: float, targets: dict[str, dict]):
    """Returns transcribe_many-compatible fn. Corruption rate grows as em shrinks (rate=noise at 8 px)."""
    def transcribe_many(jobs: list[dict], workers: int = 8) -> list[dict]:
        res = []
        for j in jobs:
            meta = j["meta"]
            t = targets[meta["key"]]
            seed = zlib.crc32(meta["key"].encode())
            rng = random.Random(seed)
            pfac = 0.7 + 0.6 * random.Random(zlib.crc32(meta["passage_id"].encode())).random()
            em = max(t["em"], 0.5)
            rate = min(0.95, noise * pfac * (8.0 / em) ** 2)
            text = _corrupt(t["target"], rate, rng)
            res.append({"text": text, "model": "fake", "input_tokens": int(t["vision_tokens"]),
                        "output_tokens": int(est_output_tokens(text)), "cost_usd": 0.0,
                        "latency_s": 0.0, "cached": False, "error": None, "meta": meta})
        return res
    return transcribe_many


# ------------------------------------------------------------------------------------------- run
def job_key(variant, model, passage_id, scheme, budget) -> str:
    return f"{variant}|{model}|{passage_id}|{scheme}|{budget:g}"


def read_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("error") or r.get("skipped"):     # failed calls are retried
                done.add(job_key(r["variant"], r["model"], r["passage_id"], r["scheme"], r["budget"]))
    return done


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schemes", nargs="+", default=None, help=f"default: {' '.join(DEFAULT_SCHEMES)}")
    ap.add_argument("--budgets", nargs="+", type=float, default=None,
                    help=f"px^2/char grid (default: {' '.join(map(str, DEFAULT_BUDGETS))})")
    ap.add_argument("--variant", choices=["natural", "scrambled"], default="natural")
    ap.add_argument("--n-passages", type=int, default=None, help="first N passages (default: all)")
    ap.add_argument("--passage-chars", type=int, default=None,
                    help="truncate passages at a sentence/word boundary to ~N chars")
    ap.add_argument("--model", default=None, help="Claude model (default: vbpe.claude_eval.DEFAULT_MODEL)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-name", default=None, help="default: [dryrun-]pilot | [dryrun-]main-<variant>")
    ap.add_argument("--out-dir", default=str(config.RESULTS / "phase1"), help="parent of the run dir")
    ap.add_argument("--dry-run", action="store_true", help="fake transcriber; never calls Claude")
    ap.add_argument("--fake-noise", type=float, default=0.03,
                    help="dry run: char corruption rate at em=8 px (grows ~1/em^2)")
    ap.add_argument("--pilot", action="store_true",
                    help="preset: 1 passage, all schemes in vbpe.schemes.SCHEMES, budgets 2..100")
    ap.add_argument("--max-cost", type=float, default=None, help="abort if estimated USD exceeds this")
    ap.add_argument("--cost-per-call", type=float, default=None, help="override the per-call estimate (USD)")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (estimate)")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens (estimate)")
    ap.add_argument("--call-overhead", type=float, default=CALL_OVERHEAD_USD,
                    help="fixed USD per call in the estimate (CLI system prompt etc.)")
    ap.add_argument("--yes", action="store_true", help="really submit paid calls after the estimate")
    ap.add_argument("--chunk", type=int, default=None, help="jobs per transcribe_many call (default 4*workers)")
    ap.add_argument("--no-analysis", action="store_true")
    a = ap.parse_args(argv)
    if a.pilot:
        from vbpe.schemes import SCHEMES
        a.schemes = a.schemes or list(SCHEMES)
        a.budgets = a.budgets or list(PILOT_BUDGETS)
        a.n_passages = a.n_passages or 1
    a.schemes = a.schemes or list(DEFAULT_SCHEMES)
    a.budgets = a.budgets or list(DEFAULT_BUDGETS)
    if a.run_name is None:
        a.run_name = ("dryrun-" if a.dry_run else "") + ("pilot" if a.pilot else f"main-{a.variant}")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    run_dir = Path(a.out_dir) / a.run_name
    img_dir = run_dir / "images"
    runs_path = run_dir / "runs.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(a), indent=1, default=str))

    claude_eval = None
    if a.dry_run:
        model = "fake"
    else:
        from vbpe import claude_eval  # lazy: WP-E1
        model = a.model or getattr(claude_eval, "DEFAULT_MODEL", "default")

    passages, src = get_passages(a.variant, a.n_passages, a.dry_run)
    schemes = load_schemes(a.schemes)
    if not schemes:
        print("no usable schemes", file=sys.stderr)
        return 2
    done = read_done(runs_path)

    # ---- build jobs / no-call rows ----
    pending, immediate = [], []        # (job, info) ; finished rows needing no call
    key_tokens: dict[str, float] = {}
    for sname, sch in schemes.items():
        keys = sch.vlm_key_images()
        key_tokens[sname] = sum(metrics.vision_tokens(*im.size) for im in keys)
        if keys:
            (img_dir / sname).mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(keys):
                im.save(img_dir / sname / f"_key_{i}.png")
    for p in passages:
        text = truncate_text(p["text"], a.passage_chars)
        n = len(metrics.normalize(text))
        for sname, sch in schemes.items():
            units = None
            target = None
            for b in a.budgets:
                k = job_key(a.variant, model, p["id"], sname, b)
                if k in done:
                    continue
                side = int(round(math.sqrt(b * n)))
                base = {"run": a.run_name, "phase": 1, "model": model, "variant": a.variant,
                        "passage_id": p["id"], "corpus": src, "scheme": sname, "budget": b,
                        "n_chars": n, "W": side, "H": side, "px_per_char": metrics.px_per_char(side, side, text),
                        "vision_tokens": metrics.vision_tokens(side, side),
                        "key_image_tokens": key_tokens[sname],
                        "text_tokens_est": metrics.text_tokens_est(text)}
                empty = {"output_text": "", "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
                         "latency_s": 0.0, "cached": False}
                if side > config.CLAUDE_MAX_EDGE or side * side > config.CLAUDE_MAX_AREA:
                    immediate.append({**base, "em": None, "n_lines": 0, "fit": None, "skipped": True,
                                      "cer": None, "symbol_cer": None, **empty,
                                      "error": f"too_large: {side}x{side} exceeds Claude limits"})
                    continue
                if units is None:
                    units = sch.encode(text)
                    target = sch.transcription_target(text)
                img, info = render_page(units, side, side)
                (img_dir / sname).mkdir(parents=True, exist_ok=True)
                img.save(img_dir / sname / f"{p['id']}_{b:g}.png")
                base.update(em=info["em"], n_lines=info["n_lines"], fit=bool(info["fit"]))
                if not info["fit"]:
                    immediate.append({**base, "skipped": True, "cer": 1.0, "symbol_cer": 1.0, **empty,
                                      "error": None})
                    continue
                job = {"images": sch.vlm_key_images() + [img], "system": sch.vlm_system_prompt(),
                       "prompt": sch.vlm_prompt(), "meta": {"key": k, "passage_id": p["id"]}}
                if a.model:
                    job["model"] = a.model
                info2 = {"base": base, "text": text, "target": target, "scheme": sch, "em": info["em"],
                         "vision_tokens": base["vision_tokens"], "key_image_tokens": key_tokens[sname],
                         "prompt_tokens": (len(job["system"]) + len(job["prompt"])) / 4 + 10,
                         "out_tokens_est": est_output_tokens(target)}
                pending.append((job, info2))

    # ---- cost estimate & gate ----
    prices = price_for(model, a.price_in, a.price_out)
    per = [a.cost_per_call if a.cost_per_call is not None else est_call_cost(i, prices, a.call_overhead)
           for _, i in pending]
    est = sum(per)
    print(f"run={a.run_name} model={model} variant={a.variant} passages={len(passages)} ({src}) "
          f"schemes={list(schemes)} budgets={[f'{b:g}' for b in a.budgets]}")
    print(f"already done: {len(done)}  no-call rows: {len(immediate)}  calls to make: {len(pending)}")
    print(f"ESTIMATED COST: ${est:.2f} (~${(est / max(1, len(pending))):.3f}/call; prices in/out "
          f"${prices[0]}/${prices[1]} per Mtok + ${a.call_overhead} overhead; claude_eval cache may lower it)")
    if not a.dry_run and pending:
        if a.max_cost is not None and est > a.max_cost:
            print(f"ABORT: estimate ${est:.2f} exceeds --max-cost ${a.max_cost:.2f}", file=sys.stderr)
            return 3
        if not a.yes:
            print("Not submitting: re-run with --yes to make the paid calls.", file=sys.stderr)
            return 1

    with open(runs_path, "a") as f:
        for r in immediate:
            f.write(json.dumps(r) + "\n")

    # ---- transcribe in chunks, append rows as we go ----
    if a.dry_run:
        tmany = make_fake_transcriber(a.fake_noise, {j["meta"]["key"]: i for j, i in pending})
    else:
        tmany = claude_eval.transcribe_many
    chunk = a.chunk or max(4 * a.workers, 8)
    spent, t0, n_done = 0.0, time.time(), 0
    for c0 in range(0, len(pending), chunk):
        part = pending[c0:c0 + chunk]
        results = tmany([j for j, _ in part], workers=a.workers)
        if len(results) != len(part):
            raise RuntimeError(f"transcribe_many returned {len(results)} results for {len(part)} jobs")
        by_key = {r["meta"]["key"]: r for r in results if isinstance(r.get("meta"), dict) and "key" in r["meta"]}
        with open(runs_path, "a") as f:
            for idx, (job, info) in enumerate(part):
                res = by_key.get(job["meta"]["key"], results[idx])
                out = res.get("text") or ""
                err = res.get("error")
                sch = info["scheme"]
                row = {**info["base"], "skipped": False}
                if err:
                    row.update(cer=None, symbol_cer=None)
                else:
                    row.update(cer=metrics.cer(info["text"], sch.decode(out)),
                               symbol_cer=metrics.cer(info["target"], out))
                row.update(output_text=out, cost_usd=res.get("cost_usd") or 0.0,
                           input_tokens=res.get("input_tokens"), output_tokens=res.get("output_tokens"),
                           latency_s=res.get("latency_s"), cached=bool(res.get("cached")), error=err)
                if res.get("model") and not a.dry_run:
                    row["served_model"] = res["model"]
                spent += row["cost_usd"]
                f.write(json.dumps(row) + "\n")
        n_done += len(part)
        print(f"  {n_done}/{len(pending)} calls, ${spent:.2f} spent, {time.time() - t0:.0f}s", flush=True)
        if not a.dry_run and a.max_cost is not None and spent > a.max_cost:
            print(f"STOP: spent ${spent:.2f} > --max-cost ${a.max_cost:.2f}", file=sys.stderr)
            break

    if not a.no_analysis and runs_path.exists():
        from vbpe import analysis
        rows = analysis.load_runs(runs_path)
        s = analysis.write_summary(rows, run_dir, title=f"Phase 1 {a.run_name}")
        for v in s["headline"]["verdicts"].values():
            print(v)
        print(f"summary: {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
