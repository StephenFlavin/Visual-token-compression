"""Analysis of CER-vs-pixel-budget sweeps (Phase 1 Claude runs and Phase 2 CRNN results).

Input rows are dicts (one JSON object per line in ``runs.jsonl``). Keys used:

  scheme       rendering scheme name                         (required)
  cer          character error rate vs source text           (required; rows with cer=None are ignored)
  px_per_char  actual canvas pixels per normalised source char (required)
  budget       nominal px^2/char of the grid point           (Phase 1; absent in Phase 2)
  em           font size in px                               (Phase 2 groups by em when budget is absent)
  passage_id   bootstrap unit (Phase 1). Phase 2 may use ``group``; if neither is present every row
               is its own unit and bootstrap resampling is stratified per grid point.
  variant      "natural" | "scrambled" (optional; analysed as separate conditions)
  error        non-empty -> row ignored (failed call)

Main entry points: ``load_runs``, ``curve``, ``p_at`` / ``p_at_ci``, ``headline``, ``write_summary``.
CLI: ``python3 -m vtc.analysis results/phase1/<run>/runs.jsonl [--out DIR]``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from vtc import config

LATIN_SCHEMES = ("latin", "latin_condensed")
THRESHOLDS = (0.03, 0.10)
DEFAULT_TEXT_TOKENS_PER_CHAR = 0.25   # metrics.text_tokens_est: ~4 chars / token

# Fixed scheme -> colour mapping (colour follows the entity, never its rank). Categorical palette
# validated for CVD separation on adjacent pairs; lines also get distinct markers/dashes so identity
# never depends on colour alone.
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_SCHEME_ORDER = ["latin", "latin_condensed", "subst300", "subst1000", "subst300_legend",
                 "quad2", "quad3", "quad4"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_DASHES = ["-", "-", "--", "--", ":", "-.", "-.", "-."]
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------------------- loading
def _row_key(r: dict) -> tuple:
    return (r.get("run"), r.get("model"), r.get("effort"), r.get("variant"), r.get("passage_id", r.get("group")),
            r.get("scheme"), r.get("budget"), r.get("em") if r.get("budget") is None else None)


def load_runs(paths: str | Path | Iterable[str | Path], dedupe: bool = True) -> list[dict]:
    """Read one or more ``runs.jsonl`` files (or directories containing one) into a list of dicts.

    With ``dedupe`` the last row per (run, model, variant, passage, scheme, budget) wins, except that
    a successful row is never replaced by a later failed one.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows: list[dict] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            p = p / "runs.jsonl"
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not dedupe:
        return rows
    best: dict[tuple, dict] = {}
    for r in rows:
        k = _row_key(r)
        old = best.get(k)
        if old is not None and _usable(old) and not _usable(r):
            continue
        best[k] = r
    return list(best.values())


def _usable(r: dict) -> bool:
    return r.get("cer") is not None and not r.get("error") and r.get("px_per_char") is not None


def usable_rows(rows: Iterable[dict]) -> list[dict]:
    return [r for r in rows if _usable(r)]


def detect_phase(rows: list[dict]) -> int:
    phases = {r.get("phase") for r in rows if r.get("phase") is not None}
    if phases:
        return max(phases)
    return 1 if any(r.get("budget") is not None for r in rows) else 2


def conditions(rows: list[dict]) -> dict[str, list[dict]]:
    """Split rows into analysis conditions: one per variant (and per model / effort if several)."""
    out: dict[str, list[dict]] = defaultdict(list)
    sfx = _setting_namer(rows)
    for r in rows:
        out[(r.get("variant") or "all") + sfx(r)].append(r)
    return dict(sorted(out.items()))


def _setting_namer(rows: list[dict]):
    """r -> "|<model>|effort=<e>" suffix, naming only the fields that vary across ``rows``."""
    many_models = len({x.get("model") for x in rows}) > 1
    many_efforts = len({x.get("effort") for x in rows}) > 1

    def name(r: dict) -> str:
        return ((f"|{r.get('model')}" if many_models else "")
                + (f"|effort={r.get('effort')}" if many_efforts else ""))
    return name


def prior_credit(rows: list[dict]) -> dict:
    """Per (setting, scheme, budget): mean CER natural, scrambled and natural - scrambled.

    Negative = the language prior helps (natural text is read better). Passages differ between
    variants, so this is an unpaired difference of means. Empty if only one variant is present.
    """
    rows = usable_rows(rows)
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    sfx = _setting_namer(rows)
    for r in rows:
        if r.get("variant") in ("natural", "scrambled"):
            acc[sfx(r).lstrip("|") or "all"][r["scheme"]][_xkey(r)][r["variant"]].append(r["cer"])
    out: dict = {}
    for setting, by_s in acc.items():
        for s in schemes_in([{"scheme": k} for k in by_s]):
            for b in sorted(by_s[s]):
                d = by_s[s][b]
                if d.get("natural") and d.get("scrambled"):
                    nat, scr = float(np.mean(d["natural"])), float(np.mean(d["scrambled"]))
                    out.setdefault(setting, {}).setdefault(s, {})[f"{b:g}"] = {
                        "natural": nat, "scrambled": scr, "diff": nat - scr,
                        "n_natural": len(d["natural"]), "n_scrambled": len(d["scrambled"])}
    return out


def refusal_rates(rows: list[dict]) -> dict:
    """Fraction of answered calls flagged ``refusal`` per condition / scheme (rows lacking the key ignored)."""
    out: dict = {}
    for c, rs in conditions([r for r in rows if r.get("refusal") is not None]).items():
        for s in schemes_in(rs):
            v = [bool(r["refusal"]) for r in rs if r["scheme"] == s]
            out.setdefault(c, {})[s] = {"rate": float(np.mean(v)), "n": len(v)}
    return out


def schemes_in(rows: list[dict]) -> list[str]:
    names = {r["scheme"] for r in rows}
    known = [s for s in _SCHEME_ORDER if s in names]
    return known + sorted(names - set(known))


# ------------------------------------------------------------------------------------ curve tables
def _xkey(r: dict):
    return r["budget"] if r.get("budget") is not None else r.get("em", r["px_per_char"])


def _unit(r: dict):
    u = r.get("passage_id")
    return r.get("group") if u is None else u


class CurveTable:
    """CER matrix (units x grid points) for one scheme. ``units=None`` -> unpaired (Phase 2 style)."""

    def __init__(self, rows: list[dict], scheme: str, units: list | None = None):
        rs = [r for r in usable_rows(rows) if r["scheme"] == scheme]
        self.scheme = scheme
        xkeys = sorted({_xkey(r) for r in rs})
        # x = mean actual px^2/char at that grid point; sort by it (em grids are not ordered by px)
        xs = {k: float(np.mean([r["px_per_char"] for r in rs if _xkey(r) == k])) for k in xkeys}
        self.xkeys = sorted(xkeys, key=lambda k: xs[k])
        self.x = np.array([xs[k] for k in self.xkeys], dtype=float)
        self.budget = [k if rs and any(r.get("budget") is not None for r in rs) else None
                       for k in self.xkeys]
        col = {k: j for j, k in enumerate(self.xkeys)}
        has_units = rs and all(_unit(r) is not None for r in rs)
        if has_units:
            self.units = units if units is not None else sorted({_unit(r) for r in rs}, key=str)
            ui = {u: i for i, u in enumerate(self.units)}
            s = np.zeros((len(self.units), len(self.xkeys)))
            n = np.zeros_like(s)
            for r in rs:
                if _unit(r) in ui:
                    s[ui[_unit(r)], col[_xkey(r)]] += r["cer"]
                    n[ui[_unit(r)], col[_xkey(r)]] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                self.M = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            self.cols = None
        else:
            self.units = None
            self.M = None
            self.cols = [np.array([r["cer"] for r in rs if _xkey(r) == k], dtype=float)
                         for k in self.xkeys]

    @property
    def empty(self) -> bool:
        return len(self.xkeys) == 0

    def mean(self) -> np.ndarray:
        if self.M is not None:
            with np.errstate(invalid="ignore"), _quiet():
                return np.nanmean(self.M, axis=0)
        return np.array([c.mean() for c in self.cols])

    def counts(self) -> np.ndarray:
        if self.M is not None:
            return np.sum(~np.isnan(self.M), axis=0).astype(float)
        return np.array([len(c) for c in self.cols], dtype=float)

    def bootstrap(self, n_boot: int = 1000, seed: int = 0, idx: np.ndarray | None = None) -> np.ndarray:
        """(n_boot, n_points) bootstrap replicates of the mean curve.

        Paired tables resample units (passages) with replacement; ``idx`` (n_boot, n_units) lets
        several schemes share the same draws. Unpaired tables resample within each grid point.
        """
        rng = np.random.default_rng(seed)
        if self.M is not None:
            if idx is None:
                idx = rng.integers(0, len(self.units), size=(n_boot, len(self.units)))
            out = np.empty((idx.shape[0], len(self.xkeys)))
            with _quiet():
                for b in range(idx.shape[0]):
                    out[b] = np.nanmean(self.M[idx[b]], axis=0)
            return out
        out = np.empty((n_boot, len(self.xkeys)))
        for j, c in enumerate(self.cols):
            out[:, j] = c[rng.integers(0, len(c), size=(n_boot, len(c)))].mean(axis=1)
        return out


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


# --------------------------------------------------------------------------- smoothing & crossing
def isotonic_decreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted least-squares non-increasing fit (pool-adjacent-violators). NaNs are left as NaN."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    ok = ~np.isnan(y)
    if not ok.any():
        return out
    yy = y[ok]
    ww = np.ones_like(yy) if w is None else np.maximum(np.asarray(w, dtype=float)[ok], 1e-12)
    # blocks: [value, weight, length]
    vals, wts, lens = [], [], []
    for v, wt in zip(yy, ww):
        vals.append(v); wts.append(wt); lens.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:   # violation of non-increasing
            v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
            nw = wts[-1] + w2
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / nw
            wts[-1] = nw
            lens[-1] += l2
    out[ok] = np.repeat(vals, lens)
    return out


def crossing(x: np.ndarray, y: np.ndarray, threshold: float, w: np.ndarray | None = None,
             smooth: bool = True) -> tuple[float | None, str | None]:
    """x at which the (isotonically smoothed) curve first drops to <= threshold.

    Linear interpolation in log(x) between the bracketing grid points. Returns (x, censored):
    censored="left" if already <= threshold at the smallest x (true value is <= returned x);
    (None, "right") if the curve never reaches the threshold within the grid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y) & (x > 0)
    if w is not None:
        w = np.asarray(w, dtype=float)[ok]
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return None, "right"
    order = np.argsort(x)
    x, y = x[order], y[order]
    if w is not None:
        w = w[order]
    s = isotonic_decreasing(y, w) if smooth else y
    below = np.nonzero(s <= threshold)[0]
    if len(below) == 0:
        return None, "right"
    i = int(below[0])
    if i == 0:
        return float(x[0]), "left"
    y0, y1 = s[i - 1], s[i]
    t = (y0 - threshold) / (y0 - y1) if y0 != y1 else 1.0
    lx = math.log(x[i - 1]) + t * (math.log(x[i]) - math.log(x[i - 1]))
    return float(math.exp(lx)), None


# ------------------------------------------------------------------------------------- public API
def curve(rows: list[dict], scheme: str, n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """Mean CER per grid point with a bootstrap CI over passages (or rows, when unpaired)."""
    t = CurveTable(rows, scheme)
    if t.empty:
        return {"scheme": scheme, "budget": [], "px_per_char": [], "mean": [], "lo": [], "hi": [],
                "smoothed": [], "n": []}
    mean = t.mean()
    boot = t.bootstrap(n_boot, seed)
    a = (1 - ci) / 2
    with _quiet():
        lo = np.nanquantile(boot, a, axis=0)
        hi = np.nanquantile(boot, 1 - a, axis=0)
    return {"scheme": scheme, "budget": t.budget, "px_per_char": t.x.tolist(), "mean": mean.tolist(),
            "lo": lo.tolist(), "hi": hi.tolist(),
            "smoothed": isotonic_decreasing(mean, t.counts()).tolist(), "n": t.counts().tolist()}


def p_at(rows: list[dict], scheme: str, threshold: float) -> float | None:
    """px^2/char at which the monotone-smoothed mean CER of ``scheme`` crosses ``threshold``."""
    t = CurveTable(rows, scheme)
    if t.empty:
        return None
    return crossing(t.x, t.mean(), threshold, t.counts())[0]


def _boot_p(t: CurveTable, threshold: float, boot: np.ndarray) -> np.ndarray:
    """Crossing per bootstrap replicate; never-crossing -> +inf."""
    out = np.empty(boot.shape[0])
    for b in range(boot.shape[0]):
        p, _ = crossing(t.x, boot[b], threshold)
        out[b] = np.inf if p is None else p
    return out


def _q(v: np.ndarray, q: float) -> float | None:
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None
    x = float(np.quantile(v, q, method="nearest"))
    return None if math.isinf(x) else x


def p_at_ci(rows: list[dict], scheme: str, threshold: float, n_boot: int = 1000, ci: float = 0.95,
            seed: int = 0) -> dict:
    """P@threshold with a bootstrap CI (resampling passages). ``hi=None`` -> CI open to the right."""
    t = CurveTable(rows, scheme)
    if t.empty:
        return {"p": None, "censored": "right", "lo": None, "hi": None, "frac_crossed": 0.0}
    p, cens = crossing(t.x, t.mean(), threshold, t.counts())
    bp = _boot_p(t, threshold, t.bootstrap(n_boot, seed))
    a = (1 - ci) / 2
    return {"p": p, "censored": cens, "lo": _q(bp, a), "hi": _q(bp, 1 - a),
            "frac_crossed": float(np.mean(np.isfinite(bp)))}


def _text_tokens_per_char(rows: list[dict]) -> float:
    v = [r["text_tokens_est"] / r["n_chars"] for r in rows
         if r.get("text_tokens_est") and r.get("n_chars")]
    return float(np.mean(v)) if v else DEFAULT_TEXT_TOKENS_PER_CHAR


def compression_at(p: float | None, text_tokens_per_char: float = DEFAULT_TEXT_TOKENS_PER_CHAR):
    """text tokens / vision tokens for the same text at px^2/char = p.

    Vision tokens use the large-image approximation px / config.CLAUDE_PX_PER_TOKEN (28x28 patches =
    784 px/token; metrics.vision_tokens adds patch rounding and ~3 tokens of overhead).
    """
    if p is None or p <= 0:
        return None
    return text_tokens_per_char / (p / config.CLAUDE_PX_PER_TOKEN)


def _ratio(a, b):
    if a is None or b is None:
        return None
    return a / b


def _analyse_condition(rows: list[dict], thresholds, n_boot: int, seed: int) -> dict:
    schemes = schemes_in(rows)
    tpc = _text_tokens_per_char(rows)
    units_all = sorted({_unit(r) for r in usable_rows(rows) if _unit(r) is not None}, key=str)
    paired = bool(units_all) and all(_unit(r) is not None for r in usable_rows(rows))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units_all), size=(n_boot, len(units_all))) if paired else None
    tables = {s: CurveTable(rows, s, units=units_all if paired else None) for s in schemes}
    boots = {s: (t.bootstrap(n_boot, seed, idx=idx) if not t.empty else None) for s, t in tables.items()}
    out = {"n_units": len(units_all) if paired else None, "paired_bootstrap": paired,
           "text_tokens_per_char": tpc, "thresholds": {}}
    for thr in thresholds:
        res = {}
        bps = {}
        for s, t in tables.items():
            if t.empty:
                continue
            p, cens = crossing(t.x, t.mean(), thr, t.counts())
            bp = _boot_p(t, thr, boots[s])
            bps[s] = bp
            res[s] = {"p": p, "censored": cens, "lo": _q(bp, 0.025), "hi": _q(bp, 0.975),
                      "frac_crossed": float(np.mean(np.isfinite(bp))),
                      "vision_tokens_per_1k_chars": None if p is None else p * 1000 / config.CLAUDE_PX_PER_TOKEN,
                      "compression": compression_at(p, tpc),
                      "grid": [float(t.x.min()), float(t.x.max())]}
        lat = [s for s in LATIN_SCHEMES if s in res]
        best = None
        if lat:
            finite = [s for s in lat if res[s]["p"] is not None]
            best = min(finite, key=lambda s: res[s]["p"]) if finite else lat[0]
        best_p = res[best]["p"] if best else None
        best_boot = np.min(np.vstack([bps[s] for s in lat]), axis=0) if lat else None
        for s, d in res.items():
            if best is None:
                d.update(ratio=None, ratio_lo=None, ratio_hi=None, saving=None)
                continue
            r = _ratio(d["p"], best_p)
            if d["p"] is not None and best_p is None:
                r = 0.0          # scheme reaches the threshold, best Latin never does
            with np.errstate(invalid="ignore", divide="ignore"):
                rb = bps[s] / best_boot
            rb[np.isinf(bps[s]) & np.isinf(best_boot)] = np.nan
            # censoring: P = smallest grid x means "<= x" (left); None means "> grid max" (right)
            cs = d["censored"]
            cb = res[best]["censored"] if best_p is not None else "right"
            bound = None
            if cs == "left" and cb == "left" and s != best:
                bound = "undetermined"
            elif cs == "left" and s != best:
                bound = "upper"            # true ratio <= reported
            elif cb == "left" and s != best:
                bound = "lower"            # true ratio >= reported
            d.update(ratio=r, ratio_lo=_q(rb, 0.025), ratio_hi=_q(rb, 0.975), ratio_bound=bound,
                     saving=None if r is None else 1 - r)
        out["thresholds"][f"{thr:g}"] = {"best_latin": best, "best_latin_p": best_p, "schemes": res}
    return out


def _fmt(v, nd=1):
    return "n/r" if v is None else f"{v:.{nd}f}"


def _verdicts(cond: dict[str, dict], phase: int) -> dict[str, str]:
    """K2 / K3 / quadrant / prediction strings per PLAN section 4."""
    v = {}
    if phase == 1:
        key, thr, bar, name = "K2", 0.10, 0.20, "K2 (Phase 1: subst P@10% >= 20% below best Latin, natural or scrambled)"
    else:
        key, thr, bar, name = "K3", 0.03, 0.30, "K3 (Phase 2: subst P@3% >= 30% below best Latin)"
    tk = f"{thr:g}"
    cands = []   # (ratio, cond, scheme, d, best)
    any_subst = False
    for c, a in cond.items():
        t = a["thresholds"].get(tk)
        if not t:
            continue
        for s, d in t["schemes"].items():
            if s.startswith("subst"):
                any_subst = True
                if d["ratio"] is not None:
                    cands.append((d["ratio"], c, s, d, t["best_latin"], t["best_latin_p"]))
    if not any_subst:
        v[key] = f"{name}: UNDETERMINED - no substitution scheme in data."
    elif not cands:
        v[key] = (f"{name}: UNDETERMINED - neither subst nor best Latin reaches CER {thr:.0%} "
                  "within the grid; extend budgets.")
    else:
        r, c, s, d, best, bp = min(cands, key=lambda z: z[0])
        detail = (f"best case {s} on {c}: P@{thr:.0%}={_fmt(d['p'])} vs {best} {_fmt(bp)} px^2/char, "
                  f"ratio {r:.2f} (95% CI {_fmt(d['ratio_lo'], 2)}-{_fmt(d['ratio_hi'], 2)})")
        if d.get("ratio_bound"):
            what = "undetermined" if d["ratio_bound"] == "undetermined" else f"only a {d['ratio_bound']} bound"
            detail += (f"; WARNING ratio is {what} (threshold reached at the smallest budget "
                       "- extend the grid downwards)")
        passed = r <= 1 - bar
        hi = d["ratio_hi"]
        lo = d["ratio_lo"]
        unsure = (passed and (hi is None or hi > 1 - bar)) or (not passed and lo is not None and lo <= 1 - bar)
        if phase == 1:
            word = "NOT TRIGGERED (zero-shot signal survives)" if passed else "TRIGGERED - zero-shot negative"
        else:
            word = "PASS" if passed else "KILL"
        v[key] = f"{name}: {word}; {detail}" + ("; CI straddles the bar - inconclusive" if unsure else "")
    # quadrant
    qthr = f"{0.03:g}" if phase == 2 else f"{0.10:g}"
    qs = []
    for c, a in cond.items():
        t = a["thresholds"].get(qthr, {})
        for s, d in t.get("schemes", {}).items():
            if s.startswith("quad"):
                qs.append(f"{s}/{c} ratio {_fmt(d['ratio'], 2)}")
    if qs:
        if phase == 2:
            killed = all((d["ratio"] is None or d["ratio"] >= 1)
                         for a in cond.values() for s, d in a["thresholds"].get(qthr, {}).get("schemes", {}).items()
                         if s.startswith("quad"))
            v["quadrant"] = (f"Quadrant (Phase 2 rule: killed if P@3% >= best Latin): "
                             f"{'KILLED' if killed else 'survives'}; " + ", ".join(qs))
        else:
            ok = [a["thresholds"].get(qthr, {}).get("schemes", {}) for a in cond.values()]
            p1 = all((d["ratio"] is None or d["ratio"] >= 1.3) for sch in ok for s, d in sch.items()
                     if s.startswith("quad"))
            v["P1"] = (f"P1 (quadrant needs >= 30% more px^2/char than best Latin, at P@{float(qthr):.0%}): "
                       f"{'CONFIRMED' if p1 else 'NOT confirmed'}; " + ", ".join(qs))
    # P2: subst300 within +-20% of latin_condensed
    p2 = []
    for c, a in cond.items():
        t = a["thresholds"].get(tk, {}).get("schemes", {})
        if "subst300" in t and "latin_condensed" in t:
            r = _ratio(t["subst300"]["p"], t["latin_condensed"]["p"])
            if r is not None:
                p2.append(f"{c}: {r:.2f} ({'within' if 0.8 <= r <= 1.2 else 'outside'} +-20%)")
    if p2:
        v["P2"] = f"P2 (subst300 / latin_condensed at P@{thr:.0%}): " + "; ".join(p2)
    # P3: Latin advantage shrinks on scrambled text
    if "natural" in cond and "scrambled" in cond:
        def best_subst(a):
            t = a["thresholds"].get(tk, {}).get("schemes", {})
            rs = [d["ratio"] for s, d in t.items() if s.startswith("subst") and d["ratio"] is not None]
            return min(rs) if rs else None
        rn, rs_ = best_subst(cond["natural"]), best_subst(cond["scrambled"])
        if rn is not None and rs_ is not None:
            v["P3"] = (f"P3 (Latin advantage shrinks on scrambled): best subst ratio natural {rn:.2f} -> "
                       f"scrambled {rs_:.2f} -> {'CONFIRMED' if rs_ < rn else 'NOT confirmed'}")
    return v


def headline(rows: list[dict], thresholds=THRESHOLDS, n_boot: int = 1000, seed: int = 0,
             phase: int | None = None) -> dict:
    """P@threshold per scheme, ratio vs best Latin, implied compression and kill-criterion verdicts."""
    rows = usable_rows(rows)
    phase = detect_phase(rows) if phase is None else phase
    cond = {c: _analyse_condition(rs, thresholds, n_boot, seed) for c, rs in conditions(rows).items()}
    return {"phase": phase, "n_rows": len(rows), "conditions": cond, "verdicts": _verdicts(cond, phase)}


# ------------------------------------------------------------------------------------------- plots
def scheme_style(name: str) -> dict:
    i = _SCHEME_ORDER.index(name) if name in _SCHEME_ORDER else len(_SCHEME_ORDER) + (hash(name) % 8)
    j = i % len(_PALETTE)
    return {"color": _PALETTE[j], "marker": _MARKERS[j], "linestyle": _DASHES[j]}


def _style_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_PRIMARY, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(rows: list[dict], path: str | Path, n_boot: int = 1000, title: str | None = None,
                floor: float = 1e-3) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    rows = usable_rows(rows)
    conds = conditions(rows)
    n = max(1, len(conds))
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.8), squeeze=False, sharey=True)
    xlabel = "pixels² per source character (log)"
    for ax, (c, rs) in zip(axes[0], conds.items()):
        _style_axes(ax)
        xs_all = []
        for s in schemes_in(rs):
            cv = curve(rs, s, n_boot=n_boot)
            if not cv["px_per_char"]:
                continue
            st = scheme_style(s)
            x = np.array(cv["px_per_char"])
            xs_all.extend(x)
            m = np.clip(np.array(cv["mean"]), floor, None)
            lo = np.clip(np.array(cv["lo"]), floor, None)
            hi = np.clip(np.array(cv["hi"]), floor, None)
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, linewidth=0)
            ax.plot(x, m, color=st["color"], linestyle=st["linestyle"], linewidth=2, marker=st["marker"],
                    markersize=6, markeredgecolor="white", markeredgewidth=1, label=s, zorder=3)
        for thr in THRESHOLDS:
            ax.axhline(thr, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(f"CER {thr:.0%}", xy=(1.0, thr), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points", ha="right", va="bottom",
                        fontsize=8, color=TEXT_SECONDARY)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(floor * 0.8, 1.3)
        if xs_all:
            ax.set_xlim(min(xs_all) / 1.15, max(xs_all) * 1.15)
        yt = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}" if v < 0.01 else f"{v:.0%}"))
        nominal = sorted({r["budget"] for r in rs if r.get("budget") is not None})
        cand = nominal if 2 <= len(nominal) <= 14 else (1, 2, 3, 4, 6, 9, 13, 18, 27, 40, 60, 100, 150, 250)
        xt = [v for v in cand if (not xs_all) or (min(xs_all) / 1.2 <= v <= max(xs_all) * 1.2)]
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(xlabel, color=TEXT_PRIMARY, fontsize=10)
        sec = ax.secondary_xaxis("top", functions=(lambda v: v * 1000 / config.CLAUDE_PX_PER_TOKEN,
                                                   lambda t: t * config.CLAUDE_PX_PER_TOKEN / 1000))
        sec.set_xlabel(f"≈ vision tokens per 1k chars (px/{config.CLAUDE_PX_PER_TOKEN})", color=TEXT_SECONDARY, fontsize=9)
        tok = [v * 1000 / config.CLAUDE_PX_PER_TOKEN for v in ax.get_xlim()]
        sec.xaxis.set_major_locator(FixedLocator([t for t in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500)
                                                  if tok[0] <= t <= tok[1]]))
        sec.xaxis.set_minor_locator(NullLocator())
        sec.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        sec.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        sec.spines["top"].set_color(TEXT_SECONDARY)
        n_units = len({_unit(r) for r in rs})
        ax.set_title(f"{c}  (n={n_units} passage{'s' if n_units != 1 else ''})", fontsize=10,
                     color=TEXT_PRIMARY, loc="left")
    axes[0][0].set_ylabel("CER (log; mean, 95% bootstrap CI)", color=TEXT_PRIMARY, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    ncol = max(1, min(len(labels), 4 * n))
    nrow = max(1, math.ceil(len(labels) / ncol))
    if nrow > 1:
        fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] + 0.3 * (nrow - 1))
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
                   fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title or "CER vs pixel budget", fontsize=12, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02 + 0.05 * nrow, 1, 0.97))
    path = Path(path)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def plot_p_bars(head: dict, path: str | Path, title: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    conds = head["conditions"]
    n = max(1, len(conds))
    thr_keys = [f"{t:g}" for t in THRESHOLDS]
    n_s = max([len({s for tk in thr_keys for s in a["thresholds"].get(tk, {}).get("schemes", {})})
               for a in conds.values()] or [1])
    fig, axes = plt.subplots(1, n, figsize=(max(6.4, 1.15 * n_s + 0.8) * n, 4.6), squeeze=False)
    width = 0.38
    for ax, (c, a) in zip(axes[0], conds.items()):
        _style_axes(ax)
        ax.grid(False, axis="x")
        schemes = schemes_in([{"scheme": s} for tk in thr_keys
                              for s in a["thresholds"].get(tk, {}).get("schemes", {})])
        ymax = 1.0
        ymin = np.inf
        for k, tk in enumerate(thr_keys):
            t = a["thresholds"].get(tk, {})
            for i, s in enumerate(schemes):
                d = t.get("schemes", {}).get(s)
                if d is None or d["p"] is None:
                    continue
                ymax = max(ymax, d["hi"] or d["p"], d["grid"][1])
                ymin = min(ymin, d["lo"] or d["p"])
        ymin = 1.0 if not np.isfinite(ymin) else ymin
        for k, tk in enumerate(thr_keys):
            t = a["thresholds"].get(tk, {})
            off = (k - 0.5) * width
            for i, s in enumerate(schemes):
                d = t.get("schemes", {}).get(s)
                st = scheme_style(s)
                if d is None:
                    continue
                if d["p"] is None:
                    ax.text(i + off, ymin * 1.05, f"not reached\n(> {d['grid'][1]:.0f})", rotation=90,
                            ha="center", va="bottom", fontsize=7, color=TEXT_SECONDARY)
                    continue
                ax.bar(i + off, d["p"], width=width * 0.92, color=st["color"],
                       alpha=1.0 if k == 0 else 0.45, hatch=None if k == 0 else "///",
                       edgecolor="white", linewidth=0.8, zorder=2)
                lo = d["lo"] if d["lo"] is not None else d["p"]
                hi = d["hi"] if d["hi"] is not None else ymax * 1.8
                ax.errorbar(i + off, d["p"], yerr=[[d["p"] - lo], [hi - d["p"]]], fmt="none",
                            ecolor=TEXT_PRIMARY, elinewidth=1, capsize=3 if d["hi"] is not None else 0,
                            zorder=3)
                if d["hi"] is None:   # CI open to the right: some replicates never reach the threshold
                    ax.annotate("", xy=(i + off, hi * 1.08), xytext=(i + off, hi * 0.92),
                                arrowprops={"arrowstyle": "-|>", "color": TEXT_PRIMARY, "lw": 1})
                if d["censored"] == "left":
                    ax.text(i + off, d["p"], "≤", ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY)
            bp = t.get("best_latin_p")
            if bp:
                ax.axhline(bp, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (1, 2)) if k == 1 else (0, (4, 3)),
                           zorder=1)
        labels = []
        for s in schemes:
            d10 = a["thresholds"].get(thr_keys[1], {}).get("schemes", {}).get(s, {})
            r = d10.get("ratio")
            pre = {"upper": "≤", "lower": "≥", "undetermined": "?"}.get(d10.get("ratio_bound"), "")
            labels.append(s if s in LATIN_SCHEMES or r is None else f"{s}\n{pre}×{r:.2f} @10%")
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.set_yscale("log")
        lo_, hi_ = ymin / 1.6, ymax * 2.0
        ax.set_ylim(lo_, hi_)
        ax.yaxis.set_major_locator(FixedLocator([v for v in (0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500)
                                                 if lo_ <= v <= hi_]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylabel("px² per char needed (log)", color=TEXT_PRIMARY, fontsize=10)
        ax.set_title(c, fontsize=10, color=TEXT_PRIMARY, loc="left")
    legend = [Patch(facecolor="#8a8984", label="P@3% (solid)"),
              Patch(facecolor="#8a8984", alpha=0.45, hatch="///", edgecolor="white", label="P@10% (hatched)")]
    fig.legend(handles=legend + [plt.Line2D([], [], color=TEXT_SECONDARY, linestyle=(0, (4, 3)),
                                            label="best Latin P@3%"),
                                 plt.Line2D([], [], color=TEXT_SECONDARY, linestyle=(0, (1, 2)),
                                            label="best Latin P@10%")],
               loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title or "Pixel budget needed to reach CER 3% / 10% (95% bootstrap CI)", fontsize=12,
                 color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    path = Path(path)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------------------- summary
def _md_table(a: dict) -> list[str]:
    tks = list(a["thresholds"])
    head = ["scheme"]
    for tk in tks:
        pct = f"{float(tk):.0%}"
        head += [f"P@{pct} [95% CI]", f"ratio@{pct} [CI]", f"vis tok/1k@{pct}", f"compr@{pct}"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    schemes = schemes_in([{"scheme": s} for tk in tks for s in a["thresholds"][tk]["schemes"]])
    for s in schemes:
        cells = [s]
        for tk in tks:
            d = a["thresholds"][tk]["schemes"].get(s)
            if d is None:
                cells += ["", "", "", ""]
                continue
            if d["p"] is None:
                p = f"> {d['grid'][1]:.0f} (not reached)"
            else:
                p = ("≤ " if d["censored"] == "left" else "") + f"{d['p']:.1f} [{_fmt(d['lo'])}, {_fmt(d['hi'])}]"
            pre = {"upper": "≤ ", "lower": "≥ ", "undetermined": "? "}.get(d.get("ratio_bound"), "")
            r = "—" if d["ratio"] is None else (f"{pre}{d['ratio']:.2f} [{_fmt(d['ratio_lo'], 2)}, "
                                                f"{_fmt(d['ratio_hi'], 2)}]")
            cells += [p, r, _fmt(d["vision_tokens_per_1k_chars"], 1), _fmt(d["compression"], 2)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_summary(rows: list[dict], out_dir: str | Path, n_boot: int = 1000, title: str | None = None,
                  plots: bool = True) -> dict:
    """Write summary.json, summary.md and (optionally) cer_curves.png / p_at_bars.png to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = rows
    rows = usable_rows(rows)
    head = headline(rows, n_boot=n_boot)
    curves = {c: {s: curve(rs, s, n_boot=n_boot) for s in schemes_in(rs)} for c, rs in conditions(rows).items()}
    n_skipped = sum(1 for r in all_rows if r.get("skipped"))
    n_err = sum(1 for r in all_rows if r.get("error"))
    cost = sum(r.get("cost_usd") or 0 for r in all_rows)
    pc = prior_credit(rows)
    rr = refusal_rates(rows)
    summary = {"headline": head, "curves": curves, "prior_credit": pc, "refusal_rates": rr,
               "counts": {"rows": len(all_rows), "usable": len(rows), "skipped_no_fit": n_skipped,
                          "errors": n_err}, "cost_usd_total": cost}
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1))

    md = [f"# {title or 'Sweep summary'} (phase {head['phase']})", "",
          f"Rows: {len(all_rows)} total, {len(rows)} usable, {n_skipped} no-fit (scored CER=1), "
          f"{n_err} errors. Total recorded cost: ${cost:.2f}.", "",
          "P@x = px²/char at which the isotonic-smoothed mean CER first reaches x (log-x interpolation). "
          "ratio = P(scheme)/P(best Latin) (best of latin, latin_condensed at that threshold); "
          "CI = paired passage bootstrap. compr = text tokens / vision tokens at P (Claude: "
          f"px/{config.CLAUDE_PX_PER_TOKEN} = 28x28 patches, "
          f"text ≈ chars/4). '≤' on P = already below threshold at the smallest budget; on a ratio "
          "'≤'/'≥'/'?' = upper/lower bound/undetermined because of that censoring.", "",
          "## Verdicts", ""]
    md += [f"* {v}" for v in head["verdicts"].values()]
    for c, a in head["conditions"].items():
        md += ["", f"## Condition: {c}", ""]
        for tk, t in a["thresholds"].items():
            md.append(f"* best Latin @ {float(tk):.0%}: **{t['best_latin']}** "
                      f"({_fmt(t['best_latin_p'])} px²/char)")
        md += [""] + _md_table(a)
        md += ["", "Mean CER by budget:", ""]
        cs = curves[c]
        budgets = sorted({(b if b is not None else round(x, 1)) for cv in cs.values()
                          for b, x in zip(cv["budget"], cv["px_per_char"])})
        md.append("| scheme | " + " | ".join(f"{b:g}" for b in budgets) + " |")
        md.append("|" + "---|" * (len(budgets) + 1))
        for s, cv in cs.items():
            m = {(b if b is not None else round(x, 1)): v for b, x, v in zip(cv["budget"], cv["px_per_char"], cv["mean"])}
            md.append(f"| {s} | " + " | ".join("" if b not in m else f"{m[b]:.3f}" for b in budgets) + " |")
    if rr:
        md += ["", "## Refusal / non-transcription rate (heuristic)", "",
               "| condition | " + " | ".join(schemes_in(rows)) + " |", "|" + "---|" * (len(schemes_in(rows)) + 1)]
        for c, d in rr.items():
            md.append(f"| {c} | " + " | ".join(f"{d[s]['rate']:.0%} (n={d[s]['n']})" if s in d else ""
                                               for s in schemes_in(rows)) + " |")
    if pc:
        md += ["", "## Prior credit: mean CER(natural) - mean CER(scrambled)", "",
               "Negative = the language prior helps. Unpaired (different passages per variant).", ""]
        for setting, by_s in pc.items():
            bs = sorted({float(b) for d in by_s.values() for b in d})
            md += [f"Setting: {setting}", "", "| scheme | " + " | ".join(f"{b:g}" for b in bs) + " |",
                   "|" + "---|" * (len(bs) + 1)]
            for sname, d in by_s.items():
                md.append(f"| {sname} | " + " | ".join(f"{d[f'{b:g}']['diff']:+.3f}" if f"{b:g}" in d else ""
                                                       for b in bs) + " |")
            md.append("")
    if plots and rows:
        plot_curves(rows, out / "cer_curves.png", n_boot=n_boot, title=title and f"{title}: CER vs pixel budget")
        plot_p_bars(head, out / "p_at_bars.png")
        md += ["", "![CER curves](cer_curves.png)", "", "![P@ bars](p_at_bars.png)"]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarise a CER-vs-pixel sweep (runs.jsonl).")
    ap.add_argument("paths", nargs="+", help="runs.jsonl file(s) or run directories")
    ap.add_argument("--out", help="output dir (default: directory of the first input)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--title")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    rows = load_runs(a.paths)
    first = Path(a.paths[0])
    out = Path(a.out) if a.out else (first if first.is_dir() else first.parent)
    s = write_summary(rows, out, n_boot=a.n_boot, title=a.title, plots=not a.no_plots)
    print((out / "summary.md").read_text())
    print(f"wrote {out}/summary.json, summary.md" + ("" if a.no_plots else ", cer_curves.png, p_at_bars.png"))
    return 0 if s["counts"]["usable"] else 1


if __name__ == "__main__":
    sys.exit(main())
