"""Capacity curve (docs/PLAN_3090.md, Run 0 / 0b): bits recovered per vision token vs chars per vision token.

For each encoding (a ``get_encoding_s`` name at S = 448), text set and (optional) page shift, every test page is
scored with Qwen2.5-VL-3B (frozen, or with a finetune_qwen.py ``--adapter``) by the teacher-forced loss of its
true transcript under three images:

  real    the page itself
  swap    another test page of the same point (page i+1), the same target
  blank   an all-white S x S page, the same target

Reference (the conservative blind, PLAN_3090 correction 5) = per-sample min(swap, blank); gap = reference - real
(nats, summed over the target's text tokens). A swap or blank loss is never trusted alone: either can be lower
(format or topic cues), and the min never inflates the gap.

Target window. A page's whole transcript is up to ~18.6k chars (latin_p5), too long for a CPU forward pass, so
each page scores ONE window of whole lines with at most ``--window-chars`` target chars (default 2000), and the
prompt asks for exactly those lines (``r3_tasks.transcribe_prompt``: "Transcribe lines a-b of the n lines on this
page.", or "Transcribe all the text on this page." when the window is the page). ``--window top`` (default)
takes the first lines, ``mid`` a window centred on the page. The row records the window's and the page's chars,
and the page's information is extrapolated per char:

  bits per vision token (page i) = gap_i / ln 2 x (page chars_i / window chars_i) / vision tokens (256 at 448 px)

The point estimate is the mean over pages, with a page-bootstrap 95% CI. Also reported per point:
  * gap per text token (nats) = sum gap / sum target tokens;
  * recovered fraction = sum gap / sum reference loss (share of the text's blind information the page supplies);
  * recovered chars per vision token = chars per vision token x recovered fraction;
  * optional greedy transcription CER of the window on the first ``--cer-pages`` pages (``--cer-max-new`` caps
    the generated tokens), so "the NLL gap tracks CER" can be checked.

Text sets (``--texts``):
  natural   the test pages exactly as ``geom.make_pages_s`` renders them (WikiText-103 test + val text);
  shuffled  the same passages with their words shuffled: for page i, the source text the natural page i consumed
            is split on whitespace and permuted with ``numpy.random.default_rng([seed, i])``; page i of the
            shuffled set is rendered from that shuffled passage (followed by the next passage's shuffled words,
            so the page is full). Word-level, not block-level, shuffling: it keeps the page's vocabulary, letters
            and word lengths (so the rendered density is the same) and removes syntax and most of the language
            prior beyond the word, which is the low-predictability control (ZeroSense) the plan asks for. Words
            keep their punctuation; capitalisation is unchanged.

Encoding names: any ``get_encoding_s`` name (latin_p<pitch>, the learned-glyph banks of ``pages.LG_BANKS``, ...),
optionally with an integer upscale suffix ``_x<f>`` (e.g. ``lg_n0_pb_h5_x2``): the page is rendered at S // f px
and every pixel repeated f x f (the same alphabet at f times the cell size; f*(S//f) < S leaves a white strip).

Shifts (Run 0b, grid alignment): ``--shifts 0 7 14 21`` scores every (dx, dy) pair of the list. The text is
rendered once at (S - margin) px, ``--shift-margin`` defaulting to the largest shift, and pasted at (dx, dy) on a
white S x S page, so no text is cut and all shifts carry the same text. Use ``--arms real blank`` there: the
blank arm does not depend on the shift and is scored once per page.

Resume: every scored sample is appended to ``samples.jsonl`` (keyed by encoding, text set, shift, page) and is
skipped on a restart; a point's row goes to ``runs.jsonl`` when all its pages are done. ``--report`` (also run at
the end) writes summary.md and capacity.png (and alignment tables when shifts exist) from the files.

    python3 encoder_probe/experiments/capacity_curve.py --encodings latin_p16 latin_p8 --texts natural shuffled \\
        --pages 32 --run-name run0_cpu
    python3 encoder_probe/experiments/capacity_curve.py --encodings latin_p8 --texts natural --pages 16 \\
        --shifts 0 7 14 21 --arms real blank --cer-pages 0 --run-name run0b_cpu
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from encoder_probe.ep import r3_tasks as T  # noqa: E402
from encoder_probe.ep.geom import get_encoding_s, text_stream_s  # noqa: E402
from encoder_probe.ep.pages import Page  # noqa: E402

S = 448
LN2 = math.log(2)


# ------------------------------------------------------------------------------------------------- pages

def parse_encoding(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(.+)_x(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, 1)


def shuffle_words(text: str, seed: int, i: int) -> str:
    words = text.split()
    perm = np.random.default_rng([seed, i]).permutation(len(words))
    return " ".join(words[j] for j in perm)


class PageSet:
    """Pages of one (encoding, text set): content rendered at ``inner`` px, upscaled x f, pasted on S x S."""

    def __init__(self, name: str, text_set: str, n: int, seed: int = 0, margin: int = 0):
        self.name, self.text_set = name, text_set
        base, self.f = parse_encoding(name)
        self.inner = (S - margin) // self.f
        self.enc = get_encoding_s(base, self.inner)
        cap = self.enc.page_chars_max
        text = text_stream_s("test", (n + 1) * cap + cap)
        pos, segs, pages = 0, [], []
        for _ in range(n):
            if len(text) - pos < cap:
                raise ValueError(f"{name}: only {len(pages)} of {n} test pages fit the text")
            pg, used = self.enc.render(text, pos)
            pages.append(pg)
            segs.append((pos, used))
            pos += used
        if text_set == "shuffled":
            parts, starts, cur = [], [], 0
            for i, (p, u) in enumerate(segs + [(pos, cap)]):
                parts.append(shuffle_words(text[p:p + u], seed, i))
                starts.append(cur)
                cur += len(parts[-1]) + 1
            stream = " ".join(parts)
            pages = [self.enc.render(stream, starts[i])[0] for i in range(n)]
        elif text_set != "natural":
            raise KeyError(text_set)
        assert len({tuple(p.refs) for p in pages}) == len(pages), f"{name}/{text_set}: repeated pages"
        self.pages = pages

    def image(self, i: int, shift: tuple[int, int] = (0, 0)) -> np.ndarray:
        g = self.pages[i].img
        if self.f > 1:
            g = np.repeat(np.repeat(g, self.f, 0), self.f, 1)
        out = np.full((S, S, 3), 255, np.uint8)
        dx, dy = shift
        h, w = g.shape[:2]
        assert dy + h <= S and dx + w <= S, "shift larger than the margin"
        out[dy:dy + h, dx:dx + w] = g
        return out


def blank_image() -> np.ndarray:
    return np.full((S, S, 3), 255, np.uint8)


def window(page: Page, max_chars: int, where: str = "top") -> tuple[int, int]:
    refs, n = page.refs, len(page.refs)
    if len(T.transcript(page)) <= max_chars:
        return 0, n
    a, b = T.window_from(refs, 0, max_chars)
    if where == "mid":
        a = max(0, (n - (b - a)) // 2)
        a, b = T.window_from(refs, a, max_chars)
    elif where != "top":
        raise KeyError(where)
    return a, b


# ---------------------------------------------------------------------------------------------- statistics

def boot_mean(x: np.ndarray, reps: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), (reps, len(x)))
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def boot_ratio(num: np.ndarray, den: np.ndarray, reps: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (reps, len(num)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def sample_stats(s: dict) -> dict:
    """Per-sample derived quantities (losses are summed nats over the target's n text tokens)."""
    ref = min(x for x in (s.get("loss_swap"), s.get("loss_blank")) if x is not None)
    gap = ref - s["loss_real"]
    return {"ref": ref, "gap": gap, "n": s["n_tokens"],
            "bits_vtok": gap / LN2 * s["page_chars"] / s["window_chars"] / s["vision_tokens"]}


def cer_trunc(target: str, hyp: str) -> float:
    """CER of the hypothesis cut to the target's length (normalised whitespace): a model that reads the window
    and then keeps transcribing past its last line (zero-shot Qwen ignores "lines a-b" on most pages) is not
    charged for the extra lines; stopping early or summarising still counts as errors."""
    from rapidfuzz.distance import Levenshtein
    from vbpe.metrics import normalize
    r, h = normalize(target), normalize(hyp)
    return Levenshtein.distance(r, h[:len(r)]) / max(1, len(r))


def point_row(samples: list[dict], meta: dict) -> dict:
    st = [sample_stats(s) for s in samples]
    gap = np.array([x["gap"] for x in st])
    ref = np.array([x["ref"] for x in st])
    n = np.array([x["n"] for x in st], float)
    b = np.array([x["bits_vtok"] for x in st])
    cpt = float(np.mean([s["page_chars"] for s in samples]) / samples[0]["vision_tokens"])
    bits, blo, bhi = boot_mean(b)
    gpt, glo, ghi = boot_ratio(gap, n)
    frac, flo, fhi = boot_ratio(gap, ref)
    cers = [s["cer"] for s in samples if s.get("cer") is not None]
    row = {**meta, "pages": len(samples), "chars_per_token": cpt,
           "window_chars_mean": float(np.mean([s["window_chars"] for s in samples])),
           "page_chars_mean": float(np.mean([s["page_chars"] for s in samples])),
           "bits_per_vtok": bits, "bits_per_vtok_ci": [blo, bhi],
           "gap_per_text_token": gpt, "gap_per_text_token_ci": [glo, ghi],
           "recovered_frac": frac, "recovered_frac_ci": [flo, fhi],
           "recovered_chars_per_token": frac * cpt, "recovered_chars_per_token_ci": [flo * cpt, fhi * cpt],
           "loss_real_per_token": float(sum(s["loss_real"] for s in samples) / n.sum()),
           "loss_ref_per_token": float(ref.sum() / n.sum()),
           "cer": float(np.mean(cers)) if cers else None, "cer_pages": len(cers)}
    ct = [s["cer_trunc"] for s in samples if s.get("cer_trunc") is not None]
    if ct:
        row["cer_trunc"] = float(np.mean(ct))
    for arm in ("swap", "blank"):
        v = [s.get(f"loss_{arm}") for s in samples]
        if all(x is not None for x in v):
            row[f"loss_{arm}_per_token"] = float(sum(v) / n.sum())
            row[f"leak_{arm}_below_real"] = int(sum(x < s["loss_real"] for x, s in zip(v, samples)))
    return row


def alignment_stats(G: np.ndarray, shifts: list, reps: int = 10000, seed: int = 0) -> dict:
    """G[page, shift] = per-page bits per vision token. Paired across shifts (same pages, same text).

    * best_vs_worst: best minus worst shift by mean; the CI re-selects best and worst in each page bootstrap
      replicate (as pre-registered) -- this is biased upward by the selection itself, so also:
    * perm_p: permutation p-value of the observed max-min spread, shift labels permuted within each page (H0:
      shifts exchangeable);
    * crossval: best/worst chosen on one half of the pages, the difference measured on the other half (both
      halves, averaged), with a bootstrap CI over random splits x page resamples;
    * aligned_vs_mean: shift (0, 0) minus the mean of the other shifts (no selection), paired page-bootstrap CI.
    """
    rng = np.random.default_rng(seed)
    P, K = G.shape
    m = G.mean(0)
    bi, wi = int(m.argmax()), int(m.argmin())
    obs = float(m[bi] - m[wi])
    idx = rng.integers(0, P, (reps, P))
    mb = G[idx].mean(1)                                     # (reps, K)
    bw = mb.max(1) - mb.min(1)
    perm = np.empty(min(reps, 5000))
    for r in range(len(perm)):
        Gp = np.take_along_axis(G, np.argsort(rng.random((P, K)), 1), 1)
        mp = Gp.mean(0)
        perm[r] = mp.max() - mp.min()

    def cv(Gs, r):
        o = r.permutation(len(Gs))
        h1, h2 = Gs[o[:len(Gs) // 2]], Gs[o[len(Gs) // 2:]]
        d = []
        for A, B in ((h1, h2), (h2, h1)):
            ma = A.mean(0)
            d.append(B[:, ma.argmax()].mean() - B[:, ma.argmin()].mean())
        return float(np.mean(d))
    cvs = np.array([cv(G[rng.integers(0, P, P)], rng) for _ in range(min(reps, 2000))])
    cv_point = float(np.mean([cv(G, np.random.default_rng(seed + k)) for k in range(200)]))
    a0 = shifts.index((0, 0)) if (0, 0) in shifts else 0
    others = [k for k in range(K) if k != a0]
    d = G[:, a0] - G[:, others].mean(1)
    am, alo, ahi = boot_mean(d, reps, seed)
    return {"best_shift": list(shifts[bi]), "worst_shift": list(shifts[wi]), "best_vs_worst": obs,
            "best_vs_worst_ci": [float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))],
            "perm_p": float((np.sum(perm >= obs) + 1) / (len(perm) + 1)),
            "crossval_best_vs_worst": cv_point,
            "crossval_ci": [float(np.percentile(cvs, 2.5)), float(np.percentile(cvs, 97.5))],
            "aligned_vs_mean": am, "aligned_vs_mean_ci": [alo, ahi],
            "mean_by_shift": {f"{s[0]},{s[1]}": float(v) for s, v in zip(shifts, m)}}


# --------------------------------------------------------------------------------------------- reporting

def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def fmt_ci(v, ci, d=2):
    return f"{v:.{d}f} ({ci[0]:.{d}f}-{ci[1]:.{d}f})"


def report(out: Path) -> str:
    rows = load_jsonl(out / "runs.jsonl")
    if not rows:
        return ""
    backfill_cer_trunc(rows, load_jsonl(out / "samples.jsonl"))
    lines = [f"# Capacity curve: {out.name}", "",
             "bits/vtok = conservative-blind gap (min of swap, blank) in bits per vision token, extrapolated from "
             "the scored window to the page; recovered chars/vtok = chars/vtok x recovered fraction (gap / blind "
             "loss); 95% page-bootstrap CIs. CER = greedy transcription of the window (first pages only).", ""]
    plain = [r for r in rows if tuple(r.get("shift", (0, 0))) == (0, 0) and not r.get("shifted")]
    if plain:
        lines += ["| encoding | text | pages | chars/vtok | window chars | gap nats/text tok | bits/vtok | "
                  "recovered frac | recovered chars/vtok | CER (pages) | CER cut to window |", "|" + "---|" * 11]
        for r in sorted(plain, key=lambda r: (r["text"], r["chars_per_token"])):
            cer = f"{100 * r['cer']:.1f}% ({r['cer_pages']})" if r.get("cer") is not None else "-"
            lines.append(f"| {r['encoding']} | {r['text']} | {r['pages']} | {r['chars_per_token']:.1f} | "
                         f"{r['window_chars_mean']:.0f} | {fmt_ci(r['gap_per_text_token'], r['gap_per_text_token_ci'], 3)} | "
                         f"{fmt_ci(r['bits_per_vtok'], r['bits_per_vtok_ci'])} | "
                         f"{fmt_ci(r['recovered_frac'], r['recovered_frac_ci'])} | "
                         f"{fmt_ci(r['recovered_chars_per_token'], r['recovered_chars_per_token_ci'])} | {cer} | "
                         + (f"{100 * r['cer_trunc']:.1f}% |" if r.get("cer_trunc") is not None else "- |"))
        lines.append("")
        _plot(plain, out / "capacity.png")
        lines += ["![capacity](capacity.png)", ""] + run0_checks(plain)
    shifted = [r for r in rows if r.get("shifted")]
    if shifted:
        S_ = load_jsonl(out / "samples.jsonl")
        lines += ["## Grid alignment (paired over pages; bits per vision token)", "",
                  "| encoding | text | pages | best shift | worst shift | best - worst (CI, re-selected) | perm p | "
                  "cross-validated best - worst (CI) | (0,0) - mean of others (CI) |", "|" + "---|" * 9]
        for (e, t) in sorted({(r["encoding"], r["text"]) for r in shifted}):
            G, shifts = alignment_matrix(S_, e, t)
            if G is None:
                continue
            a = alignment_stats(G, shifts)
            lines.append(f"| {e} | {t} | {G.shape[0]} | {tuple(a['best_shift'])} | {tuple(a['worst_shift'])} | "
                         f"{fmt_ci(a['best_vs_worst'], a['best_vs_worst_ci'], 3)} | {a['perm_p']:.3f} | "
                         f"{fmt_ci(a['crossval_best_vs_worst'], a['crossval_ci'], 3)} | "
                         f"{fmt_ci(a['aligned_vs_mean'], a['aligned_vs_mean_ci'], 3)} |")
            with open(out / "alignment.json", "a") as f:
                f.write(json.dumps({"encoding": e, "text": t, **a}) + "\n")
            lines += ["", "mean bits/vtok by (dx, dy): " + ", ".join(f"({k}) {v:.3f}"
                                                                    for k, v in a["mean_by_shift"].items()), ""]
        lines.append("")
    md = "\n".join(lines)
    (out / "summary.md").write_text(md)
    return md


def spearman(x, y, reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Spearman rho (average ranks for ties) and a two-sided permutation p."""
    def rank(v):
        v = np.asarray(v, float)
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[o] = np.arange(len(v))
        for u in np.unique(v):
            r[v == u] = r[v == u].mean()
        return r
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(reps)])
    return rho, float((np.sum(np.abs(null) >= abs(rho) - 1e-12) + 1) / (reps + 1))


def run0_checks(rows: list[dict]) -> list[str]:
    """The Run 0 criteria as operationalised in PLAN_3090 amendment 2b (descriptive lines for summary.md)."""
    out = ["## Run 0 checks (PLAN_3090 amendment 2b)", ""]
    nat = sorted([r for r in rows if r["text"] == "natural"], key=lambda r: r["chars_per_token"])
    if len(nat) >= 3:
        b = [r["bits_per_vtok"] for r in nat]
        k = int(np.argmax(b))
        out.append(f"* natural: sparsest recovered fraction {nat[0]['recovered_frac']:.2f} (needs >= 0.8); max bits/vtok "
                   f"{b[k]:.2f} at {nat[k]['encoding']} (interior: {0 < k < len(nat) - 1}); densest point CI upper "
                   f"{nat[-1]['bits_per_vtok_ci'][1]:.2f} vs max {b[k]:.2f}")
    pts = [r for r in rows if r.get("cer") is not None]
    if len(pts) >= 3:
        rho, p = spearman([r["gap_per_text_token"] for r in pts], [r["cer"] for r in pts])
        out.append(f"* gap per text token vs CER over {len(pts)} points: Spearman rho = {rho:.2f} (permutation "
                   f"p = {p:.3g}; needs <= -0.8)")
        pt = [r for r in pts if r.get("cer_trunc") is not None]
        if len(pt) >= 3:
            rho, p = spearman([r["gap_per_text_token"] for r in pt], [r["cer_trunc"] for r in pt])
            out.append(f"* (added after the first rows, amendment 2c) the same with CER cut to the window: rho = "
                       f"{rho:.2f} (p = {p:.3g})")
    by = {(r["encoding"], r["text"]): r for r in rows}
    pairs = [(by[(e, "natural")], by[(e, "shuffled")]) for e, t in by if t == "natural" and (e, "shuffled") in by]
    if pairs:
        worse = sum(s["recovered_frac"] < n["recovered_frac"] for n, s in pairs)
        out.append(f"* shuffled recovered fraction below natural at {worse} of {len(pairs)} pitches (needs >= 4 of 6): "
                   + ", ".join(f"{n['encoding']} {n['recovered_frac']:.2f}/{s['recovered_frac']:.2f}"
                               for n, s in pairs))
    return out + [""]


def backfill_cer_trunc(rows: list[dict], samples: list[dict]) -> None:
    """Rows written before cer_trunc existed: compute it from their samples' hypotheses (targets re-rendered)."""
    for r in rows:
        if r.get("cer_trunc") is not None or not r.get("cer_pages"):
            continue
        sel = [s for s in samples if s["encoding"] == r["encoding"] and s["text"] == r["text"]
               and s["shift"] == r["shift"] and s.get("hyp") is not None]
        if not sel:
            continue
        ps = None
        if any(s.get("target") is None for s in sel):
            ps = PageSet(r["encoding"], r["text"], r["pages"], r.get("seed", 0), r.get("margin", 0))
        base, f = parse_encoding(r["encoding"])
        enc = ps.enc if ps else get_encoding_s(base, (S - r.get("margin", 0)) // f)
        r["cer_trunc"] = float(np.mean([
            cer_trunc(s["target"] if s.get("target") is not None else T.transcript(ps.pages[s["page"]], *s["window"]),
                      T.clean_transcript(enc, s["hyp"])) for s in sel]))


def alignment_matrix(samples: list[dict], enc: str, text: str):
    sel = [s for s in samples if s["encoding"] == enc and s["text"] == text and "loss_real" in s]
    shifts = sorted({tuple(s["shift"]) for s in sel})
    pages = sorted({s["page"] for s in sel})
    by = {(s["page"], tuple(s["shift"])): s for s in sel}
    pages = [p for p in pages if all((p, sh) in by for sh in shifts)]
    if not pages or len(shifts) < 2:
        return None, shifts
    G = np.array([[sample_stats(by[(p, sh)])["bits_vtok"] for sh in shifts] for p in pages])
    return G, shifts


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    colours = {"natural": "#1f6fb4", "shuffled": "#d9622b"}
    for text in sorted({r["text"] for r in rows}):
        rs = sorted([r for r in rows if r["text"] == text], key=lambda r: r["chars_per_token"])
        x = [r["chars_per_token"] for r in rs]
        for ax, key in ((axs[0], "bits_per_vtok"), (axs[1], "recovered_chars_per_token")):
            y = np.array([r[key] for r in rs])
            ci = np.array([r[key + "_ci"] for r in rs])
            ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], marker="o", capsize=3, label=text,
                        color=colours.get(text))
    xs = sorted({r["chars_per_token"] for r in rows})
    axs[1].plot(xs, xs, ":", color="grey", label="all chars recovered")
    for ax, t in ((axs[0], "bits recovered per vision token"), (axs[1], "recovered chars per vision token")):
        ax.set_xscale("log")
        ax.set_xlabel("chars per vision token")
        ax.set_title(t)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encodings", nargs="+", default=["latin_p16"])
    ap.add_argument("--texts", nargs="+", default=["natural", "shuffled"], choices=["natural", "shuffled"])
    ap.add_argument("--pages", type=int, default=32)
    ap.add_argument("--window-chars", type=int, default=2000)
    ap.add_argument("--window", default="top", choices=["top", "mid"])
    ap.add_argument("--arms", nargs="+", default=["real", "swap", "blank"], choices=["real", "swap", "blank"])
    ap.add_argument("--shifts", type=int, nargs="*", default=None, help="Run 0b: score every (dx, dy) pair")
    ap.add_argument("--shift-margin", type=int, default=None, help="default: the largest shift")
    ap.add_argument("--cer-pages", type=int, default=3, help="greedy-transcribe the first N pages per point")
    ap.add_argument("--cer-max-new", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run-name", default="capacity")
    ap.add_argument("--out-root", default=str(ROOT / "encoder_probe" / "results"))
    ap.add_argument("--tiny", action="store_true", help="random-init tiny model (CPU tests)")
    ap.add_argument("--adapter", default=None, help="finetune_qwen.py adapter (.pt): measure the TRAINED model")
    ap.add_argument("--report", action="store_true", help="only rewrite summary.md / plots from the files")
    ap.add_argument("--max-points", type=int, default=None, help="stop after this many new points (tests)")
    a = ap.parse_args(argv)
    out = Path(a.out_root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    if a.report:
        print(report(out))
        return 0
    assert "real" in a.arms and len(a.arms) >= 2, "need the real arm and at least one reference arm"
    from encoder_probe.ep import qwen_ft as Q
    from encoder_probe.ep.encoders import resolve_device
    from vbpe.metrics import cer as cer_fn

    shifts = [(0, 0)] if a.shifts is None else [(dx, dy) for dy in a.shifts for dx in a.shifts]
    margin = (max(a.shifts) if a.shifts else 0) if a.shift_margin is None else a.shift_margin
    done = {(s["encoding"], s["text"], tuple(s["shift"]), s["page"]): s for s in load_jsonl(out / "samples.jsonl")}
    rows_done = {(r["encoding"], r["text"], tuple(r["shift"])) for r in load_jsonl(out / "runs.jsonl")}
    todo = [(e, t, sh) for e in a.encodings for t in a.texts for sh in shifts if (e, t, sh) not in rows_done]
    if not todo:
        print("[cap] all points done", flush=True)
        report(out)
        return 0

    dev = resolve_device(a.device)
    torch.set_num_threads(a.threads)
    if dev.type == "cpu" and not a.tiny:
        torch.set_flush_denormal(True)
    t0 = time.time()
    model, tok = Q.load_model(dev, tiny=a.tiny, bf16_cpu=dev.type == "cpu" and not a.tiny)
    if a.adapter:
        st = torch.load(a.adapter, map_location="cpu", weights_only=False)
        lo = st["lora"]
        model = Q.make_trainable(model, lo["r"], lo["alpha"], lo["dropout"], lo["vision_blocks"], lo["train_merger"])
        Q.load_trainable_state(model, st["trainable"])
        print(f"[cap] adapter {Path(a.adapter).name} ({st['encoding']}, {st['steps']} steps)", flush=True)
    model.eval()
    print(f"[cap] model loaded in {time.time() - t0:.0f}s; {len(todo)} points to do", flush=True)
    V = Q.image_tokens(S)
    autocast = dev.type == "cuda"

    def loss_of(img, prompt, target):
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=autocast):
            loss, n = Q.lm_loss(model, Q.collate([{"imgs": [img], "prompt": prompt, "answer": target}], tok, dev))
        return loss.item() * n, n

    sets: dict = {}
    new_points = 0
    for (e, t, sh) in todo:
        key = (e, t)
        if key not in sets:
            sets.clear()
            sets[key] = PageSet(e, t, a.pages, a.seed, margin)
        ps = sets[key]
        t1 = time.time()
        samples = []
        for i, pg in enumerate(ps.pages):
            k = (e, t, sh, i)
            if k in done:
                samples.append(done[k])
                continue
            wa, wb = window(pg, a.window_chars, a.window)
            prompt = T.transcribe_prompt(wa, wb, len(pg.refs))
            target = T.transcript(pg, wa, wb)
            s = {"encoding": e, "text": t, "shift": list(sh), "page": i, "window": [wa, wb],
                 "lines": len(pg.refs), "window_chars": sum(len(r) for r in pg.refs[wa:wb]),
                 "page_chars": pg.n_chars, "vision_tokens": V, "prompt": prompt}
            s["loss_real"], s["n_tokens"] = loss_of(ps.image(i, sh), prompt, target)
            s["loss_swap"] = loss_of(ps.image((i + 1) % len(ps.pages), sh), prompt, target)[0] \
                if "swap" in a.arms else None
            if "blank" in a.arms:
                prev = next((v for kk, v in done.items() if kk[:2] == (e, t) and kk[3] == i
                             and v.get("loss_blank") is not None and v["window"] == [wa, wb]), None)
                s["loss_blank"] = prev["loss_blank"] if prev else loss_of(blank_image(), prompt, target)[0]
            else:
                s["loss_blank"] = None
            s["cer"] = s["hyp"] = s["cer_trunc"] = None
            s["target"] = target
            if i < a.cer_pages:
                n_tgt = len(tok.encode(target))
                (hyp,) = Q.generate(model, tok, [{"imgs": [ps.image(i, sh)], "prompt": prompt}], dev,
                                    min(a.cer_max_new, int(1.25 * n_tgt) + 16), autocast=autocast)
                s["hyp"] = hyp
                s["cer"] = cer_fn(target, T.clean_transcript(ps.enc, hyp))
                s["cer_trunc"] = cer_trunc(target, T.clean_transcript(ps.enc, hyp))
            done[k] = s
            samples.append(s)
            with open(out / "samples.jsonl", "a") as f:
                f.write(json.dumps(s) + "\n")
            st = sample_stats(s)
            print(f"[cap] {e} {t} {sh} page {i}: real {s['loss_real'] / s['n_tokens']:.3f} ref "
                  f"{st['ref'] / s['n_tokens']:.3f} nats/tok, {st['bits_vtok']:.2f} bits/vtok"
                  + (f", CER {100 * s['cer']:.1f}%" if s["cer"] is not None else "")
                  + f" ({time.time() - t1:.0f}s)", flush=True)
        meta = {"phase": "capacity_curve", "encoding": e, "text": t, "shift": list(sh), "shifted": a.shifts is not None,
                "margin": margin, "upscale": ps.f, "window_mode": a.window, "window_chars_max": a.window_chars,
                "arms": a.arms, "seed": a.seed, "adapter": a.adapter, "device": str(dev),
                "model": "tiny" if a.tiny else Q.QWEN_REPO, "time_s": time.time() - t1}
        row = point_row(samples, meta)
        with open(out / "runs.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[cap] POINT {e} {t} {sh}: {row['chars_per_token']:.1f} chars/vtok, "
              f"{fmt_ci(row['bits_per_vtok'], row['bits_per_vtok_ci'])} bits/vtok, recovered "
              f"{fmt_ci(row['recovered_frac'], row['recovered_frac_ci'])}, CER {row['cer']}", flush=True)
        new_points += 1
        if a.max_points and new_points >= a.max_points:
            break
    report(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
