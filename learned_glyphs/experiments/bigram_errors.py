"""Do the learned glyph alphabets read worse on FREQUENT letter pairs? (user question, 2026-11-06)

Hypothesis: glyphs that often appear next to each other may have shapes that clash when adjacent, so a learned
alphabet would be worse on frequent bigrams than its per-character error rate predicts. The joint training saw
real WikiText lines (frequent pairs in proportion), which predicts the opposite.

Per saved CRNN reader (characters-only banks), on held-out test lines:
* Align ref/hyp (Levenshtein opcodes); each ref char i >= 1 is an error if replaced or deleted. Its context is the
  bigram (ref[i-1], ref[i]); bigram and char frequencies are counted on the same test lines.
* **Stratified test (the answer):** within each char c, split its occurrences at the median log-frequency of
  their bigrams (frequent vs rare contexts) and take error(frequent) - error(rare). Average over chars, weighted by
  occurrences. Line-bootstrap 95% CI. The hypothesis is supported if the CI is above 0.
* Also reported:
  * error rate by bigram-frequency quintile (unstratified);
  * the share of substitution errors between glyph pairs that are (near-)identical in the bank (< 1 px);
  * the fixed-font bank (lg_fixed_n0_h5, ordinary Latin) as the control.

    python3 learned_glyphs/experiments/bigram_errors.py --n-test 3000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ARMS = {   # tag -> (bank, reader)
    "learned_n0_pb_h5": ("e1long/banks/lg_learned_n0_pb_h5.pt", "e1long/readers/lg_learned_n0_pb_h5.pt"),
    "learned_n0_h5": ("e1/banks/lg_learned_n0_h5.pt", "e1/readers/lg_learned_n0_h5.pt"),
    "fixed_n0_h5 (Latin)": ("e1/banks/lg_fixed_n0_h5.pt", "e1/readers/lg_fixed_n0_h5.pt"),
    "learned_n0_h4": ("e1long/banks/lg_learned_n0_h4.pt", "e1long/readers/lg_learned_n0_h4.pt"),
}


def ref_errors(ref: str, hyp: str) -> np.ndarray:
    """1 where ref char i is replaced or deleted in the alignment, else 0."""
    from rapidfuzz.distance import Levenshtein
    err = np.zeros(len(ref), dtype=np.int8)
    for tag, i1, i2, _, _ in Levenshtein.opcodes(ref, hyp):
        if tag in ("replace", "delete"):
            err[i1:i2] = 1
    return err


def substitutions(ref: str, hyp: str) -> list[tuple[str, str]]:
    from rapidfuzz.distance import Levenshtein
    out = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
        if tag == "replace" and i2 - i1 == j2 - j1:
            out += list(zip(ref[i1:i2], hyp[j1:j2]))
    return out


def analyse(pairs: list[tuple[str, str]], near_identical: set, n_boot: int = 1000, seed: int = 0) -> dict:
    lines = []
    big, uni = Counter(), Counter()
    for ref, hyp in pairs:
        e = ref_errors(ref, hyp)
        lines.append((ref, e))
        uni.update(ref)
        big.update(zip(ref, ref[1:]))
    # occurrence table: (line, char, log bigram freq, error)
    occ = [(k, ref[i], math.log(big[(ref[i - 1], ref[i])]), int(e[i]))
           for k, (ref, e) in enumerate(lines) for i in range(1, len(ref))]
    L = np.array([o[0] for o in occ])
    C = np.array([o[1] for o in occ])
    F = np.array([o[2] for o in occ])
    E = np.array([o[3] for o in occ], dtype=float)
    chars = [c for c, n in uni.items() if n >= 40]
    med = {c: np.median(F[C == c]) for c in chars}

    def strat(idx):
        num = den = 0.0
        Ci, Fi, Ei = C[idx], F[idx], E[idx]
        for c in chars:
            m = Ci == c
            if m.sum() < 20:
                continue
            hi, lo = m & (Fi > med[c]), m & (Fi <= med[c])
            if hi.sum() < 5 or lo.sum() < 5:
                continue
            num += m.sum() * (Ei[hi].mean() - Ei[lo].mean())
            den += m.sum()
        return num / den if den else float("nan")

    d = strat(np.arange(len(E)))
    rng = np.random.default_rng(seed)
    by_line = [np.where(L == k)[0] for k in range(len(lines))]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(lines), len(lines))
        boots.append(strat(np.concatenate([by_line[k] for k in pick])))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    q = np.quantile(F, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(F, q)
    quint = [float(E[bins == b].mean()) for b in range(5)]   # 0 = rarest bigrams
    subs = [s for ref, hyp in pairs for s in substitutions(ref, hyp)]
    ni = sum(1 for a, b in subs if frozenset((a, b)) in near_identical)
    return {"n_lines": len(pairs), "n_chars": int(len(E)), "cer_ref": float(E.mean()),
            "strat_freq_minus_rare": float(d), "ci": [float(lo), float(hi)],
            "err_by_bigram_quintile_rare_to_freq": quint, "n_subs": len(subs),
            "subs_between_near_identical_glyphs": ni, "share_near_identical": ni / max(1, len(subs))}


def near_identical_pairs(bank_path: Path) -> set:
    """Glyph pairs < 1 px apart (L1 at 1x), from the bank bitmaps."""
    from learned_glyphs.lg.scheme import LearnedGlyphScheme
    sc = LearnedGlyphScheme(bank_path)
    out = set()
    maps = sc.bitmaps
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            a, b = maps[i], maps[j]
            if a.shape != b.shape:
                continue
            if np.abs(a - b).sum() / (sc.ss * sc.ss) < 1.0:
                out.add(frozenset((sc.symbols[i], sc.symbols[j])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out", default=str(ROOT / "learned_glyphs" / "results" / "bigram_errors"))
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from learned_glyphs.lg.report import test_pass
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for tag in a.arms:
        bank, reader = (ROOT / "learned_glyphs" / "results" / p for p in ARMS[tag])
        pairs = test_pass(str(bank), str(reader), n_test=a.n_test)
        res[tag] = analyse(pairs, near_identical_pairs(bank))
        res[tag]["n_near_identical_pairs"] = len(near_identical_pairs(bank))
        r = res[tag]
        print(f"{tag}: CER {r['cer_ref']:.4f}; frequent - rare context (within char) {r['strat_freq_minus_rare']:+.4f} "
              f"(CI {r['ci'][0]:+.4f} to {r['ci'][1]:+.4f}); near-identical share of subs "
              f"{r['share_near_identical']:.2f}", flush=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    md = ["# Bigram error check (learned_glyphs/experiments/bigram_errors.py)", "",
          f"{a.n_test} held-out test lines per reader, characters-only banks. 'Frequent - rare' = within each char, "
          "error rate in its more-frequent bigram contexts minus its rarer ones (occurrence-weighted, line "
          "bootstrap 95% CI). Positive = frequent pairs read worse (the hypothesis).", "",
          "| reader | CER | frequent - rare (CI) | error by bigram quintile, rare → frequent | near-identical glyph "
          "pairs | share of substitutions between them |", "|---|---|---|---|---|---|"]
    for tag, r in res.items():
        md.append(f"| {tag} | {r['cer_ref']:.4f} | {r['strat_freq_minus_rare']:+.4f} ({r['ci'][0]:+.4f} to "
                  f"{r['ci'][1]:+.4f}) | {' / '.join(f'{x:.3f}' for x in r['err_by_bigram_quintile_rare_to_freq'])} "
                  f"| {r['n_near_identical_pairs']} | {r['share_near_identical']:.2f} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
