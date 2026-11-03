# Visual BPE: results so far (living document)

Status: **2026-11-03 04:30 UTC.** Phase 2 (K3) is decided. Learned-glyph experiments are still running
(see `results/queue.txt`).

Metric throughout: **px²/char** = canvas area / source characters. **P@x** = px²/char at which CER
first reaches x (isotonic fit, log-x interpolation). Lower is better.

## One-paragraph answer

Hand-designed Visual BPE (one CJK proxy glyph per frequent character chain, or Hangul-style quadrant
packing) does **not** beat ordinary condensed Latin text per pixel:
* with a zero-shot OCR reader (PP-OCR), it reaches 10% CER at only 0.89× Latin's pixels (pre-registered bar: 0.80×) and is worse at 3% CER;
* with a reader trained from scratch on each scheme to convergence (Phase 2, CRNN), plain DejaVu Latin is the most pixel-efficient scheme at every error level we measured; substitution needs ≥1.18× its pixels at 3% CER (K3 required ≤0.70×);
* quadrant packing needs about 2× the pixels of Latin.

Claude cannot be used to score any non-natural rendering at all (safety classifiers refuse). The one
open thread is the **learned glyph alphabet** sub-project (`learned_glyphs/`): it optimises glyph
shapes jointly with a reader. Its first pilot looks interesting but has not yet passed its anti-cheating
checks (transfer to a fresh reader, harsh channel).

## Kill criteria (pre-registered in PLAN.md)

| ID | Test | Result | Verdict |
|---|---|---|---|
| K1 | substitution saves ≥15% line width vs condensed Latin (analytic) | 32% at N=1000 → only 1.19× larger em on a fixed page | pass (weak) |
| K2 | zero-shot reader: subst ≥20% fewer px² at 10% CER | PP-OCR: 0.89× at 10%, 1.17× (worse) at 3%; scrambled text gives the same → not a language-prior artefact | **negative** |
| K3 | trained reader: subst P@3% ≥30% below best Latin | converged (10.5k steps): latin 1.73% CER at 7.5 px²/char; subst300 2.26% at 8.9; subst1000 5.75% at 7.8 → subst ≥1.18× **worse** | **KILL** |
| P1 | quadrant needs ≥30% more px² | quad3 ≥3.2×, quad4 ≥3.4× vs converged Latin (quad at 3.5k steps) | confirmed |

## Readers

* **Claude (natural Latin only):** DejaVu Sans P@3% = 29.4 px²/char [28.6, 30.2] ≈ 37 vision tokens per
  1k chars (~6.7× text-token cost). Condensed fonts are *worse* for Claude (1.11–1.2×).
* **PP-OCR (rapidocr, zero-shot):** see `results/phase1b/summary/`.
* **CRNN trained per scheme and em (Phase 2):** see `results/phase2/merged/` (longest run per scheme and em).
  This is far more pixel-efficient than Claude (latin_condensed ≈ 12 px²/char at 3% CER vs Claude's 29), because
  the reader is specialised to a single font and has no "natural image" overhead.

## Phase 2 detail (longest run per scheme/em: `results/phase2/merged/`)

Frontier points (10.5k steps unless noted):

| scheme | em | px²/char | CER |
|---|---|---|---|
| latin (DejaVu) | 3.5 | 7.53 | 1.73% |
| latin | 4 | 9.84 | 0.58% |
| latin | 5 | 15.39 | 0.10% |
| latin_condensed | 4.5 | 9.02 | 2.43% |
| latin_condensed | 5 | 11.15 | 0.92% |
| subst300 | 5 | 8.91 | 2.26% |
| subst300 | 6 | 12.84 | 0.67% |
| subst1000 | 5 | 7.81 | 5.75% |
| subst1000 | 6 | 11.26 | 1.01% |
| quad3 / quad4 (3.5k) | 10 | 26.0 / 23.3 | 2.1% / 3.5% |

Why substitution loses even with a trained reader: a 1000-token dictionary saves ~32% of line *width*
(K1), but each proxy glyph must still be legible at that em. A CJK ideograph has far more detail than a
Latin letter, so it needs a larger em to reach the same symbol error rate, and that eats the width saving.
Each symbol error also costs several characters (a whole token), not one.

Convergence mattered a lot: at 3.5k steps subst1000 em7 had 5.6% CER and 10.5k steps brought it to 0.58%.
But Latin improved just as much (em4 2.5% → 0.58%), so the verdict did not change.

**Bug found and fixed (2026-11-03):** latin em4 first scored 62% CER because its test and val sets came from an old
render cache. That cache was written before per-line heights were stored, so its lines were one pixel row taller
than the training batches. The same checkpoint scores 3.8% on fresh renders. Affected rows are in
`runs_stale_cache.jsonl` and were re-run. Plain DejaVu Latin is the best scheme once this is fixed.

## Learned glyph alphabet (learned_glyphs/), status 2026-11-03 06:00 UTC

E0 calibration: fixed DejaVu glyphs through the `default` channel reproduce Phase-2 latin P@3% (1.01×).

Joint training (glyphs + reader, 3.5k steps, default channel). "held-out" = the same model tested on
unseen degradations (real JPEG, unseen resize filter):

| bank | line h (px) | px²/char | CER | held-out CER |
|---|---|---|---|---|
| fixed DejaVu, N=0 | 5 | 10.8 | 6.65% | 54% |
| learned, N=0 | 5 | 10.0 | 0.65% | 11% |
| learned, N=0 | 4 | 8.0 | 1.26% | 20% |
| learned, N=0 | 3 | 3.0* | 26.7% | 62% |
| learned, N=300 | 5 | 7.35 | 3.46% | 30% |
| learned, N=300 | 4 | 5.9 | 4.17% | 46% |
| learned, N=0, whole-pixel (`--pixel-block`) | 5 | 10.0 | 3.08% | 6% |
| learned, N=0, whole-pixel | 4 | 8.0 | 7.01% | 13% |
| learned, N=300, whole-pixel | 5 | 7.35 | 8.31% | 20% |

*h=3 cells are 1 px wide.

Reference (Phase 2, PIL rendering): latin at 3.5k steps reaches 2.54% at 9.8 px²/char. Converged, it reaches 1.73% at 7.5 and 4.51% at 5.5 (P@3% ≈ 6.3).

Checks:
* **Transfer (E2):** fresh readers on the frozen pilot banks (h=7) reach 0.04% (N=0) and 0.30% (N=300), so the glyphs are
  intrinsically legible in this channel.
* **Harsh channel (h=7):** learned N=300 reaches 12.0% CER against 48.5% for fixed DejaVu, so learning helps more, not less.
* **Bridge (E3, unchanged Phase-2 PIL pipeline, 3.5k steps):** learned N=0 h5 gets 0.12% at 10.0 px²/char; N=300 h5 gets 0.64% at 7.35;
  N=300 h6 gets 0.18% at 12.6. That beats converged Latin at a similar budget (1.73% at 7.5).

Interpretation so far:
1. **The learned glyphs are grey-level pixel codes, not letterforms.** Sub-pixel structure becomes grey levels after
   area downsampling (see `learned_glyphs/results/*/atlas/`).
2. **Forcing whole-pixel glyphs removes most of the gain** (h5: 0.65% → 3.08%, about Latin level). So most of the
   advantage comes from grey-level/sub-pixel coding. The plan flagged this as likely channel exploitation, since a VLM
   image encoder or real capture may not preserve it.
   Whole-pixel glyphs are, however, more robust on held-out degradations (6% vs 11%).
3. **Tokens (the Visual-BPE part) do not help:** at 3% CER, learned N=0 needs fewer px² than learned N=300 (S2 ≥ 1).
   The gain that exists is a "better low-res font" effect, not a "more symbols" effect.

Pending: bridge for h4 banks and the whole-pixel bank; 10.5k-step learned runs (h3, h4) for a converged comparison
with converged Latin.
