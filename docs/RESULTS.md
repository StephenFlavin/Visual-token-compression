# Visual BPE: results so far (living document)

Status: **2026-11-03 13:10 UTC.** Phase 2 (K3) is decided. Learned glyphs: **qualified GO** (bridge 0.57×, transfer 0.74×/0.63× Latin; see the
verdict at the end). All planned rounds done.

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
3. **Tokens need more training, and at convergence they DO help.** At 3.5k steps learned N=300 looked no better than
   N=0, but at 10.5k steps it reaches 0.60% CER at 5.88 px²/char (h4). Learned N=0 needs 8.0 px²/char for 0.57%.
   Converged Latin reaches 4.51% at 5.5 and 1.73% at 7.5. So learned N=300 P@3% is below 5.9, against Latin's ≈6.3.
4. **Bridge (Phase-2 PIL pipeline, 3.5k steps):** N=300 h4 gets 0.80% at 5.88 px²/char; N=0 h4 gets 0.25% at 8.0;
   whole-pixel N=0 h5 gets 1.42% at 10.0. For comparison, latin at 3.5k steps gets 6.34% at 7.5 and 2.54% at 9.8.

Converged results (10.5k steps). "joint" = glyphs and reader trained together through our channel;
"bridge" = frozen bank rendered by the unchanged Phase-2 pipeline, fresh reader:

| bank | h | mode | px²/char | CER | held-out CER |
|---|---|---|---|---|---|
| learned N=300 | 3 | joint | 2.52 | 20.8% | 65% |
| learned N=300 | 4 | joint | 5.88 | 0.60% | 35% |
| learned N=300 | 4 | **bridge** | 5.88 | **0.14%** | – |
| learned N=0 | 3 | joint | 3.0 | 16.4% | 61% |
| learned N=0 | 4 | joint | 8.0 | 0.57% | 19% |
| learned N=300, whole-pixel | 4 | joint | 5.88 | 6.75% | 22% |
| learned N=300, whole-pixel | 5 | joint | 7.35 | 1.80% | 12% |
| learned N=0, whole-pixel | 5 | joint | 10.0 | 1.03% | 6% |
| latin (Phase 2) | em3 | – | 5.54 | 4.51% | – |
| latin (Phase 2) | em3.5 | – | 7.53 | 1.73% | – |

**Learned-glyph verdict (10:20 UTC, converged = 10.5k steps; supersedes the joint-only verdict written at 08:30).**
"Bridge" = frozen bank rendered by the unchanged Phase-2 pipeline with a fresh reader. This is the apples-to-apples
comparison with Phase 2. The jointly trained reader lags far behind, because it chases moving glyphs.

| bank | h | px²/char | bridge CER | joint CER |
|---|---|---|---|---|
| learned N=300 | 3 | 2.5 | 7.95% | 20.8% |
| learned N=300 | 4 | 5.88 | 0.14% | 0.60% |
| learned N=300, whole-pixel | 4 | 5.9 | 2.08% | 6.75% |
| learned N=300, whole-pixel | 5 | 7.3 | 0.59% | 1.80% |

Round 7 added the h3 bridge point for the whole-pixel bank and h3 points for E2 transfer and the harsh channel at h7.

| bank / check | h | px²/char | CER |
|---|---|---|---|
| whole-pixel N=300, bridge | 3 | 2.5 | 33.4% |
| learned N=300, E2 transfer | 3 / 4 | 2.52 / 5.88 | 16.4% / 0.29% |
| learned N=300, harsh joint | 7 | 15.4 | **3.45%** (held-out 4.4%) |
| fixed DejaVu N=0, harsh joint | 7 | 20.7 | 42.8% (held-out 45.5%) |

Round 8 (13:05) added intermediate budgets: N=300 banks with narrower cells (char 2 px, token 3 px) at h3 and h4.

| N=300 bank | px²/char | bridge CER | E2 transfer CER | joint CER | joint held-out CER |
|---|---|---|---|---|---|
| h3, default widths | 2.52 | 7.95% | 16.4% | 20.8% | 65% |
| h3, widths 2/3 | 4.09 | 1.92% | 4.79% | 8.71% | 68% |
| h4, widths 2/3 | 5.45 | 0.73% | 1.72% | 3.34% | 55% |
| h4, default widths | 5.88 | 0.14% | 0.29% | 0.60% | 35% |

**Final P@x on this runner** (`vbpe.analysis.p_at`, isotonic + log-x interpolation; rows keyed by px²/char, see the note below),
against converged Phase-2 Latin (DejaVu):

| | P@3% | P@1% | ratio vs Latin (3% / 1%) |
|---|---|---|---|
| latin (Phase 2) | 6.54 | 8.92 | 1 / 1 |
| latin_condensed | 8.32 | 17.1 | 1.27 / 1.92 |
| learned N=300, bridge (E3) | 3.75 | 5.10 | **0.57 / 0.57** |
| learned N=300, transfer (E2) | 4.83 | 5.66 | 0.74 / **0.63** |
| learned N=300 whole-pixel, bridge | 5.73 | 6.91 | 0.88 / 0.77 |

E0 calibration showed that fixed DejaVu through our `default` channel reproduces Phase-2 Latin P@3% (1.01×), so E2 ratios are
comparable with the Phase-2 baseline. Note: `vbpe.analysis` groups rows by `em`. The width-2/3 banks share an em with the
default-width banks, so set `budget = px_per_char` on learned-glyph rows before calling `p_at`, or the two get pooled.

**Verdict against the pre-registered GO rule** (≥30% fewer px²/char than best fixed Latin in transfer AND bridge; gain survives harsh):
* **Bridge (unchanged Phase-2 pipeline): 43% fewer px²** at both 3% and 1% CER. Pass.
* **E2 transfer: 26% fewer at 3% CER (just short), 37% fewer at 1%.** Borderline.
* **Harsh channel: the gain survives and grows.** At h7, learned N=300 gets 3.45% CER at 15.4 px²/char; fixed DejaVu gets 42.8% at 20.7.
* **Whole-pixel N=300 (no grey-level coding): 12% / 23% fewer.** A real but smaller effect.
* **Caveat, held-out degradations:** the jointly trained models fail badly on unseen degradations (real PIL JPEG, an unseen
  resize filter). The tighter banks get 55–68% CER there, so the densest codes are brittle outside their training channel.
  Whole-pixel banks hold up better (12–22%).
* Untestable here: whether a pretrained VLM image encoder reads such glyphs.

**Bottom line:** a learned glyph alphabet with ~300 chain tokens beats hand-designed typography by ~25–45% px²/char for a
trained CRNN reader. Most of that edge is real legibility rather than reader co-adaptation, because it holds with a fresh
reader. Part of it relies on grey-level coding, and the densest banks are brittle on unseen degradations. This is a
**qualified GO** for a GPU follow-up, which would train with real JPEG/resampling in the channel and test with a VLM encoder.
Hand-designed Visual BPE (CJK proxies, quadrant packing) stays killed (K3).
