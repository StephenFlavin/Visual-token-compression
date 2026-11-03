# Visual BPE: results so far (living document)

Status: **2026-11-03, overnight run in progress.** Numbers marked *(pending)* are being refined by
longer (10.5k-step) training runs; see `results/queue.txt` for what is running.

Metric throughout: **px²/char** = canvas area / source characters. **P@x** = px²/char at which CER
first reaches x (isotonic fit, log-x interpolation). Lower is better.

## One-paragraph answer

Hand-designed Visual BPE (one CJK proxy glyph per frequent character chain, or Hangul-style quadrant
packing) does **not** beat ordinary condensed Latin text per pixel:
* with a zero-shot OCR reader (PP-OCR), it reaches 10% CER at only 0.89× Latin's pixels (pre-registered bar: 0.80×) and is worse at 3% CER;
* with a reader trained from scratch on each scheme (Phase 2, CRNN), it is about even or worse at 3% CER, never the ≥30% gain the plan required (K3);
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
| K3 | trained reader: subst P@3% ≥30% below best Latin | 3.5k steps: subst300 1.16× (worse), subst1000 1.94× (worse); 10.5k-step frontier runs *(pending)* | **negative** (pending final) |
| P1 | quadrant needs ≥30% more px² | quad3 1.95×, quad4 2.10× at 3% | confirmed |

## Readers

* **Claude (natural Latin only):** DejaVu Sans P@3% = 29.4 px²/char [28.6, 30.2] ≈ 37 vision tokens per
  1k chars (~6.7× text-token cost). Condensed fonts are *worse* for Claude (1.11–1.2×).
* **PP-OCR (rapidocr, zero-shot):** see `results/phase1b/summary/`.
* **CRNN trained per scheme and em (Phase 2):** see `results/phase2/merged/` (longest run per scheme and em).
  This is far more pixel-efficient than Claude (latin_condensed ≈ 12 px²/char at 3% CER vs Claude's 29), because
  the reader is specialised to a single font and has no "natural image" overhead.

## Phase 2 detail (3.5k steps; 10.5k-step reruns supersede where available)

| scheme | P@3% px²/char | ratio vs best Latin |
|---|---|---|
| latin_condensed | 12.2 | 1.00 |
| latin (DejaVu) | 15.1 *(em4/5 being re-run, see below)* | 1.24 |
| subst300 | 14.2 | 1.16 |
| subst1000 | 23.7 *(not converged at 3.5k; em7 drops 5.6%→0.6% at 10.5k)* | 1.94 |
| quad3 / quad4 | 23.9 / 25.7 | 1.95 / 2.10 |

Convergence matters: at 10.5k steps, subst1000 em7 improves 5.6% → 0.58% and subst300 em7 1.5% → 0.21%.
Latin also improves (em5 0.5% → ~0.1%). The frontier ems (latin_condensed 4.5 and 5, subst300 5 and 6,
subst1000 5 and 6, latin 4.5) are being re-run at 10.5k steps so that every scheme is compared at convergence.

**Bug found and fixed (2026-11-03):** latin em4 scored 62% CER because its test and val sets came from an old render
cache. That cache was written before per-line heights were stored, so its lines were one pixel row taller than the
training batches. The same checkpoint scores 3.8% on fresh renders. Affected rows (latin em4, em5 and em8) were moved to
`runs_stale_cache.jsonl` and are being re-run.

## Learned glyph alphabet (learned_glyphs/)

* E0 calibration: fixed DejaVu glyphs through the `default` channel reproduce Phase-2 latin P@3% (1.01×).
* Pilot at h=7 px (3.5k steps, joint training, default channel):

  | bank | px²/char | CER |
  |---|---|---|
  | fixed DejaVu, N=0 | 20.7 | 0.81% |
  | learned, N=0 | 21.0 | 0.19% |
  | learned, N=300 tokens | 15.4 | 0.27% |

  The learned glyphs are blob-like grey-level codes, not letterforms (`learned_glyphs/results/lg_pilot/atlas/`).
  This is exactly the "channel exploitation" risk the plan guards against. The pilot proves nothing until it passes
  E2 (fresh reader on the frozen bank) and harsh-channel training. Lower heights (h=5, 6), transfer and harsh-channel
  runs are queued.
