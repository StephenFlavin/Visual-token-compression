# Phase 1 latin_natural (phase 1)

Rows: 84 total, 84 usable, 0 no-fit (scored CER=1), 0 errors. Total recorded cost: $1.92.

P@x = px²/char at which the isotonic-smoothed mean CER first reaches x (log-x interpolation). ratio = P(scheme)/P(best Latin) (best of latin, latin_condensed at that threshold); CI = paired passage bootstrap. compr = text tokens / vision tokens at P (Claude: px/784 = 28x28 patches, text ≈ chars/4). '≤' on P = already below threshold at the smallest budget; on a ratio '≤'/'≥'/'?' = upper/lower bound/undetermined because of that censoring.

## Verdicts

* K2 (Phase 1: subst P@10% >= 20% below best Latin, natural or scrambled): UNDETERMINED - no substitution scheme in data.

## Condition: natural

* best Latin @ 3%: **latin** (29.4 px²/char)
* best Latin @ 10%: **latin** (25.1 px²/char)

| scheme | P@3% [95% CI] | ratio@3% [CI] | vis tok/1k@3% | compr@3% | P@10% [95% CI] | ratio@10% [CI] | vis tok/1k@10% | compr@10% |
|---|---|---|---|---|---|---|---|---|
| latin | 29.4 [28.6, 30.2] | 1.00 [1.00, 1.00] | 37.5 | 6.66 | 25.1 [24.2, 25.7] | 1.00 [1.00, 1.00] | 32.1 | 7.79 |
| latin_condensed | 32.5 [31.5, 35.6] | 1.11 [1.06, 1.23] | 41.5 | 6.03 | 29.0 [26.3, 29.8] | 1.15 [1.06, 1.19] | 37.0 | 6.76 |

Mean CER by budget:

| scheme | 14 | 17 | 20 | 23 | 27 | 32 | 40 |
|---|---|---|---|---|---|---|---|
| latin | 0.692 | 0.345 | 0.276 | 0.167 | 0.043 | 0.010 | 0.004 |
| latin_condensed | 0.711 | 0.676 | 0.403 | 0.139 | 0.133 | 0.026 | 0.009 |

## Refusal / non-transcription rate (heuristic)

| condition | latin | latin_condensed |
|---|---|---|
| natural | 5% (n=42) | 7% (n=42) |

![CER curves](cer_curves.png)

![P@ bars](p_at_bars.png)
