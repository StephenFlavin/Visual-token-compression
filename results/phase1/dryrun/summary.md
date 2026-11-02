# Phase 1 dryrun (phase 1)

Rows: 96 total, 96 usable, 0 no-fit (scored CER=1), 0 errors. Total recorded cost: $0.00.

P@x = px²/char at which the isotonic-smoothed mean CER first reaches x (log-x interpolation). ratio = P(scheme)/P(best Latin) (best of latin, latin_condensed at that threshold); CI = paired passage bootstrap. compr = text tokens / vision tokens at P (Claude: px/750, text ≈ chars/4). '≤' = already below threshold at the smallest budget.

## Verdicts

* K2 (Phase 1: subst P@10% >= 20% below best Latin, natural or scrambled): UNDETERMINED - no substitution scheme in data.
* P1 (quadrant needs >= 30% more px^2/char than best Latin, at P@10%): NOT confirmed; quad3/natural ratio 0.61

## Condition: natural

* best Latin @ 3%: **latin_condensed** (26.7 px²/char)
* best Latin @ 10%: **latin_condensed** (8.0 px²/char)

| scheme | P@3% [95% CI] | ratio@3% [CI] | vis tok/1k@3% | compr@3% | P@10% [95% CI] | ratio@10% [CI] | vis tok/1k@10% | compr@10% |
|---|---|---|---|---|---|---|---|---|
| latin | 37.0 [33.8, 40.6] | 1.38 [1.28, 1.49] | 49.4 | 5.06 | 11.6 [10.7, 12.8] | 1.44 [1.35, 1.55] | 15.4 | 16.19 |
| latin_condensed | 26.7 [24.7, 30.5] | 1.00 [1.00, 1.00] | 35.7 | 7.01 | 8.0 [7.5, 8.5] | 1.00 [1.00, 1.00] | 10.7 | 23.37 |
| quad3 | 15.5 [14.8, 17.1] | 0.58 [0.51, 0.65] | 20.7 | 12.07 | 4.9 [4.6, 5.3] | 0.61 [0.56, 0.64] | 6.5 | 38.28 |

Mean CER by budget:

| scheme | 4 | 6 | 9 | 13 | 18 | 27 | 40 | 60 |
|---|---|---|---|---|---|---|---|---|
| latin | 0.289 | 0.190 | 0.130 | 0.086 | 0.059 | 0.041 | 0.027 | 0.018 |
| latin_condensed | 0.215 | 0.133 | 0.087 | 0.064 | 0.048 | 0.030 | 0.020 | 0.013 |
| quad3 | 0.118 | 0.082 | 0.053 | 0.037 | 0.024 | 0.017 | 0.011 | 0.007 |

![CER curves](cer_curves.png)

![P@ bars](p_at_bars.png)
