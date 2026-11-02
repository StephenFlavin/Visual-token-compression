# Sweep summary (phase 1)

Rows: 585 total, 585 usable, 0 no-fit (scored CER=1), 0 errors. Total recorded cost: $0.00.

P@x = px²/char at which the isotonic-smoothed mean CER first reaches x (log-x interpolation). ratio = P(scheme)/P(best Latin) (best of latin, latin_condensed at that threshold); CI = paired passage bootstrap. compr = text tokens / vision tokens at P (Claude: px/784 = 28x28 patches, text ≈ chars/4). '≤' on P = already below threshold at the smallest budget; on a ratio '≤'/'≥'/'?' = upper/lower bound/undetermined because of that censoring.

## Verdicts

* K2 (Phase 1: subst P@10% >= 20% below best Latin, natural or scrambled): TRIGGERED - zero-shot negative; best case subst1000 on natural: P@10%=24.2 vs latin_condensed 27.2 px^2/char, ratio 0.89 (95% CI 0.85-0.91)
* P2 (subst300 / latin_condensed at P@10%): natural: 0.92 (within +-20%); scrambled: 0.99 (within +-20%)
* P3 (Latin advantage shrinks on scrambled): best subst ratio natural 0.89 -> scrambled 0.90 -> NOT confirmed

## Condition: natural

* best Latin @ 3%: **latin** (38.1 px²/char)
* best Latin @ 10%: **latin_condensed** (27.2 px²/char)

| scheme | P@3% [95% CI] | ratio@3% [CI] | vis tok/1k@3% | compr@3% | P@10% [95% CI] | ratio@10% [CI] | vis tok/1k@10% | compr@10% |
|---|---|---|---|---|---|---|---|---|
| latin | 38.1 [37.7, 38.6] | 1.00 [1.00, 1.00] | 48.6 | 5.15 | 29.2 [29.1, 29.2] | 1.07 [1.05, 1.10] | 37.2 | 6.72 |
| latin_condensed | 62.2 [56.9, 65.9] | 1.63 [1.50, 1.73] | 79.3 | 3.15 | 27.2 [26.5, 27.7] | 1.00 [1.00, 1.00] | 34.7 | 7.20 |
| subst300 | 52.5 [51.3, 53.7] | 1.38 [1.36, 1.41] | 66.9 | 3.74 | 25.1 [24.9, 25.2] | 0.92 [0.91, 0.94] | 32.0 | 7.82 |
| subst1000 | 44.5 [41.3, 47.4] | 1.17 [1.09, 1.23] | 56.8 | 4.40 | 24.2 [23.0, 24.9] | 0.89 [0.85, 0.91] | 30.8 | 8.11 |
| subst3000 | > 90 (not reached) | — | n/r | n/r | 31.0 [27.6, 33.7] | 1.14 [1.04, 1.23] | 39.5 | 6.33 |

Mean CER by budget:

| scheme | 12 | 14 | 16 | 18 | 20 | 23 | 26 | 30 | 34 | 40 | 48 | 60 | 90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latin | 0.757 | 0.634 | 0.296 | 0.280 | 0.272 | 0.262 | 0.226 | 0.068 | 0.059 | 0.018 | 0.015 | 0.004 | 0.005 |
| latin_condensed | 0.741 | 0.609 | 0.496 | 0.348 | 0.179 | 0.107 | 0.109 | 0.082 | 0.029 | 0.021 | 0.063 | 0.032 | 0.012 |
| subst300 | 0.814 | 0.623 | 0.424 | 0.257 | 0.207 | 0.160 | 0.072 | 0.053 | 0.047 | 0.057 | 0.038 | 0.017 | 0.016 |
| subst1000 | 0.732 | 0.523 | 0.304 | 0.251 | 0.196 | 0.119 | 0.070 | 0.054 | 0.058 | 0.039 | 0.023 | 0.011 | 0.014 |
| subst3000 | 0.655 | 0.459 | 0.383 | 0.310 | 0.246 | 0.161 | 0.130 | 0.107 | 0.081 | 0.060 | 0.041 | 0.047 | 0.038 |

## Condition: scrambled

* best Latin @ 3%: **latin** (37.7 px²/char)
* best Latin @ 10%: **latin_condensed** (27.2 px²/char)

| scheme | P@3% [95% CI] | ratio@3% [CI] | vis tok/1k@3% | compr@3% | P@10% [95% CI] | ratio@10% [CI] | vis tok/1k@10% | compr@10% |
|---|---|---|---|---|---|---|---|---|
| latin | 37.7 [37.4, 38.4] | 1.00 [1.00, 1.00] | 48.1 | 5.19 | 29.1 [29.0, 29.2] | 1.07 [1.04, 1.13] | 37.2 | 6.73 |
| latin_condensed | 65.0 [56.2, 69.6] | 1.72 [1.50, 1.81] | 82.9 | 3.01 | 27.2 [25.7, 28.0] | 1.00 [1.00, 1.00] | 34.7 | 7.20 |
| subst300 | 51.4 [49.3, 53.0] | 1.36 [1.32, 1.39] | 65.6 | 3.81 | 27.1 [25.0, 28.2] | 0.99 [0.96, 1.01] | 34.5 | 7.24 |
| subst1000 | 46.8 [45.8, 48.1] | 1.24 [1.22, 1.25] | 59.7 | 4.19 | 24.5 [22.7, 25.4] | 0.90 [0.88, 0.91] | 31.2 | 8.01 |
| subst3000 | > 90 (not reached) | — | n/r | n/r | 32.9 [28.5, 36.6] | 1.21 [1.11, 1.31] | 42.0 | 5.95 |

Mean CER by budget:

| scheme | 12 | 14 | 16 | 18 | 20 | 23 | 26 | 30 | 34 | 40 | 48 | 60 | 90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latin | 0.762 | 0.486 | 0.291 | 0.285 | 0.267 | 0.263 | 0.222 | 0.068 | 0.060 | 0.014 | 0.011 | 0.003 | 0.005 |
| latin_condensed | 0.754 | 0.614 | 0.511 | 0.369 | 0.193 | 0.109 | 0.106 | 0.087 | 0.031 | 0.021 | 0.063 | 0.034 | 0.014 |
| subst300 | 0.825 | 0.675 | 0.455 | 0.267 | 0.216 | 0.172 | 0.117 | 0.050 | 0.047 | 0.061 | 0.037 | 0.014 | 0.013 |
| subst1000 | 0.790 | 0.595 | 0.363 | 0.272 | 0.224 | 0.128 | 0.071 | 0.063 | 0.054 | 0.051 | 0.026 | 0.012 | 0.013 |
| subst3000 | 0.723 | 0.520 | 0.414 | 0.365 | 0.301 | 0.196 | 0.156 | 0.111 | 0.096 | 0.066 | 0.047 | 0.049 | 0.032 |

## Refusal / non-transcription rate (heuristic)

| condition | latin | latin_condensed | subst300 | subst1000 | subst3000 |
|---|---|---|---|---|---|
| natural | 0% (n=78) | 0% (n=78) | 0% (n=78) | 0% (n=78) | 0% (n=78) |
| scrambled | 0% (n=39) | 0% (n=39) | 0% (n=39) | 0% (n=39) | 0% (n=39) |

## Prior credit: mean CER(natural) - mean CER(scrambled)

Negative = the language prior helps. Unpaired (different passages per variant).

Setting: all

| scheme | 12 | 14 | 16 | 18 | 20 | 23 | 26 | 30 | 34 | 40 | 48 | 60 | 90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latin | -0.005 | +0.149 | +0.005 | -0.005 | +0.005 | -0.001 | +0.003 | +0.001 | -0.001 | +0.004 | +0.004 | +0.001 | -0.000 |
| latin_condensed | -0.014 | -0.005 | -0.015 | -0.021 | -0.014 | -0.002 | +0.003 | -0.005 | -0.002 | +0.000 | +0.001 | -0.002 | -0.002 |
| subst300 | -0.010 | -0.053 | -0.032 | -0.010 | -0.009 | -0.013 | -0.045 | +0.004 | +0.001 | -0.004 | +0.002 | +0.003 | +0.003 |
| subst1000 | -0.057 | -0.072 | -0.059 | -0.020 | -0.028 | -0.009 | -0.001 | -0.009 | +0.004 | -0.012 | -0.003 | -0.001 | +0.001 |
| subst3000 | -0.068 | -0.061 | -0.031 | -0.055 | -0.054 | -0.035 | -0.026 | -0.004 | -0.016 | -0.006 | -0.006 | -0.002 | +0.006 |


![CER curves](cer_curves.png)

![P@ bars](p_at_bars.png)
