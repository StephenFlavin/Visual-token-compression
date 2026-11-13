# Sweep summary (phase 2)

Rows: 32 total, 32 usable, 0 no-fit (scored CER=1), 0 errors. Total recorded cost: $0.00.

P@x = px²/char at which the isotonic-smoothed mean CER first reaches x (log-x interpolation). ratio = P(scheme)/P(best Latin) (best of latin, latin_condensed at that threshold); CI = paired passage bootstrap. compr = text tokens / vision tokens at P (Claude: px/784 = 28x28 patches, text ≈ chars/4). '≤' on P = already below threshold at the smallest budget; on a ratio '≤'/'≥'/'?' = upper/lower bound/undetermined because of that censoring.

## Verdicts

* K3 (Phase 2: subst P@3% >= 30% below best Latin): KILL; best case subst300 on all: P@3%=8.2 vs latin 6.5 px^2/char, ratio 1.25 (95% CI 1.25-1.25)
* Quadrant (Phase 2 rule: killed if P@3% >= best Latin): KILLED; quad3/all ratio 3.65, quad4/all ratio 3.93
* P2 (subst300 / latin_condensed at P@3%): all: 0.99 (within +-20%)

## Condition: all

* best Latin @ 3%: **latin** (6.5 px²/char)
* best Latin @ 10%: **latin** (5.5 px²/char)

| scheme | P@3% [95% CI] | ratio@3% [CI] | vis tok/1k@3% | compr@3% | P@10% [95% CI] | ratio@10% [CI] | vis tok/1k@10% | compr@10% |
|---|---|---|---|---|---|---|---|---|
| latin | 6.5 [6.5, 6.5] | 1.00 [1.00, 1.00] | 8.3 | 29.96 | ≤ 5.5 [5.5, 5.5] | 1.00 [1.00, 1.00] | 7.1 | 35.41 |
| latin_condensed | 8.3 [8.3, 8.3] | 1.27 [1.27, 1.27] | 10.6 | 23.57 | ≤ 7.1 [7.1, 7.1] | ? 1.29 [1.29, 1.29] | 9.1 | 27.47 |
| subst300 | 8.2 [8.2, 8.2] | 1.25 [1.25, 1.25] | 10.5 | 23.92 | ≤ 7.2 [7.2, 7.2] | ? 1.30 [1.30, 1.30] | 9.2 | 27.15 |
| subst1000 | 9.7 [9.7, 9.7] | 1.48 [1.48, 1.48] | 12.3 | 20.30 | ≤ 7.8 [7.8, 7.8] | ? 1.41 [1.41, 1.41] | 10.0 | 25.08 |
| quad3 | 23.9 [23.9, 23.9] | 3.65 [3.65, 3.65] | 30.4 | 8.21 | ≤ 16.7 [16.7, 16.7] | ? 3.01 [3.01, 3.01] | 21.3 | 11.76 |
| quad4 | 25.7 [25.7, 25.7] | 3.93 [3.93, 3.93] | 32.8 | 7.62 | ≤ 14.9 [14.9, 14.9] | ? 2.69 [2.69, 2.69] | 19.0 | 13.16 |

Mean CER by budget:

| scheme | 5.5 | 7.1 | 7.2 | 7.5 | 7.8 | 8.9 | 9 | 9.8 | 11.1 | 11.3 | 12.5 | 12.8 | 14.9 | 15.3 | 15.4 | 16.1 | 16.7 | 17.5 | 20 | 21.8 | 22.1 | 22.8 | 23.3 | 26 | 28.5 | 30.1 | 31.3 | 33.5 | 35.7 | 37.5 | 39.4 | 44.6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latin | 0.045 |  |  | 0.017 |  |  |  | 0.006 |  |  | 0.002 |  |  |  | 0.001 |  |  |  |  |  | 0.003 |  |  |  |  | 0.001 |  |  |  |  | 0.001 |  |
| latin_condensed |  | 0.041 |  |  |  |  | 0.024 |  | 0.009 |  |  |  |  |  |  | 0.014 |  |  |  | 0.005 |  |  |  |  | 0.002 |  |  |  |  |  |  | 0.001 |
| subst300 |  |  | 0.041 |  |  | 0.023 |  |  |  |  |  | 0.007 |  |  |  |  |  | 0.002 |  |  |  | 0.011 |  |  |  |  |  |  | 0.004 |  |  |  |
| subst1000 |  |  |  |  | 0.058 |  |  |  |  | 0.010 |  |  |  | 0.006 |  |  |  |  | 0.004 |  |  |  |  |  |  |  | 0.008 |  |  |  |  |  |
| quad3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.066 |  |  |  |  |  |  | 0.021 |  |  |  |  |  | 0.009 |  |  |
| quad4 |  |  |  |  |  |  |  |  |  |  |  |  | 0.095 |  |  |  |  |  |  |  |  |  | 0.035 |  |  |  |  | 0.018 |  |  |  |  |

![CER curves](cer_curves.png)

![P@ bars](p_at_bars.png)
