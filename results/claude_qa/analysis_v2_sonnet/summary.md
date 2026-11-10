# Claude QA v2: pooled over documents

Lossless (pre-registered) = upper 95% bound of paired (text - image) accuracy < 2%.

## effort high, image at 117.6 px²/char: 3 documents, 360 paired questions, 2.19x fewer input tokens

* text - image: +0.003 (95% CI -0.011 to +0.017); McNemar text-only 2, image-only 1, p = 1; **LOSSLESS** (margin 2%)
* complex only: text - image +0.008 (95% CI -0.031 to +0.051)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 99.7% (98%-100%) | 99.4% (98%-100%) |
| needle | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| number | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| complex | 120 | 99.2% (95%-100%) | 98.3% (94%-100%) |
| two_hop | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| three_hop | 24 | 100.0% (86%-100%) | 100.0% (86%-100%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 94.4% (74%-99%) | 88.9% (67%-97%) |

## effort high, image at 58.8 px²/char: 3 documents, 360 paired questions, 4.15x fewer input tokens

* text - image: +0.139 (95% CI +0.106 to +0.178); McNemar text-only 50, image-only 0, p = 1.78e-15; **loss not excluded** (margin 2%)
* complex only: text - image +0.350 (95% CI +0.266 to +0.438)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 99.7% (98%-100%) | 85.8% (82%-89%) |
| needle | 120 | 100.0% (97%-100%) | 95.0% (90%-98%) |
| number | 120 | 100.0% (97%-100%) | 98.3% (94%-100%) |
| complex | 120 | 99.2% (95%-100%) | 64.2% (55%-72%) |
| two_hop | 30 | 100.0% (89%-100%) | 66.7% (49%-81%) |
| three_hop | 24 | 100.0% (86%-100%) | 41.7% (24%-61%) |
| sum | 30 | 100.0% (89%-100%) | 86.7% (70%-95%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 94.4% (74%-99%) | 16.7% (6%-39%) |

## effort high, image at 88.2 px²/char: 3 documents, 360 paired questions, 2.87x fewer input tokens

* text - image: +0.017 (95% CI +0.000 to +0.037); McNemar text-only 7, image-only 1, p = 0.0703; **loss not excluded** (margin 2%)
* complex only: text - image +0.042 (95% CI -0.005 to +0.097)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 99.7% (98%-100%) | 98.1% (96%-99%) |
| needle | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| number | 120 | 100.0% (97%-100%) | 99.2% (95%-100%) |
| complex | 120 | 99.2% (95%-100%) | 95.0% (90%-98%) |
| two_hop | 30 | 100.0% (89%-100%) | 96.7% (83%-99%) |
| three_hop | 24 | 100.0% (86%-100%) | 100.0% (86%-100%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 94.4% (74%-99%) | 72.2% (49%-88%) |

## effort low, image at 117.6 px²/char: 3 documents, 360 paired questions, 2.19x fewer input tokens

* text - image: +0.003 (95% CI -0.016 to +0.022); McNemar text-only 5, image-only 4, p = 1; **loss not excluded** (margin 2%)
* complex only: text - image +0.008 (95% CI -0.042 to +0.061)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 98.9% (97%-100%) | 98.6% (97%-99%) |
| needle | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| number | 120 | 99.2% (95%-100%) | 99.2% (95%-100%) |
| complex | 120 | 97.5% (93%-99%) | 96.7% (92%-99%) |
| two_hop | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| three_hop | 24 | 95.8% (80%-99%) | 100.0% (86%-100%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 88.9% (67%-97%) | 77.8% (55%-91%) |

## effort low, image at 58.8 px²/char: 3 documents, 360 paired questions, 4.15x fewer input tokens

* text - image: +0.217 (95% CI +0.175 to +0.262); McNemar text-only 79, image-only 1, p = 1.34e-22; **loss not excluded** (margin 2%)
* complex only: text - image +0.467 (95% CI +0.374 to +0.554)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 98.9% (97%-100%) | 77.2% (73%-81%) |
| needle | 120 | 100.0% (97%-100%) | 94.2% (88%-97%) |
| number | 120 | 99.2% (95%-100%) | 86.7% (79%-92%) |
| complex | 120 | 97.5% (93%-99%) | 50.8% (42%-60%) |
| two_hop | 30 | 100.0% (89%-100%) | 30.0% (17%-48%) |
| three_hop | 24 | 95.8% (80%-99%) | 20.8% (9%-40%) |
| sum | 30 | 100.0% (89%-100%) | 86.7% (70%-95%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 88.9% (67%-97%) | 16.7% (6%-39%) |

## effort low, image at 88.2 px²/char: 3 documents, 360 paired questions, 2.87x fewer input tokens

* text - image: +0.031 (95% CI +0.008 to +0.057); McNemar text-only 14, image-only 3, p = 0.0127; **loss not excluded** (margin 2%)
* complex only: text - image +0.075 (95% CI +0.017 to +0.142)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 98.9% (97%-100%) | 95.8% (93%-97%) |
| needle | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| number | 120 | 99.2% (95%-100%) | 97.5% (93%-99%) |
| complex | 120 | 97.5% (93%-99%) | 90.0% (83%-94%) |
| two_hop | 30 | 100.0% (89%-100%) | 93.3% (79%-98%) |
| three_hop | 24 | 95.8% (80%-99%) | 95.8% (80%-99%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 88.9% (67%-97%) | 50.0% (29%-71%) |

* no document, effort high: 1.9% of 360 questions
* no document, effort low: 1.9% of 360 questions
