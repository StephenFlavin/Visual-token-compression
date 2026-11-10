# Claude QA v2: pooled over documents

Lossless (pre-registered) = upper 95% bound of paired (text - image) accuracy < 2%.

## effort high, image at 117.6 px²/char: 3 documents, 360 paired questions, 2.19x fewer input tokens

* text - image: +0.000 (95% CI -0.013 to +0.013); McNemar text-only 1, image-only 1, p = 1; **LOSSLESS** (margin 2%)
* complex only: text - image -0.008 (95% CI -0.046 to +0.023)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 99.7% (98%-100%) | 99.7% (98%-100%) |
| needle | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| number | 120 | 100.0% (97%-100%) | 99.2% (95%-100%) |
| complex | 120 | 99.2% (95%-100%) | 100.0% (97%-100%) |
| two_hop | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| three_hop | 24 | 100.0% (86%-100%) | 100.0% (86%-100%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 94.4% (74%-99%) | 100.0% (82%-100%) |

## effort high, image at 58.8 px²/char: 3 documents, 360 paired questions, 4.15x fewer input tokens

* text - image: +0.081 (95% CI +0.055 to +0.113); McNemar text-only 29, image-only 0, p = 3.73e-09; **loss not excluded** (margin 2%)
* complex only: text - image +0.200 (95% CI +0.133 to +0.280)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 99.7% (98%-100%) | 91.7% (88%-94%) |
| needle | 120 | 100.0% (97%-100%) | 97.5% (93%-99%) |
| number | 120 | 100.0% (97%-100%) | 98.3% (94%-100%) |
| complex | 120 | 99.2% (95%-100%) | 79.2% (71%-85%) |
| two_hop | 30 | 100.0% (89%-100%) | 90.0% (74%-97%) |
| three_hop | 24 | 100.0% (86%-100%) | 66.7% (47%-82%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 94.4% (74%-99%) |
| count | 18 | 94.4% (74%-99%) | 27.8% (12%-51%) |

## effort high, image at 88.2 px²/char: 3 documents, 360 paired questions, 2.87x fewer input tokens

* text - image: -0.003 (95% CI -0.016 to +0.008); McNemar text-only 0, image-only 1, p = 1; **LOSSLESS** (margin 2%)
* complex only: text - image -0.008 (95% CI -0.046 to +0.023)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 99.7% (98%-100%) | 100.0% (99%-100%) |
| needle | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| number | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| complex | 120 | 99.2% (95%-100%) | 100.0% (97%-100%) |
| two_hop | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| three_hop | 24 | 100.0% (86%-100%) | 100.0% (86%-100%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 94.4% (74%-99%) | 100.0% (82%-100%) |

## effort low, image at 117.6 px²/char: 3 documents, 360 paired questions, 2.19x fewer input tokens

* text - image: +0.003 (95% CI -0.008 to +0.016); McNemar text-only 1, image-only 0, p = 1; **LOSSLESS** (margin 2%)
* complex only: text - image +0.000 (95% CI -0.031 to +0.031)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 100.0% (99%-100%) | 99.7% (98%-100%) |
| needle | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| number | 120 | 100.0% (97%-100%) | 99.2% (95%-100%) |
| complex | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| two_hop | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| three_hop | 24 | 100.0% (86%-100%) | 100.0% (86%-100%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |

## effort low, image at 58.8 px²/char: 3 documents, 360 paired questions, 4.15x fewer input tokens

* text - image: +0.153 (95% CI +0.118 to +0.194); McNemar text-only 55, image-only 0, p = 5.55e-17; **loss not excluded** (margin 2%)
* complex only: text - image +0.375 (95% CI +0.288 to +0.464)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 100.0% (99%-100%) | 84.7% (81%-88%) |
| needle | 120 | 100.0% (97%-100%) | 98.3% (94%-100%) |
| number | 120 | 100.0% (97%-100%) | 93.3% (87%-97%) |
| complex | 120 | 100.0% (97%-100%) | 62.5% (54%-71%) |
| two_hop | 30 | 100.0% (89%-100%) | 53.3% (36%-70%) |
| three_hop | 24 | 100.0% (86%-100%) | 37.5% (21%-57%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 94.4% (74%-99%) |
| count | 18 | 100.0% (82%-100%) | 16.7% (6%-39%) |

## effort low, image at 88.2 px²/char: 3 documents, 360 paired questions, 2.87x fewer input tokens

* text - image: +0.011 (95% CI -0.001 to +0.028); McNemar text-only 4, image-only 0, p = 0.125; **loss not excluded** (margin 2%)
* complex only: text - image +0.017 (95% CI -0.017 to +0.059)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 100.0% (99%-100%) | 98.9% (97%-100%) |
| needle | 120 | 100.0% (97%-100%) | 100.0% (97%-100%) |
| number | 120 | 100.0% (97%-100%) | 98.3% (94%-100%) |
| complex | 120 | 100.0% (97%-100%) | 98.3% (94%-100%) |
| two_hop | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| three_hop | 24 | 100.0% (86%-100%) | 100.0% (86%-100%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 100.0% (82%-100%) | 88.9% (67%-97%) |

* no document, effort high: 6.9% of 360 questions
* no document, effort low: 5.0% of 360 questions
