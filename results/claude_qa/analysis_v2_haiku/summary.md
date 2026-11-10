# Claude QA v2: pooled over documents

Lossless (pre-registered) = upper 95% bound of paired (text - image) accuracy < 2%.

## effort high, image at 117.6 px²/char: 3 documents, 360 paired questions, 2.19x fewer input tokens

* text - image: +0.058 (95% CI +0.030 to +0.090); McNemar text-only 25, image-only 4, p = 0.000104; **loss not excluded** (margin 2%)
* complex only: text - image +0.175 (95% CI +0.100 to +0.256)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 98.1% (96%-99%) | 92.2% (89%-95%) |
| needle | 120 | 100.0% (97%-100%) | 98.3% (94%-100%) |
| number | 120 | 98.3% (94%-100%) | 100.0% (97%-100%) |
| complex | 120 | 95.8% (91%-98%) | 78.3% (70%-85%) |
| two_hop | 30 | 96.7% (83%-99%) | 90.0% (74%-97%) |
| three_hop | 24 | 100.0% (86%-100%) | 66.7% (47%-82%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 77.8% (55%-91%) | 16.7% (6%-39%) |

## effort high, image at 58.8 px²/char: 3 documents, 360 paired questions, 4.15x fewer input tokens

* text - image: +0.525 (95% CI +0.471 to +0.576); McNemar text-only 190, image-only 1, p = 1.22e-55; **loss not excluded** (margin 2%)
* complex only: text - image +0.808 (95% CI +0.721 to +0.864)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 98.1% (96%-99%) | 45.6% (40%-51%) |
| needle | 120 | 100.0% (97%-100%) | 55.8% (47%-64%) |
| number | 120 | 98.3% (94%-100%) | 65.8% (57%-74%) |
| complex | 120 | 95.8% (91%-98%) | 15.0% (10%-22%) |
| two_hop | 30 | 96.7% (83%-99%) | 6.7% (2%-21%) |
| three_hop | 24 | 100.0% (86%-100%) | 0.0% (0%-14%) |
| sum | 30 | 100.0% (89%-100%) | 30.0% (17%-48%) |
| max | 18 | 100.0% (82%-100%) | 38.9% (20%-61%) |
| count | 18 | 77.8% (55%-91%) | 0.0% (0%-18%) |

## effort high, image at 88.2 px²/char: 3 documents, 360 paired questions, 2.87x fewer input tokens

* text - image: +0.111 (95% CI +0.079 to +0.149); McNemar text-only 42, image-only 2, p = 1.13e-10; **loss not excluded** (margin 2%)
* complex only: text - image +0.325 (95% CI +0.242 to +0.411)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 98.1% (96%-99%) | 86.9% (83%-90%) |
| needle | 120 | 100.0% (97%-100%) | 98.3% (94%-100%) |
| number | 120 | 98.3% (94%-100%) | 99.2% (95%-100%) |
| complex | 120 | 95.8% (91%-98%) | 63.3% (54%-71%) |
| two_hop | 30 | 96.7% (83%-99%) | 63.3% (46%-78%) |
| three_hop | 24 | 100.0% (86%-100%) | 37.5% (21%-57%) |
| sum | 30 | 100.0% (89%-100%) | 100.0% (89%-100%) |
| max | 18 | 100.0% (82%-100%) | 100.0% (82%-100%) |
| count | 18 | 77.8% (55%-91%) | 0.0% (0%-18%) |

## effort low, image at 117.6 px²/char: 3 documents, 360 paired questions, 2.19x fewer input tokens

* text - image: -0.492 (95% CI -0.541 to -0.437); McNemar text-only 1, image-only 178, p = 4.7e-52; **INVALID: a call did not engage** (margin 2%)
* not engaged: qav2h_d0_low text: 80 unanswered, says "can't answer these"
* complex only: text - image -0.700 (95% CI -0.775 to -0.607)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 39.2% (34%-44%) | 88.3% (85%-91%) |
| needle | 120 | 95.0% (90%-98%) | 98.3% (94%-100%) |
| number | 120 | 22.5% (16%-31%) | 96.7% (92%-99%) |
| complex | 120 | 0.0% (-0%-3%) | 70.0% (61%-77%) |
| two_hop | 30 | 0.0% (-0%-11%) | 86.7% (70%-95%) |
| three_hop | 24 | 0.0% (0%-14%) | 37.5% (21%-57%) |
| sum | 30 | 0.0% (-0%-11%) | 96.7% (83%-99%) |
| max | 18 | 0.0% (0%-18%) | 94.4% (74%-99%) |
| count | 18 | 0.0% (0%-18%) | 16.7% (6%-39%) |

## effort low, image at 58.8 px²/char: 3 documents, 360 paired questions, 4.15x fewer input tokens

* text - image: +0.247 (95% CI +0.192 to +0.301); McNemar text-only 105, image-only 16, p = 3.16e-17; **INVALID: a call did not engage** (margin 2%)
* not engaged: qav2h_d0_low text: 80 unanswered, says "can't answer these"; qav2h_d1_low image: 0 unanswered, says "can't answer these"
* complex only: text - image -0.025 (95% CI -0.071 to +0.010)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 39.2% (34%-44%) | 14.4% (11%-18%) |
| needle | 120 | 95.0% (90%-98%) | 25.0% (18%-33%) |
| number | 120 | 22.5% (16%-31%) | 15.8% (10%-23%) |
| complex | 120 | 0.0% (-0%-3%) | 2.5% (1%-7%) |
| two_hop | 30 | 0.0% (-0%-11%) | 0.0% (-0%-11%) |
| three_hop | 24 | 0.0% (0%-14%) | 0.0% (0%-14%) |
| sum | 30 | 0.0% (-0%-11%) | 0.0% (-0%-11%) |
| max | 18 | 0.0% (0%-18%) | 16.7% (6%-39%) |
| count | 18 | 0.0% (0%-18%) | 0.0% (0%-18%) |

## effort low, image at 88.2 px²/char: 3 documents, 360 paired questions, 2.87x fewer input tokens

* text - image: -0.417 (95% CI -0.467 to -0.362); McNemar text-only 3, image-only 153, p = 1.39e-41; **INVALID: a call did not engage** (margin 2%)
* not engaged: qav2h_d0_low text: 80 unanswered, says "can't answer these"
* complex only: text - image -0.483 (95% CI -0.572 to -0.390)

| kind | n | text | image |
|---|---|---|---|
| all | 360 | 39.2% (34%-44%) | 80.8% (76%-85%) |
| needle | 120 | 95.0% (90%-98%) | 98.3% (94%-100%) |
| number | 120 | 22.5% (16%-31%) | 95.8% (91%-98%) |
| complex | 120 | 0.0% (-0%-3%) | 48.3% (40%-57%) |
| two_hop | 30 | 0.0% (-0%-11%) | 26.7% (14%-44%) |
| three_hop | 24 | 0.0% (0%-14%) | 8.3% (2%-26%) |
| sum | 30 | 0.0% (-0%-11%) | 100.0% (89%-100%) |
| max | 18 | 0.0% (0%-18%) | 100.0% (82%-100%) |
| count | 18 | 0.0% (0%-18%) | 0.0% (0%-18%) |

* no document, effort high: 0.6% of 360 questions
* no document, effort low: 0.3% of 360 questions
