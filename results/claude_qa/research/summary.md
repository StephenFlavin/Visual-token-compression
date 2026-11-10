# Claude QA, deep-research variant: results

3 corpora x 120 questions; arms paired with `text` question by question (Newcombe 95% interval for arm - text, exact McNemar). Lossless = lower bound of (arm - text) above -2 points.

| arm | input tokens (mean) | fewer tokens | list-price $ per corpus (incl. notes) | all | lookups | synthesis | arm - text, all (95% CI) | McNemar p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| none | 5,060 | 53.88x | 0.07 | 6.9% | 12.2% | 1.7% | -92.5 (-94.7 to -89.1) | 1.1e-100 | loss |
| notes | 46,349 | 5.88x | 2.70 | 70.3% | 70.6% | 70.0% | -29.2 (-34.1 to -24.5) | 1.3e-30 | loss |
| notes_pages | 163,656 | 1.67x | 3.27 | 99.4% | 99.4% | 99.4% | -0.0 (-1.3 to +1.3) | 1 | lossless |
| pages | 122,404 | 2.23x | 0.77 | 99.7% | 99.4% | 100.0% | +0.3 (-0.8 to +1.6) | 1 | lossless |
| text | 272,627 | 1.00x | 1.48 | 99.4% | 99.4% | 99.4% | - | - | reference |

Per question type (accuracy, pooled; n per type in brackets):

| arm | needle (90) | number (90) | citation (36) | conflict (36) | chain (36) | source_count (36) | total (36) |
|---|---|---|---|---|---|---|---|
| none | 0% | 24% | 6% | 0% | 0% | 3% | 0% |
| notes | 89% | 52% | 97% | 75% | 75% | 28% | 75% |
| notes_pages | 100% | 99% | 100% | 100% | 100% | 97% | 100% |
| pages | 100% | 99% | 100% | 100% | 100% | 100% | 100% |
| text | 100% | 99% | 100% | 100% | 100% | 97% | 100% |

Arm - text by group (points, 95% CI):

| arm | lookups | synthesis |
|---|---|---|
| none | -87.2 (-91.2 to -81.2) | -97.8 (-99.0 to -93.8) |
| notes | -28.9 (-35.9 to -22.5) | -29.4 (-36.5 to -22.6) |
| notes_pages | -0.0 (-2.1 to +2.1) | -0.0 (-2.6 to +2.6) |
| pages | -0.0 (-2.1 to +2.1) | +0.6 (-1.6 to +3.1) |
