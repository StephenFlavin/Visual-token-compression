| density | calls | input tokens (mean) | small-word mean | − text (95% CI) | verdict | calls with every passive wrong | original 120 mean | − text (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| page118 | 15 | 114,188 | 97.2% | -2.8 (-5.2 to -1.2) | loss | 2 of 15 | 99.9% | -0.1 (-0.2 to +0.0) | no loss shown |
| page88 | 15 | 87,445 | 96.2% | -3.8 (-6.3 to -1.7) | loss | 3 of 15 | 99.3% | -0.7 (-0.9 to -0.4) | no loss shown |
| text | 6 | 248,971 | 100.0% | (reference) |  | 0 of 6 | 100.0% | (reference) |  |

Per-call small-word accuracy (document: calls in repeat order, Test A's first; passives right of 5 in brackets):

* page118: d0 97.5% (5), 97.5% (5), 97.5% (5), 97.5% (5), 97.5% (5); d1 87.5% (0), 100.0% (5), 100.0% (5), 100.0% (5), 100.0% (5); d2 87.5% (0), 100.0% (5), 97.5% (4), 100.0% (5), 97.5% (4)
* page88: d0 100.0% (5), 100.0% (5), 100.0% (5), 100.0% (5), 100.0% (5); d1 95.0% (4), 97.5% (5), 85.0% (0), 97.5% (5), 85.0% (0); d2 100.0% (5), 85.0% (0), 100.0% (5), 100.0% (5), 97.5% (5)
* text: d0 100.0% (5), 100.0% (5); d1 100.0% (5), 100.0% (5); d2 100.0% (5), 100.0% (5)

Per-call original-120 accuracy:

* page118: d0 100.0%, 100.0%, 100.0%, 100.0%, 100.0%; d1 100.0%, 100.0%, 99.2%, 100.0%, 100.0%; d2 100.0%, 100.0%, 100.0%, 100.0%, 100.0%
* page88: d0 99.2%, 100.0%, 100.0%, 100.0%, 100.0%; d1 99.2%, 100.0%, 98.3%, 97.5%, 98.3%; d2 99.2%, 99.2%, 100.0%, 99.2%, 100.0%
* text: d0 100.0%, 100.0%; d1 100.0%, 100.0%; d2 100.0%, 100.0%

| density | all_but | before_after | comparative | from_to | instead_of | negated_count | negation | passive | lure answers (of small-word) |
|---|---|---|---|---|---|---|---|---|---|
| page118 | 100% | 100% | 100% | 100% | 100% | 100% | 93% | 84% | 2.0% |
| page88 | 100% | 91% | 100% | 100% | 100% | 100% | 100% | 79% | 3.8% |
| text | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0.0% |

* page118 passives right, by place: split_page - (n=0), top_line - (n=0), bottom_line - (n=0), one_line 86% (n=50), two_plus_lines 80% (n=25); calls by passives right (0..5): {0: 2, 1: 0, 2: 0, 3: 0, 4: 2, 5: 11}
* page88 passives right, by place: split_page - (n=0), top_line - (n=0), bottom_line - (n=0), one_line 75% (n=60), two_plus_lines 93% (n=15); calls by passives right (0..5): {0: 3, 1: 0, 2: 0, 3: 0, 4: 1, 5: 11}
* invalid calls: none
