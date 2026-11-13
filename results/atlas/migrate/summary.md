*14 projects with all three arms; spend $24.01 at list price (cap $30.0).*

Large end (pilot): plain 1/2 all pass, 97.8% tests, atlas 1/2 all pass, 97.8% tests, atlas_only 1/2 all pass, 97.8% tests.
Smaller end: plain 11/12 all pass, 91.7% tests, atlas 11/12 all pass, 91.7% tests, atlas_only 11/12 all pass, 91.7% tests.
First 10 projects: plain 8/10 all pass, 89.6% tests, atlas 8/10 all pass, 89.6% tests, atlas_only 8/10 all pass, 89.6% tests.

| Arm | Projects | All tests pass | Mean share passing | Builds | Median turns | Mean cost | Median seconds | Median output tokens | unsafe (total) |
|---|---|---|---|---|---|---|---|---|---|
| plain | 14 | 12 | 92.5% | 14 | 6.5 | $0.54 | 71.5 | 8209.5 | 0 |
| atlas | 14 | 12 | 92.5% | 14 | 6.0 | $0.57 | 62.5 | 7138.5 | 0 |
| atlas_only | 14 | 12 | 92.5% | 14 | 6.5 | $0.60 | 79.5 | 9524.5 | 0 |

| Project | C lines | Tests | plain: pass / turns / $ / s | atlas: pass / turns / $ / s | atlas_only: pass / turns / $ / s |
|---|---|---|---|---|---|
| Genetic_neural_network_for_simple_control | 3889 | 23 | 22/23 (SUSPECT) / 26 / 1.72 / 488 | 22/23 (SUSPECT) / 28 / 2.58 / 991 | 22/23 (SUSPECT) / 31 / 2.14 / 746 |
| recordManager | 3759 | 3 | 3/3 ALL / 17 / 1.79 / 428 | 3/3 ALL (SUSPECT) / 19 / 1.97 / 468 | 3/3 ALL (SUSPECT) / 18 / 2.51 / 636 |
| SimpleXML | 806 | 3 | 3/3 ALL / 7 / 0.29 / 65 | 3/3 ALL / 5 / 0.22 / 51 | 3/3 ALL / 6 / 0.29 / 78 |
| utf8 | 806 | 25 | 25/25 ALL / 6 / 0.27 / 58 | 25/25 ALL (SUSPECT) / 7 / 0.26 / 53 | 25/25 ALL / 4 / 0.22 / 49 |
| clhash | 824 | 5 | 5/5 ALL / 6 / 0.25 / 56 | 5/5 ALL / 5 / 0.26 / 55 | 5/5 ALL / 6 / 0.40 / 103 |
| Simple_Sparsehash | 845 | 12 | 12/12 ALL / 6 / 0.30 / 76 | 12/12 ALL / 8 / 0.28 / 75 | 12/12 ALL / 5 / 0.23 / 63 |
| proj_2DPartInt | 855 | 6 | 6/6 ALL / 7 / 0.35 / 67 | 6/6 ALL / 6 / 0.38 / 70 | 6/6 ALL / 7 / 0.41 / 97 |
| Phills_DHT | 875 | 29 | 29/29 ALL / 4 / 0.11 / 23 | 29/29 ALL / 5 / 0.12 / 23 | 29/29 ALL / 5 / 0.11 / 23 |
| lib2bit | 911 | 1 | 0/1 / 9 / 0.56 / 143 | 0/1 / 6 / 0.46 / 111 | 0/1 / 7 / 0.58 / 164 |
| rubiksolver | 923 | 6 | 6/6 ALL / 7 / 0.29 / 54 | 6/6 ALL / 5 / 0.26 / 48 | 6/6 ALL / 7 / 0.31 / 74 |
| c_string | 929 | 15 | 15/15 ALL / 4 / 0.21 / 50 | 15/15 ALL / 4 / 0.19 / 39 | 15/15 ALL / 4 / 0.20 / 48 |
| simple_lang | 944 | 3 | 3/3 ALL / 6 / 0.45 / 92 | 3/3 ALL / 5 / 0.28 / 55 | 3/3 ALL / 7 / 0.33 / 81 |
| clog | 1060 | 1 | 1/1 ALL (SUSPECT) / 8 / 0.59 / 154 | 1/1 ALL / 7 / 0.31 / 79 | 1/1 ALL / 4 / 0.24 / 60 |
| roaring_bitmap | 1083 | 15 | 15/15 ALL (tests restored) (SUSPECT) / 6 / 0.42 / 111 | 15/15 ALL (tests restored) / 9 / 0.43 / 109 | 15/15 ALL (tests restored) (SUSPECT) / 8 / 0.43 / 122 |

Paired tests (exact; McNemar on all-tests-pass, sign tests on the rest):

* **atlas vs plain:** all-pass atlas only 0, plain only 0, p = 1.000; share: median diff +0 (0+ / 0- / 14=, p = 1.000); turns: median diff -0.5 (6+ / 7- / 1=, p = 1.000); cost_usd: median diff -0.0101 (6+ / 8- / 0=, p = 0.791); agent_s: median diff -3.5 (3+ / 10- / 1=, p = 0.092); output_tokens: median diff -517 (5+ / 9- / 0=, p = 0.424)
* **atlas_only vs plain:** all-pass atlas_only only 0, plain only 0, p = 1.000; share: median diff +0 (0+ / 0- / 14=, p = 1.000); turns: median diff +0 (5+ / 5- / 4=, p = 1.000); cost_usd: median diff +0.00416 (7+ / 7- / 0=, p = 1.000); agent_s: median diff +12 (8+ / 5- / 1=, p = 0.581); output_tokens: median diff +1.44e+03 (8+ / 6- / 0=, p = 0.791)
* **atlas_only vs atlas:** all-pass atlas_only only 0, atlas only 0, p = 1.000; share: median diff +0 (0+ / 0- / 14=, p = 1.000); turns: median diff +0.5 (7+ / 5- / 2=, p = 0.774); cost_usd: median diff +0.0214 (8+ / 6- / 0=, p = 0.791); agent_s: median diff +19.5 (9+ / 4- / 1=, p = 0.267); output_tokens: median diff +3.08e+03 (10+ / 4- / 0=, p = 0.180)
