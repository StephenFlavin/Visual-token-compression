# E0 pixel statistics (64 val lines, corpus wikitext103)

dist = hist L1 + |d mean ink|/ref + |d sharpness|/ref vs the PIL source of the same mode (eval vs Phase-2 test images; train vs Phase-2 train render + augment_batch).

| em | h | source | mean ink | p99 | sharp | sharp/ink | bg std | dist |
|---|---|---|---|---|---|---|---|---|
| 5 | 6 | pil_eval | 0.1000 | 0.643 | 0.1851 | 1.851 | 0.0000 | 0.000 |
| 5 | 6 | pil_train | 0.1109 | 0.570 | 0.1844 | 1.662 | 0.0304 | 0.000 |
| 5 | 6 | clean_eval | 0.1007 | 0.643 | 0.1891 | 1.879 | 0.0000 | 0.111 |
| 5 | 6 | clean_train | 0.1007 | 0.643 | 0.1889 | 1.876 | 0.0000 | 0.959 |
| 5 | 6 | default_eval | 0.1102 | 0.561 | 0.1901 | 1.725 | 0.0247 | 0.899 |
| 5 | 6 | default_train | 0.1093 | 0.565 | 0.1780 | 1.629 | 0.0297 | 0.102 |
| 5 | 6 | harsh_eval | 0.1216 | 0.329 | 0.1066 | 0.877 | 0.0435 | 2.031 |
| 5 | 6 | harsh_train | 0.1184 | 0.396 | 0.1238 | 1.045 | 0.0514 | 0.907 |
| 5 | 6 | harsh+heldout_eval | 0.1220 | 0.329 | 0.0910 | 0.746 | 0.0413 | 2.168 |
| 8 | 10 | pil_eval | 0.1141 | 0.800 | 0.2185 | 1.915 | 0.0000 | 0.000 |
| 8 | 10 | pil_train | 0.1231 | 0.734 | 0.2180 | 1.772 | 0.0299 | 0.000 |
| 8 | 10 | clean_eval | 0.1143 | 0.800 | 0.2261 | 1.978 | 0.0000 | 0.105 |
| 8 | 10 | clean_train | 0.1143 | 0.800 | 0.2264 | 1.980 | 0.0000 | 0.994 |
| 8 | 10 | default_eval | 0.1216 | 0.694 | 0.2231 | 1.834 | 0.0248 | 0.930 |
| 8 | 10 | default_train | 0.1210 | 0.710 | 0.2093 | 1.730 | 0.0297 | 0.101 |
| 8 | 10 | harsh_eval | 0.1311 | 0.384 | 0.1038 | 0.792 | 0.0437 | 2.123 |
| 8 | 10 | harsh_train | 0.1276 | 0.451 | 0.1225 | 0.960 | 0.0520 | 1.073 |
| 8 | 10 | harsh+heldout_eval | 0.1314 | 0.376 | 0.0883 | 0.673 | 0.0413 | 2.245 |

Pixel-aligned relative MAD (phase 0 both sides, eval degradation); yardstick = PIL vs PIL shifted by 1/4 px.

| em | preset | prefix (12 ch) | full line | PIL 1/4-px prefix | PIL 1/4-px full | clip max |
|---|---|---|---|---|---|---|
| 5 | clean | 0.121 | 0.326 | 0.333 | 0.323 | 0.050 |
| 5 | default | 0.401 | 0.538 | 0.333 | 0.323 | 0.050 |
| 5 | harsh | 1.036 | 1.031 | 0.333 | 0.323 | 0.050 |
| 5 | harsh+heldout | 1.033 | 1.031 | 0.333 | 0.323 | 0.050 |
| 8 | clean | 0.194 | 0.435 | 0.333 | 0.327 | 0.050 |
| 8 | default | 0.435 | 0.613 | 0.333 | 0.327 | 0.050 |
| 8 | harsh | 1.071 | 1.077 | 0.333 | 0.327 | 0.050 |
| 8 | harsh+heldout | 1.076 | 1.079 | 0.333 | 0.327 | 0.050 |

| em | px2/char bank (h x W / chars) | 1.2 em x W / chars |
|---|---|---|
| 5 | 15.25 | 15.25 |
| 8 | 40.89 | 39.26 |

Mean dist per preset (lower = closer to PIL):

* clean_eval: 0.108
* default_eval: 0.915
* harsh_eval: 2.077
* harsh+heldout_eval: 2.207
* clean_train: 0.976
* default_train: 0.102
* harsh_train: 0.990
