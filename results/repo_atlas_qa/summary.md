### Per arm (judge total 5-100; entity recall 0-1; 95% bootstrap intervals over questions)

| arm | questions | judge score | entity recall | recall, non-test entities | missing answers | mean input tokens | cost per question | seconds per question | turns |
|---|---|---|---|---|---|---|---|---|---|
| closed | 144 | 75.6 (74.5-76.6) | 0.71 (0.68-0.75) | 0.71 (0.68-0.75) | 0 | 1,271 | $0.012 | 5 | 1.0 |
| map | 144 | 76.1 (75.2-76.9) | 0.71 (0.68-0.74) | 0.72 (0.68-0.75) | 0 | 28,402 | $0.016 | 5 | 1.0 |
| atlas | 144 | 75.4 (74.4-76.4) | 0.70 (0.67-0.74) | 0.72 (0.69-0.75) | 0 | 67,918 | $0.023 | 5 | 1.0 |
| text | 96 | 75.5 (74.2-76.7) | 0.71 (0.66-0.75) | 0.71 (0.67-0.76) | 0 | 97,062 | $0.028 | 4 | 1.0 |
| agent | 72 | 79.1 (77.8-80.3) | 0.79 (0.76-0.83) | 0.78 (0.74-0.82) | 0 | 25,081 | $0.045 | 11 | 4.3 |
| closed1 | 36 | 76.2 (73.8-78.2) | 0.74 (0.68-0.79) | 0.73 (0.67-0.79) | 0 | 819 | $0.022 | 9 | 1.0 |

Invalid (pages dropped) answers left out: 0.

### Paired differences (same questions; 95% bootstrap intervals)

| difference | questions | judge score | entity recall |
|---|---|---|---|
| atlas − map | 144 | -0.6 (-1.3 to +0.0) | -0.004 (-0.024 to +0.015) |
| atlas − agent | 72 | -3.3 (-4.4 to -2.3) | -0.115 (-0.152 to -0.079) |
| atlas − text | 96 | +1.1 (+0.4 to +1.8) | +0.016 (-0.010 to +0.043) |
| atlas − closed | 144 | -0.1 (-0.9 to +0.6) | -0.008 (-0.032 to +0.015) |
| map − closed | 144 | +0.5 (-0.2 to +1.2) | -0.003 (-0.027 to +0.021) |
| text − closed | 96 | -0.8 (-1.7 to +0.1) | -0.025 (-0.057 to +0.007) |
| agent − closed | 72 | +3.5 (+2.3 to +4.7) | +0.077 (+0.045 to +0.108) |
| agent − closed1 | 36 | +3.1 (+1.7 to +4.7) | +0.030 (-0.011 to +0.072) |
| closed1 − closed | 36 | +1.0 (-0.3 to +2.4) | +0.027 (-0.021 to +0.078) |
| atlas − closed1 | 36 | +0.5 (-0.6 to +1.6) | -0.065 (-0.120 to -0.014) |

### Per category (judge score; n per cell in brackets)

| category | closed | map | atlas | text | agent | closed1 | atlas − map | atlas − agent |
|---|---|---|---|---|---|---|---|---|
| what | 73.5 (37) | 74.4 (37) | 74.3 (37) | 72.6 (25) | 77.9 (18) | 71.9 (9) | -0.0 (-1.2 to +1.2) | -4.4 (-7.2 to -1.9) |
| why | 75.4 (35) | 75.7 (35) | 75.1 (35) | 75.6 (23) | 79.5 (18) | 78.1 (9) | -0.6 (-1.6 to +0.3) | -3.8 (-5.1 to -2.6) |
| where | 77.4 (36) | 77.7 (36) | 76.0 (36) | 76.5 (24) | 81.1 (18) | 79.1 (9) | -1.8 (-3.6 to -0.1) | -4.0 (-5.9 to -2.1) |
| how | 76.0 (36) | 76.5 (36) | 76.4 (36) | 77.3 (24) | 77.8 (18) | 75.8 (9) | -0.1 (-1.4 to +1.1) | -1.1 (-3.2 to +0.9) |
| cross-file (reference names >= 2 .py files) | 75.9 (100) | 76.4 (100) | 75.5 (100) | 75.4 (64) | 78.8 (53) | 75.2 (25) | -0.9 (-1.8 to -0.1) | -3.4 (-4.8 to -2.0) |
| single-file or none | 74.9 (44) | 75.4 (44) | 75.3 (44) | 75.6 (32) | 79.8 (19) | 78.5 (11) | -0.0 (-1.2 to +1.1) | -3.2 (-4.3 to -1.8) |

### Per repository (judge score)

| repository | closed | map | atlas | text | agent | closed1 |
|---|---|---|---|---|---|---|
| flask | 75.8 (48) | 77.1 (48) | 76.7 (48) | 75.7 (48) | 79.7 (24) | 75.5 (12) |
| pytest | 74.2 (48) | 74.1 (48) | 73.2 (48) |  | 77.5 (24) | 75.1 (12) |
| requests | 76.8 (48) | 76.9 (48) | 76.4 (48) | 75.2 (48) | 80.1 (24) | 78.1 (12) |

Score spread over all judged answers: min 46, quartiles 74 / 77 / 80, max 89.

### Spend (list price, every uncached call)

| part | spent |
|---|---|
| agent | $3.32 |
| atlas | $3.32 |
| closed | $1.73 |
| closed1 | $0.78 |
| harness checks | $0.68 |
| judge | $6.01 |
| map | $2.30 |
| text | $2.69 |
| **total** | **$20.83** |
