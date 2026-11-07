# Capacity curve: run0b_cpu_p6

bits/vtok = conservative-blind gap (min of swap, blank) in bits per vision token, extrapolated from the scored window to the page; recovered chars/vtok = chars/vtok x recovered fraction (gap / blind loss); 95% page-bootstrap CIs. CER = greedy transcription of the window (first pages only).

## Grid alignment (paired over pages; bits per vision token)

| encoding | text | pages | best shift | worst shift | best - worst (CI, re-selected) | perm p | cross-validated best - worst (CI) | aligned shift - mean of others (CI) |
|---|---|---|---|---|---|---|---|---|
| latin_p6 | natural | 16 | (14, 21) | (14, 14) | 2.237 (1.740-2.968) | 0.000 | 1.910 (0.886-2.831) | (28, 28): 0.080 (-0.282-0.454) |

mean bits/vtok by (dx, dy): (7,7) 12.408, (7,14) 12.192, (7,21) 14.036, (7,28) 13.371, (14,7) 12.915, (14,14) 11.923, (14,21) 14.160, (14,28) 13.306, (21,7) 12.757, (21,14) 12.081, (21,21) 14.056, (21,28) 13.361, (28,7) 12.705, (28,14) 12.089, (28,21) 13.899, (28,28) 13.097

