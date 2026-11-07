# Capacity curve: run0b_cpu

bits/vtok = conservative-blind gap (min of swap, blank) in bits per vision token, extrapolated from the scored window to the page; recovered chars/vtok = chars/vtok x recovered fraction (gap / blind loss); 95% page-bootstrap CIs. CER = greedy transcription of the window (first pages only).

## Grid alignment (paired over pages; bits per vision token)

| encoding | text | pages | best shift | worst shift | best - worst (CI, re-selected) | perm p | cross-validated best - worst (CI) | (0,0) - mean of others (CI) |
|---|---|---|---|---|---|---|---|---|
| latin_p12 | natural | 16 | (14, 7) | (7, 0) | 0.158 (0.140-0.328) | 0.377 | 0.037 (-0.063-0.313) | -0.046 (-0.127-0.031) |

mean bits/vtok by (dx, dy): (0,0) 8.063, (0,7) 8.052, (0,14) 8.111, (0,21) 8.100, (7,0) 8.017, (7,7) 8.075, (7,14) 8.124, (7,21) 8.065, (14,0) 8.089, (14,7) 8.175, (14,14) 8.157, (14,21) 8.162, (21,0) 8.060, (21,7) 8.138, (21,14) 8.144, (21,21) 8.168

| latin_p8 | natural | 16 | (21, 21) | (14, 0) | 1.517 (1.248-2.000) | 0.000 | 1.294 (0.927-1.919) | -1.081 (-1.329--0.838) |

mean bits/vtok by (dx, dy): (0,0) 11.024, (0,7) 12.308, (0,14) 12.241, (0,21) 12.358, (7,0) 11.196, (7,7) 12.441, (7,14) 12.325, (7,21) 12.296, (14,0) 10.999, (14,7) 12.397, (14,14) 12.234, (14,21) 12.346, (21,0) 11.219, (21,7) 12.440, (21,14) 12.268, (21,21) 12.515

| lg_pb_h5 | natural | 16 | (21, 7) | (21, 0) | 0.124 (0.091-0.207) | 0.005 | 0.064 (0.003-0.199) | -0.011 (-0.060-0.043) |

mean bits/vtok by (dx, dy): (0,0) 0.435, (0,7) 0.471, (0,14) 0.458, (0,21) 0.423, (7,0) 0.489, (7,7) 0.483, (7,14) 0.448, (7,21) 0.413, (14,0) 0.418, (14,7) 0.423, (14,14) 0.482, (14,21) 0.427, (21,0) 0.374, (21,7) 0.498, (21,14) 0.461, (21,21) 0.418

