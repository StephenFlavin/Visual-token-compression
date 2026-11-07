# Capacity curve: run0b_cpu_m28

bits/vtok = conservative-blind gap (min of swap, blank) in bits per vision token, extrapolated from the scored window to the page; recovered chars/vtok = chars/vtok x recovered fraction (gap / blind loss); 95% page-bootstrap CIs. CER = greedy transcription of the window (first pages only).

## Grid alignment (paired over pages; bits per vision token)

| encoding | text | pages | best shift | worst shift | best - worst (CI, re-selected) | perm p | cross-validated best - worst (CI) | (0,0) - mean of others (CI) |
|---|---|---|---|---|---|---|---|---|
| latin_p8 | natural | 16 | (28, 28) | (0, 0) | 1.600 (1.363-1.882) | 0.000 | 1.551 (1.286-1.871) | -1.011 (-1.228--0.800) |

mean bits/vtok by (dx, dy): (0,0) 11.146, (0,28) 12.470, (28,0) 11.255, (28,28) 12.746

