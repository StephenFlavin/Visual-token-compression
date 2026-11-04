# Quick test Q1: linear readout of frozen Qwen2.5-VL ViT features (2026-11-04)

Closed-form ridge from each feature to the clean pixels it covers (14-px patches; merger: 28-px tokens), 64 train pages (lambda picked on pages 48-63), 16 held-out test pages (WikiText test/val), 448-px pages. bN = after blocks 0..N-1; final = after block 31, before the merger. A linear readout is a lower bound on what a feature holds.

## R^2 of the test-page pixels

| encoding | chars/Qwen token | raw | embed | b8 | b16 | b24 | b28 | final | merger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb3_lg_pb_h5 | 321.6 | 1.000 | 1.000 | 0.691 | 0.606 | 0.425 | 0.343 | 0.230 | 0.019 |
| rgb3_lg_free_h4 | 405.5 | 1.000 | 1.000 | 0.640 | 0.458 | 0.230 | 0.161 | 0.091 | 0.008 |
| latin_p8 | 28.1 | 1.000 | 1.000 | 0.970 | 0.836 | 0.456 | 0.281 | 0.239 | 0.251 |
| lg_pb_h5 | 107.3 | 1.000 | 1.000 | 0.983 | 0.946 | 0.763 | 0.622 | 0.430 | 0.033 |
| rgb3_p8 | 72.8 | 1.000 | 1.000 | 0.799 | 0.665 | 0.449 | 0.343 | 0.292 | 0.279 |
| latin_p16 | 6.8 | 1.000 | 1.000 | 0.969 | 0.841 | 0.420 | 0.220 | 0.174 | 0.157 |

## Ink MCC (Matthews correlation of the ink masks; 0 = chance, 1 = exact; rgb3 codes R/G/B)

| encoding | raw | embed | b8 | b16 | b24 | b28 | final | merger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb3_lg_pb_h5 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.89/0.84/0.84 | 0.82/0.78/0.76 | 0.66/0.64/0.61 | 0.58/0.57/0.54 | 0.48/0.48/0.45 | 0.29/0.29/0.29 |
| rgb3_lg_free_h4 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.61/0.55/0.56 | 0.46/0.43/0.41 | 0.23/0.23/0.20 | 0.17/0.17/0.14 | 0.09/0.09/0.07 | 0.00/0.00/-0.00 |
| latin_p8 | 0.999 | 0.997 | 0.888 | 0.673 | 0.176 | 0.022 | 0.023 | 0.034 |
| lg_pb_h5 | 1.000 | 1.000 | 1.000 | 0.999 | 0.914 | 0.806 | 0.650 | 0.295 |
| rgb3_p8 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.65/0.64/0.61 | 0.48/0.50/0.45 | 0.19/0.22/0.17 | 0.07/0.09/0.06 | 0.04/0.05/0.04 | 0.09/0.10/0.09 |
| latin_p16 | 1.000 | 1.000 | 0.935 | 0.851 | 0.387 | 0.077 | 0.041 | 0.058 |

## Ink F1 (rgb3 codes R/G/B); chance = 2p/(1+p) for ink fraction p (an all-ink guess)

| encoding | F1 chance | raw | embed | b8 | b16 | b24 | b28 | final | merger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb3_lg_pb_h5 | 0.634 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.94/0.91/0.91 | 0.90/0.88/0.87 | 0.82/0.80/0.79 | 0.77/0.77/0.75 | 0.72/0.71/0.70 | 0.62/0.62/0.62 |
| rgb3_lg_free_h4 | 0.326 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.68/0.62/0.63 | 0.53/0.50/0.47 | 0.22/0.22/0.18 | 0.15/0.15/0.11 | 0.05/0.05/0.04 | 0.00/0.00/0.00 |
| latin_p8 | 0.281 | 0.999 | 0.997 | 0.905 | 0.695 | 0.112 | 0.004 | 0.006 | 0.016 |
| lg_pb_h5 | 0.636 | 1.000 | 1.000 | 1.000 | 0.999 | 0.954 | 0.896 | 0.811 | 0.617 |
| rgb3_p8 | 0.258 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.66/0.66/0.62 | 0.46/0.48/0.42 | 0.11/0.14/0.09 | 0.02/0.03/0.02 | 0.01/0.01/0.01 | 0.04/0.05/0.04 |
| latin_p16 | 0.251 | 1.000 | 1.000 | 0.944 | 0.870 | 0.356 | 0.027 | 0.009 | 0.024 |

## Calibration: DeepEncoder projector tokens (512-px pages, 64-px tokens, 1280-d -> 12288 pixels)

| encoding | chars/DeepEncoder token | R^2 | ink MCC (R/G/B) | ink F1 (R/G/B) |
|---|---:|---:|---|---|
| latin_p8 | 147.6 | 0.001 | 0.00/0.00/0.00 | 0.00/0.00/0.00 |
| rgb3_lg_pb_h5 | 1692.6 | 0.017 | 0.29/0.29/0.29 | 0.62/0.62/0.62 |

Check: merger output rebuilt from the raster/window bookkeeping vs the module's pooler_output, max |diff| 0.00e+00 (fp32).

## Reading (written after the run; `--summarize` regenerates the tables above, not this section)
* **Use ink MCC to compare codes.** F1 has a chance level of 2p/(1+p), which is high for the dense binary codes
  (ink fraction p ~ 0.45, chance 0.63) and low for Latin (0.25-0.28).
* **A linear pixel readout falls for every encoding, the Latin controls included.** latin_p16 MCC goes 0.94 (b8) ->
  0.39 (b24) -> 0.08 (b28) -> 0.06 (merger), latin_p8 0.89 -> 0.18 -> 0.02 -> 0.03. Qwen reads Latin pages, so
  deep layers re-code text nonlinearly, and a low deep-layer linear readout does not mean the content is lost. What
  this test can show is which codes stay linearly pixel-like, and where.
* **Whole-pixel binary banks stay pixel-like deep into the ViT, and the merger costs the most.** lg_pb_h5 MCC
  1.00 (b8/b16) -> 0.91 (b24) -> 0.81 (b28) -> 0.65 (final) -> 0.30 (merger). The moonshot rgb3_lg_pb_h5 goes
  0.86 -> 0.79 -> 0.64 -> 0.56 -> 0.47 -> 0.29 (mean over R/G/B). Its overlay costs ~0.15 MCC by b8, against the
  single bank. Merger level ~0.29 equals the DeepEncoder calibration (0.29, where 1280 dims cannot hold 12288
  pixels). So after the merger only coarse ink density survives linearly. Both codes are therefore still largely
  linear at b28 (the input to R3-1's unfrozen blocks) and lose most of it in blocks 28-31 + merger, which R3-1 trains.
* **Grey-level / antialiased codes fade early, like Latin.** rgb3_lg_free_h4 goes 0.57 (b8) -> 0.43 (b16) -> 0.22
  (b24) -> 0.16 (b28) -> 0.00 (merger). rgb3_p8 goes 0.63 -> 0.48 -> 0.19 -> 0.07 -> 0.10. Whether the ViT re-codes
  them (as it does Latin) or destroys them, a linear readout cannot tell. For rgb3_lg_free_h4 the loss is already
  large at b8-b16, below the 4 blocks R3-1 unfreezes.
* col_b4 was dropped for time.
