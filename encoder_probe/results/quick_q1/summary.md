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

## Ink F1 (mean over R, G, B; rgb3 codes per channel R/G/B)

| encoding | raw | embed | b8 | b16 | b24 | b28 | final | merger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb3_lg_pb_h5 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.94/0.91/0.91 | 0.90/0.88/0.87 | 0.82/0.80/0.79 | 0.77/0.77/0.75 | 0.72/0.71/0.70 | 0.62/0.62/0.62 |
| rgb3_lg_free_h4 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.68/0.62/0.63 | 0.53/0.50/0.47 | 0.22/0.22/0.18 | 0.15/0.15/0.11 | 0.05/0.05/0.04 | 0.00/0.00/0.00 |
| latin_p8 | 0.999 | 0.997 | 0.905 | 0.695 | 0.112 | 0.004 | 0.006 | 0.016 |
| lg_pb_h5 | 1.000 | 1.000 | 1.000 | 0.999 | 0.954 | 0.896 | 0.811 | 0.617 |
| rgb3_p8 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.66/0.66/0.62 | 0.46/0.48/0.42 | 0.11/0.14/0.09 | 0.02/0.03/0.02 | 0.01/0.01/0.01 | 0.04/0.05/0.04 |
| latin_p16 | 1.000 | 1.000 | 0.944 | 0.870 | 0.356 | 0.027 | 0.009 | 0.024 |

## Calibration: DeepEncoder projector tokens (512-px pages, 64-px tokens, 1280-d -> 12288 pixels)

| encoding | chars/DeepEncoder token | R^2 | ink F1 (R/G/B) |
|---|---:|---:|---|
| latin_p8 | 147.6 | 0.001 | 0.00/0.00/0.00 |
| rgb3_lg_pb_h5 | 1692.6 | 0.017 | 0.62/0.62/0.62 |

Check: merger output rebuilt from the raster/window bookkeeping vs the module's pooler_output, max |diff| 0.00e+00 (fp32).
