# Quick test Q1: linear readout of frozen Qwen2.5-VL ViT features (2026-11-04)

Closed-form ridge from each feature to the clean pixels it covers (14-px patches; merger: 28-px tokens), 64 train pages (lambda picked on pages 48-63), 16 held-out test pages (WikiText test/val), 448-px pages. bN = after blocks 0..N-1; final = after block 31, before the merger. A linear readout is a lower bound on what a feature holds.

## R^2 of the test-page pixels

| encoding | chars/Qwen token | raw | embed | b8 | b16 | b24 | b28 | final | merger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb3_lg_pb_h5 | 321.6 | 1.000 | 1.000 | 0.691 | 0.606 | 0.425 | 0.343 | 0.230 | 0.019 |
| latin_p16 | 6.8 | 1.000 | 1.000 | 0.969 | 0.841 | 0.420 | 0.220 | 0.174 | 0.157 |

## Ink MCC (Matthews correlation of the ink masks; 0 = chance, 1 = exact; rgb3 codes R/G/B)

| encoding | raw | embed | b8 | b16 | b24 | b28 | final | merger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb3_lg_pb_h5 | - | - | - | - | - | - | - | - |
| latin_p16 | - | - | - | - | - | - | - | - |

## Ink F1 (rgb3 codes R/G/B); chance = 2p/(1+p) for ink fraction p (an all-ink guess)

| encoding | F1 chance | raw | embed | b8 | b16 | b24 | b28 | final | merger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb3_lg_pb_h5 | - | 1.00/1.00/1.00 | 1.00/1.00/1.00 | 0.94/0.91/0.91 | 0.90/0.88/0.87 | 0.82/0.80/0.79 | 0.77/0.77/0.75 | 0.72/0.71/0.70 | 0.62/0.62/0.62 |
| latin_p16 | - | 1.000 | 1.000 | 0.944 | 0.870 | 0.356 | 0.027 | 0.009 | 0.024 |

Check: merger output rebuilt from the raster/window bookkeeping vs the module's pooler_output, max |diff| 0.00e+00 (fp32).
