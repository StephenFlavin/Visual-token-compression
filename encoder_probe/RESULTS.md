# Frozen-encoder probe: results (2026-11-03)

Design, pre-registered predictions, decision rules and **every probe change** are in [PLAN.md](PLAN.md).
Rows: `results/main/` (grid), `results/controls/` (post-hoc controls incl. the SigLIP layer-6 set),
`results/controls_tmlp2/`, `results/controls_tp1000/` (probe-capacity controls). Tables below come from
`python3 encoder_probe/experiments/analyse.py` (main + controls).

## Verdict
* **Colour pixels: dead for VLMs with a frozen pretrained encoder.** With raw pixels (identity) the probe reads
  1×1 colour blocks almost perfectly (col_b1 0.06% CER = 256 chars per token). Through frozen SigLIP-B/16 no colour
  encoding gets below 62% CER at any block size (1-4 px), at the last layer or at layer 6, clean or JPEG q75.
  For 1-3 px blocks even the *training* CTC stays near the unigram plateau (2.4-2.8 nats/char), so the colour
  codes are not decodable from the tokens by this probe at all, not just badly generalised; col_b4 at layer 6
  fits its training pages (CTC 0.7) but does not generalise (62%). Plain Latin through the same encoder and probe does reach
  4-9% CER (at large text), so the encoder/probe pair is informative for shapes but not for colours.
  Decision rule 1 (colour ≥ 2× best Latin chars/token at 3% through siglip/jpeg75): **fails**; colour never
  reaches 3% (or 10%) anywhere through SigLIP.
* **Learned glyphs: constrain to whole pixels.** At the identity ceiling grey-level banks win (lg_free_h4 0.01% vs
  lg_pb_h4 2.6% at the same 5.9 px²/char), but through SigLIP layer 6 the whole-pixel banks keep far more
  (lg_pb_h4 54.7% vs lg_free_h4 75.9% clean). Decision rule 2 (use grey-level only if lg_free still beats lg_pb
  through siglip/jpeg75): **fails** -> whole-pixel banks for the GPU plan.
* **Frozen SigLIP-B/16 at 224 px destroys almost all fine text detail for this probe.** Last layer: every encoding
  58-85% CER, Latin included up to 306 px²/char. Layer 6: Latin needs ≈ 122 px²/char (2.1 chars per vision
  token) for 5% CER, i.e. *fewer* characters per token than text tokens (≈ 4). Nothing reaches 3%.
  Interesting side result: learned whole-pixel glyphs survive layer 6 better than DejaVu at a third of the area
  (lg_pb_h4 54.7% at 5.9 px²/char vs latin_p6 62.5% at 16 px²/char).

**Caveat (applies to every number through SigLIP):** a small probe trained from scratch on 200 pages of frozen
tokens is not an LLM reading them. Through SigLIP the probe overfits badly (latin_p8 last layer: train CTC 0.13,
test 58%; latin_p24: train 0.002, test 40%), and 5× more training pages took latin_p10 from 61% to 43%, a
2-block token MLP to 52%. So the SigLIP CERs are upper bounds on the information the encoder destroys, and
absolute levels are probe-limited. The *comparisons* (colour vs Latin vs learned glyphs, same probe, same data)
are what the decisions rest on: colour (1-3 px) stays at the unigram floor even in training while Latin and lg_pb
learn, which more probe data would not obviously reverse. VLMs that fine-tune the encoder (PaliGemma, Qwen-VL) are outside
this test.

## Pre-registered predictions
| # | prediction | result | verdict |
|---|---|---|---|
| 1 | identity/clean col_b1 < 3% | 0.06% (256 chars/token) | **pass** |
| 2 | identity/jpeg75 col_b1 > 20%; col_b2 also degrades | col_b1 28.8% (from 0.06%); col_b2 0.92% (from 0.00%) | **pass** |
| 3 | siglip/clean col_b1 > 20%; far below 256 chars/token | 72.6% (unigram level); no colour encoding below 62% through SigLIP at any layer | **pass** |
| 4 | through siglip, lg_free's grey-level advantage over lg_pb shrinks or reverses | identity: lg_free_h4 0.01% vs lg_pb_h4 2.57%; siglip: 79.7% vs 75.1%; siglip_l6: 75.9% vs 54.7% (jpeg75: 75.3% vs 58.7%) | **pass (reverses)** |

## What survives the encoder (same encoding, identity/clean ceiling vs through the encoder)
Best (lowest-CER) encoding per family; unigram-level CER on this text is ≈ 72-75%.

| family | identity/clean | identity/jpeg75 | siglip/clean | siglip_l6/clean | siglip_l6/jpeg75 |
|---|---:|---:|---:|---:|---:|
| latin | 0.02% (p8) | 0.09% (p10) | 26.8% (p16, 2.1 c/tok) | 4.9% (p16) | 4.0% (p16) |
| lg_free | 0.01% (h4) | 0.07% (h4) | 79.7% (h4) | 75.9% (h4) | 75.3% (h4) |
| lg_pb | 0.75% (h5) | 0.76% (h5) | 75.1% (h4) | 53.4% (h5) | 56.4% (h5) |
| colour | 0.00% (b2) | 0.07% (b4) | 72.6% (b1) | 62.5% (b4) | 67.0% (b4) |

* **JPEG alone is not what kills colour.** Without an encoder, colour blocks survive q75 well from 2×2 up
  (col_b2 0.92%, col_b3 0.88%, col_b4 0.07%); only 1×1 blocks break (28.8%, chroma subsampling). Identity/jpeg75
  colour reaches C@3% = 71 chars/token vs 23.5 for Latin (3×): a pixel-level reader could use colour codes. The
  pretrained encoder is what removes them.
* Through SigLIP the ranking flips relative to the ceiling: lg_pb > latin (per area) > colour ≈ lg_free.
  Grey-level detail (lg_free, colour) is what the encoder discards; binary-ish shapes survive best.
* Last layer vs layer 6: layer 6 keeps far more (latin_p16 26.8% -> 4.9%, lg_pb_h5 75% -> 53%), consistent with the
  last layer being shaped by the pooled contrastive objective. JPEG q75 on top of layer 6 costs little
  (≤ 5 points) — the encoder, not compression, is the bottleneck.
* Probe-capacity controls on siglip/clean latin_p10 (61.3%): 2 token-MLP blocks 52.2%, 1000 train pages 43.5%.
  More data helps a lot, so absolute SigLIP numbers are probe/data-limited (see caveat).

## Probe and protocol notes
* Probe (identical for every row): tokens -> 3×3 conv map + linear skip -> unpatchified 3-channel page with an
  auxiliary pixel-reconstruction loss (×10, training pages only) -> column-shared dilated-conv CTC line reader;
  3000 steps, 200 train / 16 test pages. Four probe versions were tried on identity/pilot runs before this one
  (PLAN.md "Probe changes"); none was chosen on SigLIP results.
* Colour rows are scored with unstripped hypotheses (`score_v: 2`); the first 4 identity/clean colour rows had a
  0.4% scoring floor and were re-run.
* siglip (last layer)/jpeg75 was not run: siglip/clean is already at the floor for every encoding. The
  layer-6 set (post-hoc control, PLAN.md) stands in for it in the decision rules.
* C@x = chars per vision token (256 / px²/char) at which the isotonic CER curve crosses x; `>=` = already below x at
  the densest encoding of the family (left-censored), `-` = never reached.

## Tables

### CER by encoding

| encoding | px²/char | chars/token | identity/clean | siglip/clean | siglip_l6/clean | siglip_l6/jpeg75 | identity/jpeg75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| latin_p24 | 306.42 | 0.8 | 0.11% | 39.50% | 7.14% | 8.74% |  |
| latin_p16 | 122.01 | 2.1 | 0.12% | 26.75% | 4.88% | 3.98% |  |
| latin_p10 | 46.26 | 5.5 | 0.05% | 61.34% | 37.27% | 34.96% | 0.09% |
| latin_p8 | 28.68 | 8.9 | 0.02% | 58.02% | 26.37% | 29.70% | 0.13% |
| latin_p6 | 16.03 | 16.0 | 0.17% | 74.66% | 62.51% |  | 0.39% |
| latin_p5 | 11.14 | 23.0 | 0.62% | 74.98% | 71.96% |  | 2.28% |
| latin_p4 | 6.97 | 36.8 | 3.49% | 73.62% | 68.37% |  | 18.39% |
| lg_free_h4 | 5.86 | 43.7 | 0.01% | 79.68% | 75.94% | 75.31% | 0.07% |
| lg_free_w23_h3 | 4.10 | 62.4 | 0.08% | 84.82% | 85.45% | 85.76% | 0.82% |
| lg_pb_h5 | 7.46 | 34.3 | 0.75% | 75.25% | 53.41% | 56.35% | 0.76% |
| lg_pb_h4 | 5.86 | 43.7 | 2.57% | 75.13% | 54.65% | 58.68% | 2.62% |
| col_b4 | 16.00 | 16.0 | 0.04% | 75.32% | 62.47% | 66.98% | 0.07% |
| col_b3 | 9.16 | 27.9 | 0.01% | 75.21% | 70.78% | 72.51% | 0.88% |
| col_b2 | 4.00 | 64.0 | 0.00% | 75.05% | 70.74% | 70.97% | 0.92% |
| col_b1 | 1.00 | 256.0 | 0.06% | 72.63% | 71.90% | 72.06% | 28.77% |

### C@10% (chars per vision token; text tokens ≈ 4 chars/token)

| family | identity/clean | siglip/clean | siglip_l6/clean | siglip_l6/jpeg75 | identity/jpeg75 |
|---|---:|---:|---:|---:|---:|
| latin | >=36.8 | - | 2.4 | 2.4 | 28.8 |
| lg_free | >=62.4 | - | - | - | >=62.4 |
| lg_pb | >=43.7 | - | - | - | >=43.7 |
| colour | >=256.0 | - | - | - | 100.6 |

### C@3% (chars per vision token; text tokens ≈ 4 chars/token)

| family | identity/clean | siglip/clean | siglip_l6/clean | siglip_l6/jpeg75 | identity/jpeg75 |
|---|---:|---:|---:|---:|---:|
| latin | 33.9 | - | - | - | 23.5 |
| lg_free | >=62.4 | - | - | - | >=62.4 |
| lg_pb | >=43.7 | - | - | - | >=43.7 |
| colour | >=256.0 | - | - | - | 71.0 |

### C@1% (chars per vision token; text tokens ≈ 4 chars/token)

| family | identity/clean | siglip/clean | siglip_l6/clean | siglip_l6/jpeg75 | identity/jpeg75 |
|---|---:|---:|---:|---:|---:|
| latin | 24.5 | - | - | - | 17.9 |
| lg_free | >=62.4 | - | - | - | >=62.4 |
| lg_pb | 35.5 | - | - | - | 35.4 |
| colour | >=256.0 | - | - | - | 64.3 |
