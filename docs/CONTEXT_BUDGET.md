# Context budget: what each encoding could buy on the 3090 (living table)

Goal (user, 2026-11-04): show a local model on the RTX 3090 holding much more context than it can by default, without
shortcutting the research. This page turns each encoding's chars per context position (cpt) into context, KV and
compute numbers, and says how much of that is measured. **Update the "Evidence" table after every round-3 step.**

## Baselines

| | Qwen2.5-VL-3B (round-3 test model) | DeepSeek-V4.1-Flash (user's reference) |
|---|---|---|
| Size | 3.75B params (3.09B LLM + ~0.67B ViT), bf16 ~7.5 GB, fits the 3090 | 552B MoE, 8B active (prefill) / 16B (decode) [paper] |
| KV per position | 36,864 B (36 layers x 2 KV heads x 128 x 2 x bf16; verified from config) | 890 B global KV [paper]; ~5.4 KB measured in the community 3090 offload config (1.5 GiB fp8 = 297k tokens) |
| Native window | 32,768 positions | 1,048,576 tokens [NVIDIA card]; 262k in the 3090 offload config |
| Chars in window as text | ~142k (4.33 chars/token, measured on held-out WikiText) | ~4.5M (assumes ~4.3 chars/token: not measured) |
| On this PC | runs | **does not run as configured:** the offload config pins 190 GiB of experts in host RAM (needs 256+ GiB); this PC has 32 GB |
| 3090 speed | not measured yet | prefill ~690-810 tok/s, decode ~19-21 tok/s (community D119/D141 runs, 8k-261k) |
| Binding limit on context | the 32k window (KV at 32k is only 1.2 GB) | the 1M window and prefill time (1M tokens ~ 22 min); KV is not the limit (1M tokens ~ 5.4 GB measured, 0.9 GB at the paper's figure) |

Sources: DeepSeek-V4.1-Flash paper (arXiv 2609.19969), NVIDIA model card, community offload wiki
(github.com/0xSero/dsv41-flash-offload). How V4.1-Flash tokenises images is not public in the abstract: unknown.

## Potential (upper bounds IF the model reads the encoding perfectly)

cpt = chars per context position. Compute/char = prefill FLOPs per document char vs plain text on Qwen-3B, estimated as
2 x params per position, with the ViT adding ~5.4 GFLOP per vision position (4 patches x 0.67B x 2); attention ignored.

| Input | cpt | Positions per 1M chars | KV cut vs text | Qwen-3B: KV per 1M chars | Qwen-3B: chars in 32k | V4.1-Flash: chars in 1M window | Prefill compute/char (Qwen-3B) |
|---|---|---|---|---|---|---|---|
| Plain text (default) | 4.33 | 231k | 0% | 8.5 GB | 0.14M | ~4.5M | 1.00x |
| Latin page images, Glyph-level | ~15 | 67k | 71% | 2.5 GB | 0.49M | ~16M | 0.54x |
| Colour blocks col_b4 | 49 | 20k | 91% | 0.75 GB | 1.6M | ~51M | 0.17x |
| rgb3_p8 | ~73 | 14k | 94% | 0.50 GB | 2.4M | ~77M | 0.11x |
| Learned glyphs lg_pb_h5 | ~107 | 9.3k | 96% | 0.34 GB | 3.5M | ~112M | 0.08x |
| Moonshot rgb3_lg_pb_h5 | 322 | 3.1k | 98.7% | 0.11 GB | 10.6M | ~338M | 0.03x |
| Moonshot rgb3_lg_free_h4 | 405 | 2.5k | 98.9% | 0.09 GB | 13.3M | ~425M | 0.02x |

Notes:
* Images cost ~1.9x text per position on Qwen-3B (the ViT), so they break even on compute at ~8 cpt.
* Decode: each generated token reads the whole KV, so fewer positions also speed long-context decoding on Qwen-3B.
  On V4.1-Flash decode is PCIe/expert-bound, not KV-bound: expect no decode gain there.
* Dense codes have no language prior, so a single misread glyph can corrupt a name or number. Exact-recall tasks
  (needle) are the hard test, not the gist.

**Literature calibration (docs/research/2026-11-04_rgb3_lg_go_no_go_report.md):** plausible go ~30-60 chars/token
(already ~1.3M chars in Qwen's 32k window, ~9x text), stretch 60-120, 322-405 implausible at LoRA-3B scale.

## Evidence (what is measured so far)

| Input | Raw pixels (identity ceiling, probe) | Frozen encoder (probe) | Fine-tuned Qwen (R3-1) | Long-context QA (R3-2) |
|---|---|---|---|---|
| Latin images | latin_p16 0.06% (DeepEncoder geometry) | DeepEncoder: 8.5 cpt at 15-19%; 36 cpt at floor. SigLIP l6: ~2 cpt for 5%. Qwen: R3-0 not run | not run | not run (prior art: Glyph 3-4x at Qwen3-8B-level accuracy, with heavy training) |
| col_b4 | col_b2 0.02%, colour < 3% | floor (71-73%) on SigLIP and DeepEncoder | not run | not run |
| rgb3_p8 | 0.42% | floor (83.5%) | not run | not run |
| lg_pb_h5 | 0.52% | floor (82.4%) | not run | not run |
| rgb3_lg_pb_h5 | partial: 15.2% at step 500/3000 (identity28) | not run | not run | not run |
| rgb3_lg_free_h4 | partial: 37.0% at step 500/3000 | not run | not run | not run |

Q1 (2026-11-04, linear pixel readout of frozen Qwen ViT layers, encoder_probe/results/quick_q1/summary.md):
ink MCC at b28 / merger: rgb3_lg_pb_h5 0.56 / 0.29, lg_pb_h5 0.81 / 0.30, rgb3_lg_free_h4 0.16 / 0.00, rgb3_p8
0.07 / 0.10, latin_p8 0.02 / 0.03. Latin also fails this linear test and is still read by Qwen, so it is a lower bound,
not a readability result.

A row counts as a real context gain only when R3-2 shows QA accuracy within the ROUND3.md margins at the stated cpt.
Until then the Potential table is a ceiling, not a result.

## What this means for the demo
* On this PC the demo model is Qwen2.5-VL-3B (or a larger Qwen VL that fits). There the window, not KV, caps context:
  Glyph-level Latin would take it from ~142k to ~490k chars in 32k positions, and a working dense code into the
  millions.
* V4.1-Flash can't be the local demo as is (RAM), and can't be fine-tuned locally. Its KV is already tiny, so for it the
  gains would be window size (past ~4.5M chars) and prefill time, and only if its own vision encoder reads the
  encoding. A cheap, honest check would be a zero-shot test of V4.1-Flash on rendered Latin pages through a hosted API.
  Not done; it needs the user's go-ahead (external, paid).
