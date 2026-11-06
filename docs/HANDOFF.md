# Hand-off (current state, 2026-11-06)

Nothing is running. Everything is on `claude/validation-harness`. Read [AGENTS.md](../AGENTS.md) first (gotchas), then
**[COLOUR_GIST.md](COLOUR_GIST.md) (the current direction: targets, results, amendments, runbook)**; round-3 design
in [encoder_probe/ROUND3.md](../encoder_probe/ROUND3.md). Older session notes, incl.
hardware, sizing and samples of the RTX 4090 session: [history/HANDOFF_2026-11.md](history/HANDOFF_2026-11.md).

## Status
| Line of work | Verdict | Details |
|---|---|---|
| Phases 1-2: hand-designed glyph substitution vs Latin | **KILL**: plain Latin wins | [RESULTS.md](RESULTS.md) |
| Learned glyph alphabet (pixel space, own reader) | **qualified GO**: 0.57x Latin's px²/char | [learned_glyphs/RESULTS.md](../learned_glyphs/RESULTS.md) |
| Frozen encoders (SigLIP, DeepEncoder) | dense codes lost; only sparse Latin reads | [encoder_probe/RESULTS.md](../encoder_probe/RESULTS.md) |
| Round 3, G0: moonshot `rgb3_lg_pb_h5` (322 chars/token) from raw pixels | **1.07% CER**: the page itself is readable | ROUND3.md |
| Round 3, Q1: linear readout of frozen Qwen ViT | glyph codes pixel-like to the last block; everything collapses linearly at the merger | ROUND3.md |
| Round 3, G2-lite (CPU probe, 500 pages) | **"ViT output is the wall at this budget"**: no-merger arm matches ~29% of chars vs ~15% | ROUND3.md |
| Round 3, R3-1 (LoRA fine-tune, 1000 steps, RTX 4090) | **latin_p8 (28 chars/token)** is read; latin_p5 (72) gets the gist only; the glyph codes are not read; **G3 undecidable** | below |
| Analysis of all of the above against blind controls | glyph codes' information is lost inside the frozen ViT, upstream of R3-1's trained blocks; next: train from block 8 | [ANALYSIS.md](ANALYSIS.md) |
| Own encoder, Stage 0: text compressor into frozen Qwen LLM (RTX 4090) | **inconclusive**: the ratio-4 positive control reached only 64% CER at 8000 steps; recipe too slow | [OWN_ENCODER.md](OWN_ENCODER.md) |
| **Colour gist** A0/A0b: Latin in 2-3 colour channels, untrained Qwen (CPU) | reads a blend of the streams, can't pick one by colour; info per vision token R(2) = 0.51 at 16 px (below 1 at every size) | [COLOUR_GIST.md](COLOUR_GIST.md) |
| **Colour gist** A1-cpu: 2 streams, LoRA only, vision frozen, 500 steps (CPU) | **missed** the > 1 bar: R(2) = 0.72 (0.54-0.94); one stream read (68% of a single stream's info), the other not | [COLOUR_GIST.md](COLOUR_GIST.md) |
| **Colour gist** diagnostics (CPU) | both streams survive equally to ViT block 16, then blend in the later blocks (not a brightness problem) | [COLOUR_GIST.md](COLOUR_GIST.md) |

**R3-1 short budget** (1000 steps, 1000 train / 16 test pages, clean, one fair_hash `05dc0c9e8507`):
| Arm (chars/token) | CER | blind CER | cloze (blind) | MC (blind) | val loss (blind) |
|---|---|---|---|---|---|
| latin_p8 (28.1) | **55.0%** | 88.5% | 0.52 (0.22) | 0.88 (0.25) | 1.125 (2.48) |
| latin_p5 (72.4) | 80.9% | 77.8% | 0.31 (0.30) | 0.56 (0.28) | 2.163 (2.402) |
| lg_pb_h5 (107.3) | 78.4% | blind died at step 500 | 0.17 | 0.31 | 2.274 (2.285 at step 500) |
| rgb3_lg_pb_h5 (321.6) | 83.6% | 89.9% | 0.22 (0.22) | 0.34 (0.28) | 2.518 (2.517) |

latin_p8 transcribes the first line(s) of a window nearly exactly, then drifts into invented text; its loss was still
falling. latin_p5's val loss is 9.9% below its blind's (MC 18/32 vs 9/32, p = 0.02): gist, not words. The moonshot's
losses equal its blind run's to 3 decimals. No arm is near 3% CER, so G3 can't be decided.

## What ran and where the rows are
* G0: `encoder_probe/results/round3_cpu_tp200/`. Q1: `encoder_probe/results/quick_q1/summary.md`.
* G2-lite: `encoder_probe/results/g2lite/` (rows), `g2lite_ops/` (op splits), `g2lite_blind/` (floors),
  `g2lite_diag*/` (merger-LR and sanity diagnostics).
* R3-1: `encoder_probe/results/round3_ft_gpu3h/` and `round3_ft_gpu3h_blind/` (one row per eval cond, clean and
  jpeg75); logs `results/queue_logs/gpu3h_{1,2}/`. The LoRA adapters were lost with the instance; nothing depends on them.

## Current direction (2026-11-06): gist through colour channels
User's plan, one variable at a time:
1. Latin in 2, then 3 colour channels (no BPE, no glyphs, large font).
2. Learned glyphs without BPE in a "pure" VLM.
3. Merge.

The target is **gist** (a "system 1" background context), not exact copying.

**Targets** (pre-registered; full table in [COLOUR_GIST.md](COLOUR_GIST.md) "At a glance"). Information per vision
token is measured against each arm's own blind control:
* **headline:** 2 streams carry > 1x the info per vision token of 1 stream at the same font, CI above 1;
* **strong:** ≥ 1.5x, each stream's MC ≥ 0.8x the 1-stream MC, other-stream lure ≤ 15%;
* **useful:** 2 streams at 16 px ≥ 1 stream at 12 px (equal chars per token).

**State after the CPU work:**
* **Untrained**, Qwen reads the superimposed streams as one blended text.
* **A LoRA-only fine-tune with the ViT frozen** lifted one stream but not the other (R(2) 0.72).
* **Where the streams blend:** both streams survive equally in the frozen ViT up to block 16, then the pretrained
  letter and word features blend them.

* **S1, a small VLM with a vision encoder trained from scratch (CPU, 2026-11-06): closed, inconclusive.**
  * It never learned to read a single stream: gap vs swapped images ≤ 0.001 nats/token in all three runs, with a
    collapse fix and an alignment stage. That is a bootstrap and budget limit, not an overlay result
    (COLOUR_GIST.md "S1 result").

So the GPU A1 trains the ViT from block 16 (amendment 3); the prompt below runs it. Remaining CPU options are
builds, not tests (the phase B from-scratch encoder).

## Own encoder, Stage 0 (ran 2026-11-05/06, RunPod RTX 4090)
The text-side compression ceiling: how many chars per slot a learned encoder can pack into Qwen's frozen LLM.
Design, amendment 1 and results: [OWN_ENCODER.md](OWN_ENCODER.md); code `encoder_probe/experiments/text_compress.py`,
table and curves `analyse_stage0.py`, rows `encoder_probe/results/stage0/runs.jsonl`.
* **2026-11-06: positive control failed, Stage 0 inconclusive.** Ratio 4 (17.6 chars/slot) reached CER 74.8% at
  3000 steps and **64.0% at 8000** (blind 77.8%), though its loss is 1.9 nats/token below blind: the decoder
  paraphrases instead of copying. Ratio 16 at 8000 steps: 78.1% (blind 80.3%). Ratios 8, 32 and 80 not run.
  No D3 or D10. Next: a recipe amendment (faster slot learning), not a density conclusion.

## Open items, in priority order
1. **Colour gist A1 on a GPU** (the prompt below):
   * 1 and 2 streams at 16 px with blinds (headline, ~2.3 h on a 4090);
   * 1 stream at 12 px with blind (equal-density bar, +1.2 h);
   * 3 streams (A2, +1.2 h).
2. **If A1 passes:** smaller fonts (A3), then the system-1 use test (gist QA against a text summary at equal KV
   bytes).
3. **Phase B, glyphs without BPE:** a small vision encoder trained from scratch into Qwen2.5-0.5B. It is also the
   fairest test of the overlay itself, since it has no pretrained text features to blend the streams. Then
   phase C (merge).
4. **Parked (earlier lines; still valid, lower priority under the user's direction):**
   * Stage A' on the glyph codes (ROUND3.md R3-1b);
   * the Latin density sweep;
   * Stage 0 amendment 2 ([OWN_ENCODER.md](OWN_ENCODER.md));
   * 5000-step Latin arms.

## Prompt for the next GPU session
```text
Pull branch claude/validation-harness. Read AGENTS.md (gotchas), docs/HANDOFF.md and docs/COLOUR_GIST.md:
targets ("At a glance"), A0/A0b/A1-cpu results, amendments 1-3 and the A1 runbook. Mechanics: docs/GPU_RUNNER.md
sections 0-2 (any Linux GPU) or docs/LOCAL_GPU.md section 8 (Windows RTX 3090).

1. Setup (GPU_RUNNER.md section 1), then the CPU tests encoder_probe/tests/test_colour.py and test_round3.py.
   Note GPU, free disk and torch/transformers/peft versions in your first commit.
2. Smoke with the amendment-3 recipe:
   python3 encoder_probe/experiments/finetune_qwen.py --encoding rgb2s_p16 --smoke --vision-blocks 16 --lr-vision 1e-4
   Check device=cuda bf16=True, the row has per_stream (cyan, magenta) and val_per_sample; record s/step and peak
   GPU memory. If it does not fit, use --vision-blocks 12 for EVERY arm and record that as a dated amendment.
3. Run the A1 runbook loop in COLOUR_GIST.md exactly (rgb1s_p16, rgb2s_p16, rgb1s_p12, rgb3s_p16; each with its
   --blind-images run; identical flags, one fair_hash) under nohup, pushing after each arm. If time is short,
   stop after a complete arm pair (never leave an arm without its blind). Keep >= 5 GB disk free.
4. Then: python3 encoder_probe/experiments/analyse_colour.py --real colour_a1 --pitch 16 --equal-density rgb2s_p16 rgb1s_p12
   Add its table, the per-stream table and the verdicts as a dated "A1 results" section in docs/COLOUR_GIST.md
   (headline R(2) with CI; strong; equal density; A2 only if the headline passed), update the "At a glance" table
   and docs/HANDOFF.md, and give me the results table.

Stop and ask me if the smoke fails or a result would change the plan; record any change as a dated amendment in
COLOUR_GIST.md before running it. Never choose anything on test-page CER. No PRs, no Claude API calls, no model
IDs in commits. Stop the pod when finished: it bills by the hour.
```

## Gotchas for GPU runners
* **Disk quota:** the 4090 runner's quota filled at ~10 GB (checkpoints ~1.75 GB, adapters ~0.56 GB each, page caches
  ~3 GB), killing a blind job and truncating 4 docs in a commit. Keep >= 3 GB free and check `df` before committing.
* **Every code needs its own blind control** at the same settings: blind floors differ by code (latin_p5 77.8% CER,
  latin_p8 88.5%, moonshot 89.9%). A CER is evidence of reading only well below its own blind.
* **Merger / vision LR:** pretrained modules collapse at the probe's LR 2e-3; keep pretrained parameters at <= 1e-4
  (`--lr-vision`, default 2e-5; `run_g2lite.py --merger-lr 1e-4`).
* Fairness guard: never change a flag for one arm only; all compared rows must share one `fair_hash`.
* Measured on the RTX 4090: 1.2 s/step (batch 4, bf16, grad checkpointing), peak 11.4 GB, ~7 min load + eval per job.
