# Hand-off (current state, 2026-11-05)

Nothing is running. Everything is on `claude/validation-harness`. Read [AGENTS.md](../AGENTS.md) first (gotchas), then
[encoder_probe/ROUND3.md](../encoder_probe/ROUND3.md) (round-3 design, gates, amendments). Older session notes, incl.
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
Plan, gates and the first (zero-shot) result: [COLOUR_GIST.md](COLOUR_GIST.md). Untrained Qwen reads one stream
through an overlapping one (14% CER) but cannot pick a stream by colour, so stream separation has to be trained:
A1 (2 streams, fine-tuned, ~2 h on a 4090) is next: code built and CPU-tested; runbook and GPU prompt in COLOUR_GIST.md.
Headline bar (user): the overlay carries > 1x the information per vision token of one stream (CI above 1).

## Own encoder, Stage 0 (ran 2026-11-05/06, RunPod RTX 4090)
The text-side compression ceiling: how many chars per slot a learned encoder can pack into Qwen's frozen LLM.
Design, amendment 1 and results: [OWN_ENCODER.md](OWN_ENCODER.md); code `encoder_probe/experiments/text_compress.py`,
table and curves `analyse_stage0.py`, rows `encoder_probe/results/stage0/runs.jsonl`.
* **2026-11-06: positive control failed, Stage 0 inconclusive.** Ratio 4 (17.6 chars/slot) reached CER 74.8% at
  3000 steps and **64.0% at 8000** (blind 77.8%), though its loss is 1.9 nats/token below blind: the decoder
  paraphrases instead of copying. Ratio 16 at 8000 steps: 78.1% (blind 80.3%). Ratios 8, 32 and 80 not run.
  No D3 or D10. Next: a recipe amendment (faster slot learning), not a density conclusion.

## Open items, in priority order
Reordered 2026-11-05 after [ANALYSIS.md](ANALYSIS.md): R3-1 trained too little of the ViT to test the glyph codes, and
latin_p8 is underfitting, so the recipe is tested before the long runs.
1. **Stage A' vision-side diagnostic** (ROUND3.md R3-1b, pre-registered): `--vision-blocks 24 --lr-vision 1e-4`,
   1000 steps, latin_p8, lg_pb_h5 and rgb3_lg_pb_h5, each with `--blind-images`. Glyph line continues only if its
   val-loss gain vs blind is > 0.02 nats/token; latin_p8's val loss < 1.0 makes A' the recipe for later arms.
   ~6 jobs of ~35 min on a 4090 (measure in the smoke: peak ~18-19 GB estimated).
2. **Latin density sweep:** latin_p12 and latin_p10 (~12.5 and ~18 chars/token), 1000 steps, on the better recipe,
   each with a blind. Maps how readability falls with density: the demo's operating point.
3. **5000-step Latin arms** (latin_p8, p6, p5 with blinds) on the better recipe: the G3 baseline. ~1 h 45 min per
   job on a 4090 with the R3-1 recipe.
4. R3-2 long-context benchmark, only for an arm that passes G3.
5. Lower priority: a per-line eval of latin_p8 (recognition vs losing its place; needs a flag); the G2-lite direct
   probe on block-8/16 states (CPU, needs a block option); rgb3_p8 (~73); moonshot levers (ROUND3.md).

## Prompt for the next GPU session
```text
Pull branch claude/validation-harness. Read AGENTS.md, docs/HANDOFF.md, and in encoder_probe/ROUND3.md the R3-1
short-budget results, R3-1b and both amendments. Runbook: docs/GPU_RUNNER.md (any Linux GPU; measure s/step and
peak memory in the smoke, then size), or docs/LOCAL_GPU.md section 8 on the Windows RTX 3090 PC.

1. Setup, CPU tests, then finetune_qwen.py --encoding latin_p8 --smoke. Pre-render pages on the CPU first.
2. Stage A' (ROUND3.md R3-1b, pre-registered; read docs/ANALYSIS.md): 1000 steps, one run name, flags
   --vision-blocks 24 --lr-vision 1e-4 for every arm: latin_p8, lg_pb_h5, rgb3_lg_pb_h5, and for EACH the same
   command with --blind-images. Smoke it first and check peak memory (estimate ~18-19 GB); if it does not fit, use
   --vision-blocks 16 for every arm and say so. Never above 1e-4 for pretrained weights. Then run
   python3 encoder_probe/experiments/analyse_r3.py adapted to the new run names, and apply A''s decision rules.
3. Latin density sweep: latin_p12 and latin_p10, 1000 steps, on the better recipe (A' if latin_p8's val loss < 1.0,
   else R3-1's), each with --blind-images.
4. If time remains: latin_p8, latin_p6, latin_p5 at 5000 steps on that recipe, each with --blind-images. Long jobs
   with nohup and checkpoints (--ckpt-every 250); push rows often.
5. Report every arm against its own blind: val-loss gain first, then MC/cloze, then CER with the op split (ops:
   sub/ins/del, len_ratio) and prefix_frac. Apply ROUND3.md G3; never choose anything on test-page CER. R3-2 only
   for an arm that passes G3.
6. Keep >= 3 GB of disk free (df -h . ; delete finished checkpoints and page caches you no longer need).

Stop and ask me if the smoke fails, if fine-tuned Latin stays at its blind floor, or if a result would change the
plan. Record design changes as dated amendments in ROUND3.md before running them. At the end update
encoder_probe/ROUND3.md, docs/HANDOFF.md and docs/RESULTS.md and give me a results table. No PRs, no Claude API
calls, no model IDs in commits.
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
