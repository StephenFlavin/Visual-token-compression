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
| Round 3, R3-1 (LoRA fine-tune, 1000 steps, RTX 4090) | only **latin_p8 (28 chars/token)** is read; every denser code sits at its own blind floor; **G3 undecidable** | below |

**R3-1 short budget** (1000 steps, 1000 train / 16 test pages, clean, one fair_hash `05dc0c9e8507`):
| Arm (chars/token) | CER | blind CER | cloze (blind) | MC (blind) | val loss (blind) |
|---|---|---|---|---|---|
| latin_p8 (28.1) | **55.0%** | 88.5% | 0.52 (0.22) | 0.88 (0.25) | 1.125 (2.48) |
| latin_p5 (72.4) | 80.9% | 77.8% | 0.31 (0.30) | 0.56 (0.28) | 2.163 (2.402) |
| lg_pb_h5 (107.3) | 78.4% | blind died at step 500 | 0.17 | 0.31 | 2.274 (2.285 at step 500) |
| rgb3_lg_pb_h5 (321.6) | 83.6% | 89.9% | 0.22 (0.22) | 0.34 (0.28) | 2.518 (2.517) |

latin_p8 transcribes the first line(s) of a window nearly exactly, then drifts into invented text; its loss was still
falling. The moonshot's losses equal its blind run's to 3 decimals. No arm is near 3% CER, so G3 can't be decided.

## What ran and where the rows are
* G0: `encoder_probe/results/round3_cpu_tp200/`. Q1: `encoder_probe/results/quick_q1/summary.md`.
* G2-lite: `encoder_probe/results/g2lite/` (rows), `g2lite_ops/` (op splits), `g2lite_blind/` (floors),
  `g2lite_diag*/` (merger-LR and sanity diagnostics).
* R3-1: `encoder_probe/results/round3_ft_gpu3h/` and `round3_ft_gpu3h_blind/` (one row per eval cond, clean and
  jpeg75); logs `results/queue_logs/gpu3h_{1,2}/`. The LoRA adapters were lost with the instance; nothing depends on them.

## Open items, in priority order
1. **latin_p8, latin_p6, latin_p5 at 5000 steps, each with its own `--blind-images` control.** Does latin_p8's drift
   go away, and how far can fine-tuned Latin be pushed? This sets the G3 baseline. On a 4090 ~1 h 45 min per job.
2. **Short vision-side diagnostic** on lg_pb_h5 or the moonshot (R3-1b stage B, light): `--lr-vision 1e-4` and/or
   more `--vision-blocks`, 1000 steps, loss vs its own blind at the same settings. A long dense-code run only if
   the loss leaves the blind's.
3. R3-2 long-context benchmark, only for an arm that passes G3.
4. Lower priority: rgb3_p8 (~73) arm; G2 at scale on a GPU (GPU_RUNNER.md section 6); moonshot levers (ROUND3.md).

## Prompt for the next GPU session
```text
Pull branch claude/validation-harness. Read AGENTS.md, docs/HANDOFF.md, and in encoder_probe/ROUND3.md the R3-1
short-budget results, R3-1b and both amendments. Runbook: docs/GPU_RUNNER.md (any Linux GPU; measure s/step and
peak memory in the smoke, then size), or docs/LOCAL_GPU.md section 8 on the Windows RTX 3090 PC.

1. Setup, CPU tests, then finetune_qwen.py --encoding latin_p8 --smoke. Pre-render pages on the CPU first.
2. Arms at identical settings, 5000 steps, one run name (one fair_hash): latin_p8, latin_p6, latin_p5, and for
   EACH the same command with --blind-images (run name gets _blind). Order: latin_p8 + blind, latin_p5 + blind,
   latin_p6 + blind. Long jobs with nohup and checkpoints (--ckpt-every 250); push rows often.
3. Then a short vision-side diagnostic (new run name, 1000 steps) on lg_pb_h5 or rgb3_lg_pb_h5: --lr-vision 1e-4
   and/or --vision-blocks 8, plus its own --blind-images run at the same settings. Report train/val loss vs the
   blind's; never above 1e-4 for pretrained weights (AGENTS.md: merger LR collapse).
4. Report every CER next to its own blind floor, with the op split (ops: sub/ins/del, len_ratio) and prefix_frac.
   Apply ROUND3.md G3; never choose anything on test-page CER. R3-2 only for an arm that passes G3.
5. Keep >= 3 GB of disk free (df -h . ; delete finished checkpoints and page caches you no longer need).

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
