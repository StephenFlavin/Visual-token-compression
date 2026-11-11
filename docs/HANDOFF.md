# Hand-off (current state, 2026-11-11)

**The goals were reframed on 2026-11-11** ([REVIEW_2026-11-11.md](REVIEW_2026-11-11.md), README "Goals"). Read
[AGENTS.md](../AGENTS.md) first (gotchas), then this page.

| Goal | Why | State |
|---|---|---|
| 1. Page images as cheaper context for Claude, stretched from the text side | The vision encoder is the only compressor a user can reach on a closed model; 2.2x is measured lossless; denser rendering is exhausted | **Test A** (prune the text, then render it): [results/claude_qa/prune](../results/claude_qa/prune/README.md) |
| 2. Text slots against page pixels on a model we can train | Once the reader is trained, pixels are optional; the glyph runs were never compared with a text-side compressor | **Test B**, pre-registered in [TEXT_SLOTS.md](TEXT_SLOTS.md); code and tests are on `claude/sharp-mendel-kkhg4g`; **the 3090 prompt is the next section** |
| 3. Close the original idea (braille-like glyphs, then BPE and the RGB overlay) with a fair test | Stage A (is a glyph alphabet learnable through a trained ViT at all) and the channel capacity are the last untested links, already queued | Runs 1-2 driver on the 3090 since 2026-11-09 ([PLAN_3090.md](PLAN_3090.md)); **Stage C and colour Run 4 are held until Test B is in** |

Branches: the 3090 PC works on `claude/validation-harness`; the review, Test A and Test B are on
`claude/sharp-mendel-kkhg4g` (one commit line ahead of it) and are merged in by step 1 of the prompt. Parked:
the colour-gist line ([COLOUR_GIST.md](COLOUR_GIST.md)) and Stage 0 ([OWN_ENCODER.md](OWN_ENCODER.md), replaced by
Test B). Older session notes: [history/HANDOFF_2026-11.md](history/HANDOFF_2026-11.md).

## Prompt for the 3090 (2026-11-11): Test B first, then the image arms with Stage C held
Order and reasons: Test B takes about one evening of GPU time and decides whether the remaining image arms are a
comparison (Latin, Stages A-B) or a dead end (Stage C). The Latin arms of Runs 1-2 are the image side of Test B's
comparison, so they resume right after it.
```text
Work in /c/git/visual-bpe-run (Git Bash; `. .venv/env.sh`; `python`, not `python3`; PYTHONUTF8=1). Read AGENTS.md
(gotchas, especially GPU runners and Windows), docs/HANDOFF.md, docs/TEXT_SLOTS.md and docs/LOCAL_GPU.md section 9.
TEXT_SLOTS.md is pre-registered: do not change its arms, gates, rules or predictions after seeing data; if
something must change, write a dated amendment in it first. One GPU job at a time. Keep >= 10 GB of disk free.

0. State. Show: tail of results/r12_logs/driver.log, results/r12_logs/done.txt, whether results/r12_logs/STOP
   exists, nvidia-smi, free disk. If the Runs 1-2 driver is running: wait for the current run's next checkpoint
   (every 250 steps, ~16 min), then stop the driver process first and its python child second (the run resumes
   exactly from the checkpoint). If that leaves a STOP file, it comes from the kill: delete it. Commit and push
   any finished r12 rows on claude/validation-harness as the driver does.
1. Code. git fetch origin claude/sharp-mendel-kkhg4g && git merge origin/claude/sharp-mendel-kkhg4g
   (into claude/validation-harness). If it conflicts, stop and report; do not resolve by hand. Then:
   python -m pytest -q encoder_probe/tests/test_text_slots.py encoder_probe/tests/test_text_compress.py
2. Smoke. python encoder_probe/experiments/text_slots.py --ratio 1 --smoke   (ratio 1 = the longest sequence)
   Check: device cuda, bf16 True, the loss falls, a row is written. Record s/step and peak GiB (nvidia-smi and the
   row). If nvidia-smi shows more than 20 GB: use --batch-size 4 and twice the steps for EVERY arm, and write
   that as a dated line under "Results" in docs/TEXT_SLOTS.md before the first arm.
3. Gates. Run arms L0, L1, L2 exactly as in the TEXT_SLOTS.md runbook (one at a time; the same command resumes
   after a power-off). Then: python encoder_probe/experiments/analyse_slots.py encoder_probe/results/tb
   Commit and push encoder_probe/results/tb. G1 FAIL: stop and report. G2 FAIL: apply the pre-registered rule
   (one rerun of L2 at 3000 steps only if the validation loss was still falling; otherwise stop and report).
4. Ladder. L3, L4, L5, then F2, F4, F8. Commit and push the rows after each arm.
5. Analysis and write-up. Run analyse_slots.py with one --image per finished trained Latin arm of Runs 1-2
   (name:chars per token:upper CI of bits per token on natural:on shuffled, from
   encoder_probe/results/analysis_run12/summary.md; "-" where missing). Add a dated "Results" section to
   docs/TEXT_SLOTS.md: the table, G1 and G2, each prediction as PASS / FAIL / not evaluable, the win lines, two
   sample outputs each of L4 and L5, s/step, peak VRAM, problems. One line in AGENTS.md "Findings" and one row in
   the docs/HANDOFF.md table. Commit, push.
6. Image arms, Stage C held. Resume the driver without stage C (finished steps are skipped):
   nohup python encoder_probe/experiments/run12_driver.py --steps 6000 --stages A B chan >> results/r12_logs/driver.out 2>&1 &
   When it logs "all requested stages done": do step 7 of the Runs 1-2 prompt below (write-up), rerun
   analyse_slots.py with the trained latin_p8, latin_p6 and latin_p5 numbers, and update the win lines in
   docs/TEXT_SLOTS.md. Stage C runs only if TEXT_SLOTS.md's rule says so (the text arms lost) and gates A1 and A2
   passed; otherwise it stays held.

Stop and ask if the smoke fails, a gate fails, the merge conflicts, or a result would change the plan. Report
failures as plainly as successes. No pull requests, no Claude API calls, no model IDs in commits.
```

## 3090 session state: Runs 1-2 (in progress since 2026-11-09; now step 6 of the prompt above)
Steps 0-2 of the prompt below are done (PLAN_3090 "Results (RTX 3090 ...)": S = 6000, 3.93 s/step, 17.9 GB;
amendment 7: page counts). Steps 3-6 run under `encoder_probe/experiments/run12_driver.py`, which records finished
steps in `results/r12_logs/done.txt` and resumes after a power-off (the PC is on in 6-7 h stretches). After a
restart, in Git Bash:
```bash
cd /c/git/visual-bpe-run && . .venv/env.sh && git pull --rebase origin claude/validation-harness
cat results/r12_logs/driver.log | tail; ls results/r12_logs/STOP 2>/dev/null   # a STOP file = a failed check: read it first
nohup python encoder_probe/experiments/run12_driver.py --steps 6000 --stages A B chan >> results/r12_logs/driver.out 2>&1 &
```
Then do step 7 (write-up) when the driver logs "all requested stages done". Stage C is held (2026-11-11): it was
`--stages A chan B C` before the reframing.

## Earlier prompt, still the reference for the driver's steps: PLAN_3090 Runs 1-2 (amendment 4: stage A learnability, B Latin, C dense glyphs; amendment 5: the bar; amendment 6: channel capacity)
```text
Pull branch claude/validation-harness (git pull --rebase origin claude/validation-harness). Read AGENTS.md
(gotchas, especially GPU runners), docs/PLAN_3090.md (the ladder, amendments 1-6 and the Results sections;
amendment 4 sets the budget, the arm order and the gates; amendment 5 which number the > 20 bar reads;
amendment 6 the channel-capacity diagnostic) and docs/LOCAL_GPU.md section 8 (Windows RTX 3090:
Git Bash, `. .venv/env.sh`, `python` not `python3`, PYTHONUTF8=1, EP_TRAIN_R2_CAP=400000000, one finetune job at a
time; Windows pages VRAM instead of raising OOM, so watch progress and checkpoint mtimes). Everything below is
pre-registered: do not change it after seeing data; if something must change, add a dated amendment to PLAN_3090.md
first. Never commit adapters (encoder_probe/cache is git-ignored).

0. Setup and tests: python -m pytest -q encoder_probe/tests/test_capacity_curve.py encoder_probe/tests/test_colour.py
   encoder_probe/tests/test_round3.py   (tiny model, CPU; test_rgb3_unsupported_pitch may fail on Windows: no raqm,
   known). Make sure the Qwen2.5-VL-3B weights are cached (LOCAL_GPU.md section 8 one-liner).
1. Frozen baseline (inference only, minutes):
   python encoder_probe/experiments/capacity_curve.py --device cuda --encodings latin_p8 latin_p6 latin_p5
     lg_n0_pb_c4_h7 lg_n0_pb_c7_h7 --texts natural shuffled --pages 200 --shifts 7 --cer-pages 4 --run-name r12_frozen
   Expect latin_p8 near the CPU Run 0 (12.6 natural / 25.7 shuffled bits per vision token, maybe ~10% higher with
   the margin) and the glyph encodings near 0. If latin_p8 is far off, stop and report.
2. Smoke: set
   R="--lora-r 0 --vision-blocks 16 --lr-vision 2e-5 --task-mix 1.0 --text shuffled --top-margin 7
      --train-pages 12000 --test-pages 16 --batch-size 8 --max-target-chars 2000 --eval-conds clean --device cuda"
   python encoder_probe/experiments/finetune_qwen.py --encoding latin_p8 $R --smoke
   Check: loss and val_loss fall; peak VRAM (nvidia-smi) < ~21 GB, else add --micro-batch 4 (same effective
   batch); note s/step and RAM (12000 pages ~7 GB). S = 6000 steps unless one run would exceed 12 h; then the
   largest multiple of 500 that fits, not below 3000. Write S, s/step and VRAM into PLAN_3090 Results before the
   first arm.
3. For every arm <enc> below, one job at a time, in order:
     python encoder_probe/experiments/finetune_qwen.py --encoding <enc> $R --steps S --run-name r12
     python encoder_probe/experiments/finetune_qwen.py --encoding <enc> $R --steps S --run-name r12 --blind-images
     python encoder_probe/experiments/capacity_curve.py --device cuda --encodings <enc> --texts natural shuffled
       --pages 200 --shifts 7 --cer-pages 4 --adapter encoder_probe/cache/r3_adapters/r12__<enc>__S.pt
       --run-name r12_cap_trained
     python encoder_probe/experiments/capacity_curve.py ... same ... --adapter
       encoder_probe/cache/r3_adapters/r12_blind__<enc>__S.pt --run-name r12_cap_blind
     python encoder_probe/experiments/capacity_curve.py --device cuda --encodings <enc> --texts natural shuffled
       --pages 64 --shifts 7 14 21 28 --shift-dx 7 --shift-margin 28 --arms real blank --cer-pages 0
       --adapter encoder_probe/cache/r3_adapters/r12__<enc>__S.pt --run-name r12_sweep   (trained adapter only)
     python encoder_probe/experiments/analyse_run12.py --frozen encoder_probe/results/r12_frozen
       --trained encoder_probe/results/r12_cap_trained --blind encoder_probe/results/r12_cap_blind
       --sweep encoder_probe/results/r12_sweep --chan encoder_probe/results/r12_chan
   Then commit and push encoder_probe/results/r12* and encoder_probe/results/analysis_run12.
   Sanity after every arm (amendment 3): the blind adapter's bits per token on shuffled text <= 1.0, and the trained
   arm is not below its frozen point by more than its CI. If either fails, stop and report.
4. Stage A, in order: latin_p8, lg_n0_pb_c4_h7, lg_n0_pb_c7_h7. Then read gates A1 and A2 in
   analysis_run12/summary.md and report them (send a short status message if you can).
4b. Channel capacity (amendment 6), whether or not A1/A2 passed: random bits as colour blocks, no blind needed.
   python encoder_probe/experiments/channel_capacity.py --cell 4 --smoke --device cuda
   (200 steps: check that the loss falls below 2.079 nats/digit, peak VRAM < ~21 GB else add --micro-batch 4,
   note s/step). Then, one at a time:
     for c in 7 4 2; do python encoder_probe/experiments/channel_capacity.py --cell $c --steps S --device cuda
       --run-name r12_chan; done
   Re-run analyse_run12 (with --chan), commit and push encoder_probe/results/r12_chan. Sanity: every arm's
   swapped-image upper CI <= 0.5 bits/token, else stop and report.
5. Stage B: latin_p6, latin_p5 (Run 1 verdict).
6. Stage C, only if A1 and A2 both PASS and the banks lg_n0_pb_c2_h7md / lg_pb_c2t4_h7md exist on the branch
   (they are on the branch since 2026-11-07): lg_n0_pb_c2_h7md, then lg_pb_c2t4_h7md (Run 2 verdict).
   If A1 or A2 fails, skip stage C: the glyph line stops per amendment 4.
7. Write the results into docs/PLAN_3090.md (Results: a "Runs 1-2" subsection with analyse_run12's tables, the gates
   and verdicts vs amendments 3-4, the > 20 recovered-chars bar table with its amendment-5 status (PASS,
   placement-sensitive pass, below bar), the channel-capacity table and its readout (amendment 6), s/step, peak
   VRAM, S, problems) and one line in
   AGENTS.md "Findings". Commit, pull --rebase, push.
Report: the analyse_run12 summary.md tables (incl. the bar table), gates A1/A2, the Run 1 and Run 2 verdicts, the sanity checks, timings
and problems.
```

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

## Earlier direction (2026-11-06, parked): gist through colour channels
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

**2026-11-06: superseded by the user's 3090 plan ([PLAN_3090.md](PLAN_3090.md)); the colour A1 prompt below is parked (colour returns as Run 4).** So the GPU A1 trains the ViT from block 16 (amendment 3); the prompt below runs it. Remaining CPU options are
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
**As of 2026-11-11:** (1) Test B and then the held-back image arms (the prompt at the top); (2) Test A's follow-ups
in its README; (3) after a Test B win, the use test: needles and chains from slots (`longctx_bench.py`) against
tokens and page images at equal context positions. The list below is the earlier one; all of it is parked.

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

## Prompt for the next GPU session (colour A1: parked 2026-11-07, superseded by the 3090 prompt above)
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
