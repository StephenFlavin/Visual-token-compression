# GPU runner session (2026-11-05, 10:59-13:59 UTC, 3 h, docs/GPU_RUNNER.md sections 0-5)

**Hardware:** NVIDIA GeForce RTX 4090 24 GB (bf16 supported), 32 CPU cores, 124 GB RAM, Linux, torch 2.8.0+cu128,
transformers 5.22.1, peft 0.21.2, Pillow with raqm. Full Qwen2.5-VL-3B weights loaded (3,754,622,976 parameters).

**Setup/smoke:** CPU tests green (137 passed, 1 skipped). `finetune_qwen.py --smoke` (latin_p8) **ran as built: no
code fixes were needed.** Measured: **1.21-1.22 s/step** (batch 4, grad checkpointing, bf16), **peak allocated
11.4 GB** (nvidia-smi ~12.3 GB), model load ~1 min, eval ~2.3 min per cond (64 windows + 64 cloze + 32 MC). Page
pre-render (1000 train pages): latin_p8 3 min, latin_p5 8 min, moonshot 11 min (parallel CPU jobs).

**Sizing:** K = floor(0.85 x 24 / 11.4) = 1 job at a time. The section-3 formula gave 1250 steps, but it ignores the
~7 min/job load+eval overhead (5 jobs), so **STEPS = 1000** for every job (~25 min/job). One fair_hash for all rows:
`05dc0c9e8507`. Rows: `encoder_probe/results/round3_ft_gpu3h/` and `round3_ft_gpu3h_blind/`; logs
`results/queue_logs/gpu3h_1/`.

**Results: R3-1 short-budget (1000 steps, 1000 train pages, 16 test pages; 5000+ steps was the plan).**
| arm | chars/token | cond | CER | blind CER | cloze | blind cloze | MC | blind MC | final val loss (blind) |
|---|---|---|---|---|---|---|---|---|---|
| latin_p8 | 28.1 | clean | **55.0%** | 88.5% | 0.52 | 0.22 | 0.88 | 0.25 | 1.125 (2.48) |
| latin_p8 | 28.1 | jpeg75 | 57.5% | 86.3% | 0.48 | 0.23 | 0.91 | 0.25 | |
| rgb3_lg_pb_h5 (moonshot) | 321.6 | clean | **83.6%** | 89.9% | 0.22 | 0.22 | 0.34 | 0.28 | 2.518 (2.517) |
| rgb3_lg_pb_h5 (moonshot) | 321.6 | jpeg75 | 87.6% | 88.6% | 0.19 | 0.23 | 0.38 | 0.31 | |
| latin_p5 | 72.4 | clean | 80.9% | (none; latin_p8's 88.5%) | 0.31 | (0.22) | 0.56 | (0.25) | 2.163 (no blind run) |
| latin_p5 | 72.4 | jpeg75 | 79.0% | | 0.33 | | 0.53 | | |
| lg_pb_h5 | 107.3 | clean | 78.4% | (none) | 0.17 | | 0.31 | | 2.274 (no blind run) |
| lg_pb_h5 | 107.3 | jpeg75 | 78.5% | | 0.17 | | 0.34 | | |

* **latin_p5 reads a little:** it has no blind control of its own (5 jobs fitted, a 6th didn't). Against latin_p8's blind floor
  (same text targets) its cloze/MC (0.31/0.56 vs 0.22/0.25) and val loss (2.16, still falling, vs ~2.48-2.52 for
  both blind runs) are above chance, but most generations are invented text. Read it as "starting to read".

* **latin_p8 is read:** far below its blind floor on CER, cloze and MC, and val loss 1.13 vs 2.48 blind. Its CER is
  high because it transcribes the first line(s) of a window nearly exactly, then drifts into fluent invented
  continuation (samples in the rows); the loss was still falling at 1000 steps.
* **The moonshot is not read at this budget:** CER, cloze and MC sit at the blind floor; train/val loss stayed flat
  (~2.5) and is identical to the blind run's to 3 decimals; real and blind generations are the same generic text
  ("in the 1970s, and the 1980s. The first was the 1973 film ...") whatever the page.
* The op split (sub/ins/del) and hyp/ref length ratio are not in `finetune_qwen.py` rows (only CER + 4 samples), so
  they were not reported; add them to `run_eval` before the next run.

**Against ROUND3.md gate G3** (best code's chars/token at < 3% CER >= 1.5x the best Latin's at < 3%): **not
decidable at this budget, no arm reaches 3% CER.** Short-budget signal: Latin learns to read within 1000 steps; the
moonshot shows no sign of reading (at its blind floor), so on this evidence it is the weaker candidate. A code at
its floor after 1000 steps is undecided, not failed, but its loss showed no movement at all, unlike Latin's.

**Extra (user-approved, end of session):** lg_pb_h5 (single-channel learned glyph bank, ~107 chars/token) at the
same settings and fair_hash, started 13:32 while the write-up ran, to ask whether the moonshot fails because of
density (322) or because of the glyph code itself. It finished (rows above, no blind run): its val loss fell below
the ~2.5 blind floors while the moonshot's did not, and its CER is a few points under latin_p5's, but cloze/MC are at
chance. A weak hint that density, not the glyph code alone, is what stalls the moonshot; it needs its own blind
control and longer training before it means anything.

**What did not run:** 5000-step arms; latin_p5 blind control; latin_p6, rgb3_p8; R3-2; the optional frozen-Qwen probe (R3-0);
G2 at scale (section 6). Adapters (`encoder_probe/cache/r3_adapters/`, ~0.56 GB each) were not copied off the
machine. Nothing was chosen on test-page CER.

**Suggested next:** the 5000-step arms on the 3090 (overnight), with latin_p8 first to see where its drift error
goes; for the moonshot, check whether an easier intermediate density (lg_pb_h5 ~107, rgb3_p8 ~73) moves off its
blind floor before spending a long run on 322 chars/token.

---

# Handoff: cloud CPU runner -> any GPU runner (2026-11-05, morning)

**For a GPU session of any hardware, start with [GPU_RUNNER.md](GPU_RUNNER.md)** (a time-boxed plan; measure, then
size). The Windows/3090 runbook is [LOCAL_GPU.md](LOCAL_GPU.md). Read AGENTS.md first: it has the round-3 gotchas
(merger LR collapse, probe bias, disk and memory limits). Design, pre-registrations and amendments are in
[encoder_probe/ROUND3.md](../encoder_probe/ROUND3.md); the evidence table is in [CONTEXT_BUDGET.md](CONTEXT_BUDGET.md).

**State of the evidence for the moonshot (`rgb3_lg_pb_h5`, 322 chars per Qwen token):**
| Gate | Result | Where |
|---|---|---|
| G0: raw-pixel ceiling (identity28, 200 pages, CPU) | **1.07% CER**, passes the 3% go bar | `encoder_probe/results/round3_cpu_tp200/` |
| Q1: linear pixel readout of the frozen Qwen ViT | glyph codes stay pixel-like to the last block, no channel collapse; everything collapses linearly at the merger | `encoder_probe/results/quick_q1/summary.md` |
| G2-lite: frozen / trained / no merger (500 pages, 3000 steps, CPU) | in progress at hand-off; the no-merger arm leads by ~14 points at equal steps, no arm near usable | `encoder_probe/results/g2lite/` |
| G3 / R3-1: fine-tune Qwen, LLM as reader | **not run: the next step, needs a GPU** | GPU_RUNNER.md |

**What went wrong overnight, and the fixes (details in ROUND3.md amendments):**
* **The trained-merger arm collapsed at the shared LR 2e-3.** Amendment 2 sets merger LR 1e-4, chosen on collapse stats,
  never on test CER. The as-registered latin_p8 trained row (77.8%) is kept and labelled.
* **GridProbe can't read Latin through Qwen at CPU-scale data** (latin_p16: 75%; cross-check 74.5%). So the
  comparative "1.5x vs pushed Latin" rule can't be applied with the probe; it moves to R3-1.
* **Amendment 1 (user):** sweep Latin to ~100 chars/token in G2/G3 (p8 ~28, p6 ~51, p5 ~72, p4 ~110 after its ceiling).
* **Ops:** 2 container restarts, a full disk, 2 OOM kills. All recovered from checkpoints; no finished row was lost.

**GPU decision (user, 2026-11-05):** the only rentable options were a Colab T4 (Turing: no bf16, 16 GB) and a TPU
v5e-1 (needs a torch_xla port). Neither runs R3-1 as built, so **R3-1 waits for the user's RTX 3090** (docs/LOCAL_GPU.md
section 8, or docs/GPU_RUNNER.md, which works unchanged on any Ampere+ GPU). The CPU G2-lite runs finish here,
including the blind floors.

**Prompt for the next GPU session** (the user's RTX 3090, or any Ampere+ GPU). Paste into Claude Code in the repo:
```text
Pull branch claude/validation-harness. Read AGENTS.md, the top section of docs/HANDOFF.md, and in
encoder_probe/ROUND3.md: R3-1, R3-1b, both amendments and the G2-lite notes/results. Then the runbook for this
machine: docs/LOCAL_GPU.md section 8 on the Windows RTX 3090 PC, otherwise docs/GPU_RUNNER.md (measure
s/step and peak memory in the smoke, then size the runs).

Goal: R3-1 (gate G3). LoRA fine-tune Qwen2.5-VL-3B so its own LLM reads each code.
1. Setup, tests, then the smokes. The fine-tune and benchmark have never run on a GPU or with real weights:
   fix small bugs minimally, keep the CPU tests green, commit each fix.
2. Arms at identical settings (one run name, one fair_hash), 5000 steps each:
   latin_p8, latin_p5, latin_p6 (the Latin baseline pushed toward ~100 chars/token, per amendment 1),
   rgb3_lg_pb_h5 (moonshot), lg_pb_h5, rgb3_p8. Order: latin_p8, moonshot, latin_p5, then the rest.
   Pre-render pages on the CPU first. Run long jobs with nohup and checkpoints; push rows often.
3. Blind control: run finetune_qwen.py --blind-images (built and CPU-tested: each page's image swapped for
   another page's, text targets unchanged, same fair_hash, run name gets _blind) for latin_p8 and the moonshot at
   the same settings. Report every CER next to its blind floor, with the substitution/insertion/deletion split and the
   hypothesis/reference length ratio. A score near the blind floor means the model is not reading.
4. Apply ROUND3.md: G3 asks whether the best code's density at < 3% CER is >= 1.5x the densest Latin reaching
   the same CER under the same budget. Never choose anything on test-page CER.
5. Run R3-2 (long-context benchmark) only for arms that pass G3; otherwise stop and report.

Stop and ask me if the smokes can't be made to run, if fine-tuned Latin stays near its blind floor, or if a
result would change the plan. Record any design change as a dated amendment in ROUND3.md before running it. At
the end, write up round 3 (encoder_probe/RESULTS.md, docs/RESULTS.md, the AGENTS.md findings bullet,
docs/CONTEXT_BUDGET.md evidence table) and give me a results table. No PRs, no Claude API calls, no model IDs
in commits.
```
On the 3090, R3-1 is roughly 15-25 h in total (an estimate: 2-4 h per 5000-step arm), so run it overnight.

**Open items, in priority order:**
1. R3-1 short arms on a GPU: latin_p8, moonshot, latin_p5 at equal steps (GPU_RUNNER.md).
2. G2-lite on a GPU at 1000+ pages. `run_g2lite.py` is CPU-only: port it with `--device`, or use `run_probe2.py
   --encoder qwen25vl` for the frozen arm.
3. Latin p6/p4 points; lg_pb_h5 arms; moonshot levers (grid-aligned 4/7 px pitch, merger redesign): ROUND3.md.
4. R3-2 long-context benchmark, once a fine-tuned code passes G3.

---

# Handoff: cloud runner -> local RTX 3090 (2026-11-04, evening)

The cloud runner is idle; nothing runs there. Everything below is on `claude/validation-harness`.

**Since the local handoff (section below), the cloud did:**
* Cross-check closed: latin_p32 deepencoder/clean 200 / 1000 / 2000 pages = 89.0% / 27.6% / 24.9% (CPU) vs local
  85.4% / 25.8% / 25.3% (3000 pages). CPU/GPU and Linux/Windows rendering agree.
* **Round-3 code built, CPU-tested (117 tests incl. 14 in test_round3.py on a tiny random-init Qwen2.5-VL), never run
  on GPU or with real weights:** `ep/qwen_ft.py`, `ep/r3_tasks.py`, `experiments/finetune_qwen.py` (R3-1),
  `experiments/longctx_bench.py` (R3-2). Runbook: docs/LOCAL_GPU.md section 8. Deviation from ROUND3.md:
  transcription targets are windows of whole lines up to `--max-target-chars 2000` (same for every arm; dense pages
  hold 7k-82k chars). KV verified: 36,864 B/position (36 layers x 2 KV heads x 128 x bf16); 32k window = 1.21 GB.
* **Moonshot encodings built:** `rgb3_lg_pb_h5` (321.6 chars/Qwen token) and `rgb3_lg_free_h4` (405.5), in
  `ep/rgb_overlay.py` (notes in ROUND3.md). Their identity28 ceilings are only partial: at step 500/3000, 15.2% and
  37.0% val CER (CPU, 7.8 s/step, stopped). Re-run them locally (cloud checkpoints were deleted: not portable).

**Do next, in order (commands in docs/LOCAL_GPU.md section 8):**
1. Pull; `pip install "peft>=0.17"`; `python -m pytest -q encoder_probe/tests`; fetch the full Qwen2.5-VL-3B weights
   (~7.5 GB; the load check prints 3754622976 parameters).
2. Smokes, one at a time: frozen qwen25vl probe, `finetune_qwen.py --encoding latin_p8 --smoke`, `longctx_bench.py` at
   10k chars. Expect small GPU-only bugs: fix minimally, keep the CPU tests green, commit each fix.
3. Moonshot ceilings (the round-2 guard: < 10% CER, else the combined page is unreadable):
   `python encoder_probe/experiments/run_probe2.py --device cuda --encoding E --encoder identity28 --cond C --steps 3000 --run-name round3_qwen_tp200`
   for E in rgb3_lg_pb_h5 rgb3_lg_free_h4, C in clean jpeg75.
4. R3-0 (frozen Qwen probe grid) -> pick codes; R3-1 fine-tuning arms (latin_p5, latin_p8, R3-0 picks, both
   `rgb3_lg_*`; all `round3_ft` rows must share one `fair_hash`); R3-2 long-context benchmark. Decision rules and
   predictions: ROUND3.md.
   After each step, update the Evidence table in docs/CONTEXT_BUDGET.md (context/KV/compute per encoding vs
   the default model; user's demo goal).
5. Write up in encoder_probe/RESULTS.md (round-3 section), AGENTS.md findings bullet, docs/RESULTS.md. Commit the
   `round3_*` runs.jsonl and push (git pull --rebase first).

**Limits:** at most 2 Qwen-token GPU jobs; finetune/bench jobs alone on the GPU (Windows pages VRAM and stalls
silently: watch checkpoint mtimes). Qwen token caches ~30 GB disk. VRAM/time figures in the runbook are estimates.

---

# Handoff: local RTX 3090 session -> cloud runner (2026-11-04)

Branch `claude/validation-harness`. Read AGENTS.md first (it has the round-2 findings bullet and new Windows/GPU
gotchas), then encoder_probe/ROUND3.md (the next plan). This file says what the local GPU session did and what is
open.

## Done on the local PC (Windows 11, RTX 3090, Ryzen 9 5900X, 32 GB)
* **Round 2 finished for DeepEncoder.** Write-up: encoder_probe/RESULTS.md (round-2 section), AGENTS.md findings
  bullet, docs/RESULTS.md. Rows: `encoder_probe/results/round2_gpu_tp{200,1000,3000,5000}/runs.jsonl`; job logs in
  `results/queue_logs/gpu_p0*`, `gpu_p1*` (force-added). Analyse with
  `python3 encoder_probe/experiments/analyse2.py encoder_probe/results/round2_gpu_tp1000/runs.jsonl encoder_probe/results/round2_gpu_tp200/runs.jsonl`
  (pass files explicitly: the cloud `round2/` 200-page rows collide with `round2_gpu_tp200`).
* **P0 (data scaling, latin through deepencoder/clean):** latin_p32 (8.5 chars/token) 85.4% (200 pages) -> 18.9%
  (1000) -> 17.2% (3000) -> 14.9% (5000) at 10k steps; latin_p16 (36 chars/token) ~74% at every page count and
  cannot fit its training pages (train CTC ~2.8). N = 1000 for P1.
* **P1 (1000 pages):** identity32 ceilings all < 3% (guard passes). Through DeepEncoder only latin_p32 reads (18.9%
  clean, 19.1% jpeg75); latin_p24 and denser Latin/mono 72-77%, col_b2/b4 71-73%, rgb3 83.5%, lg_free_h4 82.2% vs
  lg_pb_h5 82.4%. Verdict: colour pixels and RGB overlay dead for frozen DeepEncoder; grey-level banks not supported
  (stay whole-pixel). No family reaches 10%, so the decision rules hold only in weaker form. P2 (RGB overlay grid on
  SigLIP) and P3 (rgb6) were gated off and not run. Clean rows other than latin_p32/p16/rgb3_p8 ran 5k steps (time).
* **Code fixes pushed:** portable file lock (`ep/flock.py`, msvcrt on Windows), `phase2_driver.py` imports on Windows
  (so `job_queue.py` runs), DejaVu fallback to `assets/fonts` in `vbpe/config.py`, VLM encoders in fp32 on CUDA with
  TF32 off (CUDA bf16 gave corr 0.9974 vs CPU fp32). The cloud session's own CUDA support (`--device`, page cache,
  `--pages-only`) worked first time on the GPU. None of this changes the Linux CPU path.

## Caveats for comparing rows
* Local rows were rendered on Windows without raqm (Pillow hinted layout): some Latin pitches differ slightly from
  the runner (latin_p32 483.2 vs 479.5 px²/char; test-page digests differ). Compare local rows with local rows; the
  200-page cross-check still agrees (latin_p32 85.4% local vs 89.0% cloud, both memorising).
* `test_rgb3_unsupported_pitch` fails on Windows only for that reason; it should pass on the runner.

## Open items
1. **Cloud cross-check, 1000 pages: agrees.** `round2_tp/` (cloud CPU, fp32 probe, Linux renderer): latin_p32
   deepencoder/clean 1000 pages 3k steps = 27.6% vs 25.8% local (GPU, bf16 probe, Windows renderer). Still pending:
   `round2_tp2000/` (2000 pages); compare with local 3000-page 3k-step 25.3%.
   **Closed 2026-11-04 ~17:20: agrees.** Cloud 2000 pages, 3k steps = 24.9% (train CTC 0.59; curve 36.0% -> 25.6% ->
   23.5% at steps 2000/2500/3000). Cloud CPU rows: 89.0% (200) -> 27.6% (1000) -> 24.9% (2000); local GPU 3k-step: 25.8%
   (1000), 25.3% (3000). Cloud queue is now empty.
2. **Round 3** (encoder_probe/ROUND3.md): fine-tune Qwen2.5-VL-3B to read dense codes and benchmark KV-cache savings
   on long-context QA. Order: R3-0 frozen Qwen probe (code exists; CUDA path for qwen25vl never run) -> R3-1 LoRA
   fine-tune arms (Latin baseline, best single codes, and the moonshot RGB-overlay-of-learned-glyphs arm, not built
   yet) -> R3-2 long-context benchmark. R3-1/R3-2 need new code (`finetune_qwen.py`, `longctx_bench.py`) and are
   GPU work: on the 4-CPU runner, build and test the code (and the `rgb3_lg_*` encodings with their identity28
   ceilings, which are CPU-feasible at small page counts), and leave the GPU runs for the local PC.
3. Optional GPU extras, low priority: latin_p24 at 3000-5000 pages (does the 8.5 -> 15 chars/token cliff move?),
   clean rows at 10k steps, P2 for completeness.

## Local PC state (for the next local session)
* Clone `C:\git\visual-bpe-run` (sparse checkout: three `learned_glyphs/results/*/phase2/samples` dirs excluded,
  their files have `:` in names; `git update-index --skip-worktree` on them after each pull). `C:\git\visual-bpe` is
  the broken original checkout (every file staged as deleted by the failed Windows checkout): leave it or delete it.
* venv `.venv` (Python 3.12, torch 2.11 cu128, transformers 5.22.1); env: `PYTHONUTF8=1 EP_PAGE_CACHE=1
  EP_TRAIN_R2_CAP=400000000`, PATH with `.venv/Scripts` first; queue lines use `python`. Helper:
  `. .venv/env.sh` in Git Bash. Weights, page and token caches in `encoder_probe/cache/` (~25 GB, git-ignored).
* At most 3 GPU jobs (2 for Qwen-token jobs): with 5, VRAM filled and Windows paged GPU memory; all jobs stalled
  silently for 20 min. Watch checkpoint mtimes.
