# Running the experiments: every command

*Moved out of AGENTS.md on 2026-11-12; regrouped by area and re-checked on 2026-11-13. Setup is in
[AGENTS.md](../AGENTS.md); things that go wrong are in [GOTCHAS.md](GOTCHAS.md); where rows land is in
[results/README.md](../results/README.md).*

**Read this first.**

* The study is finished. **One experiment is ready and not yet run** (section 5); the rest are finished, parked or
  closed, and their commands are kept so a result can be reproduced or extended.
* Every script and flag named here was checked against the code after the restructure (`--help`, 2026-11-13).
  Nothing was re-run: commands marked **Claude** spend money or subscription quota (answers already paid for are
  read back from `results/cache/claude/`), and commands marked **GPU** need a CUDA machine.
* Status words: **finished** = ran, result written up; **parked** = stopped part-way, can be resumed;
  **closed** = will not be run; **never ran** = built and tested only.
* On the Windows RTX 3090 PC use `python`, not `python3` ([gpu/LOCAL_GPU.md](gpu/LOCAL_GPU.md)).

| Section | Area | Status |
|---|---|---|
| 1 | Phases 1-2: hand-designed glyphs against Latin | finished |
| 2 | Learned glyph alphabet | finished |
| 3 | Frozen-encoder probes (rounds 1-2) and Qwen2.5-VL round 3 | finished, except the long-context benchmark (never ran) |
| 4 | The 3090 plan: capacity curve, trained encoder, channel capacity | capacity curve finished; training parked |
| 5 | **Local Qwen3.8-27B: page images against text** | **ready, not run** |
| 6 | Other GPU lines: own encoder, text slots, colour | finished inconclusive / closed |
| 7 | Claude: page images against text on long documents | finished |
| 8 | Claude: retrieval, compaction, coding agents | finished (EnterpriseRAG-Bench: preparation only) |
| 9 | Claude: the repository atlas and plan quality | finished, closed |
| 10 | Utilities: job queue, analysis, figures, checks | |

## 1. Phases 1-2: glyph substitution against Latin (finished)

Write-up: [glyphs/RESULTS.md](glyphs/RESULTS.md). Phase 1 asks Claude to transcribe pages; Phase 1b uses an OCR
model; Phase 2 trains a small reader (a CRNN) per rendering scheme and font size (`em`).

| What | Command | Notes |
|---|---|---|
| Phase 1 pilot (**Claude**) | `python3 experiments/phases/phase1_sweep.py --pilot --max-cost 10`, then add `--yes` | prints a cost estimate first; the estimate is ~2x the real cost |
| Phase 1, Latin only (**Claude**) | `python3 experiments/phases/phase1_sweep.py --variant natural --n-passages 6 --schemes latin latin_condensed --budgets 14 17 20 23 27 32 40 --run-name latin_natural --max-cost 5 --yes` | ~$1.9 for 84 calls |
| Phase 1 dry run (free) | add `--dry-run [--fake-noise 0.03]` | a fake transcriber, no Claude |
| Phase 1b, OCR zero-shot | `python3 experiments/phases/phase1b_ocr.py --variant natural --n-passages 6` (and `--variant scrambled --n-passages 3`) | CPU, ~5 s/page, 2 workers; resumable |
| Phase 2 grid | `nohup python3 experiments/phases/phase2_driver.py > results/phase2_driver.log 2>&1 &` | one process per (scheme, em), largest em first, keeps 4 cores busy; per-job logs in `results/phase2/<run>/driver/`. `experiments/phases/phase2_main.sh` is the older 3-worker wrapper |
| Phase 2 single runs | `python3 experiments/phases/phase2_train.py --schemes latin --ems 5 8 --steps 3500 --parallel 3 --threads 1 --run-name X` | `--smoke` for a quick check |
| Merge Phase 2 rows | `python3 scripts/merge_phase2.py` | keeps the longest run per (scheme, em) in `results/phase2/merged/`, the rows to cite. **Do not pass `--help`** ([GOTCHAS.md](GOTCHAS.md)) |
| Analysis | `python3 -m vbpe.analysis <runs.jsonl> [more] --out DIR` | Phase 1, 1b and 2 rows; `summary.md` / `summary.json` and plots |

## 2. Learned glyph alphabet (finished)

Write-up: [learned_glyphs/RESULTS.md](../learned_glyphs/RESULTS.md); design
[learned_glyphs/PLAN.md](../learned_glyphs/PLAN.md).

| What | Command | Notes |
|---|---|---|
| Train a glyph bank with its reader | `python3 learned_glyphs/experiments/train_joint.py --dry-run` prints the configurations; `--grid e1` is the 2x2 grid | whole-pixel, patch-aligned banks: `--pixel-block --guard-ss 0 --height-px 7 --char-w 2 --token-w 4 --steps 10500`; add `--min-dist-px 2 --min-dist-weight 1 --tag-suffix md` for banks with no identical glyphs ([gpu/PLAN_3090.md](gpu/PLAN_3090.md)) |
| Calibrate the degradation channel | `python3 learned_glyphs/experiments/calibrate_channel.py` | |
| Judge a bank ("bridge") | `python3 learned_glyphs/experiments/transfer_eval.py --bridge --ems 5.8333 --steps 10500` | the frozen bank through the unchanged Phase-2 reader pipeline; this, not the joint error, is the number to quote |

## 3. Frozen-encoder probes and round 3 (finished; the long-context benchmark never ran)

Write-ups: [encoder_probe/RESULTS.md](../encoder_probe/RESULTS.md), [encoder_probe/ROUND3.md](../encoder_probe/ROUND3.md),
[background/ANALYSIS.md](background/ANALYSIS.md). A "probe" is a small reader trained on a frozen encoder's output
tokens. Every arm needs a blind control.

| What | Command | Notes |
|---|---|---|
| Round 1 (SigLIP), CPU | `python3 encoder_probe/experiments/run_probe.py --print-grid --steps 3000 --run-name main > results/queue_X.txt`, then the job queue (section 10); one run: `run_probe.py --encoding latin_p8 --encoder siglip_l6 --cond jpeg75` | ~20-30 min per run at 1 thread; tables: `encoder_probe/experiments/analyse.py` |
| Round 2 (DeepSeek-OCR's DeepEncoder), **GPU** | `python3 encoder_probe/experiments/run_probe2.py --device cuda --encoding latin_p32 --encoder deepencoder --cond clean --train-pages 1000 --run-name round2_gpu_tp1000` | runbook [gpu/LOCAL_GPU.md](gpu/LOCAL_GPU.md) sections 3-7 (`EP_PAGE_CACHE=1`); `--pages-only` pre-renders; tables: `analyse2.py <runs.jsonl files>` |
| Round 3, probe on Qwen's vision states (G2-lite), CPU | `python3 encoder_probe/experiments/run_g2lite.py --print-queue` | arms frozen / trained / direct (no merger) and blind; `--merger-lr 1e-4`; tables: `analyse_g2lite.py`; rows `encoder_probe/results/g2lite*/` |
| Round 3, fine-tune Qwen2.5-VL-3B (R3-1), **GPU** | `python3 encoder_probe/experiments/finetune_qwen.py --encoding latin_p8 --smoke`, then `--steps N --run-name R`, and the same with `--blind-images` (rows go to `R_blind`) | runbooks [gpu/GPU_RUNNER.md](gpu/GPU_RUNNER.md), [gpu/LOCAL_GPU.md](gpu/LOCAL_GPU.md) section 8. Ran once: RTX 4090, 1000 steps |
| R3-1 against its blinds | `python3 encoder_probe/experiments/analyse_r3.py` | exact tests, loss gain, bits per vision token -> `encoder_probe/results/analysis_r3/summary.md` |
| Round 3, long-context questions (R3-2), **GPU** | `python3 encoder_probe/experiments/longctx_bench.py --inputs text latin_p8 --lengths 10000 --tasks needle cloze [--adapter <file>.pt]` | **never ran** |

## 4. The 3090 plan: capacity curve and the trained encoder (curve finished; training parked)

Plan, amendments and results: [gpu/PLAN_3090.md](gpu/PLAN_3090.md). What resuming takes: [HANDOFF.md](HANDOFF.md).

| What | Command | Notes |
|---|---|---|
| Capacity curve of the frozen model (Run 0), CPU or GPU. **Finished** | `python3 encoder_probe/experiments/capacity_curve.py --encodings latin_p16 latin_p12 latin_p10 latin_p8 latin_p6 latin_p5 --texts natural shuffled --pages 32 --run-name run0_cpu` | bits recovered per vision token; needs the full Qwen2.5-VL-3B weights (7.5 GB); ~2.6 h on a CPU with AMX |
| Placement on the patch grid (Run 0b). **Finished** | `capacity_curve.py --encodings latin_p6 --texts natural --pages 16 --shifts 7 14 21 28 --shift-margin 28 --arms real blank --cer-pages 0` | every (dx, dy) of the listed shifts; `--shift-dx 7` holds dx fixed |
| Train the vision encoder, language model frozen (Runs 1-2), **GPU. Parked** | `python encoder_probe/experiments/run12_driver.py --steps 6000 --stages A chan B` | resumable driver for the pre-registered commands; each arm is `finetune_qwen.py --encoding <enc> --lora-r 0 --vision-blocks 16 --lr-vision 2e-5 --task-mix 1.0 --text shuffled --top-margin 7 --train-pages 12000 --batch-size 8 ...`, then its `--blind-images` run, then `capacity_curve.py --adapter <adapter> --pages 200 --shifts 7 --allow-fewer` |
| Channel capacity: random bits as colour blocks, **GPU. Parked** | `python encoder_probe/experiments/channel_capacity.py --cell 4 --smoke --device cuda`, then `--cell {7,4,2} --steps 6000 --device cuda --run-name r12_chan` | 96 / 294 / 1176 bits offered per token; no blind needed; on a CPU pass `--bf16 on` |
| Verdicts for Runs 1-2 | `python encoder_probe/experiments/analyse_run12.py --frozen encoder_probe/results/r12_frozen --trained encoder_probe/results/r12_cap_trained --blind encoder_probe/results/r12_cap_blind --sweep encoder_probe/results/r12_sweep --chan encoder_probe/results/r12_chan` | only `r12_frozen` and `r12_smoke` exist in the repo |

## 5. Local Qwen3.8-27B: page images against text (ready, not run)

**This is the one run left.** Paste the prompt at the top of [HANDOFF.md](HANDOFF.md) into a session on the 3090
PC. Plan and predictions: [gpu/LOCAL_QWEN.md](gpu/LOCAL_QWEN.md). No training; the model is served by llama.cpp's
`llama-server` and the harness talks to it over HTTP.

| Step | Command |
|---|---|
| Tests against a mock server | `python -m pytest -q tests/test_local_qa.py` |
| Sizes only, no server | `python experiments/claude_qa/local_qa.py run --dry-run --doc 0 1 2` |
| Start the server | `llama-server -m models/qwen38/Qwen3.8-27B-UD-Q4_K_M.gguf --mmproj models/qwen38/mmproj-F16.gguf -ngl 99 -c 57344 -np 1 --image-max-tokens 4096 --port 8080` (flag names vary by build: check `--help`) |
| Thinking off | `X='--extra {"chat_template_kwargs":{"enable_thinking":false}}'` |
| Run the arms | `python experiments/claude_qa/local_qa.py run --doc 0 $X`, check, then `run --doc 1 2 $X` |
| Generation speed | `python experiments/claude_qa/local_qa.py speed --label plain $X`, and `--label mtp` with the server restarted with its multi-token module |
| Table and verdicts | `python experiments/claude_qa/local_qa.py analyse` -> `results/local_qa/qwen38_27b/summary.md` |

Other options: `--arms none text page118 page88 page59`, `--base-url`, `--run-name`, `--page-px` (a multiple of
32), `--questions-per-call`, `--max-tokens`.

## 6. Other GPU lines (finished inconclusive, or closed)

| What | Command | Status and notes |
|---|---|---|
| Text-side compressor into a frozen language model ("Stage 0") | `python3 encoder_probe/experiments/text_compress.py --ratio 16 --smoke`, then `--ratio R --steps 3000 --run-name stage0`; table: `analyse_stage0.py` | **Finished, inconclusive** (ratios 4 and 16 ran). [gpu/OWN_ENCODER.md](gpu/OWN_ENCODER.md); tests `encoder_probe/tests/test_text_compress.py` |
| Test B: text slots against page pixels | `python encoder_probe/experiments/text_slots.py --ratio 1 --smoke`, then the ladder in [gpu/TEXT_SLOTS.md](gpu/TEXT_SLOTS.md); `analyse_slots.py encoder_probe/results/tb --image name:chars_per_token:bits_natural:bits_shuffled` | **Closed, never ran on a GPU.** Tests `encoder_probe/tests/test_text_slots.py` |
| Colour layers, untrained model (A0) | `python3 encoder_probe/experiments/zeroshot_colour.py --pages 12 --transcribe-pages 3` | **Finished** (CPU, ~6 min with AMX). [gpu/COLOUR_GIST.md](gpu/COLOUR_GIST.md) |
| Colour layers, fine-tuned (A1/A2) | `finetune_qwen.py --encoding rgb{1,2,3}s_p16 --steps 1000 --eval-mc 4 --eval-cloze 4 --val-samples 64 --run-name colour_a1` (and `--blind-images`), then `analyse_colour.py --real colour_a1` | CPU version **finished** (missed its bar); the GPU version is **closed** |

## 7. Claude: page images against text on long documents (finished)

Write-up: [claude/CLAUDE_QA.md](claude/CLAUDE_QA.md). All of these call **Claude**; `--dry-run` builds pages and
questions without calls.

| What | Command | Notes |
|---|---|---|
| The main test | `python3 experiments/claude_qa/claude_qa_render.py --chars 500000 --dry-run`, then `--yes` | options: `--budget` (px² per character), `--arms`, `--n-complex`; pooled analysis: `analyse_claude_qa.py <run dirs> [--out DIR]` |
| Deep-research sources | `python3 experiments/claude_qa/claude_qa_research.py --dry-run`, `--yes`, `--analyse` | 3 corpora x 5 arms |
| Test A: pruned text against pages | `python3 experiments/claude_qa/claude_qa_prune.py build`, then its run and analysis steps ([results/claude_qa/prune](../results/claude_qa/prune/README.md)) | LLMLingua-2 on CPU, ~7 min per document; $17.52; tests `tests/test_claude_qa_prune.py` |
| Repeat calls of Test A's arms | `python3 experiments/claude_qa/claude_qa_repeat.py check`, `locate`, `run`, `analyse` | rows `results/claude_qa/repeat/`; $16.17 |
| Vision checks | `python3 scripts/check_claude_vision.py` | token formula, legibility, latency; ~$0.5 |

## 8. Claude: retrieval, compaction, coding agents (finished)

| What | Command | Notes |
|---|---|---|
| Retrieval (RAG) with page images | `python3 experiments/rag/run_rag.py --stages A B C`, then `--analyse` | plan [claude/RAG_PLAN.md](claude/RAG_PLAN.md); result [results/rag](../results/rag/README.md) |
| EnterpriseRAG-Bench | setup in `experiments/erb/erb_common.py`; `build_bm25.py` (~4 min, 4 cores), then `run_erb.py --pilot [--dry-run]`, `judge.py`, `analyse_erb.py` | **preparation only**; plan [claude/ERB_PLAN.md](claude/ERB_PLAN.md) |
| Compaction with a page archive | `python3 experiments/compaction/run_compact.py --corpora 0 1 2` | [results/compaction](../results/compaction/README.md) |
| Compaction at scale, real hooks | `python3 experiments/compaction/run_scale.py --sessions research:0 code:0 logs:0` (pilot), `--sessions all`; `analyse_scale.py` | [results/compaction_scale](../results/compaction_scale/README.md) |
| Coding agents reading pages: smoke | `python3 experiments/readhook/smoke/run_smoke.py --arms text image --reps 1` | [experiments/readhook/README.md](../experiments/readhook/README.md) |
| Coding agents on SWE-bench | in `experiments/readhook/swe/`: `run_swe.py validate`, `run --arms text vbhook`, `report`; `run_ml.py select`, `all` (run with the harness's own virtualenv) | [claude/SWE_IMAGES.md](claude/SWE_IMAGES.md) |

## 9. Claude: the repository atlas and plan quality (finished, closed)

Plan and results: [claude/REPO_ATLAS_PLAN.md](claude/REPO_ATLAS_PLAN.md), [results/README.md](../results/README.md)
(one folder per test under `results/atlas/`). Runners are in `experiments/atlas/`; each takes sub-commands and a
spend cap (`--cap`).

| What | Command |
|---|---|
| Build an atlas of a repository (no Claude calls) | `python3 experiments/atlas/repo_atlas.py stats <repo dir> ...` (sizes), `build <repo dir> --out DIR` (pages, `index.json`, `index.txt`) |
| Pilot: where does the change go? | `experiments/atlas/repo_atlas_eval.py` (arms tree / map / atlas / text / agent) |
| Questions about a repository (SWE-QA) | `repo_atlas_qa.py prepare`, `estimate`, `run --arms closed map atlas text --cap 22`, `judge`, `analyse` |
| Finding a security flaw | `repo_atlas_sec.py select`, `prepare`, `run`, `analyse` |
| Hard coding tasks | `repo_atlas_swe.py inspect`, `select`, `run`, `analyse`; the repeat with auto-memory off: `repo_atlas_swe_rerun.py check`, `validate`, `order`, `run`, `analyse`; on Sonnet: `repo_atlas_swe_sonnet.py` |
| C to Rust migration | `repo_atlas_migrate.py prepare`, `run`, `report` |
| A whole-repository port (Go to Rust) | `repo_atlas_port.py prepare`, `reference`, `smoke`, `run --arms plain atlas`, `finish`, `grade`, `analyse` |
| Plan quality: briefs from the atlas | `python3 experiments/atlas/plan_quality.py select`, `prepare`, `estimate`, `plan`, `judge`, `work --cap N`, `extend`, `analyse` ([claude/PLAN_QUALITY_TEST.md](claude/PLAN_QUALITY_TEST.md); tests `tests/test_plan_quality.py`) |

## 10. Utilities

| What | Command | Notes |
|---|---|---|
| CPU job queue | one single-core command per line in `results/queue_<round>.txt`; `nohup python3 scripts/job_queue.py results/queue_<round>.txt --log-dir results/queue_logs/<round> [--after-phase2] [--cores N] > results/queue_<round>.log 2>&1 &` | backfills free cores; lines can be appended while it runs. **Restarting re-claims from line 1:** use a new queue file and a new `--log-dir` per round |
| Save rows while long jobs run | `nohup scripts/autosave_results.sh 600 &` | commits and pushes `runs.jsonl` files every 10 min |
| All tests | `OMP_NUM_THREADS=1 python3 -m pytest -q tests encoder_probe/tests learned_glyphs/tests` | CPU, no network, no Claude calls |
| Links and paths in the docs | `python3 scripts/check_links.py [files]` | |
| Figures | `python3 scripts/make_readme_figures.py`, `make_claude_qa_figures.py`, `make_gpu_figures.py`, and the other `scripts/make_*_figures.py` | each reads result rows and writes `docs/img/*.png` ([scripts/README.md](../scripts/README.md)) |

Outputs: `results/phase{1,1b,2}/<run>/`, `encoder_probe/results/<run>/`, `learned_glyphs/results/<run>/`, each
with `runs.jsonl`. Claude responses are cached in `results/cache/claude/` (keyed on model, prompt and image
bytes), so re-runs are free.
