# AGENTS.md — how to work in this repo (read before running anything)

Research harness testing **Visual BPE**: draw text more densely (substitution glyphs, packed cells, learned glyph
banks, colour codes) to get more characters per image pixel / vision token. Overview and current state:
`README.md`. Design, pre-registered predictions and kill criteria: `PLAN.md` (phases 1-2),
`learned_glyphs/PLAN.md`, `encoder_probe/PLAN.md` and `ROUND3.md`. Results: `docs/RESULTS.md`,
`encoder_probe/RESULTS.md`; where rows live: `results/README.md`; scripts: `scripts/README.md`.

## Environment setup

Cloud CPU runner: 4 CPU, ~15.7 GB RAM (cgroup), ~39 GB disk, no GPU. GPU runners: `docs/GPU_RUNNER.md` (any Linux
GPU) and `docs/LOCAL_GPU.md` (the user's Windows RTX 3090).

```bash
pip install -r requirements.txt                       # torch line may fail: use the CPU index below
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install rapidocr-onnxruntime                      # Phase 1b OCR reader (installs 1.2.3 on py3.13)
scripts/fetch_fonts.sh                                # Noto Sans CJK SC, Archivo Narrow, Roboto Condensed
python3 scripts/build_data.py                         # WikiText-103 -> data/ passages, splits, dict_{100,300,1000,3000}.json (~5 min)
OMP_NUM_THREADS=1 python3 -m pytest -q tests encoder_probe/tests learned_glyphs/tests   # ~310 tests, ~4 min, no network, no Claude calls
```
Round 3 also needs `transformers` (runner: 5.22.1) and `peft`; Qwen/DeepEncoder weights download from HF Hub into
`encoder_probe/cache/`. `data/`, fonts, image dirs, caches, checkpoints and adapters are git-ignored; everything
regenerates from the commands above.

## Running the experiments

| What | Command | Notes |
|---|---|---|
| Phase 1 Claude pilot | `python3 experiments/phase1_sweep.py --pilot --max-cost 10` then add `--yes` | Prints estimate first; estimate is ~2x the real cost |
| Phase 1 Claude, Latin only | `python3 experiments/phase1_sweep.py --variant natural --n-passages 6 --schemes latin latin_condensed --budgets 14 17 20 23 27 32 40 --run-name latin_natural --max-cost 5 --yes` | ~$1.9 for 84 calls |
| Phase 1 dry run (free) | add `--dry-run [--fake-noise 0.03]` | fake transcriber, no Claude |
| Phase 1b OCR zero-shot | `python3 experiments/phase1b_ocr.py --variant natural --n-passages 6` (and `--variant scrambled --n-passages 3`) | CPU, ~5 s/page, 2 workers; resumable |
| Phase 2 CRNN grid | `nohup python3 experiments/phase2_driver.py > results/phase2_driver.log 2>&1 &` | one process per (scheme, em), largest em first, keeps 4 cores busy; per-job logs in `results/phase2/<run>/driver/`. (`experiments/phase2_main.sh` is the older 3-worker wrapper: leaves cores idle.) |
| Phase 2 single runs | `python3 experiments/phase2_train.py --schemes latin --ems 5 8 --steps 3500 --parallel 3 --threads 1 --run-name X` | `--smoke` for a quick check |
| CPU job queue (any experiments) | one single-core command per line in `results/queue_<round>.txt`; `nohup python3 scripts/job_queue.py results/queue_<round>.txt --log-dir results/queue_logs/<round> [--after-phase2] > results/queue_<round>.log 2>&1 &` | backfills free cores (subtracts busy Phase-2 processes only); lines can be appended while it runs. **Restarting re-claims from line 1**: new queue file + new `--log-dir` per round. Move finished queue files to `results/archive/` |
| Learned glyphs | `learned_glyphs/experiments/train_joint.py`, `calibrate_channel.py`, `transfer_eval.py` | see `learned_glyphs/PLAN.md`; `--dry-run` prints configs; `--grid e1` for the 2x2 grid |
| Frozen-encoder probe, round 1 | `python3 encoder_probe/experiments/run_probe.py --print-grid --steps 3000 --run-name main > results/queue_X.txt`, then the job queue; single run: `run_probe.py --encoding latin_p8 --encoder siglip_l6 --cond jpeg75` | ~20-30 min/run at 1 thread (+1-3 min SigLIP features); `analyse.py` prints the tables |
| Frozen-encoder probe, round 2 | `python3 encoder_probe/experiments/run_probe2.py --device cuda --encoding latin_p32 --encoder deepencoder --cond clean --train-pages 1000 --run-name round2_gpu_tp1000` | `docs/LOCAL_GPU.md` (`EP_PAGE_CACHE=1`, P0-P3 queue, `analyse2.py`); `--device auto` = cuda if available; `--pages-only` pre-renders |
| Round 3 R3-1: LoRA fine-tune Qwen2.5-VL-3B (GPU) | `python3 encoder_probe/experiments/finetune_qwen.py --encoding latin_p8 --smoke`, then `--steps N --run-name R` (and `--blind-images` → `R_blind`) | runbooks `docs/GPU_RUNNER.md` (any Linux GPU), `docs/LOCAL_GPU.md` §8 (3090); design `encoder_probe/ROUND3.md`. Every arm needs its blind control |
| Round 3 G2-lite: probe on Qwen ViT states | `python3 encoder_probe/experiments/run_g2lite.py --print-queue` (arms frozen / trained / direct = no merger, plus blind) | `--merger-lr 1e-4`; `analyse_g2lite.py`; rows `encoder_probe/results/g2lite*/` |
| Round 3 R3-2: long-context QA (GPU) | `python3 encoder_probe/experiments/longctx_bench.py --inputs text latin_p8 --lengths 10000 --tasks needle cloze [--adapter <finetune result>.pt]` | not run yet; `docs/LOCAL_GPU.md` §8 |
| Analysis | `python3 -m vbpe.analysis <runs.jsonl> [more] --out DIR` | Phase 1, 1b and 2 rows; summary.md/json + plots. Phase 2: run `scripts/merge_phase2.py` first |
| Claude vision checks | `python3 scripts/check_claude_vision.py` | token formula, legibility, latency (~$0.5) |

Outputs: `results/phase{1,1b,2}/<run>/`, `encoder_probe/results/<run>/`, `learned_glyphs/results/<run>/`, each with
`runs.jsonl` (map in `results/README.md`). Claude responses are cached in `results/cache/claude/` (keyed on
model+prompt+image bytes), so re-runs are free.

## Things that DON'T work / gotchas (save yourself the time)

**Network / installs**
- `apt-get` packages (e.g. fonts-noto-cjk) are unavailable. `github.com/.../raw/...` returns 403 via the proxy, but **`raw.githubusercontent.com` works**. `fonts.gstatic.com` 404s. HF Hub and the PyTorch CPU wheel index work.

**Claude as a reader (headless `claude -p`)**
- Send images as base64 in `--input-format stream-json`; no tools needed. `vbpe/claude_eval.py` does this correctly.
- **Run the child CLI with a clean env** (`env -i` + PATH/HOME/proxy/CA vars) in an empty temp dir. Otherwise it reuses the parent session id, adds an extra haiku summary call and ~830 cached tokens per call.
- Image tokens = `ceil(W/28)*ceil(H/28)+3` (not W·H/750). No downscaling up to 1568×1568. Use canvas sides that are multiples of 28.
- Use `--effort low`: adaptive thinking doubles cost with no CER gain. 8 concurrent calls: no rate limiting. `total_cost_usd` is list price; on this runner calls draw from the user's subscription quota.
- **Claude cannot score non-natural text.** Substitution, quadrant pages and even word-scrambled English get refused or cut off by safety classifiers; `subst*_legend` (cipher key in prompt) gets a hard API AUP error. Do NOT reword/disguise prompts to get past this: that is classifier evasion. Use Claude only for natural Latin; use PP-OCR (Phase 1b) or trained readers for everything else.
- Below ~18 px²/char Claude hallucinates fluent English or refuses; the base OCR prompt says "never refuse, give best guess".
- Condensed fonts LOSE for Claude (≈1.11–1.2x more px²/char at 3% CER) despite fitting more per line; Glyph's tuned config also picked a wide font.

**PP-OCR via rapidocr_onnxruntime 1.2.3**
- It **ignores `use_det=False`** (detects on any crop taller than 30 px) and **resets `text_score` to 0.5 whenever any kwarg is passed**, silently dropping lines. Call `engine.text_recognizer([bgr_crops])` directly (`vbpe/readers/rapidocr_reader.py::read_lines`).
- Full-page detection + reading-order stitching drops/reorders lines. Use exact line strips from the layout: strip i = `[margin + i*pitch, margin + (i+1)*pitch)`; do NOT pad (neighbour ascenders kill recognition).
- It drops inter-word spaces and emits full-width punctuation → score with `cer_nospace` (NFKC + whitespace removed). Its training is Chinese-heavy: biased towards CJK glyphs.

**Layout / rendering**
- `fit_em` must floor, not round (rounding up overflowed the canvas).
- Breaks are allowed after units whose source ends in whitespace AND before units whose source starts with whitespace (dictionary tokens look like `" the "`).
- Phase 2 renders with `supersample=4` (true sub-pixel phase, no hinting differences); Phase 1/1b render natively.

**CRNN (Phase 2) and learned glyphs**
- Keep 2-px columns (4-px stride: latin em8 2.35% vs 0.19% CER). CTC frames: each column expanded E times so T ≥ 2L+1.
- **Mixed-em training learns far slower per step** than per-em models (latin em7 18% vs ~2% at 1500 steps). Use per-em runs.
- **Render caches written before per-line `heights.npy` existed load 1 row taller** than fresh batches (latin em4: 62% CER vs 3.8% fresh). `build_lineset` treats them as misses. Low train loss + terrible val CER → suspect train/val geometry mismatch first.
- GroupNorm worse than BatchNorm. Substitution (1000+ classes) and quadrant models learn much slower than Latin: check `curves/` for convergence before trusting a crossing; re-run critical ems with more steps if still improving.
- Cite `results/phase2/merged/` (`scripts/merge_phase2.py`: longest run per scheme/em), not single runs.
- Learned banks: train ≥ 10.5k steps (tokens look useless at 3.5k); judge banks by the bridge (frozen bank, Phase-2 pipeline), not joint CER. `vbpe.analysis` keys rows by em: set `budget=px_per_char` for banks with non-default cell widths.

**Analysis**
- K2 verdict text can say "neither subst nor best Latin reaches CER x" when only subst fails: read the table. With 1 passage, bootstrap CIs collapse to a point.

**Frozen-encoder probe and round 3 (encoder_probe/)**
- CTC alone cannot bootstrap a learned token->pixel map; it needs the auxiliary MSE loss (x10) to the clean page. Per-token output heads and extra "free" feature channels memorise the training pages (13.7% / 77% test CER). Check val CER, not train loss.
- Through SigLIP the probe overfits (train CTC ~0, test 40-60%); more pages help. SigLIP's last layer loses much more than layer 6 (`--encoder siglip_l6`). The probe is the limit for Latin, so absolute probe CERs are upper bounds on the loss: decide on same-probe comparisons.
- **GridProbe reads through a pixel reconstruction, so it favours pixel-like codes.** Latin stops being pixel-decodable from Qwen after block 24; through frozen Qwen it reads held-out Latin at ~75% (latin_p16). Probe Latin numbers through VLM encoders are lower bounds; compare Latin with codes only with the LLM as reader (R3-1).
- **Every probe or fine-tune arm needs its own blind control** (images/features swapped between pages): floors differ by code (R3-1: latin_p5 77.8%, latin_p8 88.5% CER) and an LLM writes fluent English from any image.
- Colour refs are raw fixed-length chunks with boundary spaces: do not whitespace-normalise colour hypotheses (~0.4% CER floor). Rows with `score_v: 2` are scored correctly.
- **Quote the row's `cer` (16 test pages), not the last `curve` point (4 pages).** `analyse2.py` keys rows by (encoder, cond, train_pages, encoding), later row wins: pass runs.jsonl files explicitly (`round2/` 200-page rows collide with `round2_gpu_tp200`).
- **Fine-tuning a pretrained module at the probe's LR (2e-3) wrecks it:** Qwen's merger lost 11-76% of output dims and token cosine rose to 0.81-0.91 (pretrained 0.10). Use LR ≤ 1e-4 for pretrained parameters (`run_g2lite.py --merger-lr`, `finetune_qwen.py --lr-vision 2e-5`) and check output std, dead dims and token cosine early.
- **DeepEncoder on CUDA must run in fp32 with TF32 off** (bf16: corr 0.9974 vs CPU fp32). The probe uses bf16 autocast on CUDA. Rows record `device`, `enc_compute`, `bf16`.
- **CPU bf16 depends on the host:** without AMX/AVX512-BF16 (`grep -o 'amx_bf16\|avx512_bf16' /proc/cpuinfo` empty) bf16 runs 3-10x slower than fp32. `encoder_probe/ep/vlm_encoders.cpu_has_bf16()` picks the dtype; do the same elsewhere. In fp32 call `torch.set_flush_denormal(True)` (denormals: 5.5x slower). Sudden slowdown after a restart → suspect this first.
- `finetune_qwen.py` adapters go to `encoder_probe/cache/r3_adapters/` (~0.56 GB each, git-ignored): copy them off a rented GPU before it shuts down.

**GPU runners**
- **The user's RTX 3090 PC is Windows** (Git Bash, the venv's `python`, `PYTHONUTF8=1`). Its Pillow has no raqm, so Latin px²/char differs by up to ~3%, test-page digests differ and `test_rgb3_unsupported_pitch` fails for this reason only. Compare local rows with local rows.
- **Windows pages GPU memory instead of raising OOM** (5 parallel jobs stalled): at most 3 on the 3090. Elsewhere size concurrency from the smoke's peak memory (sum < ~85% of VRAM).
- Measured speeds: RTX 3090 probe 0.015-0.06 s/step (cloud CPU ~3), DeepEncoder ~0.05 s/page, Latin rendering ~45 s/1000 pages (`EP_PAGE_CACHE=1`). RTX 4090 R3-1: 1.2 s/step, 11.4 GB peak, ~7 min load+eval per job.

**Cloud runner: resources, restarts, resuming**
- Speed: ~0.6–0.8 s/step per Phase-2 run at 1 thread, so a 3500-step run ≈ 40 min; 26 runs ≈ 6 h on 3 cores, ~4.5 h on 4.
- **Disk is fixed (~39 GB) and has filled up** (checkpoint saves crashed; a full disk also truncated docs in a commit). Checkpoints are atomic (.tmp then replace): delete `*.tmp` before restarting. Snapshots (~725 MB), qwen25vl_h32 caches (~1.35 GB/500 pages) and old checkpoints add up: keep ≥ 3 GB free and log `df`.
- **cgroup memory ~15.7 GB:** the OOM killer took the largest job twice (5.2 GB probe, 5.8 GB cache build). Budget peak RSS per job; labels for 500 rgb3_lg pages alone are ~1.5 GB.
- **The container is suspended soon after the session goes idle** (no turn, no running subagent): nohup'd jobs die 5-15 min later and the next wake shows `uptime` 0 min (not memory: logged < 7 GB). Keep a background subagent waiting on the queue (run_in_background loops that re-arm). The container can also restart when the session resumes. So **make long jobs checkpoint and resume** (`run_probe2.py --ckpt-every 200`, `finetune_qwen.py --ckpt-every 250`; resume is exact), and on every wake check `uptime`.
- A *restart* keeps the disk (data/, fonts, caches, pip): restart `phase2_driver.py` / `job_queue.py` (fresh queue file with only the remaining runs, fresh `--log-dir`) / `autosave_results.sh` with nohup. A *reclaim* loses all git-ignored state: redo setup first. `nohup scripts/autosave_results.sh 600` commits+pushes `runs.jsonl` every 10 min.
- Resuming: Phase 2 (no Claude calls) skips finished (scheme, em) rows; in-flight runs restart from step 0 (Phase 2 writes checkpoints only at the end). Phase 1 re-runs skip finished rows and read the Claude cache; errors (incl. usage limits) are not cached and are retried.
- `job_queue.py` only subtracts Phase-2 processes, so two queues oversubscribe the cores. To reorder a running queue, atomically rewrite only its unclaimed tail, keeping the first <claimed> lines byte-identical (claimed = number of `start job` lines in its log).
- Never edit a running bash script (bash reads it incrementally); to change plans, SIGKILL the wrapper bash only (its python child keeps running).

**Git / shell**
- Never `pkill -f <name>` from a shell whose command line contains <name> (it kills itself); a `while pgrep -f <name>` loop matches itself. Use `pgrep -f "[j]ob_queue.py"`, and only if the plain string appears nowhere else in that command line (`pkill -f "[r]un_probe.py --encoding x"` then `nohup python3 run_probe.py --encoding x` in one call kills its shell, exit 144). Put the pkill in its own call.
- A stop hook demands a clean tree. Live Phase-2 logs/curves are git-ignored while running; `git add -f` them when a grid finishes. Other long jobs: snapshot-commit (`WIP:`) and push.

## Findings so far (2026-11-05)
Summary with figures: `README.md`; details per stage in the linked docs.
- **Phases 1-2 (K1-K3): KILL.** Substitution saves only ~1.19x em (K1); PP-OCR zero-shot misses the K2 bar (0.89x); trained CRNNs: plain Latin wins (1.73% CER at 7.5 px²/char vs subst300 2.26% at 8.9). Claude reads natural Latin at 29.4 px²/char for 3% CER and cannot be used on codes. `docs/RESULTS.md`.
- **Learned glyph alphabet: qualified GO for raw pixels.** Learned N=300 needs 0.57x Latin's px²/char at 3% CER (bridge), 0.88x for whole-pixel banks; the densest banks are brittle on held-out degradations. `docs/RESULTS.md`, `learned_glyphs/RESULTS.md`.
- **Frozen encoders (rounds 1-2, SigLIP, DeepEncoder): dense codes are lost.** Colour/RGB codes are dead; grey-level banks are not supported; only low-density Latin is readable through the encoder. Learned glyphs should be whole-pixel. `encoder_probe/RESULTS.md`.
- **Round 3 (Qwen2.5-VL-3B), CPU:** the moonshot `rgb3_lg_pb_h5` (322 chars/token) is readable from raw pixels (G0 1.07% CER); glyph codes stay pixel-like through the ViT but collapse at the merger (Q1); G2-lite: no arm usable. `encoder_probe/ROUND3.md`.
- **Round 3 R3-1 (RTX 4090, 1000 steps, blind controls): only latin_p8 (28 chars/token) is read** (55% CER vs blind 88.5%; it reads the first line, then drifts). latin_p5 (72), lg_pb_h5 (107; by val loss only, its blind run died before eval) and the moonshot sit at their own blind floors. **G3 is undecidable** (no arm near 3% CER). `docs/HANDOFF.md`.
- **Next:** 5000-step Latin arms (latin_p8 first) with their blind controls; for the glyph codes, a vision-side diagnostic (higher `--lr-vision` / more `--vision-blocks` vs the blind loss) before any long run.
