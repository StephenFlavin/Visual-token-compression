# AGENTS.md — how to work in this repo (read before running anything)

Research harness testing **Visual BPE** (render frequent character chains as single glyphs or
Hangul-style packed cells to get more text per image pixel). `PLAN.md` = design, pre-registered
predictions and kill criteria. `docs/` = prior art, dictionary/K1 report, Claude eval notes.

## Environment setup (cloud runner: 4 CPU, 15 GB RAM, no GPU)

```bash
pip install -r requirements.txt                       # torch line may fail: use the CPU index below
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install rapidocr-onnxruntime                      # Phase 1b OCR reader (installs 1.2.3 on py3.13)
scripts/fetch_fonts.sh                                # Noto Sans CJK SC, Archivo Narrow, Roboto Condensed
python3 scripts/build_data.py                         # WikiText-103 -> data/ passages, splits, dict_{100,300,1000,3000}.json (~5 min)
python3 -m pytest -q tests                            # ~100 tests, <15 s, no network, no Claude calls
```

`data/`, fonts, image dirs, caches and checkpoints are git-ignored; everything regenerates from the commands above.

## Running the experiments

| What | Command | Notes |
|---|---|---|
| Phase 1 Claude pilot | `python3 experiments/phase1_sweep.py --pilot --max-cost 10` then add `--yes` | Prints estimate first; estimate is ~2x the real cost |
| Phase 1 Claude, Latin only | `python3 experiments/phase1_sweep.py --variant natural --n-passages 6 --schemes latin latin_condensed --budgets 14 17 20 23 27 32 40 --run-name latin_natural --max-cost 5 --yes` | ~$1.9 for 84 calls |
| Phase 1 dry run (free) | add `--dry-run [--fake-noise 0.03]` | fake transcriber, no Claude |
| Phase 1b OCR zero-shot | `python3 experiments/phase1b_ocr.py --variant natural --n-passages 6` (and `--variant scrambled --n-passages 3`) | CPU, ~5 s/page, 2 workers; resumable |
| Phase 2 CRNN grid | `experiments/phase2_main.sh` (per-em, 3500 steps, 26 runs) | resumable; logs in `results/phase2_main.log`. Runs groups sequentially with a 3-worker pool → idle cores; prefer the driver below |
| Phase 2 core-filling scheduler | `nohup python3 experiments/phase2_driver.py > results/phase2_driver.log 2>&1 &` | one process per (scheme, em), largest em first, keeps 4 cores busy (subtracts other busy phase2 processes); per-job logs in `results/phase2/<run>/driver/` |
| Phase 2 single runs | `python3 experiments/phase2_train.py --schemes latin --ems 5 8 --steps 3500 --parallel 3 --threads 1 --run-name X` | `--smoke` for a quick check |
| CPU job queue (any experiments) | append single-core commands to `results/queue.txt`; `nohup python3 scripts/job_queue.py results/queue.txt [--after-phase2] > results/queue.log 2>&1 &` | backfills free cores (subtracts busy Phase-2 processes); new lines can be appended while it runs; per-job logs in `results/queue_logs/`. **Restarting it re-claims from line 1** (it does not remember finished lines): start a new queue file (`--log-dir` too) for each new round |
| Learned glyphs (sub-project) | see `learned_glyphs/PLAN.md`; `learned_glyphs/experiments/train_joint.py`, `calibrate_channel.py`, `transfer_eval.py` | `--dry-run` prints configs; `--grid e1` for the 2x2 grid |
| Frozen-encoder probe (sub-project) | `python3 encoder_probe/experiments/run_probe.py --print-grid --steps 3000 --run-name main > results/queue_X.txt`, then the job queue; single run: `run_probe.py --encoding latin_p8 --encoder siglip_l6 --cond jpeg75` | see `encoder_probe/PLAN.md`; ~20-30 min/run at 1 thread (+1-3 min SigLIP features); `encoder_probe/experiments/analyse.py` prints the tables |
| Any Linux GPU runner (time-boxed, e.g. 3 h) | `docs/GPU_RUNNER.md` | hardware-agnostic: measure s/step + peak memory in the smoke, then size equal-step arms; R3-1 fine-tune is the priority |
| Encoder probe round 2 on a local GPU | `python3 encoder_probe/experiments/run_probe2.py --device cuda --encoding latin_p32 --encoder deepencoder --cond clean --train-pages 1000 --run-name round2_gpu_tp1000` | runbook: `docs/LOCAL_GPU.md` (setup, `EP_PAGE_CACHE=1`, P0-P3 queue, `encoder_probe/experiments/analyse2.py`); `--device auto` = cuda if available |
| Analysis | `python3 -m vbpe.analysis <runs.jsonl> [more runs.jsonl] --out DIR` | works for Phase 1, 1b and 2 rows; writes summary.md/json + plots |
| Claude vision checks | `python3 scripts/check_claude_vision.py` | token formula, legibility, latency (~$0.5) |

Outputs: `results/phase1/<run>/`, `results/phase1b/<run>/`, `results/phase2/<run>/` (each has `runs.jsonl`).
Claude responses are cached in `results/cache/claude/` (keyed on model+prompt+image bytes), so re-runs are free.

## Things that DON'T work / gotchas (save yourself the time)

**Network / installs**
- `apt-get` packages (e.g. fonts-noto-cjk) are unavailable. `github.com/.../raw/...` returns 403 via the proxy, but **`raw.githubusercontent.com` works**. `fonts.gstatic.com` 404s. HF Hub and the PyTorch CPU wheel index work.

**Claude as a reader (headless `claude -p`)**
- Send images as base64 in `--input-format stream-json`; no tools needed. `vbpe/claude_eval.py` does this correctly.
- **Must run the child CLI with a clean env** (`env -i` + PATH/HOME/proxy/CA vars) in an empty temp dir. Inheriting the parent session's env makes the child reuse the parent session id, adds an extra haiku summary call and ~830 cached tokens per call.
- Image tokens = `ceil(W/28)*ceil(H/28)+3` (not W·H/750). No downscaling up to 1568×1568. Use canvas sides that are multiples of 28.
- Use `--effort low`: adaptive thinking doubles cost with no CER gain. 8 concurrent calls: no rate limiting.
- `total_cost_usd` is list price; on this runner calls draw from the user's subscription quota.
- **Claude cannot score non-natural text.** Substitution (English with CJK symbols), quadrant-packed pages and even word-scrambled English get refused or cut off by safety classifiers ("my previous response was stopped by a safety classifier"); `subst*_legend` (cipher key in prompt) gets a hard API AUP error. Do NOT try to reword/disguise prompts to get past this — that is classifier evasion. Use Claude only for natural Latin baselines; use PP-OCR (Phase 1b) or trained CRNNs (Phase 2) for composite schemes.
- Below ~18 px²/char Claude hallucinates fluent English or refuses; the base OCR prompt now says "never refuse, give best guess".
- Condensed fonts LOSE for Claude (≈1.11–1.2x more px²/char at 3% CER) despite fitting more per line; Glyph's tuned config also picked a wide font.

**PP-OCR via rapidocr_onnxruntime 1.2.3**
- It **ignores `use_det=False`** (runs detection on any crop taller than 30 px) and **resets `text_score` to 0.5 whenever any kwarg is passed**, silently dropping whole lines. Call `engine.text_recognizer([bgr_crops])` directly (see `vbpe/readers/rapidocr_reader.py::read_lines`).
- Full-page detection + reading-order stitching drops/reorders lines → non-monotonic CER curves. Use exact line strips from our layout geometry: strip i = `[margin + i*pitch, margin + (i+1)*pitch)`; do NOT pad (padding pulls in neighbour ascenders and kills recognition).
- It drops inter-word spaces and emits full-width punctuation → score all schemes with `cer_nospace` (NFKC + whitespace removed).
- Its training is Chinese-heavy: biased towards CJK glyphs (the opposite bias of Claude).

**Layout / rendering**
- `fit_em` must floor, not round (rounding up overflowed the canvas).
- Breaks are allowed after units whose source ends in whitespace AND before units whose source starts with whitespace (top dictionary tokens look like `" the "`).
- Phase 2 renders with `supersample=4` (Pillow snaps glyph origins to whole pixels; supersampling gives true sub-pixel phase and removes hinting differences). Phase 1/1b render natively (supersample=1).

**CRNN (Phase 2)**
- A 4-px column stride was a bottleneck (latin em8: 2.35% vs 0.19% CER with 2-px columns). Keep 2-px columns.
- CTC frames: each column expanded E times so T ≥ 2L+1 (quad cells emit 2–4 labels per cell).
- **Mixed-em training (one model across ems) learns far slower per step** than per-em models (latin em7 18% vs ~2% at 1500 steps) — not a way to save compute. Use per-em runs.
- **Render caches written before per-line `heights.npy` existed load at the padded max height** (1 row taller than fresh train batches). That silently wrecked latin em4 (62% CER; 3.8% on fresh renders with the same checkpoint). `build_lineset` now treats such caches as misses. If a run has low train loss but terrible val CER, suspect train/val geometry mismatch first.
- GroupNorm worse than BatchNorm. Substitution (1000+ classes) and quadrant models learn much slower than Latin: check `curves/` for convergence before trusting a crossing; re-run critical ems with more steps if still improving.

**Analysis**
- K2 verdict text can say "neither subst nor best Latin reaches CER x" when only subst fails — read the table, not just the verdict line.
- With 1 passage, bootstrap CIs collapse to a point.

- Real speed on this runner: ~0.6–0.8 s/step per run at 1 thread (latin em6–8), so a 3500-step run ≈ 40 min; budget ~6 h for 26 runs on 3 cores, ~4.5 h on 4.
- Never edit a running bash script (bash reads it incrementally). To change plans mid-run, SIGKILL the wrapper bash only — its python child keeps running.

**Resuming after a crash, usage limit or container reclaim**
- Phase 2 makes no Claude calls, so usage limits don't affect it. Resume = re-run `nohup python3 experiments/phase2_driver.py > results/phase2_driver.log 2>&1 &` (covers the whole grid incl. latin); finished (scheme, em) rows in `runs.jsonl` are skipped. In-flight runs restart from step 0 (checkpoints are only written at the end of a run).
- Phase 1 Claude sweeps: re-run the same command; finished rows are skipped and responses come from `results/cache/claude/`. Errors (incl. usage-limit errors) are not cached, so they are retried on re-run.
- After a container reclaim, git-ignored state is gone (data/, fonts, render caches, checkpoints, Claude cache): redo the setup block first. `scripts/autosave_results.sh 600` (run with nohup) commits+pushes `runs.jsonl` every 10 min so finished rows survive.

- Observed 2026-11-03: a container *restart* (not reclaim) kills all processes but keeps the disk (data/, fonts, caches, pip packages). Recovery = restart `phase2_driver.py`, `job_queue.py ... --after-phase2`, `autosave_results.sh` with nohup; nothing else needed.
- **Observed 2026-11-04: the container is suspended soon after the session goes idle** (no turn and no running subagent). nohup'd jobs die ~5-15 min after the last turn, and the next wake shows `uptime` 0 min. Memory was ruled out (logged at under 7 GB). Long unattended runs need something that keeps the session active: a background subagent that waits on the queue with run_in_background Bash loops and re-arms. Round 1 survived 10 h only because its subagent was running. Even then the container restarted when the session resumed on a user message (06:17, jobs healthy until that moment). So **make long jobs checkpoint and resume** (`run_probe2.py --ckpt-every 200`; resume is exact, tested), and on every wake check `uptime` and restart the queue with a fresh `--log-dir`.
- **Hosts differ after a restart: check bf16 support.** On 2026-11-04 the post-restart host had no AMX/AVX512-BF16 (`grep -o 'amx_bf16\|avx512_bf16' /proc/cpuinfo` empty), and bf16 autocast/weights ran 3-10x slower than fp32 (probe 1.8 -> 8.4 s/step). `encoder_probe/ep/vlm_encoders.cpu_has_bf16()` picks the dtype automatically for round 2; anything else using bf16 on CPU should do the same. If speeds suddenly drop after a restart, suspect this first. In fp32, also call `torch.set_flush_denormal(True)`: denormals made the round-2 latin probes 5.5x slower (17 vs 3.1 s/step); bf16 on AMX had hidden it.

**Frozen-encoder probe (encoder_probe/)**
- CTC alone cannot bootstrap a learned token->pixel map (unigram-level CER for 1500 steps even on raw-pixel tokens). The fix is an auxiliary MSE loss (x10) from the map to the clean training page. Two things memorise the 200 training pages: per-token output heads (13.7% test CER at identity/latin_p8, train loss 0.006) and extra "free" feature channels (identity/latin_p10 77%). Check val CER, not train loss.
- Through SigLIP the probe overfits (train CTC ~0, test 40-60%). More training pages help (1000 pages: latin_p10 61% -> 43%). Read SigLIP numbers as upper bounds on the information lost. The last layer loses much more than layer 6 (`--encoder siglip_l6`).
- Colour refs are raw fixed-length chunks, so they keep their boundary spaces: do not whitespace-normalise colour hypotheses (doing so adds a ~0.4% CER floor). Rows with `score_v: 2` are scored correctly.
- `job_queue.py` only subtracts Phase-2 processes, so two queues oversubscribe the cores. To reorder a running queue, atomically rewrite only its unclaimed tail and keep the first <claimed> lines byte-identical; the claimed count is the number of `start job` lines in its log.
- **The user's local GPU PC (RTX 3090) is Windows** (use Git Bash, the venv's `python`, `PYTHONUTF8=1`). Its Pillow has no raqm, so Latin layout is hinted: px²/char differs from the Linux runner by up to ~3%, test-page digests differ and `test_rgb3_unsupported_pitch` fails for this reason only. Compare local rows with local rows.
- **Windows pages GPU memory instead of raising OOM:** 5 parallel GPU jobs stalled when VRAM filled. Run at most 3 on the 3090 (resume from checkpoints). On any other GPU, size concurrency from the smoke's measured peak memory (keep the sum under ~85% of VRAM); Linux raises OOM instead of stalling.
- **DeepEncoder on CUDA must run in fp32 with TF32 off** (CUDA bf16: corr 0.9974 vs CPU fp32; fp32: 0.999998). The probe itself uses bf16 autocast on CUDA. Rows record `device`, `enc_compute`, `bf16`.
- **Quote the row's `cer` (16 test pages), not the last `curve` point (4 pages, ~0.5 points off).** `analyse2.py` keys rows by (encoder, cond, train_pages, encoding), not steps, and the later row wins; pass the runs.jsonl files explicitly (cloud `round2/` 200-page rows collide with `round2_gpu_tp200`).
- Measured RTX 3090 speeds (other GPUs: measure in the smoke): probe 0.015-0.06 s/step (cloud CPU ~3 s/step), DeepEncoder ~0.05 s/page, Latin rendering ~45 s per 1000 pages (cache with `EP_PAGE_CACHE=1`).

- **Fine-tuning a pretrained module at the probe's LR wrecks it.** G2-lite trained Qwen's patch merger with the probe's
  AdamW LR 2e-3: within a few hundred steps 11-76% of output dims went dead and random token pairs reached cosine
  0.81-0.91 (pretrained 0.10), so the "trained" arm measured a collapsed merger. Use a separate LR ≤ 1e-4 for
  pretrained parameters (`run_g2lite.py --merger-lr`, `finetune_qwen.py --lr-vision 2e-5`), and check output std,
  dead dims and token cosine early.
- **GridProbe reads through a pixel reconstruction, so it favours pixel-like codes.** Q1: Latin is no longer linearly
  pixel-decodable from Qwen states after block 24, while the whole-pixel glyph codes stay pixel-like to the last block.
  Through frozen Qwen, Latin memorises 100 pages and reads held-out text at 75% (latin_p16; run_probe2 cross-check
  74.5%). Probe Latin numbers through VLM encoders are lower bounds; compare Latin with codes only with the LLM as
  reader (R3-1).
- **The cloud runner's disk allowance is fixed (~39 GB here) and filled up overnight** (0 MB free: checkpoint saves
  and a cache concat crashed with rc=1). Checkpoints are atomic (.tmp then replace), so the last good one survives;
  delete `*.tmp` before restarting. Diagnostic snapshots (~725 MB each), qwen25vl_h32 caches (~1.35 GB per 500
  pages) and old checkpoints add up: keep ≥ 3 GB free and log `df`.
- **cgroup memory limit ~15.7 GB:** the OOM killer took the largest job twice (a 5.2 GB 1185-class probe; a 5.8 GB
  cache build). Budget peak RSS per job; label strings for 500 rgb3_lg pages alone are ~1.5 GB.
- **After a restart, don't restart an old queue file from line 1** if it holds lines whose runs were killed and
  superseded (job_queue re-claims every line; only finished rows and caches skip). Write a fresh queue file with
  just the remaining runs.

**Git**
- Never `pkill -f <name>` from a shell whose own command line contains <name> (it kills itself); likewise a `while pgrep -f <name>` wait loop matches itself and never exits. Use a bracket pattern, e.g. `pgrep -f "[j]ob_queue.py"`. The bracket only helps if the plain string appears nowhere else in that command line: `pkill -f "[r]un_probe.py --encoding x"` followed by `nohup python3 run_probe.py --encoding x ...` in the same Bash call kills its own shell (exit 144). Put the pkill in its own call. A stop hook demands a clean tree. Live Phase-2 logs/curves are git-ignored while runs are going (see .gitignore); force-add them (`git add -f`) when a grid finishes. Other long jobs: snapshot-commit (`WIP:`) and push.

## Findings so far (details in results/*/summary.md)
- K1 (analytic): N=1000 substitution saves 32% line width vs condensed Latin at equal em → only ~1.19x larger em on a fixed page.
- Claude, natural text: DejaVu P@3% = 29.4 px²/char [28.6, 30.2] ≈ 37 vision tokens / 1k chars (~6.7x vs text tokens).
- PP-OCR zero-shot: subst1000 P@10% = 0.89x best Latin (K2 bar 0.80 → negative); at 3% subst is 1.17x worse; subst3000 worse than subst1000. Scrambled passages give the same numbers (prior credit ≈ 0), so PP-OCR's result is not a language-prior artefact.
- Quadrant: letters end up 22–29% smaller than condensed Latin at the same canvas; Claude cannot read it at any budget.
- Phase 2 (decisive, K3): **KILL.** Converged CRNNs: latin (DejaVu) 1.73% CER at 7.5 px²/char; subst300 2.26% at 8.9; subst1000 5.75% at 7.8; quadrant ≥3× worse. Plain Latin wins. Use `scripts/merge_phase2.py` (longest run per scheme/em) before `vbpe.analysis`. Details in docs/RESULTS.md.
- Learned glyph alphabet (learned_glyphs/): **qualified GO.** Converged (10.5k) learned N=300 needs 0.57x Latin's px²/char at 3% and 1% CER in the bridge (frozen bank, unchanged Phase-2 pipeline) and 0.74x / 0.63x in E2 transfer; the gain survives harsh (h7: 3.5% vs DejaVu 43%). Whole-pixel banks: 0.88x / 0.77x. The densest banks are brittle on held-out degradations (55–68% CER). Judge banks by the bridge, not joint CER. Train ≥10.5k steps: tokens look useless at 3.5k. `vbpe.analysis` keys rows by em: set `budget=px_per_char` for banks with non-default cell widths. Untested: VLM encoders. Details in docs/RESULTS.md.
- Frozen-encoder probe (encoder_probe/): **colour pixels and RGB overlay dead for frozen-encoder VLMs; learned glyphs should be whole-pixel.** Round 1 (SigLIP-B/16): with raw pixels the probe reads 1x1 colour blocks at 0.06% CER (256 chars/token), and colour survives JPEG q75 from 2x2 blocks up; through SigLIP no colour encoding gets below 62% CER, grey-level banks lose their ceiling lead (layer 6: lg_free_h4 76% vs lg_pb_h4 55%), and Latin needs ~2 chars/token for 5% CER at layer 6. Round 2 (DeepSeek-OCR's DeepEncoder, 512 px, 64 tokens; Qwen dropped; 1000 train pages, clean and JPEG q75): identity32 ceilings are < 3% for every encoding (guard passes), but through the encoder colour blocks (71-73%), rgb3 (83.5%) and both bank families (lg_free_h4 82.2% vs lg_pb_h5 82.4%) all sit at the floor, so colour/RGB is dead (P2/P3 gated off) and grey-level banks are not supported. Only latin_p32 (8.5 chars/token) is readable (19%, JPEG costs nothing); latin_p24 (15.4) and denser are at the floor, so no family reaches 10% and the decision rules apply only in weaker form. The probe is the limit for Latin (latin_p32 85% -> 15% from 200 to 5000 pages; latin_p16 ~74% and cannot fit its training pages), so absolute levels are upper bounds on the loss; decisions rest on same-probe comparisons. Details in encoder_probe/RESULTS.md.
- Round 3 on CPU (2026-11-04/05, encoder_probe/ROUND3.md): **moonshot page readable, encoder is the open question.**
  * G0: `rgb3_lg_pb_h5` (322 chars/Qwen token) read from raw pixels at 1.07% CER on 16 test pages.
  * Q1 (linear readout of frozen Qwen ViT): the whole-pixel glyph codes stay pixel-like to the last block with no
    colour-channel collapse; every code collapses linearly at the 2x2 merger. The grey-level `rgb3_lg_free_h4` fades
    early (dropped).
  * G2-lite (500 pages, CPU): the probe reads the moonshot better from pre-merger states than through the merger, but
    no arm is near a usable level, and the probe can't measure Latin through Qwen at this scale.
  * The fair test is R3-1 on a GPU (docs/GPU_RUNNER.md).
