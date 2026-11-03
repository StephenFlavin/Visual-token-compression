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

**Git**
- Never `pkill -f <name>` from a shell whose own command line contains <name> (it kills itself). A stop hook demands a clean tree. Live Phase-2 logs/curves are git-ignored while runs are going (see .gitignore); force-add them (`git add -f`) when a grid finishes. Other long jobs: snapshot-commit (`WIP:`) and push.

## Findings so far (details in results/*/summary.md)
- K1 (analytic): N=1000 substitution saves 32% line width vs condensed Latin at equal em → only ~1.19x larger em on a fixed page.
- Claude, natural text: DejaVu P@3% = 29.4 px²/char [28.6, 30.2] ≈ 37 vision tokens / 1k chars (~6.7x vs text tokens).
- PP-OCR zero-shot: subst1000 P@10% = 0.89x best Latin (K2 bar 0.80 → negative); at 3% subst is 1.17x worse; subst3000 worse than subst1000. Scrambled passages give the same numbers (prior credit ≈ 0), so PP-OCR's result is not a language-prior artefact.
- Quadrant: letters end up 22–29% smaller than condensed Latin at the same canvas; Claude cannot read it at any budget.
- Phase 2 (decisive, K3): **KILL.** Converged CRNNs: latin (DejaVu) 1.73% CER at 7.5 px²/char; subst300 2.26% at 8.9; subst1000 5.75% at 7.8; quadrant ≥3× worse. Plain Latin wins. Use `scripts/merge_phase2.py` (longest run per scheme/em) before `vbpe.analysis`. Details in docs/RESULTS.md.
- Learned glyph alphabet (learned_glyphs/): converged (10.5k) bridge, i.e. frozen bank through the Phase-2 pipeline: learned N=300 P@3% ≈ 4.3 vs Latin 6.3 (≈32% fewer px², borderline GO); whole-pixel N=300 bank 0.59% at 7.3 vs Latin 1.73% at 7.5. Judge banks by the bridge, not joint CER (the joint reader lags far behind). Train ≥10.5k steps: tokens look useless at 3.5k. Details in docs/RESULTS.md.
