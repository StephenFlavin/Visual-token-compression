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
1. **Cloud cross-check rows** `encoder_probe/results/round2_tp/` (1000 pages) and `round2_tp2000/` (2000 pages,
   latin_p32 deepencoder/clean, CPU) were not on the branch when the local session ended. When they land, compare
   with `round2_gpu_tp1000` (18.9% at 10k steps, 25.8% at 3k) as an approximate check (different renderer, 3k steps).
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
