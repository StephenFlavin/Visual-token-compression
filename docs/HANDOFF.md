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
