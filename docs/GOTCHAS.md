# Gotchas: things that don't work, by area

*Moved out of AGENTS.md on 2026-11-12; headings tidied and three sections added at the end on 2026-11-13 (nothing
was removed). Read the section for the area you are about to touch. Commands are in [RUNNING.md](RUNNING.md);
setup is in [AGENTS.md](../AGENTS.md).*

| Section | Read it before you |
|---|---|
| Network / installs | install anything or download weights in a cloud session |
| Claude as a reader | send page images to Claude from a script |
| PP-OCR | use the OCR reader of Phase 1b |
| Layout / rendering | change how pages are drawn |
| CRNN and learned glyphs | train a pixel reader or a glyph bank |
| Analysis | read a Phase 1-2 verdict |
| Frozen-encoder probe and round 3 | run anything under `encoder_probe/` |
| GPU runners | work on the Windows RTX 3090 PC or a rented GPU |
| Cloud runner | start a long job in a cloud session |
| Git / shell | kill a process or commit from a long job |
| Headless Claude Code agents | run `claude -p` agents as test subjects |
| Local model server | run `local_qa.py` against `llama-server` |
| Scripts without `--help` | call a script to see its options |

Terms used below: CER = character error rate; em = font size in px; px²/char = page area per character; a "blind"
run = the same training with images swapped between pages; ViT = the vision transformer of a vision-language model
(VLM); LoRA = a small trainable adapter on the language model (LLM); bf16 = 16-bit floating point (bfloat16).

## Network / installs

- **A cloud session in the `Default` environment reaches package registries only: Hugging Face is refused** (403 on CONNECT; `scripts/build_data.py` and every weight download fail). Start sessions in the `visual-bpe` environment (allows huggingface.co, its cdn-lfs / xethub hosts and us.aws.cdn.hf.co). Still refused there: `download.pytorch.org` (PyPI's torch runs on CPU) and `openaipublic.blob.core.windows.net` (tiktoken's vocabulary, which LLMLingua loads). `build_data.py` writes the split texts before it builds the dictionaries; the dictionary step needs `scripts/fetch_fonts.sh`. The WebFetch tool cannot reach huggingface.co even there; `curl` through the proxy can (`https://huggingface.co/<repo>/raw/main/README.md`, `/api/models/<repo>/tree/main`).
- `apt-get` packages (e.g. fonts-noto-cjk) are unavailable. `github.com/.../raw/...` returns 403 via the proxy, but **`raw.githubusercontent.com` works**. `fonts.gstatic.com` 404s. HF Hub and the PyTorch CPU wheel index work.

## Claude as a reader (headless `claude -p`)

- Send images as base64 in `--input-format stream-json`; no tools needed. `vbpe/claude_eval.py` does this correctly.
- **Run the child CLI with a clean env** (`env -i` + PATH/HOME/proxy/CA vars) in an empty temp dir. Otherwise it reuses the parent session id, adds an extra haiku summary call and ~830 cached tokens per call.
- Image tokens = `ceil(W/28)*ceil(H/28)+3` (not W·H/750). No downscaling up to 1568×1568. Use canvas sides that are multiples of 28.
- **Many-image calls silently lose their earliest pages above some image-token total:** ~175k image tokens arrived whole (122 small or 38 large pages); ~345k were cut to ~145-157k (243 small pages / 30 MB, and 75 large pages / 27 MB), with no error. Not bytes or image count: 7 noise pages of 26 MB arrived whole. Not in the vision docs (600 images, 128 MB). `experiments/claude_qa/claude_qa_render.py` compares `input_tokens` with the image-token formula and marks such arms invalid. Opus 5.5 reads images up to 1932x1932 px (4,764 tokens) without downscaling: `claude_eval.transcribe(..., hires=True)`.
- Use `--effort low`: adaptive thinking doubles cost with no CER gain. 8 concurrent calls: no rate limiting. `total_cost_usd` is list price; on this runner calls draw from the user's subscription quota.
- **Claude cannot score non-natural text.** Substitution, quadrant pages and even word-scrambled English get refused or cut off by safety classifiers; `subst*_legend` (cipher key in prompt) gets a hard API AUP error. Do NOT reword/disguise prompts to get past this: that is classifier evasion. Use Claude only for natural Latin; use PP-OCR (Phase 1b) or trained readers for everything else.
- Below ~18 px²/char Claude hallucinates fluent English or refuses; the base OCR prompt says "never refuse, give best guess".
- Condensed fonts LOSE for Claude (≈1.11–1.2x more px²/char at 3% CER) despite fitting more per line; Glyph's tuned config also picked a wide font.

## PP-OCR via rapidocr_onnxruntime 1.2.3

- It **ignores `use_det=False`** (detects on any crop taller than 30 px) and **resets `text_score` to 0.5 whenever any kwarg is passed**, silently dropping lines. Call `engine.text_recognizer([bgr_crops])` directly (`vbpe/readers/rapidocr_reader.py::read_lines`).
- Full-page detection + reading-order stitching drops/reorders lines. Use exact line strips from the layout: strip i = `[margin + i*pitch, margin + (i+1)*pitch)`; do NOT pad (neighbour ascenders kill recognition).
- It drops inter-word spaces and emits full-width punctuation → score with `cer_nospace` (NFKC + whitespace removed). Its training is Chinese-heavy: biased towards CJK glyphs.

## Layout / rendering

- `fit_em` must floor, not round (rounding up overflowed the canvas).
- Breaks are allowed after units whose source ends in whitespace AND before units whose source starts with whitespace (dictionary tokens look like `" the "`).
- Phase 2 renders with `supersample=4` (true sub-pixel phase, no hinting differences); Phase 1/1b render natively.

## CRNN (Phase 2) and learned glyphs

- Keep 2-px columns (4-px stride: latin em8 2.35% vs 0.19% CER). CTC frames: each column expanded E times so T ≥ 2L+1.
- **Mixed-em training learns far slower per step** than per-em models (latin em7 18% vs ~2% at 1500 steps). Use per-em runs.
- **Render caches written before per-line `heights.npy` existed load 1 row taller** than fresh batches (latin em4: 62% CER vs 3.8% fresh). `build_lineset` treats them as misses. Low train loss + terrible val CER → suspect train/val geometry mismatch first.
- GroupNorm worse than BatchNorm. Substitution (1000+ classes) and quadrant models learn much slower than Latin: check `curves/` for convergence before trusting a crossing; re-run critical ems with more steps if still improving.
- Cite `results/phase2/merged/` (`scripts/merge_phase2.py`: longest run per scheme/em), not single runs. After merging, regenerate its summary with `python3 -m vbpe.analysis results/phase2/merged/runs.jsonl --out results/phase2/merged` (the summary went stale once: 28 rows summarised, 32 merged).
- Learned banks: train ≥ 10.5k steps (tokens look useless at 3.5k); judge banks by the bridge (frozen bank, Phase-2 pipeline), not joint CER. `vbpe.analysis` keys rows by em: set `budget=px_per_char` for banks with non-default cell widths.

## Analysis

- K2 verdict text can say "neither subst nor best Latin reaches CER x" when only subst fails: read the table. With 1 passage, bootstrap CIs collapse to a point.

## Frozen-encoder probe and round 3 (encoder_probe/)

- CTC alone cannot bootstrap a learned token->pixel map; it needs the auxiliary MSE loss (x10) to the clean page. Per-token output heads and extra "free" feature channels memorise the training pages (13.7% / 77% test CER). Check val CER, not train loss.
- Through SigLIP the probe overfits (train CTC ~0, test 40-60%); more pages help. SigLIP's last layer loses much more than layer 6 (`--encoder siglip_l6`). The probe is the limit for Latin, so absolute probe CERs are upper bounds on the loss: decide on same-probe comparisons.
- **GridProbe reads through a pixel reconstruction, so it favours pixel-like codes.** Latin stops being pixel-decodable from Qwen after block 24; through frozen Qwen it reads held-out Latin at ~75% (latin_p16). Probe Latin numbers through VLM encoders are lower bounds; compare Latin with codes only with the LLM as reader (R3-1).
- **Above ~75% CER, CER can't separate partial reading from none** (unrelated fluent text scores ~80%). Judge fine-tune arms first on the val-loss gain vs their blind run (same targets, paired, noise ~0.005 nats/token), then MC; `encoder_probe/experiments/analyse_r3.py` computes both with exact tests.
- **A from-scratch vision encoder trained only through an LLM's loss collapses to a constant soft prompt** (S1 run 1:
  output std across images 2e-5, gap vs swapped images 0.000 after 4 h). Add an auxiliary reconstruction loss
  (`encoder_probe/experiments/small_vlm.py --recon`, default 10) and watch `image_spread` and the swapped-image gap at every eval.
- **peft's `disable_adapter()` context does not switch off LoRA when the inner modules are called directly, and
  gradient checkpointing recomputes layers in backward under whatever adapter state is current then.**
  `text_compress.py` toggles the LoRA layers itself (`AdapterSwitch`) and passes a checkpoint `context_fn` that
  restores the forward-time state (tested: same gradients with and without checkpointing).
- **Every probe or fine-tune arm needs its own blind control** (images/features swapped between pages): floors differ by code (R3-1: latin_p5 77.8%, latin_p8 88.5% CER) and an LLM writes fluent English from any image.
- Colour refs are raw fixed-length chunks with boundary spaces: do not whitespace-normalise colour hypotheses (~0.4% CER floor). Rows with `score_v: 2` are scored correctly.
- **Quote the row's `cer` (16 test pages), not the last `curve` point (4 pages).** `analyse2.py` keys rows by (encoder, cond, train_pages, encoding), later row wins: pass runs.jsonl files explicitly (`round2/` 200-page rows collide with `round2_gpu_tp200`).
- **Fine-tuning a pretrained module at the probe's LR (2e-3) wrecks it:** Qwen's merger lost 11-76% of output dims and token cosine rose to 0.81-0.91 (pretrained 0.10). Use LR ≤ 1e-4 for pretrained parameters (`run_g2lite.py --merger-lr`, `finetune_qwen.py --lr-vision 2e-5`) and check output std, dead dims and token cosine early.
- **DeepEncoder on CUDA must run in fp32 with TF32 off** (bf16: corr 0.9974 vs CPU fp32). The probe uses bf16 autocast on CUDA. Rows record `device`, `enc_compute`, `bf16`.
- **CPU bf16 depends on the host:** without AMX/AVX512-BF16 (`grep -o 'amx_bf16\|avx512_bf16' /proc/cpuinfo` empty) bf16 runs 3-10x slower than fp32. `encoder_probe/ep/vlm_encoders.cpu_has_bf16()` picks the dtype; do the same elsewhere. In fp32 call `torch.set_flush_denormal(True)` (denormals: 5.5x slower). Sudden slowdown after a restart → suspect this first.
- **Runs 1-2 (`--vision-blocks 16 --lora-r 0`) train 352M fp32 parameters: a resume checkpoint is 4.0 GB, written next to the old one (8 GB at save), and each kept adapter ~1.4 GB.** On the 3090 PC the first save failed with `RuntimeError: [enforce fail at inline_container.cc:672] unexpected pos` when the disk ran short (2026-11-09); keep >= 10 GB free (`run12_driver.py` checks). Rendering 12000 train pages takes ~1 h single-threaded: `capacity_curve.PageSet` caches page sets on disk with `EP_PAGE_CACHE=1`.
- `finetune_qwen.py` adapters go to `encoder_probe/cache/r3_adapters/` (~0.56 GB each, git-ignored): copy them off a rented GPU before it shuts down.

## GPU runners

- **The user's RTX 3090 PC is Windows** (Git Bash, the venv's `python`, `PYTHONUTF8=1`). Its Pillow has no raqm, so Latin px²/char differs by up to ~3%, test-page digests differ and `test_rgb3_unsupported_pitch` fails for this reason only. Compare local rows with local rows.
- **Windows pages GPU memory instead of raising OOM** (5 parallel jobs stalled): at most 3 on the 3090. Elsewhere size concurrency from the smoke's peak memory (sum < ~85% of VRAM).
- Measured speeds: RTX 3090 probe 0.015-0.06 s/step (cloud CPU ~3), DeepEncoder ~0.05 s/page, Latin rendering ~45 s/1000 pages (`EP_PAGE_CACHE=1`). RTX 4090 R3-1: 1.2 s/step, 11.4 GB peak, ~7 min load+eval per job.

## Cloud runner: resources, restarts, resuming

- Speed: ~0.6–0.8 s/step per Phase-2 run at 1 thread, so a 3500-step run ≈ 40 min; 26 runs ≈ 6 h on 3 cores, ~4.5 h on 4.
- **Disk is fixed (~39 GB) and has filled up** (checkpoint saves crashed; a full disk also truncated docs in a commit). Checkpoints are atomic (.tmp then replace): delete `*.tmp` before restarting. Snapshots (~725 MB), qwen25vl_h32 caches (~1.35 GB/500 pages) and old checkpoints add up: keep ≥ 3 GB free and log `df`.
- **cgroup memory ~15.7 GB:** the OOM killer took the largest job twice (5.2 GB probe, 5.8 GB cache build). Budget peak RSS per job; labels for 500 rgb3_lg pages alone are ~1.5 GB.
- **The container is suspended soon after the session goes idle** (no turn, no running subagent): nohup'd jobs die 5-15 min later and the next wake shows `uptime` 0 min (not memory: logged < 7 GB). Keep a background subagent waiting on the queue (run_in_background loops that re-arm). The container can also restart when the session resumes. So **make long jobs checkpoint and resume** (`run_probe2.py --ckpt-every 200`, `finetune_qwen.py --ckpt-every 250`; resume is exact), and on every wake check `uptime`.
- A *restart* keeps the disk (data/, fonts, caches, pip): restart `phase2_driver.py` / `job_queue.py` (fresh queue file with only the remaining runs, fresh `--log-dir`) / `autosave_results.sh` with nohup. A *reclaim* loses all git-ignored state: redo setup first. `nohup scripts/autosave_results.sh 600` commits+pushes `runs.jsonl` every 10 min.
- Resuming: Phase 2 (no Claude calls) skips finished (scheme, em) rows; in-flight runs restart from step 0 (Phase 2 writes checkpoints only at the end). Phase 1 re-runs skip finished rows and read the Claude cache; errors (incl. usage limits) are not cached and are retried.
- `job_queue.py` only subtracts Phase-2 processes, so two queues oversubscribe the cores. To reorder a running queue, atomically rewrite only its unclaimed tail, keeping the first <claimed> lines byte-identical (claimed = number of `start job` lines in its log).
- Never edit a running bash script (bash reads it incrementally); to change plans, SIGKILL the wrapper bash only (its python child keeps running).

## Git / shell

- Never `pkill -f <name>` from a shell whose command line contains <name> (it kills itself); a `while pgrep -f <name>` loop matches itself. Use `pgrep -f "[j]ob_queue.py"`, and only if the plain string appears nowhere else in that command line (`pkill -f "[r]un_probe.py --encoding x"` then `nohup python3 run_probe.py --encoding x` in one call kills its shell, exit 144). Put the pkill in its own call.
- A stop hook demands a clean tree. Live Phase-2 logs/curves are git-ignored while running; `git add -f` them when a grid finishes. Other long jobs: snapshot-commit (`WIP:`) and push.

## Headless Claude Code agents (`claude -p` as the test subject)

These were learned in the repo-atlas and plan-quality tests; the entries with their context are in
[FINDINGS.md](FINDINGS.md).
- **Claude Code's auto-memory (`~/.claude/projects/`) is shared across headless runs:** one run read another's
  notes. Set `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`, or delete each run's store.
- **Block the package index for agents solving tasks from public repositories:** one run leaked the fix by
  downloading a later PyPI release of the package under test.
- **A background job inside an agent run produces several result events:** sum turns and tokens over all of them.
  `assistant` stream events carry only a message's starting output count: use `--include-partial-messages` and
  read `message_delta`.
- **The CLI prices Opus cache reads at $0.20 per million tokens (0.05x input), not $0.40:** a live cost guard that
  assumed 0.1x ran 40% high and cut two runs early (they were resumed with `claude -p --resume`).
- **A context is read from the prompt cache only when it is its own first user message,** separate from the
  questions.
- **Repeat calls need `claude_eval.transcribe(..., cache_salt=...)`,** or the response cache returns the first
  answer again.
- **A subagent stopped by a session restart cannot be resumed:** have long runners write rows that the main
  session can analyse.
- **Grade each cargo test binary separately, with its own timeout,** and key tests by binary: one slow test hid
  the others.
- **Docker has no daemon on the cloud runner;** a benchmark's pytest suite can usually be run directly.
- **`repo_atlas.collect` leaves out files that contain its marker characters** (`§`, `▶`).

## Local model server (`experiments/claude_qa/local_qa.py`, not yet run against a real model)

- **Finished arms are skipped on a rerun,** so to redo an arm (after fixing the server) delete its row from
  `results/local_qa/<run>/runs.jsonl` first. A failed call is never written as a row.
- **`run --dry-run` still writes `questions_d*.json`** into `results/local_qa/<run>/`. On the cloud runner delete
  the directory afterwards (it is not git-ignored); on the 3090 PC it is part of the results.
- **Pass the thinking switch as one unquoted shell variable:**
  `X='--extra {"chat_template_kwargs":{"enable_thinking":false}}'` then `$X`. The JSON must contain no spaces.
- **`llama-server` flag names change between builds** (image token limits, multi-token prediction): check
  `llama-server --help` before trusting [gpu/LOCAL_QWEN.md](gpu/LOCAL_QWEN.md)'s command line.

## Scripts without `--help`

- **`scripts/merge_phase2.py` takes run names as bare arguments and has no option parser:**
  `merge_phase2.py --help` looks for a run called `--help`, finds nothing and **overwrites
  `results/phase2/merged/runs.jsonl` with an empty file** (seen 2026-11-13; `git checkout` restores it, and the
  plain command reproduces the committed file).
- **`experiments/rag/analyse_rag.py` and `experiments/compaction/analyse_scale.py` ignore `--help` and run the
  analysis;** `encoder_probe/experiments/analyse_g2lite.py --help` fails with a file-not-found error (it expects
  result paths). Read the module docstring instead.
