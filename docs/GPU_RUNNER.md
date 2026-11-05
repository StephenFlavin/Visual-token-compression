# GPU runner: 3-hour plan (any Linux + NVIDIA GPU)

For a rented or borrowed Linux machine with one NVIDIA GPU, for a limited time (planned: about 3 h). Nothing here
assumes a particular GPU or CPU: measure first, then size the run. The Windows/RTX 3090 specifics live in
[LOCAL_GPU.md](LOCAL_GPU.md); the design is in [encoder_probe/ROUND3.md](../encoder_probe/ROUND3.md); current state
and open items are in [HANDOFF.md](HANDOFF.md). Every time and memory figure below is an **estimate** unless it says
measured.

## Goal for the session
The one test the CPU runner cannot do: **R3-1 (gate G3)**, fine-tuning Qwen2.5-VL-3B so its own LLM reads each code.
The comparison is plain Latin against the moonshot `rgb3_lg_pb_h5` at an identical budget. The CPU probe can't measure
Latin through Qwen at feasible data sizes (HANDOFF.md), so the LLM-as-reader result is the first fair comparison.
In about 3 h: setup, smoke, then short arms at equal steps. A short result is a signal, not a verdict. Label it so.

## 0. Before you start (5 min)
* Check `git push` works from the runner. The machine is ephemeral: anything not pushed is lost. If push doesn't
  work, copy `encoder_probe/results/` out before the session ends.
* Python ≥ 3.11, CUDA driver ≥ 12.x, **~40 GB free disk**:
  * Qwen weights 7.5 GB;
  * page caches ~1 GB per 1000 pages per code;
  * finetune checkpoints ~1.75 GB each, adapters ~0.6 GB each.
* Note the hardware in your first commit message:
  ```bash
  nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g; df -h .
  python3 -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
  ```
  **R3-1 needs bf16** (Ampere or newer: A10/A100/L4/L40/H100/RTX 30xx+). On a pre-Ampere GPU (T4, V100) skip
  sections 2-5 and run **section 6 (G2 at scale)** instead: the probe runs in fp32 there.

## 1. Setup (~20-30 min, mostly downloads; run the steps in parallel where marked)
```bash
git clone https://github.com/StephenFlavin/visual-bpe && cd visual-bpe && git checkout claude/validation-harness
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu128 torch   # pick the wheel your driver supports
pip install "peft>=0.17"
scripts/fetch_fonts.sh                                   # needs raw.githubusercontent.com
ls /usr/share/fonts/truetype/dejavu/ || sudo apt-get install -y fonts-dejavu-core   # DejaVu is required
export EP_TRAIN_R2_CAP=400000000 EP_PAGE_CACHE=1         # BEFORE the data build: the moonshot pages need ~82M train chars
python3 scripts/build_data.py                            # WikiText-103 -> data/ (~5 min)
python3 -m pytest -q encoder_probe/tests                 # ~2 min, CPU only (if a test fails, fix or report before GPU work)
# (parallel) full Qwen2.5-VL-3B weights, ~7.5 GB, pinned revision; expect 3754622976 parameters:
python3 -c "import torch; from encoder_probe.ep.qwen_ft import load_model; m,t=load_model(torch.device('cuda')); print(sum(p.numel() for p in m.parameters()))"
```
**(parallel) Pre-render the training pages on the CPU.** This is CPU-bound and single-threaded per job: the moonshot
takes ~1 s/page, so ~17 min for 1000 pages; Latin ~1 min. Start it as soon as `data/` exists, with one job per code:
```bash
for e in latin_p8 latin_p5 rgb3_lg_pb_h5; do
  nohup python3 encoder_probe/experiments/run_probe2.py --pages-only --encoding $e --encoder qwen25vl --train-pages 1000 > results/pages_$e.log 2>&1 &
done
```
Linux renders with raqm: this matches the cloud runner's pages (the Windows PC's Latin pages differ by up to ~3% px²/char).

## 2. Smoke (~10-15 min). Do not skip: the fine-tune has never run with real weights on a GPU
```bash
python3 encoder_probe/experiments/finetune_qwen.py --encoding latin_p8 --smoke        # 500 steps, 100 train / 4 test pages
```
Check:
* the log shows `device=cuda bf16=True`;
* the loss falls;
* generation returns text;
* write down **s/step** and **peak GPU memory** (`nvidia-smi`).

If it fails, fix minimally, keep the CPU tests green, commit the fix, rerun. If it OOMs, add `--micro-batch 2`
(same effective batch), and then use it for **every** arm.

## 3. Size the run from the smoke
* Time left for training: `T_min = (minutes remaining) - 25` (eval ~5-10 min per arm + write-up + push margin).
* Arms, in priority order: `latin_p8` (~28 chars/token), `rgb3_lg_pb_h5` (moonshot, 322), `latin_p5` (~72, the
  pushed Latin baseline), then `lg_pb_h5` (~107) if time allows.
* **Blind controls (required):** `--blind-images` runs of latin_p8 and the moonshot at the same settings. Each
  page's image is swapped for another page's, the text targets stay with the true page, and the run name gets
  `_blind`. An LLM can write fluent English from any image, so a CER is only evidence of reading when it is well
  below its blind control. That makes 5 jobs: 3 arms + 2 blind.
* Jobs at once `K = floor(0.85 * VRAM / peak_memory_from_smoke)`, at most 5:
  * 24 GB cards (3090, 4090): K = 1 likely;
  * 48 GB cards (A40, A6000, L40, L40S): K = 2-3 likely.
  Measure, don't assume. With K > 1 each job runs somewhat slower, so re-measure s/step on the first parallel steps.
* Steps per arm, **identical for all 5 jobs** (fairness guard: same steps, batch, LRs, pages, every flag):
  `STEPS = floor(T_min*60 * K / (5 * s_per_step_at_K) / 250) * 250`, minimum 1000. If that's < 1000, drop latin_p5
  first, then the latin_p8 blind control. Never drop the moonshot's blind control while the moonshot runs.

## 4. Run
```bash
RUN=round3_ft_gpu3h   # one run name for all arms; rows must share one fair_hash
F=encoder_probe/experiments/finetune_qwen.py
{ echo "python3 $F --encoding latin_p8 --steps $STEPS --run-name $RUN"
  echo "python3 $F --encoding rgb3_lg_pb_h5 --steps $STEPS --run-name $RUN"
  echo "python3 $F --encoding rgb3_lg_pb_h5 --steps $STEPS --run-name $RUN --blind-images"   # -> ${RUN}_blind
  echo "python3 $F --encoding latin_p8 --steps $STEPS --run-name $RUN --blind-images"
  echo "python3 $F --encoding latin_p5 --steps $STEPS --run-name $RUN"; } > results/queue_gpu3h.txt
# keep EP_PAGE_CACHE=1 EP_TRAIN_R2_CAP=400000000 exported in this shell: the jobs must read the pre-rendered pages
nohup python3 scripts/job_queue.py results/queue_gpu3h.txt --cores 1 --log-dir results/queue_logs/gpu3h_1 > results/queue_gpu3h_1.log 2>&1 &
nohup scripts/autosave_results.sh 600 > results/autosave.log 2>&1 &     # commits+pushes runs.jsonl every 10 min
```
* `--cores K`: the number of concurrent GPU jobs from section 3 (replace the `--cores 1` above).
* Resume after a crash: start the queue again with a **new** `--log-dir`. Finished rows are skipped, and a job continues
  from its last checkpoint (every 250 steps). Never edit a running queue except by atomically rewriting its unclaimed
  tail (AGENTS.md).
* Watch: `tail -f results/queue_logs/gpu3h_1/*.log`, plus `nvidia-smi` and `df -h .` (keep ≥ 3 GB free; the CPU
  runner died of a full disk once).
* Optional, only if a GPU slot and memory are free: the frozen-Qwen probe for the moonshot (R3-0),
  `run_probe2.py --device cuda --encoding rgb3_lg_pb_h5 --encoder qwen25vl --cond clean --train-pages 1000 --steps 10000 --run-name round3_qwen_tp1000`.
  It measures the absolute question only: GridProbe can't read Latin through Qwen at these page counts.
* **Required baseline for every GPU probe run (run_probe2 / run_g2lite): the blind floor.** Run the same probe,
  recipe, pages and steps with each page's features replaced by another random page's
  (`run_g2lite.py --arm blind --encoding <code> ... --run-name <run>_blind`; seeded derangement of train and of test
  pages, labels and recon targets unchanged). That is the CER the probe gets from label statistics alone.
  * Quote every probe CER next to its code's blind CER.
  * Quote its op split (`ops`: sub / ins / del as fractions of ref chars, `len_ratio` = hyp / ref length).
  * A CER at or near the blind floor, with mostly deletions and a short `len_ratio`, is "not read", whatever its
    absolute value (2026-11-05: frozen moonshot 86% and direct 74.5% are short letter-frequency soup, 56-83%
    deletions).

## 5. Results and hand-back (last ~20 min, whatever finished)
* Rows: `encoder_probe/results/round3_ft_gpu3h/runs.jsonl`, one per eval cond, with `cer`, `cloze_acc`, `mc_acc`,
  `chars_per_token`, `curve`, `s_per_step`, `peak_gpu_bytes`, `fair_hash`. Check that all rows share one `fair_hash`.
* Read against ROUND3.md: G3 asks whether the best code's density at < 3% CER is ≥ 1.5x the best Latin density at
  < 3% CER. Report CER per arm with chars/token. Label the result **short-budget (STEPS steps)**: 5000+ steps was the
  plan, and a code that is still falling is undecided, not failed.
* Commit and push:
  ```bash
  git add encoder_probe/results/round3_ft_gpu3h/ && git add -f results/queue_logs/gpu3h_1
  git commit -m "Round 3 GPU (3 h, <GPU name>): R3-1 short arms, STEPS steps"
  git pull --rebase origin claude/validation-harness && git push origin claude/validation-harness
  ```
* Add a dated "GPU runner session" note to HANDOFF.md: hardware, measured s/step and memory, what ran, results, what
  was not run.
* Adapters (`encoder_probe/cache/r3_adapters/*.pt`, ~0.6 GB each) are git-ignored. Copy them off the machine if R3-2
  should reuse them; otherwise they're lost with the runner.

## 6. G2 at scale: the probe on any NVIDIA GPU, T4 included (a fallback or extra to R3-1)
G2-lite (encoder_probe/ROUND3.md) ran on CPU at 500 pages. Only the no-merger moonshot arm read above label statistics
(~29% vs ~15% correct chars), so the probe looked data-limited. This repeats it at 2000 pages with the blind floors.
`run_g2lite.py --device cuda` runs the cache build (Qwen ViT, fp32, TF32 off), the merger and the probe on the GPU.
The probe uses bf16 autocast if the GPU supports bf16, else fp32 (a T4 is fine: the probe is small). The CPU path is
unchanged (identical CER and curves to the CPU runs), but **the CUDA path has not run on a GPU yet: smoke first.**
Setup is section 1, minus the Qwen full-weight fetch (the ViT downloads its own ~1.3 GB of tensors) and minus peft.
```bash
export EP_PAGE_CACHE=1 EP_TRAIN_R2_CAP=400000000
R=encoder_probe/experiments/run_g2lite.py; TP=2000
# smoke (~5 min): tiny cache + 200 steps on the GPU
python3 $R --device cuda --build-only --encoding latin_p24 --train-pages 20 --test-pages 4
python3 $R --device cuda --arm direct --encoding latin_p24 --train-pages 20 --test-pages 4 --steps 200 --eval-every 100 --run-name g2gpu_smoke
# caches (one at a time on the GPU; page rendering is CPU-bound, ~1 s/page for the moonshot, so start them early)
for e in rgb3_lg_pb_h5 latin_p8 latin_p5; do echo "python3 $R --device cuda --build-only --encoding $e --train-pages $TP --test-pages 16"; done > results/queue_g2gpu_cache.txt
# arms at equal steps (same order as overnight: moonshot first), each code with its blind floor
for e in rgb3_lg_pb_h5 latin_p8 latin_p5; do for arm in direct frozen blind; do
  echo "python3 $R --device cuda --arm $arm --encoding $e --train-pages $TP --steps 6000 --run-name g2gpu_tp$TP"; done
  echo "python3 $R --device cuda --arm trained --merger-lr 1e-4 --encoding $e --train-pages $TP --steps 6000 --run-name g2gpu_tp$TP"
done > results/queue_g2gpu_runs.txt
nohup python3 scripts/job_queue.py results/queue_g2gpu_cache.txt --cores 1 --log-dir results/queue_logs/g2gpu_cache_1 > results/queue_g2gpu_cache_1.log 2>&1 &
# after the caches exist (runs wait for a missing cache under its lock anyway):
nohup python3 scripts/job_queue.py results/queue_g2gpu_runs.txt --cores 2 --log-dir results/queue_logs/g2gpu_runs_1 > results/queue_g2gpu_runs_1.log 2>&1 &
python3 encoder_probe/experiments/analyse_g2lite.py encoder_probe/results/g2gpu_tp$TP/runs.jsonl
```
* **Sizes (estimates):**
  * hidden-state cache ~5.4 GB per code at 2000 pages (bf16), plus the post-merger token cache ~2 GB, so keep
    ≥ 25 GB of disk free for 3 codes;
  * host RAM per run ~6-12 GB at 2000 pages (the states are memory-mapped; frozen precomputes fp32 tokens, ~4 GB);
  * rendering 2000 moonshot pages takes ~35 min on one core, so on a 2-core Colab start the moonshot cache first.
* **Concurrency:** `--cores 2` = two probe runs on the GPU at once. Raise it only if `nvidia-smi` shows headroom
  and host RAM allows.
* **Read it with section 4's blind-floor rule:** quote CER, op split and correct chars next to each code's blind floor.
  The pre-registered G2-lite decision rule (ROUND3.md) applies unchanged; the larger page count is the only change.
  Record it as a dated note before running.
* Steps 6000 is a starting point. Measure s/step on the first run and keep every arm at the same steps.
