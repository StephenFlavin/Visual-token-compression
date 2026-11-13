# Running the encoder-probe experiments on the local Windows PC (RTX 3090 + Ryzen 9 5900X)

**Status (2026-11-13): one job left for this PC, section 10** (page images on a local Qwen3.8-27B, no training;
the prompt is at the top of [HANDOFF.md](../HANDOFF.md)). Everything else here is a record of finished or parked
rounds, kept for its setup notes, limits and measured speeds.

For any other GPU machine (rented, cloud, unknown hardware) start from [GPU_RUNNER.md](GPU_RUNNER.md) instead.

The PC: RTX 3090 (24 GB, native bf16), Ryzen 9 5900X (12 cores / 24 threads, AVX2 only), **Windows 11**. Use Git
Bash, the venv's `python` (not `python3`) and `PYTHONUTF8=1`; the repo is at `/c/git/visual-token-compaction-run` and
`. .venv/env.sh` sets the environment. Sections 1-7 were written for Linux before the PC turned out to be Windows,
so their commands say `python3`. Windows pitfalls: [GOTCHAS.md](../GOTCHAS.md), "GPU runners". Times are
**estimates** unless marked measured.

| Section | What | Status |
|---|---|---|
| 1-3 | Setup, CUDA notes, smoke test for the encoder-probe code | Reference; used for round 2 |
| 4-7 | Round 2: probes on DeepSeek-OCR's frozen DeepEncoder | **Finished 2026-11-04.** Only sparse Latin (8.5 characters per vision token) is read: 18.9% character error at 1000 training pages, 14.9% at 5000; every denser code sits at the floor. [encoder_probe/RESULTS.md](../../encoder_probe/RESULTS.md) |
| 8 | Round 3 on Qwen2.5-VL-3B: frozen probe (R3-0), fine-tune (R3-1), long-context benchmark (R3-2) | **Not run on this PC.** R3-1 ran once on a rented RTX 4090 ([GPU_RUNNER.md](GPU_RUNNER.md)); R3-0 on CUDA and R3-2 never ran. The setup block is still what the PLAN_3090 runs use |
| 8 (setup) + [PLAN_3090.md](PLAN_3090.md) | Runs 1-2: Qwen's vision encoder trained on Latin and glyphs | **Parked.** Frozen baseline and smoke measured here on 2026-11-09 (3.93 s/step at batch 8, 17.9 GB); the driver was stopped in the wind-down. Resume command: [HANDOFF.md](../HANDOFF.md) section 3 |
| 9 | Test B: text slots against page pixels | **Closed, never run** ([TEXT_SLOTS.md](TEXT_SLOTS.md)) |
| 10 | Qwen3.8-27B: page images and generation speed | **Ready to run** ([LOCAL_QWEN.md](LOCAL_QWEN.md)) |

**Measured on this PC** (for sizing anything new): probe 0.015-0.06 s/step; DeepEncoder ~0.05 s/page; Latin
rendering ~45 s per 1000 pages; Qwen2.5-VL-3B with 16 vision blocks trained, batch 8: 3.93 s/step, 17.9 GB;
at most 3 probe jobs or one fine-tune at a time (Windows pages GPU memory instead of raising out-of-memory).

## 1. Setup

```bash
git clone https://github.com/StephenFlavin/visual-token-compaction && cd visual-token-compaction
git checkout claude/validation-harness
python3 -m venv .venv && . .venv/bin/activate          # Python >= 3.11 (the runner uses 3.13)
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cu128 torch   # any CUDA 12.x wheel your driver supports
pip install transformers safetensors huggingface_hub requests          # encoder code (runner: transformers 5.22.1)
sudo apt-get install fonts-dejavu-core                  # DejaVu Sans / Mono from /usr/share/fonts (often preinstalled)
scripts/fetch_fonts.sh                                  # other fonts into assets/fonts/
export EP_TRAIN_R2_CAP=400000000                        # before the first run: see "Text supply" below
python3 scripts/build_data.py                           # WikiText-103 -> data/ (~5 min)
python3 -m pytest -q encoder_probe/tests                # 84 tests, about 1 min, no GPU needed
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

* **Weights** are fetched automatically on first use into `encoder_probe/cache/weights/` (git-ignored). Only the
  vision tensors are downloaded, with HTTP range requests on the pinned Hugging Face revision (`ep/hf_partial.py`):
  DeepEncoder is about 0.8 GB of a 6.7 GB shard (Qwen vision tower 1.34 GB, only if you run `qwen25vl`; about 2.1 GB
  if you fetch both). No HF token is needed. To fetch them up front:
  `python3 -c "from encoder_probe.ep.vlm_encoders import load_vlm; load_vlm('deepencoder', 'cuda')"`.
* `data/`, `encoder_probe/cache/` (weights, token caches, page caches, checkpoints) are git-ignored and rebuild on demand.
* **Text supply.** Pages never repeat. Above about 1000 train pages the densest encodings need more than the default
  64M train chars. `data/train_r2.txt` is built on demand, so set `EP_TRAIN_R2_CAP` before it is first built (above).
  The prefix is identical, so the 200-page sets equal the cloud runner's. If the file already exists at 64M:
  `rm data/train_r2.txt` and rerun with the variable set. Measured train-page limits at 512 px with 64M chars:
  col_b1 ~240, col_b2 ~970, lg_free_h4 ~1400, rgb3_p6 ~1480, lg_pb_h5 ~1760, latin_p5 ~2600. Latin p8 and up fit
  5000 easily. A shortage fails loudly ("only N of M full train pages fit the text"), never silently.

## 2. What changed for CUDA (all scripts: `--device {auto,cpu,cuda}`, auto = cuda if available)

* `run_probe2.py` (round 2), `run_probe.py` (round 1) and `build_tokens2.py` take `--device`. On CUDA the probe and
  test tokens go to the GPU. Train tokens and uint8 pixels also go there if they fit in `--gpu-data-gb` (default
  4 GB per job), else each batch is copied over. The probe uses bf16 autocast, and the numpy batch sampling is
  identical. VLM encoders run in fp32 on CUDA (TF32 off; bf16 on the 3090 gave corr 0.9974 vs CPU fp32) with larger batches; SigLIP (round 1) stays fp32.
* Rows record `device` and `enc_compute` (the device/dtype that made the token cache). The skip key is unchanged:
  (encoding, encoder, cond, steps). **Use one `--run-name` per `--train-pages` value.** Checkpoints are per page count.
* Resume works across devices: checkpoints load with `map_location="cpu"`, the CUDA RNG state is saved too.
* `EP_PAGE_CACHE=1` pickles rendered pages to `encoder_probe/cache/pages/` so that jobs sharing (encoding, page size,
  page counts) render them once. Rendering is single-threaded CPU work (61-166 s per 216-page job on the runner, so
  roughly 20-40 min for 5000 pages): **use it on this PC.** A 5000-page Latin set is about 4 GB on disk. Delete the
  directory if you change a renderer. `run_probe2.py --pages-only ...` only renders and caches (a CPU job).
* **On the 5900X CPU**, `cpu_has_bf16()` is False (no AVX512-BF16/AMX), so CPU runs use fp32 with denormals flushed.
  That is intended: emulated bf16 was 3-10x slower on the runner.

## 3. Smoke test (about 2 minutes, run first)

```bash
export EP_PAGE_CACHE=1
python3 encoder_probe/experiments/run_probe2.py --device cuda --encoding latin_p32 --encoder deepencoder \
    --cond clean --train-pages 20 --test-pages 4 --steps 200 --eval-every 100 --run-name round2_smoke_gpu
python3 encoder_probe/experiments/run_probe2.py --device cuda --encoding latin_p16 --encoder identity32 \
    --cond clean --train-pages 20 --test-pages 4 --steps 200 --eval-every 100 --run-name round2_smoke_gpu
# DeepEncoder on CUDA vs CPU (both fp32) on 2 pages: expect corr >= 0.9999 (measured 0.999998 on the 3090)
python3 -c "
import numpy as np, torch
from encoder_probe.ep.token_cache import split_pages
from encoder_probe.ep.vlm_encoders import load_vlm
_, tr, _ = split_pages('latin_p8', 512, 2, 4)
x = np.stack([p.img for p in tr])
a, b = load_vlm('deepencoder', 'cuda')(x).float(), load_vlm('deepencoder', 'cpu')(x).float()
print('corr', torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item())"
```
The log line should say `device=cuda bf16=True train-data-on-gpu=True`. These are smoke rows (`smoke: true`), not results.
The first run also builds the `tr20` DeepEncoder token cache on the GPU. If anything fails here, fix it before queueing.

## 4. Round 2: state before the run (2026-11-04, superseded by the result)

*Superseded 2026-11-04 evening: round 2 finished on this PC (1000-5000 pages). Only latin_p32 (8.5 chars/token) reads
through DeepEncoder (18.9% at 1000 pages, 14.9% at 5000); every denser code sits at the floor. Results:
[RESULTS.md](../glyphs/RESULTS.md) and encoder_probe/RESULTS.md. The text below is the pre-run state.*

Round 2 targets DeepSeek-OCR's DeepEncoder (512 px Tiny mode, 8x8 = 64 tokens of 1280-d); Qwen was dropped (user
decision). Rows: `encoder_probe/results/round2/runs.jsonl` (cloud runner, fp32 probe, 200 train pages, 3000 steps).

* Through DeepEncoder with 200 training pages, **every encoding sits near the unigram floor**: latin_p16
  (36 chars/token) 78.3%, col_b4 72.9%, lg_free_h4 82.3%, lg_pb_h5 84.3%, and even the easy positive control
  latin_p32 (8.5 chars/token, far below the ~40 chars/token at which DeepSeek-OCR itself decodes) 89.0%, **with
  train CTC 0.04**. The probe memorises its training pages and is data-limited, so **none of these rows is a
  verdict yet**.
* With raw pixels at the same page geometry (identity32 ceiling) the same probe reads latin_p16 at 0.06%, latin_p8
  at 0.42%, and col_b2 at 8.7% (step 1500, still running). The page, reader and scoring work. The open question
  is whether the encoder or the 200-page probe is the bottleneck.
* Queued on the cloud runner: latin_p32 deepencoder/clean with 1000 and 2000 train pages (`results/round2_tp/`).
  Round 1 (SigLIP) found the same pattern: 5x more pages took latin_p10 from 61% to 43%.

## 5. Experiments, in priority order

Run names below start with `round2_gpu_`, so GPU rows never mix with the runner's rows. Write each block's commands
to a queue file and run it as in section 6.

### P0: data-scaling positive control (settles: is the probe data-limited, and at what page count N to run P1)

latin_p32 and latin_p16 through deepencoder/clean at 200 / 1000 / 3000 / 5000 train pages (the 200-page runs repeat
the runner's rows on GPU/bf16, as a cross-device check):

```bash
for N in 200 1000 3000 5000; do for e in latin_p32 latin_p16; do
  echo "python3 encoder_probe/experiments/run_probe2.py --pages-only --encoding $e --encoder deepencoder --train-pages $N" >> results/queue_cpu_p0.txt
  echo "python3 encoder_probe/experiments/run_probe2.py --encoding $e --encoder deepencoder --cond clean --steps 3000 --train-pages $N --run-name round2_gpu_tp$N" >> results/queue_gpu_p0.txt
done; done
```
If a curve (the row's `curve`, val CER every 500 steps) is still falling at step 3000, run the same command
with `--steps 10000` (same run name: steps is part of the key). Rough 3090 times: page rendering 20-40 min per
5000-page set (once, cached), DeepEncoder tokens about 1-3 min per 5000 pages, probe about 2-5 min per 3000 steps.
All are estimates.

**Decision.** If Latin reaches low CER with more data (latin_p32 < 3%, latin_p16 well below the floor), choose
**N** = the smallest page count beyond which latin_p32/latin_p16 improve by less than about a fifth (relative), and
re-run the round-2 grid at N (P1). If Latin stays near the floor at 5000 pages and 10k steps, more data is not the
fix: stop and report back (the probe design is the limit, or DeepEncoder tokens need a different reader).
Do not start P1 before this is settled.

### P1: the round-2 grid at N (settles the ROUND2.md decision rules for DeepEncoder)

All `encodings_for("deepencoder")` through deepencoder **clean and jpeg75** at N train pages, plus the identity32
ceilings:

```bash
N=3000   # the P0 winner
ENCS="latin_p5 latin_p6 latin_p8 latin_p10 latin_p16 latin_p24 latin_p32 col_b2 col_b4 lg_free_h4 lg_pb_h5 rgb3_p6 rgb3_p8 mono_p6 mono_p8"
for c in clean jpeg75; do for e in $ENCS; do
  echo "python3 encoder_probe/experiments/run_probe2.py --encoding $e --encoder deepencoder --cond $c --steps 3000 --train-pages $N --run-name round2_gpu_tp$N"
done; done > results/queue_gpu_p1.txt
for e in $ENCS col_b1; do
  echo "python3 encoder_probe/experiments/run_probe2.py --encoding $e --encoder identity32 --cond clean --steps 3000 --run-name round2_gpu_tp200"
done >> results/queue_gpu_p1.txt
```
* Add `--steps` as P0 showed. Pre-render the pages with `--pages-only` lines in a CPU queue, as in P0.
  `col_b1` gets at most ~240 train pages at 64M chars (~1500 at 400M), so run it at the largest count that fits.
* **identity32 ceilings at 200 pages.** identity32 tokens are 3.1 MB/page in fp32, so 3000 pages need ~19 GB of host
  RAM while they are built. A ceiling only gets better with more data, so a 200-page ceiling under 10% satisfies
  the guard. Run them at N only if RAM allows (1000 pages is about 7 GB).
* Estimate: 46 probe jobs (30 deepencoder + 16 ceilings), each a few minutes on the GPU plus a one-off page
  render per encoding. With 3 jobs at a time, expect several hours, dominated by rendering at large N.

**Decision rules** (ROUND2.md; "real encoder" = deepencoder):
* **Colour blocks / rgb3** are worth GPU work only if, through deepencoder with jpeg75, they reach **>= 2x the chars
  per vision token of the best Latin rendering at 3% CER** (C@3%, same encoder and condition). Otherwise the idea is
  dead for VLMs with a pretrained encoder.
* **Grey-level learned banks** stay in the plan only if **lg_free_h4 beats lg_pb_h5** (lower CER at its 1.26x
  density) through deepencoder/jpeg75. Otherwise banks are constrained to whole pixels.
* **Informativeness guard:** a row through an encoder is interpreted only if its identity ceiling at that geometry is
  **< 10% CER**. If no Latin encoding reaches 3% through the encoder, use C@10% and report it as weaker evidence.
* Caveat: a probe on frozen tokens is not an LLM reading them. A positive result means the information survives
  the encoder.

### P2: RGB overlay grid (only if colour survives P1)
Gate: run it only if colour or rgb3 passes, or nearly passes, the colour rule in P1. If colour is dead through
deepencoder, skip it. The 48 commands are in `encoder_probe/experiments/rgb_overlay_grid.txt`: round-1
`run_probe.py`, 224 px, identity / siglip_l6, clean / jpeg75, run name `rgb_overlay`. They run on CUDA as they are
(`--device auto`); SigLIP is downloaded from HF on first use.
`cp encoder_probe/experiments/rgb_overlay_grid.txt results/queue_gpu_p2.txt`. Each job takes a few minutes (estimate).
Decision rule (RGB_OVERLAY.md): pursue RGB overlay only if rgb3 beats the best latin/mono C@3% through the real
encoder with jpeg75. rgb3_p5 repeats some pages under round-1 `make_pages` (RGB_OVERLAY.md caveat).

### P3: rgb6 (only if rgb3 keeps most of its 3x through the real encoder with jpeg75)
Not built yet. Per docs/background/IDEAS.md: add `rgb6_p{6,8}` to `encoder_probe/ep/rgb_overlay.py`, with 2 binary-thresholded
layers per channel at ink depths 1/2 and 1/4, channel-tagged classes and 6 interleaved labels per cell (check the
CTC frame budget), and run it with the rgb3 conditions. Predicted to work only for identity/clean.

## 6. Running jobs on this PC

* **GPU: 3 jobs at once** to start with. One probe job (40M parameters, 2 pages x 16 lines per step) leaves most of
  the 3090 idle and needs about one CPU core to feed it. Each job uses ~1-2 GB of GPU memory plus up to
  `--gpu-data-gb`. Watch `nvidia-smi`: go to 4-6 jobs if utilisation stays under ~70% and memory allows. Run at most
  2 of the 5000-page jobs together (each peaks at ~9 GB of host RAM while it loads).
* `scripts/job_queue.py` works for GPU jobs too: `--cores` is just the number of concurrent jobs (default 4; no code
  edit is needed for 12 cores). It sets OMP_NUM_THREADS=1, passes the environment through (so `EP_PAGE_CACHE=1`
  reaches the jobs) and only subtracts Phase-2 processes, so two queues don't see each other. Pick `--cores` so the
  two sum to at most 12 physical cores:
  ```bash
  export EP_PAGE_CACHE=1 EP_TRAIN_R2_CAP=400000000
  nohup python3 scripts/job_queue.py results/queue_cpu_p0.txt --cores 8 --log-dir results/queue_logs/cpu_p0 > results/queue_cpu_p0.log 2>&1 &
  nohup python3 scripts/job_queue.py results/queue_gpu_p0.txt --cores 3 --log-dir results/queue_logs/gpu_p0 > results/queue_gpu_p0.log 2>&1 &
  ```
  CPU work (page pre-rendering, `scripts/build_data.py`, any CPU-only grid) goes in the CPU queue with about 8-9
  workers while 3 GPU jobs run (10-11 if the GPU runs one job), leaving cores free to feed the GPU.
  Restarting a queue re-claims from line 1. Finished probe rows are skipped and `--pages-only` exits at once if
  cached, but use a fresh `--log-dir` per round anyway.
* If a probe job starts before its pages are cached, it renders them itself (and caches them). If its token cache is
  missing, it builds it under the global VLM lock (one VLM load at a time; other jobs wait). To build token caches
  in one go instead:
  `python3 encoder_probe/experiments/build_tokens2.py --encoders deepencoder --conds clean jpeg75 --train-pages N`.
* **CPU-only fallback** (`--device cpu`): the 5900X should give roughly 3x the runner's throughput (12 single-thread
  jobs vs 4). Estimate: about 2 s/step per job, so a 3000-step job takes ~1.7 h.

## 7. Analysis and sending results back

```bash
python3 encoder_probe/experiments/analyse2.py encoder_probe/results/round2*/runs.jsonl
```
It prints, per setting (encoder, cond, train pages): test CER by encoding, C@10% / C@3% per family on the
chars-per-VLM-token axis, and the identity-ceiling guard. (`analyse.py` is the round-1 table: 224 px, 256 tokens.)
Train CTC is only in the job logs (`[probe2] step ... ctc ...`): compare it with val CER to spot memorisation.

Send the rows (and the small text logs) back on the branch:
```bash
git add encoder_probe/results/round2_gpu_*/runs.jsonl
git add -f results/queue_logs/gpu_p0 results/queue_logs/gpu_p1   # optional, git-ignored by default
git commit -m "Round 2 GPU: <what ran>"
git pull --rebase origin claude/validation-harness && git push origin claude/validation-harness
```

## 8. Round 3 on the 3090 (encoder_probe/ROUND3.md)

**Status (2026-11-05):** `finetune_qwen.py` ran as built on a Linux RTX 4090 (smoke + 1000-step arms with blind
controls, [GPU_RUNNER.md](GPU_RUNNER.md); results in [ANALYSIS.md](../background/ANALYSIS.md)): 1.2 s/step, peak 11.4 GB at batch 4.
Nothing below has run on this Windows PC yet, and R3-0 (frozen Qwen probe on CUDA) and R3-2 (`longctx_bench.py`)
have not run on any GPU. The follow-up arms planned then (latin_p8/p6/p5 at 5000 steps with blinds, then a short vision-side diagnostic)
were overtaken by [PLAN_3090.md](PLAN_3090.md), whose runs use this section's setup; the R3-1 block below is the
original grid and was never run here. *(Status updated 2026-11-13.)*

### Setup (Windows, Git Bash, the round-2 venv)
```bash
cd /c/git/visual-token-compaction-run && . .venv/env.sh            # PYTHONUTF8=1 EP_PAGE_CACHE=1 EP_TRAIN_R2_CAP=400000000, venv on PATH
git pull --rebase origin claude/validation-harness
pip install "peft>=0.17"                               # now in requirements.txt (runner tested 0.21.2); pulls accelerate
python -m pytest -q encoder_probe/tests/test_round3.py # ~1 min on the CPU, no download (tiny model)
# full Qwen2.5-VL-3B-Instruct weights, ~7.5 GB, into the Hugging Face cache (HF_HOME or ~/.cache/huggingface),
# pinned revision; fetch once before queueing so two jobs never download at the same time:
python -c "import torch; from encoder_probe.ep.qwen_ft import load_model; m, t = load_model(torch.device('cuda')); print(sum(p.numel() for p in m.parameters()))"
```
Expect 3754622976 parameters (counted on the meta device from the config). No torchvision and no HF processor
are needed: inputs are built by `ep/qwen_ft.py` (checked equal to the checkpoint's chat template and image-token
expansion). Windows: use `python`, not `python3`, in queue lines; keep `PYTHONUTF8=1`; file locks go through
`ep/flock.py` (msvcrt), as in round 2.

### GPU job limits
* **At most 2 Qwen jobs at once** (R3-0 probe jobs with Qwen tokens), and **one `finetune_qwen.py` or
  `longctx_bench.py` job at a time, alone on the GPU**, until `nvidia-smi` shows headroom. Windows pages GPU memory
  to host RAM instead of failing with OOM: jobs then stall silently (round 2: 20 min with 5 jobs). Watch log
  progress and checkpoint mtimes.
* VRAM: finetune, batch 4, windows of <= 2000 chars (~800-token sequences): estimated ~12-15 GB (bf16 weights
  7.5 GB + fp32 trainables, grads and AdamW for 145.5M params 2.3 GB + checkpointed activations 2-4 GB); **measured
  11.4 GB peak allocated (~12.3 GB in nvidia-smi) on the RTX 4090**, so one finetune job at a time on 24 GB.
  If it pages, use `--micro-batch 2` (same effective batch). longctx_bench at 32k positions: weights 7.5 GB + KV
  1.2 GB + prefill activations 2-3 GB = **~11-13 GB**; text input at 32k is the peak (LM-head logits are computed for
  the last position only).
* Disk: a finetune resume checkpoint is ~1.75 GB (trainables + AdamW, `encoder_probe/cache/ckpt/r3ft__*.pt`, deleted
  when the job ends); the kept adapter is ~0.6 GB (`encoder_probe/cache/r3_adapters/<run>__<encoding>__<steps>.pt`).
  Keep >= 3 GB free: the 4090 runner's disk quota filled and killed a job.

### Smokes (run first, one at a time)
```bash
# R3-0: the qwen25vl CUDA path has never run (it builds a 20-page token cache under the VLM lock)
python encoder_probe/experiments/run_probe2.py --device cuda --encoding latin_p16 --encoder qwen25vl --cond clean \
    --train-pages 20 --test-pages 4 --steps 200 --eval-every 100 --run-name round3_qwen_smoke
# R3-1: 500 steps, 100 train / 4 test pages; check the loss falls, val_loss falls, generation returns text
python encoder_probe/experiments/finetune_qwen.py --encoding latin_p8 --smoke          # -> results/round3_ft_smoke/
# R3-2: base model, 10k-char documents (text ~2.4k positions, latin_p8 2 pages)
python encoder_probe/experiments/longctx_bench.py --inputs text latin_p8 --lengths 10000 --tasks needle cloze \
    --depths 0 0.5 1 --n-cloze 2 --run-name round3_lc_smoke
```
Each should take minutes (estimate). The finetune log prints `device=cuda bf16=True trainable={...}` and s/step;
the bench prints positions, KV MB and prefill seconds per item. If anything fails here, fix it before queueing.

### R3-0: frozen Qwen probe (run names `round3_qwen_tp1000`, ceilings `round3_qwen_tp200`)
```bash
ENCS="latin_p5 latin_p6 latin_p8 latin_p10 latin_p16 latin_p24 mono_p6 mono_p8 col_b2 col_b4 rgb3_p6 rgb3_p8 lg_free_h4 lg_pb_h5"
for e in $ENCS; do   # CPU queue: pre-render pages (identity28 and qwen25vl share the 448-px page cache)
  echo "python encoder_probe/experiments/run_probe2.py --pages-only --encoding $e --encoder qwen25vl --train-pages 1000"
  echo "python encoder_probe/experiments/run_probe2.py --pages-only --encoding $e --encoder qwen25vl --train-pages 200"
done > results/queue_cpu_r3_0.txt
{ echo "python encoder_probe/experiments/build_tokens2.py --encoders qwen25vl --conds clean jpeg75 --train-pages 1000 --encodings $ENCS"
  for e in $ENCS; do
    echo "python encoder_probe/experiments/run_probe2.py --encoding $e --encoder identity28 --cond clean --steps 10000 --run-name round3_qwen_tp200"
  done
  for c in clean jpeg75; do for e in $ENCS; do
    echo "python encoder_probe/experiments/run_probe2.py --encoding $e --encoder qwen25vl --cond $c --steps 10000 --train-pages 1000 --run-name round3_qwen_tp1000"
  done; done; } > results/queue_gpu_r3_0.txt
nohup python scripts/job_queue.py results/queue_cpu_r3_0.txt --cores 8 --log-dir results/queue_logs/cpu_r3_0 > results/queue_cpu_r3_0.log 2>&1 &
nohup python scripts/job_queue.py results/queue_gpu_r3_0.txt --cores 2 --log-dir results/queue_logs/gpu_r3_0 > results/queue_gpu_r3_0.log 2>&1 &
python encoder_probe/experiments/analyse2.py encoder_probe/results/round3_qwen_tp1000/runs.jsonl encoder_probe/results/round3_qwen_tp200/runs.jsonl
```
* `--cores 2`: Qwen token jobs. The builder holds the VLM lock; probe jobs whose cache is missing wait for it.
  Qwen tokens are 256 x 2048 bf16 per page (~1 MB): 1000 pages ~1 GB per (encoding, cond) cache, ~30 GB for the
  grid on disk.
* To see the moonshot codes frozen too, add `rgb3_lg_pb_h5 rgb3_lg_free_h4` to ENCS (their identity28 ceilings run
  on the cloud runner, `round3_cpu_tp200`; the 1185-class probe is ~2x slower per step).
* Decision (ROUND3.md): pick the 1-2 best dense codes by chars/token at 10% CER; if none survives, still take
  col_b4 and rgb3_p8.
* Quote every probe CER next to its code's blind floor and op split (`run_g2lite.py --arm blind`; GPU_RUNNER.md
  section 4). GridProbe under-reads Latin through Qwen (ROUND3.md), so Latin probe rows are lower bounds.

### R3-1: fine-tuning arms (run name `round3_ft`; smokes `round3_ft_smoke`)
```bash
ARMS="latin_p5 latin_p8 col_b4 rgb3_p8 rgb3_lg_pb_h5 rgb3_lg_free_h4"   # replace col_b4 rgb3_p8 by the R3-0 picks
for e in $ARMS; do echo "python encoder_probe/experiments/run_probe2.py --pages-only --encoding $e --encoder qwen25vl --train-pages 1000"; done > results/queue_cpu_r3_1.txt
for e in $ARMS; do echo "python encoder_probe/experiments/finetune_qwen.py --encoding $e --smoke"; done > results/queue_gpu_r3_1.txt
for e in $ARMS; do echo "python encoder_probe/experiments/finetune_qwen.py --encoding $e --steps 5000 --run-name round3_ft"; done >> results/queue_gpu_r3_1.txt
for e in $ARMS; do echo "python encoder_probe/experiments/finetune_qwen.py --encoding $e --steps 5000 --run-name round3_ft --blind-images"; done >> results/queue_gpu_r3_1.txt   # -> round3_ft_blind
nohup python scripts/job_queue.py results/queue_cpu_r3_1.txt --cores 8 --log-dir results/queue_logs/cpu_r3_1 > results/queue_cpu_r3_1.log 2>&1 &
nohup python scripts/job_queue.py results/queue_gpu_r3_1.txt --cores 1 --log-dir results/queue_logs/gpu_r3_1 > results/queue_gpu_r3_1.log 2>&1 &
```
* **Fairness guard:** never change a flag for one arm only. Every row stores `fair` and `fair_hash` (steps, batch,
  lrs, LoRA config, unfrozen layers, pages, task mix, window size, eval settings); all `round3_ft` rows must share
  one `fair_hash`. Changing a setting = a new run name for every arm.
* The pages-only lines fill the cache key `(encoding, 448, 1000, 16)` that the 5000-step runs read (the
  `rgb3_lg_*` pages need ~82M train chars for 1000 pages, hence `EP_TRAIN_R2_CAP=400000000`). Smokes use 100/4
  pages and render their own (~1-3 min).
* **Blind control for every arm** (`--blind-images`: each page's image swapped for another page's, targets
  unchanged, same `fair_hash`, run name gets `_blind`). Floors differ by code (latin_p5 77.8% CER, latin_p8 88.5%),
  so quote every CER next to its own blind.
* Rows: `encoder_probe/results/round3_ft/runs.jsonl`, one per eval cond (clean, jpeg75): `cer`, `cloze_acc`,
  `mc_acc`, `chars_per_token`, `curve` [(step, train-loss EMA, val loss)], `s_per_step`, `peak_gpu_bytes`, and `ops`
  (sub / ins / del, `len_ratio`, `prefix_frac`): report the op split with each CER. Resume:
  restart the queue with a new `--log-dir`; a job continues from its last `--ckpt-every` (250) checkpoint, exactly.
* Estimate: 5000 steps at ~1.2 s/step (4090, measured; the 3090 is likely slower) = ~1 h 45 min per job incl. load
  and eval, so 6 arms + 6 blinds ≈ 20+ h. Run the Latin arms first if time is short; prediction 2 needs the best Latin
  at 3%.

### R3-2: long-context benchmark (run names `round3_lc_base`, `round3_lc_ft`)
```bash
LENS="10000 30000 100000 300000 1000000"
A=encoder_probe/cache/r3_adapters
{ echo "python encoder_probe/experiments/longctx_bench.py --inputs text latin_p8 --lengths $LENS --run-name round3_lc_base"
  for e in latin_p8 col_b4 rgb3_lg_pb_h5; do   # each fine-tuned model reads its own encoding; text as the reference
    echo "python encoder_probe/experiments/longctx_bench.py --inputs text $e --lengths $LENS --adapter $A/round3_ft__${e}__5000.pt --run-name round3_lc_ft"
  done; } > results/queue_gpu_r3_2.txt
nohup python scripts/job_queue.py results/queue_gpu_r3_2.txt --cores 1 --log-dir results/queue_logs/gpu_r3_2 > results/queue_gpu_r3_2.log 2>&1 &
```
* Default tasks: needle at depths 0/0.25/0.5/0.75/1 and 5 cloze questions per document (one prefill each). Inputs
  over 32768 KV entries are written as `skipped: true` rows (text beyond ~140k chars, latin_p8 at 1M). Rows hold
  `acc`, `positions`, `kv_bytes` (from the cache tensors; = 36,864 B x positions in bf16), `peak_gpu_bytes`,
  `prefill_s`, `vision_s`, `chars_per_position`, and the per-item answers.
* Beyond 32k (optional, record it): `--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-positions 131072`
  with its own run name. Untested with M-RoPE on the real model; the KV at 128k is ~4.8 GB plus larger prefill
  activations: check VRAM on a single length first.
* Estimate (not measured): a 32k-position prefill ~5-20 s on the 3090, so ~1-2 h per command line above.

### Sending results back
```bash
git add encoder_probe/results/round3_qwen_tp*/runs.jsonl encoder_probe/results/round3_ft*/runs.jsonl encoder_probe/results/round3_lc_*/runs.jsonl
git add -f results/queue_logs/gpu_r3_0 results/queue_logs/gpu_r3_1 results/queue_logs/gpu_r3_2   # optional
git commit -m "Round 3 GPU: <what ran>"
git pull --rebase origin claude/validation-harness && git push origin claude/validation-harness
```

## 9. Test B on the 3090: text slots against page pixels (docs/gpu/TEXT_SLOTS.md)

**Closed without being run (2026-11-12).** It would repeat published work for a payoff current local models
already get from their architecture. The section is kept so the test can be run as written if that changes.

**Why it was to run before the remaining image arms (as written on 2026-11-11).** It needs about one evening of GPU time and decides what the
other arms are for: if slots made from the text tokens carry more per position than vision tokens at the same
budget, the dense glyph arms (Stage C) and the colour overlay (Run 4) are dropped; the Latin arms stay, as the
image side of the comparison. Design, gates, rules and predictions are pre-registered in
[TEXT_SLOTS.md](TEXT_SLOTS.md); the session prompt is in the git history (`git show a46cd47:docs/HANDOFF.md`).

**Nothing here has run on a GPU.** The code passed its tests on the tiny CPU model only, so the smoke is the first
real run: read its log before queueing anything.

### Setup
Same venv, weights and data as section 8 (`. .venv/env.sh`, `python`, `PYTHONUTF8=1`; `data/train.txt` and
`data/test.txt` from `scripts/build_data.py`). No pages are rendered and no vision tower runs.
```bash
cd /c/git/visual-token-compaction-run && . .venv/env.sh
git fetch origin claude/sharp-mendel-kkhg4g && git merge origin/claude/sharp-mendel-kkhg4g
python -m pytest -q encoder_probe/tests/test_text_slots.py encoder_probe/tests/test_text_compress.py   # tiny model, ~20 s
```

### GPU job limits
* **One job at a time**, and not while the Runs 1-2 driver is training (17.9 GB per run): stop the driver at a
  checkpoint first (step 0 of the prompt in [HANDOFF.md](../HANDOFF.md)).
* **Memory (estimate):** Stage 0 peaked at 10.1 GB at batch 4 with two passes through the LLM. Test B makes one
  pass at batch 8 over at most 969 positions (ratio 1; 729 at ratio 2), so expect 10-18 GB. If `nvidia-smi` shows more than 20 GB
  in the smoke, use `--batch-size 4` and twice the steps for every arm. Windows pages GPU memory instead of
  raising an error: a sudden slowdown means the job no longer fits.
* **Time (estimate, scaled from Stage 0's measured 0.93 s/step on an RTX 4090: ~5,200 token positions per second
  with LoRA gradients and checkpointing):** on a 4090, 0.8 s/step at ratio 16 to 1.5 at ratio 1, so 2.9 h for the
  LoRA ladder (12,000 steps) and 1.5 h for the frozen-reader arms (5,750), plus ~5 min of loading and evaluation per
  arm: ~5 h. **On this 3090 (a 4090 is 1.3-1.8x faster in FP16 training): ~8 h, two power-on stretches.** Gates G1
  and G2 are known after ~1.2 h, and the same-budget arm L4 (8x) after ~3.5 h. An A100 is in the 4090's class
  (~5 h, ~4 h with `--no-grad-ckpt`, which its 80 GB allows); an H100 ~2.5-3 h. Every arm checkpoints every 500
  steps and resumes exactly with the same command.
* **Disk:** a checkpoint is up to ~2 GB (written next to the old one), each finished arm keeps ~0.2-0.7 GB in
  `encoder_probe/cache/ts_encoders/`, about 4 GB for all nine arms. The next ratio needs the previous arm's file,
  so do not delete them before the ladder is done.

### Smoke, then the arms
```bash
python encoder_probe/experiments/text_slots.py --ratio 1 --smoke     # 20 steps at the longest sequence: s/step, peak GiB, loss falls
```
Then the runbook block of [TEXT_SLOTS.md](TEXT_SLOTS.md), one command at a time, checking gates G1 and G2 with
`analyse_slots.py` after the third arm.

### What to look at
* `[ts] step N val_loss X blind Y`: the gap between the two is what the slots carry. At ratio 1 after 250 steps
  the validation loss should be near 0.
* `[ts] TEST ... read shuffled` and `read natural`: CER, bits per slot, recovered fraction, each next to its blind
  control and the unmodified LLM's prior.
* A loss that stops falling far above 0 at ratio 2 is the Stage 0 failure again: stop and report (gate G2).

### Sending results back
```bash
python encoder_probe/experiments/analyse_slots.py encoder_probe/results/tb
git add encoder_probe/results/tb docs/gpu/TEXT_SLOTS.md && git commit -m "Test B: <arms>"
git pull --rebase origin claude/validation-harness && git push origin claude/validation-harness
```

## 10. Qwen3.8-27B on the 3090: page images and generation speed, no training (docs/gpu/LOCAL_QWEN.md)

**What it answers.** Whether rendering a document as page images lets a local model hold about twice the text in
the same memory without getting answers wrong, as it does on Claude, and how much faster the model writes with the
multi-token prediction module it ships with. Design, predictions and what each outcome means are pre-registered in
[LOCAL_QWEN.md](LOCAL_QWEN.md), which also has the full runbook; the session prompt is at the top of
[HANDOFF.md](../HANDOFF.md).

**Nothing here has run against a real model.** `experiments/claude_qa/local_qa.py` passed its tests against a mock server
only. The first real request (step 2 of the prompt) is the check.

* **Different stack from sections 1-9:** no PyTorch job. The model runs in llama.cpp's `llama-server` (Windows CUDA
  release) and `local_qa.py` talks to it over HTTP; the venv is only needed for rendering and scoring.
* **GPU:** one server at a time, and not while the Runs 1-2 driver is training: stop the driver at a checkpoint
  first (step 0 of the prompt).
* **Memory (estimate):** 16.6 GB weights + 0.9 GB vision projector + ~3.7 GB cache for a 57k-token context = ~21 GB
  plus buffers. Fallbacks, in order: an 8-bit cache (`--cache-type-k q8_0 --cache-type-v q8_0`), the 15.4 GB
  `Q4_K_S` weights. Windows pages GPU memory instead of failing: a sudden slowdown means it no longer fits.
* **Disk:** ~19 GB of model files under `models/qwen38/` (git-ignored: add `models/` to `.git/info/exclude` if
  `git status` lists it).
* **Time (estimate):** under an hour of GPU time for the 15 arms (3 documents x 5 arms) with thinking off, plus
  downloads and setup.
* **Commands** (the same as in the prompt and in [LOCAL_QWEN.md](LOCAL_QWEN.md); checked against
  `local_qa.py --help` on 2026-11-13):
```bash
cd /c/git/visual-token-compaction-run && . .venv/env.sh
python -m pytest -q tests/test_local_qa.py                           # 5 tests, mock server
python experiments/claude_qa/local_qa.py run --dry-run --doc 0 1 2   # sizes only, no server
llama-server -m models/qwen38/Qwen3.8-27B-UD-Q4_K_M.gguf --mmproj models/qwen38/mmproj-F16.gguf \
  -ngl 99 -c 57344 -np 1 --image-max-tokens 4096 --port 8080          # in its own terminal; check flags with --help
X='--extra {"chat_template_kwargs":{"enable_thinking":false}}'
python experiments/claude_qa/local_qa.py run --doc 0 $X              # then analyse and the checks, then --doc 1 2
python experiments/claude_qa/local_qa.py speed --label plain $X      # and --label mtp after restarting the server
python experiments/claude_qa/local_qa.py analyse                     # -> results/local_qa/qwen38_27b/summary.md
```
* **Sending results back:**
```bash
git add results/local_qa docs/gpu/LOCAL_QWEN.md docs/FINDINGS.md docs/HANDOFF.md
git commit -m "Local Qwen3.8-27B: page images vs text, results"
git pull --rebase origin claude/validation-harness && git push origin HEAD:claude/validation-harness
```
