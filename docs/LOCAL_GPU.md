# Running the encoder-probe experiments on a local GPU (RTX 3090 + Ryzen 9 5900X)

The remaining round-2 encoder-probe work (`encoder_probe/ROUND2.md`) is slow on the 4-vCPU cloud runner. This is
the runbook for a Linux PC with an RTX 3090 (24 GB, Ampere, native bf16) and a Ryzen 9 5900X (12 cores / 24 threads,
AVX2 only). All run times below are **estimates**, not measurements: the CUDA path was written on a machine without
a GPU and has not been run yet (see "Smoke test" first).

## 1. Setup

```bash
git clone https://github.com/StephenFlavin/visual-bpe && cd visual-bpe
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

## 4. Status in brief (2026-11-04)

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
Not built yet. Per docs/IDEAS.md: add `rgb6_p{6,8}` to `encoder_probe/ep/rgb_overlay.py`, with 2 binary-thresholded
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
