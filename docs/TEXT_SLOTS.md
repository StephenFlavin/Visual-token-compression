# Test B: text-side slots against page pixels, same reader, same budget

*Pre-registered 2026-11-11, before any row. Code: `encoder_probe/experiments/text_slots.py`, analysis
`analyse_slots.py`, tests `encoder_probe/tests/test_text_slots.py` (tiny CPU model). Background:
[REVIEW_2026-11-11.md](REVIEW_2026-11-11.md). The 3090 prompt is in [HANDOFF.md](HANDOFF.md).*

## Question

With the same LLM (Qwen2.5-VL-3B's), the same training text, the same number of training samples and the same
measure, does a slot built from r text tokens carry more per context position than a vision token built from a
rendered page?

**Why it matters.** Every glyph, colour and overlay code in this repo is a way to put more text into one context
position through the vision tower. If a slot made straight from the tokens carries more at the same budget, the
vision tower is a detour for any model we can train, and the dense-code line has no use left. If it does not,
pixels are the better pipe and Stage C goes ahead.

## Why not Stage 0 again

Stage 0 ([OWN_ENCODER.md](OWN_ENCODER.md)) asked a similar question and its 4x control failed: 64.0% CER at 8000
steps, outputs on topic but almost never an exact prefix (0.3%). Three causes, each changed here:

| Stage 0 | Here | Why |
|---|---|---|
| m global memory slots: any slot may hold any part of the window | **local slots**: slot i is a linear map of its own r token embeddings, initialised to their mean | copying becomes unpacking r tokens per slot; at r = 1 the slots are the token embeddings, so the starting point is exact |
| frozen reader | **reader with LoRA** (r = 16, alpha 32, lr 2e-4: the R3-1 image arms' LoRA) | every image arm trained something on the reader's side (R3-1: LoRA; Runs 1-2: 352M ViT parameters) |
| natural text | **trained on word-shuffled text**, as the Runs 1-2 image arms are | with no language prior the gist shortcut does not lower the loss; only copying does |
| each ratio from scratch | **ratio ladder with warm starts** | each step only has to learn to unpack twice as much |

A frozen-reader variant (`--decoder frozen`: the encoder alone is trained, the LLM is unmodified) is the exact
analogue of Runs 1-2 and runs second. Passing slots as key-value states per layer (500xCompressor) was considered
and is not built.

## Setup

* **Encoder:** frozen token embeddings, a 2-layer transformer over the window whose output starts at zero, then
  one linear map per slot over its r embeddings. About 80M to 140M trainable parameters at r = 2 to 16.
* **Reader:** the LLM reads `[m slots]["Repeat the text exactly:"]` and rebuilds the 480 tokens.
* **Data:** 480-token windows of WikiText-103 (`data/train.txt`), words shuffled within the window for training;
  64 held-out windows (`data/test.txt`) read as natural and as shuffled text.
* **Budget:** batch 8. The image arms of Runs 1-2 saw 6000 x 8 = 48,000 pages of up to 2000 characters; a window
  here is ~2080 characters. The ladder reaches ratio 8 with exactly 48,000 samples in total.
* **Measure:** bits per slot = (reference loss − loss) x 481 tokens / m / ln 2, per window, bootstrap 95% CI.
  The reference is the lower of the blind control (the same reader given another window's slots) and the prior (the
  unmodified LLM with no slots). This is `capacity_curve.py`'s bits per vision token with slots in place of vision
  tokens. Also: greedy CER, recovered fraction, recovered chars per slot.

## Arms

| Arm | Ratio (chars per slot) | Steps | Starts from | Samples so far | Role |
|---|---|---|---|---|---|
| L0 | 1 (4.3) | 0 | – | 0 | the untrained reader copying its own embeddings (reported, not a gate) |
| L1 | 1 (4.3) | 250 | – | 2,000 | **gate G1** |
| L2 | 2 (8.7) | 1000 | L1 | 10,000 | **gate G2** |
| L3 | 4 (17) | 2000 | L2 | 26,000 | |
| L4 | 8 (35) | 2750 | L3 | 48,000 | the same-budget arm: denser than latin_p8 (27-28 chars per token) |
| L5 | 16 (69) | 6000 | L4 | 96,000 | twice the budget: reported as such, cannot give a same-budget win |
| F2, F4, F8 | 2, 4, 8 | 1000, 2000, 2750 | –, F2, F4 | 46,000 | frozen reader |

The smoke runs at ratio 1, the longest sequence (969 positions). If batch 8 does not fit (smoke peak above
20 GB), use batch 4 and twice the steps for every arm.

## Gates and decision rules

* **G1:** L1 reads shuffled text at ≤ 1% CER. A failure is a pipeline or prompt fault: stop and report.
* **G2:** L2 reads shuffled text at ≤ 3% CER. If it fails with the validation loss still falling over its last
  three evaluations, rerun L2 once at 3000 steps (a dated amendment, and the later arms then exceed the budget);
  otherwise stop and report.
* **Win** (per image arm and text set, computed by `analyse_slots.py --image ...`): a text arm with at least the
  image arm's chars per position whose lower CI bound of bits per slot is above the image arm's upper bound, with
  at most 48,000 samples. Image arms: frozen latin_p8 (28.1 chars per token; 12.6 bits natural, 25.7 shuffled) and
  the trained latin_p8 / p6 / p5 arms of Runs 1-2 as they finish.
* **If the text arms win** against the trained Latin arms at equal or higher density: Stage C and colour Run 4 are
  dropped; the glyph line closes with Stage A. Next is the use test below.
* **If they lose:** pixels are the better pipe for this recipe and budget; Stage C proceeds as planned.
* Never choose anything on test-window CER: curves are on the 16 validation windows.

## Predictions (before any row)

1. G1 and G2 pass.
2. L3 (ratio 4) reads shuffled text at ≤ 5% CER.
3. L4 (ratio 8, 35 chars per slot) carries more than frozen latin_p8's upper bounds (13.1 bits natural, 26.4
   shuffled) and reads shuffled text at ≤ 20% CER.
4. L5 (ratio 16) is read in part (CER between 20% and 70%) and carries more bits per slot than L4.
5. The frozen reader is far behind at every ratio (at least twice the CER of the LoRA arm).

Predictions 3-4 rest on published results with about 1B training tokens (prior-art.md); these arms see 23M to 46M,
so they are the uncertain ones.

## What this does not show

* **Reconstruction is not use.** Published compressors of both kinds lose factual recall against full context. A
  win here earns the use test: needles and chains from slots (`longctx_bench.py`) against the same text as tokens
  and as page images, at equal context positions.
* **One LLM, one seed, 64 windows of English Wikipedia.**
* **A LoRA-adapted reader:** the slots are readable by the adapted LLM only. The frozen-reader arms say what the
  unmodified LLM reads.

## Runbook

```bash
python encoder_probe/experiments/text_slots.py --ratio 1 --smoke            # s/step, peak GiB, loss falls
T="python encoder_probe/experiments/text_slots.py --run-name tb"
E=encoder_probe/cache/ts_encoders
$T --ratio 1 --steps 0
$T --ratio 1 --steps 250
$T --ratio 2 --steps 1000 --init-from $E/tb__r1__shuffled__lora.pt          # then check G1 and G2
$T --ratio 4 --steps 2000 --init-from $E/tb__r2__shuffled__lora.pt
$T --ratio 8 --steps 2750 --init-from $E/tb__r4__shuffled__lora.pt
$T --ratio 16 --steps 6000 --init-from $E/tb__r8__shuffled__lora.pt
$T --decoder frozen --ratio 2 --steps 1000
$T --decoder frozen --ratio 4 --steps 2000 --init-from $E/tb__r2__shuffled__frozen.pt
$T --decoder frozen --ratio 8 --steps 2750 --init-from $E/tb__r4__shuffled__frozen.pt
python encoder_probe/experiments/analyse_slots.py encoder_probe/results/tb \
    --image latin_p8_trained:<chars/token>:<upper bits natural>:<upper bits shuffled>   # from analysis_run12
```

Every arm resumes from its last checkpoint (every 500 steps) when the same command is run again, and is skipped
once its row exists. Each finished arm leaves its encoder and LoRA in `encoder_probe/cache/ts_encoders/` (up to
~0.7 GB, git-ignored); the next ratio needs the previous one's file. Rows: `encoder_probe/results/tb/runs.jsonl`.

## Results

Not run yet.
