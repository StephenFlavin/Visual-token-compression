# Round 3: can a small VLM learn to read dense codes, and does it shrink the KV cache?

Written 2026-11-04, before any round-3 code or run. Rounds 1-2 (RESULTS.md, ROUND2.md) are unchanged.

## Why
Rounds 1-2 showed the information is there in the pixels and lost in frozen encoders: with raw pixels (identity
ceilings) colour col_b2 reads at 0.02% CER at 1024 chars/token and rgb3_p6 at 0.01% at 677, but through frozen SigLIP
and frozen DeepEncoder every dense code sits at the floor (71-84%), and only latin_p32 (8.5 chars/token) reads
through DeepEncoder (19%). A frozen probe cannot say whether a VLM can *learn* to read these codes. The end goal is
practical: fit more document per GB of KV cache in a long-context LLM, measured locally on the RTX 3090.

**Key fact.** KV cache grows with context positions, and a vision token is one position like a text token. Saving =
chars per position. Qwen2.5-VL-3B (36 layers, 2 KV heads x 128, bf16) costs 36 x 2 x 2 x 128 x 2 B = 36,864 B
(36 KiB) of KV per position. **Verified 2026-11-04** from config.json at the pinned revision 66285546 (= hub `main`):
num_hidden_layers 36, num_key_value_heads 2, hidden_size 2048 / 16 heads = head_dim 128, max_position_embeddings
128000, rope_theta 1e6, M-RoPE sections [16, 24, 24], tied embeddings; the CPU tests check the cache-tensor bytes
against this formula. 1M chars as text (4.33 chars/token measured with the Qwen tokenizer on held-out WikiText)
~ 231k positions ~ 8.5 GB; as Latin images at ~40 chars/token ~ 0.9 GB; as a code at 100 chars/token ~ 0.37 GB.

**The trap.** The saving is real only if the model answers from the image tokens directly. If it transcribes the
page back to text first, the output tokens refill the KV cache. The benchmark is therefore QA over encoded context,
not OCR.

## Model
Qwen2.5-VL-3B-Instruct (open weights, already a VLM, encoder barely compresses: 28x28x3 = 2352 values -> 2048-d
token; 448 px page = 256 tokens). Fits the 3090 in bf16 for inference; LoRA training on single pages fits too.
DeepSeek-OCR is the fallback/second target (its encoder compresses ~10x and destroyed every code frozen).
A text-only LLM would need its own encoder + projector alignment: only after a positive Qwen result.

## Steps
### R3-0: frozen Qwen probe (code exists, ~2 h GPU)
The round-2 Qwen geometry, never run: identity28 ceilings + qwen25vl clean/jpeg75 at 1000 train pages, 10k steps,
encodings latin_p{5,6,8,10,16,24}, mono_p{6,8}, col_b{2,4}, rgb3_p{6,8}, lg_free_h4, lg_pb_h5
(run names `round3_qwen_tp1000`, ceilings `round3_qwen_tp200`). First a CUDA smoke of `qwen25vl` (its CUDA path has
never run). Qwen tokens are 256 x 2048 per page: run 2 GPU jobs at a time, not 3. Applies ROUND2.md's decision rules
for Qwen. Picks the dense codes for R3-1 (best 1-2 by chars/token at 10% CER; if none survive, still take col_b4 and
rgb3_p8 as the fine-tuning test, since fine-tuning is the point).

### R3-1: teach Qwen to read (fine-tuning smoke, then runs)
* LoRA (r=16) on the LLM's attention+MLP; unfreeze the patch merger and the last 4 vision blocks; everything else
  frozen. bf16, gradient checkpointing, single 448 px pages (256 vision tokens + prompt + answer, short sequences).
* Data: rendered WikiText train pages (same renderers as rounds 1-2, page cache), unlimited and free. Task mix per
  page: 60% full transcription, 40% questions about the page (cloze: "what word follows X", "which of these
  phrases appears", copy a span).
* Arms at matched chars/token: Latin (the fair baseline: it gets the same fine-tuning budget) vs the R3-0 picks.
* **Moonshot arm (user, 2026-11-04): first go throws everything at it.** Besides rgb3 and the learned glyph banks
  on their own, add a combined code: an RGB overlay of learned-glyph layers (one glyph-bank text per channel, e.g.
  `rgb3_lg_pb_h5` / `rgb3_lg_free_h4`; not built yet: extend `ep/rgb_overlay.py` to draw bank glyphs per channel,
  with channel-tagged classes as rgb3). Nominal density = 3x the bank's: about 3 x 107 = 320 (lg_pb_h5) and
  3 x 135 = 405 (lg_free_h4) chars per Qwen token, against latin_p5 at 72. Run its identity28 ceiling first (the
  round-2 guard: < 10% CER, else the combined page itself is unreadable). It goes into R3-1 even if R3-0 shows the
  overlay at the floor frozen, since fine-tuning is the test; if it fails, fall back to the best single code.
  Smoke: 500 steps per arm (~minutes) to check the loop, loss falls, generation works. Then ~5k steps per arm.
  * **Moonshot encodings (built 2026-11-04).** `rgb3_lg_pb_h5`, `rgb3_lg_free_h4` (any `rgb3_<LG_BANKS key>`):
    `ep/rgb_overlay.py::RGB3BankEncoding`, resolved by `get_encoding` (224) and `get_encoding_s` (448/512), so
    `run_probe2.py --encoding rgb3_lg_pb_h5` works. Each channel is the bank's own word-wrapped layout at its native
    pitch (2 px chars, 4 px dictionary tokens); consecutive wrapped lines go R, G, B. Classes: channel-tagged bank
    labels (3 x 395 = 1185); labels interleaved by glyph centre x (= rgb3's R[k], G[k], B[k] for equal advances);
    decode routes by tag. Strips: `docs/img/rgb3_lg_{pb_h5,free_h4}_strip.png` (page, then R, G, B; 4x).
    Tests: `encoder_probe/tests/test_rgb3_lg.py`.
    Density at S=448 (16x16 tokens of 28 px): 320 / 403 chars per token (2.98x the single banks' 107 / 136).
    CTC budget: T = 3S frames per line; a full-width line could hold S/2 2-px glyphs per stream, L = 3S/2, and
    2L+1 = 3S+1 misses by one frame, so lines wrap at S-1 px (worst case 223 glyphs/stream at 448, L <= 669,
    1339 <= 1344; at 512: 255, 1531 <= 1536). Real lines peak at ~600-620 labels (2L+1 ~ 1240), so the 1 px
    costs ~0.2% density. Cost: the 1185-class head and L ~ 600 make a CPU probe step ~7.5 s at 1 thread (~2x
    lg_pb_h5's 4 s), i.e. ~6.5 h per 3000-step run.
    Identity28 ceilings (200 pages, 3000 steps, `round3_cpu_tp200`): pending.
* Metric: CER of generated transcriptions on held-out test pages (WikiText test/val, as rounds 1-2), clean and jpeg75;
  cloze accuracy.
* **Code: built 2026-11-04, CPU-tested on a tiny random-init model, never run on a GPU or with the real weights.**
  `experiments/finetune_qwen.py` (shared code: `ep/qwen_ft.py` model/LoRA/inputs/KV, `ep/r3_tasks.py` prompts and
  answers; tests `tests/test_round3.py`). As planned: LoRA r=16 (alpha 32, dropout 0.05) on q/k/v/o/gate/up/down of
  all 36 LLM layers (29.9M params), merger + last 4 vision blocks unfrozen (115.5M), trainables fp32, bf16 weights +
  autocast, gradient checkpointing; 448-px pages from `split_pages` (any registered encoding, `rgb3_lg_*`
  included); answers are the decoded characters; loss on answer tokens only; defaults 5000 steps, batch 4,
  1000 train / 16 test pages, lr 2e-4 (LoRA) / 2e-5 (vision), 5% warm-up + cosine. Rows: `results/round3_ft/`, one
  per eval cond, with `fair` / `fair_hash` (the fairness-guard settings; compare hashes across arms).
  **Deviation: "full transcription" is windowed.** A whole dense page is 7k (latin_p8) to 82k (rgb3_lg_pb_h5) chars,
  i.e. 2k-19k answer tokens, so a page that holds more than `--max-target-chars` (default 2000, the same for every
  arm) is transcribed as a window of whole lines ("Transcribe lines a-b of the n lines on this page."); pages that
  fit (latin_p24) get the whole page. Eval = greedy generation of 4 evenly spaced windows per test page (all
  windows of a short page), CER = edit distance / ref chars with rounds 1-2's per-line `clean_hyp`. Whole-page
  targets for every arm: `--max-target-chars 100000` (long sequences; VRAM untested).
  Checked without the 7.5 GB weights: the hand-built Qwen2.5-VL chat template + 256 `<|image_pad|>` per page equals
  the checkpoint's own chat template and processor expansion token for token (real tokenizer, 2026-11-04), and the
  hub's old key layout (`model.layers.*`, `visual.*`, tied head) loads into transformers 5.22.1 bit for bit (test).

### R3-2: long-context benchmark (the goal)
* Documents: concatenated held-out WikiText, 10k-1M chars. Inputs: (a) plain text tokens, (b) Latin page images,
  (c) dense-code page images, all into the fine-tuned model (and (a), (b) into the base model as references).
* Tasks: needle-in-a-haystack (a random fact sentence inserted at depth 0-100%), and short-answer QA / cloze on
  spans of the document. Answers must be short (no transcription).
* Measure per input and length: accuracy, positions used, actual KV bytes (from the model's cache tensors) and peak
  GPU memory, prefill time. Native window 32k positions (= 1.2 GB KV); go beyond only with the model's long-context
  setting, noted in rows.
* Output: accuracy vs KV-cache-GB curves for (a)/(b)/(c). Rows to `encoder_probe/results/round3_*/runs.jsonl`.
* **Code: built 2026-11-04, CPU-tested on a tiny random-init model, never run on a GPU.**
  `experiments/longctx_bench.py`. Inputs `text` or any page encoding (the document rendered onto consecutive
  448-px pages, 256 positions each); base model or `--adapter` (a finetune_qwen.py result, LoRA merged). Needle =
  'The secret number of <name> is <6 digits>.' at `--depths`; cloze = the word after a 6-word context unique in the
  document; answers <= 12 tokens, greedy. Per item: correct, positions (= KV entries), M-RoPE span, KV bytes summed
  over the cache tensors after prefill (= the analytic formula, tested), peak GPU memory, vision and LLM prefill
  time. One row per (input, length, task). Inputs above `--max-positions` (default 32768 KV entries) get a row with
  `skipped: true` and the reason; `--rope-scaling` (e.g. YaRN) is recorded in every row. Note: with M-RoPE a
  448-px page advances the rope position by only 16 while using 256 KV entries, so the 32k limit is applied to KV
  entries (the memory axis), not to rope positions (`rope_span` is in the rows). At the default limit, text fits
  up to ~140k chars, latin_p8 pages up to ~0.9M chars, latin_p5 and denser codes the full 1M.

## Pre-registered predictions (before any round-3 run)
1. R3-0: Qwen preserves more than DeepEncoder: Latin C@3% through qwen25vl/clean >= 15 chars/token (ROUND2 (a)),
   and col_b4 between 3% and 50% CER clean (ROUND2 (b)).
2. R3-1: after equal fine-tuning, at least one dense code reaches < 3% transcription CER at >= 1.5x the chars/token
   of the best fine-tuned Latin at 3%, clean; the advantage shrinks under jpeg75.
3. R3-2: at equal KV bytes the dense-code input answers needle/QA at least as well as Latin images, and both beat
   plain text once the text no longer fits the 32k window.

## Decision rules
* **Go** (worth scaling to bigger models / other LLMs): prediction 2 holds AND in R3-2 the dense code reaches the
  same accuracy as Latin images with <= 0.67x the KV bytes.
* **Latin-only**: fine-tuned Latin images give the KV win over text but no dense code beats Latin: the gain is
  "optical compression", not the encoding; stop the dense-code line.
* **Kill**: neither images nor codes keep QA accuracy within 10 points of plain text at any length that fits.
* Fairness guard: Latin and every code get the same steps, data, LoRA config and unfrozen layers.

## Runbook notes (this PC: Windows, RTX 3090, see docs/LOCAL_GPU.md and AGENTS.md gotchas)
* Clone at `C:\git\visual-bpe-run` (sparse: three learned_glyphs sample dirs have `:` in file names); venv `.venv`,
  `PYTHONUTF8=1 EP_PAGE_CACHE=1 EP_TRAIN_R2_CAP=400000000`, queue lines use `python`, not `python3`.
* At most 3 GPU jobs (2 for Qwen-token jobs): Windows pages VRAM instead of OOM and jobs stall silently. Watch for
  checkpoints not updating. Keep `--ckpt-every`.
* Code built (2026-11-04, CPU-tested only): R3-1 trainer (`encoder_probe/experiments/finetune_qwen.py`,
  transformers + peft) and R3-2 benchmark (`encoder_probe/experiments/longctx_bench.py`). Weights: full
  Qwen2.5-VL-3B (~7.5 GB, into the Hugging Face cache), not just the vision tensors. Commands, in order, with VRAM
  estimates: docs/LOCAL_GPU.md, "Round 3 on the 3090".
* Estimates (not measured): R3-0 ~2 h; R3-1 smoke 30 min, full arms 2-4 h each; R3-2 harness ~1 day of coding,
  runs ~1-2 h per configuration.
