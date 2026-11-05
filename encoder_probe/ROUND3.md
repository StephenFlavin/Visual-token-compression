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

**Q1 (2026-11-04, cloud CPU, quick test): layer-wise linear readout of frozen Qwen ViT features.**
`experiments/quick_linear_readout.py`; table: `results/quick_q1/summary.md`, rows `results/quick_q1/rows.jsonl`,
panel `docs/img/q1_readout_rgb3_lg_pb_h5.png` (one test page, readout per layer, R/G/B crops). Closed-form ridge
from each feature to the clean pixels it covers. 64 train pages (lambda chosen on 16 of them), 16 held-out test
pages, 448 px. Hidden states are put back in raster order: the merger rebuilt from them equals pooler_output exactly,
and the patch-embed readout gives R^2 = 1. Ink MCC (0 = chance) by depth, b8 / b16 / b24 / b28 / final / merger:
rgb3_lg_pb_h5 (322 chars/token) 0.86 / 0.79 / 0.64 / 0.56 / 0.47 / 0.29; rgb3_lg_free_h4 (406) 0.57 / 0.43 / 0.22
/ 0.16 / 0.08 / 0.00; lg_pb_h5 (107) 1.00 / 1.00 / 0.91 / 0.81 / 0.65 / 0.30; rgb3_p8 (73) 0.63 / 0.48 / 0.19 /
0.07 / 0.05 / 0.10; latin_p8 (28) 0.89 / 0.67 / 0.18 / 0.02 / 0.02 / 0.03; latin_p16 (6.8) 0.94 / 0.85 / 0.39 /
0.08 / 0.04 / 0.06. DeepEncoder calibration: latin_p8 0.00, rgb3_lg_pb_h5 0.29.
What it shows:
* The Latin controls lose their linear pixel readout by b24-b28, yet Qwen reads Latin. **A linear readout is a lower
  bound**, and a low deep-layer value is not evidence of loss.
* The whole-pixel binary codes (lg_pb_h5, rgb3_lg_pb_h5) stay pixel-like through b28. They drop mostly in blocks
  28-31 and the merger, down to the ~0.29 coarse-density level that DeepEncoder also gives. R3-1 unfreezes exactly
  that part, so this is a weak good sign for R3-1 on rgb3_lg_pb_h5.
* The grey-level moonshot rgb3_lg_free_h4 and rgb3_p8 already fade by b8-b16, as Latin does. A linear readout
  cannot tell re-coding from destruction there. If it is destruction, only stage B (full ViT) could help them.
* Not a readability result: a frozen or tuned probe (R3-0 / R3-0b) is still needed.

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
    Identity28 ceilings, PARTIAL (cloud CPU, 200 train pages, `round3_cpu_tp200`, stopped 2026-11-04 at step ~650
    of 3000 so the user can run them locally on the 3090): val CER (4 test pages) at step 500: `rgb3_lg_pb_h5`
    clean 15.2% (321.6 chars/token, train CTC 0.25), `rgb3_lg_free_h4` clean 37.0% (405.5 chars/token, train CTC
    1.15); 7.8 s/step at 1 CPU thread (fp32, no native bf16); jpeg75 not started. Both still falling: these are
    not ceilings yet and say nothing about the < 10% guard. **The full ceilings (identity28 clean + jpeg75, 3000
    steps) are to be run locally**; the cloud resume checkpoints were deleted (Windows renders differ, so they
    would not resume onto the same pages). For comparison, identity32 ceilings (round2_gpu_tp200, 3000 steps):
    rgb3_p6 0.01% (169 chars/32-px token), rgb3_p8 0.42% (96), lg_pb_h5 0.52% (141), lg_free_h4 0.05% (177); no
    identity28 rows exist yet for rgb3_p*/lg_*.
    **Update (2026-11-04/05): the clean ceilings were re-run on the cloud after all (G0, run `round3_cpu_tp200`).**
    * `rgb3_lg_pb_h5` val CER (4 pages): 15.2% at step 500, 2.3% at 1000, 1.5% at 1500.
    * `rgb3_lg_free_h4` was killed by the cgroup OOM killer (5.2 GB RSS) at about step 400. Not restarted: Q1 dropped
      it.
    * **Precision note for the `rgb3_lg_pb_h5` row:** steps 1-2400 ran in fp32 (host without native bf16). After a
      container restart it resumed at step 2400 on an AMX host, so steps 2401-3000 ran in bf16 autocast. The row's
      `bf16: true` therefore describes only the last 600 steps.
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

**R3-1 short-budget results (GPU runner, RTX 4090, 2026-11-05; details and samples in docs/HANDOFF.md).**
1000 steps per arm (5000 planned), 1000 train / 16 test pages, one fair_hash `05dc0c9e8507`, with --blind-images
controls. Rows: `encoder_probe/results/round3_ft_gpu3h{,_blind}/`.
| Arm (chars/token) | CER clean | blind CER | cloze | MC | val loss (blind) |
|---|---|---|---|---|---|
| latin_p8 (28) | 55.0% | 88.5% | 0.52 (0.22) | 0.88 (0.25) | 1.12 (2.48) |
| latin_p5 (72) | 80.9% | **77.8%** (own blind, run after) | 0.31 (0.30) | 0.56 (0.28) | 2.16 (2.40) |
| lg_pb_h5 (107) | 78.4% | blind died at step 500 (disk full) | 0.17 | 0.31 | 2.27 (2.29 at 500) |
| rgb3_lg_pb_h5 (322) | 83.6% | 89.9% | 0.22 (0.22) | 0.34 (0.28) | 2.518 (2.517) |
* **The LLM reads Latin at 28 chars/token within 1000 steps:** well below its blind control on every metric. The
  first line of a window comes out near-exact, then the output drifts into fluent invented continuation (the
  language prior takes over). latin_p5 (72) does **not** transcribe yet: its own blind control gets lower CER (77.8% vs 80.9%) and the same
  cloze; only MC and val loss show some image use. Blind CER floors differ by code, so each arm needs its own.
* **The moonshot is not used at all at this budget.** Its losses match the blind run's to 3 decimals, and its
  generations are the same generic text as the blind run's, whatever the page. This agrees with G2-lite (the
  information dies at or before the merger); the vision side (merger + last 4 blocks at lr_vision 2e-5) didn't
  move in 1000 steps.
* **G3 is undecidable** (no arm reaches 3% CER). Short-budget signal only: the moonshot is the weaker candidate.
  Undecided, not failed.
* `finetune_qwen.py` rows now carry `ops` (sub/ins/del, len_ratio) and `prefix_frac` (the share of ref chars
  before the first error, the read-then-drift measure). Added after this run.
* Next, before any long moonshot run: whether an intermediate density moves off its blind floor (lg_pb_h5 ~107: also at
  its floor at 1000 steps, val loss = its blind's; rgb3_p8 ~73 not run), and a vision-first stage-B diagnostic (R3-1b) for the moonshot (higher
  lr_vision / more unfrozen blocks, loss vs blind). For Latin: 5000 steps for latin_p8/p5 to see where the drift goes.

### R3-1b: deeper training if LoRA is not enough (added 2026-11-04, user: "LoRA first, then proper training")
Staged so each step is only run when the one before shows a reason to. The fairness guard holds at every stage:
Latin gets the identical recipe, steps and data. Memory and time figures are estimates for the 3090.
* **Gate G0 (before any fine-tuning):** the code's identity28 ceiling is < 10% CER. If the raw page is unreadable, no
  amount of model training helps; drop that code.
* **Stage A = R3-1 as above** (LoRA r=16 + merger + last 4 ViT blocks, 5k steps). Go to stage B for a code only if
  its held-out CER is still falling at 5k, or it plateaus well above its ceiling while Latin converges.
* **Stage B, vision-first training (standard VLM recipe):**
  * B1: the full ViT (~0.67B) + merger trained on transcription only, LLM frozen. AdamW state for 0.67B is about
    11 GB; fits with gradient checkpointing at batch 1.
  * B2: freeze the ViT, LoRA r=64 on the LLM for the QA mix.
  * Curriculum for the dense codes: start at the next-densest code that reads (e.g. rgb3_p8 or lg_pb_h5), then
    switch to the moonshot page. 20-50k steps. Same budget for Latin.
* **Stage C, beyond the 3090 (only after a positive stage-B result):** full fine-tune of the 3B LLM (about 60 GB with
  AdamW: rented A100/H100, user's call), a larger Qwen VL, or continued pretraining on long rendered documents
  (Glyph-style).
* **What would make a moonshot result credible (all required before calling it):**
  * Held-out text only (WikiText test/val), and needle facts never seen in training: random names, numbers, IDs.
  * Answers come straight from the image tokens, with no transcription in the output (KV trap above).
  * Same-recipe Latin baseline next to it, and compute/char measured (ViT + LLM prefill time on the 3090), not only
    the estimate in docs/CONTEXT_BUDGET.md.
  * The result repeats with a second seed.
  * Exact recall (needle accuracy) is reported separately from gist QA, because a dense code with no language prior
    fails first on exact strings.

### Re-scope after the literature report (2026-11-04)
Source: [docs/research/2026-11-04_rgb3_lg_go_no_go_report.md](../docs/research/2026-11-04_rgb3_lg_go_no_go_report.md)
(external research task; several of its sources are recent preprints, verify before citing).
* **Density sweep, not one extreme point.** Published go region: ~30-90 chars/token; the moonshot's 322-405 is ~7x past
  the best trained optical result at high accuracy and ~2x past the best text-side compressor (C3 ~170 at 93%).
  Capacity studies report an abrupt wall (unstable band ~2.2x wide), so sweep at ~2x spacing: about 15, 30, 60, 120,
  240, 322 chars/token. Existing codes at Qwen geometry: latin_p8 ~28, rgb3_p8 ~73, lg_pb_h5 ~107, rgb3_lg_pb_h5 322.
* **Gates (adopted):**
  * G0: identity ceiling < 3% CER (go); > 10% kills that point.
  * G1: nonlinear glyph-ID probe on frozen pre-merger ViT states, ≥ 97%.
  * G2: train the merger (+ last 4 blocks) with a CTC head, LLM frozen, across the sweep. Kill rgb3_lg at 322/405 if it
    is < 90% glyph accuracy at ≥ 120 chars/token.
  * G3: the R3-1 transcription bar (≥ 1.5x fine-tuned Latin chars/token at < 3% CER).
  * G4: needle within 3 points and cloze within 5 points of Latin images at ≤ 0.67x KV, AND within 5 points of a
    text-side compressor (500xCompressor/ICAE-style LoRA on Qwen2.5-3B) at the same KV.
  * G4b: the same QA with a transcribe-first prompt, reporting the use-vs-read gap.
* **Grid alignment:** 5 px pitch does not divide the 28 px token grid. Add 4/7 px pitch variants and a pixel-shift
  test (report: unstable-band errors come from patch misalignment).
* **Negatives are reported as "not reachable at this budget"**, not "impossible".

**Q1 result (layer-wise linear pixel readout, frozen Qwen2.5-VL ViT; ink MCC per channel, chance-corrected):**

| Code (chars/token) | block 8 | block 24 | final | after merger |
|---|---|---|---|---|
| latin_p8 (~28) | 0.89 | 0.18 | 0.02 | 0.03 |
| rgb3_p8 (~73) | ~0.64 | ~0.19 | ~0.05 | ~0.10 |
| lg_pb_h5 (~107) | 1.00 | 0.91 | 0.65 | 0.30 |
| rgb3_lg_pb_h5 (322) | ~0.85 | ~0.64 | ~0.47 | 0.29 |
| rgb3_lg_free_h4 (405) | ~0.57 | ~0.22 | ~0.08 | ~0.00 |

Reading: Latin's pixels stop being linearly decodable by block 24 even though Qwen reads Latin well, so this is not a
readability score. The whole-pixel glyph codes stay pixel-like through the whole ViT, with no colour-channel collapse
for rgb3_lg_pb_h5. Every code collapses linearly at the 2x2 merger, which points at the merger as the wall (G2). The
grey-level free_h4 variant fades by block 24: drop it, consistent with rounds 1-2 (whole-pixel banks only).

### G2-lite: does a trained merger let dense codes through? (pre-registered 2026-11-04, before any run)
CPU version of the report's gate G2. Runs overnight on the 4-CPU runner; repeat on the 3090 at more pages/steps.

**Design.**
* Qwen2.5-VL-3B ViT frozen through all 32 blocks. Cache the final-block hidden states before the merger: 1024
  patches x 1280 per 448 px page, bf16, resumable. Use ep/qwen_split.py and Q1's code, which verified the window
  and raster ordering.
* Three arms per code, all with the same GridProbe reader, the round-2 recipe (CTC + aux MSE x10), 3000 steps,
  500 train / 16 held-out test pages:
  * **frozen**: pretrained merger frozen + probe. This is R3-0 at this budget, the control.
  * **trained**: merger initialised from pretrained and trained + probe. This is G2.
  * **direct**: probe on the four concatenated pre-merger patch states (5120-d per token, no merger). It is the
    nonlinear upper bound of what the frozen ViT output carries (the report's G1, at token level).
* Codes (chars per Qwen token): latin_p8 (~28), lg_pb_h5 (~107), rgb3_lg_pb_h5 (322, the moonshot). free_h4
  is dropped (Q1).
* Run order (CPU budget): moonshot trained, latin_p8 trained, moonshot frozen, moonshot direct, lg_pb_h5 trained,
  latin_p8 frozen; then lg_pb_h5 frozen/direct and latin_p8 direct if time is left.
* Quote the row's `cer` on the 16 test pages, with chars/token, the curve and the train CTC.

**Predictions (my priors, before running).**
1. trained beats frozen by ≥ 20 CER points for both glyph codes, and by less for latin_p8.
2. direct ≤ trained + 2 points: the merger can only lose information. A bigger gap means the merger's 2048-d output
   is the wall.
3. latin_p8 trained < 5%. The moonshot trained does not reach < 10% at this budget (literature prior: 322 is ~7x
   past the best trained optical result). lg_pb_h5 at 107 is a coin flip.

**Decision rule** for the moonshot at 322 (≥ 120 chars/token, so the report's kill rule applies; CER stands in for
glyph error):
* trained < 10% CER: strong signal. The moonshot becomes a main R3-1 arm, and the 3090 rerun is at 1000+ pages.
* trained 10-30% and still falling: undecided. Rerun longer on the 3090 before judging.
* trained > 30% and direct > 30%: the ViT output is the wall. Next comes stage B (train the ViT) or a lower-density
  sweep point.
* direct good, trained and frozen bad: the merger's capacity is the wall. Next comes a merger redesign (levers below).
* The probe is data-limited at 500 pages (round 2: Latin through DeepEncoder 85% -> 15% from 200 to 5000 pages).
  Only same-budget comparisons between arms carry weight; absolute levels are upper bounds on the loss.

**Implementation note (2026-11-04, before the first row).** The design above is implemented as written:
`ep/g2lite.py`, `experiments/run_g2lite.py`, tests in `tests/test_g2lite.py`. The cache is `qwen25vl_h32`, built
with `build_tokens2.py --split 32` in chunks of 25 pages. The queue is `results/queue_g2lite.txt` (print it with
`run_g2lite.py --print-queue`). Rows go to `encoder_probe/results/g2lite/runs.jsonl`, keyed by (encoding, arm,
steps, train_pages, seed). These choices are not fixed by the design:
* **Standardisation.**
  * frozen and trained: per-dim mean/std of the *pretrained* merger output on the train pages, as run_probe2 does
    with its tokens. In the trained arm this is a fixed affine map after the trained merger.
  * direct: per-dim mean/std of the 5120-d concatenated states.
* **Optimiser.**
  * The trained arm's merger (RMSNorm weight and both MLP layers, 36.7M parameters) is in the same AdamW group,
    with the same LR 2e-3 and OneCycle schedule.
  * The grad-norm clip of 5 covers all trainable parameters.
  * The probe is re-seeded right before it is built, so every arm starts from the same probe init. direct differs
    only in its first layer's input width.
* **Frozen arm.** Its merger output is computed once from the cached states. The merger has no dropout, so this is
  the same as running it every step.
* **Check on real weights.** The pretrained merger on the cached bf16 states vs the model's own tokens (pooler
  output from fp32 states): rel L2 0.22%, corr 0.999997. Each row stores this as `merger_check`.

**Feasibility (measured 2026-11-04): the full set does not fit one night on this runner.**
* Cost at 1 thread, fp32:
  * ViT cache: 13.7 s/page, so about 2 core-hours per code.
  * Probe steps (profiled, s/step):

    | Code | frozen | trained | direct |
    |---|---|---|---|
    | rgb3_lg_pb_h5 | 7.8 | 9.2 | 9.9 |
    | latin_p8 | 2.6 | 3.7 | 4.0 |
    | lg_pb_h5 | 3.7 | 4.9 | 4.9 |

  * The moonshot's CTC over about 560 labels per line and its 1185-class reader dominate.
* The 9 runs total about 42 core-hours, plus about 6 for the caches. The night offers about 24 on 2-3 cores.
* So the runs go in the stated priority order. Whatever has not finished by morning stays as queue lines, for the
  3090 rerun or the next night. The design is unchanged.

**Amendment (2026-11-04 ~21:50, before any G2-lite row; user): push the Latin baseline as far as the code.**
Once the encoder and merger are trained, fine-tuned Latin need not stay at Glyph's ~15 chars/token:
* its raw-pixel ceiling at Qwen-equivalent density is ~51 (p6, 0.25%) and ~72 (p5, 0.27%), from round-2 identity32
  rows rescaled to 28 px tokens;
* DeepSeek-OCR's trained encoder reaches ~45.
So the 1.5x bar can move to ~25-70+. Comparing codes against latin_p8 (~28) alone would test against an under-pushed
baseline.
* **G2-lite:** add a Latin density sweep under the same arms. latin_p8 (~28) is kept, and latin_p5 (~72) is added,
  plus latin_p6 (~51) if time allows. Exploratory: latin_p4 (~110), but only after its identity28 ceiling is < 3%.
  Tonight's priority on CPU: moonshot trained > latin_p8 trained > latin_p5 trained > moonshot frozen; the rest goes
  to the 3090.
* **Comparative rule (added; the absolute moonshot rule above is unchanged):** a code passes only if its density at a
  given CER is ≥ 1.5x the densest Latin that reaches the same CER under the same arm and budget. Latin is swept up to
  ~100 chars/token in G2 and G3, so the code is compared against a pushed baseline.
* **G3 / R3-1 arms:** Latin at p8, p6, p5 (and p4 if its ceiling passes), same recipe as every code.

**Amendment 2 (2026-11-05 ~02:30, after the first trained row, before any amended row; approved by main): the
trained arm uses merger LR 1e-4.** At the shared LR of 2e-3, AdamW plus OneCycle wrecks the pretrained merger. Its
output collapses to near-identical tokens, so the trained arm as registered measures a broken merger, not merger
capacity.
* **Evidence.** Checkpoints of the as-registered runs, plus 500-step latin_p8 diagnostics (`results/g2lite_diag*`,
  not pre-registered arms). Standardised output std uses the pretrained statistics (pretrained = 1.03). "Dead dims"
  is the share of the 2048 dims with std < 0.05. Weight change is the relative norm change of mlp.0 / mlp.2.

  | Run (step) | train CTC | output std | dead dims | token-pair cosine | weight change |
  |---|---|---|---|---|---|
  | pretrained merger | - | 1.03 | 0% | 0.10 | 0 / 0 |
  | latin_p8 trained, LR 2e-3, 3000-step run (2200) | 2.84 | 0.11 | 76% | 0.91 | 1.01 / 0.69 |
  | moonshot trained, LR 2e-3, 3000-step run (600) | 5.5 | 0.14 | 24% | 0.82 | 0.47 / 0.26 |
  | latin_p8 trained, LR 2e-3, 500-step diag (400) | 2.98 | 0.15 | 11% | 0.81 | 0.67 / 0.41 |
  | latin_p8 trained, merger LR 1e-4, 500-step diag (400) | 3.03 | 0.37 | 0% | 0.20 | 0.10 / 0.06 |
  | latin_p8 frozen, 500-step diag (500) | 3.00 | (1.03) | - | (0.10) | - |

* **Choice.** Merger LR 1e-4, picked on train CTC and the collapse statistics, never on test CER:
  * It is the only setting that keeps the merger's output diverse (no dead dims, cosine 0.20).
  * Its train CTC equals the frozen and 2e-3 runs at 500 steps. Every Latin arm is still at unigram level there
    (about 2.9-3.0 nats/char), so no setting was "clearly too slow" and 3e-4 was not needed.
* **Scope.**
  * It applies to every trained arm (moonshot, latin_p8, latin_p5, latin_p6, lg_pb_h5) as `--merger-lr 1e-4`. Rows
    carry `merger_lr` in their key.
  * The probe LR (2e-3) and every other setting stay as pre-registered.
  * The two as-registered trained rows are reported as such: latin_p8 trained, 77.8%, collapsed. moonshot trained was
    stopped at step 600 (99.8% at step 500, collapsed); its checkpoint is kept in `cache/ckpt/old_collapsed/`.
* **Precision.** The new host has native bf16, so all G2-lite probe runs restarted at 00:15 and run in bf16 autocast
  from step 1 (`precision_segments` in each row). The ViT caches are fp32 on every host.

**Limitation (2026-11-05): the probe favours pixel-like codes, so the Latin comparison is not fair here.**
* GridProbe reads text only through a 3-channel reconstruction of the page (aux MSE), then a conv reader.
* Q1 found that Latin pixels stop being linearly decodable in Qwen's states after block 24 (merger ink MCC 0.03).
  The glyph codes stay pixel-like (lg_pb_h5 final 0.65).
* So this probe under-reads codes that Qwen turns into non-pixel features, Latin above all. The reconstruction MSE at
  step 500 shows it: 0.14-0.22 for latin_p8 here vs about 0.001 for identity28.
* G2-lite's Latin numbers are lower bounds, and the comparative 1.5x-vs-Latin rule above cannot be applied fairly
  with this probe. The fair Latin-vs-code comparison is G3 (the LLM itself as reader, on the 3090).
* G2-lite still answers the absolute question: does glyph-code information get through the frozen or trained
  merger, and does bypassing the merger (direct) change that?
* A latin_p16 frozen sanity run (about 7 chars/token, 100 train pages) decides whether Latin arms stay in tonight's
  CPU queue.

**Sanity check (2026-11-05 ~03:55): the pipeline learns; Latin through Qwen overfits at this budget.**
Setup: latin_p16 frozen, 6.8 chars/token, 100 train / 16 test pages, 1000 steps (diagnostic, `results/g2lite_diag`).

| Step | train CTC | val CER |
|---|---|---|
| 250 | 3.40 | 99.8% |
| 500 | 2.80 | 99.96% |
| 750 | 1.78 | 82% |
| 1000 | 0.97 | 75% |

* Test CER **75.5%**, final train CTC 0.82, so it fits its training pages far below unigram level.
* The same check through the independent round-2 pipeline (`run_probe2.py --encoder qwen25vl` on the cached merger
  tokens, `results/g2lite_diag_probe2`) gives **74.5%**, with a matching curve. So there is no G2-lite-specific bug.
* Latin is readable from Qwen's tokens but data-limited at 100 pages. This matches round 2 (Latin through
  DeepEncoder: 85% -> 15% from 200 to 5000 pages).
* Consequence for tonight's CPU: only the three moonshot arms run (frozen, trained at merger LR 1e-4, direct). The
  latin_p5 cache (its chunks are kept) and the Latin sweep go to the 3090 at more pages. The amended latin_p8 trained
  run gets a slot only if one frees before about 07:00, and is a lower bound.

**Required reference (2026-11-05 ~06:40, outside review): the blind floor and the CER op split.**
* **The problem.** Decoding the moonshot direct checkpoint (step 1800, 4 test pages, 356 lines) gives distinct but
  unaligned letter-frequency soup.
  * Example: ref 'Robert Boulter is an English film…' -> hyp 'ld nola s i wistililielegr arreri btti…'.
  * Its 74.5% CER splits into sub 18.4% / ins 0.3% / del 55.8%, with hyp/ref length 0.45.
  * Frozen moonshot (3 stored samples): del 83%, length 0.17.
  * latin_p16 frozen (3 stored samples): del 69%, length 0.31.
  * A CER near 75-86% can therefore come from label statistics alone: a longer hyp of frequent letters lowers CER
    without reading the page.
* **Blind arm** (`--arm blind`, diagnostic, never a pre-registered arm):
  * frozen-merger features, but each train and test page gets another random page's features (seeded derangement);
  * labels and recon targets stay with the true page;
  * same probe, recipe and pages.
  * Its CER is the probe's floor for that code. Runs: `results/g2lite_blind/` (moonshot 1500 steps, latin_p16 1000
    steps).
* **Every G2-lite number, diagnostics included, is quoted against its code's blind floor and with its op split.**
  * Rows written from now on carry `ops` (sub / ins / del / len_ratio over all 16 test pages).
  * The rows already running get theirs from their last checkpoint, resumed to step 3000 under another run name.
    Resume is exact.

**G2-lite results (2026-11-05 ~08:40, CPU: 500 train / 16 test pages, 3000 steps, bf16 probe).**
**Verdict, by the pre-registered decision rule:**
* The moonshot (322 chars/token) has trained 84.6% > 30% and direct 71.9% > 30%.
* That is the branch **"the ViT output is the wall at this budget"**. Next: stage B (train the ViT) or a lower-density
  sweep point.
* Not "trained < 10%", not "10-30% and falling", not "direct good, merger arms bad".

Rows: `results/g2lite/runs.jsonl`. Op splits come from `results/g2lite_ops/` (exact resumes; their CER equals the
original rows bit for bit). Floors come from `results/g2lite_blind/`. "correct" = (hyp length - sub - ins) / ref
chars, the share of reference characters the alignment matches.

| Code (chars/token) | frozen | trained (merger LR 1e-4) | direct | trained, LR 2e-3 (as registered) | blind floor |
|---|---|---|---|---|---|
| latin_p8 (28.1) | not run | 79.2%: del 74 / sub 5, len 0.26, **correct 21%**, CTC 2.79 | not run | 77.8% (collapsed merger), CTC 2.77 | not run |
| lg_pb_h5 (~107) | not run | not run | not run | not run | not run |
| rgb3_lg_pb_h5 (321.6) | 86.1%, CTC 4.55; 3 stored samples: del 83 / sub 2, len 0.17, correct ~15% | 84.6%: del 82 / sub 2, len 0.18, **correct 15%**, CTC 4.51 | 71.9%: del 46 / sub 25, len 0.55, **correct 29%**, CTC 3.31 | stopped at step 600 (99.8% at 500, collapsed) | 99.5% at 1500 steps: del 99, len 0.005, correct 1%, CTC 4.73 |
| latin_p16 (6.8; 100 pages, 1000 steps, diagnostic) | 75.5% (run_probe2: 74.5%); 3 stored samples: len 0.31, correct ~27% | - | - | - | 87.3%: del 81 / sub 6, len 0.19, correct 13%, CTC 1.07 |

* Val curves (steps 500-3000):
  * moonshot frozen 99 96 93 90 89 86
  * moonshot trained 100 95 91 88 87 85
  * moonshot direct 76 75 75 74 73 72
  * latin_p8 trained 86 87 80 81 80 79
  * All were still falling slowly at 3000.
* Precision: bf16 autocast throughout (`precision_segments`). Frozen resumed twice and trained once after container
  restarts.

**Pre-registered predictions:**
1. trained beats frozen by >= 20 points for both glyph codes, by less for latin_p8: **FAIL**.
   * Moonshot: 1.5 points (86.1 -> 84.6).
   * lg_pb_h5 and latin_p8 frozen: not run.
2. direct <= trained + 2: **FAIL**.
   * direct is 12.7 points better than trained (71.9 vs 84.6).
   * This hints that the merger loses information the ViT output still carries.
3. latin_p8 trained < 5%: **FAIL** (79.2%; 77.8% as registered).
   * Moonshot trained not < 10%: **PASS** (84.6%).
   * lg_pb_h5: not run.

**Post-hoc observations (not part of the decision rule):**
* Direct reads more than the merger arms. It matches about 29% of reference characters vs 15% for trained and about
  15% for frozen, and it emits 3x longer lines with far more substitutions.
* Its train CTC (3.31) is the only one clearly below the moonshot's label-statistics level (about 4.5-4.7). So the
  frozen ViT output carries more of the moonshot than the 2x2 merger passes on, but far from enough at this budget.
* CER below the blind floor is **not comparable across codes**:
  * The moonshot floor (99.5%) is an empty-output floor (length 0.005).
  * The latin_p16 floor (87.3%) is a matched-length floor: blind latin learns to emit frequent letters and also
    memorises its 100 training pages (train CTC 1.07).
  * Lead with the op split and the correct-character share. The correct-character share of a blind probe that does
    emit text is about 13%.
  * On that scale the moonshot frozen and trained arms (about 15%) are barely above label statistics, and direct
    (29%) is clearly above it.
* Latin through Qwen is barely above its floor here:
  * latin_p8 trained has correct 21% at unigram-level train CTC.
  * latin_p16 frozen: 75.5% vs blind 87.3%.
  * So **no comparative claim against Latin is made**. The comparative 1.5x rule (amendment 1) cannot be applied
    with this probe.

**Limitations:** 500 pages, 3000 steps, CPU. The probe is data-limited (round 2: Latin 85% -> 15% from 200 to 5000
pages). GridProbe reads through a pixel reconstruction, which under-reads Latin. Only one seed was run. The lg_pb_h5
arms, the latin_p5/p6 sweep, the latin_p8 frozen and direct arms, and a 3000-step blind floor were not run tonight:
* the latin_p5 cache chunks are kept locally (the lg_pb_h5 cache was never built); all of these go to the 3090
  rerun at 1000+ pages;
* the queue lines are in `results/queue_g2lite.txt` (old order, do not re-run from line 1) and `--print-queue`.

### Moonshot levers (beyond the go/no-go; user: "shoot for the moon")
Ways to push density past the literature's 30-90 chars/token, each testable on its own:
1. **Grid-aligned pitch.** 4 or 7 px line pitch divides the 28 px token, plus the pixel-shift protocol.
2. **Merger redesign.**
   * A wider projector output.
   * A 1x2 merge instead of 2x2: twice the positions, 161 chars/token for the moonshot.
   * A trained conv compressor like DeepSeek-OCR's.
   This trades positions for capacity per position.
3. **Redundancy inside the code.** Error-correcting bits or repeated glyph streams cost raw density for exact recall,
   and help most where needle tests fail.
4. **Per-page colour calibration palette** (JAB Code style) for the rgb3 channels.
5. **Curriculum.** Train at an easy density first (rgb3_p8 / lg_pb_h5), then the moonshot.
6. **Stage B / C.** Full ViT training, then larger models or continued pretraining (R3-1b).

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
