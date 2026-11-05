# Learned glyph alphabet: results report (2026-11-03)

**Verdict: qualified GO.** A glyph alphabet learned jointly with a reader, with ~300 frequent character chains as extra
symbols, needs **0.57×** the pixels of plain DejaVu text for the same error rate when scored in the unchanged
Phase-2 pipeline. It needs **0.63–0.74×** with a fresh reader trained on the frozen glyphs, and its lead grows under
harsh degradation. Two things qualify it:
1. Part of the gain comes from grey-level coding. Glyphs restricted to whole pixels still win, but only by 12–23%.
2. The densest alphabets break down on degradations they never saw in training (real JPEG, an unseen resize filter).

Nothing here tests a pretrained VLM encoder. `encoder_probe/` covers that.

**VLM status (2026-11-05):** frozen VLM encoders (SigLIP, DeepEncoder) lose the banks, and a LoRA-fine-tuned Qwen2.5-VL-3B
does not read `lg_pb_h5` (107 chars/token) or its RGB overlay (322) after 1000 steps: both sit at their blind floors
([encoder_probe/ROUND3.md](../encoder_probe/ROUND3.md), [results](../encoder_probe/RESULTS.md#round-3-qwen25-vl-3b-can-a-vlm-learn-to-read-the-codes-2026-11-05)). The CRNN result below stands;
the VLM case is open.

Design and pre-registered criteria: `PLAN.md`. Raw rows: `results/<run>/runs.jsonl`. The round-by-round log is in
`../docs/RESULTS.md` (history) and `../results/queue*.txt` (exact commands).

## Setup in one paragraph
Text → symbol ids (95 printable ASCII chars, plus N dictionary chains for N=300) → `GlyphBank` cells (4× supersampled,
learned ink, hard-binarised from mid-training) → line canvas → `Channel` (random sub-pixel phase, area downsample, blur,
contrast, noise, 8-bit; `harsh` adds random rescale and filters, stronger blur and noise, and differentiable JPEG) →
CRNN + CTC (`vbpe.crnn`). Metric: **px²/char** = line pitch × line width / source chars. P@x is the px²/char where CER
reaches x (isotonic fit, log-x interpolation). The baseline is converged Phase-2 DejaVu Latin: **P@3% = 6.54, P@1% = 8.92**.
E0 calibration showed that fixed DejaVu through our `default` channel reproduces that baseline (1.01×), so the two pipelines are comparable.

Three ways to score a bank, from least to most trustworthy:
* **joint**: glyphs and reader trained together through our channel. This understates the bank, because the reader chases moving glyphs.
* **E2 transfer**: frozen bank, a fresh reader trained from scratch through our channel. This measures intrinsic legibility.
* **E3 bridge**: frozen bank exported as a bitmap-font `Scheme` (`lg:` hook) and trained and scored by the *unchanged*
  Phase-2 pipeline (PIL rendering, Phase-2 CRNN). It is scored exactly like latin and subst, so it is the apples-to-apples number.

All headline numbers come from converged runs (10.5k steps). At 3.5k steps chain tokens looked useless; they need the longer training.

## Headline: pixels needed vs plain text

| | P@3% | P@1% | vs Latin (3% / 1%) |
|---|---|---|---|
| latin, DejaVu (Phase 2) | 6.54 | 8.92 | 1 / 1 |
| latin_condensed (Phase 2) | 8.32 | 17.1 | 1.27 / 1.92 |
| learned N=300, **bridge** | 3.75 | 5.10 | **0.57 / 0.57** |
| learned N=300, E2 transfer | 4.83 | 5.66 | 0.74 / 0.63 |
| learned N=300 whole-pixel, bridge | 5.73 | 6.91 | 0.88 / 0.77 |

## Frontier data (N=300, converged)

| bank (h = line pitch px; widths char/token px) | px²/char | bridge CER | E2 CER | joint CER | joint held-out CER |
|---|---|---|---|---|---|
| h3, widths 1/3 | 2.52 | 7.95% | 16.4% | 20.8% | 65% |
| h3, widths 2/3 | 4.09 | 1.92% | 4.79% | 8.71% | 68% |
| h4, widths 2/3 | 5.45 | 0.73% | 1.72% | 3.34% | 55% |
| h4, widths 2/4 | 5.88 | 0.14% | 0.29% | 0.60% | 35% |
| whole-pixel h3 | 2.52 | 33.4% | – | – | – |
| whole-pixel h4 | 5.88 | 2.08% | – | 6.75% | 22% |
| whole-pixel h5 | 7.35 | 0.59% | – | 1.80% | 12% |

"held-out" = the joint model tested on degradations never used in training (real PIL JPEG, an unseen resize filter).

## Do the chain tokens matter? (the Visual-BPE effect)
At matched accuracy and line height (joint, converged, h4), N=300 reaches 0.60% CER at 5.88 px²/char, while chars-only N=0
needs 8.0 px²/char for 0.57%. That is **≈0.74×**, so the chain tokens contribute on top of the better glyph shapes. Whole-pixel
banks show the same pattern: N=300 h5 gets 1.80% at 7.35, against N=0 h5's 1.03% at 10.0. Chars-only learned glyphs (the "font
effect") already beat DejaVu: joint N=0 h4 gets 0.57% at 8.0, where converged Latin needs ~9.8 for 0.58%.

## Robustness
* **Harsh channel (converged, h7):** learned N=300 gets 3.45% CER at 15.4 px²/char (held-out 4.4%). Fixed DejaVu is unusable at
  42.8% CER on 20.7 px²/char (held-out 45.5%). At h5 both are poor (39% vs 68%). Learning helps more under harsh degradation, not less.
* **Held-out degradations (default-channel models):** unconstrained banks degrade badly (35–68% CER) and whole-pixel banks
  much less (6–22%). This is the clearest sign that part of the default-channel gain exploits that exact channel.

## Against the pre-registered GO rule
"≥30% fewer px²/char than best fixed Latin in transfer (E2) and bridge (E3), and the gain survives the harsh channel."

| condition | result | pass? |
|---|---|---|
| bridge ≥30% fewer | 43% fewer at 3% and 1% CER | yes |
| E2 transfer ≥30% fewer | 26% fewer at 3% CER, 37% fewer at 1% | borderline (passes at 1%) |
| survives harsh | gain grows (3.5% vs 43% CER at h7) | yes |
| sanity ceiling (>3× gain = suspect) | ≤1.75× | not triggered |

## What the glyphs look like
Atlases are in `results/e1long/atlas/`. Unconstrained banks are blob- and grey-level codes, not letterforms: sub-pixel structure
becomes intermediate grey after area downsampling. Whole-pixel banks (`--pixel-block`) are small binary pixel patterns.
Neither is human-readable.

## Lessons (also in ../AGENTS.md)
* Score banks by the bridge, not by joint CER: the joint reader lags 2–4× behind a fresh reader on the same frozen bank.
* Train ≥10.5k steps. Tokens look useless at 3.5k steps, and that was the first, wrong interim conclusion.
* `vbpe.analysis` groups rows by `em`. Banks with non-default cell widths share an em, so set `budget = px_per_char` first.
* Restarting `scripts/job_queue.py` on the same file re-runs it from line 1. Use a new queue file per round.

## Open questions → next steps
1. **Does a pretrained VLM image encoder preserve these glyphs?** `encoder_probe/` (running) trains a probe on frozen SigLIP
   tokens for DejaVu, these banks (grey-level and whole-pixel) and colour blocks. If grey-level banks lose their edge there, the
   GPU plan should use whole-pixel banks.
2. **GPU follow-up (if 1 is not negative):** train with real JPEG and resampling in the channel (removes the held-out brittleness),
   use larger N (1000), and fine-tune a small VLM on rendered pages of learned glyphs vs DejaVu at equal vision-token budgets.

## Reproduce
```bash
python3 learned_glyphs/experiments/train_joint.py --grid e1 --modes learned --n-tokens-list 300 --heights 4 --steps 10500 --run-name e1long
python3 learned_glyphs/experiments/transfer_eval.py --bridge --run-name e3_bridge_long --steps 10500 \
    --banks learned_glyphs/results/e1long/banks/lg_learned_n300_h4.pt --ems 3.3333
python3 learned_glyphs/experiments/transfer_eval.py --run-name e2_transfer_long --steps 10500 \
    --banks learned_glyphs/results/e1long/banks/lg_learned_n300_h4.pt
# whole-pixel: add --pixel-block --guard-ss 0 --scheme-suffix _pb; narrower cells: --char-w 2 --token-w 3 --scheme-suffix _w23;
# harsh: --channel harsh --run-name e1long_harsh
```
About 35 min per 10.5k-step run at 1 thread. All runs: `results/queue*.txt`.
