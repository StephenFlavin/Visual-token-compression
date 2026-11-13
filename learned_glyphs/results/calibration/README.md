# E0: channel calibration (WP-LG1)

Tooling: `learned_glyphs/experiments/calibrate_channel.py`. Modules: `lg/glyphs.py` (GlyphBank),
`lg/compose.py`, `lg/channel.py`. Tests: `learned_glyphs/tests/test_render.py`.

The fixed DejaVu bank is `GlyphBank.from_font(ASCII95, DejaVuSans, h=round(1.2·em), em=em)`. It holds PIL
coverage at 4x supersampling, uses the vtc baseline (0.9·em below the top of the line box) and cells
that are each glyph's advance at 4·em, rounded to whole ss px. It is composed and pushed through each preset,
then compared with the Phase-2 rendering of the same lines (`render_sample`, `RenderParams()`).
The comparison uses the first 256 val lines, which are the Phase-2 test lines.

## Presets (lg/channel.py)
| preset | train-mode ranges | eval (fixed) |
|---|---|---|
| clean | x/y sub-pixel phase (1/4 px), area downsample, 8-bit | the same, with a seeded phase |
| default | clean + blur σ U[0,0.5] px, gamma logU[0.8,1.25], contrast U[0.7,1.0], bg lift U[0,0.05], noise σ U[0,0.06] (the augment_batch ranges) | σ 0.25, γ 1, c 0.85, b 0.025, noise 0.03 |
| harsh | phase, then a resample round trip ×logU[0.8,1.25] (filters area/bilinear/bicubic chosen per line and direction), blur σ U[0.3,1.2], γ logU[0.7,1.4], c U[0.5,0.95], b U[0,0.1], noise U[0.01,0.1], differentiable JPEG q U[40,90] | scale 0.9 bilinear, σ 0.75, c 0.725, b 0.05, noise 0.055, q 65 |
| `<preset>+heldout` (test only) | refuses train=True | after the preset's eval pipeline: a PIL LANCZOS round trip at 0.87 and real PIL JPEG q50 |

## Pixel statistics, full stats run (5 ems, 256 lines; `stats.md`, `stats.json`, `strips/`)
dist = histogram L1 + relative Δ mean ink + relative Δ edge sharpness, measured against the PIL source for the same mode.
The eval reference is the Phase-2 test images. The train reference is the Phase-2 train rendering plus augment_batch.

| preset | mean dist, eval vs PIL test | mean dist, train vs PIL train |
|---|---|---|
| clean | **0.092** | 1.010 (no photometric augmentation) |
| default | 0.941 (eval adds noise σ 0.03; Phase-2 test has none) | **0.075** |
| harsh | 2.090 | 1.102 |
| harsh+heldout | 2.230 | n/a |

* **clean eval reproduces the Phase-2 test images.** Mean ink is within 1%, edge sharpness within about 4%
  (the bank is slightly sharper because glyph origins are snapped to the ss grid), and the histograms match.
* **default train mode reproduces the Phase-2 train images** (PIL plus augment_batch).
* Pixel-aligned relative MAD (phase 0 on both sides) over the first 12 chars is 0.13–0.17 for clean against PIL.
  For scale, shifting the same PIL rendering by 1/4 px gives 0.28–0.37. Over full lines the MAD is 0.32–0.60,
  because rounding each advance to whole ss px accumulates drift of up to about 0.5–1 px by the end of the line.
  The drift is largest at em 7: the bank's line is about 1.4 px wider, so the local glyph spacing is right but
  the line is slightly stretched.
* Up to 5% of a glyph's ink is clipped at its cell edge (negative side bearings, e.g. `j`). That is negligible.
* The integer pitch shifts the budget against Phase 2's 1.2·em: px²/char is +4% at em 4 (h=5), −3% at em 6
  (h=7) and −5% at em 7 (h=8). Rows record both `em` and `height_px`.
* harsh roughly halves edge sharpness (0.10 vs 0.21) and cuts peak ink to about 0.35 (PIL 0.65–0.80).
  At em ≤ 5 it may be close to unreadable for DejaVu (see `strips/em5_*.png`).

**Recommendation:** to reproduce Phase 2 exactly, train through `default` and evaluate through `clean`.
The script reports this automatically as `extra_eval_cer.clean`. Use `default` eval and `harsh`/`+heldout` as the stress tests.

## Training smoke (results/calibration_quick)
Setup: em 5, train `default`, 300 steps, 5k train lines.
Result: the loss went 4.34 → 3.15 and CER was 1.0 throughout. That is the normal CTC plateau:
Phase-2 latin em5 was also at CER 1.0 at step 250 (loss 4.36) and first left it by step 500.
The run cost 0.29 s/step at 1 thread, against about 0.33 s/step for Phase 2.
It checks the mechanics only. The smoke gives no CER evidence about whether the channel is calibrated.

## Running the full E0
```bash
# pixel stats + 15 training runs (3 presets x ems 4..8, 3500 steps, Phase-2 data sizes), 3 workers x 1 thread
nohup python3 learned_glyphs/experiments/calibrate_channel.py --train --parallel 3 \
    > learned_glyphs/results/calibration/e0.log 2>&1 &
# results: runs.jsonl (phase="lg", scheme="lg_fixed_n0", channel=<preset>, extra_eval_cer{clean, harsh+heldout}),
#          e0_training.md (P@3% per preset and per eval channel vs Phase-2 latin 15.1 px²/char), curves/, logs/, samples/
# resumable: finished (channel, em, steps) rows are skipped.
```
Each run takes about 20–40 min, depending on em.
* `--train-presets default --train-ems 4 5 6` gives a cheaper first pass. P@3% for Phase-2 latin lies between em 4 and em 5.
* `--pil-ref` reruns the unchanged Phase-2 trainer when `--steps` is not 3500.

Pass criterion (PLAN E0): P@3% within about 15% of Phase-2 latin (15.1 px²/char).
* If clean or default comes out **easier** than Phase 2, harden the channel.
* If harsh is far harder, that is expected: harsh is the stress test, not the calibration point.
