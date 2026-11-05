# docs/ index

Start with [HANDOFF.md](HANDOFF.md) (current state and next steps), then [RESULTS.md](RESULTS.md). The repo
overview is the top-level [README.md](../README.md); how to run things and the gotchas are in [AGENTS.md](../AGENTS.md).

## Current status
* [HANDOFF.md](HANDOFF.md): status table, where the rows are, open items, the prompt for the next GPU session.

## Results
* [RESULTS.md](RESULTS.md): cross-project summary: Phases 1-2 (KILL), learned glyphs (qualified GO), frozen encoders, round 3.
* [CONTEXT_BUDGET.md](CONTEXT_BUDGET.md): chars per context position per encoding, turned into context, KV and
  compute on Qwen2.5-VL-3B; potential vs measured (chart `img/ctx_budget.png`).
* [learned_glyphs/RESULTS.md](../learned_glyphs/RESULTS.md): learned glyph alphabet, full report.
* [encoder_probe/RESULTS.md](../encoder_probe/RESULTS.md): frozen-encoder probe, rounds 1-2 (SigLIP, DeepEncoder).
* [encoder_probe/ROUND3.md](../encoder_probe/ROUND3.md): round 3 (Qwen2.5-VL-3B) design, gates, amendments and results (G0, Q1, G2-lite, R3-1).

## Runbooks
* [GPU_RUNNER.md](GPU_RUNNER.md): any Linux + NVIDIA GPU, time-boxed: measure in the smoke, then size equal-step arms with blind controls.
* [LOCAL_GPU.md](LOCAL_GPU.md): the user's Windows RTX 3090 PC: setup, round-2 queue, round 3 (section 8).

## Plans and ideas
* [PLAN.md](../PLAN.md): original design, pre-registered predictions and kill criteria (Phases 1-2).
* [learned_glyphs/PLAN.md](../learned_glyphs/PLAN.md), [encoder_probe/PLAN.md](../encoder_probe/PLAN.md),
  [encoder_probe/ROUND2.md](../encoder_probe/ROUND2.md), [encoder_probe/RGB_OVERLAY.md](../encoder_probe/RGB_OVERLAY.md): sub-project plans.
* [IDEAS.md](IDEAS.md): idea backlog (RGB overlay, rgb6, braille control, fallbacks if the moonshot fails), each with a dated status.
* [MOONSHOT_BRIEF.md](MOONSHOT_BRIEF.md): brief for the external research task on the RGB-overlaid learned-glyph code.

## Reference
* [prior-art.md](prior-art.md): prior art for Visual BPE (Glyph, DeepSeek-OCR, PIXEL, text-side compression).
* [learned_glyphs/docs/prior-art.md](../learned_glyphs/docs/prior-art.md): prior art for learned glyph alphabets.
* [research/2026-11-04_rgb3_lg_go_no_go_report.md](research/2026-11-04_rgb3_lg_go_no_go_report.md): external literature report (go region ~30-90 chars/token).
* [claude-eval-notes.md](claude-eval-notes.md): measurements of Claude as a zero-shot OCR reader (token formula, legibility, cost).
* [dictionary-report.md](dictionary-report.md): Visual-BPE dictionary and kill check K1.
* `img/`: figures; `ctx_*.png` come from `python3 scripts/make_docs_figures.py`, `readme_*` from `scripts/make_readme_figures.py`.

## History
* [history/HANDOFF_2026-11.md](history/HANDOFF_2026-11.md): the stacked session hand-offs of 2026-11-04/05 (local
  3090 round 2, cloud CPU round 3, RTX 4090 R3-1 session with hardware, sizing and samples), verbatim.
