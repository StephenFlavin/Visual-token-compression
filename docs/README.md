# docs/ index

Start with [HANDOFF.md](HANDOFF.md) (current state and next steps), then [RESULTS.md](RESULTS.md). The repo
overview is the top-level [README.md](../README.md); how to run things and the gotchas are in [AGENTS.md](../AGENTS.md).

## Current status
* [REVIEW_2026-11-11.md](REVIEW_2026-11-11.md): review of goals, methods and next steps (the goals were reframed
  from it): keep the Claude page-image line, stop the dense codes after the queued runs, and two tests of text-side
  compression.
* [LOCAL_QWEN.md](LOCAL_QWEN.md): the last planned run, pre-registered: page images and generation speed on a local
  Qwen3.8-27B (RTX 3090), no training; runbook and the 3090 prompt.
* [PLAN_QUALITY_TEST.md](PLAN_QUALITY_TEST.md): design, queued: are briefs an orchestrator writes from the atlas
  good enough to replace scouting and to pick the right worker model and reasoning level per ticket?
* [REPO_ATLAS_PLAN.md](REPO_ATLAS_PLAN.md): draft for approval: a whole repository as page images read up front
  (`experiments/repo_atlas.py`, built), which repositories fit, and a two-stage test against a repo map and a
  searching agent.
* [TEXT_SLOTS.md](TEXT_SLOTS.md): Test B (built, not scheduled as of 2026-11-12), pre-registered: slots made from text tokens against vision tokens made from
  a page, same reader and budget, on the 3090. Test A (prune the text, then render it, on Claude):
  [results/claude_qa/prune](../results/claude_qa/prune/README.md).
* [HANDOFF.md](HANDOFF.md): where the work stands (being wound down), the prompt for the 3090, status table, where the rows are.
* [PLAN_3090.md](PLAN_3090.md): the current go-forward plan (user's 3090 research plan, reviewed 2026-11-06): capacity curve, trained-ViT Latin vs glyphs, colour as Run 4.
* [COLOUR_GIST.md](COLOUR_GIST.md): (parked 2026-11-06 for PLAN_3090.md; colour returns as Run 4) gist through colour channels. Latin in 2 then 3
  channels, then glyphs without BPE, then merge; zero-shot result (A0) and the A1 gate.
* [OWN_ENCODER.md](OWN_ENCODER.md): training our own encoder for the moonshot: what it involves, GPUs, time and
  cost, compute grants, and Stage 0 (text-side compression ceiling), pre-registered with a RunPod prompt.
* [ANALYSIS.md](ANALYSIS.md): the results measured against blind controls (significance, information per vision
  token, where the glyph codes lose their information), an outlook per code and what to run next.

## Results
* [RESULTS.md](RESULTS.md): cross-project summary: Phases 1-2 (KILL), learned glyphs (qualified GO), frozen encoders, round 3.
* [CLAUDE_QA.md](CLAUDE_QA.md): can Claude answer questions from page images as well as from text? Pre-registered: 2.2x fewer tokens lossless, 4.2x loses multi-step questions; charts and cost table.
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
