# docs/ index

The story in order is the top-level [README.md](../README.md); setup and working rules are in
[AGENTS.md](../AGENTS.md). The research is wound down (2026-11-12): every document below is a record unless its
line says otherwise.

## Top level

* [HANDOFF.md](HANDOFF.md): where the work stands, the status table, where the rows are, and the prompts for the RTX 3090. Current.
* [RUNNING.md](RUNNING.md): every command, one row per experiment. Current.
* [GOTCHAS.md](GOTCHAS.md): things that don't work, by area (network, Claude as a reader, PP-OCR, rendering, training, GPU runners, cloud runner, git). Current.
* [FINDINGS.md](FINDINGS.md): the findings log in the order they were made, with numbers and row locations. Current.
* `img/`: figures used by the docs and the README (`scripts/make_*_figures.py` write them).

## claude/: page images as context for Claude

* [CLAUDE_QA.md](claude/CLAUDE_QA.md): questions answered from page images vs text on long documents. Result: 2.2x fewer tokens lossless on the original questions, 4.2x loses multi-step questions.
* [SWE_IMAGES.md](claude/SWE_IMAGES.md): coding agents reading code and output as page images on SWE-bench. Result: cost does not fall.
* [RAG_PLAN.md](claude/RAG_PLAN.md): RAG with page images. Plan; rows and result in [results/rag](../results/rag/README.md).
* [ERB_PLAN.md](claude/ERB_PLAN.md): EnterpriseRAG-Bench with page images. Plan, prep done ([results/erb](../results/erb/README.md)).
* [REPO_ATLAS_PLAN.md](claude/REPO_ATLAS_PLAN.md): a whole repository as page images. Plan; the tests are closed, results in [results/atlas](../results/README.md).
* [PLAN_QUALITY_TEST.md](claude/PLAN_QUALITY_TEST.md): briefs an orchestrator writes from the atlas. Design and result: not worth its cost.
* [CONTEXT_BUDGET.md](claude/CONTEXT_BUDGET.md): chars per context position per encoding, as context, KV and compute on Qwen2.5-VL-3B. Living table, dated 2026-11-05.
* [claude-eval-notes.md](claude/claude-eval-notes.md): Claude as a zero-shot OCR reader: token formula, legibility, cost. Reference.

## gpu/: runbooks and plans for GPU work

* [GPU_RUNNER.md](gpu/GPU_RUNNER.md): runbook for any Linux GPU, time-boxed, with blind controls.
* [LOCAL_GPU.md](gpu/LOCAL_GPU.md): runbook for the Windows RTX 3090 PC: setup, round 2, round 3, Test B.
* [LOCAL_QWEN.md](gpu/LOCAL_QWEN.md): page images and generation speed on a local Qwen3.8-27B. Pre-registered, not run: the one test left.
* [PLAN_3090.md](gpu/PLAN_3090.md): capacity curve, trained-ViT Latin vs glyphs, channel capacity; amendments 1-6 and results. Run 0 done; the Runs 1-2 driver is stopped.
* [TEXT_SLOTS.md](gpu/TEXT_SLOTS.md): Test B, text slots against page pixels. Pre-registered and built, not scheduled.
* [OWN_ENCODER.md](gpu/OWN_ENCODER.md): training an own encoder; Stage 0 (text-side compression ceiling). Result: inconclusive.
* [COLOUR_GIST.md](gpu/COLOUR_GIST.md): Latin in 2-3 colour channels, gist as the target. A0, A1-cpu and S1 results; Run 4 closed.
* [MOONSHOT_BRIEF.md](gpu/MOONSHOT_BRIEF.md): brief for the external research task on the RGB-overlaid learned-glyph code. Kept as sent.

## glyphs/: the original idea, Phases 1-2

* [PLAN.md](glyphs/PLAN.md): original design, pre-registered predictions and kill criteria (was the root `PLAN.md`).
* [RESULTS.md](glyphs/RESULTS.md): cross-project results: Phases 1-2 (KILL), learned glyphs (qualified GO), frozen encoders, round 3.
* [dictionary-report.md](glyphs/dictionary-report.md): the Visual-BPE dictionary and kill check K1 (`scripts/build_data.py --report` writes it).

## background/: prior art, reviews, history

* [prior-art.md](background/prior-art.md): prior art for Visual BPE (Glyph, DeepSeek-OCR, PIXEL, text-side compression).
* [IDEAS.md](background/IDEAS.md): idea backlog, each with a dated status.
* [ANALYSIS.md](background/ANALYSIS.md): the results against blind controls: significance, information per vision token, where the glyph codes lose their information.
* [REVIEW_2026-11-11.md](background/REVIEW_2026-11-11.md): review of goals, methods and next steps; the goals were reframed from it.
* [history/HANDOFF_2026-11.md](background/history/HANDOFF_2026-11.md): the session hand-offs of 2026-11-04/05, verbatim.
* [research/2026-11-04_rgb3_lg_go_no_go_report.md](background/research/2026-11-04_rgb3_lg_go_no_go_report.md): external literature report on the moonshot code.

## In the packages

* [learned_glyphs/PLAN.md](../learned_glyphs/PLAN.md), [learned_glyphs/RESULTS.md](../learned_glyphs/RESULTS.md), [learned_glyphs/docs/prior-art.md](../learned_glyphs/docs/prior-art.md): the learned glyph alphabet.
* [encoder_probe/PLAN.md](../encoder_probe/PLAN.md), [RGB_OVERLAY.md](../encoder_probe/RGB_OVERLAY.md), [ROUND2.md](../encoder_probe/ROUND2.md), [ROUND3.md](../encoder_probe/ROUND3.md), [RESULTS.md](../encoder_probe/RESULTS.md): the frozen-encoder probe, rounds 1-3.
* [experiments/readhook/README.md](../experiments/readhook/README.md): the vbhook hooks that show files and command output as page images.
* [results/README.md](../results/README.md): where each stage's rows live. [scripts/README.md](../scripts/README.md): setup, housekeeping and figure scripts.

`python3 scripts/check_links.py` checks every link in these files.
