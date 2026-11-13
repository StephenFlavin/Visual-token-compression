# scripts/

Setup, housekeeping and figure scripts. Run from the repo root (`python3 scripts/<name>`). Experiments live in `experiments/`,
`learned_glyphs/experiments/` and `encoder_probe/experiments/` (commands in [docs/RUNNING.md](../docs/RUNNING.md)).

## Setup and housekeeping

| Script | What it does |
|---|---|
| `fetch_fonts.sh` | Downloads the fonts (Noto Sans CJK SC, Archivo Narrow, Roboto Condensed, README UI font) into `assets/fonts/` |
| `build_data.py` | WikiText-103 -> `data/`: split texts, evaluation passages, `dict_{100,300,1000,3000}.json`; `--report` rewrites `docs/glyphs/dictionary-report.md` (kill check K1), including its status line: restore that by hand |
| `check_claude_vision.py` | Claude vision checks (token formula, legibility, latency, determinism, hidden calls) -> `results/claude_checks/`; spends ~$0.3-0.5 |
| `job_queue.py` | Runs one single-core command per line of a queue file, backfilling free cores; per-job logs in `--log-dir` |
| `autosave_results.sh` | Commits and pushes `runs.jsonl` and other result files every N seconds while long jobs run |
| `merge_phase2.py` | Keeps the longest run per (scheme, font size) from `phase2_main` and `phase2_long` -> `results/phase2/merged/runs.jsonl`, the Phase 2 rows to cite. Follow it with `python3 -m vtc.analysis results/phase2/merged/runs.jsonl --out results/phase2/merged`, or the summary there goes stale |
| `check_links.py` | Checks that every relative link and image in the tracked `.md` files exists (exit code 1 if not) and lists inline-code paths that do not exist as warnings |

## Figure scripts

All write PNGs to `docs/img/` and read their numbers from result rows unless a row below says otherwise; none
calls a model. Each file's docstring
lists its figures and sources. Re-running one overwrites its images (the README's demo figures come out
byte-identical).

| Script | Figures (file prefix) | Used in |
|---|---|---|
| `make_readme_figures.py` | the README's demo figures (`glyphs.png`, `colour.png`, `rgb.png`, `moonshot.png`; needs torch and the fonts) and charts (`readme_density.png`, `readme_r3_1.png`); also the shared chart style the other scripts import | [README.md](../README.md) |
| `make_glyphs_figures.py` | `glyphs_*`: px² per character at 3% error per scheme, error against density, grid-aligned banks | [docs/glyphs/RESULTS.md](../docs/glyphs/RESULTS.md), [learned_glyphs/RESULTS.md](../learned_glyphs/RESULTS.md) |
| `make_docs_figures.py` | `ctx_budget.png`: characters that fit in Qwen2.5-VL-3B's context window per encoding. Its densities and read / not-read marks are typed in from the docs named in its docstring, not read from rows | [docs/glyphs/RESULTS.md](../docs/glyphs/RESULTS.md), [docs/claude/CONTEXT_BUDGET.md](../docs/claude/CONTEXT_BUDGET.md) |
| `make_probe_figures.py` | `probe_*`: codes through frozen encoders, colour layers | [encoder_probe/RESULTS.md](../encoder_probe/RESULTS.md), [docs/gpu/COLOUR_GIST.md](../docs/gpu/COLOUR_GIST.md) |
| `make_gpu_figures.py` | `gpu_*`: the capacity curve of frozen Qwen2.5-VL-3B (Run 0) and the grid-phase sweep | [docs/gpu/PLAN_3090.md](../docs/gpu/PLAN_3090.md) |
| `make_claude_qa_figures.py` | `claude_qa_*`: verdict grid, pages at actual pixels, per-question-type grid, cost against accuracy, effort, position | [docs/claude/CLAUDE_QA.md](../docs/claude/CLAUDE_QA.md) |
| `make_claude_docs_figures.py` | `claude_*`: every tested use at a glance, repeat calls, pruned text against pages | [docs/claude/CLAUDE_QA.md](../docs/claude/CLAUDE_QA.md) |
| `make_rag_figure.py` | `rag_cost_accuracy.png` | [results/rag](../results/rag/README.md) |
| `make_compaction_figure.py`, `make_compaction_scale_figure.py` | `compaction_cost_accuracy.png`, `compaction_scale.png` | [results/compaction](../results/compaction/README.md), [results/compaction_scale](../results/compaction_scale/README.md) |
| `make_swe_figures.py` | `swe_cost_*` | [docs/claude/SWE_IMAGES.md](../docs/claude/SWE_IMAGES.md) |
| `make_atlas_figures.py` | `atlas_*`: the repo-atlas tests and the plan-quality test | [docs/claude/REPO_ATLAS_PLAN.md](../docs/claude/REPO_ATLAS_PLAN.md), [docs/claude/PLAN_QUALITY_TEST.md](../docs/claude/PLAN_QUALITY_TEST.md) |
