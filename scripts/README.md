# scripts/

Setup, housekeeping and figure scripts. Run from the repo root. Experiments live in `experiments/`,
`learned_glyphs/experiments/` and `encoder_probe/experiments/` (commands in [AGENTS.md](../AGENTS.md)).

| Script | What it does |
|---|---|
| `fetch_fonts.sh` | Downloads the fonts (Noto Sans CJK SC, Archivo Narrow, Roboto Condensed, README UI font) into `assets/fonts/` |
| `build_data.py` | WikiText-103 -> `data/`: split texts, evaluation passages, `dict_{100,300,1000,3000}.json`; `--report` rewrites `docs/dictionary-report.md` (K1) |
| `check_claude_vision.py` | Claude vision checks (token formula, legibility, latency, determinism, hidden calls) -> `results/claude_checks/`; spends ~$0.3-0.5 |
| `job_queue.py` | Runs one single-core command per line of a queue file, backfilling free cores; per-job logs in `--log-dir` |
| `autosave_results.sh` | Commits and pushes `runs.jsonl` and other result files every N seconds while long jobs run |
| `merge_phase2.py` | Keeps the longest run per (scheme, em) -> `results/phase2/merged/runs.jsonl`, the Phase-2 rows to cite |
| `make_readme_figures.py` | Draws the README demo figures (`docs/img/glyphs.png`, `colour.png`, `rgb.png`) |
| `make_docs_figures.py` | Draws the docs figures (`docs/img/ctx_*.png`, context-window budget per encoding) |
