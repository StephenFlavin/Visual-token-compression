# EnterpriseRAG-Bench (prep only, 2026-11-11)

Plan, predictions, commands and costs: [docs/claude/ERB_PLAN.md](../../docs/claude/ERB_PLAN.md). No Claude calls yet.

* `retrieval.jsonl`: per question, the official BM25 top-10 (from the leaderboard's `answers_bm25.jsonl`), our
  Lucene-BM25 reimplementation's top-60 with scores, the merged ranking used by the arms (official top-10, then ours),
  overlap@10 with the official set (mean 0.992) and gold-document recall at 10-50.
* `pilot_qids.json`: the 50 pilot questions (stratified by type).
* `judge/`: the provisional judge's inputs and outputs (`answers_<name>_pilot.jsonl`, `results_<name>.json`,
  `questions_updated_<name>.jsonl`, `usage.jsonl`).
* After a run: `runs.jsonl` (one row per answer call), `answers_<arm>.jsonl` (the benchmark's submission format),
  `summary.md` / `summary.json` (`experiments/erb/analyse_erb.py`).
