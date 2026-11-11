# RAG with page images

Plan, arms and predictions: [docs/RAG_PLAN.md](../../docs/RAG_PLAN.md), committed (9810725) before any call.
Harness: `experiments/rag/run_rag.py` (resumable: finished calls in `runs.jsonl`, responses cached; failed calls
neither recorded nor cached; it stops itself when most of a batch fails, e.g. on a usage limit).

Dry run (no calls): BM25 recall of the gold evidence at 5 / 15 passages: planted facts 0.88 / 0.88, MultiHop-RAG
0.55 / 0.79, HotpotQA 0.72 / 0.90. Budget-matched pages (`Pmatch`) carry 13.7 / 14.3 / 11.5 passages in the tokens of
the top-5 as text; the top-5 as pages cost 700 / 797 / 352 image tokens against ~1,830 / 2,110 / 770 as text.
Deviation from the plan: MultiHop-RAG has 148 questions (37 per type), not 150.
