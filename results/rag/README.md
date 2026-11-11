# RAG with page images

Plan, arms and predictions: [docs/RAG_PLAN.md](../../docs/RAG_PLAN.md), committed (9810725) before any call.
Harness: `experiments/rag/run_rag.py` (resumable: finished calls in `runs.jsonl`, responses cached; failed calls
neither recorded nor cached; it stops itself when most of a batch fails, e.g. on a usage limit).

Dry run (no calls): BM25 recall of the gold evidence at 5 / 15 passages: planted facts 0.88 / 0.88, MultiHop-RAG
0.55 / 0.79, HotpotQA 0.72 / 0.90. Budget-matched pages (`Pmatch`) carry 13.7 / 14.3 / 11.5 passages in the tokens of
the top-5 as text; the top-5 as pages cost 700 / 797 / 352 image tokens against ~1,830 / 2,110 / 770 as text.
Deviation from the plan: MultiHop-RAG has 148 questions (37 per type), not 150.

**Scoring fix after stage A (applied to every arm; all rows are rebuilt from the response cache at no cost):** the
first scorer took the first number on the answer's first line. The model often shows its working on that line
(`4543-3710=833`, `Lengav: 2,007 + 904 = 2,911`), more often with text than with pages, so correct answers were scored
wrong and the text arms looked worse than they were (e.g. "total" questions 22-33% from text vs 89% from pages). The
fix reads the whole answer and takes the number after the last `=`/arrow, else after "total" / "difference" /
"answer", else the only number, else the last.

**Free-text scoring (added during stage B, applied to every arm):** at low effort the model sometimes reasons before
answering on the answer line. Primary score: the candidate after the last "answer:" (else the last line) equals the
gold answer or starts with a gold answer of at most 4 words; the strict first-line exact match is kept as `em_strict`.
