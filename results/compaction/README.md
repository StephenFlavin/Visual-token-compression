# Context compaction with a page-image archive: pre-registration (2026-11-11, before any agent call)

Question (user): when Claude Code compacts a long session, can the full pre-compaction transcript be kept as page
images that the model opens on demand, and what does that cost and gain against normal compaction?

Mechanism (verified headless with Haiku): a PreCompact hook receives `transcript_path` (which still holds the
pre-compaction messages) and can render it; a SessionStart hook with matcher `compact` adds a note after compaction,
which the model sees. No hook can edit the summary itself. In this test the archive pointer goes in the question
prompt instead, so all arms fork from one compacted session.

Design (`experiments/compaction/run_compact.py`), Opus 5.5, per corpus (the 3 deep-research corpora, 34-50 real
Wikipedia sources with invented facts, ~240-290k tokens of text, 120 exact-answer questions: 60 lookups, 60 across
sources):

1. One session reads every source file with Read (low effort; shared by all arms); the source files are then removed.
2. **D, no compaction:** a fork answers the questions (high effort).
3. `/compact` on the session (one summary shared by A-C).
4. Forks of the compacted session answer the same questions with Read only:
   **A, normal compaction** (summary only); **B, page archive** (the sources from the transcript as 88.2 px²/char
   pages, ~3x fewer tokens than text, plus a text index of which sources each page holds); **C, text archive** (the
   same chunks as text files, same index).

Metrics: accuracy (all, lookups, synthesis) and the list-price cost of the question phase; reading and compaction
costs reported separately (shared).

Predictions:
1. A (normal compaction) loses at least 30 points against D: a summary cannot hold 60 planted facts and numbers.
2. B recovers to within 5 points of D, and within 5 points of C.
3. B's question phase costs less than C's (at least 1.5x less), because looking things up in an archive is reading.
4. Reported without a threshold: B's cost and accuracy against A (the user's two key metrics), points gained per
   extra dollar, and how many archive files each arm opened.
