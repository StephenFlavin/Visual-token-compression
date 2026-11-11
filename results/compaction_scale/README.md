# Compaction archive at scale: pre-registration (2026-11-11, written before any agent call)

Follow-up to [results/compaction](../compaction/README.md) (3 research sessions: normal compaction 36.9%, page archive
100%, 31% cheaper). That test left four gaps, which this one closes:

| | first test | this test |
|---|---|---|
| sessions | 3, all reading Wikipedia sources | **9 in 3 kinds**: research (Read, 14 Wikipedia articles with planted facts, new seeds), codebase exploration (Read, ~22 Python files of this repository), log triage (Bash, synthetic service logs with planted events) |
| archive delivery | pointer written into the question prompt | **real hooks** (`experiments/compaction/hooks/`): PreCompact renders the whole transcript (files read, command output, the agent's own messages) to 88.2 px²/char pages plus an index; SessionStart(compact) tells the model where it is |
| archive content | Read results only | everything in the transcript, code and logs minified with their line structure (`vbhook` minifiers) |
| questions | 120 in one turn | 40 per session in **4 turns of 10**, then a **task turn**: write a file that needs 5 details from before the compaction |

Arms per session (Opus 5.5; reading at low effort, everything after it at high effort):
* **D, no compaction:** a fork of the reading session answers.
* **A, normal compaction:** a fork runs `/compact` with no hooks, then answers (Claude Code's summary includes the
  raw transcript path, which the model may read).
* **B, compaction + page archive:** a fork runs `/compact` with the two hooks installed, then answers. Nothing about
  the archive is in any prompt.

Questions have exact numeric answers, scored automatically. Research: lookups (needle, number) and cross-source
(citation, conflict, chain, source count, total), 20 + 20. Code: value of a module-level constant, line of a
top-level `def`/`class`, and which file (by reading order) defines a function. Logs: a request's latency, a deploy's
build number, an incident's affected requests, differences between days' p99, which day had the highest p99.

Metrics: accuracy per arm (all, per kind, per turn, task items); cost of everything after the reading (compaction +
answering) from each invocation's usage at Opus 5.5 list prices; cost per correct answer; archive pages opened;
whether the hook's note reached the model.

Predictions:
1. A loses at least 30 points against D (pooled over 360 questions).
2. B is within 5 points of D pooled, and within 10 points of D within each session kind.
3. B costs at most half of A per correct answer (pooled).
4. The hooks work unaided: in every B session the archive note is in the post-compaction context and B opens at least
   one archive page.
5. Task turn: B beats A by at least 30 points on task items (pooled, 45 items).
6. Reported without a threshold: results per kind, cost of compaction vs answering, hook run time, archive size.

Budget: ~$6-7 per session, ~$60 for 9; one session of each kind first as a pilot (~$20), checked before the rest.
