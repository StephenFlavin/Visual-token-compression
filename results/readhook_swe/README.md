# SWE-bench Verified (Django), text vs page images: pre-registration (2026-11-11, before any agent call)

Question: does an agent fix real issues as well when the code and command output it reads are shown as page images
(vbhook at 88.2 px²/char, ~2.9x denser than text for prose) instead of text?

* **Tasks:** `selection.json`: Django tasks from SWE-bench Verified that the two newest Opus submissions on the
  SWE-bench leaderboard both resolved (mini-SWE-agent v2 with Opus 4.6, and with Opus 4.5 at high effort), excluding
  "<15 min fix", hardest first (all ten "1-4 hours", then "15 min - 1 hour" by the mean number of agent steps those runs
  needed). The first 50 that pass validation are run. Selecting tasks Opus can solve means a failure from images is
  about reading, not about the task.
* **Environment** (no Docker): uv virtualenv per task with the version's Python (3.8 for Django 3.0-4.0, where
  SWE-bench uses 3.6 for 3.0-3.2; 3.9 for 4.1-4.2; 3.11 for 5.0), Django editable + tests/requirements/py3.txt.
  `validate`: the gold patch must resolve the task and the unpatched code must not, or the task is dropped.
* **Arms**, Claude Code headless, Opus 5.5 at high effort, tools Bash/Read/Edit/Write, all allowed, max 250 turns:
  `text` (no hooks) and `vbhook` (`experiments/readhook/vbhook/install.py`, budget 88.2). Same prompt (the issue, how
  to run tests, "do not modify the existing tests").
* **Grading** (SWE-bench rules): model patch without its tests/ changes, then the task's test patch; run the test
  modules the test patch touches; swebench's Django log parser; resolved = all FAIL_TO_PASS and PASS_TO_PASS pass.

Predictions:
1. `text` resolves at least 80% of the tasks.
2. `vbhook` resolves within 10 points of `text` (paired; exact McNemar reported). With 50 tasks this is a coarse check:
   a 10-point gap is about 5 tasks.
3. The content the vbhook arm saw as images cost at least 2x fewer tokens than the same content as text (from the
   hook log).
4. Reported without a threshold: total input tokens (incl. cache reads) and list-price cost per task by arm, turns,
   wall time, and for every task vbhook fails but text resolves, the cause (misread, edit failure, other).

## Results (2026-11-11): 50 tasks x 2 arms, all runs complete, $11.86 at list price

| | text | vbhook (pages at 88.2) |
|---|---|---|
| resolved | **50/50** | **50/50** |
| list-price cost (sum) | $5.78 | $6.08 |
| input tokens incl. cache reads (sum / median per task) | 3,451,101 / 50,770 | 3,371,578 / 54,118 |
| output tokens (sum) | 146,034 | 158,785 |
| turns (median) | 6.5 | 7.0 |
| wall time (median) | 31 s | 32 s |

* Per task, vbhook/text input tokens: median 1.05, geometric mean 1.04; cost: median 1.01, geometric mean 1.06. vbhook used less input on 21 of 50 tasks.
* What the vbhook arm saw as images: 126 views (2.5 per task) costing 39,468 tokens against 123,300 for the same content as text (3.12x fewer).
* One vbhook run took 20 minutes (django__django-15128): the model ran Django's whole test suite before and after its change; not the hooks.

Pre-registered verdicts:
1. text resolves >= 80%: **PASS** (100%).
2. vbhook within 10 points of text: **PASS** (100% vs 100%; no discordant task, McNemar p = 1). With every task solved by both arms the comparison has no power to find a small difference: it rules out a large one.
3. content seen as images >= 2x fewer tokens than as text: **PASS** (3.12x).
4. Totals: no saving overall. The agent reads little on these tasks (median 6.5 turns, ~30 s, targeted `grep`/`sed` of a few dozen lines that stay text below the size threshold), so the file content that became images is a few percent of the input; the rest is Claude Code's system prompt and tool definitions and the conversation, re-read (from the prompt cache) on every turn. Small differences in the number of turns move the total more than the images save.

Conclusion: reading code and command output as page images at 2.9x density did not cost Opus 5.5 any task (50/50, the same as text) on the hardest Django tasks recent Opus models solve, and the images themselves cost 3x fewer tokens than the text they replaced; but on tasks this short, file reading is too small a share of an agent's input for the total to fall. The saving needs work dominated by reading: large files, many files, long outputs.

### Caching (added 2026-11-11 on review): warm vs cold, and what the token totals mean

The "input tokens" above add uncached input, cache writes and cache reads as if they cost the same. Split from each
run's usage record (`stream.jsonl`):

| | text | vbhook |
|---|---|---|
| uncached input | 636 | 660 |
| cache writes (1.25x input price) | 451,201 | 463,075 |
| cache reads (0.1x input price) | 2,999,264 | 2,907,843 |
| output | 146,034 | 158,785 |
| cost (CLI, list price) | $5.78 | $6.08 |

* Warm vs cold is the same for both arms: the first call of every run (100/100) read the start of Claude Code's system
  prompt (~3.4k tokens) from a warm cache; median first-call write ~3.5k in both arms.
* 87% of input is cache reads at 0.1x, so the 3.1x saving on image content saves mostly the cheapest category (plus
  one cache write per image). Output tokens, about half the cost at $20/M, are not reduced by images; the vbhook arm
  wrote 8.7% more output, which is most of its 5% higher cost. Dollar savings need image savings that are large next
  to output and cache writes: long-context reading (large files kept in context over many turns).
