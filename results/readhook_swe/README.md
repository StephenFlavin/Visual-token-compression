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
