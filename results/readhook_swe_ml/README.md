# SWE-bench Multilingual, Rust and Java: text vs page images, pre-registration (2026-11-11, before any agent call)

Same question and arms as `results/readhook_swe/` (Django), now on brace languages, where vbhook's minifier drops
cosmetic whitespace and shows indentation as `»` levels (edit-safe mode; +3.5% page tokens over no levels).

* **Tasks:** `selection.json`: the Rust and Java tasks of SWE-bench Multilingual that both Opus submissions on its
  leaderboard resolved (mini-SWE-agent v2.0.0a0 with Opus 4.6 and Opus 4.5), hardest first by mean agent steps:
  47 tasks (Rust: tokio 8, bat 6, ruff 5, nushell 5, axum 4, ripgrep 2, coreutils 2; Java: lucene 5, gson 4, druid 3,
  javaparser 2, rxjava 1). Lombok (13 solved tasks) is left out: its tests need Apache Ant, which is not installed and
  could not be installed in this session.
* **Environment** (no Docker): installed Rust 1.97 and JDK 21 (the task images may use other versions), Maven for
  `mvnd`, the projects' Gradle wrappers; one blobless clone per repository, a git worktree per run, Rust builds
  sharing a target dir per repository with debug info off.
* **Validation and grading:** the task's own eval_script on the worktree; swebench's parser for the task; the gold
  patch must pass FAIL_TO_PASS and the base must not; PASS_TO_PASS tests that fail with the gold patch here are left
  out for every arm (at most half, or the task is dropped).
* **Arms:** as for Django: Claude Code headless, Opus 5.5 at high effort, tools Bash/Read/Edit/Write, `text` vs
  `vbhook` (88.2 px²/char), same prompt.

Predictions (same as Django):
1. `text` resolves at least 80% of the valid tasks.
2. `vbhook` resolves within 10 points of `text` (paired; exact McNemar).
3. The content the vbhook arm saw as images cost at least 2x fewer tokens than as text.
4. Reported: total input tokens and cost by arm, turns, wall time, failure causes for every discordant task; and
   whether multi-line Edits in brace languages fail more often with images (the `»`-level fix is new).
