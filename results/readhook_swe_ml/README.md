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

## Results (2026-11-11): 29 valid tasks x 2 arms, $9.58 at list price

Valid here: ripgrep 2, bat 5, axum 3, tokio 8, coreutils 2, ruff 4 (Rust); Lucene 4, Druid 1 (Java). Dropped by
validation: gson 4 and javaparser 2 (do not build or pass with JDK 21), RxJava 1 and Lucene 1 (Gradle under JDK 21),
Druid 2, nushell 5 (gold patches fail here), bat 1, axum 1 (test code does not compile with Rust 1.97), ruff 1.

| | text | vbhook (pages at 88.2) |
|---|---|---|
| resolved | 28/29 | 29/29 |
| list-price cost | $4.83 | $4.75 |
| median turns / wall time | 8 / 120 s | 11 / 119 s |
| Edit/Write calls (failed) | 18 (0) | 37 (0) |
| content seen as images | - | 48,162 tokens vs 151,633 as text (3.15x) |

Pre-registered verdicts:
1. text resolves >= 80%: **PASS** (97%).
2. vbhook within 10 points of text: **PASS** (+3.4 points; one discordant task, astral-sh__ruff-15443, resolved only
   with images; exact McNemar p = 1).
3. images >= 2x fewer tokens than the same content as text: **PASS** (3.15x).
4. Reported: cost $4.75 vs $4.83; no failed Edit or Write in either arm (the `»` indentation levels for brace languages
   worked), though the image arm made twice as many edit calls and more turns. Cost anatomy and why totals do not fall:
   `docs/SWE_IMAGES.md`.

Incidents, all fixed during the run and applying to both arms: an agent's `grep -r` and an orphaned `target/debug/bat`
from bat's own tests hung `cargo test | grep` pipelines (harness now gives agents an empty stdin; `watchdog.sh` stops
programs from the build dirs after 3 minutes, test binaries after 15); ruff's build filled the disk (runs restarted
after the user cleared caches; no finished row was lost); Gradle needed per-test logging for the task parser
(`~/.gradle/init.d/swebench-test-logging.gradle`).
