# Repo atlas on hard coding tasks, again: two clean runs per task and arm (Q1)

*Pre-registered 2026-11-12, before the first scored agent run. Design: `docs/claude/REPO_ATLAS_PLAN.md`, "Queued
(2026-11-12)", Q1. Code: `experiments/atlas/repo_atlas_swe_rerun.py` (tests `tests/test_repo_atlas_swe_rerun.py`), which
imports the first test's runner `experiments/atlas/repo_atlas_swe.py` unchanged. First test:
[results/repo_atlas_swe/README.md](../repo_atlas_swe/README.md). Results go below the "Results" heading; nothing above
it changes after the first scored run, except dated amendments.*

## Question

The first test (one run per task and arm) found Claude Code with the repo atlas in its first message solved 11 of 16
hard SWE-rebench tasks against 8 without (3 discordant tasks, all the atlas's way, McNemar p = 0.25), in fewer turns
(median 10 vs 12.5, sign test p = 0.006), at 2.6x the cost. Two things could make that 11-vs-8 hint wrong: run-to-run
variation (never measured on these tasks), and Claude Code's auto-memory (`~/.claude/projects/<cwd>/memory/`), which
was on. Does the hint survive two fresh runs per task and arm with auto-memory off and the package index blocked
from the start?

## First, free: did the first test's runs read or write a memory store?

`repo_atlas_swe_rerun.py check` (output `first_test_memory_check.json`), run on the 51 first-test transcripts, which
are still on disk in the scratchpad:

* **Each run had its own store.** Every run ran in its own directory (`runs/<task>/<arm hash>/<repo>`), so each init
  event names a different auto-memory path: 51 distinct paths for 51 runs. No two runs could share a store.
* **No memory was written or read.** None of the 51 stores holds a memory file now (the stores exist; the only file
  in any of them is one persisted tool output, below). No tool call in any transcript reads, lists or writes a path
  under `~/.claude/`, a `MEMORY.md` or a `memory/` directory.
* The one hit: in jsonpickle-587's `map` run a Bash output over the size limit was saved by the CLI to
  `~/.claude/projects/<that run's slug>/<session>/tool-results/...txt`; the agent did not open it. That is the CLI's
  own spill file, not memory.

So the first test was not contaminated through auto-memory: a store is keyed by the run's directory, and no two of its
runs shared one. This re-run still turns memory off, as the user asked, and checks every run.

## Tasks

The first test's 16 compared tasks (17 accepted, minus celestoai__smolvm-172, left out there for a leak): fromager-1124,
hermes-webui-330_interface, tox-3904, pyfakefs-1286, pyjanitor-1593, docstub-123, jsonpickle-588, jsonpickle-587,
pyinaturalist-643, fabric-cicd-670, sqlglot-6067, anyio-975, dspy-8607, virtualenv-2940, arta-50, sqlglot-5359 (ids,
dates and sizes: `results/repo_atlas_swe/selection.json`).

Environments are rebuilt from the first test's records: the base commit alone from today's bare git stores
(`pq/git`, re-used, not re-downloaded), a fresh one-commit repository, a uv venv with the task's Python and install
command. Each is validated exactly as the first test did (`repo_atlas_swe.validate`: the base with the test patch
fails some FAIL_TO_PASS, gold + test patch passes all, at most half of PASS_TO_PASS env-broken) before any agent
run; rows in `validate.jsonl`, with the env-broken PASS_TO_PASS compared to the first test's. **A task that does not
validate is dropped and reported, not replaced.**

**Validation (done before this pre-registration was committed): all 16 tasks validate**, and on every task the set of
env-broken PASS_TO_PASS is identical to the first test's (tox-3904 1, jsonpickle-587/588 2 each, dspy-8607 9, others 0).
Installs took 1-6 s (uv cache), test runs 2-217 s.

## Held fixed from the first test

* Model `claude-opus-5-5`, `--effort high`, Claude Code CLI 2.1.322 headless (`claude -p`, stream-json in and out),
  tools `Bash,Read,Edit,Write` (all allowed), `--setting-sources ""`, `--strict-mcp-config`,
  `--disable-slash-commands`, `--no-session-persistence`, `--max-turns 250`, 5400 s timeout, clean environment
  (`claude_eval.build_env`), no web tools.
* The first message: `repo_atlas_swe.first_message` (the same prompt text). The atlas arm sends the **same page
  files** the first test sent (its builds in `ra_swe/ctx`, linked, not rebuilt) and its index.
* Checkout, environment build (`fresh_repo`, `make_env`), grading (`exclude_files`, the task's test command,
  `parse_pytest_log`, `grade_status`, the env-broken PASS_TO_PASS of this validation left out), the page-arrival
  check (first call's input >= 97% of the image-token formula), turns and tokens summed over every result event
  (`stream_totals`).

## What differs

1. **Arms:** `plain` and `atlas` only (the repo map changed nothing and is dropped).
2. **Two independent runs per task and arm:** 64 runs. Each run gets its own fresh checkout and venv in a neutral
   directory `runs/<hash of task, arm, run>/` (no arm or task name in any path), its own cwd, a clean env. Nothing is
   read from the harness's response cache (agent runs never use it). The API's prompt cache may still serve the
   identical atlas pages to a second run of the same task if it starts within minutes; that lowers its cost, not its
   input.
3. **Auto-memory off:** `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` in the child env. Before and after each run its project
   store (`~/.claude/projects/<slug of the cwd>/`) is listed and deleted; each row records whether a memory file
   existed before or was written during the run (`memory_before`, `memory_after`).
4. **Package index blocked for every run from the start** (`repo_atlas_swe.agent_env`: `PIP_NO_INDEX`, `UV_OFFLINE`,
   index URLs on a closed port). The first test added this after its 4-task pilot.
5. **Leak scan widened:** the first test's pattern (pip download/index, a pip install of the package, pypi.org, git
   fetch/clone/pull/remote, curl, wget, github.com, the dataset) plus any path into the scratch directories outside
   the run's own directory (the old transcripts and patches, other runs, the dataset, git stores), this harness's
   repository, or a memory store. The run's own directory (checkout + venv, which sits in the scratchpad) is removed
   from each command before matching, so the venv path is not a hit. Every hit is read by hand; a run that read a
   release, upstream copy, old patch or transcript of the fix is listed and its task left out of the paired tests
   (both arms).
6. **Order:** all 32 first runs in an order shuffled with seed 20261112, then all 32 second runs in a second shuffled
   order. The first plain and first atlas job of that order run first as the smoke runs: they are scored like the
   rest and their cost counts. At most 2 runs at once (as in the first test).
7. Patches are taken and applied as bytes (`git diff --binary`; the plan-quality runner needed this for a non-UTF-8
   patch on pyfakefs-1286). The grading command and rule are unchanged.

## Scores and tests

Per run: resolved (all FAIL_TO_PASS and the non-broken PASS_TO_PASS pass), turns, cost (`total_cost_usd`, list
price), seconds, tokens. Per task and arm: resolved runs of 2.

* Per arm: task-runs resolved of 32; tasks resolved in both / one / neither run; median turns; mean and median cost
  per run; median seconds.
* **Sign test over tasks** on resolved runs per task (0-2), atlas vs plain.
* **Exact McNemar** on "resolved in both runs", atlas vs plain.
* **Sign test over tasks on mean turns** (and the same, descriptively, for cost and seconds).
* **Bootstrap 95% interval over tasks** (10,000 resamples, seed 20261112) for atlas minus plain resolved task-runs.
* **Run-to-run agreement:** per arm, the tasks whose run 1 and run 2 differ in resolved (the noise the first test
  could not see); and atlas vs plain within run 1 and within run 2 separately.
* The first test's rows (one run each) are shown beside the new ones per task and arm, **never pooled**.

## Predictions (from the design, unchanged)

* **A1:** the atlas resolves at least as many task-runs as plain.
* **A2:** the gap is smaller than the first test's 3 of 16: atlas minus plain under 19 points of task-runs, i.e. at
  most 5 of 32 (6 of 32 = 18.75 points equals the first test's gap).
* **A3:** fewer turns with the atlas: sign test over tasks on mean turns, p < 0.05, atlas lower on the majority.
* **A4:** the atlas costs at least twice plain (mean cost per run).

## Spend and stop rules

* **Hard cap $40 at list price** (the CLI's `total_cost_usd`). Estimate ~$30 (32 plain at ~$0.26, 32 atlas at ~$0.68,
  from the first test). Every run's cost goes to `spend.jsonl` before it is graded. Before each run the runner
  reserves an upper bound (the larger of $1.50 / $2.50 for plain / atlas and 1.5x the dearest run of that arm so far,
  in the first test or here) and does not start a run that could cross the cap with the runs in flight.
* If the cap would cut the design, the shuffled order already finishes every first run before any second run, so a
  truncated test is still one full replication.
* **Usage limit:** a run that ends in a usage-limit error is not recorded as a result (its cost is booked); the runner
  stops launching, the rows so far are committed and pushed, and what remains is reported. The runner is resumable
  (finished (task, arm, run) rows are skipped).
* Refusals are reported, not worked around; prompts are not reworded. Tests are never edited or skipped.
* A run with no result event is retried once (as in the first test).

## Amendment 1 (2026-11-12, after the two smoke runs, before the third run)

* Smoke runs: docstub-123 plain (not resolved, 11 turns, $0.20, 62 s) and pyjanitor-1593 atlas (resolved, 12 turns
  over 3 result events, $3.61, 2,637 s: the agent ran the full test suite with `-n 4` in the background for ~30
  minutes and waited on it; the atlas's 131k image tokens were re-read, and re-written after cache expiry, at each
  turn). Pages arrived (140,782 first-call input vs 131,169 formula). Neither run's init event lists an auto-memory
  path (memory off); no project store existed before or after either run.
* **The leak scan flagged the CLI's own background-task output files** (`/tmp/claude-0/<slug of the run's cwd>/<session>/tasks/*.output`,
  read by `tail` while waiting). That directory is now treated like the run's own directory
  (`own_prefixes`); a path to any other run's temp dir is still a hit. Stored suspects are re-scanned with the current
  pattern at analysis. No scoring rule changes.
* The atlas run's cost bound is now 1.5 x $3.61 = $5.41 per run (the rule above). The $30 estimate assumed $0.68 per
  atlas run; if more runs behave like this one, the cap may cut the second replication. The order already puts all
  first runs first.

## Results

**Status (2026-11-12, 21:20 UTC): complete, 64 of 64 runs, $35.07 of the $40 cap.** The test was interrupted at 54
runs by a container restart at ~19:23 UTC and finished later the same day with the same code, the same order and the
same reservation rule (`run --cap 40 --workers 2`, which skips finished rows). Two runs that were in flight at the
restart (pyjanitor-1593 plain r2, sqlglot-6067 plain r2) were never booked: their half-written run directories and
CLI temp directories were deleted and both were run again from scratch; whatever they had been charged before the
restart is not in `spend.jsonl` (no result event, so no cost figure). Near the end the reservation rule refused
jsonpickle-588 atlas r2 ($33.20 spent + $1.50 reserved for the run in flight + $5.41 bound > $40) and stopped the
runner; once the run in flight was booked ($34.16) the runner was started again with `--workers 1` and the same rule
admitted both remaining runs ($34.16 + $5.41 < $40). `summary.md` is the runner's `analyse` output on these rows.

**The first test's 11-vs-8 hint does not repeat: on two fresh runs per task and arm, plain solves 19 of 32 task-runs
and the atlas 17.** The interim reading (from 54 runs) stands: the 10 added runs changed no conclusion.

| | plain | atlas |
|---|---|---|
| task-runs solved of 32 | **19** | **17** |
| tasks solved in both / one / neither run | 9 / 1 / 6 | 8 / 1 / 7 |
| run 1: solved of 16, median turns, mean cost | 10, 11.5, $0.25 | 9, 9.5, $0.85 |
| run 2: solved of 16, median turns, mean cost | 9, 11, $0.28 | 8, 12, $0.81 |
| median turns per run (32 runs) | 11 | 11.5 |
| mean / median cost per run | $0.27 / $0.24 | $0.83 / $0.61 |
| median seconds per run | 82 | 103 |
| first test (one run, auto-memory on): solved of 16, median turns, mean cost | 8, 12.5, $0.26 | 11, 10, $0.68 |

Per task (resolved in run 1 / run 2; first test's single run beside it, never pooled):

| task | plain | atlas | first test plain / atlas |
|---|---|---|---|
| fromager-1124 | no / no | no / no | no / no |
| hermes-webui-330_interface | no / no | no / no | no / no |
| tox-3904 | yes / yes | yes / yes | yes / yes |
| pyfakefs-1286 | yes / yes | yes / yes | yes / yes |
| pyjanitor-1593 | yes / yes | yes / yes | yes / yes |
| docstub-123 | no / no | no / no | no / no |
| jsonpickle-588 | yes / yes | yes / yes | yes / yes |
| jsonpickle-587 | yes / yes | **no / no** | no / yes |
| pyinaturalist-643 | yes / yes | yes / yes | yes / yes |
| fabric-cicd-670 | no / no | no / no | no / no |
| sqlglot-6067 | yes / yes | yes / yes | yes / yes |
| anyio-975 | yes / no | yes / yes | yes / yes |
| dspy-8607 | no / no | no / no | no / yes |
| virtualenv-2940 | no / no | no / no | no / no |
| arta-50 | yes / yes | yes / yes | yes / yes |
| sqlglot-5359 | yes / yes | yes / no | no / yes |

**Pre-registered paired tests (16 tasks, all validated, none left out for a leak):**

* Resolved runs per task (0-2): atlas higher on 1 task (anyio-975), plain higher on 2 (jsonpickle-587,
  sqlglot-5359), 13 tied; sign test p = 1.0.
* Resolved in both runs: atlas only 1, plain only 2; exact McNemar p = 1.0.
* Atlas minus plain: -2 of 32 task-runs (-6.2 points); bootstrap 95% interval over tasks -7 .. +2.
* Mean turns per task: atlas lower on 10 of 15 untied tasks, sign test p = 0.30 (medians 10.75 vs 12.25). Within
  run 1 alone 6 lower / 6 higher; within run 2 11 lower / 5 higher. Descriptively: mean cost atlas higher on all 16
  (p < 0.0001, 3.1x); mean seconds atlas lower on 7 of 16 (p = 0.80).
* **Run-to-run flips:** one task per arm (plain: anyio-975 yes then no; atlas: sqlglot-5359 yes then no). The other
  15 tasks per arm give the same answer twice. Within run 1 the arms disagree on one task (jsonpickle-587, plain's
  way); within run 2 on three (two plain's way, one the atlas's).
* **Against the first test:** of its three atlas-only tasks, jsonpickle-587 is now solved by plain in both runs and
  by the atlas in neither, sqlglot-5359 by plain twice and the atlas once, dspy-8607 by neither arm in four runs.
  Plain rose from 8 to 9-10 solved per replication with nothing changed for it but memory off and the index blocked
  from the start; the atlas fell from 11 to 8-9. With one flip per arm across 16 tasks, run-to-run noise alone is
  small here; the first test's 3-0 split was more likely a lucky draw on a few borderline tasks (jsonpickle-587,
  sqlglot-5359, dspy-8607) than a stable atlas effect.

**Predictions:**

* **A1 FAIL:** the atlas resolves fewer task-runs than plain (17 vs 19).
* **A2 PASS:** the gap is -6.2 points, well under 19 (and in plain's direction).
* **A3 FAIL:** atlas lower on mean turns on 10 of 15 tasks, sign test p = 0.30 (needed p < 0.05). The first test's
  "fewer turns" (p = 0.006) is at most a weak tendency here.
* **A4 PASS:** the atlas costs 3.1x plain per run ($0.83 vs $0.27).

**Checks:**

* Pages arrived in every atlas run (first-call input >= 97% of the image-token formula); no permission denials, no
  timeouts, no errors. Three runs had more than one result event (background test runs: pyjanitor-1593 atlas r1 and
  plain r2, anyio-975 atlas r2); turns and cost are summed over all of them.
* **Memory:** no run had a memory file before or after (0 of 64); no init event lists an auto-memory path. Two
  project stores held only the CLI's own persisted tool outputs (`<session>/tool-results/*.txt`: dspy-8607 atlas r1,
  jsonpickle-587 plain r2), deleted after the run.
* **Leaks:** 6 runs are flagged by the widened pattern after the amendment-1 re-scan. Read by hand, every hit is a
  path into the run's own directory inside a longer command (the stored suspect strings are truncated, which cuts
  the own-directory prefix and leaves a bare scratchpad path) or a scratch file the agent wrote itself under
  `/tmp/claude-0/` (`t_reason.py`, `t_slots.py`, `patch_expand.py`, `before.txt` / `after.txt`). `/tmp/claude-0/`
  is shared between runs, so all 64 transcripts were checked for a read of a file there that the same run had not
  written first, or a listing of `/tmp`: none. No download, package index, release, git-history or old-patch access
  in any run. No task is left out.
* **Spend:** $35.07 at list price for the 64 booked runs ($27.19 before the restart, $7.88 after), plus the unknown
  charge of the two runs killed by the restart.
