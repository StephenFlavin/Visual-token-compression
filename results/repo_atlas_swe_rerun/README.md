# Repo atlas on hard coding tasks, again: two clean runs per task and arm (Q1)

*Pre-registered 2026-11-12, before the first scored agent run. Design: `docs/REPO_ATLAS_PLAN.md`, "Queued
(2026-11-12)", Q1. Code: `experiments/repo_atlas_swe_rerun.py` (tests `tests/test_repo_atlas_swe_rerun.py`), which
imports the first test's runner `experiments/repo_atlas_swe.py` unchanged. First test:
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

*(to be filled in after the runs)*
