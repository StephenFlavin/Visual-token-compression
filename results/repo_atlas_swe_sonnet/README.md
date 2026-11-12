# Sonnet with the repo atlas on the hard coding tasks (Q3)

*Pre-registered 2026-11-12, after the legibility gate and the environment validation, before the first scored agent
run. Design: `docs/REPO_ATLAS_PLAN.md`, "Queued", Q3. Code: `experiments/repo_atlas_swe_sonnet.py` (tests
`tests/test_repo_atlas_swe_sonnet.py`), a thin runner that imports Q1's runner `experiments/repo_atlas_swe_rerun.py`
and the first test's `experiments/repo_atlas_swe.py` unchanged. Results go below the "Results" heading; nothing above
it changes after the first scored run, except dated amendments.*

## Question

Every atlas test so far used Opus, which solves these tasks about as often with the atlas as without (Q1: 9 vs 10 in
run 1). A cheaper model explores and plans less, so a map of the whole repository up front could matter more to it.
Does the repo atlas (the whole repository as page images in the first message) help Claude Code on **Sonnet 5.5**
(`claude-sonnet-5-5`, `--effort high`) solve the same 16 hard SWE-rebench tasks?

## Gate (run before this pre-registration; $0.63)

*Rule (from the design):* 20 lookup questions with answers known from the source, on two of the tasks' atlases;
Sonnet (high effort) and Opus (high effort) answer with the atlas pages (and their index) as their own first message
and the questions as the second; exact or normalised match. Sonnet under 80%: re-render for it (117.6 px² per
character, pages of at most 1568 px, only if every task still fits 150k image tokens; tasks that no longer fit are
dropped and named) and ask again. Still under 80%: stop, and "Sonnet cannot read the atlas" is the result. Per call:
did all pages arrive (first-turn input >= 97% of the image-token formula)?

*Questions* (`gate_questions.json`, `gate-build`): on the atlases of docstub-123 (5 pages) and pyfakefs-1286 (10
pages), 10 each, drawn with a fixed seed so that every atlas has questions on its early, middle and late thirds of
pages (the page of each question is the page that holds its anchor) and of three kinds, answers from the source with
`ast`: the first function or class defined at or after a tagged line `§n` of a named file (9), the parameter names of
a named function in order (9), a module-level constant's value (2; `0o777` and `511` both count).

*Result* (`gate.jsonl`):

| model | docstub-123 | pyfakefs-1286 | total | first-turn input vs formula |
|---|---|---|---|---|
| Sonnet 5.5 high | 10/10 | 10/10 | **20/20 (100%)** | 21,354 vs 19,818; 45,299 vs 43,983 |
| Opus 5.5 high | 10/10 | 10/10 | 20/20 (100%) | 21,350 vs 19,818; 45,295 vs 43,983 |

**The gate passes without re-rendering.** Every answer was the exact string. All pages arrived in all four calls.
**Page size:** the atlas pages are 1932 px on the long side; Sonnet's first-turn input equals Opus's to within 4
tokens, so Sonnet reads 1932-px pages without downscaling, as Opus does (a 1568-px limit would have cut each full page
from 4,761 to about 3,136 image tokens).

## Tasks and environments

The 16 tasks of Q1 (the first test's 17 accepted minus smolvm-172, left out there for a leak): fromager-1124,
hermes-webui-330_interface, tox-3904, pyfakefs-1286, pyjanitor-1593, docstub-123, jsonpickle-588, jsonpickle-587,
pyinaturalist-643, fabric-cicd-670, sqlglot-6067, anyio-975, dspy-8607, virtualenv-2940, arta-50, sqlglot-5359.

Environments are rebuilt in this test's own work dir (scratchpad `swe_sonnet`) exactly as Q1 did (base commit from the
existing bare git stores, read only; a fresh one-commit repository; a uv venv with the task's Python and install
command) and validated with `repo_atlas_swe.validate` (the base with the test patch fails some FAIL_TO_PASS, gold +
test patch passes all, at most half of PASS_TO_PASS env-broken), rows in `validate.jsonl`, env-broken PASS_TO_PASS
compared with the first test's. A task that does not validate is dropped and reported, not replaced. Validation
status at the time of this commit: see "Validation" below (written before the first scored run).

## Held fixed from Q1

* Claude Code CLI headless (`claude -p`, stream-json in and out), tools `Bash,Read,Edit,Write` (all allowed),
  `--setting-sources ""`, `--strict-mcp-config`, `--disable-slash-commands`, `--no-session-persistence`,
  `--max-turns 250`, 5400 s timeout, clean environment, no web tools: Q1's `run_one`, called unchanged.
* The first message (`repo_atlas_swe.first_message`, same prompt text); the atlas arm sends the same page files as
  the first test and Q1 (`ra_swe/ctx`, linked read only, not rebuilt) and its index.
* Auto-memory off (`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`); each run's project store listed and deleted before and after.
  Package index blocked for every run (`repo_atlas_swe.agent_env`).
* Each run in its own fresh checkout and venv under `swe_sonnet/runs/<hash>/` (a hash of task, arm and run number
  with its own salt, so never Q1's directory), its own cwd; nothing in Q1's or the port test's directories is written.
* Grading (`exclude_files`, the task's test command, `parse_pytest_log`, `grade_status`, this validation's
  env-broken PASS_TO_PASS left out), patches as bytes, the page-arrival check, turns and tokens summed over every
  result event, Q1's widened leak scan (amendment 1's own-temp-dir rule included).

## What differs from Q1

1. **Model:** `claude-sonnet-5-5` at `--effort high` (Q1: `claude-opus-5-5`, high).
2. **Order:** all 32 first runs shuffled with seed 20261113, then all 32 second runs in another shuffled order (Q1:
   seed 20261112); the first plain and first atlas job of the order run first as the smoke runs, scored and counted.
   At most 2 runs at once.
3. **Cost bound:** before each run the runner reserves the larger of $1.00 (plain) / $2.00 (atlas) and 1.5x the
   dearest run of that arm so far in this test, and does not start a run that could cross the cap with the runs in
   flight.

## Scores and tests (as Q1)

Per run: resolved (all FAIL_TO_PASS and the non-broken PASS_TO_PASS pass), turns, cost (`total_cost_usd`, list
price), seconds, tokens. Per task and arm: resolved runs of 2.

* Per arm: task-runs resolved of 32; tasks resolved in both / one / neither run; run 1 and run 2 separately; median
  turns; mean and median cost per run; median seconds.
* **Sign test over tasks** on resolved runs per task (0-2), atlas vs plain.
* **Exact McNemar** on "resolved in both runs", atlas vs plain.
* **Sign test over tasks on mean turns** (and, descriptively, cost and seconds).
* **Bootstrap 95% interval over tasks** (10,000 resamples, seed 20261113) for atlas minus plain resolved task-runs.
* **Flips** between run 1 and run 2 per arm; atlas vs plain within run 1 and within run 2.
* Cost per run and seconds per arm.
* **Shown beside, never pooled:** Q1's Opus rows (`results/repo_atlas_swe_rerun/runs.jsonl`, read at analysis time;
  it may still be gaining its last runs) and the plan-quality test's Sonnet-medium-with-atlas-brief rows
  (`results/plan_quality/work.jsonl`, arm `sonnet_atlas`, 8 of 16, one run each).

## Predictions (from the design, unchanged)

* **S1:** the gate passes without re-rendering. *(Settled above: PASS.)*
* **S2:** Sonnet with the atlas solves no more task-runs than Sonnet plain (difference under 3 of 32).
* **S3:** Sonnet with the atlas stays below Opus plain (Q1's rows, on the tasks complete in both).
* **S4:** the atlas multiplies Sonnet's cost per run by at least 3 (mean cost per run, atlas over plain).

## Spend and stop rules

* **Hard cap $20 at list price** (the CLI's `total_cost_usd`), gate included ($0.63). Estimate ~$15-19 (Q1's Opus
  costs at Sonnet's half price: plain ~$0.13-0.25, atlas ~$0.45-0.85 per run, more if Sonnet takes more turns).
  Every run's cost goes to `spend.jsonl` before it is graded.
* If the cap would cut the design, the order already finishes every first run before any second run, so a truncated
  test is still one full replication; the paired tests then run on the tasks with both runs of both arms, and run 1
  is reported on all 16.
* **Usage limit:** a run that ends in a usage-limit error is not recorded (its cost is booked); the runner stops
  launching, the rows so far are committed and pushed, and what remains is reported. The runner is resumable
  (finished (task, arm, run) rows are skipped): `python3 experiments/repo_atlas_swe_sonnet.py run --cap 20`.
* Refusals are reported, not worked around; prompts are not reworded. Tests are never edited or skipped.
* A run with no result event is retried once (as Q1).
* **Leaks:** every run's Bash commands and file paths are scanned with Q1's pattern and every hit read by hand; a run
  that read a release, upstream copy, old patch or transcript of the fix is listed and its task left out of the
  paired tests (both arms). Transcripts are kept until the leak and memory checks are done.
* `python3 experiments/repo_atlas_swe_sonnet.py analyse` is standalone and runs on whatever rows exist
  (`summary.md`).

## Validation

**All 16 tasks validate** (done before this commit, in `swe_sonnet/validate/`, removed after each task), and on
every task the set of env-broken PASS_TO_PASS is identical to the first test's (tox-3904 1, jsonpickle-587/588 2
each, dspy-8607 9, others 0). Installs took 1-8 s, the two test runs together 2-253 s (anyio-975, dspy-8607 and
virtualenv-2940 about 4 minutes, run beside Q1's and the port test's agents), each under the 300 s test timeout.

## Results
