# Repo atlas on hard, fresh coding tasks: does it change what Claude Code solves?

*Pre-registered 2026-11-12, before the first agent run. Code: `experiments/repo_atlas_swe.py` (tests
`tests/test_repo_atlas_swe.py`). Plan and pilot: [docs/REPO_ATLAS_PLAN.md](../../docs/REPO_ATLAS_PLAN.md),
[results/repo_atlas/README.md](../repo_atlas/README.md). Results go below the "Results" heading. Nothing above it
changes after the first agent run, except dated amendments.*

## Question

Claude Code solves a real issue in a Python repository. Does it solve more, or the same with fewer turns, when its
first message also holds the whole repository as page images (the "atlas")? The yesterday pilot could not tell:
its public SWE-bench tasks were located from a bare file list. So this test uses tasks collected recently and
filters out the ones a closed-book call already locates. It measures solving, graded by each task's own tests.

## What the benchmarks hold (inspected 2026-11-12)

| Dataset | Tasks | Created | Notes |
|---|---|---|---|
| `nebius/SWE-rebench-leaderboard` (test = union of the monthly splits 2025_01 .. 2026_03) | 860, 413 repos, all Python | 2025-01-03 .. **2026-05-12** | install command, Python version and test command per task; parser `parse_log_pytest` for all |
| `nebius/SWE-rebench-V2` | 32,079, many languages | 2014-11 .. 2025-10 | older |
| `SWE-bench-Live/SWE-bench-Live` (full, test, lite, verified) | 1,000-1,888 | 2021-07 .. 2025-09 | older |

**No task from 2026-07 or later exists in either benchmark**, so the brief's fallback applies: the newest
available, SWE-rebench leaderboard, newest first. The newest tasks (2026-03 .. 2026-05) are shortly before the
model's stated training cutoff (June 2026). The closed-book screen is the guard against memorised tasks.

## Selection rule (fixed before any agent run)

1. Python tasks of the SWE-rebench leaderboard, newest first (`created_at`); one task per (base commit, patch)
   (pyfakefs-1286 appears under two repository names).
2. **Gold patch changes at least 3 source files** (`repo_atlas_eval.gold_files`: tests, docs, changelogs, lock
   files left out).
3. The install command must run here: candidates whose install needs `apt-get`, `conda`, `uv sync`/`uv run`
   (a uv-managed project environment), `docker` or `sudo` are skipped without trying.
4. The atlas at the base commit fits: at most 150k image tokens (`repo_atlas.stats`, 88.2 px²/char, tests left out).
5. Screen: one closed-book call (issue + the atlas's file list, the pilot's `tree` prompt, Opus 5.5 high effort).
   Dropped when its top 5 files include every gold file that exists at the base commit.
6. Local validation without Docker (about 10 minutes at most: install 480 s, each test run 300 s): a uv venv with
   the task's Python, the task's install command (`pip install` goes through `uv pip` and falls back to pip);
   the base with the test patch must fail some FAIL_TO_PASS; gold + test patch must pass all FAIL_TO_PASS; at most
   half of PASS_TO_PASS may fail with the gold patch (those are left out of grading for every arm).
7. Stop at 24 accepted or 80 candidates tried.

Steps 4-6 ran in the order fit, screen, validate (the screen is cheaper than validating). That does not change
which tasks pass.

### Funnel

| Step | Count |
|---|---|
| Candidates tried (newest first, >= 3 gold source files; 105 exist) | 80 (created 2025-07-04 .. 2026-05-12) |
| Install needs apt / conda / uv sync (skipped) | 17 |
| Atlas over 150k image tokens | 26 |
| Screened | 37 |
| Dropped by the screen (closed-book named every gold file) | 12 of 37 |
| Validated | 25 |
| Invalid: gold fails FAIL_TO_PASS here | 6 (pyrealm-593, universal_pathlib-529, pdfly-219, basic-memory-341, a2a-python-472, cubed-799) |
| Invalid: base already passes FAIL_TO_PASS | 1 (sqlglot-6174) |
| Invalid: install over 480 s | 1 (pgmpy-2271) |
| **Accepted** | **17** |

Two screen calls hit the subscription's usage limit (an error, not a result). They were retried after the reset,
and both tasks were accepted. Fewer than 24 tasks passed, and more than 12 have at least 3 gold files, so the rule
was not relaxed to 2 files. Rows: `funnel.jsonl` (each candidate, its stage and the screen's answer),
`validate.jsonl`, `selection.json`.

### Tasks (run order = newest first)

| # | Task | Created | Gold source files | FAIL_TO_PASS | Files in atlas | Atlas pages / image tokens | Map chars |
|---|---|---|---|---|---|---|---|
| 1 | python-wheel-build__fromager-1124 | 2026-05-07 | 3 | 7 | 182 | 18 / 83,559 | 60,000 |
| 2 | celestoai__smolvm-172 | 2026-04-18 | 6 | 8 | 92 | 20 / 90,390 | 57,328 |
| 3 | nesquena__hermes-webui-330_interface | 2026-04-12 | 3 | 4 | 42 | 20 / 91,080 | 20,404 |
| 4 | tox-dev__tox-3904 | 2026-03-26 | 3 | 2 | 126 | 14 / 64,584 | 75,512 |
| 5 | pytest-dev__pyfakefs-1286 | 2026-02-22 | 6 | 28 | 29 | 10 / 43,953 | 40,028 |
| 6 | pyjanitor-devs__pyjanitor-1593 | 2026-02-11 | 5 | 3 | 161 | 28 / 131,169 | 70,805 |
| 7 | scientific-python__docstub-123 | 2026-01-14 | 7 | 6 | 41 | 5 / 19,803 | 23,035 |
| 8 | jsonpickle__jsonpickle-588 | 2026-01-12 | 3 | 1 | 44 | 6 / 23,598 | 19,251 |
| 9 | jsonpickle__jsonpickle-587 | 2026-01-11 | 4 | 2 | 44 | 5 / 23,460 | 19,191 |
| 10 | niconoe__pyinaturalist-643 | 2026-01-07 | 3 | 1 | 70 | 9 / 41,538 | 33,146 |
| 11 | microsoft__fabric-cicd-670 | 2025-12-07 | 4 | 2 | 104 | 9 / 41,607 | 22,443 |
| 12 | tobymao__sqlglot-6067 | 2025-10-09 | 3 | 1 | 94 | 31 / 144,555 | 114,033 |
| 13 | agronholm__anyio-975 | 2025-09-14 | 5 | 10 | 42 | 10 / 44,712 | 78,209 |
| 14 | stanfordnlp__dspy-8607 | 2025-08-07 | 4 | 2 | 124 | 18 / 80,523 | 66,146 |
| 15 | pypa__virtualenv-2940 | 2025-08-05 | 3 | 2 | 89 | 7 / 31,464 | 32,218 |
| 16 | MAIF__arta-50 | 2025-07-21 | 4 | 2 | 15 | 2 / 7,521 | 3,975 |
| 17 | tobymao__sqlglot-5359 | 2025-07-07 | 3 | 1 | 92 | 28 / 129,168 | 105,408 |

The pilot is tasks 1-4.

## Checkouts and leakage

The base commit alone is fetched (`git fetch --depth 1`). Each run gets `git archive` of it in a new repository
with one commit ("base") plus one commit with what the install wrote ("env"): no history, and no object after the
base. Runs live in neutral directories (a hash, not the arm's name). The agent never sees the gold patch, the
test patch or the dataset. Checked per run: Bash commands that reach the web, git history, a release of the
package, the dataset or the scratch directories (`leak_suspects`), and Read/Edit/Write paths outside the checkout.

## Arms

Same for all: Claude Code headless (`claude -p`, CLI 2.1.322), `claude-opus-5-5`, `--effort high`, tools
`Bash,Read,Edit,Write` (all allowed), `--max-turns 250`, 5400 s timeout, clean environment (`claude_eval.build_env`),
no settings sources, no MCP, no slash commands, the first message sent as stream-json. The task prompt is the one
from `experiments/readhook/swe/run_swe.py`, made generic: the repository name, the venv path and how to run
pytest, "do not modify the existing tests", reply DONE, then the issue (`problem_statement`).

1. `plain`: the task prompt only.
2. `map`: a repo map (path, then every class/def signature line, from `repo_atlas_eval.build_map`) of the atlas's
   file set, then the task prompt.
3. `atlas`: the atlas pages as images (1932 px wide, base64), then a note on the minified format, the sentence
   "The page images show the repository's source at the start of the task; read them for orientation and edit
   from the files, not from the pages.", `index.txt`, then the task prompt.

Checked on every atlas run: the first API call's input tokens are at least 97% of the image-token formula (pages
arrived whole, not downscaled), else the run is marked invalid.

## Grading and scores

The agent's diff against the "env" commit, minus files the test patch replaces, is applied to a reset checkout.
Then the test patch is applied and the task's test command is run, parsed like SWE-bench's `parse_log_pytest`.
**Resolved** = every FAIL_TO_PASS and every PASS_TO_PASS that passes with the gold patch here.

Per task and arm: resolved; share of gold source files the patch touches; turns; cost (`total_cost_usd`, list
price); seconds; input and output tokens. Analysis: paired counts with exact McNemar for resolved; paired medians
with sign tests for turns and cost.

**What this size can show:** with 8-17 paired tasks, only large differences. McNemar needs at least 6 discordant
tasks, all one way, for p < 0.05. Turns and cost are paired per task, so a consistent shift shows sooner.

## Predictions (the coordinator's, kept as given)

* **H1** resolved tasks: atlas and plain differ by at most 2 tasks (this size cannot show a smaller difference).
* **H2** the atlas arm uses fewer turns than plain (median, paired).
* **H3** the atlas arm costs more per task than plain.
* **H4** map and plain differ by at most 1 resolved task.
* **H5** the atlas arm's patches touch the gold files at least as often as plain's.

## Cost estimate and cap

Hard cap **$35** at list price, over every call (screen calls included: $0.88 spent on 37 screens). Assumed list
prices: cache write $5/M, cache read about $0.2/M (fitted to earlier rows), output $20/M. Earlier Claude Code
runs on SWE-bench Verified cost $0.12 per task (median 6.5 turns). These tasks are harder: about $0.5-1.5 per plain
run is assumed. The map adds about 15k tokens (~$0.2 per run at ~30 turns). The atlas adds 7-145k image tokens:
one cache write (~$0.3 for 60k) plus a cache read per turn (~$0.4 over 30 turns), ~$0.7 per run. So about
**$2.5-4.5 per task** for the three arms; at $35 that is **about 8-13 tasks, not all 17**. The runner books a bound
before each task ($4 until measured, then 1.5x the dearest task so far) and does not start a task that could pass the
cap; a started task is finished in all three arms. At most 2 agent runs at once (shared machine).

## Results

*(added after the runs)*
