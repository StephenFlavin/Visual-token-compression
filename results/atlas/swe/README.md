# Repo atlas on hard, fresh coding tasks: does it change what Claude Code solves?

*Pre-registered 2026-11-12, before the first agent run. Code: `experiments/atlas/repo_atlas_swe.py` (tests
`tests/test_repo_atlas_swe.py`). Plan and pilot: [docs/claude/REPO_ATLAS_PLAN.md](../../../docs/claude/REPO_ATLAS_PLAN.md),
[results/atlas/pilot/README.md](../pilot/README.md). Results go below the "Results" heading. Nothing above it
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

## Amendment 1 (2026-11-12, after the pilot's 4 tasks, before task 5)

* **Leak found in the pilot.** On celestoai__smolvm-172 the `map` arm ran `pip download smolvm==0.0.36` (a PyPI
  release made after the base commit), unzipped it, read the released `smolvm/host/setup.py` and
  `smolvm/cli/main.py` (which already hold the fix's `for_bake`, `skip_kvm_check` and `firecracker_version`), then
  wrote its change. That run resolved the task; plain and atlas did not.
* **Change, for every arm from task 5 on:** the agent's environment has no package index (`PIP_NO_INDEX=1`,
  `UV_OFFLINE=1`, index URLs pointed at a closed local port). The agent can still reach the web through other tools,
  so every run's Bash commands are still scanned (now also for any `pip download` / `pip index` and `pypi.org`), and
  each hit is read by hand.
* **Analysis rule:** a task where any arm read a release or an upstream copy of the fix is left out of the paired
  comparisons and listed (`contaminated.json`). smolvm-172 is left out on this rule. The other pilot runs' flagged
  commands were read and are harmless: the run's own venv path, `git log` of the one-commit checkout, and
  listing the run's own installed package.
* Two concurrent agent runs, not three (the coordinator's limit; the machine is shared).

## Results (2026-11-12): 17 tasks x 3 arms, all runs complete, $28.45 of the $35 cap

### Verdict

* **The atlas arm solved the most tasks, but this sample cannot show that the atlas causes it.** On the 16 compared
  tasks: atlas 11, map 9, plain 8. Every discordant task went the atlas's way (atlas only 3, plain only 0), but exact
  McNemar p = 0.25. One run per arm cannot separate an effect from run-to-run variation. Three wins in 16 tasks is the
  size of change worth testing with more tasks and repeats, not a result.
* **Turns fall, cost rises.** The atlas run used fewer turns than plain on 11 of the 12 tasks that differ (median 10 vs
  12.5; sign test p = 0.006). It cost more on all 16 tasks: $0.68 against $0.26 per task on average (2.6x). Wall time
  did not change (median 74 s vs 100 s; lower on 7 of 15, p = 1.0).
* **The repo map changed nothing measurable:** 9 vs 8 resolved (map only 1, p = 1.0), the same turns, and $0.49 per task.
* Pages arrived whole in every atlas run (first-call input 106-194% of the image-token formula). There were no
  refusals and no usage-limit rows. One task (smolvm-172) is left out for a leak (amendment 1).

### Per arm (16 compared tasks)

| arm | tasks | resolved | mean share of gold files touched | median turns | mean cost | median seconds | median input tokens (incl. cache reads) | median output tokens | pages dropped |
|---|---|---|---|---|---|---|---|---|---|
| plain | 16 | 8 | 0.58 | 12.5 | $0.26 | 100 | 171,882 | 5,501 | n/a |
| map | 16 | 9 | 0.59 | 12 | $0.49 | 90 | 488,662 | 6,030 | n/a |
| atlas | 16 | **11** | 0.63 | **10** | $0.68 | 74 | 564,516 | 5,038 | 0 |

With smolvm-172 included (17 tasks): plain 8, map 10 (the leaked run), atlas 11.

### Per task

| task | gold files | plain: resolved / gold touched / turns / cost | map | atlas |
|---|---|---|---|---|
| python-wheel-build__fromager-1124 | 3 | no / 1.00 / 13 / $0.27 | no / 1.00 / 12 / $0.48 | no / 1.00 / 12 / $0.83 |
| celestoai__smolvm-172 (left out: leak in map) | 6 | no / 0.83 / 20 / $0.77 | yes / 0.83 / 23 / $0.96 | no / 0.83 / 24 / $1.63 |
| nesquena__hermes-webui-330_interface | 3 | no / 0.33 / 14 / $0.52 | no / 0.33 / 17 / $0.44 | no / 0.33 / 9 / $0.83 |
| tox-dev__tox-3904 | 3 | yes / 0.67 / 18 / $0.50 | yes / 0.67 / 14 / $0.76 | yes / 0.67 / 17 / $1.06 |
| pytest-dev__pyfakefs-1286 | 6 | yes / 1.00 / 16 / $0.32 | yes / 1.00 / 15 / $0.41 | yes / 1.00 / 11 / $0.54 |
| pyjanitor-devs__pyjanitor-1593 | 5 | yes / 0.40 / 14 / $0.50 | yes / 0.40 / 21 / $1.64 | yes / 0.40 / 11 / $1.34 |
| scientific-python__docstub-123 | 7 | no / 0.43 / 7 / $0.17 | no / 0.43 / 9 / $0.22 | no / 0.43 / 7 / $0.25 |
| jsonpickle__jsonpickle-588 | 3 | yes / 1.00 / 12 / $0.29 | yes / 1.00 / 17 / $0.42 | yes / 1.00 / 12 / $0.46 |
| jsonpickle__jsonpickle-587 | 4 | no / 0.25 / 14 / $0.19 | no / 0.25 / 10 / $0.24 | **yes** / 0.25 / 11 / $0.33 |
| niconoe__pyinaturalist-643 | 3 | yes / 0.33 / 6 / $0.10 | yes / 0.33 / 8 / $0.24 | yes / 0.33 / 5 / $0.35 |
| microsoft__fabric-cicd-670 | 4 | no / 0.00 / 11 / $0.18 | no / 0.00 / 12 / $0.23 | no / 0.00 / 10 / $0.46 |
| tobymao__sqlglot-6067 | 3 | yes / 1.00 / 7 / $0.12 | yes / 1.00 / 14 / $0.62 | yes / 1.00 / 7 / $1.03 |
| agronholm__anyio-975 | 5 | yes / 0.60 / 12 / $0.26 | yes / 0.80 / 12 / $0.51 | yes / 0.80 / 9 / $0.83 |
| stanfordnlp__dspy-8607 | 4 | no / 0.75 / 13 / $0.27 | no / 0.75 / 18 / $0.63 | **yes** / 1.00 / 15 / $0.95 |
| pypa__virtualenv-2940 | 3 | no / 0.33 / 7 / $0.17 | no / 0.33 / 11 / $0.26 | no / 0.33 / 7 / $0.34 |
| MAIF__arta-50 | 4 | yes / 0.50 / 9 / $0.11 | yes / 0.50 / 9 / $0.11 | yes / 0.50 / 5 / $0.12 |
| tobymao__sqlglot-5359 | 3 | no / 0.67 / 15 / $0.21 | yes / 0.67 / 11 / $0.57 | yes / 1.00 / 10 / $1.08 |

The three atlas-only wins: on jsonpickle-587 all three arms edited only `pickler.py`, and the atlas run passed the
second FAIL_TO_PASS test. On dspy-8607 and sqlglot-5359 the atlas run also changed one more gold file
(`dspy/utils/dummies.py`, `sqlglot/generator.py`) that the other arms left alone. "Gold files touched" counts files,
not correct edits: many tasks change docs-adjacent or helper files that a different correct fix does not need, so
0.3-0.6 does not mean a wrong fix.

### Paired tests (exact, 16 tasks)

* Resolved: atlas vs plain 3-0 discordant, McNemar p = 0.25; map vs plain 1-0, p = 1.0; atlas vs map 2-0, p = 0.5.
* Turns: atlas lower than plain on 11 of 12 untied tasks (median paired difference -1), p = 0.006; map vs plain lower
  on 5 of 14, p = 0.42.
* Cost: atlas dearer on 16 of 16 (median +$0.29), map on 15 of 16 (median +$0.14); both p ≤ 0.001.
* Input tokens (incl. cache reads): atlas +359k, map +238k per task (median); every task.
* Gold files touched: atlas higher on 3 tasks, lower on 0.

**What 16 tasks can and cannot show:** a consistent shift in turns or cost shows up (each task is its own control).
A difference in solved tasks needs about 6 discordant tasks all one way for p < 0.05; this run had 3. The arms were
run once each. Earlier coding tests here (`results/readhook_swe/`) did not repeat runs either, so run-to-run variation
on these tasks is unknown. It could be as large as the 3-task gap.

### Predictions

* **H1** atlas and plain differ by at most 2 resolved tasks: **FAIL** (3: 11 vs 8). Not significant (p = 0.25).
* **H2** atlas uses fewer turns than plain (median, paired): **PASS** (10 vs 12.5; 11 of 12, p = 0.006).
* **H3** atlas costs more per task than plain: **PASS** ($0.68 vs $0.26; 16 of 16).
* **H4** map and plain differ by at most 1 resolved task: **PASS** (9 vs 8).
* **H5** atlas patches touch the gold files at least as often as plain's: **PASS** (mean share 0.63 vs 0.58; higher on
  3 tasks, never lower).

### Selection funnel

80 candidates (newest first, at least 3 gold source files, 2025-07-04 .. 2026-05-12) -> 17 skipped for an
apt/conda/uv-sync install -> 63 sized, 26 atlases over 150k image tokens -> 37 screened, **12 dropped** because the
closed-book call named every gold file -> 25 validated, 8 invalid (6 gold fails here, 1 base passes, 1 install over
480 s) -> **17 accepted**. All 17 were run.

### Cost

| what | calls | list price |
|---|---|---|
| closed-book screen | 37 | $0.88 |
| plain | 17 | $4.96 |
| map | 17 | $8.73 |
| atlas | 17 | $12.44 |
| map run of pyjanitor-1593 cut by the usage limit (not recorded, re-run) | 1 | $1.45 |
| **total** | | **$28.45** (cap $35) |

The cost estimate ($2.5-4.5 per task, 8-13 tasks) was too high: these tasks took 5-24 turns, so all 17 fit. The atlas
costs one cache write of its pages plus a cache read per turn. On a 64-145k-token atlas that is $0.8-1.1 per run,
about 3-5x a plain run.

### Problems and changes from the brief

1. **Freshness.** No task in either benchmark is from 2026-07 or later. The tasks run from 2025-07 to 2026-05, before
   the model's stated training cutoff (June 2026). The closed-book screen dropped 12 of 37 as located from the file
   list alone. The rest may still be partly known.
2. **Leak (amendment 1).** In the pilot, one map run downloaded a later PyPI release of the package and read the fix.
   From task 5 on, the agent had no package index. The task is left out of the comparison. No later run reached the
   web, a release or git history (Bash commands scanned; hits read by hand: only the run's own paths). The block cost
   something: on fabric-cicd-670 the plain and map runs tried to build a Python 3.13 venv with pip and could not;
   all three arms failed that task.
3. **Background jobs.** One map run (pyjanitor-1593) left the full test suite running in the background. Its
   notifications started two extra short turns after the first result. The runner first kept only the last result
   (turns = 2). `repair` recomputed turns and tokens over all result events from the transcript (21 turns). The cost
   was right (it is cumulative).
4. **Usage limits** stopped the work twice (screen calls during selection, and the pyjanitor map run). Neither was
   counted as a result. Both were redone after the reset; the cut-off run's $1.45 is in the total.
5. **Changes from the brief:** fit, screen, validate in that order (same set); install commands needing apt/conda/uv
   sync skipped untried; no settings sources for the agent (no repository settings or hooks loaded) instead of
   `project,local`; 2 concurrent runs (the coordinator's limit) instead of 3; the leak amendment.

### Files

`runs.jsonl` (one row per task and arm: grade, patch files, turns, cost, tokens, first-call input, leak scan),
`spend.jsonl`, `funnel.jsonl`, `validate.jsonl`, `selection.json`, `contaminated.json`, `summary.md`
(`repo_atlas_swe.py analyse`). Transcripts, patches, venvs and pages stayed in the scratch directory.
