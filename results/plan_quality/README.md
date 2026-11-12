# Plan quality: can an orchestrator that reads the repo atlas once write good briefs for worker agents?

*Pre-registered 2026-11-12, before the first scored call (selection screens and a model-id smoke test came before
it; $0.50). Design: [docs/PLAN_QUALITY_TEST.md](../../docs/PLAN_QUALITY_TEST.md), kept as written; this file fixes
what the design leaves open. Code: `experiments/plan_quality.py` (tests `tests/test_plan_quality.py`). Results go
below the "Results" heading. Nothing above it changes after the first scored call, except dated amendments written
before the runs they affect. Run by an Opus 5.5 subagent; hard cap **$45** at list price over every call.*

## Question

A backlog of tickets against one repository. One orchestrator reads the repository once and writes a brief per
ticket; workers get the brief, not the repository view. Are briefs written from the atlas (the repository as page
images, `experiments/repo_atlas.py`) as good as scouting with tools and better than briefs from a file list? Do they
invent paths or symbols? Can the planner size a ticket and pick the right worker (model and reasoning level)? Does a
brief help the worker?

## Tickets

**Source:** SWE-rebench leaderboard, the fresh-task test's rules ([results/repo_atlas_swe/README.md](../repo_atlas_swe/README.md)):
Python tasks newest first, one per (base commit, patch), fix changes at least 3 source files, atlas at the base
commit at most 150k image tokens, and the closed-book screen (issue + file list, Opus 5.5 high effort) does not
name every fix file in its top 5. The install rule (no apt / conda / uv sync) is **not** applied here: layers 1-2
run nothing. The fresh-task test's funnel rows are reused for its 80 candidates (its install-skipped candidates were
fitted and screened now); then the candidate list continued newest first by the same rule until 40 tickets passed.

### Funnel (`funnel.jsonl`)

| Step | Count |
|---|---|
| Candidates tried (newest first, >= 3 fix source files; 105 exist), 2025-05-03 .. 2026-05-12 | 91 |
| Atlas over 150k image tokens | 33 (26 from the fresh-task funnel, 7 now) |
| Screened | 58 (37 earlier, 21 now; the 21 new screens cost $0.49) |
| Dropped by the screen (file list alone named every fix file) | 18 (12 earlier, 6 now) |
| **Passed: layer-1 tickets** | **40** (stopped at 40; 14 candidates left untried) |

### Ticket list (layer-1 order = run order)

Layer-1 order: the 16 layer-3 tickets first, in the fresh-task test's order (its compared tasks; smolvm-172 is left
out of layer 3 for its leak), then the other 24 newest first. "Fix files" = source files of the fix at the base
commit (+ files the fix creates); functions and lines over the fix's source files (`fix_size`).

| # | Ticket | Created | Layer 3 | Fix files | Fix functions | Fix lines | Files in atlas | Pages / image tokens | Map chars |
|---|---|---|---|---|---|---|---|---|---|
| 1 | python-wheel-build__fromager-1124 | 2026-05-07 | yes | 3 | 5 | 25 | 182 | 18 / 83,559 | 60,212 |
| 2 | nesquena__hermes-webui-330_interface | 2026-04-12 | yes | 3 | 2 | 27 | 42 | 20 / 91,080 | 20,609 |
| 3 | tox-dev__tox-3904 | 2026-03-26 | yes | 3 | 1 | 24 | 126 | 14 / 64,584 | 75,708 |
| 4 | pytest-dev__pyfakefs-1286 | 2026-02-22 | yes | 6 | 23 | 150 | 29 | 10 / 43,953 | 40,231 |
| 5 | pyjanitor-devs__pyjanitor-1593 | 2026-02-11 | yes | 5 | 12 | 122 | 161 | 28 / 131,169 | 71,014 |
| 6 | scientific-python__docstub-123 | 2026-01-14 | yes | 7 | 7 | 59 | 41 | 5 / 19,803 | 23,244 |
| 7 | jsonpickle__jsonpickle-588 | 2026-01-12 | yes | 3 | 2 | 22 | 44 | 6 / 23,598 | 19,456 |
| 8 | jsonpickle__jsonpickle-587 | 2026-01-11 | yes | 4 | 4 | 39 | 44 | 5 / 23,460 | 19,396 |
| 9 | niconoe__pyinaturalist-643 | 2026-01-07 | yes | 3 | 5 | 22 | 70 | 9 / 41,538 | 33,351 |
| 10 | microsoft__fabric-cicd-670 | 2025-12-07 | yes | 4 | 6 | 22 | 104 | 9 / 41,607 | 22,649 |
| 11 | tobymao__sqlglot-6067 | 2025-10-09 | yes | 3 | 3 | 5 | 94 | 31 / 144,555 | 114,232 |
| 12 | agronholm__anyio-975 | 2025-09-14 | yes | 5 | 4 | 37 | 42 | 10 / 44,712 | 78,408 |
| 13 | stanfordnlp__dspy-8607 | 2025-08-07 | yes | 4 | 5 | 36 | 124 | 18 / 80,523 | 66,347 |
| 14 | pypa__virtualenv-2940 | 2025-08-05 | yes | 3 | 3 | 17 | 89 | 7 / 31,464 | 32,417 |
| 15 | MAIF__arta-50 | 2025-07-21 | yes | 4 | 2 | 10 | 15 | 2 / 7,521 | 4,168 |
| 16 | tobymao__sqlglot-5359 | 2025-07-07 | yes | 3 | 6 | 32 | 92 | 28 / 129,168 | 105,607 |
| 17 | celestoai__smolvm-172 | 2026-04-18 |  | 6 | 8 | 237 | 92 | 20 / 90,390 | 57,528 |
| 18 | sissbruecker__linkding-1291 | 2026-01-25 |  | 6 | 3 | 41 | 238 | 14 / 64,239 | 52,786 |
| 19 | imperialcollegelondon__pyrealm-593 | 2026-01-19 |  | 3+1 | 25 | 668 | 110 | 20 / 92,805 | 44,861 |
| 20 | fsspec__universal_pathlib-529 | 2026-01-04 |  | 5 | 12 | 41 | 45 | 9 / 42,021 | 50,688 |
| 21 | py-pdf__pdfly-219 | 2025-12-19 |  | 3 | 2 | 33 | 36 | 4 / 14,145 | 7,356 |
| 22 | PyPSA__PyPSA-1498 | 2025-12-17 |  | 3 | 3 | 38 | 91 | 28 / 129,651 | 85,246 |
| 23 | plone__cookieplone-templates-322 | 2025-12-05 |  | 3+1 | 2 | 7 | 277 | 11 / 51,336 | 29,323 |
| 24 | tobymao__sqlglot-6174 | 2025-10-23 |  | 4 | 6 | 37 | 94 | 31 / 144,762 | 114,800 |
| 25 | basicmachines-co__basic-memory-341 | 2025-10-08 |  | 3 | 3 | 26 | 170 | 27 / 125,511 | 63,464 |
| 26 | a2aproject__a2a-python-472 | 2025-09-13 |  | 3 | 3 | 56 | 95 | 13 / 56,442 | 69,128 |
| 27 | cubed-dev__cubed-799 | 2025-09-09 |  | 4 | 10 | 175 | 78 | 10 / 44,022 | 42,553 |
| 28 | Spenhouet__confluence-markdown-exporter-92 | 2025-09-05 |  | 3 | 13 | 57 | 14 | 3 / 11,523 | 11,300 |
| 29 | stellarmesh__stellarmesh-51 | 2025-09-05 |  | 4 | 15 | 311 | 8 | 2 / 5,796 | 5,300 |
| 30 | pgmpy__pgmpy-2271 | 2025-08-06 |  | 2+1 | 1 | 30 | 113 | 29 / 135,378 | 52,589 |
| 31 | beeware__briefcase-2401 | 2025-07-19 |  | 3 | 1 | 10 | 87 | 19 / 88,320 | 62,077 |
| 32 | libp2p__py-libp2p-750 | 2025-07-07 |  | 3 | 2 | 15 | 167 | 16 / 72,588 | 94,175 |
| 33 | tobymao__sqlglot-5352 | 2025-07-04 |  | 8 | 13 | 85 | 92 | 28 / 128,892 | 105,309 |
| 34 | lincc-frameworks__nested-pandas-303 | 2025-07-03 |  | 5 | 14 | 113 | 25 | 5 / 21,114 | 14,100 |
| 35 | python-wheel-build__fromager-653 | 2025-07-01 |  | 8 | 15 | 111 | 123 | 12 / 52,716 | 41,625 |
| 36 | qutech__qupulse-900 | 2025-07-01 |  | 3 | 6 | 39 | 74 | 19 / 88,044 | 118,306 |
| 37 | feder-observatory__stellarphot-519 | 2025-06-09 |  | 3 | 7 | 143 | 62 | 12 / 54,855 | 21,807 |
| 38 | tobymao__sqlglot-5189 | 2025-06-08 |  | 4 | 4 | 175 | 89 | 27 / 125,442 | 102,205 |
| 39 | ultrabug__py3status-2286 | 2025-05-15 |  | 3 | 1 | 26 | 146 | 24 / 111,642 | 44,722 |
| 40 | pgmpy__pgmpy-2034 | 2025-05-03 |  | 3 | 2 | 19 | 111 | 27 / 122,751 | 44,597 |

Every fix has at least 3 source files, so by the design's own size rule every real fix is M (3 files) or L (4+).

### Layer 2: not evaluable

Groups (rule fixed here): tickets of one repository, oldest first, in consecutive groups of 3, a remainder of 1-2
joining the last group; planned on the group's earliest base commit, provided every ticket's fix files exist there.
Only one repository has 3 or more tickets: sqlglot (5189, 5352, 5359, 6067, 6174), one usable group of 5. The
design needs at least three groups, so **layer 2 is not evaluable** and is not run; its budget goes to layer 1.
Prediction 4 is reported as not evaluable.

## Planners

All four: Opus 5.5 (`claude-opus-5-5`), `--effort high`, the same system prompt, ticket block and instruction. The
ticket id the planner sees is neutral (`T1`), not the instance id. No planner sees the fix, the test patch, the
dataset or any later history.

* `tree`: the atlas's file list (`Repository <repo>: its <n> source files (tests, docs and data files left out).`
  then one path per line), one call.
* `map`: the repo map of the same files (`repo_atlas_eval.build_map`: path, then every class / def signature line),
  one call.
* `atlas`: the atlas pages as images (1932 px wide, base64), then the note on the minified format and `index.txt`,
  one call. Checked on every call: input tokens at least 97% of the image-token formula, else marked invalid.
* `scout`: Claude Code headless (`claude -p`, CLI 2.1.322) in an export of the base commit **without git** (`git
  archive`), in a neutral directory (a hash, no ticket id or arm name). Tools `Read,Grep,Glob` only (no Bash, no web
  tools), `--permission-prompts none` (anything that would need approval is denied; smoke-tested: a Read outside the
  directory was denied), `--setting-sources ""`, no MCP, no slash commands, no session persistence, at most 40 turns.
  No package index in its environment. **Auto-memory: off** (`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`), and the run's
  project store under `~/.claude/projects/` is deleted before and after each run; the row records whether anything
  was there. The system prompt below is appended to Claude Code's own. The reply is the brief.

System prompt: `You are the planning lead for a team of coding agents. You look at a repository and write a short
brief per ticket for the worker agent that will implement it. Answer only in the format asked for.`

The user message (context first; for `scout` the context is the line `The current directory is a checkout of the
<repo> repository. Use your tools (Read, Grep, Glob) to look at the code as much as you need; do not change
anything. Then reply with the brief only.`):

```
<CONTEXT>
<ticket id="T1">
<issue text>
</ticket>

Write a brief for the worker agent that will implement the ticket below. The worker gets the ticket and your brief, nothing else; it reads the files it edits and may depart from your brief where the code disagrees.

Use exactly this format, one field per line; repeat the FILES: line once per file:

TICKET: <id>
SUMMARY: <one sentence>
FILES: <path> : <what changes>            (at most 6, most important first)
FUNCTIONS: <names>
APPROACH: <at most 3 sentences>
RISKS: <at most 2>
COLLIDES_WITH: <ticket ids that touch the same files, or none>
ORDER: <tickets that should land first, or none>
SIZE: S | M | L
WORKER: <haiku | sonnet | opus> / <low | medium | high>
CONFIDENCE: low | medium | high

Rules:
* Paths exactly as they appear in the repository. A file or function that does not exist yet and has to be created: write " (new)" after it.
* FUNCTIONS: comma-separated, as function or Class.method.
* SIZE: S = one file, a local change, no design choice; M = two or three files, or one decision that is not obvious; L = four or more files, or a change across modules.
* WORKER: the model and reasoning level that should implement the ticket. Default per size: S = haiku / low, M = sonnet / medium, L = opus / high. Choose the cheapest worker you expect to resolve the ticket; if you choose other than the default for the size, say why in RISKS.
* This is the only ticket in this batch: COLLIDES_WITH: none and ORDER: none.
```

The brief format is the design's, unchanged. Added to fix open details: "(new)" marks for files and functions to be
created (so a planned new file is not counted as invented), "repeat the FILES: line", and "choose the cheapest
worker you expect to resolve the ticket" (the right-sizing goal). A brief that breaks the format is scored as parsed
and its problems are counted; nothing is re-asked.

## Layer 1 scores (per brief; `parse_brief`, `file_scores`, `func_score`, `invented`)

* **Fix files named:** share of the fix's source files that exist at the base commit named on a FILES line (all
  FILES lines, at most 6). Files the fix creates are reported apart.
* **Named files not in the fix:** named source files that are not among the fix's source files (incl. created ones),
  as a share of named source files. Named tests and docs are counted apart, not as wrong.
* **Fix functions named:** share of the fix's changed functions (`repo_atlas_eval.gold_functions`) that occur as a
  word in FUNCTIONS, the FILES descriptions or APPROACH.
* **Things that do not exist:** named paths that are neither a file nor a directory in `git ls-files` at the base
  commit, plus named FUNCTIONS whose last component (`Class.method` -> `method`) occurs nowhere as an identifier in
  the repository's tracked files; items marked "(new)" (or a FILES line saying it creates a new file) are left out.
  Reported as a share of everything named (paths + functions).
* **Approach, judged blind:** one call per brief, Opus 5.5 at `--effort low`, calls in a shuffled order (seed
  20261112). The judge sees the ticket, the real fix (source files only, cut at 24,000 characters with a note) and
  the brief's SUMMARY, FILES, FUNCTIONS, APPROACH and RISKS lines (no ticket id, size, worker or confidence; no
  planner name anywhere). Rubric (system prompt, verbatim):

```
You grade a brief that a planner wrote for a coding ticket, against the change that actually fixed the ticket. The brief was written before the fix existed. It does not have to match the fix line by line: a different approach that would also work counts as right. Judge three things:
CAUSE: does the brief identify the actual cause of the problem (for a feature request: what has to be added)?
PLACE: does it point to the right place in the code: the files and functions the fix changes, or others where the change would work as well?
OUTCOME: would a competent engineer who follows the brief arrive at the behaviour the ticket asks for?
Answer each with yes, partly or no. Reply in exactly four lines:
CAUSE: yes|partly|no
PLACE: yes|partly|no
OUTCOME: yes|partly|no
REASON: <one sentence>
```

  Score: yes 2, partly 1, no 0, summed (0-6); also the share with OUTCOME yes.
* **Sizing:** SIZE (S 1, M 2, L 3) against the real fix's size. Primary: Spearman rank correlation with a composite
  real size, the mean of the ranks of fix files, fix functions and fix lines changed. Also reported: the correlation
  with each of the three, and a confusion table against the design's own rule applied to the fix's files (S one, M
  two or three, L four or more).
* Cost (list price, `total_cost_usd`), seconds, turns per brief.

Paired comparisons over tickets with all four briefs: mean difference and an exact sign test.

## Layer 3: workers

Tickets: the fresh-task test's 16 compared tasks (rows 1-16 above), in its order. Environments are rebuilt per run
from its records (`repo_atlas_swe.fresh_repo`, `make_env`: uv venv, the task's own install command), and grading is
its own: the agent's diff (minus files the test patch replaces) on a reset checkout, then the test patch and the
task's test command; **resolved** = every FAIL_TO_PASS and every PASS_TO_PASS that passed with the gold patch in its
validation (`validate.jsonl`, `env_broken_p2p` left out).

Worker: Claude Code headless, tools `Bash,Read,Edit,Write` (allowed), `--max-turns 250`, 5400 s, clean
environment, no settings sources, no MCP, no slash commands, no session persistence, one-commit checkout with no
history in a neutral directory, **no package index** (`repo_atlas_swe.agent_env`), **auto-memory off** and its
store deleted before and after each run (as for the scout). The first message is the fresh-task test's task prompt
(repository, venv, "do not modify the existing tests", reply DONE, the issue), then:

```
A planner who looked at the repository wrote this brief for the issue. Use it as a starting point; read the files before you edit them, and depart from the brief where the code disagrees.

<brief>
<the layer-1 brief's SUMMARY, FILES, FUNCTIONS, APPROACH and RISKS lines>
</brief>
```

The worker gets no atlas and no pages.

| arm | model | effort | brief |
|---|---|---|---|
| `opus_atlas` | claude-opus-5-5 | high | atlas planner's |
| `opus_tree` | claude-opus-5-5 | high | tree planner's |
| `sonnet_atlas` | claude-sonnet-5-5 | medium | atlas planner's |
| `haiku_atlas` | claude-haiku-5-5 | low | atlas planner's |
| no brief | claude-opus-5-5 | high | none: the fresh-task test's `plain` rows (`results/repo_atlas_swe/runs.jsonl`); its first four tasks ran before its package-index block |

Per run: resolved, turns (summed over all result events), cost, seconds, tokens, share of fix files touched, a scan
of Bash commands for the web, git history, package releases and this repository's result files, and Read/Edit/Write
paths outside the checkout (read by hand). A run where the agent read a release or upstream copy of the fix is
left out of the comparisons and listed.

**Right-sizing:** the measured tier of a ticket is the cheapest of Haiku low, Sonnet medium, Opus high (all with the
atlas brief) that resolves it, "none" if none does. The atlas planner's WORKER model is scored against it: right,
too small, too large, or unresolvable (a planner's effort that differs from the tier's default is reported, and the
run of that model's tier is used). Policies on tickets resolved and total worker cost: always Opus high, always
Haiku low (the cheapest tier; always Sonnet medium also shown), the planner's choice. Also SIZE against the fix's
size as in layer 1, on these tickets.

## Predictions (the design's, unchanged; operational reading in brackets)

1. The `atlas` brief names more of the fix's files than the `tree` brief (paired, at least 5 points). [mean share of
   fix files named, atlas minus tree >= 5 points over tickets with both briefs]
2. Things that do not exist: at most 2% of what the `atlas` brief names, and not more than 2 points above `scout`.
3. `scout` names at least as many of the fix's files as `atlas` (within 5 points either way). [PASS if scout's mean
   share is at least atlas's minus 5 points]
4. Collisions: the `atlas` planner finds more true colliding pairs than `tree` at no more false ones. [not evaluable:
   layer 2 has one group]
5. Sizing: the `atlas` planner's SIZE rises with the real fix's size (rank correlation at least 0.3). [primary:
   composite real size]
6. Right-sizing: following the planner's WORKER choice resolves within 1 ticket of always-Opus-high at no more
   than 70% of its cost.
7. A worker with the `atlas` brief uses fewer turns than one with no brief (paired median) and resolves within 2
   tickets of it. [median turns of opus_atlas below the no-brief rows' median, and resolved counts within 2]

## Cost estimate, order and stop rules

Estimate before any call (list price, cache write $5/M, output $20/M): layer 1 on all 40 tickets: tree $2.9, map
$6.0, atlas $17.3, scout ~$16 (assumed $0.40 a run), judge ~$8 (assumed $0.05 a call): **~$50**, above the $45 cap
by itself. Layer 3: ~$0.8-1.0 per ticket for the four worker runs (the fresh-task test's Opus runs cost $0.26 on
average without a brief), ~$13-16 for 16 tickets.

So "layer 1 in full, then layer 3" cannot both fit, and layer 3 carries predictions 6 and 7. **Order (fixed
here):**

1. Layer-1 pilot: tickets 1-6 (all four planners, then their 24 judge calls). Re-estimate from the measured costs
   (amendment if anything changes).
2. Layer 1 on tickets 7-16 (the rest of the layer-3 tickets).
3. Layer 3, ticket by ticket in order, all four worker runs of a ticket or none. A ticket is started only when
   spent + its bound stays under the cap (bound $3.00 until measured, then 1.5x the dearest ticket so far).
4. Layer 1 on tickets 17-40 in order, one whole ticket at a time (four planners + four judge calls), started only
   when spent + 1.2x its projection (`ticket_projection`, from measured costs) stays under the cap.

Each call books a bound against the cap before it starts and nothing is launched past the cap. At most 2 calls or
agent runs at once. **Usage limit:** a call or run that fails with a usage-limit error is not recorded, nothing more
is launched, the work done so far is committed and the reset time reported; no retry loop. A refusal or classifier
cut-off is recorded as the result. Rows are committed and pushed as they arrive (`WIP:` commits); every step skips
finished rows, so the runner resumes.

**What the sample can show:** with 16-40 tickets, paired file shares can show differences of a few points; judge
scores and sizing correlations only moderate effects; layer 3 (16 tickets, one run per arm) only large differences
(McNemar needs about 6 discordant tickets all one way for p < 0.05).

## Results
