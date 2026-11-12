# Repo atlas, stage 1 pilot: where does the change go?

*Pre-registered 2026-11-12, before any scored call. Plan: [docs/claude/REPO_ATLAS_PLAN.md](../../docs/claude/REPO_ATLAS_PLAN.md).
Code: `experiments/atlas/repo_atlas_eval.py` (tests `tests/test_repo_atlas_eval.py`). Results are added below this
pre-registration once the calls are in; nothing above the "Results" heading changes after the first call, except
dated amendments.*

## Question

Does having a whole repository in context as page images (the "atlas") lead to better decisions about where a
change goes than (a) a repo map of signatures and (b) Claude Code searching the checkout by itself?

This pilot proves the harness, measures the real cost per arm (the agent arm is the uncertain one) and gives a first
look at the scores. **With 8 tasks it cannot decide anything statistically.** The predictions below are read as
directions only.

## Tasks

* Pool: the 8 psf/requests tasks of SWE-bench Verified (`princeton-nlp/SWE-bench_Verified`) and the gin-gonic/gin
  (8), sharkdp/bat (10) and preactjs/preact (15) tasks of SWE-bench Multilingual (`SWE-bench/SWE-bench_Multilingual`).
* **Gold files** = the files in the task's gold `patch`, leaving out tests, docs (`.md`, `.rst`, `.txt`, `.adoc`,
  `docs/`), changelogs, lock files and everything else `repo_atlas.wanted` leaves out (`is_gold`).
* **Selection rule** (`select`): per repository, order the tasks by the number of gold files (most first; ties: lowest
  instance id) and keep the first 2 whose atlas at the base commit is at most 150k image tokens. No task was skipped
  for size. Every requests task in Verified changes one source file, so the two requests tasks are the two lowest ids.

| Task | Gold files scored | Gold files the fix creates (not scored) | Functions/classes touched (approximate) | Atlas pages / image tokens |
|---|---|---|---|---|
| psf__requests-1142 | requests/models.py | | prepare_body | 16 / 71,415 |
| psf__requests-1724 | requests/sessions.py | | request | 17 / 78,315 |
| gin-gonic__gin-1957 | binding/binding.go, context.go | binding/header.go | 9 names | 7 / 29,187 |
| gin-gonic__gin-1805 | routergroup.go | | createStaticHandler | 7 / 28,635 |
| sharkdp__bat-1892 | src/assets.rs, src/assets/ignored_suffixes.rs, src/bin/bat/app.rs, src/bin/bat/clap_app.rs, src/syntax_mapping.rs | src/syntax_mapping/ignored_suffixes.rs | 11 names | 7 / 28,842 |
| sharkdp__bat-2393 | src/bin/bat/assets.rs, src/bin/bat/main.rs | | 14 names | 8 / 33,396 |
| preactjs__preact-3689 | hooks/src/index.d.ts, hooks/src/index.js, hooks/src/internal.d.ts | | 5 names | 8 / 37,536 |
| preactjs__preact-3010 | src/diff/children.js, src/index.d.ts | | diffChildren | 8 / 35,052 |

Full lists: `selection.json`. Context sizes: `contexts.json`. Atlas indexes: `index/<task>.txt`.

## Checkouts

Full clones from GitHub, a worktree per task at its `base_commit` (in a scratch directory outside this repo). Every
context is built from that worktree, so nothing of the fix is visible. The agent works in a `git archive` export of
the base commit (no `.git`, so no history to read).

## Arms

All arms: Opus 5.5 (`claude-opus-5-5`), `--effort high`. One call per task and arm, no retries except on errors
(errors are not cached or recorded).

1. `tree`: the paths of the atlas's file set (`repo_atlas.collect` at the base commit: tracked source files; tests,
   docs dirs, changelogs, lock files, vendor dirs and files without a minifier left out). 1-3k characters.
2. `map`: per file of the same set, its path, then the first line of every declaration (`signatures`): Python:
   every `class` / `def` / `async def` header at every nesting level from `ast` (decorators, docstrings and bodies
   left out; regex fallback for Python 2 files); Go: lines starting `func` or `type`; Rust: `fn`, `struct`, `enum`,
   `trait`, `type`, `mod`, `union`, `impl` and `macro_rules!` lines; JS/TS: `function`, `class`, arrow and
   `function` assignments, `X.prototype.y =`, `interface` / `type` / `enum` / `namespace`, and indented method heads;
   other files: the path only. 20-24k characters (about 6-7k tokens).
3. `atlas`: the pages of `repo_atlas.build` at the base commit (88.2 px² per character, 1932 px wide, sent at the
   high-resolution tier), in order, then a note on the minified format and `index.txt` as text.
4. `text`: the same files as plain text, each after `=== <path> ===`, on requests and gin only.
5. `agent`: Claude Code headless (`claude -p`) in the export, clean environment (`claude_eval.build_env`), tools
   `Read,Grep,Glob` only (no Bash, edits, web or MCP; `--strict-mcp-config`, no settings sources), Claude Code's own
   system prompt, `--max-turns 40`. Recorded: cost, turns, input and output tokens, seconds, tool calls, any path
   outside its directory and any permission denial.

Same question for every arm: the context (arms 1-4), then the issue text (`problem_statement`) in `<issue>` tags,
then: "List the files that must change to fix this issue, most likely first: at most 5 paths, exactly as they
appear in the repository, one per line as `FILE: <path>`. Then one line `PLAN: <one sentence>`." Arms 1-4 have a
two-sentence system prompt (`SYSTEM`). The agent's prompt starts with one line saying the directory is a checkout of
the repository and that it should look at the code as much as it needs and change nothing.

## Checks

* Atlas calls: `input_tokens` at least 97% of the image-token formula (sum of ceil(W/28) x ceil(H/28)), else the
  call is invalid (pages dropped) and is reported as such.
* Every atlas is at most 150k image tokens at the base commit (largest: 78,315).
* Paths not in the repository at the base commit (`git ls-files`) count as wrong and are reported.
* Refusals or cut-offs are recorded as the result (no files named); nothing is reworded.

## Scores per task and arm

* **Primary:** all scored gold files among the (at most 5) named files.
* First named file is a gold file; share of gold files named.
* Functions or classes touched by the gold patch named in the PLAN line or the paths (whole-word match). The names
  come from the hunk headers (the enclosing definition git shows) and from declarations on changed lines of gold
  files, so the list is approximate and includes common words (`request`, `run`, `default`): a secondary score only.
* Cost (`total_cost_usd`, list price), input tokens, seconds, turns.

## Predictions (the coordinator's, kept as given)

* **P1** atlas scores at least as high as map on the primary score (8 tasks: a direction only).
* **P2** atlas and text agree on the requests and gin tasks (at most 1 of the 4 differs on the primary score).
* **P3** the agent arm scores at least as high as the atlas on the primary score.
* **P4** on requests and gin the atlas call costs less than the agent run; on bat it costs more.

## Training-data caveat

These are public repositories and issues the model may have seen, fixes included. That applies to every arm alike;
the `tree` arm shows how far the names alone (and memory) go.

## Cost estimate (list price, before any call)

Prices fitted to earlier Opus 5.5 rows (`results/claude_qa/repeat`): $5 per million input tokens written to cache,
$20 per million output tokens; 4,000 output tokens per one-shot call assumed (high effort). The agent arm is assumed
at $0.50 per task: earlier full-solve Claude Code runs on SWE-bench in this repo cost $0.05-0.35 at 4-17 turns
(`results/readhook_swe*`).

| Arm | Calls | Estimate |
|---|---|---|
| tree | 8 | $0.68 |
| map | 8 | $0.94 |
| atlas | 8 | $2.40 |
| text (requests ~200k tokens each, because these old commits carry vendored urllib3 and chardet under `requests/packages/`) | 4 | $3.27 |
| agent | 8 | $4.00 |
| **Total** | 36 | **$11.30** (cap $12) |

The runner books a generous bound per call before making it (8k output tokens; for the agent $1.50 until measured,
then twice the largest measured run) and does not launch a call that could take the total over $12. Order: tree,
map, atlas, then agent, then text, so that a cap hit drops text calls (named in the results) rather than the agent.

## Notes fixed before the first call

* The scored gold leaves out files the fix creates (gin-1957 `binding/header.go`, bat-1892
  `src/syntax_mapping/ignored_suffixes.rs`): they cannot be named "exactly as they appear in the repository". The
  brief counted every gold file; this is the one change, made before any call.
* The file set is `repo_atlas.collect`'s: it keeps top-level files such as `README.rst`, `Makefile` and `setup.py`,
  and requests' vendored `requests/packages/` (the repository as it was).

## Results (2026-11-12)

### What the pilot shows, and what it cannot

* **The harness works.** 36 calls, no errors, no refusals, no cut-offs. Every atlas call arrived whole (input
  tokens 103-108% of the image-token formula). The agent stayed in its directory and had no permission denials.
* **The question is too easy for these tasks.** Every arm named every scored gold file on every task, the `tree` arm
  too (paths only, about 2k tokens). So the primary score is at the ceiling and cannot tell the arms apart.
* **The tasks look memorised, or the issues give the answer away.** With nothing but the file list, the model wrote
  plans that match the real fixes in detail: `to_native_string` for requests-1724, `bigint` in `diffChildren` for
  preact-3010, `c.handlers = group.engine.noRoute` for gin-1805, the `IgnoredSuffixes` struct for bat-1892. The
  calls used almost no thinking (0-274 thinking tokens at high effort, 2-6 seconds).
* **The agent is the cheapest arm that reads code, not the dearest.** Claude Code needed 2-5 turns, 1-4 tool calls
  (mostly one Grep) and $0.02-0.06 per task. The atlas cost 4-17 times more per task.
* **8 tasks cannot decide anything statistically.** Here it does not matter: all arms tie.

### Per arm

| arm | tasks | all gold files in top 5 | first file right (or a file the fix creates) | gold functions named (approx.) | mean input tokens | mean cost | mean seconds | turns | wrong paths | pages dropped |
|---|---|---|---|---|---|---|---|---|---|---|
| tree | 8 | 8 of 8 | 7 of 8 (8) | 0.64 | 2,155 | $0.015 | 3 | 1 | 0 | 0 |
| map | 8 | 8 of 8 | 6 of 8 (7) | 0.65 | 11,063 | $0.060 | 3 | 1 | 0 | 0 |
| atlas | 8 | 8 of 8 | 6 of 8 (7) | 0.64 | 45,403 | $0.232 | 5 | 1 | 0 | 0 |
| text | 4 | 4 of 4 | 2 of 4 (3) | 0.58 | 235,519 | $1.182 | 4 | 1 | 0 | n/a |
| agent | 8 | 8 of 8 | 6 of 8 (7) | 0.55 | 17,599 | $0.038 | 10 | 3.1 (2-5) | 1 | n/a |

"First file right" misses on two tasks for most arms: requests-1724 (models.py named before sessions.py, the
gold file) and gin-1957 (the new file `binding/header.go` named first; in brackets it counts as right). "Wrong
paths" leaves out `binding/header.go`, which every arm named correctly as a new file; the agent's one wrong path is
a new test file it proposed (`hooks/test/browser/useErrorBoundary.test.js`), in fourth place. Seconds for one-shot
arms are wall time per call; the agent's include start-up.

### Per task (all = every scored gold file in the top 5; 1st = first file is gold)

| task | gold files | tree | map | atlas | text | agent |
|---|---|---|---|---|---|---|
| psf__requests-1142 | 1 | all, 1st | all, 1st | all, 1st | all, 1st | all, 1st |
| psf__requests-1724 | 1 | all, 1st | all | all | all | all |
| gin-gonic__gin-1957 | 2 (+1 new) | all | all | all | all | all |
| gin-gonic__gin-1805 | 1 | all, 1st | all, 1st | all, 1st | all, 1st | all, 1st |
| sharkdp__bat-1892 | 5 (+1 new) | all, 1st | all, 1st | all, 1st | | all, 1st |
| sharkdp__bat-2393 | 2 | all, 1st | all, 1st | all, 1st | | all, 1st |
| preactjs__preact-3689 | 3 | all, 1st | all, 1st | all, 1st | | all, 1st |
| preactjs__preact-3010 | 2 | all, 1st | all, 1st | all, 1st | | all, 1st |

### Predictions

* **P1** atlas at least as high as map on the primary score: **PASS** (8 of 8 each), but uninformative: a tie at
  the ceiling, which the file list alone also reaches.
* **P2** atlas and text agree on the 4 requests and gin tasks: **PASS** (4 of 4 agree), at the ceiling.
* **P3** agent at least as high as atlas: **PASS** (8 of 8 each), at the ceiling.
* **P4** atlas cheaper than the agent on requests and gin, dearer on bat: **FAIL.** The atlas cost more everywhere:
  requests $0.37 and $0.42 vs agent $0.024 and $0.044; gin $0.16 and $0.16 vs $0.031 and $0.040. On bat it was
  dearer as predicted ($0.16 and $0.18 vs $0.037 and $0.058).

### Real cost

| arm | calls | spent (list price) | per call |
|---|---|---|---|
| tree | 8 | $0.12 | $0.015 |
| map | 8 | $0.48 | $0.060 |
| atlas | 8 | $1.86 | $0.232 (about $5 per million image tokens) |
| text | 4 | $4.73 | $1.18 (requests $1.73 and $1.87: 345-373k tokens with the vendored packages) |
| agent | 8 | $0.30 | $0.038 |
| **total** | 36 | **$7.49** | estimate was $11.30; cap $12 |

The one-shot arms used far fewer output tokens than assumed (95-430, not 4,000). The agent cost a tenth of what was
assumed: it reads Claude Code's ~10k-token system prompt from cache and stops after one or two searches.

**Re-estimate of the full stage 1 (~48 tasks over the 11 repositories of the plan):** atlas about $0.35 per task
(mean atlas near 70k image tokens) = ~$17; text on requests, gin and flask (~18 tasks; requests alone ~$1.8 each)
= ~$18; map ~$3; tree ~$1; agent ~$0.05-0.10 per task on the larger repositories = ~$2-5. **About $40-45**, or
about $25 without the text arm on requests. But see below: as designed, it would most likely buy another ceiling.

### Problems and what to change before a full stage 1

1. **Ceiling.** SWE-bench issues for these repositories usually name the API or the symptom precisely enough, and
   the model has likely seen the fixes. A full run with this score would cost ~$40 to tie again. Options: keep only
   tasks the `tree` arm gets wrong (a free screen at $0.015 per task); score something harder (top-1 file, the
   functions, or the lines the fix touches); or use tasks from after the model's training data (e.g. issues fixed in
   the last months, or SWE-bench-Live style collections), where memory cannot help.
2. **The agent baseline is cheap.** At $0.04 per task and 3 turns, the atlas cannot win on cost for locating a
   change; it would have to win on quality on tasks where search fails (spread-out changes, no greppable name).
3. **requests at these base commits vendors urllib3 and chardet** (`requests/packages/`), tripling its atlas
   (71-78k image tokens) and making the text arm cost ~$1.8 per call. Either leave vendored packages out (a change
   to `repo_atlas`'s file set) or drop requests from the text arm.
4. **Gold files the fix creates** (gin-1957, bat-1892) were taken out of the scored gold before the first call (see
   the notes above). Every arm named gin's new file exactly; none named bat's.
5. The gold function list is approximate (hunk headers plus changed declarations, including generic names such as
   `run` and `default`); treat that column as a rough signal only.

### Files

`runs.jsonl` (one row per task and arm, with the full answer), `spend.jsonl` (every uncached call's cost),
`selection.json`, `contexts.json`, `index/` (atlas indexes), `run*.log`. Clones, checkouts, page images and agent
transcripts stayed in a scratch directory and are not committed.
