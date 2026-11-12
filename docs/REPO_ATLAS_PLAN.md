# Repo atlas: does the whole repository as page images improve decisions on code? (draft for approval)

*2026-11-12. Nothing here has called Claude. Built and tested: `experiments/repo_atlas.py`
(`tests/test_repo_atlas.py`). The test below needs approval of its spend.*

## The idea

A repo map (aider-style) gives the model the names and signatures in a repository, a few thousand tokens. The
coding hooks in this repo can go further: minify every source file (only whitespace is removed) and draw the
whole repository as pages, so the model starts a task having read all of the code. As text that costs too much; as
pages it costs about 3.5 times less.

**What `repo_atlas.py` does:** every tracked source file (tests, docs, lock files, vendored and binary files left
out) is minified with the hooks' minifiers, joined with a `▶path◀` header per file, and rendered at the hooks'
density (88.2 px² per character). It also writes an index (file -> pages) of a few hundred text tokens.

**It cannot be one image.** A page is at most 1932 px wide, and a request above ~175k image tokens silently loses
its first pages. So a repository fits if its atlas is at most ~150k image tokens: about 1.3M characters of source.

| Repository (HEAD, source only) | Files | Source chars | Pages | Image tokens | As text (est.) | Fits one request |
|---|---|---|---|---|---|---|
| requests | 29 | 225k | 5 | 23,736 | 78k | yes |
| gin | 67 | 261k | 7 | 31,119 | 113k | yes |
| flask | 33 | 361k | 8 | 37,536 | 125k | yes |
| click | 25 | 463k | 11 | 47,748 | 160k | yes |
| preact | 138 | 453k | 12 | 55,062 | 189k | yes |
| axios | 104 | 564k | 14 | 64,170 | 223k | yes |
| bat | 121 | 622k | 15 | 67,137 | 261k | yes |
| fmt | 38 | 731k | 18 | 81,903 | 339k | yes |
| jq | 53 | 818k | 20 | 91,287 | 386k | yes |
| seaborn | 58 | 1.04M | 23 | 106,881 | 359k | yes |
| gson | 142 | 1.01M | 24 | 114,057 | 393k | yes |
| axum | 128 | 1.14M | 26 | 120,405 | 464k | yes |
| pytest | 118 | 1.56M | 35 | 162,633 | 544k | no (just over) |
| ripgrep | 120 | 1.62M | 37 | 172,983 | 657k | no (just over) |
| pylint, xarray, caddy, sphinx | 141-456 | 2.1M-4.8M | 46-102 | 215k-482k | 0.75M-1.75M | no |

Django, sympy, scikit-learn and the like are far too large. For those the atlas would have to cover one package,
which is a different test.

## What the repo already knows that bears on this

* **Reading is not where a coding agent's cost or failures are.** Tool results were 17% of the cost, and Opus
  solved the same tasks from pages and from text ([SWE_IMAGES.md](SWE_IMAGES.md)). An agent already finds the right
  files by searching. So the likely gain is fewer exploration turns, and better choices only where a task needs
  knowledge of code the agent would not think to open.
* **Pages are safe for looking things up and connecting facts, less so where one small word or symbol decides**
  (about 3 points lost on wording-sensitive questions, [CLAUDE_QA.md](CLAUDE_QA.md)). Edits must still be made from
  the file as text: the atlas is for deciding, not for copying.
* **Cost of carrying it:** the atlas is re-read on every turn at the cached price (a tenth of the input price). A
  60k-token atlas costs about $0.30 to load and about $0.024 per further turn at Opus list prices.

## Test (two stages, the second only if the first earns it)

**Stage 1: where does the change go?** One call per task and arm, no agent loop, scored automatically.

* **Tasks:** SWE-bench Verified and Multilingual tasks in repositories whose atlas fits at the task's base commit
  (pool: requests, flask, seaborn, gin, preact, axios, bat, fmt, jq, gson, axum: 86 tasks). At most 6 per
  repository, tasks whose fix touches two or more source files first: about 48 tasks.
* **Question put to the model:** the issue text, then "list the files that must change to fix this, most likely
  first, at most 5, and one sentence on the approach".
* **Arms:**
  1. `tree`: the list of file paths only (floor);
  2. `map`: a repo map: every class, function and method signature per file (about 5-20k tokens);
  3. `atlas`: the whole repository as pages, with its index;
  4. `text`: the whole repository as text, on the repositories where that is under ~130k tokens (requests, gin,
     flask), to separate "has the whole code" from "has it as images";
  5. `agent`: Claude Code on a checkout with its search and read tools, no edits, asked the same question. This is
     what the atlas has to beat in practice.
* **Scores:** all changed source files named in the top 5 (primary; changelogs and docs in the fix are not
  counted); the first file right; functions changed by the fix named in the answer; cost, turns and seconds per
  task. Paired exact tests, atlas against map and atlas against agent.
* **Predictions:** (1) atlas beats map by at least 5 points on the primary score, more on multi-file tasks;
  (2) atlas is within 3 points of text; (3) atlas and agent are within 5 points of each other, and the atlas is
  cheaper per task on repositories under ~60k image tokens and dearer above.
* **Go to stage 2 only if** the atlas is not more than 3 points below the agent and is either cheaper at equal
  quality or at least 5 points better on multi-file tasks.

**Stage 2: does it solve more, or solve as much for less?** Claude Code with the atlas read at the start against
plain Claude Code, on the multi-file tasks of stage 1, graded by each task's own tests (the no-Docker runners of
the earlier coding tests). Scores: resolved, turns, cost, overlap of the changed files with the fix. To be
pre-registered after stage 1.

## Cost (list price, estimates)

| Step | Calls | Estimate |
|---|---|---|
| Pilot: 8 tasks (requests, gin, bat, preact), all arms | ~35 | ~$7 |
| Stage 1 in full: ~48 tasks | ~205 | ~$55 (atlas ~$17, agent ~$17, text ~$7, map and tree ~$4, output ~$10) |
| Stage 2 | to be estimated from stage 1 | |

The agent arm's cost is the least certain; the pilot measures it.

## Use outside a test

A hook cannot put images into a session by itself. In Claude Code the atlas would be a folder of pages with its
index, built once per commit, and a session-start note telling the model to read the pages before planning (5 to
26 image reads for the repositories above), the same pattern as the compaction archive.

## Pilot results (2026-11-12)

*8 tasks (2 each from requests, gin, bat, preact), 5 arms, 36 calls, $7.49 at list price. Pre-registration, tables
and problems: [results/repo_atlas/README.md](../results/repo_atlas/README.md).*

* **Every arm found every changed source file on every task**, the file list alone included (tree, map, atlas and
  agent 8 of 8; text 4 of 4). The primary score is at the ceiling, so the pilot cannot tell the arms apart.
* With only the file list, the model described the real fixes in detail (`to_native_string`, `bigint` in
  `diffChildren`, `noRoute`): these tasks look memorised, or the issues give the location away.
* **Claude Code with Read/Grep/Glob is the cheapest reader:** 2-5 turns, $0.038 per task on average, against
  $0.23 for the atlas (4-17 times more per task) and $1.18 for the text arm. Prediction P4 (atlas cheaper than the
  agent on the small repositories) fails; P1-P3 pass only as ties at the ceiling.
* All atlas pages arrived (input tokens 103-108% of the image-token formula).
* **Before a full stage 1 (re-estimated at ~$40-45):** change the task so that it can fail: keep only tasks the
  `tree` arm misses, score the functions or lines rather than the files, or use issues from after the model's
  training data. As planned, it would most likely buy another tie.

## SWE-QA results (2026-11-12)

*SWE-QA (Peng et al., arXiv 2509.14635): all 144 questions of requests, flask and pytest, the three SWE-QA
repositories whose atlas fits one request at the benchmark's pinned commits (20k / 36k / 141k image tokens). Opus 5.5
at high effort, the benchmark's own judge rubric applied by Opus 5.5, 5-100 scale. $20.83 of a $22 cap.
Pre-registration, tables and problems: [results/repo_atlas_qa/README.md](../results/repo_atlas_qa/README.md).*

* **The atlas adds nothing over memory.** Closed-book (no code) 75.6, map 76.1, atlas 75.4, text 75.5; atlas −
  closed-book −0.1 (−0.9 to +0.6), atlas − map −0.6 (−1.3 to +0.0), atlas − text +1.1 (+0.4 to +1.8). Closed-book
  already names 71% of the code entities in the reference answers: these repositories are too well known.
* **Only Claude Code searching the checkout beats memory:** 79.1 on a 72-question subset, +3.5 over closed-book and
  +3.3 (2.3 to 4.4) over the atlas, at $0.045 per question (4.3 turns) against the atlas's $0.023 (8 questions per
  call, context read from cache). A one-question-per-call closed-book control gains only +1.0, so the agent's lead is
  mostly from reading code, partly from tests the atlas leaves out.
* Predictions: S1 (context ≥ 10 points over closed-book) FAIL; S2 (atlas > map) FAIL; S3 (atlas ≈ text) PASS, but
  both sit at the closed-book level; S4 (atlas within 3 of the agent and cheaper) FAIL; S5 (atlas gain largest on
  cross-file questions) FAIL.
* All atlas pages arrived (102-109% of the image-token formula). Batching needs the context as its own first user
  message: in one message with the questions, `claude -p` never reads it from the prompt cache.
* Like the pilot, this cannot show a benefit of the atlas on code the model already knows. The next test of the idea
  needs repositories or code from after the model's training data.

## Security flaw discovery: results (2026-11-12)

*12 published, already fixed flaws from the GitHub Advisory Database (published 2026-11-09 to 11-11, fixes committed
2026-07-01 or later, after the model's training data), in 10 small libraries (filegator, zt-zip, nuxt/content,
fast-xml-parser, shadow, ag-psd, compose-lint, kiwi, urwid, forge; atlases 27k-118k image tokens). Opus 5.5 at high
effort, asked to audit the vulnerable commit and report up to 5 findings (file, function, weakness class; no exploit
code). $20.70 of a $25 cap; the usage limit ended the run during task 13, so it closed on 12 tasks. Pre-registration,
tables and problems: [results/repo_atlas_sec/README.md](../results/repo_atlas_sec/README.md).*

* **The atlas did not help find the flaw.** A gold file in the top 5 findings: file list only 10 of 12, atlas 9,
  Claude Code with Read/Grep/Glob 7, the same agent with the atlas 10. File names give the location away even for
  flaws the model cannot know (`security.ts`, `ZipUtil.java`, `kiwi_auth/views.py`), so the score sits near its
  ceiling and cannot separate the arms. Stricter scores (gold file first, gold function, weakness class) are flat too.
* **One direction for a larger test:** the agent with the atlas beat the plain agent by 3 tasks to 0 (exact p =
  0.25) in about half the turns (median 8 vs 18; fewer on 11 of 12 tasks), at $0.67 against $0.46 per task. The
  plain agent lost one task to its 40-turn limit and one to reporting the repository's other advisory.
* Predictions: V1 (file list at most 30%) FAIL (83%); V2 (atlas beats file list by 3) FAIL; V3 (agent at least as good
  as the atlas) FAIL (7 vs 9); V4 (agent with atlas within 1 task of the agent, fewer turns) FAIL on tasks, PASS on
  turns.
* All pages arrived (102-127% of the image-token formula); no refusals; no tool call left the export.
* A sharper test would score functions or line ranges, or use flaws in files whose names say nothing about security.

## Hard fresh tasks: results (2026-11-12)

*17 Python tasks from the SWE-rebench leaderboard (created 2025-07 to 2026-05; nothing newer exists in SWE-rebench or
SWE-bench-Live), each with a gold patch over at least 3 source files, an atlas of at most 150k image tokens, local
validation without Docker, and a closed-book screen that dropped 12 of 37 tasks as already located. Claude Code
(Opus 5.5, high effort) solved each task once per arm: plain, with a repo map, with the atlas pages in the first
message. Graded by the tasks' own tests. $28.45 of a $35 cap. Pre-registration, tables and problems:
[results/repo_atlas_swe/README.md](../results/repo_atlas_swe/README.md).*

* **The atlas arm solved the most, but not significantly so:** 11 of 16 compared tasks against 8 for plain and 9
  for the map. All 3 discordant tasks went to the atlas (exact McNemar p = 0.25). One run per arm cannot separate
  this from run-to-run variation.
* **Fewer turns, higher cost.** The atlas run took fewer turns on 11 of 12 untied tasks (median 10 vs 12.5, p =
  0.006). It cost more on every task: $0.68 against $0.26 (2.6x). Wall time did not change. The map cost $0.49 and
  changed nothing measurable.
* Predictions: H1 (atlas within 2 tasks of plain) FAIL (3), not significant; H2 (fewer turns) PASS; H3 (dearer) PASS;
  H4 (map within 1 of plain) PASS; H5 (gold files touched at least as often) PASS.
* All pages arrived (106-194% of the image-token formula). In the pilot one map run downloaded a later PyPI release
  holding the fix. That task is left out, and later runs had no package index.
* Next step, if wanted: the same 16 tasks with 2-3 repeats per arm (about $45 at these prices), to tell a 3-task
  effect from noise.

## Migration to Rust: results (2026-11-12)

*CRUST-Bench (C to safe Rust over whole repositories; hand-written Rust interfaces and tests, graded by `cargo test`).
14 C projects (the 2 largest validated ones, 3,759-3,889 C lines, then the 12 smallest, 806-1,083) x 3 arms of Claude
Code (Opus 5.5, high effort): `plain` (C source + Rust skeleton), `atlas` (the same plus the C source as 1-3 pages in
the first message), `atlas_only` (Rust skeleton and pages, no C files). $24.01 of a $30 cap. Pre-registration,
amendments, tables and problems: [results/repo_atlas_migrate/README.md](../results/repo_atlas_migrate/README.md).*

* **The atlas changes nothing in a migration:** every arm has the same outcome on every project (12 of 14 pass all
  tests; lib2bit fails its one test and Genetic 1 of 23 in every arm). Turns, cost and time differ by noise (atlas vs
  plain: median -0.5 turns, p = 1.0; mean cost $0.57 vs $0.54). The atlas agents still opened the C files in all 14
  runs, usually with one `cat`: at 1-4k lines, reading everything costs the plain agent about one tool call.
* **Pages could stand in for the source here:** `atlas_only` matched `plain` on every project. That says little about
  page fidelity, since the Rust interfaces and tests carry much of the behaviour and the benchmark has been public
  since 2025.
* Predictions: M1 (plain and atlas within 1 project) PASS; M2 (fewer turns with the atlas) PASS on the letter only
  (7 fewer, 6 more); M3 (atlas_only worse) FAIL; M4 (atlas cost within 20%) PASS (+5%).
* Large projects cost about $2 per run, small ones about $0.30, so the pre-registered switch to the smaller end
  applied after the pilot. Fixed along the way, for all arms alike: per-binary grading (an hours-long test hid other
  binaries), and turns summed over the extra result events that background tasks add. Claude Code's auto-memory under
  `~/.claude/projects/` is readable by every run: one `atlas` run read another project's notes there (no `atlas_only`
  run did). Turn it off in future agent tests.
* A migration test that can separate the arms needs projects large enough that reading the source is real work
  (beyond what one `cat` delivers), tests that the interfaces do not give away, and code from after the model's
  training data.
