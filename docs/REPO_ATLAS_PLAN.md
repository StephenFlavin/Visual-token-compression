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
