# Repo atlas against SWE-QA: does the atlas help answer questions about a repository?

*Pre-registered 2026-11-12, before any scored call. Plan and the earlier pilot: [docs/claude/REPO_ATLAS_PLAN.md](../../docs/claude/REPO_ATLAS_PLAN.md),
[results/repo_atlas/README.md](../repo_atlas/README.md). Code: `experiments/repo_atlas_qa.py` (tests
`tests/test_repo_atlas_qa.py`). Results go below the "Results" heading; nothing above it changes after the first
scored call, except dated amendments.*

## Question

The pilot asked "which files must change?" and every arm, even a bare file list, got every answer right. This test
asks about comprehension directly, with a benchmark built for it. **Does having the whole repository as page images
(the atlas) give better answers to repository questions than a repo map, than the same code as text, and than Claude
Code searching the checkout?**

## The benchmark, as found

SWE-QA (Peng et al., "SWE-QA: Can Language Models Answer Repository-level Code Questions?", arXiv 2509.14635, ACL
Findings 2026).

* **Data.** GitHub `peng-weihan/SWE-QA-Bench` (cloned at `c13deac`), `Benchmark/<repo>.jsonl`: 15 Python
  repositories x 48 questions = 720. The Hugging Face copy `swe-qa/SWE-QA-Benchmark` is byte-identical for the three
  files used here (checked with `cmp`). Each row has only `question` and `answer` (a reference answer, mean 278
  words, usually naming files, classes and functions, often with line numbers and short code).
* **Pinned commits.** `repo_commit.txt` gives a short hash per repository (v1: 12 repositories, v2 adds conan,
  streamlink, reflex). All resolved on GitHub. The commits used here: psf/requests `46e939b` (2025-08-06),
  pallets/flask `85c5d93` (2025-06-12), pytest-dev/pytest `5989efe` (2025-08-05). The references match these
  commits (for example requests' `src/` layout).
* **Categories.** The rows carry no category label. The paper's taxonomy has four top-level types, What / Why /
  Where / How, and each repository has 12 questions of each (a few start "In ..." or "At ..."; the type is the first
  of those four words in the question). The finer labels the brief mentions (intention, cross-file reasoning,
  multi-hop) are **not in the released data**. As a stand-in for "cross-file" this test uses: the reference answer
  names at least two distinct `.py` files (`cross_file`; 100 of the 144 questions).
* **Judge.** `Benchmark construction/score/llm-as-a-judge.py`: one call per answer; the judge sees the question, the
  reference and the candidate, and scores five criteria (correctness, completeness, relevance, clarity, reasoning),
  each 1-20, "STRICT and CONSERVATIVE"; total 5-100. The paper used GPT-5 with a 1,024-token thinking budget.

## Repositories and questions

Every SWE-QA repository whose atlas (`repo_atlas.collect` defaults: tracked source files; tests, docs, examples,
vendored and data files left out) is at most 150k image tokens at its pinned commit:

| repository | files | source chars | pages | image tokens | text tokens (est.) | fits | arms |
|---|---|---|---|---|---|---|---|
| requests | 29 | 197k | 5 | 20,424 | 68k | yes | all five |
| flask | 33 | 346k | 8 | 35,604 | 120k | yes | all five |
| pytest | 110 | 1.37M | 30 | 140,622 | 478k | yes | no text (over 130k) |
| streamlink | 271 | 1.94M | 42 | 198,030 | 672k | no | |
| pylint | 203 | 1.90M | 41 | 191,613 | 669k | no | |
| reflex | 291 | 2.14M | 47 | 223,560 | 746k | no | |
| conan | 335 | 2.33M | 51 | 239,292 | 805k | no | |
| xarray | 137 | 2.69M | 59 | 278,898 | 935k | no | |
| sqlfluff | 279 | 3.24M | 67 | 314,640 | 1.12M | no | |
| sphinx | 434 | 4.60M | 98 | 463,059 | 1.68M | no | |
| astropy, django, matplotlib, scikit-learn, sympy | | | | | | no (not measured: each is several times sphinx's or pytest's size) | |

(Image tokens of the built pages; `selection.json` also has `repo_atlas.stats`' figures, up to 1.4% higher.)

**All 144 questions** of requests, flask and pytest (48 each, 12 per category; requests has 13 What and 11 Why).
No sampling. `selection.json` lists them with their category, the cross-file flag and the reference's entities.

## Arms

All arms: Opus 5.5 (`claude-opus-5-5`), `--effort high`, the same instruction and question wording.

1. `closed`: no repository context, only "the questions that follow are about the {repo} repository at commit
   {sha}. No code from it is shown." Measures prior knowledge (these are public repositories).
2. `map`: the repo map of the atlas's file set (`repo_atlas_eval.build_map`: per file, every class / def header from
   `ast`). requests 12k chars, flask 30k, pytest 144k.
3. `atlas`: the pages of `repo_atlas.build` at the pinned commit (88.2 px² per character, 1932 px wide, sent at the
   high-resolution tier), then the pilot's note on the minified format and `index.txt` as text.
4. `text`: the same files as plain text, each after `=== <path> ===` (requests 198k chars, flask 348k); not on
   pytest (~478k tokens).
5. `agent`: Claude Code headless (`claude -p`) in a `git archive` export of the pinned commit (no `.git`; the whole
   tree, tests and docs included), tools `Read,Grep,Glob` only, no settings, no MCP, `--max-turns 40`, Claude Code's
   own system prompt, one question per run. **On a seeded subset of 72 questions** (24 per repository, 6 per
   repository and category; `agent_subset(72)`) for cost, see below.

**Batching (arms 1-4).** 8 questions per call, the same batches for every arm: per repository, each category
shuffled (seed 20261112) and dealt round-robin, so every batch holds 2 What, 2 Why, 2 Where and 2 How questions;
6 calls per repository and arm (`selection.json`, `batches`).

**Call layout (arms 1-4).** Each call is two user messages in one `claude -p` session: (1) the context (images
first for the atlas) followed by "Reply only OK."; (2) the instruction and the 8 questions. Reason, measured before
this pre-registration (6 probe calls, $0.31): with context and questions in one message the CLI never reads the
context from the prompt cache on the next call (13.7k tokens written again on every call); with the context as its
own first message, the next call reads it from cache. The first batch of each (repository, arm) runs alone so that
it writes the cache, then the other five run in parallel. The extra "OK" turn is the same for every arm.

**Instruction (arms 1-4 and agent, verbatim):** "Answer each question below about the {repo} repository (commit
{sha}). For each one, start a new line with its id exactly as given (for example `Q7:`), then one paragraph of at
most 120 words: the answer, naming the specific files, classes and functions involved. No code blocks, no headings,
no other text. If you are not sure, give your best answer." then the questions as `Q<n>: <question>`. Arms 1-4 have
a two-sentence system prompt (`SYSTEM`). The agent's prompt starts with one line: the directory is a checkout of the
repository at that commit, look at the code as much as needed, change nothing.

**Length cap:** 120 words per answer for every arm (the references average 278 words; the judge penalises shorter
answers, equally for every arm).

## Scoring

* **(a) Primary: the benchmark's judge.** The rubric of `llm-as-a-judge.py` verbatim, split at "INPUT:" so that the
  fixed text is the system prompt (read from cache) and the INPUT block (question, reference, candidate) is the user
  message, plus one sentence saying where the INPUT is. Judge: Opus 5.5 at `--effort low` (Claude as judge, as the
  brief asks; the same model as the answerer, for every arm alike). One call per answer, in a seeded random order;
  the judge sees no arm name, batch or other answer. Total = sum of the five criteria (5-100). An empty or missing
  answer is not sent to the judge and scores the minimum, 5. An unparsable judge reply is reported and left out.
* **(b) Objective: entity recall.** Code entities in the reference (file paths, dotted names, identifiers in
  backticks, CamelCase names, snake_case names with an underscore; code blocks skipped; shorter forms of the same
  thing merged; `extract_entities`, about 10 per question): the share named in the answer, as whole tokens. A path
  also counts by its last two components or its file name; a dotted name by its last two components, or its last
  component when that has an underscore or an inner capital (`entity_variants`). Also reported over non-test
  entities only, because 62 of the 144 references cite test files, which arms 2-4 do not have and the agent does.
* Cost (list price, `total_cost_usd`), input tokens, seconds and (agent) turns per question. For the batched arms a
  call's cost and seconds are split equally over its 8 questions.

## Validity

* Atlas calls: the question turn's input tokens must be at least 97% of the image-token formula (sum of
  ceil(W/28) x ceil(H/28)), else the call is invalid (pages dropped) and its answers are reported and left out.
* Refusals and classifier cut-offs are recorded as the result (empty answer = 5); nothing is reworded. Errors are
  retried (not cached, not recorded as rows); their cost is booked.
* Answers missing from a batched reply count as empty (5).

## Analysis

Per arm: mean judge score and entity recall with 95% bootstrap intervals over questions (10,000 resamples).
Paired differences on the questions both arms answered: atlas − map, atlas − agent (72), atlas − text (96),
atlas − closed and each arm − closed, each with an interval. Per category (What / Why / Where / How, and the
cross-file stand-in). Before reading any difference: the closed-book arm and the spread of scores (ceiling check).

## Predictions (the coordinator's, kept; scale = the judge total, 5-100, so 10% ≈ 10 points and 3% ≈ 3 points)

* **S1** closed-book is clearly below every arm with context: each context arm's mean is at least 10 points above
  `closed` on the paired questions.
* **S2** atlas beats map: the 95% interval of the paired difference atlas − map is above 0.
* **S3** atlas and text differ by less than 3 points (96 questions). PASS if the whole 95% interval of atlas − text
  lies inside (−3, +3); FAIL if the point estimate lies outside; otherwise not evaluable (too uncertain).
* **S4** atlas is within 3 points of the agent (72 questions; same rule as S3) **and** costs less per question than
  the agent (mean list price per question). Both parts must pass.
* **S5** the atlas's advantage over map is largest on the cross-file / multi-hop category. The data has no such
  label, so it is evaluated on the stand-in: PASS if atlas − map on the cross-file questions is larger than atlas −
  map on the other questions (point estimates). The stand-in cuts across What / Why / Where / How, so the per-type
  differences are reported alongside, not used for the verdict.

Entity recall is reported for the same comparisons as a check that needs no judge; the verdicts above use (a).

## Training-data caveat

The repositories, their code at these commits (mid-2025) and the benchmark (published 2025-09) are public and may be
in Opus 5.5's training data, references included. The closed-book arm measures how far memory goes; differences
between context arms are what this test reads.

## Cost estimate (list price, before any scored call)

Prices fitted to earlier Opus 5.5 rows: $5 per million tokens written to cache, $0.20 per million read, $20 per
million output. Measured on streamlink (not a selected repository; harness check, $0.37, no rows): one closed batch
of 8 questions $0.098 (4.5k output tokens, 2.5k of them thinking); agent runs $0.045-0.082 (4-6 turns, 11-18 s);
judge calls $0.016 with the rubric not yet cached (~$0.006-0.008 once it is read from cache).

| part | calls | estimate |
|---|---|---|
| closed | 18 | $1.49 |
| map | 18 | $1.91 |
| atlas | 18 | $2.93 (context written once per repository, then read at 0.04x) |
| text | 12 | $2.22 |
| agent, 72 questions | 72 | $5.04 (at $0.07) |
| judge | 600 answers | $4.80 (at $0.008) |
| harness checks already spent (6 cache probes, 1 batch, 3 agent runs, 6 judge calls) | 16 | $0.68 |
| **total** | | **$19.07** (cap $22) |

All 144 agent questions would add ~$5.6 (agent and judge) and break the cap, so the agent arm runs on the 72-question
subset. **Order:** arms 1-4, then the judge on their answers (to measure the judge's real cost), then the agent arm
on the largest prefix of `agent_subset` that fits what is left (72, or fewer in steps of 12, which keeps it balanced),
then the judge on the agent's answers. The runner books a bound per call before making it and launches nothing that
could take the total (`spend.jsonl`, harness checks included) over $22.

## Amendments

* **2026-11-12, 09:20 UTC, before the rest of the agent arm (written after arms 1-4 and their judging, after 44 of
  the 72 agent runs).** The agent arm was cut off by the subscription's 5-hour usage limit (reset 09:10 UTC) and the
  container restarted. Failed runs were never written as rows (the runner records an answer only after a successful
  run), so the 44 agent rows are all complete; 4 runs that failed with the limit error had cost $0.10, which stays in
  `spend.jsonl`. **Subset rule, unchanged:** the agent arm runs on the pre-registered `agent_subset(72)` (24 per
  repository, 6 per repository and category); the 28 questions of it not yet answered are run now. Spent so far
  $17.76; the remaining 28 runs (~$0.045 each, measured mean of the 44) and the judging of the 72 agent answers
  (~$0.0094 each, measured) are estimated at ~$1.95, total ~$19.7 under the $22 cap. If the cap or the limit stops
  it short, the agent arm is reported on the largest balanced prefix completed (multiples of 12) and the rest is
  named. Other changes: at most 3 calls run at once (coordinator); a usage-limit error now stops the runner instead
  of being retried; the shared `git archive` export is made atomically (a race between parallel runs crashed the
  first agent start; the 5 runs that completed then had a complete tree: only the thread that extracted it went on
  to run).
* **Reservation floor.** Before the agent arm started, its per-run cost reservation was lowered from $0.50 to $0.25
  (measured runs $0.03-0.08) so that in-flight bookings would not skip the tail of the subset under the cap.

* **2026-11-12, ~09:45 UTC, after all pre-registered arms were judged (exploratory, not a prediction).** The agent
  arm scored ~3.5 points above every other arm, but it differs from them in two ways besides searching the code: it
  answers one question per call (arms 1-4 answer 8 per call) and its answers are longer (108 words on average vs
  85-93; the judge rewards completeness). To see how much of the gap the call format alone explains, one control is
  added with the money left (~$2.30 under the cap): `closed1`, the closed-book arm asked **one question per call**
  (same system prompt, context message and instruction), on the first 36 questions of `agent_subset` (12 per
  repository, 3 per repository and category), judged the same way. Estimated ~$1.1. It is reported as exploratory
  and changes no verdict of S1-S5.

## Results (2026-11-12)

### Verdict

**No. On SWE-QA the atlas does not help Opus 5.5 answer questions about requests, flask and pytest.** Its answers
score the same as answers written with no code at all (atlas − closed-book: −0.1 points on a 5-100 scale, interval
−0.9 to +0.6, 144 questions). The repo map and the full text do no better. Only Claude Code searching the checkout
beats memory, by about 3.5 points, and it beats the atlas by 3.3 (2.3 to 4.4), at about twice the atlas's cost per
question ($0.045 vs $0.023). Fairly sure: every arm has 72-144 paired questions and the intervals are narrow. The
catch is the size of the effect: on these well-known repositories the model already knows most of what the
references say (closed-book names 71% of the reference's code entities), so the test measures what context adds on
top of memory, and for the atlas, map and text that is nothing.

### Per arm (judge total 5-100, entity recall 0-1; 95% bootstrap intervals over questions)

| arm | questions | judge score | entity recall | recall, non-test entities | mean input tokens | cost per question | seconds per question | turns |
|---|---|---|---|---|---|---|---|---|
| closed | 144 | 75.6 (74.5-76.6) | 0.71 (0.68-0.75) | 0.71 (0.68-0.75) | 1,271 | $0.012 | 5 | 1 |
| map | 144 | 76.1 (75.2-76.9) | 0.71 (0.68-0.74) | 0.72 (0.68-0.75) | 28,402 | $0.016 | 5 | 1 |
| atlas | 144 | 75.4 (74.4-76.4) | 0.70 (0.67-0.74) | 0.72 (0.69-0.75) | 67,918 | $0.023 | 5 | 1 |
| text (requests, flask) | 96 | 75.5 (74.2-76.7) | 0.71 (0.66-0.75) | 0.71 (0.67-0.76) | 97,062 | $0.028 | 4 | 1 |
| agent (subset) | 72 | **79.1 (77.8-80.3)** | **0.79 (0.76-0.83)** | 0.78 (0.74-0.82) | 25,081 | $0.045 | 11 | 4.3 (2-9) |
| closed1 (exploratory, one question per call) | 36 | 76.2 (73.8-78.2) | 0.74 (0.68-0.79) | 0.73 (0.67-0.79) | 819 | $0.022 | 9 | 1 |

Arms 1-4: cost and seconds are a call's (8 questions) divided by 8; input tokens are the question turn's prompt (the
context plus the questions). No answer was missing, refused or cut off; no judge reply was unparsable; every atlas
call arrived whole (input tokens 102-109% of the image-token formula: 22.2k / 37.4k / 144.0k for 20.4k / 35.6k /
140.6k image tokens).

### Paired differences (same questions)

| difference | questions | judge score | entity recall |
|---|---|---|---|
| atlas − map | 144 | −0.6 (−1.3 to +0.0) | −0.004 (−0.024 to +0.015) |
| atlas − text | 96 | +1.1 (+0.4 to +1.8) | +0.016 (−0.010 to +0.043) |
| atlas − agent | 72 | **−3.3 (−4.4 to −2.3)** | **−0.115 (−0.152 to −0.079)** |
| atlas − closed | 144 | −0.1 (−0.9 to +0.6) | −0.008 (−0.032 to +0.015) |
| map − closed | 144 | +0.5 (−0.2 to +1.2) | −0.003 (−0.027 to +0.021) |
| text − closed | 96 | −0.8 (−1.7 to +0.1) | −0.025 (−0.057 to +0.007) |
| agent − closed | 72 | +3.5 (+2.3 to +4.7) | +0.077 (+0.045 to +0.108) |
| agent − closed1 (exploratory) | 36 | +3.1 (+1.7 to +4.7) | +0.030 (−0.011 to +0.072) |
| closed1 − closed (exploratory) | 36 | +1.0 (−0.3 to +2.4) | +0.027 (−0.021 to +0.078) |

### Per category (judge score; questions in brackets)

| category | closed | map | atlas | text | agent | atlas − map | atlas − agent |
|---|---|---|---|---|---|---|---|
| What | 73.5 (37) | 74.4 (37) | 74.3 (37) | 72.6 (25) | 77.9 (18) | −0.0 (−1.2 to +1.2) | −4.4 (−7.2 to −1.9) |
| Why | 75.4 (35) | 75.7 (35) | 75.1 (35) | 75.6 (23) | 79.5 (18) | −0.6 (−1.6 to +0.3) | −3.8 (−5.1 to −2.6) |
| Where | 77.4 (36) | 77.7 (36) | 76.0 (36) | 76.5 (24) | 81.1 (18) | −1.8 (−3.6 to −0.1) | −4.0 (−5.9 to −2.1) |
| How | 76.0 (36) | 76.5 (36) | 76.4 (36) | 77.3 (24) | 77.8 (18) | −0.1 (−1.4 to +1.1) | −1.1 (−3.2 to +0.9) |
| cross-file stand-in (reference names ≥ 2 .py files) | 75.9 (100) | 76.4 (100) | 75.5 (100) | 75.4 (64) | 78.8 (53) | −0.9 (−1.8 to −0.1) | −3.4 (−4.8 to −2.0) |
| the other questions | 74.9 (44) | 75.4 (44) | 75.3 (44) | 75.6 (32) | 79.8 (19) | −0.0 (−1.2 to +1.1) | −3.2 (−4.3 to −1.8) |

Per repository (judge score): flask closed 75.8, map 77.1, atlas 76.7, text 75.7, agent 79.7; pytest 74.2, 74.1,
73.2, —, 77.5; requests 76.8, 76.9, 76.4, 75.2, 80.1. Full output: `summary.md`.

### Ceiling and spread

Not a ceiling: the judge is strict (all 600+ answers: min 46, quartiles 74 / 77 / 80, max 89; no answer above 89).
It is a **floor of memory**: closed-book already scores 75.6 and names 71% of the reference's code entities (69% of
those not named in the question). These are three of the best-known Python repositories, and the benchmark (2025)
may itself be in the training data. The judge does see real differences: the largest atlas − closed gain (+17,
flask/Q39) is the closed-book answer naming `TagList` where the code has `PassList`, which the atlas got right; the
largest loss (−19, pytest/Q38) is a question about a test in `testing/`, which the atlas leaves out, where the atlas
answer guessed the fixture's scope wrong.

### Predictions

* **S1** closed-book at least 10 points below every context arm: **FAIL.** map +0.5, atlas −0.1, text −0.8, agent
  +3.5 above closed-book. Memory carries almost everything here.
* **S2** atlas beats map (interval above 0): **FAIL.** −0.6 (−1.3 to +0.0); if anything the map is slightly better.
* **S3** atlas and text within 3 points (whole interval inside ±3): **PASS**, +1.1 (+0.4 to +1.8), but empty: both
  are at the closed-book level, so pages and text tie because neither is used to any effect.
* **S4** atlas within 3 points of the agent and cheaper per question: **FAIL.** −3.3 (−4.4 to −2.3), point estimate
  outside ±3. The cost half passes ($0.023 vs $0.045 per question), but only because 8 questions share one atlas
  call and the context is read from cache; one question per atlas call would cost ~$0.11-0.85 (written context).
* **S5** atlas − map largest on the cross-file stand-in: **FAIL.** −0.9 on cross-file questions vs −0.0 on the
  others. (The stand-in is not the paper's label; the data has none.)

### Exploratory control: is the agent's lead just one question per call?

The agent answers one question per run and writes longer answers (108 words vs 85-94 for arms 1-4 on the same
questions). `closed1` (amendment 2) asks the closed-book question one at a time: its answers are as long (101
words) and it thinks less per question, yet it gains only +1.0 (−0.3 to +2.4) over batched closed-book, and the
agent stays +3.1 (+1.7 to +4.7) above it. So most of the agent's lead comes from reading the code, not from the call
format. Part of it can come from the tests: 62 of 144 references cite test files, which only the agent can open;
over non-test entities its recall lead shrinks a little (0.78 vs 0.72), it does not vanish.

### Cost

| part | calls | spent (list price) |
|---|---|---|
| closed | 18 | $1.73 |
| map | 18 | $2.30 |
| atlas | 18 | $3.32 |
| text | 12 | $2.69 |
| agent | 72 runs (+4 cut off by the usage limit) | $3.32 |
| closed1 (exploratory) | 36 | $0.78 |
| judge | 636 | $6.01 |
| harness checks | 16 | $0.68 |
| **total** | | **$20.83** (estimate $19.07; cap $22) |

The batched arms used ~2.2k thinking tokens per call (the closed batch cost about as much as estimated); the first
call per repository and arm wrote the context (atlas: requests $0.21, flask $0.27, pytest $0.85), later calls read
it ($0.09-0.17). Two first-wave requests calls were billed near the write price again (requests atlas b1/b2, text
b1), probably a cache miss on a parallel route.

### Problems

1. **Memory dominates.** On repositories this well known, SWE-QA measures what the model already knows; context can
   only add a few points, and the batched one-shot arms added none. A comprehension test of the atlas needs code the
   model has not seen (private or recent repositories), as the pilot also concluded.
2. **The atlas, map and text leave tests out; the agent sees the whole tree.** 62 of 144 references cite tests. This
   favours the agent, as in the pilot; the non-test recall column shows the lead is not only that.
3. **Batching.** Arms 1-4 answer 8 questions per call (needed for the cost); the agent one. The closed1 control
   suggests this explains about 1 of the agent's 3.5 points at most.
4. **Same model as answerer and judge** (Opus 5.5 at low effort), for every arm alike; the benchmark used GPT-5.
5. **Usage limit.** The 5-hour subscription limit stopped the agent arm after 44 runs; the runner had written no row
   for the failed runs, so they were rerun after the reset (amendment 1), and a usage-limit error now stops the
   runner. A race on the shared export crashed the first agent start (fixed; no row affected).
6. The cross-file category is a stand-in (≥ 2 `.py` files in the reference); the released data has no such labels.

### Files

`selection.json` (questions, categories, batches, entities), `contexts.json`, `index/` (atlas indexes),
`calls.jsonl` (one row per batched call, with the raw reply), `answers.jsonl` (one row per question and arm),
`judged.jsonl`, `spend.jsonl` (every uncached call), `summary.md` / `summary.json`. Clones, pages and agent
transcripts stayed in a scratch directory.
