# Repo atlas against SWE-QA: does the atlas help answer questions about a repository?

*Pre-registered 2026-11-12, before any scored call. Plan and the earlier pilot: [docs/REPO_ATLAS_PLAN.md](../../docs/REPO_ATLAS_PLAN.md),
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

## Results

*(after the calls)*
