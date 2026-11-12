# Repeat calls on wording-sensitive questions (Claude QA)

*2026-11-12.* Pre-registered below, committed before any scored call. Follows Test A
([../prune/README.md](../prune/README.md)); the v2 test and its rules: [docs/CLAUDE_QA.md](../../../docs/CLAUDE_QA.md).

(Results go here once the calls are in.)

---

# Pre-registration (committed before any scored call)

## The question

The repo's headline: Claude Opus 5.5 answers from page images at 117.6 px²/char (2.2x fewer input tokens than text)
with no measurable loss. Test A added 40 "small-word" questions per document (right only if "not", "all but", a
passive, a comparative's direction, before/after, from/to or "instead of" is read; each sentence holds a lure number,
the answer under the wrong reading). On those, the 117.6 pages scored 90.8% (text 100%, the denser 88.2 pages 98.3%).
In documents 1 and 2 all 5 passive questions were answered with the lure; in document 0 all 5 were right. With one
call per arm and document, this may be call-to-call variation or a real weakness of 2.2x pages on wording-sensitive
text. Settle which.

## Design

* **Same calls, repeated.** The same 3 documents, 160 questions per call, system prompt, prompt, page images
  (1932-px pages), model (Opus 5.5, the harness default) and `--effort high` as Test A's `text`, `page118` and
  `page88` arms. Checked before spending (`claude_qa_repeat.py check`): for all 9 (document, arm) pairs the unsalted
  cache key of the rebuilt call is the key under which Test A's response is stored (so model, system, prompt, effort
  and every image's PNG bytes are identical), the rendered pages equal Test A's saved PNGs pixel for pixel, and the
  image-token totals match Test A's rows.
* **Repeats.** 4 new calls per document for `page118` and for `page88` (5 per document and density with Test A's, 15
  per density), 1 new call per document for `text` (2 per document, 6 in all). 27 new calls.
* **Cache salt.** The response cache keys on model + system + prompt + image bytes + effort, so a repeat would return
  Test A's answer. `vbpe/claude_eval.py` gains an optional `cache_salt` (default None = the old keys, unit-tested)
  that enters only the cache key; repeat r uses salt `repeat{r}`. The request sent to the CLI is unchanged.
* **Concurrency:** up to 8 calls at once. Rows are appended per call (`runs.jsonl`); finished calls are skipped on
  re-run.
* **Validity (as in Test A).** Page calls: input tokens must reach 97% of the image-token formula, else pages were
  dropped and the call is invalid. Engagement: a call with unanswered questions or a disengagement phrase is invalid.
  API errors are not stored and are re-run with the same request. A refusal or safety cut-off is recorded as such and
  nothing is reworded. Invalid calls are left out of the primary analysis and listed.

## Before spending: what Test A's two failing 117.6 calls did

* **They answered every passive with the lure, and nothing else was wrong.** Document 1: W01 194 (right 531), W07 897
  (351), W14 858 (953), W22 795 (742), W34 507 (290); document 2: W05 311 (948), W14 273 (845), W19 84 (924), W20 926
  (648), W23 787 (853). 10 of 10 wrong answers are the lure, i.e. "B was paid n2 crowns by C" read as "B paid n2".
  The same two calls got all 35 other small-word questions right (negation, negated counts, "all but", comparatives,
  before/after, from/to, "instead of") and all 120 original questions. Document 0's 117.6 call got all 5 passives
  right and missed one negation (W21: 348, right 336, lure 818; neither number).
* **The replies hold only the answer lines** ("W01: 194"), so they do not show how the sentence was read.
* **The 88.2 pages made the same error once:** document 1's 88.2 call answered W01 with the lure 194 (and one
  before/after question, W38, with its lure); its other 4 passives were right.
* **Where the passive sentences sit** (`locate`, from the exact line layout of each page; `locations.json`): none of
  the 30 (15 sentences x 2 densities) is split across a page break; none is on the first or last line of a page; at
  117.6, 11 of 15 sit on one line and 4 run over 2 lines (lines 7 to 107 of 113). The two failing calls' passives
  are placed like document 0's (both groups mid-page; one-line and two-line sentences in both). Across all 240
  small-word sentence placements none is split across pages. **Nothing points to a layout artefact**; the place of
  each passive is tested again against the repeats (accuracy by one line vs two lines; top, bottom and split are
  empty).
* So the 117.6 loss in Test A is one decision per call (all five passives one way), which is why calls, not
  questions, are the unit below.

## Primary analysis (pre-registered)

* Small-word accuracy per call (40 questions). Per density: the mean over its 15 calls, and a 95% interval for
  (density − text) from a bootstrap (10,000 resamples) that draws whole calls with replacement within each document,
  for the density and for text independently. Mean = mean over all calls of the arm (5 per document for pages, 2 for
  text) minus mean over the 6 text calls.
* **Verdicts** (2-point margin, as the v2 lossless rule):
  * "2.2x (or 2.9x) loses on wording-sensitive questions" if the interval of (pages − text) excludes 0 and the mean
    loss is at least 2 points;
  * "no loss shown" if the worst case (lower end of the interval) is within 2 points of text;
  * otherwise "not settled".
* Also reported: the per-call scores; how many of the 15 calls per density miss every passive question; accuracy per
  small-word type; the share of answers equal to the lure; passive accuracy by where the sentence sits; and the same
  statistics on the original 120 questions (this also sizes the run-to-run variation of the headline lossless claim,
  which was never repeated).

## Predictions (the coordinator's)

* **R1.** The all-passives-wrong pattern recurs in some but not most 117.6 calls (2 to 8 of 15) and rarely at 88.2
  (0 to 2 of 15). PASS if both counts fall in those ranges.
* **R2.** The mean small-word loss at 117.6 is between 2 and 10 points and its interval excludes 0. PASS if both.
* **R3.** On the original 120 questions both densities stay within 2 points of text in the mean. PASS if both
  |mean difference| <= 2 points.

A prediction is not evaluable if invalid calls leave fewer than 4 valid calls per document for a density it needs.

## Cost

Estimate from Test A's measured cost of the same calls (x 1.1 for longer thinking): **$21.82 list price** for the 27
calls (page118 ~$0.71, page88 ~$0.60, text ~$1.37 per call). Cap: $25 at list price (sum of `cost_usd` over
uncached calls, `spend.jsonl`); the runner stops before a call that would cross it. The calls draw on the
subscription quota; these are not amounts billed.

## Reproduce

```bash
python3 experiments/claude_qa_prune.py build          # documents and questions (Test A; already on disk)
python3 experiments/claude_qa_repeat.py check         # free: identity with Test A, estimate
python3 experiments/claude_qa_repeat.py locate        # free: sentence placement on the pages
python3 experiments/claude_qa_repeat.py run --yes     # 27 calls, resumable
python3 experiments/claude_qa_repeat.py analyse       # -> analysis/summary.{md,json}
```
