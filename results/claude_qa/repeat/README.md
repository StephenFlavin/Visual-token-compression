# Repeat calls on wording-sensitive questions (Claude QA)

*2026-11-12.* Pre-registered below, committed before any scored call. Follows Test A
([../prune/README.md](../prune/README.md)); the v2 test and its rules: [docs/claude/CLAUDE_QA.md](../../../docs/claude/CLAUDE_QA.md).

## Verdict

**Pages lose a little on wording-sensitive questions at both densities, and 2.2x is no worse than 2.9x.** Over 15
calls each, small-word accuracy was 97.2% at 117.6 px²/char (2.2x fewer tokens) and 96.2% at 88.2 px²/char (2.85x),
against 100% for text in all 6 text calls. By the pre-registered rule both are a loss: −2.8 points (95% CI −5.2 to
−1.2) and −3.8 points (−6.3 to −1.7). The loss is small and just over the 2-point bar, so its size is uncertain (2 to
5 points); that pages lose something on these questions is fairly sure (every interval excludes 0).

**Test A's "2.2x flips the passives" was call-to-call variation, not a density effect.** The flip recurred in 2 of 15
calls at 117.6 (both Test A's) and in 3 of 15 at 88.2 (all new). In a flip call Claude answers all 5 passive
questions ("A was paid n1 by B, and B was paid n2 by C": how much did B pay?) with the lure, and gets the rest right.
In the other 25 page calls it gets 4 or 5 of 5 right. Text never flips (0 of 6). So a page call has about a 1 in 6
chance (5 of 30) of misreading every passive sentence the same way.

**On the original 120 questions the headline holds over repeats:** 117.6 pages 99.9% (−0.1, −0.2 to 0.0), 88.2 pages
99.3% (−0.7, −0.9 to −0.4); "no loss shown" for both.

All 27 calls completed; none was refused, cut off, lost pages (input tokens 7.2-7.3k above the image-token formula, as
in Test A) or failed the engagement check.

| density | calls | input tokens (mean) | small-word mean | − text (95% CI) | calls with every passive wrong | original 120 mean | − text (95% CI) |
|---|---|---|---|---|---|---|---|
| text | 6 | 248,971 | 100.0% | (reference) | 0 of 6 | 100.0% | (reference) |
| pages 117.6 px²/char (2.18x) | 15 | 114,188 | 97.2% | **−2.8 (−5.2 to −1.2): loss** | 2 of 15 | 99.9% | −0.1 (−0.2 to 0.0): no loss shown |
| pages 88.2 px²/char (2.85x) | 15 | 87,445 | 96.2% | **−3.8 (−6.3 to −1.7): loss** | 3 of 15 | 99.3% | −0.7 (−0.9 to −0.4): no loss shown |

Calls per document: 5 per density (Test A's + 4 repeats), 2 for text. Interval: bootstrap over whole calls within
document (10,000 resamples). Not pre-registered: 117.6 − 88.2 on small-word questions = +1.0 point (−2.0 to +4.2).

### Predictions

* **R1, the all-passives-wrong pattern recurs in 2-8 of 15 calls at 117.6 and in 0-2 of 15 at 88.2: FAIL.** 117.6:
  2 of 15 (in range). 88.2: 3 of 15 (above the range): the denser pages flip as often.
* **R2, the mean small-word loss at 117.6 is 2-10 points with an interval excluding 0: PASS.** −2.8 points (−5.2 to
  −1.2).
* **R3, both densities within 2 points of text on the original 120 in the mean: PASS.** −0.1 and −0.7 points.

### Per-call scores

Small-word accuracy per call (40 questions; passives right of 5 in brackets; Test A's call first):

| density | document 0 | document 1 | document 2 |
|---|---|---|---|
| 117.6 | 97.5 (5), 97.5 (5), 97.5 (5), 97.5 (5), 97.5 (5) | **87.5 (0)**, 100 (5), 100 (5), 100 (5), 100 (5) | **87.5 (0)**, 100 (5), 97.5 (4), 100 (5), 97.5 (4) |
| 88.2 | 100 (5), 100 (5), 100 (5), 100 (5), 100 (5) | 95.0 (4), 97.5 (5), **85.0 (0)**, 97.5 (5), **85.0 (0)** | 100 (5), **85.0 (0)**, 100 (5), 100 (5), 97.5 (5) |
| text | 100 (5), 100 (5) | 100 (5), 100 (5) | 100 (5), 100 (5) |

Original 120 per call: 117.6 pages 100% in 14 calls and 99.2% in 1; 88.2 pages 97.5-100% (document 1: 99.2, 100,
98.3, 97.5, 98.3; its misses are off-by-one counts on multi-step questions C35-C37); text 100% in all 6.

Accuracy per small-word type (pooled over calls):

| density | all but | before/after | comparative | from/to | instead of | negated count | negation | passive | answers = lure |
|---|---|---|---|---|---|---|---|---|---|
| 117.6 | 100% | 100% | 100% | 100% | 100% | 100% | 93% | 84% | 2.0% |
| 88.2 | 100% | 91% | 100% | 100% | 100% | 100% | 100% | 79% | 3.8% |
| text | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0.0% |

### What the failing calls have in common

* **One reading for all five passives.** In each of the 5 flip calls, all 5 passive answers are the lure (25 of 25),
  never another number; the same calls answer the other 35 small-word questions as well as the other calls of their
  density. The passive outcome per call is all-or-nothing: of 30 page calls, 5 got 0 of 5 right, 3 got 4, 22 got 5;
  none got 1-3.
* **Documents 1 and 2 only.** No call on document 0 flipped (0 of 10 page calls); 5 of 20 on documents 1 and 2 did.
  With 5 events this may be chance.
* **Not the layout.** No passive sentence is split across pages or on a page's first or last line. Passives on one
  line were right 86% of the time at 117.6 (n = 50) and 75% at 88.2 (n = 60); on two lines 80% (n = 25) and 93%
  (n = 15). The one passive missed outside a flip call (document 2, W05, twice at 117.6) runs over two lines.
* **Not the prompt cache or answer length.** Repeats that ran at the same time as another call on the same pages read
  the shared prefix from the prompt cache (8 calls, $0.16-0.19 each); 2 of the 5 flips had cache reads, 3 did not.
  Flip calls wrote 6.8-7.8k output tokens, within the range of the others (6.0-8.4k).
* **Sentence-specific misreads repeat in every call.** At 117.6, document 0's negation W21 ("did not receive 818
  maps ...; it received 336") was answered 348 or 246 in all 5 calls (neither the answer nor the lure: a misread
  number). At 88.2, document 1's W38 (before/after) was answered with the lure in all 5 calls, and document 2's W35 in
  2. Text read all three right. These are reading errors tied to how one sentence looks on one page set.

### Cost

**$16.17 at list price** for the 27 calls (`spend.jsonl`): text $1.36-1.38 per call, 117.6 pages $0.69-0.73 and 88.2
pages $0.58-0.60 when the prefix was not cached, $0.16-0.19 when it was. Below the $21.82 estimate (the estimate
assumed no cache reads). The calls drew on the subscription quota; these are not amounts billed.

### Not run, and limits

* Only Opus 5.5 at high effort, and only the 117.6 and 88.2 densities (not 58.8, which already loses on everything).
* Text had 2 calls per document; it was perfect in all 6, so its own spread is unmeasured but small.
* With 5 calls per document, the bootstrap draws from 5 values per document; the intervals are rough, and the
  loss rests on 5 flip calls and 3 repeated sentence misreads.
* The small-word questions are planted and templated (5 per type per document); real text rarely hinges on one
  passive this way.

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
