# Claude QA: can Claude answer questions from page images as well as from text?

*2026-11-10.* Each long document (~245k text tokens) is given to Claude three ways: as text, as rendered page
images, and not at all. Every arm gets the same 120 questions, scored automatically. The question: how many fewer
input tokens can pages use before the answers get worse?

## The answer

* **Opus 5.5: pages at 2.2x fewer tokens lose nothing measurable**, at low and high effort, multi-step questions
  included; at 2.9x it is still lossless at high effort.
* **Sonnet 5.5: lossless at 2.2x at high effort**; at low effort the estimate is the same (-0.3 points) but the
  worst case is 2.2 points, just outside the margin. At 2.9x it starts to lose (-1.7 to -3.1 points).
* **Haiku 5.5 is not lossless at any density tested.** It loses multi-step accuracy already at 2.2x; for Haiku, text
  is the better input.
* **Past ~3x, single facts survive but connecting them fails** for every model: at 4.2x, two- and three-hop
  questions and counts across the document break down.
* **Cost:** at 2.2x, pages cost about half of reading text for Opus ($0.68 vs $1.34) and Sonnet ($0.37 vs $0.67),
  for the same answers.

![Accuracy change from reading page images instead of text, per model, effort and token saving](img/claude_qa_verdict.png)

*How to read it:* each cell compares page images with text on the same 3 documents and questions. The big number is
the accuracy change in points (negative = worse than text). "Worst" is the 95% bound. ✓ marks the pre-registered
verdict: the worst case is within 2 points. Darker cells lose more. Numbers: [A1](#a1-opus-55),
[A2](#a2-sonnet-55) and [A3](#a3-haiku-55).

## What the pages look like

![The same passage at the three densities, actual pixels, with the vision-token grid](img/claude_qa_pages.png)

*Crops of the real pages at actual pixels.* Pages are 1932 x 1932 px, the largest these models read without
downscaling (4,764 tokens each; checked for all three). As text, the same passage costs one token per ~2.9 characters. At 2.2x a vision
token carries ~6.7 characters, at 4.2x ~13.3 (font size 13.9 px down to 9.9 px). Phase 1's densest readable setting
(29.4 px² per character, 3% transcription error) would be 7.8x; it fails this test outright ([A5](#a5-earlier-exploratory-runs)).

## Where it breaks

![Accuracy per question type, each model at high effort](img/claude_qa_breakdown.png)

*How to read it:* columns are question types, one block of rows per model: the grey row is the model reading text,
the blue rows the same model reading pages. Shading is points lost against the grey row.

* **Lookups and sums survive the longest.** Opus keeps 98-100% on them even at 4.2x.
* **Chains and counts fail first.** At 4.2x, Opus gets 67% of three-hop and 28% of count questions right. Wrong two-hop
  answers are usually another club's boat count: the facts are read, the link between them is lost.
* **Sonnet sits between them:** its first loss is counting (89% at 2.2x, 72% at 2.9x); at 4.2x three-hop is at 42%.
* **Haiku fails the same way at lower densities:** three-hop falls to 67% already at 2.2x, and at 4.2x even single
  lookups drop to 56-66%.

## What it costs

![List-price cost per call and accuracy, per model and input form](img/claude_qa_cost.png)

*How to read it:* one row per option; the bar is the list-price cost of one call (a ~245k-token document plus 120
questions), the dot its accuracy with a 95% interval. Costs are list prices as if nothing came from the prompt
cache (see [A4](#a4-cost)); these calls drew on a subscription quota, so they are not amounts billed.

* **Opus:** 2.2x pages cost $0.68 and 2.9x $0.55 against $1.34 for text, with no measurable loss.
* **Sonnet:** 2.2x pages cost $0.37 against $0.67 for text, with no measurable loss at high effort: the cheapest
  lossless option of the three models.
* **Haiku:** reading text costs $0.18 (98% right). Its 2.9x pages drop below Haiku's 100k-token price tier ($0.023)
  but lose 11 points.

## How the test works

* **Calls:** headless Claude Code (`claude -p`, no tools, isolated environment; `vbpe/claude_eval.py`). The image arm
  cannot see the source text. Input tokens come from the CLI's usage report.
* **Documents:** 3 non-overlapping ~700k-character documents (held-out WikiText-103 test and validation text), each
  with invented facts inserted between paragraphs.
* **Pages:** the same document string rendered as DejaVu Sans Latin (Phase 1's renderer) at 117.6, 88.2 and 58.8 px²
  per character.
* **Questions (120 per document, exact numeric answers):**
  * 40 invented facts ("How many tonnes of granite did Pavo Dekbry use to rebuild the bridge at Rimosk?"), which only
    the document can answer;
  * 40 numbers in real sentences ("Find the sentence that begins '...'. What number does it contain?");
  * 40 multi-step questions over invented facts: 10 two-hop (person -> chairs a rowing club -> how many boats),
    8 three-hop (person -> club -> town of its boathouse -> lamp posts), 10 sums, 6 largest-of-three, 6 counts across
    the whole document. A chain's facts are at least 1/8 of the document apart.
* **Pre-registered** (committed before any call): each image answer is paired with the same document's text answer
  and pooled over the 3 documents (360 pairs). Paired difference with Newcombe's score interval; **lossless = the
  upper 95% bound of (text - image) is below 2 points**. The Sonnet and Haiku suites were registered the same way
  before their runs.
* **Reproduce:** `experiments/claude_qa_render.py` (one run per document, density and effort; see `results/claude_qa/
  summary.md` for the commands), `experiments/analyse_claude_qa.py results/claude_qa/qav2_d*`, figures
  `scripts/make_claude_qa_figures.py`.

## Limits

* **Ceiling:** text scores 98-100%, so losses below ~1 point cannot show.
* **Length:** ~245k text tokens per document. Through the CLI, a request with more than ~175k image tokens silently
  loses its earliest pages (seen at ~345k: ~145-157k delivered; not bytes or image count; not documented). Longer
  documents need a direct API test.
* **Scope:** three models (Opus, Sonnet and Haiku 5.5), one font and layout, English Wikipedia.
* **Questions:** planted facts and numbers with exact answers, not open-ended understanding or summaries.

---

## Data

### A1. Opus 5.5

Pooled over 3 documents, 360 paired questions per row (`results/claude_qa/analysis_v2/summary.md` has every question
type).

| effort | fewer tokens than text | text | image | text - image (95% CI) | multi-step: text / image | verdict |
|---|---|---|---|---|---|---|
| low | 2.2x | 100.0% | 99.7% | +0.3 (-0.8 to +1.6) | 100% / 100% | **lossless** |
| high | 2.2x | 99.7% | 99.7% | 0.0 (-1.3 to +1.3) | 99% / 100% | **lossless** |
| low | 2.9x | 100.0% | 98.9% | +1.1 (-0.1 to +2.8) | 100% / 98% | not shown below 2 points |
| high | 2.9x | 99.7% | 100.0% | -0.3 (-1.6 to +0.8) | 99% / 100% | **lossless** |
| low | 4.2x | 100.0% | 84.7% | +15.3 (+11.8 to +19.4) | 100% / 62.5% | loss |
| high | 4.2x | 99.7% | 91.7% | +8.1 (+5.5 to +11.3) | 99% / 79% | loss |

No document: 5.0% (low) and 6.9% (high), from years and other guessable numbers. Pre-registered verdicts:
(1) lossless at 2.2x: PASS at both efforts; (2) a loss on multi-step questions at ~4x, low effort: PASS (+37.5
points, CI +28.8 to +46.4); (3) high effort halves the gap at 4.2x and makes 2.9x lossless.

### A2. Sonnet 5.5

`analysis_v2_sonnet/summary.md`; no call flagged by the engagement check.

| effort | fewer tokens than text | text | image | text - image (95% CI) | multi-step: text / image | verdict |
|---|---|---|---|---|---|---|
| high | 2.2x | 99.7% | 99.4% | +0.3 (-1.1 to +1.7) | 99% / 98% | **lossless** |
| low | 2.2x | 98.9% | 98.6% | +0.3 (-1.6 to +2.2) | 97.5% / 96.7% | not shown below 2 points |
| high | 2.9x | 99.7% | 98.1% | +1.7 (0.0 to +3.7) | 99% / 95% | not shown below 2 points |
| low | 2.9x | 98.9% | 95.8% | +3.1 (+0.8 to +5.7) | 97.5% / 90% | loss |
| high | 4.2x | 99.7% | 85.8% | +13.9 (+10.6 to +17.8) | 99% / 64% | loss |
| low | 4.2x | 98.9% | 77.2% | +21.7 (+17.5 to +26.2) | 97.5% / 51% | loss |

No document: 1.9% at both efforts. Pre-registered verdicts: (1) lossless at 2.2x: PASS at high effort, not shown at
low effort (worst case 2.2 points); (2) a loss on multi-step questions at ~4x, low effort: PASS (+46.7 points, CI
+37.4 to +55.4); (3) high effort narrows the gap (4.2x: 21.7 -> 13.9 points; 2.9x: 3.1 -> 1.7).

### A3. Haiku 5.5

High effort (`analysis_v2_haiku/summary.md`):

| fewer tokens | text | image | text - image (95% CI) | lookups: text / image | multi-step: text / image |
|---|---|---|---|---|---|
| 2.2x | 98.1% | 92.2% | +5.8 (+3.0 to +9.0) | 99.2% / 99.2% | 95.8% / 78.3% |
| 2.9x | 98.1% | 86.9% | +11.1 (+7.9 to +14.9) | 99.2% / 98.8% | 95.8% / 63.3% |
| 4.2x | 98.1% | 45.6% | +52.5 (+47.1 to +57.6) | 99.2% / 60.8% | 95.8% / 15.0% |

Verdict (1), lossless at 2.2x: **FAIL**. **Low effort is excluded** (decided after the run, from the model's own
output): Haiku answered the text arm with 400-800 output tokens, i.e. without reasoning (33-45% correct; one call
declined 80 of 120 questions: "I haven't worked those out ... I can go through them in batches"), but spent 8-15k
output tokens on the image arms, which therefore "won" by up to 49 points. That measures how much Haiku chooses to
think per input type, not how well it reads. Opus used ~3k output tokens on both arms at low effort and none of its
calls was flagged. `analyse_claude_qa.py` now flags unanswered questions and such phrases and marks the row invalid.

### A4. Cost

Per call, list price as if uncached (every input token written to the 5-minute prompt cache once, as the first call
of a document is), mean of the 3 documents. Recomputed from each call's logged tokens; this reproduces the logged
cost of all 63 uncached calls of the three models exactly.

| option | Opus low | Opus high | Sonnet low | Sonnet high | Haiku low | Haiku high |
|---|---|---|---|---|---|---|
| text (~247k tokens) | $1.29 | $1.34 | $0.66 | $0.67 | $0.16 | $0.18 |
| pages, 2.2x (~112k) | $0.63 | $0.68 | $0.34 | $0.37 | $0.10 | $0.13 |
| pages, 2.9x (~86k) | $0.51 | $0.55 | $0.29 | $0.34 | $0.016 | $0.023 |
| pages, 4.2x (~59k) | $0.37 | $0.50 | $0.20 | $0.40 | $0.008 | $0.012 |

Prices: Opus 5.5 $4 / $20 and Sonnet 5.5 $2 / $10 per million input / output tokens; Haiku 5.5 $0.10 / $0.50 for
prompts up to 100k tokens, $0.50 / $2.50 above (its 2.9x and 4.2x pages fall in the cheaper tier); cache writes
1.25x. High effort adds thinking tokens (Opus 4.2x pages: 9-12k output tokens vs
3-4k at low effort).

Total spend of all QA experiments, from the CLI's `total_cost_usd` (list price; some calls read their document from
the cache, so the logged figures are lower than the table above):

| experiment | calls (cached repeats excluded) | list-price cost |
|---|---|---|
| 500k chars, 80 questions, 4 densities | 5 | $2.27 |
| 781k tokens, 80 questions, small pages | 5 (one arm lost pages) | $6.14 |
| 783k tokens, 100 questions, first attempt (count questions leaked; discarded) | 3 | $4.49 |
| 783k tokens, 100 questions, 1932-px pages | 5 (one arm lost pages) | $6.17 |
| 700k chars, 100 questions | 5 | $3.15 |
| **v2 pre-registered, Opus 5.5: 3 documents x 2 efforts** | 30 | **$11.47** |
| v2, Haiku 5.5 | 30 | $1.24 |
| v2, Sonnet 5.5 | 30 | $10.05 |
| **Total** | 113 | **$44.99** |

Not counted: eight small probes (large-page and payload tests), under ~$0.35 together, and the first attempt of one
Sonnet call that timed out and was retried (the no-document arm at high effort on document 3; the retry spent 129,627
output tokens and $1.58 before answering, and the timed-out attempt's spend is not logged).

### A5. Earlier exploratory runs

Same harness before the pre-registered design; one document and one call per arm, so "no loss detected" rather than
lossless (with 100 questions and no disagreement, a gap up to 3.7 points cannot be excluded).

| document | questions | result |
|---|---|---|
| 500k chars (175k tokens), small pages | 80 simple | 7.8x fewer tokens (Phase 1's 3% CER density): 33-42% vs 100%; 4.3x: 85-95%; 2.2x and 1.5x: 100% |
| 2.3M chars (781k tokens), small pages | 80 simple | 4.5x: 97% / 93% (4 misses); 8.7x: 20% / 57%; the 4x arm lost pages |
| 2.3M chars (783k tokens), 1932-px pages | 100 incl. 20 multi-step | 4.4x: lookups 90-95%, multi-step 60% vs 80% from text; 8.6x fails; 4x lost pages |
| 700k chars (245k tokens), 1932-px pages | 100 incl. 20 multi-step | 2.2x and 1.5x: 100% on all; 4.2x: lookups 97%, multi-step 70% |

### A6. Problems found on the way

* The first count questions leaked: they counted fact types that the 40 needle questions enumerate, and the
  no-document arm got 3 of 3 by counting questions. Replaced with fact types no other question mentions (rows kept as
  `qa800kc_v1_countleak`).
* A sentence opening with a double quote broke the anchor lookup (fixed; earlier question sets unchanged).
* The first analysis used a bootstrap interval, which collapses to zero width when text and image never disagree;
  replaced with Newcombe's score interval before the v2 runs.
* Pages were silently dropped above ~175k image tokens (Limits); every run now checks input tokens against the
  image-token formula.
