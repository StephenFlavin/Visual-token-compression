# Claude QA: can Claude answer questions from page images as well as from text?

*2026-11-10.* Each long document (~240k tokens of text) is given to Claude three ways: as text, as rendered page
images at three densities, and not at all. Every arm gets the same 120 questions, scored automatically. The question:
how many fewer input tokens can pages use before the answers get worse? Three models were tested: Opus 5.5,
Sonnet 5.5 and Haiku 5.5.

## The answer

* **Opus 5.5: pages with 2.2x fewer tokens lose nothing measurable, at low and high effort.** On multi-step questions
  alone no loss was seen either (0.0 and +0.8 points), but with 120 of them a loss up to about 3 points cannot be
  excluded. At 2.9x Opus is still lossless at high effort.
* **Sonnet 5.5: lossless at 2.2x at high effort.** At low effort the estimate is the same (-0.3 points) but the worst
  case is 2.2 points, just outside the margin. At 2.9x it is no longer shown lossless (-1.7 points at high effort,
  borderline; -3.1 at low effort, a measurable loss).
* **Haiku 5.5 is not lossless at any density tested.** It loses multi-step accuracy already at 2.2x; for Haiku, text
  is the better input.
* **Denser pages break the links between facts first.** Past ~3x, Opus and Sonnet still find single facts but lose
  chains and counts (Opus at 4.2x, high effort: three-hop 67%, counts 28%). Haiku loses single facts too (56-66%).
* **Cost:** at list price, 2.2x pages cost about half as much as text for the same answers (Opus $0.68 vs $1.34,
  Sonnet $0.37 vs $0.67 per call at high effort; not amounts billed).

**Terms used throughout**

* **Nx fewer tokens:** input tokens of the text call ÷ input tokens of the page call (both include ~6k tokens of
  instructions and questions): measured 2.19x, 2.87x and 4.15x, shown as 2.2x, 2.9x and 4.2x. For the document alone
  the savings are 2.26x, 3.0x and 4.5x.
* **px² per character:** page area ÷ characters on the page; lower means smaller type and denser pages
  (117.6, 88.2 and 58.8 px² here).
* **Effort:** the Claude Code `--effort low|high` setting, which controls how much the model reasons before answering.
* **Lossless:** the pre-registered test. The upper 95% bound of the accuracy loss (text − image) is below 2 points:
  no loss larger than 2 points can be shown. This is a non-inferiority margin, not a claim of zero loss.
* **Worst:** that bound, the losing end of the 95% interval.
* **Loss:** the 95% interval excludes 0 and the exact McNemar test rejects (p < 0.05).
* **Lookups:** single-fact questions (invented facts and numbers in quoted sentences). **Multi-step:** two-hop,
  three-hop, sums, largest-of-three and counts across the document. (In the result files: `needle` and `number` are
  the lookups, `complex` the multi-step questions.)
* **Phase 1:** this repo's earlier transcription test, where Claude read pages of natural text aloud
  ([RESULTS.md](RESULTS.md)).

![Accuracy change from reading page images instead of text, per model, effort and token saving](img/claude_qa_verdict.png)

*How to read it:* each cell compares page images with text on the same 3 documents and questions. The big number is
the accuracy change in points (image − text: negative = worse than text); "worst" is the losing end of the 95%
interval; ✓ marks the pre-registered lossless verdict. Colour runs from green (no loss) through yellow to red (large loss). Numbers: [A1](#a1-opus-55),
[A2](#a2-sonnet-55) and [A3](#a3-haiku-55).

## How fast it falls

![Accuracy change against token saving, one line per model](img/claude_qa_curve.png)

*How to read it:* each line starts at 0 (the model reading text) and shows the accuracy change as the pages get denser,
at high effort; the dashed band is the 2-point lossless margin, and the background runs from green (no loss)
to red (large loss).

* **By point estimate, Opus and Sonnet stay within 2 points up to 2.9x;** by the pre-registered worst case, Opus is
  lossless to 2.9x and Sonnet to 2.2x. Haiku leaves the band at the first step.
* **Multi-step questions lose about 2.5 times as many points as the overall score** for Opus and Sonnet (1.5 times for
  Haiku): at 4.2x, -20, -35 and -81 points.

## What the pages look like

![The top of the first test page at the three densities, actual pixels, with the vision-token grid](img/claude_qa_pages.png)

*The top of the first test page of document 1, exactly as sent, at actual pixels.* Pages are 1932 px wide and 1876 px
tall (4,626 tokens each; 1932 px is the largest side these models read without downscaling, checked for all three);
23-24, 18 and 12 pages per document at the three densities. A vision token carries ~6.7 characters at 2.2x and ~13.3
at 4.2x; the font shrinks from 13.8 px to 9.8 px. As text, Claude's tokenizer takes ~2.9 characters per token.
Phase 1's densest readable setting (29.4 px² per character, 3% transcription error) would be ~8.5x on these pages; in
the exploratory runs that density failed outright ([A5](#a5-earlier-exploratory-runs)).

## Where it breaks

![Accuracy per question type, each model at high effort](img/claude_qa_breakdown.png)

*How to read it:* columns are question types, one block of rows per model: the grey row is the model reading text,
the coloured rows the same model reading pages, from green (no loss against the grey row) to red. A small "+1 q" marks the
four cells where pages got 1-2 more questions right than text; that is noise, not a gain (see the counts bullet).

* **Lookups and sums survive the longest.** Opus keeps 98-100% on them even at 4.2x. (Sums and largest-of-three add
  up facts that are also asked as separate questions in the same call, so they partly reuse that work.)
* **Chains and counts fail first.** At 4.2x, Opus gets 67% of three-hop and 28% of count questions right. Wrong
  two-hop answers are usually another club's boat count: the facts are read, the link between them is lost. Opus's
  wrong counts are all too low (10 for 12, 5 for 7): occurrences are missed. Text is not perfect
  on counts either: Opus and Sonnet each answered 11 for 12 once from text (17 of 18), which is why Opus's 2.2x and
  2.9x pages, with 18 of 18, appear to beat text by one question.
* **Sonnet sits between Opus and Haiku:** its first loss is counting (89% at 2.2x, 72% at 2.9x); at 4.2x three-hop is
  at 42%.
* **Haiku fails the same way with fewer tokens saved:** three-hop falls to 67% already at 2.2x, and at 4.2x even single
  lookups drop to 56-66%.

![Single-fact lookups by where the fact sits in the document](img/claude_qa_position.png)

*How to read it:* only single-fact lookups, grouped by the fifth of the document their fact is in; the dashed line is
the models reading text. 2.2x is left out (near 100% everywhere).

* **No depth effect is visible:** a fact on the last pages is found about as often as one on the first, at both
  densities and for every model (Haiku at 4.2x is low everywhere, 53-65%, not low at the end). With 42-52 questions
  per fifth only large trends would show.
* **So the loss is about connecting facts, not reaching them.** Counting all questions makes accuracy appear to fall
  towards the end, but that is because multi-step questions, placed by their latest fact, cluster there.

## Does effort help?

![Accuracy change at low and high effort](img/claude_qa_effort.png)

*How to read it:* hollow dots are low effort, filled dots high effort, for the same model, pages and questions.

* **At 2.2x effort changes little for Opus and Sonnet:** their estimates are within 0.3 points of text at both
  efforts, but only Opus (both efforts) and Sonnet at high effort are shown lossless (Sonnet at low effort: worst
  case 2.2 points).
* **Once the pages are dense, high effort narrows the loss:** by about half for Opus (4.2x: -15.3 -> -8.1 points;
  multi-step -37.5 -> -20.0) and by a quarter to a third for Sonnet (-21.7 -> -13.9; multi-step -46.7 -> -35.0). It
  also makes Opus's 2.9x lossless.
* High effort costs more output tokens: see the cost figure below and [A4](#a4-cost).

## What it costs

![List-price cost per call and accuracy, per model, input form and effort](img/claude_qa_cost.png)

*How to read it:* one row per option, low effort (light bar, hollow dot) above high effort (solid bar, filled dot).
The bar is the list-price cost of one call (a ~240k-token document plus 120 questions), the dot its accuracy with a
95% interval; ✓ marks no measurable loss against the same model reading text at that effort. Costs are list prices
with every input token priced as a cache write (1.25x the base input price), as on the first call of a document; a call
without caching would pay ~20% less for input, and the ratios between options barely change ([A4](#a4-cost)). These
calls drew on a subscription quota, so they are not amounts billed.

* **Opus:** 2.2x pages cost $0.63 (low effort) or $0.68 (high) against $1.29-1.34 for text, with no measurable loss at
  either effort; 2.9x pages at high effort ($0.55) are lossless too.
* **Sonnet:** 2.2x pages at high effort cost $0.37 against $0.67 for text, with no measurable loss: the cheapest
  lossless option of the three models. At low effort ($0.34) the worst case is just outside the margin.
* **Effort is cheap where it is not needed and worth it where it is:** for Opus and Sonnet, high effort adds 3-9% to
  the cost of text and 2.2x calls; on 4.2x pages it adds 37% (Opus) to 96% (Sonnet), and buys back 7-8 points.
* **Haiku:** reading text at high effort costs $0.18 (98% right). Its 2.9x and 4.2x pages fall into Haiku's cheaper
  tier for prompts up to 100k tokens ($0.008-0.023) but lose 11-53 points; at low effort it barely reasons over text
  (39%, marked *).

## Where this could be used

The result supports one pattern: **large amounts of reference text that the model reads and looks things up in, but
does not need to quote exactly or reason across as a whole.** Only English prose was tested; the uses below are
candidates to test, ordered from closest to the test to furthest from it.

* **RAG with more context per call.** Retrieved passages rendered at 2.2x let Opus or Sonnet (high effort) see about
  twice as many passages for the same tokens with no measurable loss, or the same passages for about half the price. A
  hybrid fits the result better still: send the top few passages as text and the long tail as pages. The tail is there
  for lookups, which survive denser pages (Opus 98% and Sonnet 97% at 4.2x).
* **Deep research: tested, and it holds at 2.2x** ([below](#tested-deep-research-sources)).
* **Long-document Q&A** (reports, contracts, manuals, filings). The test is closest to this. At 2.2x, Opus and Sonnet
  answer as well from pages as from text, at about half the cost.
* **Context compaction: tested, and it is the strongest result so far.** After `/compact`, Claude Code keeps a
  summary and the path of the raw transcript; on 3 long sessions Opus then answered 37% of the questions about what
  it had read. Adding an archive of the transcript as page images with an index (a PreCompact hook renders it; a
  SessionStart hook points to it) brought it to 100%, 63 points better and 31% cheaper than normal compaction, and
  2.2x cheaper than the same archive as text at the same accuracy. Details:
  [results/compaction](../results/compaction/README.md).
* **Coding agents: tested, no accuracy lost but no money saved** ([below](#tested-coding-agents)). Claude Code with
  file contents and command output shown as pages at 2.9x solved every SWE-bench task it solved from text, but
  coding spends little on reading, so the total cost did not fall.

### Tested: deep-research sources

A research orchestrator usually holds many fetched sources, or drops them and keeps notes. The test
([`experiments/claude_qa_research.py`](../experiments/claude_qa_research.py), pre-registered) used 3 corpora of
34-50 real Wikipedia articles each (~270k tokens of text), each article a labelled source with invented facts
planted in it, and 120 questions per corpus: 60 lookups and 60 that need several sources (which source says this,
two sources that disagree, chains across sources, how many sources mention something, totals across sources).
Opus 5.5, high effort, one call per arm and corpus, 360 paired questions:

| what the orchestrator gets | input tokens | accuracy | vs text (95% CI) | list price per corpus |
|---|---|---|---|---|
| all sources as text | 273k | 99.4% | (reference) | $1.48 |
| **all sources as 2.2x pages** | **122k** | **99.7%** | **+0.3 (−0.8 to +1.6): lossless** | **$0.77** |
| notes per source + the pages | 164k | 99.4% | 0.0 (−1.3 to +1.3): lossless | $3.27 (notes $2.33) |
| notes per source only | 46k | 70.3% | −29.2 (−34.1 to −24.5) | $2.70 (notes $2.33) |
| nothing | 5k | 6.9% | −92.5 | $0.07 |

* **Pages carry the whole job at 2.2x:** cross-source questions scored 100% from pages (99.4% from text), and every
  question type was at 99-100%.
* **Notes alone, the usual way to save context, lose 29 points**, even though they ran over their 250-word limit
  (~380 words per source). They lose most on the questions a summary drops: numbers in passing sentences (52%) and
  how many sources mention something (28%).
* **Notes add nothing on top of the pages here** and cost more to write than the text call itself; they would pay off
  only if reused across many queries.
* Pre-registered verdicts: notes + pages lossless, PASS; notes lose at least 10 points, PASS; pages recover at least 10
  points over notes, PASS; pages lossless on lookups alone, not shown (0.0, worst −2.1 with 180 questions), though
  lossless on all 360 questions. Details in [`results/claude_qa/research/summary.md`](../results/claude_qa/research/summary.md).
* **Limits:** encyclopedia articles, not mixed web pages with tables; questions with exact answers, not a written
  report; Opus only. Page calls were slower (median 56 s against 42 s for text).

### Tested: coding agents

Hooks for Claude Code ([`experiments/readhook/vbhook/`](../experiments/readhook/vbhook/)) show file contents and long
command output as minified pages at 88.2 px² per character (2.9x for prose): minifiers for 40+ languages and output
kinds, edit-safe for code (spaces kept, indentation as `»`), and an image only when it costs fewer tokens than the text.
Opus 5.5 at high effort, text vs pages, same tasks:

| test | text | pages | details |
|---|---|---|---|
| refactor-for-performance smoke tests (6 tasks x 3) | 18/18 equivalent | 54/54 over 3 hook iterations | [results/readhook_smoke](../results/readhook_smoke/README.md) |
| SWE-bench Verified, 50 hardest Django tasks recent Opus models solve | 50/50 resolved, $5.78 | 50/50, $6.08 | [results/readhook_swe](../results/readhook_swe/README.md) |
| SWE-bench Multilingual, 29 Rust and Java tasks | 28/29, $4.83 | 29/29, $4.75 | [results/readhook_swe_ml](../results/readhook_swe_ml/README.md) |

* **No task lost**, and no failed edit in the final hook version: the model edits exactly from page images.
* **The images were 3.1x cheaper than the text they replaced**, but the total cost did not fall ($10.82 against
  $10.61 over the 79 tasks). File contents and command output are only ~17% of what a coding agent costs; output (46%)
  and the re-read starting context and conversation (37%) are untouched, and the image arm made more calls (mostly
  extra Reads and smaller Edits the hooks' notes induced). Full analysis, figures and the remaining levers:
  [SWE_IMAGES.md](SWE_IMAGES.md).

**Where not to use it**

* **Counting and aggregation over the whole input** ("how many errors are in this log?"): counts are the first thing
  to fail (Opus 28% at 4.2x), and even text is not perfect on them.
* **Chains of facts** (three-hop questions) beyond 2.2x.
* **Text to copy exactly** (quotations, hashes, long literals): the model sees pixels, not characters. Code edits
  worked in the coding tests above, but only because the pages keep spacing and show indentation explicitly.
* **Haiku 5.5**, which loses accuracy even at 2.2x.
* **Latency-sensitive calls:** pages save tokens, not time. Page calls took slightly longer than text calls (median
  Opus 2.2x: 28 s against 22 s at low effort, 42 s against 37 s at high).
* **Very long inputs through the CLI:** requests above ~175k image tokens may silently lose their earliest pages
  (Limits).

## How the test works

* **Calls:** headless Claude Code (`claude -p`, no tools, isolated environment; [`vbpe/claude_eval.py`](../vbpe/claude_eval.py)).
  One call per arm: the document (as text, or as page images in order), then all 120 questions. The system prompt
  asks for a number per question and a best guess when unsure, so there is no abstaining. The image arm cannot see
  the source text. Input tokens come from the CLI's usage report.
* **Documents:** 3 non-overlapping documents of ~700k characters (held-out WikiText-103 test and validation text;
  ~240k tokens each; the text call is ~243-248k tokens with the questions), with invented facts inserted between
  paragraphs.
* **Pages:** the same document string rendered as DejaVu Sans Latin (Phase 1's renderer) at 117.6, 88.2 and 58.8 px²
  per character (see "What the pages look like").
* **Questions (120 per document, exact numeric answers):**
  * 40 invented facts ("How many tonnes of granite did Pavo Dekbry use to rebuild the bridge at Rimosk?"), which only
    the document can answer;
  * 40 numbers in real sentences ("Find the sentence that begins '...'. What number does it contain?");
  * 40 multi-step questions over invented facts: 10 two-hop (person -> chairs a rowing club -> how many boats),
    8 three-hop (person -> club -> town of its boathouse -> lamp posts), 10 sums, 6 largest-of-three, 6 counts across
    the whole document. A chain's facts are at least 1/8 of the document apart.
* **Scoring:** an answer is correct if it equals the expected number after removing commas.
* **Pairing:** each page call is paired, question by question, with the text call of the same document and effort (one
  text call per document and effort, shared by the three densities), pooled over the 3 documents: 360 pairs.
* **Pre-registered** (in [`results/claude_qa/summary.md`](../results/claude_qa/summary.md), committed before any call,
  the same for each model): (1) lossless at 2.2x at each effort, by Newcombe's paired score interval with a 2-point
  margin; (2) a loss on multi-step questions at ~4x at low effort; (3) the effect of effort, reported without a
  threshold. The Sonnet and Haiku suites were registered the same way before their runs.
* **Engagement check** (added after the Haiku run): a call that leaves questions unanswered or says it did not look
  things up is not a reading measurement, and its rows are marked "not comparable".
* **Reproduce:** [`experiments/claude_qa_render.py`](../experiments/claude_qa_render.py) (one run per document,
  density, effort and model; the commands are in `results/claude_qa/summary.md`), then
  [`experiments/analyse_claude_qa.py`](../experiments/analyse_claude_qa.py):
  ```bash
  python3 experiments/analyse_claude_qa.py results/claude_qa/qav2_d*                                              # Opus
  python3 experiments/analyse_claude_qa.py results/claude_qa/qav2s_d* --out results/claude_qa/analysis_v2_sonnet  # Sonnet
  python3 experiments/analyse_claude_qa.py results/claude_qa/qav2h_d* --out results/claude_qa/analysis_v2_haiku   # Haiku
  python3 scripts/make_claude_qa_figures.py                                                                       # figures
  ```

## Limits

* **Ceiling:** text scores 98-100%, so losses below ~1 point cannot show.
* **One call per arm and document:** the intervals cover question-to-question variation, not run-to-run variation of
  the model, and each document's text call is shared by its three densities.
* **Length:** ~240k tokens of text per document. Through the CLI, a request with many image tokens silently loses
  its earliest pages: ~175k image tokens arrived whole, ~345k arrived as ~145-157k; nothing in between was tested,
  and the vision docs do not mention it. Longer documents need a direct API test.
* **Scope:** three models (Opus, Sonnet and Haiku 5.5), one font and layout, English Wikipedia.
* **Questions:** planted facts and numbers with exact answers, not open-ended understanding or summaries.

---

## Data

The tables use the same sign as the figures: **image − text** in accuracy points, negative = pages worse.

### A1. Opus 5.5

[analysis_v2/summary.md](../results/claude_qa/analysis_v2/summary.md) has every question type; no call flagged by the
engagement check.

| effort | fewer tokens | text | image | image − text (95% CI) | lookups: text / image | multi-step: text / image | verdict |
|---|---|---|---|---|---|---|---|
| high | 2.2x (2.19x) | 99.7% | 99.7% | 0.0 (−1.3 to +1.3) | 100.0% / 99.6% | 99.2% / 100.0% | **lossless** |
| high | 2.9x (2.87x) | 99.7% | 100.0% | +0.3 (−0.8 to +1.6) | 100.0% / 100.0% | 99.2% / 100.0% | **lossless** |
| high | 4.2x (4.15x) | 99.7% | 91.7% | −8.1 (−11.3 to −5.5) | 100.0% / 97.9% | 99.2% / 79.2% | loss |
| low | 2.2x (2.19x) | 100.0% | 99.7% | −0.3 (−1.6 to +0.8) | 100.0% / 99.6% | 100.0% / 100.0% | **lossless** |
| low | 2.9x (2.87x) | 100.0% | 98.9% | −1.1 (−2.8 to +0.1) | 100.0% / 99.2% | 100.0% / 98.3% | not shown lossless |
| low | 4.2x (4.15x) | 100.0% | 84.7% | −15.3 (−19.4 to −11.8) | 100.0% / 95.8% | 100.0% / 62.5% | loss |

No document: 6.9% (high effort), 5.0% (low), from years and other guessable numbers. Pre-registered verdicts:
(1) lossless at 2.2x: **PASS** at both efforts; (2) a loss on multi-step questions at ~4x, low effort: **PASS**
(−37.5 points, CI −46.4 to −28.8); (3) effort: high effort narrows the 4.2x loss from −15.3 to −8.1 points and makes
2.9x lossless.

### A2. Sonnet 5.5

[analysis_v2_sonnet/summary.md](../results/claude_qa/analysis_v2_sonnet/summary.md); no call flagged by the engagement
check.

| effort | fewer tokens | text | image | image − text (95% CI) | lookups: text / image | multi-step: text / image | verdict |
|---|---|---|---|---|---|---|---|
| high | 2.2x (2.19x) | 99.7% | 99.4% | −0.3 (−1.7 to +1.1) | 100.0% / 100.0% | 99.2% / 98.3% | **lossless** |
| high | 2.9x (2.87x) | 99.7% | 98.1% | −1.7 (−3.7 to 0.0) | 100.0% / 99.6% | 99.2% / 95.0% | not shown lossless |
| high | 4.2x (4.15x) | 99.7% | 85.8% | −13.9 (−17.8 to −10.6) | 100.0% / 96.7% | 99.2% / 64.2% | loss |
| low | 2.2x (2.19x) | 98.9% | 98.6% | −0.3 (−2.2 to +1.6) | 99.6% / 99.6% | 97.5% / 96.7% | not shown lossless |
| low | 2.9x (2.87x) | 98.9% | 95.8% | −3.1 (−5.7 to −0.8) | 99.6% / 98.8% | 97.5% / 90.0% | loss |
| low | 4.2x (4.15x) | 98.9% | 77.2% | −21.7 (−26.2 to −17.5) | 99.6% / 90.4% | 97.5% / 50.8% | loss |

At 2.9x and high effort the interval only just reaches 0 (its upper end is −0.04 points) and the McNemar test does
not reject (p = 0.07): borderline, not a shown loss. No document: 1.9% at both efforts. Pre-registered verdicts:
(1) lossless at 2.2x: **PASS** at high effort, not shown at low effort (worst case −2.2 points); (2) a loss on
multi-step questions at ~4x, low effort: **PASS** (−46.7 points, CI −55.4 to −37.4); (3) effort: high effort narrows
the 4.2x loss from −21.7 to −13.9 points and the 2.9x loss from −3.1 to −1.7.

### A3. Haiku 5.5

[analysis_v2_haiku/summary.md](../results/claude_qa/analysis_v2_haiku/summary.md).

| effort | fewer tokens | text | image | image − text (95% CI) | lookups: text / image | multi-step: text / image | verdict |
|---|---|---|---|---|---|---|---|
| high | 2.2x (2.19x) | 98.1% | 92.2% | −5.8 (−9.0 to −3.0) | 99.2% / 99.2% | 95.8% / 78.3% | loss |
| high | 2.9x (2.87x) | 98.1% | 86.9% | −11.1 (−14.9 to −7.9) | 99.2% / 98.8% | 95.8% / 63.3% | loss |
| high | 4.2x (4.15x) | 98.1% | 45.6% | −52.5 (−57.6 to −47.1) | 99.2% / 60.8% | 95.8% / 15.0% | loss |
| low | 2.2x (2.19x) | 39.2% | 88.3% | +49.2 (+43.7 to +54.1) | 58.8% / 97.5% | 0.0% / 70.0% | not comparable* |
| low | 2.9x (2.87x) | 39.2% | 80.8% | +41.7 (+36.2 to +46.7) | 58.8% / 97.1% | 0.0% / 48.3% | not comparable* |
| low | 4.2x (4.15x) | 39.2% | 14.4% | −24.7 (−30.1 to −19.2) | 58.8% / 20.4% | 0.0% / 2.5% | not comparable* |

No document: 0.6% (high effort), 0.3% (low). Pre-registered verdicts: (1) lossless at 2.2x: **FAIL** at high effort
(−5.8 points, worst −9.0); not evaluable at low effort; (2) a loss on multi-step questions at ~4x, low effort: not
evaluable (at high effort: −80.8 points, CI −86.4 to −72.1); (3) effort: at low effort Haiku does not reason over
long text (below).

\* **Low effort is excluded** (decided after the run, from the model's own output). Haiku answered the text arm with
424-821 output tokens, i.e. without reasoning (33-45% correct; one call declined 80 of 120 questions: "I haven't
worked those out ... I can go through them in batches"), but spent 8-15k output tokens on its 2.2x and 2.9x page calls
(0.8-2.3k on the 4.2x pages), which therefore "won" by up to 49 points. That measures how much Haiku chooses to think
per input type, not how well it reads. Opus used ~3-4k output tokens on both arms at low effort, and no Opus or
Sonnet call was flagged.

![Output tokens per call, text vs pages, per model and effort](img/claude_qa_engagement.png)

*How to read it:* one mark per call on a log scale (squares: text, 3 per row; circles: pages, 9 per row: 3 densities
x 3 documents). The shaded row is Haiku at low effort, the excluded one: 424-821 output tokens on text against
0.8-15k on pages. Opus and Sonnet spend thousands of output tokens on both.

### A4. Cost

Per call, list price with every input token priced as a cache write (1.25x the base input price, as on the first call
of a document), mean of the 3 documents. Recomputed from each call's logged tokens; this reproduces the logged cost of
all 63 uncached calls of the three models exactly.

| option | Opus low | Opus high | Sonnet low | Sonnet high | Haiku low | Haiku high |
|---|---|---|---|---|---|---|
| text (~246k input tokens) | $1.29 | $1.34 | $0.66 | $0.67 | $0.16 | $0.18 |
| pages, 2.2x (~112k) | $0.63 | $0.68 | $0.34 | $0.37 | $0.097 | $0.13 |
| pages, 2.9x (~86k) | $0.51 | $0.55 | $0.29 | $0.34 | $0.016 | $0.023 |
| pages, 4.2x (~59k) | $0.36 | $0.50 | $0.20 | $0.40 | $0.008 | $0.012 |

Prices: Opus 5.5 $4 / $20 and Sonnet 5.5 $2 / $10 per million input / output tokens; Haiku 5.5 $0.10 / $0.50 for
prompts up to 100k tokens, $0.50 / $2.50 above (its 2.9x and 4.2x pages fall in the cheaper tier). High effort adds
thinking tokens (Opus 4.2x pages: 9-12k output tokens against ~3-4k at low effort).

Total spend of all QA experiments, from the CLI's `total_cost_usd` (list price; some calls read their document from
the prompt cache, so logged figures can be lower than the table above):

| experiment | calls (cached repeats excluded) | list-price cost |
|---|---|---|
| 500k chars, 80 questions, 4 densities | 6 | $2.27 |
| 781k tokens, 80 questions, small pages | 5 (one arm lost pages) | $6.14 |
| 783k tokens, 100 questions, first attempt (count questions leaked; discarded) | 3 | $4.49 |
| 783k tokens, 100 questions, 1932-px pages | 5 (one arm lost pages) | $6.18 |
| 700k chars, 100 questions | 5 | $3.15 |
| **v2 pre-registered, Opus 5.5: 3 documents x 2 efforts** | 30 | **$11.48** |
| v2, Haiku 5.5 | 30 | $1.24 |
| v2, Sonnet 5.5 | 30 | $10.05 |
| Deep-research variant, Opus 5.5: 120 note calls + 15 arms (55 note calls failed once and were retried) | 135 | $17.89 |
| **Total** | 249 | **$62.89** |

Not counted: eight small probes (large-page and payload tests), under ~$0.35 together, and the first attempt of one
Sonnet call that timed out after 30 minutes and was retried: the no-document arm at high effort on document 3. The
retry used 114,591 input and 129,627 output tokens (the CLI apparently ran several internal turns; every other
no-document call used ~6k input tokens) and cost $1.58, included above; the timed-out attempt's spend is not logged.

### A5. Earlier exploratory runs

Same harness before the pre-registered design; one document and one call per arm, so "no loss detected" rather than
lossless (with 100 questions and no disagreement, a gap up to 3.7 points cannot be excluded). Densities are given as
token savings, like the rest of this document; the page sizes differ from v2, so the same px² per character gives a
slightly different saving.

| document | questions | result |
|---|---|---|
| 500k chars (175k tokens), small pages | 80 simple | 7.8x fewer tokens (Phase 1's 3% CER density): 33-42% vs 100%; 4.3x: 85-95%; 2.2x and 1.5x: 100% |
| 2.3M chars (781k tokens), small pages | 80 simple | 4.5x: 97% / 93% (4 misses); 8.7x: 20% / 57%; the 117.6 px²/char arm (~2.3x fewer tokens) lost pages |
| 2.3M chars (783k tokens), 1932-px pages | 100 incl. 20 multi-step | 4.4x: lookups 90-95%, multi-step 60% vs 80% from text; 8.6x fails; the 117.6 px²/char arm lost pages |
| 700k chars (245k tokens), 1932-px pages | 100 incl. 20 multi-step | 2.2x and 1.5x: 100% on all; 4.2x: lookups 97%, multi-step 70% |

### A6. Problems found on the way

* The first count questions leaked: they counted fact types that the 40 invented-fact questions enumerate, and the
  no-document arm got 3 of 3 by counting questions. Replaced with fact types no other question mentions (rows kept as
  `qa800kc_v1_countleak`).
* A sentence opening with a double quote broke the anchor lookup (fixed; earlier question sets unchanged).
* The first analysis used a bootstrap interval, which collapses to zero width when text and image never disagree;
  replaced with Newcombe's score interval before the v2 runs.
* Pages were silently dropped somewhere between ~175k and ~345k image tokens per request (Limits); every run now
  checks input tokens against the image-token formula.
* Haiku at low effort did not engage with the text (A3); the engagement check was added after that run.
