# Test A: pruned text against page images (Claude QA)

*2026-11-12.* Pre-registered (commit 4fb07a5, before any scored call; the pre-registration is unchanged below the
results). Design draft: [docs/background/REVIEW_2026-11-11.md](../../../docs/background/REVIEW_2026-11-11.md), "Test A". It extends the v2
test in [docs/claude/CLAUDE_QA.md](../../../docs/claude/CLAUDE_QA.md) (A1, A4).

## Verdict

**Pruning the text does not replace pages, and it does not stack with them.** At the same token count as 2.2x pages,
pruned text is 11.7 points worse (86.0% against 97.7%), and it is worse on the original 120 questions too (−3.6
points, where the pages lose nothing). Pruned text drawn as pages (3.4x fewer tokens) loses 9.4 points against text:
no better than pages alone at 4.1x, and clearly worse than pages at 2.85x. The loss sits where the review expected:
LLMLingua-2 deletes the small words ("not", "all but", "was ... by"), and Claude then answers with the other number
in the sentence. Lookups survive pruning almost untouched.

All 24 calls (3 documents x 8 arms) completed; none was refused, cut off, flagged by the engagement check, or lost
pages. Opus 5.5, effort high, 160 questions per call, pooled over 3 documents (480 paired questions per arm; 360
original, 120 small-word).

| arm | input tokens (mean) | fewer than text | all 480 | original 120 | small-word | arm − text, all (95% CI) | original | small-word | verdict (all) |
|---|---|---|---|---|---|---|---|---|---|
| text | 248,971 | 1x | 100.0% | 100.0% | 100.0% | (reference) | | | |
| pages 117.6 px²/char | 114,188 | 2.18x | 97.7% | 100.0% | 90.8% | −2.3 (−4.1 to −1.0) | 0.0 (−1.1 to +1.1) | −9.2 (−15.7 to −4.1) | loss (small-word only) |
| **pruned text, matched to those pages** | 115,114 | 2.16x | **86.0%** | 96.4% | **55.0%** | **−14.0 (−17.3 to −11.0)** | −3.6 (−6.1 to −1.8) | −45.0 (−53.9 to −35.8) | loss |
| lightly pruned text | 168,210 | 1.48x | 96.5% | 99.4% | 87.5% | −3.5 (−5.6 to −2.0) | −0.6 (−2.0 to +0.6) | −12.5 (−19.6 to −6.8) | loss |
| **lightly pruned text as pages** | 73,812 | 3.37x | **90.6%** | 97.8% | 69.2% | **−9.4 (−12.3 to −6.9)** | −2.2 (−4.3 to −0.7) | −30.8 (−39.6 to −22.7) | loss |
| pages 88.2 px²/char | 87,445 | 2.85x | 99.0% | 99.2% | 98.3% | −1.0 (−2.4 to 0.0) | −0.8 (−2.4 to +0.4) | −1.7 (−5.9 to +1.7) | not shown lossless (borderline) |
| pages 58.8 px²/char | 60,701 | 4.10x | 90.6% | 93.3% | 82.5% | −9.4 (−12.3 to −6.9) | −6.7 (−9.7 to −4.3) | −17.5 (−25.3 to −11.0) | loss |
| no document | 7,181 | - | 5.0% | 6.7% | 0.0% | −95.0 | | | |

"Loss": the interval excludes 0 and the exact McNemar test rejects (p < 0.001 for every "loss" above). Text never got a
question wrong, so every difference is text-only errors. Real token ratios per document: matched pruned text used
0.4-1.2% more input tokens than the 117.6 pages (within the 5% aim); light pruning 1.48-1.49x; pruned pages
3.34-3.40x. Tables per question type: [analysis/summary.md](analysis/summary.md).

### Predictions

* **P1, pruned text at 2.2x is not lossless and loses most on small-word and sentence-anchor questions: PASS.** −14.0
  points (worst −17.3). Losses by kind: small-word −45.0, number −7.5, complex −3.3, needle 0.0.
* **P2, pruned pages (3.4x) lossless on the original 120: FAIL.** −2.2 points (−4.3 to −0.7, McNemar p = 0.008): a
  measurable loss, mostly number questions (−5.8; their quoted openings are no longer in the document).
* **P3, pruned pages beat pages alone at 4.2x and are not worse than pages at 2.9x: FAIL.** Against 58.8 pages: 0.0
  points on all 480 (−3.6 to +3.6; 37 questions each way). Against 88.2 pages: −8.3 points (−11.3 to −5.7). On the
  original 120 alone, pruned pages do beat 58.8 pages (+4.4, +1.5 to +7.7, p = 0.005) and are close to 88.2 pages
  (−1.4, −3.6 to +0.6), but the small-word questions cancel that.
* **P4, full wording reads the small words and no document does not: FAIL**, on one arm. Text 100% and no document
  0% as predicted, but the 117.6 pages scored 90.8% (below the 95% bar) because of one question type (below).

### What broke

* **"All but" never survived.** LLMLingua-2 dropped the phrase in every one of the 15 sentences at both rates; all
  three pruned arms scored 0% on it, answering the lure 44 times out of 45 ("Tobin Wentis planted 837 pear trees Vardra
  206 died" → 206, not 631).
* **Passives flip.** Heavy pruning keeps "paid" and drops "was" and, in 12 of 15 sentences, "by": "Osric Falpel paid 531 crowns Floris
  Tisquen Floris Tisquen 194 by Anselm Gavgav". Pruned text 2.2x and pruned pages scored 0% on passives; light pruning
  kept "by" in all 15 and scored 100% as text, but 0% as pages.
* **Negation:** the "not" of "did not receive" was dropped 15 times in 15 at 2.2x and 14 in 15 at 1.5x ("museum
  Moroth 340 maps Kasimir Bryvar received 369"). Claude often still picked the number after "received" (all 15 right
  with light pruning), but 10 of 15 were wrong at 2.2x.
* **Answering the lure** (the number that is right under the wrong reading): 42% of small-word answers for pruned
  text at 2.2x, 30% for pruned pages, 12% for light pruning, 0-8% for pages and text. The errors are misreadings, not
  failed lookups.
* **Pruning and pages compound.** The same lightly pruned document scored 87.5% on small-word questions as text and
  69.2% as pages: the pages' own loss adds to the pruning loss.
* **Sentence-anchor questions** (find the sentence that begins "..."): no quoted opening survived whole (on average
  27% of its words survive at 2.2x, 48% at 1.5x), yet Claude still found 92.5% of them at 2.2x by partial match.
* **Pages at 117.6 px²/char flipped the passive questions in two of three calls.** In documents 1 and 2 all 5 passive
  questions were answered with the lure; in document 0 all 5 were right; text, 88.2 pages and 58.8 pages got 13-15 of
  15 right. So the call, not the sentence, chose the reading: one decision repeated five times. This is the whole of
  the 117.6 arm's small-word loss (11 errors, 10 of them passives), and it shows that the small-word questions are
  correlated within a call, so their intervals are too narrow.
* **The free survival check was not an upper bound.** It counted a question lost if any needed word was dropped; Claude
  often recovered the meaning from what was left (light pruning: 45% of small-word questions kept every needed word,
  87.5% were answered right; "paid 531 crowns by Floris Tisquen" without "was" was enough).

Examples of pruned sentences that changed meaning (document 1; light pruning = 1.5x):

| type | original | pruned, 2.2x | pruned, 1.5x | answers (right / 2.2x / 1.5x text / 1.5x pages) |
|---|---|---|---|---|
| all but | Tobin Wentis planted 837 pear trees at Vardra, and all but 206 of them died in the first winter. | Tobin Wentis planted 837 pear trees Vardra 206 died | Tobin Wentis planted 837 pear trees at Vardra 206 died first winter | 631 / 206 / 206 / 206 |
| passive | At the horse fair in Keltor, Osric Falpel was paid 531 crowns by Floris Tisquen, and Floris Tisquen was paid 194 crowns by Anselm Gavgav. | Osric Falpel paid 531 crowns Floris Tisquen Floris Tisquen 194 by Anselm Gavgav | horse fair Keltor Osric Falpel paid 531 crowns by Floris Tisquen Floris Tisquen paid 194 crowns by Anselm Gavgav | 531 / 194 / 531 / 194 |
| negated count | Of the 928 coins that Elof Ellbak catalogued at Gavvar, 244 were not silver. | 928 coins 244 not silver | Of 928 coins Ellbak catalogued at Gavvar 244 not silver | 684 / 684 / 684 / 684 |
| negation | The museum at Moroth did not receive 340 maps from Kasimir Bryvar; it received 369. | museum Moroth 340 maps Kasimir Bryvar received 369 | museum at Moroth receive 340 maps from Kasimir Bryvar received 369. | 369 / 369 / 369 / 369 |

### Cost

List price, from each call's `total_cost_usd` (`spend.jsonl`): **$17.52 in all** ($0.39 for unscored calibration
calls, $17.13 for the 24 scored calls), under the $25 cap. Per document: text $1.37, pruned text at 2.2x $0.81 (more
than the matched pages, $0.71, because it thought longer: 8.8-15.1k output tokens against 6.8-7.1k), light pruning
$0.99, pruned pages $0.55, 88.2 pages $0.60, 58.8 pages $0.59, no document $0.09. Pruning itself ran on the CPU (~7
minutes per document). These calls drew on the subscription quota; they are not amounts billed.

### Not run, and limits

* One pruner (LLMLingua-2, off the shelf, MeetingBank-trained) with the deviations listed below; a pruner trained to
  keep function words, or one trained on this kind of text, was not tested (the review's 3090 follow-up only makes
  sense if one is).
* Not run: the base-size pruner (the large one was fast enough), low effort, other models, and a rate between 1.5x
  and 2.2x.
* One call per arm and document; small-word questions are correlated within a call (see the passive flip), so their
  intervals overstate the precision.
* Small-word questions are templated (5 sentences of each of 8 types per document) and planted; real text puts less
  weight on single function words.

---

# Pre-registration (committed before any scored call; unchanged)

## The question

Opus 5.5 answers questions from page images of a ~240k-token document with 2.2x fewer input tokens and no
measurable loss (2.9x at high effort; 4.2x loses). For a closed model the only way to compress the *text* is to send
fewer real tokens. Does extractive pruning (deleting words) (a) replace pages at the same token count, and (b) stack
with pages, so that pruned text drawn as pages goes further than pages alone?

## Design

**Documents.** The same 3 documents as `qav2_d0/d1/d2` (700k chars of held-out WikiText-103, paragraph offsets 0,
1150, 2480, seeds 0, 1, 2). Regenerated with the harness and checked: the original 120 questions per document
(wording, answers, anchors) are identical to the stored `questions.json`. To each document, 40 new **small-word facts**
are added (own random stream, so the original facts and questions do not change). The documents are 3.8k-3.9k chars
longer than in v2, so every arm is re-run; nothing is paired with the stored rows.

**Small-word questions** (40 per document, 5 of each type, 120 in all). The answer is right only if the function
words are read. Each sentence holds a second number, the *lure*: the answer under the wrong reading.

| type | planted sentence (example) | question | answer / lure | words it depends on |
|---|---|---|---|---|
| negation | The museum at Norlun did not receive 502 maps from Rasmus Pelmir; it received 853. | How many maps did the museum at Norlun receive from Rasmus Pelmir? | 853 / 502 | not |
| negated count | Of the 811 coins that Nora Sundlun catalogued at Quenmir, 154 were not silver. | How many ... were silver? | 657 / 154 | not |
| all but | Hedda Mormor planted 485 pear trees at Lenmor, and all but 80 of them died in the first winter. | How many ... died in the first winter? | 405 / 80 | all, but |
| passive | At the horse fair in X, A was paid n1 crowns by B, and B was paid n2 crowns by C. | How many crowns did B pay at the horse fair in X? | n1 / n2 | was, by |
| comparative | The choir of X has n1 members, d fewer (or more) than the choir of Y. | How many members does the choir of Y have? | n1 ± d / n1 ∓ d | fewer/more, than |
| before / after | Before the flood, the granary at X held n1 sacks of rye; after it, the granary held n2. | ... hold after (or before) the flood? | n2 / n1 | before, after |
| from / to | n1 crates of slate were shipped from X to Y, and n2 crates from Y to X. | How many crates of slate were shipped to X (or Y)? | n2 / n1 | from, to |
| instead of | The ferry at Dekith carried 434 passengers on its first crossing, instead of the 43 that Liesl Peltor had planned. | How many passengers did the ferry at Dekith carry on its first crossing? | 434 / 43 | instead, of |

Each document thus has 160 questions: 40 invented facts (`needle`), 40 numbers in real sentences (`number`, the
question quotes the sentence's first words), 40 multi-step (`complex`), 40 small-word (`small`).

**Pruner.** LLMLingua-2 (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`, pip `llmlingua` 0.2.2), on CPU. The
large model needed ~7 minutes per document (4 cores), so the base-size fallback was not needed. The model runs once
per document; its per-word keep probabilities are cached (`cache/`, git-ignored), and any keep rate is a cheap
re-threshold. The re-threshold reproduces `compress_prompt_llmlingua2` exactly (checked: identical output on a 20k-char
excerpt at two rates). Known deviations from off-the-shelf LLMLingua-2:

1. **Token counter stub.** LLMLingua-2 loads OpenAI's tokenizer file from a host this runner cannot reach. It is
   replaced by the pruner's own xlm-roberta tokenizer. LLMLingua-2 uses that count only to weight words in its keep
   threshold, so **the keep rate is in xlm-roberta tokens**, not Claude's or OpenAI's.
2. **Digits.** `force_reserve_digit=True` keeps digit *tokens*, but the library averages them with other tokens of
   the same word, so the tail of "4,512" (the word ",512") could be dropped. Here a word that contains a digit is
   always kept. Every answer number survives (checked below).
3. Chunks are joined with a space (the library joins them with nothing, gluing the last word of one 512-token chunk to
   the first of the next).
4. Kept as in the library: llmlingua 0.2.2 builds its input with BERT's `[CLS]`/`[SEP]`/`[PAD]` strings, which the
   xlm-roberta tokenizer maps to `<unk>`. Fixing it changes which words are kept; the test uses the package as shipped.

**Calibrating the rate against Claude's tokens.** Unscored calls (effort low, "reply OK") on a 60k-char excerpt of
document 0 measured Claude input tokens per character: full text 2.97, pruned at rate 0.37 2.58, pruned at rate 0.57
2.80 (pruned text costs more tokens per character). Prompt overhead (CLI + 160 questions) is 7,150 tokens. The rate per
document is then set so that:

* `ptext22`: the pruned document has as many Claude tokens as the 117.6 px²/char pages have image tokens (so the
  two calls have about the same input tokens; the aim is within ~5%);
* `ptext15`: the call's input tokens are the text call's / 1.5.

Rates (xlm-roberta-token keep share): ptext22 0.36 / 0.35 / 0.36, ptext15 0.56 / 0.55 / 0.55 for documents 0 / 1 / 2;
the pruned documents keep 38-39% and 62% of the characters. For documents 1 and 2 the rates may be re-calibrated from
document 0's *measured input tokens* (never from its accuracy) if the pilot misses the token target by more than 5%;
that would be logged as an amendment. The real ratios are reported from each call's usage.

**Arms** (Opus 5.5, `--effort high`, one call per arm and document, all 160 questions in one call, through
`vbpe/claude_eval.py` as in v2):

| arm | what Claude gets | estimated input tokens (doc 0) | vs text |
|---|---|---|---|
| `none` | no document | 7k | - |
| `text` | the full text | 247k | 1x |
| `page118` / `page88` / `page59` | pages at 117.6 / 88.2 / 58.8 px²/char, 1932-px (the v2 2.2x / 2.9x / 4.2x arms) | 114k / 87k / 61k | 2.2x / 2.8x / 4.1x |
| `ptext22` | pruned text, tokens matched to `page118` | 112k | 2.2x |
| `ptext15` | lightly pruned text | 166k | 1.5x |
| `ppage15` | `ptext15`'s document drawn as 117.6 px²/char pages | 74k | 3.3x |

**Prompts.** The same system prompt and question block as v2. The three pruned arms get one added sentence before the
questions: "Words have been removed from this document to shorten it." If a pruned arm is refused or cut off by a
safety classifier, that is the result for that arm, and the arm stops (no rewording, no retry with another prompt).

**Validity checks kept from the repo.** Page arms: input tokens must reach 97% of the image-token formula, else pages
were dropped and the arm is invalid. Engagement: a call with unanswered questions or a disengagement phrase is "not
comparable". API errors are not cached and are re-run with the same prompt.

**Analysis** (`claude_qa_prune.py analyse`). Every arm is paired question by question with the same document's `text`
call and pooled over the 3 documents (480 pairs; 360 for the original questions, 120 for small-word). Difference
arm − text with Newcombe's paired score interval (95%) and exact McNemar, as in CLAUDE_QA.md. **Lossless** = the
losing end of the interval is above −2 points. **Loss** = the interval excludes 0 and McNemar p < 0.05.

## Free check before spending: what survives the pruning

For each question, are the words it needs still in the pruned document? A question passes if its anchor sentences
keep their numbers and invented names (needle, complex, small), the full quoted opening (number questions), and
the function words its answer depends on (small-word). This is an upper bound on what pruning leaves answerable by
reading alone (Claude may still guess from what is left), pooled over the 3 documents:

| arm | chars kept | answer numbers kept | invented names whole | quoted opening whole (share of its words) | function words kept (small) | all needed words: needle / number / complex / small |
|---|---|---|---|---|---|---|
| ptext22 | 38-39% | 100% | 50% | 0% (27%) | 13% | 55% / 0% / 44% / 7% |
| ptext15 | 62% | 100% | 92% | 0% (48%) | 48% | 92% / 0% / 91% / 45% |

Examples (document 0; the pruned sentence is exactly what Claude would read):

| type | original | ptext22 (2.2x) | ptext15 (1.5x) |
|---|---|---|---|
| negation | The museum at Norlun did not receive 502 maps from Rasmus Pelmir; it received 853. | museum Norlun 502 maps 853 | museum Norlun receive 502 maps Rasmus Pelmir received 853 |
| all but | Dagny Quendek planted 552 pear trees at Faloth, and all but 164 of them died in the first winter. | Quendek planted 552 pear trees Faloth 164 died | Dagny Quendek planted 552 pear trees at Faloth 164 died first winter |
| before / after | After the flood, the granary at Lenbry held 639 sacks of rye; before it, the granary held 294. | granary held 639 sacks rye 294 | After flood granary Lenbry held 639 sacks rye 294 |
| instead of | The ferry at Dekith carried 434 passengers on its first crossing, instead of the 43 that Liesl Peltor had planned. | ferry Dekith carried 434 passengers 43 Liesl Peltor | ferry at Dekith carried 434 passengers first crossing instead 43 Liesl Peltor planned |

Per-question survival: `d*/ptext*_survival.json`.

## Predictions (the reviewer's) and verdict rules

* **P1. `ptext22` is not lossless against `text`, and loses most on small-word and sentence-anchor questions.**
  PASS if (a) the losing end of (ptext22 − text) on all 480 questions is at or below −2 points, and (b) the point
  loss on `small` and on `number` questions are each larger than the loss on `needle` and on `complex` questions.
  FAIL otherwise.
* **P2. `ppage15` is lossless against `text` on the original 120 questions** (360 pairs, 2-point margin). PASS / FAIL.
* **P3. `ppage15` beats pages alone at 4.2x (`page59`) and is not worse than pages at 2.9x (`page88`).** On all 480
  questions, paired: PASS if (ppage15 − page59) has its interval above 0 with McNemar p < 0.05, AND the losing end of
  (ppage15 − page88) is above −2 points. FAIL otherwise. (Also reported on the original 120.)
* **P4. Full wording reads the small words, no document does not.** PASS if `text` and `page118` each score at least
  95% on the 120 small-word questions and `none` at most 10%. FAIL otherwise.

A prediction is **not evaluable** if an arm it needs was refused, lost pages, or failed the engagement check on any
document. Also reported, without a threshold: how often small-word answers equal the lure, per arm and type; the real
token ratios; cost.

## Cost estimate

List price (Opus 5.5, input priced as a cache write at $5 per million tokens, output $20 per million; output assumed
3k for `none`, 8-12k otherwise): **$5.67 per document, $17.03 for 3 documents**, plus $0.39 already spent on
calibration. The cap is $25 at list price (sum of `cost_usd` over uncached calls, `spend.jsonl`); the script stops
before a document whose estimate would cross it. The calls draw on the subscription quota; these are not amounts
billed.

## Order

1. Pilot: document 0, all 8 arms. If it shows a harness bug, fix it, add a dated amendment here, and re-run.
2. Documents 1 and 2.
3. `analyse`, then the results below.

## Reproduce

```bash
python3 experiments/claude_qa/claude_qa_prune.py build        # documents, questions, LLMLingua-2 probabilities (~7 min/doc, CPU)
python3 experiments/claude_qa/claude_qa_prune.py calibrate --rate22 0.37 --rate15 0.57 --yes   # ~$0.4, unscored
python3 experiments/claude_qa/claude_qa_prune.py set-rates
python3 experiments/claude_qa/claude_qa_prune.py check        # survival + cost estimate (free)
python3 experiments/claude_qa/claude_qa_prune.py run --docs 0 --yes
python3 experiments/claude_qa/claude_qa_prune.py run --docs 1 2 --yes
python3 experiments/claude_qa/claude_qa_prune.py analyse
```

Files: `d*/questions.json` (160 questions; small-word ones carry `lure` and `critical`), `d*/runs.jsonl` (one row per
arm with per-question answers), `state.json` (calibration and rates), `spend.jsonl` (every uncached call's list-price
cost), `analysis/summary.{md,json}`.
