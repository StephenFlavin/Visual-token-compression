# Test A: pruned text against page images (Claude QA)

*Pre-registered 2026-11-12, written and committed before any scored call.* Design draft:
[docs/REVIEW_2026-11-11.md](../../../docs/REVIEW_2026-11-11.md), "Test A". It extends the v2 test in
[docs/CLAUDE_QA.md](../../../docs/CLAUDE_QA.md) (A1, A4).

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
python3 experiments/claude_qa_prune.py build        # documents, questions, LLMLingua-2 probabilities (~7 min/doc, CPU)
python3 experiments/claude_qa_prune.py calibrate --rate22 0.37 --rate15 0.57 --yes   # ~$0.4, unscored
python3 experiments/claude_qa_prune.py set-rates
python3 experiments/claude_qa_prune.py check        # survival + cost estimate (free)
python3 experiments/claude_qa_prune.py run --docs 0 --yes
python3 experiments/claude_qa_prune.py run --docs 1 2 --yes
python3 experiments/claude_qa_prune.py analyse
```

Files: `d*/questions.json` (160 questions; small-word ones carry `lure` and `critical`), `d*/runs.jsonl` (one row per
arm with per-question answers), `state.json` (calibration and rates), `spend.jsonl` (every uncached call's list-price
cost), `analysis/summary.{md,json}`.
