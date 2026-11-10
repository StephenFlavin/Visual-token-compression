# Claude QA over a rendered document vs its text (2026-11-10)

`experiments/claude_qa_render.py`, headless Claude (same model as Phase 1, effort low), one document: 503,630 chars of
held-out WikiText-103 test text plus 40 invented facts ("needles"). 80 questions with exact numeric answers: 40 about
the invented facts (only the document can answer them) and 40 asking for the number in a quoted real sentence. Same
document string and questions in every arm. Tokens are the CLI's usage report (all input, incl. cache creation).
Pages: DejaVu Sans Latin (Phase 1's renderer), each <= 1.15 MP so the API does not downsample.

| Arm | px²/char | pages | input tokens | fewer tokens than text | invented facts | numbers | text-only / image-only correct (exact McNemar p) |
|---|---|---|---|---|---|---|---|
| text | - | - | 175,586 | 1x | 100% | 100% | |
| image | 29.4 (Phase 1's 3% CER point) | 14 | 22,390 | 7.8x | 33% | 42% | 50 / 0 (2e-15) |
| image | 58.8 | 27 | 41,093 | 4.3x | 95% | 85% | 8 / 0 (0.008) |
| image | 117.6 | 54 | 78,522 | 2.2x | 100% | 100% | 0 / 0 (1) |
| image | 176.4 | 80 | 115,978 | 1.5x | 100% | 100% | 0 / 0 (1) |
| no document | - | - | 3,617 | - | 0% | 25% (prior knowledge) | |

* Claude's tokenizer gives 2.87 chars per input token on this text (not the ~4 assumed earlier), so the
  savings are larger than the px²/char suggests.
* At 29.4 px²/char accuracy falls through the document (53% on the first fifth, 19% on the last); errors are
  misread digits and round-number guesses.
* Limits: one document, one call per arm, exact-retrieval questions only (no synthesis or reasoning), effort low.
  Text scores 100%, so a ceiling hides any loss at 2.2x and 1.5x; harder questions could separate them.

## 800k tokens (2026-11-10)

Same design, document 2,303,133 chars (held-out test + validation text, then a little train text) + the same kind of
40 invented facts; 80 questions. The text arm fits Claude Opus 5.5's 1M context through the CLI.

| Arm | px²/char | pages | input tokens | fewer tokens than text | invented facts | numbers | text-only / image-only correct (exact McNemar p) |
|---|---|---|---|---|---|---|---|
| text | - | - | 780,817 | 1x | 100% | 100% | |
| image | 29.4 | 61 | 89,252 | 8.7x | 20% | 57% | 49 / 0 (4e-15) |
| image | 58.8 | 122 | 174,808 | 4.5x | 97% | 93% | 4 / 0 (0.125) |
| image | 117.6 | 243 | 161,377 (expected ~345k) | **invalid** | 45% | 62% | the first ~130 pages never reached the model |
| no document | - | - | 3,662 | - | 0% | 15% | |

* Text at 780,817 tokens is 2.95 chars per token.
* 58.8 px²/char holds up at 800k: 4 misses in 80, spread through the document (not position-dependent).
* **The 117.6 px²/char run is a harness artefact, not a result.** Input tokens (161k) match ~112 pages, not 243;
  every question about the first ~55% of the document got a placeholder guess (5000 / 8000 / 3000 in rotation) and
  every question about the rest was right. The CLI or API dropped the earliest images silently (no error or warning).
  The 122-page run (17 MB of PNG) was complete; the 243-page one (30 MB) was not. Mechanism not established. Check
  `input_tokens` against the image-token formula before trusting any many-image call.

## 800k tokens, 100 questions incl. 20 complex, 1932-px pages (2026-11-10)

`--chars 2300000 --page-px 1932 --n-complex 20`. Same 40 invented facts and 40 number questions as above, plus 20
complex questions over invented facts (the prior cannot answer them; the no-document arm scores 0/20):
8 two-hop (the person who rebuilt bridge X chairs rowing club Y, which owns N boats; the three facts at least 1/8
of the document apart), 6 sums of two facts, 3 largest-of-three, 3 counts of a fact type no other question
mentions (5-12 facts each). Pages: 1932 x 1932 px (high-resolution tier, 4,764 tokens each; a probe page cost
4,772 tokens and its first line came back exact). Rows: `qa800kc_b29`, `qa800kc_b59`, `qa800kc_b118`.

| Arm | px²/char | pages | input tokens | fewer than text | invented facts (40) | numbers (40) | complex (20): two-hop / sum / max / count |
|---|---|---|---|---|---|---|---|
| text | - | - | 782,989 | 1x | 100% | 100% | 80%: 7/8, 6/6, 3/3, 0/3 |
| image | 29.4 | 19 | 91,509 | 8.6x | 0% | 38% | 0% |
| image | 58.8 | 38 | 178,094 | 4.4x | 95% | 90% | 60%: 3/8, 5/6, 3/3, 1/3 |
| image | 117.6 | 75 | 147,781 (expected 346,467) | **invalid** | 40% | 57% | pages dropped |
| no document | - | - | 4,804 | - | 0% | 23% | 0% |

* 58.8 px²/char vs text: text-only correct 11, image-only 1 (exact McNemar p = 0.006). Simple lookups stay close
  (95% / 90%), but **two-hop drops from 7/8 to 3/8**: the image arm finds the boat facts but links the wrong club
  (its wrong answers are other clubs' boat counts). Small counts per type: 8 two-hop, 3 counts.
* Counting over the whole document is hard even from text (0/3; answers 7, 6, 8 for 9, 11, 12).
* Large pages vs Phase 1's small pages: 58.8 about the same (95/90 vs 97/93 on the simple 80); 29.4 worse (0/38 vs
  20/57).
* **The first version of the 3 count questions leaked** (they counted needle types that the 40 needle questions
  enumerate): the no-document arm scored 3/3. Kept as `qa800kc_v1_countleak`; replaced before the runs above.
* **4x could not be measured through the CLI.** Both attempts lost their earliest pages: 243 small pages (342k image
  tokens, 29.7 MB) arrived as ~157k tokens, 75 large pages (346k tokens, 27.3 MB) as ~143k. Not a byte or image-count
  limit: 7 noise pages of 26 MB (33k tokens) arrived whole, and 122 / 38 pages at ~175k image tokens arrived whole.
  Something between the CLI and the model drops the earliest images once a request carries somewhere between ~176k
  and ~342k image tokens; it is not in the vision docs (600 images, 128 MB per request). The script now flags it.

## 700k chars (245k tokens), 100 questions incl. 20 complex, 1932-px pages (2026-11-10)

`--chars 700000 --page-px 1932 --n-complex 20`; sized so the least dense rendering (~159k image tokens) stays under the
page-dropping cap. Every image arm arrived complete (input tokens >= the image-token formula). Rows: `qa700kc_b176`,
`qa700kc_b118`, `qa700kc_b59`.

| Arm | px²/char | pages | input tokens | fewer than text | invented facts (40) | numbers (40) | complex (20): two-hop / sum / max / count | text-only / image-only correct (p) |
|---|---|---|---|---|---|---|---|---|
| text | - | - | 245,285 | 1x | 100% | 100% | 100%: 8/8, 6/6, 3/3, 3/3 | |
| image | 176.4 | 35 | 164,070 | 1.5x | 100% | 100% | 100%: 8/8, 6/6, 3/3, 3/3 | 0 / 0 (1) |
| image | 117.6 | 23 | 111,040 | **2.2x** | **100%** | **100%** | **100%: 8/8, 6/6, 3/3, 3/3** | 0 / 0 (1) |
| image | 58.8 | 12 | 57,947 | 4.2x | 97% | 97% | 70%: 6/8, 5/6, 3/3, 0/3 | 8 / 0 (0.008) |
| no document | - | - | 4,797 | - | 0% | 7% | 0% | |

* **2.2x is lossless here on all 100 questions, multi-step included**; 1.5x too. At 4.2x lookups stay at 97% but
  multi-step questions drop (two-hop 6/8, counts 0/3).
* Text gets the counts right at this length (3/3) but not at 800k tokens (0/3).
* Ceiling: text scores 100%, so a loss smaller than 1 question in 100 at 2.2x would not show.
