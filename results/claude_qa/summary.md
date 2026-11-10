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
