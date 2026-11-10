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
