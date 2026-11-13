# Prior art for visual token compression (WP-G)

**Status: a record.** This is the literature search done on 2026-11-02, before Phases 1-2 ran. It is kept as
written. Its claims about other people's work come from the papers in [References](#references) and were not
re-checked in the clean-up of 2026-11-13; entries marked "unverified" were never checked at all. A second search,
for learned alphabets, is in [learned_glyphs/docs/prior-art.md](../../learned_glyphs/docs/prior-art.md), and an
external report on the densest code is in
[research/](research/2026-11-04_rgb3_lg_go_no_go_report.md).

**How its expectations held up** (our own results, each re-checked against the rows):

| Expectation here | What was measured | Where |
|---|---|---|
| A condensed font may lose for a vision-language model despite fitting more per line (point 2) | Confirmed for Claude: the condensed font needs 1.11x the pixels of DejaVu Sans at 3% character error | [docs/glyphs/RESULTS.md](../glyphs/RESULTS.md) |
| Below about 9 px² per character every scheme is illegible (section 5, item 2) | True for readers used as they come (Claude needs 29.4, PP-OCR 38.1). Not true for a reader trained for one font: Latin reads at 3% error on 6.54 px² per character | same |
| The gain from a larger hand-picked alphabet is small at best; the stronger idea is a learned alphabet (point 5) | Confirmed: hand-picked substitution needs 1.25x to 1.48x *more* pixels than Latin; a learned alphabet needs 0.57x | same, [learned_glyphs/RESULTS.md](../../learned_glyphs/RESULTS.md) |
| Text-side compression matches or beats the vision path (point 3) | Not tested here on a model that can be trained (Test B was built, not run). On Claude, where only text and images go in, page images of plain text give 2.2x fewer input tokens with no measured loss, in line with "Text or Pixels? It Takes Half" | [CLAUDE_QA.md](../claude/CLAUDE_QA.md), [TEXT_SLOTS.md](../gpu/TEXT_SLOTS.md) |
| A third Latin baseline, `latin_verdana` (section 5, item 1) | Not added. DejaVu Sans, a wide Verdana-like face, was the wide-font baseline | [PLAN.md](../glyphs/PLAN.md) |

*Searched 2026-11-02. "Unverified" means I saw it only in an abstract, snippet or secondary source. "(ours)" marks our own arithmetic from published settings.*

## Bottom line on novelty and expected upside

1. **Novel as far as I can find.** No paper substitutes frequent n-grams with composite glyphs before rendering. The nearest work is PIXEL bigram-per-patch rendering (layout only), Xu Bing's Square Word (Latin packed into square cells, for humans) and FontCode/Neural Glyph (learned glyphs, but not for density). Confidence is moderate.
2. **Typography tuning buys points, not multiples.** Glyph's rendering search beat a hand-tuned config by about 2 points. It settled on a *wide* font (Verdana 9 px, 10 px line, no condensing) even though condensing was in the search space.
3. **Optical compression is itself contested.** Mean pooling and learned text compressors match or beat the vision path. C3 gets 98% at 20× where DeepSeek-OCR gets about 60%. Even a 2× gain from Visual BPE would only bring optical compression level with text-side methods.
4. **Language priors dominate zero-shot reading.** DeepSeek-OCR drops from 88–98% to 20–62% on random letters. A symbol stream gives the reader no prior to lean on, so Phase 1 is biased against substitution and only Phase 2 can be decisive.
5. **Expected upside is small, and only with a trained reader.** Glyph recognition gets less efficient as glyph complexity rises. No study shows a larger alphabet beating Latin per pixel. The stronger version of the idea is an end-to-end learned alphabet.

## 1. Optical compression state of the art

- **DeepSeek-OCR** (Oct 2025) was tested on Fox English pages of 600–1300 text tokens:

  | Vision tokens | Compression | Precision |
  |---|---|---|
  | 100 | 6.7× | 98.5% |
  | 100 | 10.6× | 91.5% |
  | 64 | 10.5× | 96.5% |
  | 64 | 19.7× | 59.1% |

  Its modes run from Tiny (512² px, 64 tokens) to Large (1280² px, 400 tokens). It trained on real PDFs and never tuned typography. My rough estimate is about 100 px²/char at 10× and about 50 px²/char at 20× (ours).
- **DeepSeek-OCR 2** (Jan 2026) swaps in a Qwen2-0.5B encoder with causal "flow" queries and uses 256–1120 tokens. It scores 91.09% on OmniDocBench (+3.7) but publishes no new compression curve.
- **Glyph** (Tsinghua/Zhipu, ACL 2026) reaches 3–4× compression at Qwen3-8B-level accuracy, and about 8× at the extreme.
  - An LLM-guided genetic search tuned the rendering settings: DPI, font, size (7–14 pt), line height, spacing and horizontal scale (0.75–1.0).
  - The released optimum is Verdana 9 pt at 72 dpi, line height 10, scale 1.0. Chinese uses Source Han Sans HW SC at the same size.
  - That works out to about 45–55 px²/char, or about 15 chars per 28×28 vision token (ours), consistent with 3–4×.
  - The search scored 45.6 against 43.6 for manual settings and 40.9 for random ones.
  - DPI matters most: on Ruler it scored 72.2 at 4.0× (72 dpi) versus 94.7 at 1.2× (120 dpi).
- **Other 2026 work:**
  - *LensVLM* (Apple, DejaVuSans) uses presets of 36, 18 and 13 px²/char (ours) at 5×, 10× and 15×. It calls 10× "small but distinguishable" and 15× mostly illegible. Naive reading at 5× scores 31.3% against 72.4% for full text.
  - *Measure Transport* (Roboto) swept font size: 10, 12 and 14 pt gave scores of 42.4, 44.4 and 45.3 for 1185, 1486 and 1716 tokens.
  - A scaling-law paper finds that characters per token matter more than local density.
  - A CVPR-W paper reportedly finds font family barely matters (unverified).
- **Nobody has tested** condensed fonts, ligatures or symbol substitution.

## 2. Critiques

- **Lee, Yang & Berg-Kirkpatrick** (2025/26) compared reconstruction perplexity:
  - Vision: 1.06 at 8.6× and 1.14 at 12.8×.
  - Parameter-free mean pooling: 1.16 at 9.8×.
  - A learned hierarchical encoder: 1.00 at 7.9× and 1.01 at 15.4×.
  - For language modelling, vision scores about the same as truncation, or slightly worse, and behind the
    hierarchical encoder at every ratio.
  - Factual recall (answer log-likelihood, 9,794 generated QA pairs): every compressor beats truncation, and vision
    never beats the best text encoder (at ~9×: vision −2.23, hierarchical −2.08, truncation −2.95; full context −1.07).
  - Setup: all encoders feed DeepSeek-OCR's own 3B MoE decoder, trained one epoch on 500k 2,000-token Wikipedia
    segments, scored on 1k-token contexts. Stated limits: single runs, English Wikipedia only, no contexts beyond
    1k, only DeepSeek-OCR's encoder tested. UC San Diego; preprint, v2 April 2026.
- **C3** (text-side latent compression): 98% at 20× and about 93% at 40×.
- **"Visual Merit or Linguistic Crutch?"** (Jan 2026) tested DeepSeek-OCR on random-letter text:

  | Mode | Clean | Random letters |
  |---|---|---|
  | Tiny | 88.0% | 19.8% |
  | Small | 95.2% | 42.1% |
  | Base | 97.9% | 61.7% |

  PaddleOCR holds at about 90%. Fewer vision tokens means more reliance on priors, and the model collapses at about 10k tokens.
- **VTCBench**: VLMs can OCR rendered text but understand long contexts in it poorly.
- **"Text or Pixels? It Takes Half"** reports roughly 50% token savings with no loss.

## 3. Script effects

- **No direct cross-script measurement of information per vision token exists.** The 2026 scaling-law paper explicitly lists Chinese as untested.
- **Indirect evidence:**
  - DeepSeek-OCR's OmniDocBench edit distance is worse for Chinese (0.181) than English (0.127).
  - Glyph's LongBench QA-Zh score is about 37 against baselines of about 60, while its QA-En score is close to baseline (my reading of their tables).
  - At equal point size an ideograph takes about 1.8× the area of a Verdana letter (ours).
  - SeeTok's biggest gain is on Chinese (+10.8 COMET), but that is measured against text-token fertility, not pixels.
- **Human psychophysics:**
  - Pelli et al. (2006, many scripts including Chinese): identification efficiency is inversely proportional to perimetric complexity.
  - Legge-lab work: Chinese characters need more spatial-frequency content and have a smaller visual span (exact figures unverified).
  - Both point against "a bigger alphabet means more information per pixel".

## 4. Closest prior art

- **Pixel language models** (PIXEL, PIXAR, PixelGPT, PHD) all render standard fonts and never fuse glyphs.
  - Lotz et al. (2023) rendered two characters per 16×16 patch ("BIGRAMS"). This helped, and a 22M model matched an 86M one. It changes alignment, not glyph shape.
- **Visual tokenizers** (SeeTok, VIST) merge patches in feature space, not in glyph space.
- **Hangul-style Latin:** Xu Bing's Square Word Calligraphy (1994 on) packs English letters into square cells, and Western readers can read it. It is art and has never been tested with machines.
- **Learned or compact glyphs:**
  - FontCode (2017) hides bits in glyph perturbations and reads them back with CNNs.
  - Neural Glyph (2020) evolves glyphs through a generator–classifier communication game.
  - Other examples are OCR-A/B and OCR-loss glyph optimisation (Word-as-Image, DiffInk).
- **Machine stenography:** nothing found.
- **No evidence** that a larger glyph alphabet beats Latin in bits per pixel for a learned reader.

## 5. Implications for our plan

1. **Add a third baseline, `latin_verdana`.** Use Verdana or the metric-similar DejaVu Sans with a 1.1 line ratio, and take the best of three Latin baselines. Glyph's search rejected condensing for a VLM, so `latin_condensed` may lose in Phase 1 despite winning K1. Define "best Latin" empirically for each reader.
2. **Change the grid.** Published legible regimes sit at roughly 13–100 px²/char, so 4–9 px²/char is likely illegible for every scheme. Concentrate on 12–60 and add 90.
3. **Revise kill criteria.**
   - Treat K2 as weak evidence. Zero-prior text costs DeepSeek-OCR 35–70 points.
   - Keep K3 decisive.
   - Report Latin's natural-minus-scrambled gap as "prior credit".
4. **State the ceiling.** Even a 30% gain over Latin still trails C3 and learned text compressors. Say so in RESULTS.md.
5. **If subst survives, try a better formulation.**
   - Learn the alphabet end to end against the reader at a fixed px² budget, using differentiable rasterisers or Neural Glyph/FontCode-style training.
   - Report bits/px² against Latin and against a barcode-like upper bound.
   - Try patch-aligned layout, following Lotz et al.

## References

1. DeepSeek-OCR. Wei, Sun, Li (DeepSeek-AI), 2025-10. https://arxiv.org/abs/2510.18234
2. DeepSeek-OCR 2. Wei, Sun, Li, 2026-01. https://arxiv.org/abs/2601.20552
3. Glyph. Cheng et al. (Tsinghua/Zhipu), 2025-10. https://arxiv.org/abs/2510.17800 ; config: https://github.com/thu-coai/Glyph/blob/main/config/config_en.json
4. Optical Context Compression Is Just (Bad) Autoencoding. Lee, Yang, Berg-Kirkpatrick (UC San Diego), 2025-12, v2 2026-04. https://arxiv.org/abs/2512.03643
5. Context Cascade Compression. Liu, Qiu, 2025-11. https://arxiv.org/abs/2511.15244
6. Visual Merit or Linguistic Crutch? Liang et al., 2026-01. https://arxiv.org/abs/2601.03714
7. VTCBench. Zhao et al., 2025-12. https://arxiv.org/abs/2512.15649
8. Text or Pixels? It Takes Half. Li, Lan, Zhou, 2025-10. https://arxiv.org/abs/2510.18279
9. LensVLM. Xie et al. (Apple/Duke), 2026-05. https://arxiv.org/abs/2605.07019
10. Visual Text Compression as Measure Transport. Tang et al., 2026. https://arxiv.org/abs/2605.06708
11. How Much Information Can a Vision Token Hold? Zhuang et al., 2026-01. https://arxiv.org/abs/2602.02539
12. Jenq et al., Adaptive rendering for VTC, CVPRW 2026 (unverified). https://openaccess.thecvf.com/content/CVPR2026W/MULA2026/papers/Jenq_Enhancing_Vision_Text_Compression_in_Vision_Language_Models_via_Adaptive_CVPRW_2026_paper.pdf
13. VIST. Xing et al., NeurIPS 2025. https://arxiv.org/abs/2502.00791
14. SeeTok. Xing et al., 2025-10. https://arxiv.org/abs/2510.18840
15. PIXEL. Rust et al., ICLR 2023. https://arxiv.org/abs/2207.06991
16. Text Rendering Strategies for Pixel LMs. Lotz et al., EMNLP 2023. https://arxiv.org/abs/2311.00522
17. PIXAR. Tai et al., 2024. https://arxiv.org/abs/2401.03321
18. PixelGPT. Chai et al., EMNLP 2024. https://arxiv.org/abs/2404.10710
19. PHD. Borenstein et al., EMNLP 2023. https://arxiv.org/abs/2310.18343
20. FontCode. Xiao, Zhang, Zheng, 2017. https://arxiv.org/abs/1707.09418
21. Neural Glyph. Park, 2020. https://arxiv.org/abs/2010.04402
22. Feature detection and letter identification. Pelli et al., Vision Research 2006. https://pubmed.ncbi.nlm.nih.gov/16808957
23. Xu Bing, Square Word Calligraphy. https://wallach.columbia.edu/content/xu-bing-square-word-calligraphy-classroom
24. Wang, He, Legge, pattern complexity and visual span, JoV 2014. https://legge.dl8.umn.edu/sites/legge.psych.umn.edu/files/2020-08/Wang%20He%20Legge%202014%20Effect%20of%20pattern%20complexity%20on%20the%20visual%20span%20for%20Chinese%20and%20alphabet%20characters%20.pdf
