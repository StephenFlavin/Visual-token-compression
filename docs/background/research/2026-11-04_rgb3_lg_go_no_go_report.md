# Visual BPE (rgb3_lg): Literature-Calibrated Go/No-Go Report

> **Status: a record, kept as received.** This is the external literature report commissioned on 2026-11-04 with
> [MOONSHOT_BRIEF.md](../../gpu/MOONSHOT_BRIEF.md), about the densest code in the repo: three pages of learned
> glyphs overlaid in the red, green and blue channels (`rgb3_lg_pb_h5`, 322 characters per vision token; "405" is
> its grey-level variant `rgb3_lg_free_h4`, dropped on 2026-11-05). "You" is the user, "the assistant" is the session that wrote the brief. Its statements about
> papers and models are the report's own and were not re-checked in the clean-up of 2026-11-13.
>
> **How it compared with what was then measured:** the report put the plausible range at 30-90 characters per
> vision token and expected a cliff. Measured on Qwen2.5-VL-3B: after a 1,000-step fine-tune only Latin at 28
> characters per token was read (55% character error against 88.5% for its blind control); the 322 page ended at
> its blind control. Without training, the model recovers at most 12.6 bits per vision token, at 28 characters per
> token, and 1.7 bits at 72. So the cliff was there, and lower than the report's range. The staged gates it
> proposes were not run as written; the trained-encoder runs that replaced them were parked
> ([PLAN_3090.md](../../gpu/PLAN_3090.md)). Summary of the whole line: [README](../../../README.md), sections 3-4.

**Bottom line: published results put a plausible go region at about 30–90 characters per vision token, not 322–405. The moonshot point is about 7x beyond the best trained optical result at high accuracy and about 2x beyond the best text-side learned compressor. Re-scope the project as a density sweep, with trained-merger gates and a matched-KV QA test against a text-side compressor.** DeepSeek-OCR's trained 16x compressor reaches 96.5% precision at 10.5 text tokens per vision token. That is about 42–47 characters per token, on prose where language priors help. Context Cascade Compression (C3; Liu & Qiu, arXiv 2511.15244), which has no pixel bottleneck at all, reaches 98.4% when compressing 1,200–1,300 text tokens into 64 latents (nearly 20x) and 93.3% into 32 latents (nearly 40x). That works out to about 86 and 170 characters per latent. A go at 30–60 characters per token would still fit about 1–2M characters in a 32k window, which is 7–14x the text baseline.

## TL;DR
- **Density:** Your success bar (≥1.5x fine-tuned Latin, roughly 25–40 chars/token) sits inside the published go region; 322 does not. DeepSeek-OCR falls from 96.5% at 10.5x to 59.1% at 19.7x (Table 1 of the C3 paper). A 2026 capacity study finds an abrupt "hard wall": the unstable band spans only about 2.2x in text length, and token load, more than glyph density, drives collapse. Run a sweep and expect a cliff, not a slope.
- **Bottleneck:** Q1 points to the merger and the downstream stage, and the literature agrees. A trained compressor/projector is what made DeepSeek-OCR, TextHawk2 and GOT-OCR work. Frozen vision encoders lose to plain mean pooling for optical compression. Embedding space itself has ample capacity (one 2048-d vector can be optimized to reconstruct about 450 tokens). But amortized encoders reach only a fraction of that, and recovering text is not the same as using it for QA.
- **Decision rule:** Kill the 322/405 points if a trained merger cannot reach ≥90% glyph accuracy at ≥120 chars/token. Keep the project alive if any code clears ≥1.5x fine-tuned Latin at <3% CER and stays within 3 points of Latin on needle QA and 5 points on cloze at ≤0.67x KV. Braille and other prebuilt dense codes give useful engineering parts (ECC, calibration, alignment), but no ready-made VLM prior.

---

## A. Answers to the research questions

### Q1. Has anyone trained a VLM/ViT to read dense synthetic visual codes?
I found no published work that trains a VLM to read QR, Data Matrix, JAB Code, HCC2D or learned glyph alphabets as a context-compression medium, or that reports capacity per vision token for such codes. The prior art splits into two groups that don't overlap:
- **Classical dense codes, read by hand-built decoders rather than ViTs.**
  - Xerox DataGlyphs store 1 bit per mark. With 5x5-pixel glyph cells at 300 dpi that is 3,600 raw bits per square inch, or about 400 bytes per square inch of data plus error correction. At 600 dpi, up to 1 KB per square inch.
  - Microsoft HCCB claims about 3,500 characters per square inch for 8-colour codes.
  - HCC2D codes stored 4,930 bytes per square inch including redundancy, about 3,972 bytes of effective data.
  - JAB Code (Fraunhofer SIT, LGPL) uses 4 or 8 colours in the current encoder/decoder. Sources disagree on the specification's ceiling (64 vs 256 colours). The code embeds its colour palette in every symbol for calibration.
  - Deep learning shows up here mainly for *detecting* codes (e.g. a Roboflow detector at 99.5% mAP that then hands off to pyzbar), not for decoding them.
- **Rendered natural text into VLMs** (DeepSeek-OCR, Glyph, TextHawk2, GOT-OCR2.0, LensVLM). Here the layers that had to be trained are documented:
  - **DeepSeek-OCR:** a 2-layer conv 16x compressor sits between SAM-base and CLIP-large. The whole DeepEncoder is trained in stage 1. SAM and the compressor are then frozen while CLIP and the decoder are trained.
  - **TextHawk2:** a new 16x resampler is trained, with the ViT and LLM updated by LoRA (ranks 16 and 128), then everything is unfrozen in SFT. The authors report that "unfreezing ViT during both pre-training and supervised fine-tuning significantly enhances performance."
  - **GOT-OCR2.0:** the encoder and its compression layers are pretrained, then trained jointly with a 0.5B decoder.

**Implication:** your learned-glyph code is novel territory. The nearest analogue for "which layers must be trained" is that every system above 4x compression trained a compressor or projector, and usually the encoder too. TextHawk2 says earlier top OCR models "have only achieved an image compression ratio of up to 4." LoRA plus the last 4 ViT blocks plus the merger is therefore the minimum credible configuration, not an over-spend.

### Q2. How do pretrained ViTs handle colour-channel-separated information?
I found no direct study of per-channel multiplexing of independent signals through CLIP, SigLIP or the Qwen2.5-VL ViT. What I did find is indirect:
- Interpretability work finds a progression from low-level attributes such as colour and line in early layers to semantic concepts later.
- Patch tokens keep spatial information through all layers except the last attention block.
- Colour-probing accuracy generally improves from early to middle layers, but its trajectory doesn't always match object recognition.

Your Q1 result (all colour channels survive; the moonshot reaches Latin-level R² from block 24) is therefore new evidence, not a replication. The literature's warning is that colour features in pretrained encoders tend to be tied to object semantics, so channel separation should be checked *after* the merger and after fine-tuning, not only in raw ViT blocks.

### Q3. DeepSeek-V4.1-Flash and DeepSeek-OCR 2
**DeepSeek-V4.1-Flash (arXiv 2609.19969, Sep 2026).** From the technical report, its KV savings come from attention and cache design, not optical compression:
- **Causal Encoder-Decoder:** a 20-layer causal encoder plus a 20-layer decoder, with decoder global KV projected from the final encoder hidden states. This roughly halves prefill compute.
- **CSA2 cross-layer reuse:** Full, Reindex and Reuse modes share main KV and indexer K, and reuse Top-K indices.
- **FP4 main KV:** E2M1 format, with one E4M3 scale per 16 channels.
- **SWA Bounded Replay.**
- **Result:** global KV of 890 bytes per token, about 1/4 of DeepSeek-V4-Flash and about 437x smaller than DeepSeek-V1. Persistent KV is about 1/8 of V4-Flash.

The model is multimodal. Images go through a from-scratch DeepSeek-ViT, then a 3x3 pixel-unshuffle (9x token reduction, inputs up to about 1344x1344), then an MLP projector. In the sections I read (abstract through architecture), nothing says long text is rendered as images or optically compressed. **Flag:** I did not read the full pre-training and data sections, so "no optical compression" is supported for the architecture but not exhaustively verified for the training data.

**DeepSeek-OCR 2 (arXiv 2601.20552, Jan 2026).**
- It keeps the 80M SAM-based tokenizer and 16x conv compression.
- It replaces CLIP with a compact LLM-style encoder: Qwen2-0.5B, which the paper's Fig. 1 labels "LM as Vision Encoder Qwen2 500M", in place of "CLIP VIT 300M".
- Learnable "causal flow" query tokens follow the visual tokens, and only the query half goes to the decoder, giving 256–1,120 tokens.
- It reports OmniDocBench v1.5 overall edit distance of 0.100 vs 0.129 for DeepSeek-OCR, which the paper summarises as "3.73% performance gains on OmniDocBench v1.5".
- The authors say it preserves DeepSeek-OCR's compression ratio. It does not push density higher.

**Useful arithmetic:**
- Qwen2.5-VL-3B: from its published config (36 layers, 2 KV heads, head dim 128, bf16), KV is about 36 KB per token. As text, that is about 8.6 KB per character.
- The moonshot at 322 chars/token: about 114 bytes per character.
- V4.1-Flash reading plain text at about 4.3 chars/token: about 207 bytes per character.

So architectural KV compression alone gets within 2x of the moonshot's bytes per character, and the two approaches compose.

### Q4. Information capacity of a vision token after merging
| Source | Setting | Density | Accuracy |
|---|---|---|---|
| Text or Pixels? It Takes Half | Off-the-shelf GPT-4.1-mini / Qwen2.5-VL-72B, no fine-tuning | ~2x text tokens per vision token (~8–9 chars) | Within 3 points of text; decoder tokens cut 38–58% |
| Glyph (Zhipu) | Continual pretraining + rendering search + post-training | 3–4x (~13–17 chars) | Comparable to Qwen3-8B on long-context benchmarks |
| LensVLM (Apple, Qwen3.5-9B) | Rendered text, presets 5x/10x/15x | 5x naive reading | 31.3% vs 72.4% text |
| DeepSeek-OCR (Fox, prose) | Trained 16x conv compressor | 10.5x (~42–47 chars) | 96.5% precision |
| DeepSeek-OCR | same | 11.8x / 19.7x | 93.8% / 59.1% |
| TextHawk2 | Trained 16x resampler | 183 words in 32 tokens (~30+ chars, single demo) | "accurate recovery" (no metric) |
| Optical Context Compression Is Just (Bad) Autoencoding (Lee, Yang, Berg-Kirkpatrick, UCSD; arXiv 2512.03643 v2, Apr 2026; row added 2026-11-10) | DeepSeek-OCR's vision encoder vs mean pooling vs a learned hierarchical text encoder (13-92M), all into DeepSeek-OCR's decoder, trained 1 epoch on 500k Wikipedia segments; 1k-token contexts | vision 2.3-12.8x; text encoders up to 83-100x | Reconstruction perplexity at ~8-15x: vision 1.06-1.14, hierarchical 1.00-1.01. LM: vision about equal to truncation, behind the hierarchical encoder at every ratio. QA recall: vision beats truncation but never the best text encoder. Single runs, English Wikipedia only |

**Capacity-law evidence.** "How Much Information Can a Vision Token Hold?" (arXiv 2602.02539) tested DeepSeek-OCR, with Qwen2.5-VL and InternVL3.5 as transfer checks. It used block-shuffled text to remove language priors. Findings:
- Three phases: stable, then unstable, then collapse.
- The unstable band always spans about 2.2x in text length, independent of resolution.
- Token load dominates density: the fitted exponent is 5.53 for load vs 2.91 for density.
- Errors in the unstable band can be fixed by shifting the image, so they are caused by misalignment with patch boundaries. Collapse persists even when glyph size is held fixed, so it is a genuine capacity limit.

**Language priors.** Liang et al., "Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR" (arXiv 2601.03714, Jan 2026), report that without linguistic support DeepSeek-OCR "plummets from approximately 90% to 20%," with "a total model collapse around 10,000 text tokens." So the 42–47 chars/token figure is inflated by priors, which your code lacks by design.

### Q5. Strongest long-context QA results, and who wins at equal KV
- **Rendered text:**
  - LensVLM-9B keeps accuracy comparable to full text at 4.3x effective compression. It scores 68.9% at the 5x preset vs 72.4% for text, and 62.1% and 52.1% at 10x and 15x, with KV 5.9–14.1x smaller than full text. It gets there by selectively expanding pages back to text, not by pure visual reading.
  - "VLMs Read Aloud" (arXiv 2611.01517) raises a frozen Qwen3-VL-8B at 5x from 27.7% to 54.8%, against 58.5% for the original text. It concludes that compression "is limited not by how much text a model can recover, but by how much it can use."
  - VTCBench finds most VLMs decode well (OCR) but show "surprisingly poor long-context understanding" under compression.
  - FocusVTC reports RULER 87.4 at 2.9x compression.
- **Text-side:**
  - C3: 98.4% reconstruction at nearly 20x (1,200–1,300 tokens into 64 latents) and 93.3% at nearly 40x (the same lengths into 32 latents), against DeepSeek-OCR's 59.1% at the same 19.7x.
  - 500xCompressor beats ICAE at every tested ratio, with ICAE's practical maximum around 15x. 500xCompressor keeps about 62–73% of uncompressed capability.
  - "Optical Context Compression Is Just (Bad) Autoencoding" (arXiv 2512.03643) finds that a learned hierarchical text encoder "outperforms vision at all compression ratios." Without fine-tuning, mean pooling beats the pretrained vision encoder.
- **Verdict at equal KV:** on reconstruction, text-side compressors win clearly at every published ratio. On QA, the best rendered-text systems at about 4–5x approach text only with tool-based expansion or a read-aloud step. No paper shows pure visual reading beating a trained text-side compressor at matched KV. This is the strongest prior against the project, and it is why the QA gate below must include a text-side baseline.

### Q6. Published curves where rendered text and text-side compression cross
I found none. C3's head-to-head table against DeepSeek-OCR shows C3 ahead at every ratio from 6.7x to about 20x (for example 99.9% vs 98.5% at 6.7x), so the curves don't cross in published data. Your density sweep, with a text-side baseline at matched KV, would be the first crossing test.

### Q7. Per-token capacity after a trained merger or projector
- The only quantitative points come from trained compressors that aren't Qwen's merger: DeepSeek-OCR's conv compressor (about 42–47 chars/token at 96.5% on prose) and TextHawk2's resampler (a qualitative demo of about 30 chars/token).
- I found no paper that retrains the Qwen2-VL/Qwen2.5-VL 2x2 merger, or a higher-ratio replacement, for dense OCR and reports a density-accuracy curve. The scaling-law paper's Qwen2.5-VL results are inference-only.
- **This is a real gap, and the single most valuable measurement the project could publish.**

### Testing the assistant's five analysis points
1. **Sweep vs one extreme point: supported.** The success bar sits around DeepSeek-OCR's high-accuracy band. The 2.2x width of the unstable band means a sweep at 2x spacing (15, 30, 60, 120, 240, 322) will land within one step of the wall.
2. **Bottleneck is the merger or downstream: supported, but the probe was wrong.** Linear pixel R² after the merger is too strict a test. The merger only needs to preserve glyph *identity*: roughly 1.5–2.8 kbits per position at 322 chars, not 2,352 raw pixel values. Replace it with a nonlinear glyph-ID probe.
3. **Bits per slot: capacity exists, but usable capacity is the issue.**
   - "Cramming 1568 Tokens into a Single Vector" gives the ceiling. A 2048-d fp16 vector has a theoretical 32,768 bits, enough for about 1,931 tokens from a 128k vocabulary. Llama-3.2-1B reconstructs 7,168 tokens from 16 vectors, about 448 tokens per vector.
   - Those vectors are optimized per sample. For trained encoders, Kuratov et al. note that "the maximum attainable lossless compression ratio is typically not higher than x10."
   - 322 chars (about 75 text tokens) per position is within the storage ceiling but about 2x beyond C3's best amortized result.
4. **Learned compressor framing: correct.** Once the merger is trained, compare against C3, 500xCompressor and ICAE at matched KV. The visual path's only structural advantages are cheap rendering and robustness to layout or script.
5. **The 0.03x prefill estimate.** This is my own arithmetic, assuming roughly a 0.67B-parameter ViT that sees 1,024 pre-merge patches per 448-px page.
   - ViT FLOPs per page come to about 1.4 TFLOP, roughly equal to the 3B LLM's cost on the page's 256 tokens.
   - The LLM-only ratio at 322 chars/token is about 0.013x (256 vs about 19k text tokens). Including the ViT gives about 0.026x, so "0.03x" holds only if it already includes the ViT.
   - At a realistic 40 chars/token the ratio rises to about 0.2x.
   - Verify the ViT parameter count against the Qwen2.5-VL config.

---

## B. Calibrated go/no-go framework

### Where a go is plausible
- **30–60 chars/token: plausible go.** About 2–4x Glyph-style Latin and about 1x DeepSeek-OCR's 96.5% point.
- **60–120: stretch.** Beyond any published high-accuracy optical point, but under C3's 40x latent level of about 170 chars.
- **322–405: implausible at LoRA-3B scale.**

Even 40 chars/token gives about 1.3M characters in a 32k window, about 9x the 140k baseline. That is a strong result on its own.

### Staged gates
| Gate | Test | Go threshold | Justification |
|---|---|---|---|
| G0 Identity ceiling | Raw-pixel CRNN, full 3,000 steps, at 322 / 405 and a 60–120 point | <3% CER | If an unconstrained reader can't decode the alphabet, no VLM will. Already <3% for single bank, rgb3 Latin and colour codes. |
| G1 Frozen pre-merger probe | Nonlinear (MLP) glyph-ID probe on ViT block 32 tokens | ≥97% per-glyph accuracy | Q1 already shows Latin-level R² from block 24. Confirms identity, not just pixels. |
| G2 Trained-merger probe | Train merger (plus last 4 blocks) with a CTC head; LLM frozen; density sweep | ≥97% glyph accuracy at the target density. Kill the moonshot if <90% at ≥120 chars/token | Fills the Q7 gap. DeepSeek-OCR's trained compressor manages about 45 chars at 96.5% with priors. |
| G3 VLM transcription sweep | LoRA + last 4 ViT blocks + merger; Latin vs code at 15, 30, 60, 120, 322 | Best code density at <3% CER ≥1.5x best Latin density at <3% CER | Your stated bar. The 2.2x unstable band means adjacent sweep points bracket the wall. |
| G4 QA at matched KV | Needle (RULER S-NIAH / VTCBench retrieval) and cloze at ≤0.67x Latin's KV | Needle within 3 points of Latin images. Cloze within 5 points. And within 5 points of a text-side compressor (500xCompressor/ICAE-style LoRA on Qwen2.5-3B) at the same KV | The 3-point tolerance matches "Text or Pixels?". The text-side baseline is required because C3 and the autoencoding paper show text latents win reconstruction. |
| G4b Use vs read | Same QA with a transcribe-first (read-aloud) prompt | Report the gap. If transcribe-first recovers most of it, the bottleneck is reasoning, not capacity | LensVLM: 31.3% vs 72.4%. Read Aloud: 27.7% → 54.8%. |

### What a negative result at LoRA-3B would and wouldn't rule out
- **Would rule out:** a cheap adaptation route to high-density codes on a 3090. It would also show that Qwen's pretrained ViT features can't be repurposed for codes with no language prior at that density.
- **Would not rule out:**
  - Embedding capacity: cramming shows about 448 tokens per vector is storable.
  - Codes trained with a from-scratch or fully trained encoder at DeepSeek-OCR or Glyph data scale (continual pretraining on large rendered corpora).
  - Larger decoders: cramming capacity grows with model size, with 8B models reaching 1,568 tokens.

Report negatives as "not reachable at this budget," not "impossible."

### Kill criteria and fallback ordering
1. **Kill rgb3_lg at 322/405** if G0 stays above 3% CER at step 3,000, or G2 is below 90% at ≥120 chars/token.
2. **Fallback 1: the same code at 60–120 chars/token**, with the line pitch changed to divide the 28-px merged-token grid (4 or 7 px rather than 5). This is my inference from the scaling-law finding that unstable-band errors come from patch misalignment and disappear with pixel shifts. Test it with their shift protocol.
3. **Fallback 2: single glyph bank at 30–60 chars/token** (it already reaches <3% CER with a CRNN), then rgb3 Latin.
4. **Fallback 3: braille-like binary codes.** Treat these as a variant of the raw-bit code, not a shortcut (see C).
5. **Project-level kill:** if no variant passes G4 against the text-side compressor, the result is "text latents dominate." That is a publishable negative result consistent with C3 and the autoencoding paper.

---

## C. Prebuilt solutions and prior art

### Braille and braille-like dot codes
- **Recognition models exist, but for photographed embossed paper, not tiny rendered glyphs.**
  - Angelina Braille Reader: object-detection CNN, with the Angelina dataset (240 annotated photos) and the DSBI double-sided dataset (114 images).
  - Vision-Braille: about 1% character error on single-sided plain braille, with a Braille-Chinese corpus of 376,846 training sentences.
  - e-Braille Tales: 99.97% OCR accuracy for Perkins Brailler output.
  - These are camera or scan pipelines with dots tens of pixels wide, and they don't transfer to 4–5 px cells.
- **Unicode braille as an alphabet.**
  - U+2800–U+28FF gives 256 eight-dot patterns, exactly one byte per cell.
  - At its densest (2x4 px, no gaps), a 28x28 token holds 14x7 = 98 cells per channel. Across 3 channels that is 294 bytes, or 2,352 bits: identical to the raw binary pixel budget. Dense braille is simply a raw-bit code under a different name.
  - With legibility gaps (about a 3x5 px cell) you get about 45 cells per channel, about 135 bytes for rgb3. Its ceiling and risks are the same as the moonshot's.
- **Pretrained-prior advantage: unverified.** I found no benchmark of VLMs reading rendered Unicode braille. A one-hour zero-shot test on Qwen2.5-VL is worth running before assuming any prior. At 2–5 px cells, any web-scale prior is unlikely to apply.
- **Grade-2 contracted braille** is a human-designed chunking scheme analogous to BPE (contractions for frequent words and letter groups). Liblouis is the standard open-source translator for it; I did not verify this in this research. Its value here is conceptual: your learned chunks (" the ") already do the same job, optimised for your reader.

### Other dense codes: what to borrow, not adopt
- **JAB Code:** the in-symbol colour palette is a ready-made per-page colour-calibration idea for rgb3. The library is open source.
- **DataGlyphs:** 1 bit per 5x5 cell is about 25x less dense per pixel than your code. Borrow the self-clocking lattice and synchronization patterns as alignment aids, given the patch-alignment sensitivity above.
- **HCC2D / HCCB:** colour error-correction ideas only.
- **None of these offers a neural reader to reuse.**

### Codebases that can replace pipeline components
| Need | Prebuilt option | What it saves |
|---|---|---|
| Capacity-law measurement | Scaling-Law-for-Vision-Token code (sxzhuang), block-shuffle + pixel-shift protocols | G1–G3 analysis, wall location |
| Trained compressor reference | DeepSeek-OCR and DeepSeek-OCR-2 code and weights | Conv 16x compressor design to replace or augment the 2x2 merger |
| Rendered-text QA | LensVLM repo (rendering presets 5x/10x/15x, eval loop); VTCBench code and data | G4 harness, Latin baselines |
| Long-context rendered pretraining | Glyph (open-sourced, per MarkTechPost) | Rendering configs, continual-pretraining recipe |
| Text-side baseline | C3 code and weights; 500xCompressor; Cramming "hidden_capacity" repo | Matched-KV baseline and per-vector capacity checks |
| Pixel LM pretraining | PIXEL (xplip/pixel, Team-PIXEL/pixel-base, rendered Wikipedia and BookCorpus datasets) | ViT-on-rendered-text pretraining code |

CLIPPO and Pix2Struct are known rendered-text pretraining lines, but I did not re-verify their current code status here.

---

## Recommended next experiments, ranked by expected information per RTX 3090 GPU-hour
1. **Finish the CRNN runs and add a CRNN density sweep (15–322)** (hours). This fixes the identity ceiling (G0) per density and rules out the alphabet itself as the failure point.
2. **Nonlinear glyph-ID probes** on frozen ViT block 32 (G1), then **merger-only training with a CTC head** across the sweep (G2) (a few GPU-hours each). This decides whether the merger is the wall and fills the literature gap.
3. **Grid-alignment test:** re-render at 4/7 px pitch and apply the pixel-shift protocol (under 1 GPU-hour). This could move points out of the unstable band for free.
4. **Zero-shot Unicode braille and tiny-font read test** on Qwen2.5-VL-3B (under 1 GPU-hour). This settles whether any pretrained prior exists.
5. **LoRA + last-4 + merger transcription sweep, Latin vs code** (G3) (about 1–2 GPU-days). This is the primary go/no-go.
6. **Matched-KV QA:** Latin images vs code vs a 500xCompressor/ICAE-style text compressor on Qwen2.5-3B, plus a transcribe-first variant (G4/G4b) (about 2–3 GPU-days). Run it only at densities that passed G3.

## Caveats
- Character-per-token conversions from text-token ratios assume about 4–4.5 chars per token for English. DeepSeek-OCR reports "precision," not CER.
- DeepSeek-OCR and Glyph numbers are on natural prose with language priors. The prior-free (shuffled) results are much lower.
- The ViT-FLOPs and KV-bytes figures are my own arithmetic from model configs as I remember them; check them against the official configs.
- Several 2026 sources (FocusVTC, VLMs Read Aloud, Progressive Cramming, the V4.1-Flash report) are recent preprints and not yet peer-reviewed.