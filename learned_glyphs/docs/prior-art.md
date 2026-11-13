# Prior art: learned glyph alphabets for machine reading (WP-LG4)

**Status: a record.** The literature search of 2026-11-02 that shaped the learned-glyph plan, kept as written.
Its claims about other work come from the papers in [References](#references) and were not re-checked in the
clean-up of 2026-11-13; entries marked "unverified" were never checked.

**How its warnings held up** (from [../RESULTS.md](../RESULTS.md), re-checked against the rows):

| Warning or recommendation here | What happened |
|---|---|
| The bank may only work on the simulated channel (point 3) | Partly true. Unrestricted banks read at 35-68% error on degradations they were not trained on; whole-pixel banks at 12-22%. The bridge score (the unchanged Phase 2 pipeline) still shows 0.57x Latin's pixels |
| A per-character gain above about 3x means the bank is exploiting the channel (point 4) | Not triggered: the largest gain is 1.74x |
| Must-haves (point 5): hard binary glyphs, test-only distortions, a fresh-reader transfer test | Done: banks are binarised from 60% of training on, every joint run has a held-out degradation score, and E2 is the transfer test |
| Must-have: a random-codebook baseline; also a 3x2 block code and a 4x6 pixel font | Never run. `GlyphBank.random_binary` exists and is unit-tested only |
| Nobody has learned glyphs as input for a vision-language model (point 1) | Tried here: frozen encoders lose the banks and a short fine-tune of Qwen2.5-VL-3B does not read them |

*Searched 2026-11-02. This adds to `../../docs/background/prior-art.md` ([P]), which already covers optical compression, PIXEL, FontCode and Neural Glyph. "Unverified" means seen only in an abstract or a secondary summary. "(ours)" marks our own arithmetic.*

## Bottom line

1. **Nobody has jointly learned glyphs and a text reader to pack more text into fewer pixels.** Nobody has learned glyphs for multi-character tokens or as VLM input either. The nearest work is learned markers and hidden-data codes (E2ETag, DeepFormableTag, StegaStamp), drawing games between agents, and hand-optimised "smartfonts". My confidence in this is moderate.
2. **Human studies suggest some real headroom.** People read smartfonts at under half Latin's size when blurred, and 3×2 block alphabets beat ordinary ones about 3× for humans in noise. The gain for a machine may be smaller, because our trained reader already reads Latin smaller than the tiniest human pixel fonts.
3. **The main risk is a glyph bank that only works on our simulated channel.** In StegaStamp, codes trained without spatial distortions decoded at chance on real photos (0.49 against 0.99 for the full model). Random sub-pixel shifts and resampling mattered more than noise.
4. **The ceiling is low (ours).** Latin carries about 0.3–0.45 bits/px. A binary code can carry about 1 bit/px, and grey-level coding through a clean channel more than 2. A per-character gain above about 3× therefore means the bank is exploiting the channel.
5. **Must-haves:** hard binary glyphs, a minimum feature size, test-only distortions the model never trained on, a fresh-reader transfer test, and a random-codebook baseline.

## 1. Fonts and visual codes optimised for machines

| Work | What is optimised | Distortions modelled | Reported result |
|---|---|---|---|
| OCR-A/B, FE-Schrift [11] | Hand-designed, distinct glyphs | none | no density claims |
| Smartfonts/Livefonts (Bragg, Kalai et al.) [1,2] | 3×2-block scripts; shapes and colours picked by exact search over a crowdsourced confusion matrix | Gaussian blur, size | readable at "smaller than half the size" of Latin when blurry (humans); harder to learn |
| Unadversarial examples [3] | Object patches and textures tuned so a fixed classifier recognises them | simulation and physical tests | gains claimed; numbers not checked |
| E2ETag [4] | 128² grey marker, learned with its detector, never binarised | motion blur ≤10 px, white balance, contrast, noise, warp; no JPEG | precision 0.93 vs 0.51 for AprilTag on simulated data; failed at 12–18 px; real-world AP 0.04 in [5]'s test |
| DeepFormableTag [5] | colour markers, through a differentiable simulator | specular highlights, lens distortion, defocus and motion blur, colour shifts, noise, bending | 36 bits; real-world AP 60.8 vs 57.6 for AprilTag and 50.2 for ArUco |
| HiDDeN [6], StegaStamp [7] | bits hidden in images | blur, crop, JPEG approximation; print-and-photograph simulation | StegaStamp: 56 bits after error correction, 98.7% bit accuracy over 1890 real captures |

- **Learned codes beat hand designs only on the distortions they trained on.** E2ETag's collapse in real photos shows what an incomplete channel costs.
- **No paper reports text density against standard fonts.**
- **Possible near-duplicates (unverified):** an Aalto MSc thesis tuning font parameters against OCR (PDF returned 403); MSTypography [8], an "OCR constraint" for artistic lettering.

## 2. Emergent visual alphabets

- **Neural Glyph [P]** adds no distortion between its agents.
  - It needed a pretrained reader: a reader trained from scratch never produced distinct symbols.
  - It also needed early stopping, because longer training produced duplicate symbols, and some runs collapsed a symbol to a dot.
  - Accuracy falls from 99.8% with 4 symbols to 88% (±18) with 512.
- **Sketching games** [9,10,12,13] find that symbols drift from pictorial to conventional, driven by mutual adaptation. Without a perceptual loss, the protocols are unreadable to humans [9].
- **None measures information per pixel or models degradation.** They are warnings about collapse and co-adaptation, not baselines.

## 3. How learned-code papers avoid channel overfitting

- **StegaStamp [7]** is the template.
  - Distortions: perspective warp of about ±10%; motion blur of 3–7 px; Gaussian blur with σ of 1–3 px; brightness ×0.5–1.5 with offset ±0.3; hue shift; noise with σ up to 0.2; JPEG at quality 50–100 with a cubic rounding approximation.
  - Every distortion ramps up from zero, and warp ramps slowest.
  - Real-world bit accuracy by training regime:

    | Distortions trained | Real-capture accuracy |
    |---|---|
    | none | 0.49 |
    | pixel-wise only | 0.51 |
    | spatial only | 0.89 |
    | all | 0.99 |
- **HiDDeN [6]** must be retrained for each new distortion type.
- **Distortion-agnostic watermarking [14]** generates distortions adversarially and adds error-correcting codes, and does better on distortions it never saw.
- **Diff-JPEG [15]** is a differentiable PyTorch JPEG close to the real codec.
- **Real validation:** marker papers test on unseen real photos; our equivalent is E3 plus test-only degradations.

**Proposed `harsh` preset:**
- Geometry:
  - random sub-pixel shift in x and y;
  - random non-integer scale 0.8–1.25, resized with a random box, bilinear, bicubic or Lanczos filter (bicubic and Lanczos ringing mimics VLM preprocessing);
  - rotation ±1.5°, baseline jitter ±0.5 px, glyph-spacing jitter ±0.5 px.
- Blur: σ 0.3–1.2 px.
- Tone: gamma 0.6–1.6, contrast 0.3–1.0, offset ±0.2.
- Noise: σ up to 0.1, plus sparse salt-and-pepper.
- JPEG at quality 40–90 (Diff-JPEG, or real JPEG with a straight-through gradient).
- Held out for test only: real PIL JPEG, a resize filter never seen in training, Poisson noise.

## 4. Is Visual BPE with learned glyphs new?

- **Nothing found combines tokenisation with learned glyph design, or feeds learned glyphs into a VLM or optical compression system.** Optical compression only searches over font, size and DPI ([P]: Glyph).
- **Name collision:** pixel BPE [16] merges quantised image codes, not rendered text, so "Visual BPE" may confuse readers. A preprint finds that whole-sequence glyph images beat index embeddings for Chinese [17] (unverified).
- **A cheap extra run worth doing:** borrow the unadversarial idea [3] and optimise glyphs against a frozen OCR model or VLM encoder. This removes reader co-adaptation.

## 5. How far are human fonts from the machine optimum?

- **No estimate exists for machine readers at 5–8 px x-height.** The capacity of 2-D blurred binary channels is known to be hard to compute; only bounds exist for specific blur kernels [18].
- **Human evidence:**
  - Letter-identification efficiency is inversely proportional to perimetric complexity [20].
  - Random 3×2 block alphabets are about 3× as efficient as traditional ones (Pelli's result, as summarised in [1]).
- **The smallest human pixel font** is Tom Thumb, at 4×6 = 24 px²/char [19]. Our reader already reads Latin at 10–15 px²/char.
- **Ceiling arithmetic (ours):**
  - English text has roughly 4.1–4.5 bits/char of single-letter entropy, so Latin carries about 0.3–0.45 bits/px.
  - A binary pixel with mild blur gives at most 1 bit/px.
  - A pixel with 8-bit grey levels and noise σ = 0.05 gives about 2.3 bits/px.
  - Honest per-character headroom is therefore about 1.5–3×, less under the harsh channel. Tokens add segmentation gain on top.
  - Anything above about 3× suggests the bank is encoding information in grey levels or in a fixed pixel alignment.

## 6. Recommendations and failure modes

**Design**
- **Binarisation:** anneal τ, then switch to hard ink; score only hard banks, or grey ink becomes a brightness code.
- **Minimum feature size:** penalise ink that changes under a one-output-pixel morphological opening or closing, measured at the 4× supersampled resolution. This removes detail that survives only one pixel alignment.
- **Gaps between cells:** force at least one supersampled column of zero ink at each cell edge, so neighbouring glyphs cannot interfere.
- **Ink coverage:** keep it within a soft band of about 10–45%.
- **Complexity:** add a small perimetric-complexity or total-variation penalty.
- **Distinctness:** add a margin loss on pairwise distances between glyphs after the channel. Monitor the confusion matrix.
- **Rare tokens:** use frequency-balanced symbol sampling, so rare tokens get trained.
- **Reader co-adaptation:** periodically reset the reader, or train against 2–3 readers.
- **Curriculum:** ramp the channel up from zero.

**Evaluation**
- Report bits/px² as well as px²/char.
- Add three baselines:
  - a frozen random binary codebook;
  - a hand-made 3×2 block code (Visibraille-style, from the smartfonts paper);
  - Tom Thumb as a pixel-font Latin.
- Base headline numbers on the test-only distortions and on E2/E3.

**Likely failure modes**
1. **Simulator-only codes:** banks that encode in grey levels or pixel alignment win on `default` and then vanish under `harsh` or in E3.
2. **Co-adapted readers:** only the co-trained reader can read the glyphs, so the E2 transfer test fails.
3. **Symbol collapse:** symbols turn into dots or duplicates.
4. **CTC misalignment:** wide token cells break CTC alignment.
5. **No VLM transfer:** a real gain may still not carry over to VLMs, which have no prior for a new script ([P] §2).

## References

1. Bragg et al. Reading and Learning Smartfonts. UIST 2016. https://www.microsoft.com/en-us/research/publication/reading-and-learning-smartfonts/
2. Bragg et al. Designing and Evaluating Livefonts. UIST 2017. https://www.microsoft.com/en-us/research/publication/designing-and-evaluating-livefonts/
3. Salman et al. Unadversarial Examples. NeurIPS 2021. https://arxiv.org/abs/2012.12235
4. Peace et al. E2ETag. BMVC 2020. https://arxiv.org/abs/2105.14184
5. Yaldiz et al. DeepFormableTag. TOG 2021. https://arxiv.org/abs/2206.08026
6. Zhu et al. HiDDeN. ECCV 2018. https://arxiv.org/abs/1807.09937
7. Tancik et al. StegaStamp. CVPR 2020. https://arxiv.org/abs/1904.05343
8. Yang et al. MSTypography. 2026. https://arxiv.org/abs/2609.37141
9. Mihai, Hare. Learning to Draw. NeurIPS 2021. https://arxiv.org/abs/2106.02067
10. Qiu et al. Emergent Graphical Conventions. NeurIPS 2022. https://arxiv.org/abs/2111.14210
11. FE-Schrift. https://en.wikipedia.org/wiki/FE-Schrift
12. Verma, Dhar. Emergence of Writing Systems. AAAI 2020. https://arxiv.org/abs/1910.00741
13. Spiegel et al. Visual Theory of Mind / Signification Game. 2025. https://arxiv.org/abs/2502.01568
14. Luo et al. Distortion Agnostic Deep Watermarking. CVPR 2020. https://arxiv.org/abs/2001.04580
15. Reich et al. Differentiable JPEG. WACV 2024. https://arxiv.org/abs/2309.06978 (code: https://github.com/necla-ml/Diff-JPEG)
16. Zhang et al. From Pixels to Tokens. ICLR 2025. https://arxiv.org/abs/2410.02155
17. Full Glyph Images Beat Token Embeddings (2026 preprint, unverified). https://pith.science/paper/2607.03994
18. 2-D ISI channel capacity (attribution unverified). https://arxiv.org/abs/cs/0502077 ; https://www.ece.mcmaster.ca/~hranilovic/publications/articles/08/halftoning_iee_postprint.pdf
19. Tom Thumb font. https://robey.lag.net/2010/01/23/tiny-monospace-font.html
20. Pelli et al. Feature detection and letter identification. Vision Research 2006. https://pubmed.ncbi.nlm.nih.gov/16808957
