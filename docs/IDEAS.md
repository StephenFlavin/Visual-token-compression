# Idea backlog

Ideas recorded for later. Each one gets a status, a short assessment, and the cheapest test that would settle it.
The dated status line under each idea is the current state; the text above it is the assessment as first written.

## RGB word overlay (user idea, 2026-11-03)

**Idea.** Overlay three words at the same position, one per colour channel, so the image holds up to 3× more text.
Words go round-robin to channels: "hello my name is jimmy" becomes red "hello" + green "my" + blue "name" in one slot,
then red "is" + green "jimmy" in the next. Each later channel is shifted slightly to the right, so letters that land on
the same spot in two words ("boom" / "zoo": the "oo") are not completely covered.

**Assessment (before any test):**
* With per-channel compositing (each word is ink in its own channel only, not paint over the others) the three
  channels are independent greyscale images. Nothing is covered, so the shift is not needed. The shift only matters
  if colours are painted over each other.
* A per-pixel reader that sees raw RGB (our CRNNs, `identity` in encoder_probe) should get close to 3× the chars per
  pixel of plain Latin at the same font size. The cost is that one line holds three words of different lengths, so
  slots need a fixed width (pad to the longest of the three) or a split into three independent text streams per line
  (simpler and denser).
* It relies on colour detail surviving, like `col_b*`:
  * JPEG 4:2:0 keeps full resolution only in luma (Y = 0.30R + 0.59G + 0.11B), which is a fixed mix of the three
    words. The separating information is in chroma at half resolution.
  * Frozen SigLIP so far gives unigram-level CER on colour-block pages, even on its own training pages
    (encoder_probe, 2026-11-03).
  * So this should be tested after the encoder-probe positive controls show whether any colour detail survives.
* Related: multiplexing 3 bits per pixel into RGB channels (steganography-style). Any version that needs the channels
  separable assumes an encoder that keeps them separable.

**Cheapest test:** add `rgb3_latin_p{5,6,8}` (three independent text streams, one per channel, black-on-white per
channel) to `encoder_probe` and run it through identity/clean, identity/jpeg75 and siglip/clean with the unchanged probe.
Go on only if it beats `latin` chars/token at 3% CER through siglip with JPEG.

**Status:** built, not run (2026-11-03). Encodings `mono_p*`, `rgb3_p*`, `rgbw_p*_{chan,paint}` in
`encoder_probe/ep/rgb_overlay.py`; design, pre-registered predictions, decision rule and the 48-command grid
in [encoder_probe/RGB_OVERLAY.md](../encoder_probe/RGB_OVERLAY.md). Only a 300-step smoke run exists.

**Status (2026-11-05): dead for frozen encoders.** From raw pixels rgb3 works (identity32: rgb3_p6 0.01% CER at
3x mono's chars/token). Through DeepEncoder rgb3 sits at the floor (83.5%, worse than mono's 72-73%), so the SigLIP
grid (P2) was gated off. In frozen Qwen, rgb3_p8 stops being linearly pixel-decodable by block 24 (Q1); it has no
fine-tuned arm yet. The overlay lives on in the moonshot (`rgb3_lg_pb_h5`, below and in ROUND3.md).

## More layers per pixel: intensity levels within each channel (follow-up, 2026-11-03)

**Idea (user):** go beyond 3 overlaid layers by adding more colours, e.g. yellow and black.

**Assessment:**
* Extra hues add no capacity. A pixel has 3 independent values (R, G, B). Yellow = R+G and black = all three, so a
  yellow or black layer overlaps the existing channel layers and can't be told apart where they overlap.
  CMYK is the same 3 dimensions inverted; K exists to save ink, not to carry information.
* The only way past 3 layers is intensity levels within each channel. Example: layer A darkens a channel by 1/2 and
  layer B by 1/4, so each value of that channel tells which of A and B are present. 2 levels per channel gives
  `rgb6` (6 layers); 8-bit channels allow up to 24 bit-plane layers in principle.
* Expected to be fragile:
  * anti-aliased edges are intermediate greys that look like the dimmer layer's ink, so levels collide even on
    clean renders;
  * JPEG, resampling and noise collapse fine level differences, the same failure as `col_b*` and the grey-level
    learned glyphs (unconstrained banks: 55-68% held-out CER vs 6-22% for whole-pixel banks);
  * frozen SigLIP already loses colour-block codes.
* Prediction: `rgb6` works only for identity/clean. It can't beat `rgb3` wherever `rgb3` already loses its advantage,
  because it relies on finer differences.

**Cheapest test (only if `rgb3` keeps most of its 3x through siglip_l6/jpeg75):** add `rgb6_p{6,8}` to
`encoder_probe/ep/rgb_overlay.py` (2 binary-thresholded layers per channel at ink depths 1/2 and 1/4, channel-tagged
classes, 6 interleaved labels per cell; check the CTC frame budget) and run it with the rgb3 grid conditions.

**Status:** recorded; gated on the rgb3 result.

**Status (2026-11-05): gated off, not built.** rgb3 failed through the encoder (DeepEncoder floor), so rgb6 (P3)
was not run.

## Braille-style fixed dot code (control for learned glyphs, 2026-11-04)

**Observation (user):** the converged whole-pixel learned glyphs look like braille. They nearly are: single-char
glyphs are 2x5 binary pixels (10 dots, vs braille's 2x4 cell) and chunk glyphs 4x5. Braille is designed to be told
apart by a blurry low-resolution sensor (a fingertip); the bank is optimised to be told apart after blur, downsampling
and noise, so it converges to the same kind of code: sparse, high-contrast dots on a tiny grid.

**Question:** how much of the learned-glyph gain (whole-pixel bridge: 0.88x / 0.77x Latin's px²/char at 3% / 1% CER)
comes from *learning*, and how much from just using a dense binary dot code on that grid? There is plenty of room:
a 2x5 grid has 1,024 patterns for 95 chars, and a 4x5 grid has about 10^6 for 300 chunks.

**Cheapest test (CPU, no new training code):** build fixed banks with the learned bank's geometry
(h5, widths 2/4 px, `pixel_block`, N=300) and score them through the E3 bridge and E2 transfer exactly like
`e1long` `lg_learned_n300_pb_h5`:
* `GlyphBank.random_binary(..., density≈0.3)`: random dots, already built and unit-tested
  (`tests/test_render.py`) but never run;
* a designed max-distance code: choose the 395 patterns by greedy max-min Hamming distance, preferring isolated
  dots (after a 1-px blur), i.e. a "braille" built for this channel.

Read: if the designed code gets close to the learned bank, the gain is mostly "dense binary code" and learning is
optional; if both fixed codes stay near DejaVu, the learned shapes matter. Also run the harsh channel, where the
learned bank's lead was largest (h7: 3.5% vs DejaVu 43%).

**Status:** recorded, not queued.

**Status (2026-11-05): still not queued.** Still the cheapest open control for the learned-glyph GO; pixel-space
only, so not blocked by the VLM results.

## If the moonshot fails: other avenues (user questions, 2026-11-04)

**User's questions:**
* "If this doesn't work out, are there more avenues worth a go? Write my own encoder? Be happy with what's already
  promising and effective?"
* Goal (user): show a local model on the RTX 3090 holding much more context than it can by default, without
  shortcutting the research (it came from DeepSeek-V4.1-Flash's long context; context and compute numbers per encoding
  are in [CONTEXT_BUDGET.md](CONTEXT_BUDGET.md)).

**1. Bank what works (do regardless): fine-tuned Latin page images + 4-bit KV + YaRN.**
* Latin page images, fine-tuned (Glyph replication on Qwen): prior art reports 3-4x; R3-1's Latin arm tests it.
* Stack two standard, orthogonal methods:
  * 4-bit KV cache: ~4x less KV memory per position.
  * YaRN RoPE scaling: Qwen's 32k window toward ~128k positions.
* Rough potential: default ~140k chars (32k positions x 4.3 chars/position) -> ~1.9M chars (128k positions x ~15
  chars/position), ~13x, with ~1.2 GB of KV. Estimate only: YaRN past the trained length costs accuracy, and the
  three have to be measured together.

**2. Write our own encoder: two very different projects.**
* **2a. Custom front-end for the codes** (= ROUND3.md R3-1b stage B): replace or train Qwen's patch stem and early
  ViT blocks on the dense codes, LLM frozen + LoRA.
  * The identity ceilings show the information is in the pixels (100-1000 chars/token at < 3% CER).
  * Novel; fits the 3090.
  * It is the honest last test of the dense-pixel idea.
* **2b. Skip pixels: learned text-side compression** (gist tokens, ICAE, C3: text -> a few memory vectors).
  * Strong in the literature (C3: 98% at 20x, ~93% at 40x; see prior-art.md), but a crowded field.
  * Once the encoder is ours, rendering to pixels is a detour. The critiques in prior-art.md find text-side
    compressors match or beat the vision path.
  * Pixels only win if the model must stay an unmodified off-the-shelf VLM, or if the documents are already images
    (scans, PDFs).

**Decision ladder:**
1. Always: Latin + 4-bit KV + YaRN (the demo).
2. If the moonshot reads as raw pixels (G0) but not through Qwen (Q1/R3-0/R3-1): try 2a once.
3. If 2a fails, the dense-pixel line is done. That is a publishable negative: dense codes carry 100-1000 chars/token
   in raw pixels, but pretrained VLM encoders can't be taught to keep them. Move to 2b only knowing it's an
   established field, not a new idea.

**Status:** recorded; gated on G0 (moonshot identity28 ceilings), Q1 (layer-wise linear readout) and R3-1 on the 3090.

**Status (2026-11-05): ladder step 2 applies, not yet decided.** G0 passed (moonshot 1.07% from raw pixels); Q1
and G2-lite put the loss in the ViT output / merger ("ViT output is the wall at this budget"); R3-1 at 1000 steps
reads latin_p8 (28 chars/token, 55% CER vs blind 88.5%) but leaves every denser code at its blind floor. Next: a
short vision-side diagnostic (higher `--lr-vision`, more `--vision-blocks`; a light version of 2a) and the 5000-step
Latin arms, which also size option 1 ([HANDOFF.md](HANDOFF.md)).
