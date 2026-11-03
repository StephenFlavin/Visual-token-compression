# Idea backlog (not yet tested)

Ideas recorded for later. Each one gets a status, a short assessment, and the cheapest test that would settle it.

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
