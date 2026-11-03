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

**Status:** recorded; not queued. Run it after the encoder_probe controls.
