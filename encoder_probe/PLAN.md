# Frozen-encoder probe: does a pretrained vision encoder keep fine pixel / colour detail?

Sub-project of Visual BPE. It tests the assumption shared by two ideas:
* **Colour-pixel encoding** (one solid b×b colour block per character; user idea, 2026-11-03), and
* **grey-level learned glyphs** (learned_glyphs/: the unconstrained banks win partly through grey-level codes).

Both only help a VLM if its frozen image encoder carries that detail through to its output tokens.
The CRNN results so far (Phase 2, learned_glyphs) read pixels directly, with no pretrained encoder in between.

## Setup
* Page: 224×224 RGB, text in horizontal lines of integer pitch p (line i = rows [i·p, (i+1)·p)), WikiText text
  (train pages from data/train.txt, test pages from data/test.txt), printable ASCII.
* Encodings (px²/char measured on the test pages):
  * `latin_p{4,5,6,8,10}`: DejaVu Sans at em = p/1.2, 4× supersampled (Phase-2 renderer), ≈7–45 px²/char.
  * `lg_free_w23_h3`, `lg_free_h4`: converged unconstrained learned banks (N=300), 4.1 / 5.9 px²/char.
  * `lg_pb_h4`, `lg_pb_h5`: converged whole-pixel learned banks (N=300), 5.9 / 7.5 px²/char.
  * `col_b{1,2,3,4}`: one b×b block per char, 95 colours from a 5-level/channel RGB grid, 1–16 px²/char.
* Encoders (frozen): `identity` (raw 16×16×3 patches = ceiling), `siglip` (google/siglip-base-patch16-224, last layer),
  optional `clip` (openai/clip-vit-base-patch16, penultimate layer). All give 196 tokens of 768 dims (one per 16×16 patch).
* Conditions: `clean` (PNG) and `jpeg75` (PIL JPEG q75, 4:2:0 chroma subsampling), applied before the encoder.
* Probe (identical for every run, trained from scratch): standardised tokens → 3×3 conv over the 14×14 grid →
  per-pixel features (unpatchify) → per-line dilated conv reader → CTC. It can only lose information, so its CER is
  an upper bound on what the frozen tokens destroy. Identity runs give each encoding's ceiling under the same probe.
* Metric: CER on held-out pages vs **chars per vision token** = 256 / px²/char (text tokens ≈ 4 chars/token).
  C@3% = chars per token at which CER reaches 3% (from `vbpe.analysis.p_at` on px²/char).

## Pre-registered predictions (written before the grid ran)
1. identity/clean: col_b1 reaches < 3% CER (the probe can read 1-px colour codes when nothing destroys them).
2. identity/jpeg75: col_b1 > 20% CER (chroma subsampling halves colour resolution); col_b2 also degrades.
3. siglip/clean: col_b1 > 20% CER; the frozen encoder keeps far less than 256 chars per token.
4. Through siglip, the grey-level advantage of lg_free over lg_pb shrinks or reverses.

## Decision rules
* **Colour pixels are worth a GPU follow-up only if**, through frozen SigLIP with JPEG q75, colour blocks reach
  ≥ 2× the chars per vision token of the best Latin rendering at 3% CER. Otherwise the idea is dead for VLMs that
  use a pretrained encoder (it would need an encoder trained for it, i.e. a learned text compressor, not OCR).
* **Learned glyphs:** use grey-level (unconstrained) banks in the GPU plan only if lg_free still beats lg_pb through
  siglip/jpeg75. Otherwise constrain to whole pixels.
* Caveat for both: a probe trained on frozen tokens is not an LLM reading them. A positive result means "the information
  survives the encoder", not "a VLM will read it".
