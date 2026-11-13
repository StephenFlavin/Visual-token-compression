# Round 1 plan: does a pretrained vision encoder keep fine pixel and colour detail?

**Status: finished 2026-11-03. This file is the pre-registered design of round 1, kept as written.** Results and
tables: [RESULTS.md, round 1](RESULTS.md#round-1-siglip-b16-2026-11-03). Follow-ups: [ROUND2.md](ROUND2.md)
(DeepSeek-OCR's encoder) and [ROUND3.md](ROUND3.md) (Qwen2.5-VL, frozen and fine-tuned).

**What was asked.** Two ideas (one colour block per character; learned glyphs that use grey levels) only help a
vision-language model if its pretrained image encoder passes that fine detail on to its output tokens. Does
SigLIP-B/16, a general image-text encoder, do so?

**What was found.** No. From raw pixels a small reader gets 1x1-px colour blocks at 0.06% character error (256
characters per token). Through frozen SigLIP no colour code gets below 62%, at any block size, at the last layer
or the middle one. Round 2 confirmed it through DeepEncoder (71-73%). Grey-level glyphs lose their advantage
behind the encoder, so learned glyph alphabets stay whole-pixel.

| Pre-registered prediction | Outcome |
|---|---|
| 1. Raw pixels: 1x1 colour blocks under 3% error | pass (0.06%) |
| 2. Raw pixels after JPEG: 1x1 blocks over 20% | pass (28.8%) |
| 3. Through SigLIP: 1x1 blocks over 20% | pass (72.6%) |
| 4. Through SigLIP the grey-level glyphs' advantage shrinks or reverses | pass (reverses at layer 6: 75.9% against 54.7%) |

Both decision rules failed: colour blocks are not worth a GPU follow-up, and grey-level glyphs are not used.

**Terms.** *Vision token:* one vector an image encoder outputs for one patch of the page. *Frozen encoder:* a
pretrained encoder whose weights are not changed. *Probe:* a small network trained from scratch to read the text
back from the encoder's tokens; the same probe on raw pixel patches (`identity`) gives the ceiling. *CER:*
character error rate; a reader that gets nothing from the page scores about 72-75% here. *CTC:* the probe's
sequence loss. *jpeg75:* the page after JPEG compression at quality 75. Code names: `latin_p<N>` plain text with
N-px lines, `col_b<N>` one NxN-px colour block per character, `lg_pb_*` / `lg_free_*` learned glyph alphabets with
whole pixels / with grey levels, `mono_p<N>` fixed-width text, `rgb3_p<N>` three fixed-width texts overlaid, one
per colour channel. Full glossary: [RESULTS.md](RESULTS.md#terms-used-on-this-page).

## The design as pre-registered

Sub-project of visual-token-compaction. It tests the assumption shared by two ideas:
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
* Probe (identical for every run, trained from scratch): standardised tokens -> [3x3 conv over the 14x14 grid
  (768->1024) -> GELU -> 1x1 conv] + 1x1 linear skip -> unpatchify to a 3-channel per-pixel map (a reconstructed
  page) -> per-line pixel strip (rows [i*p, (i+1)*p)) -> dilated residual Conv1d reader shared over columns ->
  3 frames/column -> CTC. The map gets an auxiliary MSE loss (weight 10) to the clean page pixels of the
  *training* pages, so the reader only ever sees what the tokens let the probe redraw as an image. AdamW (fused) lr 2e-3, OneCycle, 3000 steps,
  32 lines/step from 200 train pages, bf16 autocast. The probe can only lose information (the pixel loss uses
  training pages only, never test pages), so its CER is an upper bound on what the frozen tokens destroy.
  Identity runs give each encoding's ceiling under the same probe.
  **Probe changes before the grid (2026-11-03), all on identity / pilot encodings, none tuned on results:**
  1. The pilot probe (same map without skip, aux loss or gate; c=6; lr 1e-3) failed: after 1500 steps
     identity/col_b1 72% CER (unigram level), siglip/latin_p6 72%, siglip/col_b2 98%. Its reader learns in
     ~100-700 steps on true pixels, but CTC alone never bootstraps the token->pixel map (also not with a purely
     linear map or a linear skip).
  2. A token-resolution probe (per-dy input projections, per-token heads emitting 16 columns) learned fast but
     memorised the 200 train pages: identity/latin_p8 train loss 0.006, test CER 13.7% (one aborted grid row,
     kept in results/aborted_v2/).
  3. Probe 3 (= final probe + 5 gated free channels). Debug runs at 1500 steps: identity/col_b1 1.5% test CER, identity/latin_p8 2.3% at
     1000 steps (with aux weight 1 and no gate: 60%; weight 1 + gate: 25%); 1000 train pages did not help.
  4. Probe 3 had 5 extra "free" feature channels (gated, gate initialised at 0). In the first grid attempt
     (7 identity/clean rows, kept in results/aborted_c8/) latin_p10 memorised its training pages (train CTC 0.07,
     test CER 77%; p8 0.13%, p6 0.17%). Same run with seed 1: 1.5% at 1000 steps; without free channels:
     0.26% at 1000 steps. The free channels are a memorisation path, so the grid was restarted without them
     (final probe above). Decided on identity runs only, before any SigLIP grid row existed.
  5. Scoring fix (18:04, mid-grid): colour refs are raw fixed-length chunks that can start or end with a space,
     but hypotheses were whitespace-normalised, adding a ~0.4% CER floor to colour rows. Hypotheses are no
     longer stripped for colour encodings (rows carry `score_v: 2`); the 4 identity/clean colour rows scored
     before the fix were re-run after the grid.
  Pilot rows in results/pilot are from probe 1 and are not used.
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

## Post-hoc positive controls (added 2026-11-03 19:50, after the siglip/clean rows; sanity checks, not retuning)
Every siglip/clean row of the grid was 58-75% CER (near the unigram floor), including latin_p10 at 5.5 chars/token,
and the siglip latin runs overfit (latin_p8: train CTC 0.13, test 58%). That cannot separate "SigLIP's last layer
destroys the detail" from "this probe cannot decode SigLIP features". Controls (results/controls*/), queued ahead
of the remaining grid rows; siglip/jpeg75 is deferred until they are read:
1. siglip/clean latin at large pitch: latin_p16, latin_p24 (>= 1 line per patch row).
2. Intermediate layer: `siglip_l6` (output of layer 6 of 12, no final layernorm) for latin_p8/p10/p16, col_b2/b4.
3. Probe capacity on siglip/clean latin_p10 only: 2 residual token-MLP blocks before the grid map (`--token-mlp 2`),
   and 1000 instead of 200 training pages.
Rule: if none of these gets latin below ~10% CER, the probe/encoder pair is reported as uninformative, not as a
verdict on colour. If one does, the colour, latin and lg rows are rerun under that setting with the identical probe.
Control outcome (20:30): siglip_l6/clean latin_p16 = 4.9% CER (the only control below 10%); siglip/clean latin_p16
26.8%, p24 39.5% (overfitting: train CTC 0.002); siglip_l6 latin_p8 26.4%, p10 37.3%. Following the rule, the full
set of encodings (latin p4-p24, colour, lg) is run through `siglip_l6` clean and jpeg75 with the identical probe
(results/controls/). siglip (last layer)/jpeg75 is dropped: siglip/clean is already at the floor for every encoding.
