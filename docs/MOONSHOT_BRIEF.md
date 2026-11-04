# Research brief: the moonshot (RGB-overlaid learned-glyph pages as dense optical context)

Written 2026-11-04 for an external research task. Numbers are from this repo (claude/validation-harness); Q1 and G0
numbers are preliminary.

## What it is
**Visual BPE moonshot, `rgb3_lg`:** render long text as page images that a vision-language model reads in place of
text tokens, packing far more characters into each vision token than ordinary typography.
* **Learned glyph alphabet ("Visual BPE").** A bank of 395 small learned glyphs (4-5 px line pitch). Each glyph stands
  for a single character or a frequent character chunk (dictionary tokens like " the "), so text becomes a stream of
  tiny symbols.
* **RGB overlay.** Three independent text streams on one page, one per colour channel. Each channel is a black-on-white
  glyph page, so the channels don't occlude each other. That is about 3x the density of one glyph page.
* **Density.** At Qwen2.5-VL's geometry (448 px page, 256 tokens of 28x28 px), the variants hold 322 chars per
  vision token (`rgb3_lg_pb_h5`) and 405 (`rgb3_lg_free_h4`). For comparison:
  * text tokens: ~4.3 chars each;
  * plain Latin page images, Glyph-style: ~15;
  * the single glyph bank: ~107.

## Origin and aim
* **Inspiration.** DeepSeek-OCR's "contexts optical compression" (Oct 2025): render text, read it back from far fewer
  vision tokens (~97% OCR precision at ~10x, ~60% at ~20x). DeepSeek-V4.1-Flash, the user's motivating long-context
  model (552B MoE, 1M-token window, 890 B/token KV), is multimodal, but its paper abstract does not say how it encodes
  images.
* **Caveat for the research task.** Check whether V4.1-Flash itself uses optical compression. Its published KV savings
  come from attention design (CSA2, cross-layer KV reuse, FP4 KV), not OCR as far as the abstract says.
* **Aim.** Show a local model on one RTX 3090 (Qwen2.5-VL-3B) holding much more context than its default. A vision
  token is one KV position, like a text token, so chars per position is the saving.
  * Qwen-3B default: ~140k chars in its 32k window.
  * At 322 chars/position: ~10M chars in the same window, with ~98.7% less KV per char and ~0.03x the prefill compute
    per char (estimates).
* **Bar for success.** After equal fine-tuning, the code must beat fine-tuned Latin images by ≥ 1.5x chars/token at
  < 3% transcription CER, and keep long-context QA (needle, cloze) accuracy at ≤ 0.67x Latin's KV bytes.

## Evidence so far
| Test | Result |
|---|---|
| Raw pixels, trained reader (the "identity ceiling": is the page itself readable?) | Single glyph banks, rgb3 Latin and colour codes: all < 3% CER. Moonshot: 15.2% (pb_h5) and 37.0% (free_h4) at step 500 of 3000, still falling; full run (G0, gate < 10%) pending |
| Frozen SigLIP-B/16 | every dense code at the floor (≥ 62% CER); Latin needs ~2 chars/token for 5% at layer 6 |
| Frozen DeepSeek-OCR DeepEncoder (1000 train pages) | colour 71-73%, rgb3 83.5%, glyph banks 82%, Latin ≥ 15 chars/token at the floor; only Latin at 8.5 chars/token reads (15-19%) |
| Q1: layer-wise linear pixel readout of frozen Qwen2.5-VL ViT (preliminary) | moonshot R² 0.69 at block 8 vs Latin 0.97, but equal or better than Latin from block 24 on; all three colour channels survive about equally; nothing linearly recoverable after the 2x2 patch merger (expected: 2048 dims vs 2352 dense pixel values) |
| Fine-tuned Qwen (LoRA + last 4 ViT blocks + merger), long-context QA | not run yet (code built, CPU-tested; GPU runs next week) |

## Hardest parts and where it's failing
1. **Pretrained encoders throw away the fine detail the codes need.** This is the main failure so far. Every frozen
   encoder tested puts every dense code at the floor, while the same pages read near-perfectly as raw pixels. The
   information is in the pixels and lost in the encoder. Open question: can fine-tuning the encoder's later layers
   recover it, or must the early layers / patch stem be retrained (much more expensive)?
2. **Token merging bottleneck.** Qwen merges 2x2 patches (4 x 1280 dims -> 2048) and DeepEncoder compresses 16x.
   A moonshot token carries ~2352 dense pixel values and hundreds of characters. Whether 2048 dims can carry ~322
   chars (the bits per token budget) is unknown in practice.
3. **No language prior.** The glyphs are arbitrary symbols. OCR models lean heavily on language priors: DeepSeek-OCR
   drops from 88-98% to 20-62% on random letters. One misread glyph can corrupt a name or number, so exact recall
   (needle tests) is expected to fail first.
4. **Learnability and decoding.**
   * 1185 channel-tagged classes (3 channels x 395 glyphs).
   * Lines hold ~600 interleaved labels, right at the CTC frame budget (lines wrap 1 px early to fit).
   * Probes learn the moonshot ~2x slower per step than a single bank.
   * The free_h4 variant is clearly harder than pb_h5.
5. **Reading is not reasoning.** The KV saving only counts if the model answers questions directly from the image
   tokens. If it transcribes first, the output refills the KV cache. Prior art: VTCBench finds VLMs OCR rendered text
   but understand long contexts in it poorly, and naive reading at 5x scores 31% vs 72% for text (LensVLM).
6. **Training scale and data.** Probes memorise 200 training pages; ≥ 1000 are needed (Latin through DeepEncoder:
   85% -> 15% from 200 to 5000 pages). Our fine-tune is a LoRA on a 3B model on one GPU, far less than Glyph's
   continual pretraining + RL. A negative result would be weak evidence; a positive one strong.
7. **Competing approaches may dominate.** Text-side learned compression matches or beats the vision path in the
   literature: C3 98% at 20x; a learned hierarchical encoder beats vision on reconstruction perplexity. If we train
   our own encoder, rendering to pixels may be a detour.

## Questions for the research task
* Has anyone trained a VLM or ViT to read dense synthetic visual codes (2D barcodes / data matrix, colour-multiplexed
  or steganographic channels, learned glyph alphabets)? What capacity per token did they reach, and which layers had
  to be trained?
* How do pretrained ViTs handle colour-channel-separated information: is it mixed or kept in early layers? Any work
  on per-channel multiplexing through CLIP/SigLIP-style encoders?
* How does DeepSeek-V4.1-Flash (and DeepSeek-OCR 2) encode images, and does V4.1-Flash use optical compression for
  long context at all?
* Information capacity of a VLM vision token after patch merging: empirical bits/token for text rendered at
  increasing density.
* The strongest current results for long-context QA over rendered text (Glyph, LensVLM, VTCBench, "Text or Pixels? It
  Takes Half") and for text-side compression (C3, ICAE, gist tokens). At equal KV, which wins?
