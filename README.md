# Visual BPE

**Can text be drawn densely enough that a model reads more of it per vision token than per text token?**
A model holds a fixed number of positions in its context window, and a vision token takes one position just like a
text token. Plain text averages about 4.3 characters per token, so Qwen2.5-VL-3B's 32k window holds ~142k characters
of text. A page image that carries 30-300 characters per vision token, and is still read correctly, would let the
same model hold millions. This repo tests codes for doing that (glyphs for frequent letter chunks, colour pixels,
RGB overlays and learned glyph alphabets) and compares them with plain text rendered equally densely.

## Goals (2026-11-10)

1. **Can a learned glyph alphabet beat Latin through a trained vision encoder?** Stage A on the RTX 3090: the 7x7
   and 4x7 alphabets against 8-px Latin, gates A1 (learnable at all) and A2 (matches Latin at 28 chars/token).
   Stage C (the dense 2x7 and BPE alphabets against 6- and 5-px Latin) only if both pass.
   [PLAN_3090.md](docs/PLAN_3090.md) amendments 3-4.
2. **How many bits per vision token can the pipe carry at all?** Random bits painted as colour blocks (96, 294 and
   1176 bits per token offered), read through the same trained encoder into the frozen LLM; the information
   recovered is exact because the source is uniform. The result is a floor on the pipe's capacity, not a ceiling.
   Amendment 6.
3. **Already found, with no training:** rendering plain Latin text as an image gives more text per token, with a
   loss that grows with density.

   | Reader | Rendering | Chars per vision token | Text tokens' worth per vision token (at 4.33 chars) | Quality |
   |---|---|---|---|---|
   | Qwen2.5-VL-3B, frozen | 16 px Latin | 6.8 | ~1.5x | 98% of the information recovered, 1.5% transcription error: near-lossless |
   | Qwen2.5-VL-3B, frozen | 8 px Latin | 28.1 | ~3.4x in information | only 52% of the information recovered; it cannot read the page out (76% error) |
   | Claude (Phase 1) | Latin at 29.4 px²/char | ~27 | 7.8x fewer input tokens (measured) | 3% transcription error, but question answering over a 500k-char document collapses: 33-42% vs 100% from text |
   | Claude, QA test | Latin at 58.8 px²/char | ~13 | 4.3x fewer | 85-95% vs 100% from text |
   | Claude, QA test | Latin at 117.6 px²/char | ~7 | **2.2x fewer** | **100%, same as text** (also at 1.5x) |

   Sources: Run 0 in [PLAN_3090.md](docs/PLAN_3090.md), Phase 1 in [RESULTS.md](docs/RESULTS.md), the Claude QA test
   in [results/claude_qa/summary.md](results/claude_qa/summary.md) (one 500k-char document, 80 exact-answer questions;
   Claude's tokenizer measured 2.87 chars per token on it).

**Context.** For more context on a local GPU, text-side compressors and the model's own architecture (linear
attention, KV quantization) are the stronger routes: a learned text encoder matches or beats DeepSeek-OCR's vision
encoder at every compression ratio tested ([prior-art.md](docs/prior-art.md)). This repo answers the two questions
above; it does not claim that images beat text-side compression.

> **Status (2026-11-06)**
>
> | | Result |
> |---|---|
> | **Settled** | **Hand-picked chunk glyphs (Phases 1-2): KILL.** With Claude, PP-OCR and trained CRNN readers, plain Latin wins. |
> | | **Learned glyph alphabet, pixel space: qualified GO.** With a trained reader it needs about 0.57x Latin's px²/char at 3% CER. |
> | | **Frozen VLM encoders (SigLIP, DeepSeek-OCR's DeepEncoder): every dense code is lost.** Only low-density Latin (8.5 chars/token) reads through DeepEncoder. |
> | | **Qwen2.5-VL-3B, raw pixels and frozen:** the moonshot code (322 chars/token) reads from raw pixels at 1.07% CER, but its pixel detail collapses (linearly) at the ViT's 2x2 merger. A CPU probe's verdict: "the ViT output is the wall at this budget". |
> | **Open** | **R3-1, LoRA fine-tune of Qwen's LLM (1000 steps):** Latin at 28 chars/token is read (55% CER vs 88.5% blind). Latin at 72 chars/token gets the gist but not the words (val loss 10% below blind, MC above blind at p = 0.02). Learned glyphs (107) and the moonshot (322) are not read at all: their losses equal the blind runs'. Gate G3 is undecidable: no arm is near 3% CER. [Analysis](docs/ANALYSIS.md). |
> | **Open** | **Own encoder, Stage 0** (text compressor into frozen Qwen): inconclusive, its 4x positive control failed (64% CER at 8000 steps); the decoder keeps the gist, not the words. [OWN_ENCODER.md](docs/OWN_ENCODER.md) |
> | **Now** | **Gist through colour channels** ([COLOUR_GIST.md](docs/COLOUR_GIST.md)): Latin in 2-3 colour channels, then glyphs without BPE, then merge. Untrained, Qwen reads the overlay as one blended text (2-stream info per vision token 0.51x one stream's). A CPU fine-tune with the vision encoder frozen reached 0.72x. Both streams survive inside the encoder to block 16, then blend. **Next (GPU):** train the encoder from block 16, against the > 1x headline. |

**Terms.**
* **CER:** character error rate (edit distance / reference characters).
* **px²/char:** image area per source character (lower is denser).
* **Chars per vision token:** characters carried by one image token of a given encoder (higher is denser).
* **Blind control:** the same fine-tune with the images shuffled between pages, so the image says nothing about
  the answer. Its scores are the floor that the model's language prior reaches on its own.

## The codes

**1. Glyphs for frequent chunks (visual BPE).** Byte-pair encoding merges frequent letter runs (`the`, `ing`, ` of the `)
into one token. Here each of the 300 most useful chunks gets its own glyph, and every other letter is spelled
out. In the version that works, all 395 symbols are learned 5-px bitmaps, optimised together with a reader.
The haiku below goes from 66 characters to 32 symbols, and its lines get 36% shorter.

![Plain text vs learned glyphs](docs/img/glyphs.png)

**2. Colour blocks.** Each of the 95 printable characters gets its own colour, and each character is drawn as one solid block
(here 2x2 px, 60% less area than a 5-px font).

![Plain text vs one colour block per character](docs/img/colour.png)

**3. RGB overlay.** Three lines of text share one strip of pixels, one line per colour channel. That is 3x the text of a
monospace line.

![Three lines of text vs RGB overlay](docs/img/rgb.png)

**The moonshot, `rgb3_lg_pb_h5` = codes 1 + 3.** It blends the learned glyph alphabet (1) with the RGB overlay (3):
three independent streams of learned glyphs, one per colour channel, on the same pixels.
* **Why the two combine:** each channel is an ordinary black-on-white glyph page, so the three layers never
  occlude each other. Code 1 packs ~107 chars into each Qwen2.5-VL vision token; three layers make that **322**,
  ~75x plain text's 4.3 chars per token.
* **Status:** the page itself is readable (1.07% CER for a reader trained on raw pixels), but Qwen2.5-VL has not
  learned to read it: after a 1000-step fine-tune its loss equals its blind control's. Details in
  [docs/ANALYSIS.md](docs/ANALYSIS.md).

![Three lines of text vs learned glyphs overlaid one per colour channel](docs/img/moonshot.png)

A strip of a real moonshot page, magnified: the colour page (top) and its three channels.

![Moonshot: RGB overlay of learned whole-pixel glyphs](docs/img/rgb3_lg_pb_h5_strip.png)

## Results by stage

![Chars per vision token vs whether each code is read, per encoder and stage](docs/img/readme_density.png)

*Each strip is one encoder and stage. x is chars per vision token of that encoder, so one rendering sits at different x
for different encoders. The dotted line is DeepSeek-OCR's published operating point: ~97% OCR precision at 10x
compression (text tokens per vision token, Fox benchmark), converted at 4.33 chars per text token, so its position
is approximate. Generated by `scripts/make_readme_figures.py` from the result rows.*

**1. Hand-picked glyphs (Phases 1-2): KILL.** Common Chinese characters stood in for frequent chunks, and a
Hangul-style layout packed 2x2 letters per cell. A Chinese glyph needs a larger font to stay legible, which cancels
the width it saves. With a CRNN reader trained to convergence, plain DejaVu Latin reaches 1.73% CER at 7.5 px²/char;
substitution needs ≥1.18x its pixels and quadrant packing ≥3x. Claude scores only natural text (its safety classifiers
refuse the coded pages). On plain English it needs 29.4 px²/char for 3% CER, about 37 vision tokens per 1k chars.
→ [docs/RESULTS.md](docs/RESULTS.md)

**2. Learned glyph alphabet: qualified GO (pixel space).** Glyphs learned jointly with a reader need 0.57x the
pixels of plain DejaVu Latin at 3% CER in the unchanged Phase-2 pipeline (3.75 vs 6.54 px²/char), and 0.63-0.74x with
a fresh reader. Under heavy blur, noise and JPEG they get 3.5% CER where DejaVu gets 43%. There are two caveats:
* glyphs restricted to whole pixels win by only 12-23%;
* the densest alphabets break on degradations they never saw in training.

→ [learned_glyphs/RESULTS.md](learned_glyphs/RESULTS.md)

**3. Frozen VLM encoders: every dense code is lost.** A small probe reads the text back from frozen encoder tokens.
* **Raw pixels:** every code reads at 3.5% CER or better, including colour 1x1 blocks at 256 chars/token (0.06%).
* **Through SigLIP-B/16:** no colour code gets below 62% CER.
* **Through DeepSeek-OCR's DeepEncoder:** colour blocks (71-73%), the RGB overlay (83.5%) and learned glyphs (82%)
  sit at the floor. Only Latin at 8.5 chars/token reads (19%).

The probe is data-limited, so these CERs are upper bounds on what each encoder loses. The conclusions rest on
comparisons made with the same probe. → [encoder_probe/RESULTS.md](encoder_probe/RESULTS.md)

**4. Round 3, Qwen2.5-VL-3B: can a VLM be trained to read a dense code?** → [encoder_probe/ROUND3.md](encoder_probe/ROUND3.md)
* **G0 (raw pixels):** the moonshot reads at 1.07% CER at 322 chars/token.
* **Q1 (linear pixel readout of the frozen ViT):** whole-pixel glyph codes stay pixel-like up to the last ViT
  block. Every code collapses at the 2x2 merger.
* **G2-lite (CPU probe, 500 pages):** a trained merger barely helps (84.6% vs 86.1% frozen). Skipping the merger
  reads about 29% of the moonshot's characters, against about 15% through the merger. The pre-registered verdict:
  "the ViT output is the wall at this budget".
* **R3-1 (LoRA fine-tune of the LLM, plus the merger and last 4 ViT blocks):** 1000 steps on an RTX 4090, every arm
  with a blind control. Latin p8 is read: CER 55% vs 88.5% blind, multiple-choice accuracy 0.88 vs 0.25, validation
  loss 1.12 vs 2.48. It reads the first line of a window almost exactly, then drifts into invented text. The denser
  arms sit at their blind floors; the moonshot's losses match its blind run to 3 decimals.

![R3-1: CER and multiple-choice accuracy per arm next to its blind control, 1000 steps](docs/img/readme_r3_1.png)

So far, no dense code is read by a VLM. The 1000-step budget is short, so this is "not reached at this budget",
not a final no. What each code would buy in context and KV cache is in
[docs/CONTEXT_BUDGET.md](docs/CONTEXT_BUDGET.md); none of it is a working context gain until R3-2 measures it.

## How to run

Full setup, every experiment command and the known gotchas: [AGENTS.md](AGENTS.md). Quick start (CPU):

```bash
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
scripts/fetch_fonts.sh
python3 scripts/build_data.py              # WikiText-103 passages and chunk dictionaries (~5 min)
python3 -m pytest -q tests                  # fast tests, no network
python3 scripts/make_readme_figures.py      # README figures (--only charts: just the two charts)
```

GPU work has its own runbooks: [docs/GPU_RUNNER.md](docs/GPU_RUNNER.md) for any Linux GPU runner (round-3
fine-tunes, sized from a smoke run), and [docs/LOCAL_GPU.md](docs/LOCAL_GPU.md) for the local Windows RTX 3090.
The latest run's status is in [docs/HANDOFF.md](docs/HANDOFF.md).

## Repo map

| Path | What |
|---|---|
| `PLAN.md` | design, pre-registered predictions and kill criteria for Phases 1-2 |
| `vbpe/` | layout engine, rendering schemes, CRNN reader, Claude and PP-OCR readers, analysis |
| `experiments/` | Phase 1 (Claude), 1b (PP-OCR zero-shot) and 2 (trained CRNN) drivers |
| `learned_glyphs/` | learned glyph alphabet: plan, code, results, saved glyph banks |
| `encoder_probe/` | frozen-encoder probe (rounds 1-2: SigLIP, DeepEncoder) and round 3 (Qwen2.5-VL) |
| `docs/` | results, context budget, GPU runbooks, handoff, literature notes, ideas, figures; index in [docs/README.md](docs/README.md) |
| `scripts/` | data building, fonts, job queue, figure scripts |
| `results/` | Phase 1-2 rows and summaries; [results/README.md](results/README.md) says where every stage's rows live |
| `tests/` | CPU tests for the core package |
