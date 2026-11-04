# Visual BPE

**Can we fit more text into fewer image pixels?**

Vision-language models can read text from images, and an image costs a fixed number of vision tokens per pixel
area (for Claude, about one token per 28×28 px). If text could be drawn more densely and still read back
correctly, a long document would cost fewer tokens as an image than as plain text.

**What it's for.** A model can only hold a fixed number of tokens at once (its context window), and memory and cost
grow with every token. If a page image carries more characters per token than text does, the same model holds more
document: Qwen2.5-VL-3B's 32k window fits ~142k characters as text, or ~1-2M at 30-60 characters per vision token.
Numbers per encoding: [docs/CONTEXT_BUDGET.md](docs/CONTEXT_BUDGET.md).

This repo tests three ways of packing text more densely. Each was tried against plain text with readers that
were *trained* on the rendering, so neither side gets an unfair advantage.

> **Where it stands (2026-11-04).** All three codes hold far more text per pixel than plain text *if the reader sees
> the raw pixels*. Through **frozen** vision encoders (SigLIP, DeepSeek-OCR's) every dense code is lost. The open
> question, now being tested, is whether a VLM can be *trained* to read them.

> **How to read the numbers.** **px²/char** is image area per source character (lower is better). **Chars per
> vision token** is how many characters one image token carries (higher is better; a normal text token holds
> about 4). We compare renderings at the **same error rate**, usually 3% character error rate (CER).

---

## 1. Glyphs: one symbol for a frequent chunk of letters

Byte-pair encoding (BPE) merges frequent letter sequences such as `the`, `ing` and ` of the ` into single
tokens. *Visual* BPE does the same on the page: each of the 300 most useful chunks gets its own glyph, and the
rest of the text is spelled letter by letter. So `something` becomes `s` + glyph(`ome`) + `th` + glyph(`ing`).

![Plain text vs learned glyphs](docs/img/glyphs.png)

The haiku goes from **66 characters to 32 symbols**, and its lines get **36% shorter** at the same line height.
Every symbol gets a learned glyph, not just the chunks: the 95 single characters are redrawn too (grey cells),
as 5-px bitmaps optimised with a reader to stay legible at tiny sizes. The saving comes from both:
* **better letter shapes:** learned single letters alone already beat a normal font (~0.8× the pixels);
* **chunk glyphs:** several characters per symbol saves another ~26% on top (0.74×).

The full alphabet is in the [glyph atlas](learned_glyphs/results/e1long/atlas/lg_learned_n300_pb_h5.png).

An earlier version used common Chinese characters as the chunk glyphs and kept normal letters. That **failed**:
a Chinese character needs a larger font to stay legible, which cancels the width saved (≥1.18× the pixels of
plain text at equal accuracy).

**Result:** learned glyphs need **0.57×** the pixels of plain text for the same accuracy, so about 1.75× more
text fits per image (whole-pixel version: 0.77–0.88×). Through **frozen** vision encoders the glyphs don't survive
(SigLIP layer 6: 53–55% CER; DeepSeek-OCR's encoder: 82%), though whole-pixel glyphs hold up better than grey-level
ones. → [learned_glyphs/RESULTS.md](learned_glyphs/RESULTS.md)

## 2. Colour pixels: one coloured block per character

Skip letter shapes altogether: give each of the 95 printable characters its own colour and draw one small
solid block per character.

![Plain text vs one colour block per character](docs/img/colour.png)

A 2×2-pixel block per character uses **60% less area** than the haiku in a 5-px font: the whole haiku fits in
a 34×8-pixel block. A 1×1 block (a single
pixel per character) would use 90% less.

**Result:** a reader that sees the raw pixels decodes this almost perfectly (1×1 blocks: 0.06% CER, 256 chars per
token, 7.5× plain text), and 2×2 blocks still survive JPEG compression (3× plain text). But once the image passes
through a frozen pretrained vision encoder, the colour code is gone: SigLIP never gets below 62% CER, and
DeepSeek-OCR's OCR-trained encoder 71–73%. → [encoder_probe/RESULTS.md](encoder_probe/RESULTS.md)

## 3. RGB overlay: three lines in one, one per colour channel

An image has three colour channels. Write a different line of text into each one (red, green, blue) at the same
position, and one line of pixels carries three lines of text.

![Three lines of text vs RGB overlay](docs/img/rgb.png)

Each line only darkens its own channel, so nothing is painted over: each colour channel on its own still holds
one clean line. The overlay holds **exactly 3× the text** of a single monospace line (about 2.6× ordinary text).

**Result:** with raw pixels the overlay reads almost perfectly (0.01% CER at 677 chars per DeepSeek-OCR token), but
through DeepSeek-OCR's frozen encoder it is lost (83.5% CER). Inside Qwen2.5-VL's vision encoder the plain-font overlay
fades by the middle layers, while an overlay of learned glyphs is still partly recoverable at the encoder output and
loses much of the rest at the final merger step, which is why the next tests train that step. → [encoder_probe/RESULTS.md](encoder_probe/RESULTS.md)

---

## Headline results so far (best case per approach)

| Approach | Best case measured | vs plain text | Verdict |
|---|---|---|---|
| **Learned glyphs** (300 chunks, trained reader) | 3% CER at 3.75 px²/char (plain: 6.54) | **0.57× the pixels** (1.75× text per pixel) | ✅ **qualified go** |
| ↳ whole-pixel glyphs (survive encoders better) | 3% CER at 5.73 px²/char | 0.88× (0.77× at 1% CER) | ✅ use these |
| ↳ under heavy blur, noise and JPEG | 3.5% CER vs plain text 43% at the same size | lead grows | ✅ |
| ↳ through frozen VLM encoders | SigLIP layer 6: 53%; DeepSeek-OCR: 82% CER | – | ❌ frozen; ⏳ trained |
| **Colour blocks**, raw pixels | 1×1 blocks: 0.06% CER at 256 chars/token | **7.5×** chars per token | ✅ ceiling only |
| ↳ through JPEG (q75) | 2×2 blocks: 3% CER at 71 chars/token | 3× | ✅ ceiling only |
| ↳ through frozen VLM encoders | SigLIP ≥62%; DeepSeek-OCR 71–73% CER | – | ❌ frozen |
| **RGB overlay**, raw pixels | 0.01% CER at 677 chars per DeepSeek-OCR token | 3× mono line | ✅ ceiling only |
| ↳ through a frozen VLM encoder | DeepSeek-OCR: 83.5% CER | – | ❌ frozen; ⏳ trained |
| Hand-picked Chinese glyphs (trained reader) | best: 2.26% CER at 8.9 px²/char | **≥1.18× the pixels (worse)** | ❌ killed |
| Hangul-style 2×2 letter packing | ~2% CER at 26 px²/char | ≥3× worse | ❌ killed |
| *Reference:* plain text through DeepSeek-OCR's frozen encoder | 15–19% CER at 8.5 chars/token; denser Latin unreadable | ~2× text tokens | baseline |
| *Reference:* Claude reading plain English | 3% CER at 29.4 px²/char | ≈37 vision tokens per 1k chars vs ≈250 text tokens (6.7× fewer, at 3% error) | baseline |

Notes:
* "Trained reader" means a small CRNN trained from scratch on each rendering until it converged (10.5k steps),
  so every scheme is read equally well.
* Claude can only score natural text. Its safety classifiers refuse glyph-substituted or scrambled pages, so the
  dense schemes are scored with trained readers or the probe, never with Claude.
* Encoder numbers come from a small probe trained on frozen encoder outputs. They are upper bounds on what the
  encoder destroys, not what an LLM would read: the probe is data-limited (plain text through DeepSeek-OCR's encoder
  went from 85% to 15% CER as training pages grew from 200 to 5000), and DeepSeek-OCR's own decoder does far
  better on plain text. Only same-probe comparisons between codes carry the conclusions.
* Full write-ups: [docs/RESULTS.md](docs/RESULTS.md) (all phases), [PLAN.md](PLAN.md) (pre-registered predictions
  and kill criteria).

## Still to test

The frozen-encoder tests are done. Round 3 asks whether a VLM can be **trained** to read dense codes, and whether
that actually fits more document into its memory. → [encoder_probe/ROUND3.md](encoder_probe/ROUND3.md)

1. **G2-lite: train the merger (CPU, built, not run yet).** Keep Qwen2.5-VL's vision encoder frozen but train its
   final merger step, for plain text (~28 chars/token), whole-pixel learned glyphs (~107) and the moonshot below
   (322). This tests whether the merger is the wall the earlier probe suggested.
2. **Teach Qwen2.5-VL-3B to read (GPU, local RTX 3090).** LoRA fine-tuning, then deeper training if needed, on a
   density sweep of codes. Plain text is pushed to ~100 chars/token as the baseline. A code counts only if it
   reaches ≥1.5× the densest plain text at under 3% CER.
3. **Long-context benchmark: the real goal.** Ask questions about documents held as dense images versus as text,
   at equal memory (KV cache). The model must answer from the image tokens directly; transcribing back to text
   would refill the memory. Compare against a text-side compressor too.
4. **The moonshot: RGB overlay of learned glyphs** (`rgb3_lg`, 322–405 chars per token). Published results put the
   plausible range at ~30–90 chars per token, so this is a long shot.
   → [literature report](docs/research/2026-11-04_rgb3_lg_go_no_go_report.md)
5. **Smaller follow-ups:** a braille-style fixed dot code as a control for the learned glyphs, and hardening the
   glyphs against real JPEG and resizing. → [docs/IDEAS.md](docs/IDEAS.md)

Ruled out by the frozen-encoder results: the `rgb6` six-layer overlay and the standalone RGB overlay grid on SigLIP.

---

## Repo map

| Path | What |
|---|---|
| `vbpe/` | layout engine, rendering schemes (Latin, Chinese substitution, quadrant), CRNN reader, Claude eval, analysis |
| `experiments/` | Phase 1 (Claude), 1b (PP-OCR zero-shot) and 2 (trained CRNN) drivers |
| `learned_glyphs/` | learned glyph alphabet sub-project (plan, code, results, saved glyph banks) |
| `encoder_probe/` | encoder probe (rounds 1-2: frozen SigLIP and DeepEncoder) and round 3 (training Qwen2.5-VL to read codes) |
| `docs/` | results, context budget, literature report, GPU runbook, idea backlog, README figures |
| `scripts/` | data building, fonts, job queue, `make_readme_figures.py` |

**Setup and how to run everything:** [AGENTS.md](AGENTS.md). Quick start:

```bash
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch
scripts/fetch_fonts.sh
python3 scripts/build_data.py            # WikiText-103 passages and chunk dictionaries (~5 min)
python3 -m pytest -q tests                # ~100 tests, <15 s
python3 scripts/make_readme_figures.py    # regenerate the three figures above
```
