# RGB overlay through the frozen-encoder probe

Tests the user's RGB word-overlay idea (docs/IDEAS.md): put three text streams in one image, one per colour
channel, for up to 3x the text per pixel. It uses the unchanged probe and `run_probe.py` from PLAN.md.
The code is in `ep/rgb_overlay.py` (registered through one lazy branch in `pages.get_encoding`) and the tests are in
`tests/test_rgb_overlay.py`.

## Design
All pages are 224x224 RGB with line pitch p (line i = rows [i*p, (i+1)*p)). Text is rendered like `latin_p`:
em = p/1.2, 4x supersampled, then BOX-downsampled.

| encoding | what is on a line | CTC labels per line | classes |
|---|---|---|---|
| `mono_p<p>` | DejaVu Sans Mono, black on white, n_cells fixed-width cells, one fixed-length chunk of n_cells chars (no word wrap, like `col_b`), grey copied into R, G and B | the chunk | 95 printable ASCII |
| `rgb3_p<p>` (main) | a chunk of 3·n_cells chars: R draws chars [0,n), G draws [n,2n), B draws [2n,3n), all in the same mono cell grid. Each channel is composited on its own: stream c darkens only channel c (value = 255 − ink), so overlapping glyphs never hide each other. Channel c is exactly a `mono` page of stream c (this is tested) | interleaved per cell: R[k], G[k], B[k] for k = 0..n−1, left to right | 3×95 channel-tagged (`"R"+ch`, `"G"+ch`, `"B"+ch`) |
| `rgb3a_p<p>` | same pages as rgb3 | same order, untagged | 95 printable ASCII |
| `rgbw_p<p>_chan` / `_paint` | the user's literal layout. Words go round-robin to R, G, B, and each slot holds 3 consecutive words at one x position. The word in channel c is shifted right by c·s px (s = 1 by default; the `_s<k>` suffix sets it). Slot width = max_c(width_c + c·s) + one space. DejaVu Sans, slots wrap. `_chan` composites per channel as rgb3 does. `_paint` draws solid (255,0,0)/(0,255,0)/(0,0,255) ink on white, with B painted over G over R (at 4x), so overlaps hide earlier words | the line's words in source order, joined by spaces | 95 printable ASCII |

* **Scoring.** The reference is always source text: for rgb3 it is the raw chunk in source order (R+G+B), so
  `decode()` maps the interleaved labels back to source order. mono and rgb3 refs are raw fixed-length chunks,
  so `clean_hyp` does not strip them (as for `col_b`). rgbw refs are normalised word strings.
* **Why tagged classes for rgb3.** `rgb3a` has to de-interleave by position, so a single dropped or inserted label
  moves every later char into the wrong stream, and CER would then measure alignment luck, not what the tokens
  keep. With tagged labels, decode sends each label to its stream by tag, and one indel costs one char (tested). The
  emitted order is the same, and the tags carry no information that is not already in the pixels (which channel
  is dark). `rgb3a` is kept as a check and is not in the grid.
* **CTC frame budget.** The probe emits T = 224·3 = 672 frames per line. Using the strict bound 2L+1 ≤ T, rgb3
  supports p ≥ 4 (p4: L = 333, 2L+1 = 667) and fails at p3 (L = 444, 889). The constructor enforces this and the
  tests assert it. At p5 a cell is 2.5 px, which gives 7.5 frames for its 3 labels.

### Pixel math (4 test pages, measured with `make_pages`; chars/token = 256 / px²/char)

| p | em px | cell px | cells/line | lines | mono px²/char | mono chars/tok | rgb3 L/line | 2L+1 (≤ 672) | rgb3 px²/char | rgb3 chars/tok | rgbw px²/char | rgbw chars/tok | rgbw max L (2L+1) | latin_p chars/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 4.17 | 2.51 | 89 | 44 | 12.81 | 20.0 | 267 | 535 | 4.27 | 59.9 | 5.72 | 44.7 | 230 (461) | 23.0 |
| 6 | 5.00 | 3.01 | 74 | 37 | 18.33 | 14.0 | 222 | 445 | 6.11 | 41.9 | 8.19 | 31.3 | 192 (385) | 16.0 |
| 8 | 6.67 | 4.01 | 55 | 28 | 32.58 | 7.9 | 165 | 331 | 10.86 | 23.6 | 14.56 | 17.6 | 147 (295) | 8.9 |
| 10 | 8.33 | 5.02 | 44 | 22 | 51.83 | 4.9 | 132 | 265 | 17.28 | 14.8 | 23.33 | 11.0 | 119 (239) | 5.5 |

* rgb3 holds exactly 3x mono. The mono font is 0.60 em per char against DejaVu Sans's 0.51, so mono has
  about 12% fewer chars/token than `latin_p` at the same pitch, and rgb3 is about 2.6x latin. rgbw (both modes,
  same layout) is about 1.95x latin, because slots are as wide as their longest word.
* **Caveat for rgb3_p5 (text supply).** `make_pages` reads about n_lines·(224+8)·1.05 chars per page, and rgb3_p5
  uses 267 per line. 182 of its 200 train pages and 14 of its 16 test pages are therefore unique; the rest repeat
  the first pages. `make_pages` is not changed while the live grid runs. Every other encoding/pitch here fits
  (rgb3_p6 uses 222 chars per line). Do not use p4 for rgb3 under the current `make_pages`: about 146 of 200 train
  pages and 9 of 16 test pages would be unique.

## Pre-registered predictions (written 2026-11-03, before any non-smoke run)
1. **identity/clean:** rgb3 reaches 2.5–3x the chars/token of mono at equal CER. The identity probe sees raw RGB
   and the channels are independent.
2. **identity/jpeg75:** rgb3 degrades much more than mono. 4:2:0 keeps full resolution only in luma
   (0.30R + 0.59G + 0.11B, a fixed mix of the three streams), and the separating information is in chroma at
   half resolution. The B stream should suffer most, but `run_probe.py` does not score per stream, so this can only
   be seen in the samples.
3. **siglip_l6/clean:** rgb3 is no better than mono in chars/token at equal CER. Colour-block rows through SigLIP
   reached only unigram-level CER.
4. (rgbw, lower priority) On identity/clean, rgbw_chan has a higher CER than rgb3 at similar chars/token. The 1-D
   reader has to emit 2–3 spatially overlapping words in sequence instead of one char per channel per cell.
   rgbw_paint has a higher CER than rgbw_chan at every pitch, because occlusion destroys ink that per-channel
   compositing keeps.

## Decision rule
RGB overlay is worth pursuing only if rgb3 beats the best latin/mono chars/token at 3% CER (C@3%, as in
PLAN.md) through **siglip_l6 with jpeg75**. Otherwise the idea is dead for VLMs with a pretrained encoder. Identity
rows only give the ceiling.

## Grid (not run; not queued)
`experiments/rgb_overlay_grid.txt` has 48 commands, one per line, in run_probe.py syntax with `--steps 3000
--run-name rgb_overlay`:
encodings `mono_p{5,6,8,10}`, `rgb3_p{5,6,8,10}`, `rgbw_p{6,8}_chan`, `rgbw_p{6,8}_paint` crossed with
(identity, clean), (identity, jpeg75), (siglip_l6, clean), (siglip_l6, jpeg75). To run them, copy the lines into a
**new** queue file and start `scripts/job_queue.py` on it with its own `--log-dir`, after the current probe grid
has finished. The latin rows that the rule compares against come from `results/main` and `results/controls`.

## Smoke run (sanity check only, not a result)
`results/rgb_smoke/`: rgb3_p6, identity/clean, 300 steps, 40 train pages, 4 test pages, 1 thread.
Test CER (4 pages) was 74.9% at step 100, 41.0% at step 200 and 32.2% at step 300, still falling (train CTC 0.63). The samples
decode to recognisable source-order text (`'...the Raual Cosrt Thmatre. he had a guest role in the ttenisian'`), so
rendering, interleaved labels, tagged decode and scoring work end to end. These rows are smoke-only and are not
used for any prediction or decision.
