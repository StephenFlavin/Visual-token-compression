# Visual BPE: validation plan for Phases 1-2 (hand-picked glyphs against Latin)

**Status: finished, 2026-11-03. Outcome: KILL (criterion K3).** Results and charts:
[RESULTS.md](RESULTS.md). This page is the plan as it was pre-registered before any run (the prior-art search it cites is dated 2026-11-02,
K3's outcome 2026-11-03); only this
header, the outcome table and file paths (which follow the 2026-11-12 restructure) were added later.

## What was asked, and what was found

**Asked:** would "Visual BPE" pack more text into the same pixels than plain Latin text? Visual BPE here means
drawing each frequent character run (` the `, `ing `) as one glyph, either a Chinese character standing in for it
("substitution") or several letters packed into one square cell ("quadrant"). The plan was to get a go / no-go on
a CPU before paying for a GPU.

**Found:** no. The plan set three kill criteria; the decisive one killed the idea in this hand-picked form.

| Criterion | Bar set here | Outcome |
|---|---|---|
| K1, arithmetic | substitution with 1000 chunks makes lines at least 15% narrower than condensed Latin | 32% narrower: passes, but that allows only a 1.19x larger font on a fixed page |
| K2, zero-shot reader | at least 20% fewer px² per character than the best Latin at 10% character error | 0.89x with PP-OCR (bar 0.80x): fails. Claude could not be used: it refuses coded and scrambled pages |
| K3, trained reader (decisive) | at least 30% fewer px² per character than the best Latin at 3% character error | substitution needs 1.25x to 1.48x *more*: **KILL** |
| Quadrant packing | killed if it needs at least as many pixels as Latin | 3.65x to 3.93x Latin: killed (prediction P1 confirmed) |

Section 4 said what a failure would point to: "next step would be *learned* glyphs (end-to-end optimised against
the reader), not hand-designed ones". That became [learned_glyphs/PLAN.md](../../learned_glyphs/PLAN.md), which
did beat Latin in pixel space (0.57x), and then failed at every pretrained vision model
([RESULTS.md, "What happened next"](RESULTS.md#what-happened-next)).

Departures from the plan as run:

* Phase 1's zero-shot comparison of schemes was done with PP-OCR ("Phase 1b"), because Claude's safety
  classifiers refuse substitution, quadrant and word-scrambled pages. Claude was measured on natural Latin only.
* Phase 2 used font sizes from 3 to 12 px (the plan said 5 to 12): Latin was still readable below 5 px.
* Phase 2's critical points were re-run at 10,500 steps (the plan's budget gave 3,500), because the substitution
  readers had not converged.
* Phase 3 (a GPU fine-tune of these hand-picked schemes) was not run: K3 killed it.

---

## The plan as pre-registered

Goal: reach a **go / no-go** on "Visual BPE" (render frequent multi-char chains as single
glyphs / packed cells so text survives lower image resolution) *before* spending GPU money.

Runner: 4 CPU, 15 GB RAM, no GPU. Headless `claude -p` accepts base64 images directly
(no tools) → we can benchmark a frontier VLM. CPU PyTorch is installed for a small trained reader.

---

## 1. Review of the original concept doc (what changed and why)

| # | Issue in original plan | Consequence | Fix in this plan |
|---|---|---|---|
| R1 | **Quadrant packing is a rearrangement, not compression.** A Latin letter costs ≈0.5 em × 1.2 em pitch ≈ 0.6 em²; packed 2×2 in a 1 em cell at half size each letter gets (0.5 em)² of a *smaller* em — equal letter size costs ≈ the same or more area. | Predicted to lose vs plain text. | Tested anyway (cheap), with a pre-registered prediction of failure. |
| R2 | **CJK glyphs are 1.0 em wide; Latin letters 0.37–0.51 em.** "the␣" in a narrow font is ~1.5 em → one ideograph saves ~33 %, not 75 %. 2-char tokens save ~nothing. | The headline "35–45 % fewer characters" overstates area saving. | Dictionary is chosen by **width saved**, not char count; we report the analytic area ratio first (kill check K1). |
| R3 | **Bigger alphabet ⇒ each glyph needs more pixels to be discriminable** (~370 classes vs ~80). | Gains could cancel. | That's exactly what the CER-vs-pixels curves measure. |
| R4 | **Weak baseline.** "Small default font" is not the best plain-text rendering. Narrow fonts (Archivo Narrow 0.37 em/char vs DejaVu 0.51) already save ~27 % width. | Any win must be vs the *best* Latin. | Baselines: `latin` (DejaVu Sans) and `latin_condensed` (Archivo Narrow). Substitution also uses the condensed font for left-over Latin. |
| R5 | **Single data point** (1 500 words in 512²) ≈ 29 px²/char — probably illegible for *both*. | Uninformative. | Sweep pixel budget → CER curves; compare **px²/char needed to reach CER ≤ 3 % / 10 %**. |
| R6 | **Zero-shot confound:** "read symbols + apply 300-entry dictionary from prompt" mixes OCR failure with lookup failure. | Can't attribute errors. | Primary: VLM transcribes raw symbols; we decode programmatically. Legend mode is secondary. |
| R7 | **Language-prior / memorisation asymmetry:** Claude can "read" blurry English from priors (and may have memorised Wikipedia); arbitrary symbols get no prior. | Biased *against* Visual BPE zero-shot. | Sentence-shuffled passages (breaks memorised continuations) + a word-scrambled control (removes most priors). Phase 2 trains readers from scratch, so priors are equal. |
| R8 | **Stage 3 needs a GPU.** | Not runnable here. | Replaced by Phase 2: small CRNN trained from scratch per scheme on CPU — measures raw "rendering channel" density. GPU LoRA only if Phases 1–2 pass. |
| R9 | Novelty unverified. | Might be known/obsoleted. | Prior-art review (WP-G). |

## 2. Core metric

**px²/char** = canvas area / normalised source characters. For Claude, vision tokens = ceil(W/28)·ceil(H/28)+3 (measured; ≈ W·H/784),
so px²/char is directly proportional to vision tokens per source char.
For each scheme we fit the CER-vs-px²/char curve and read off **P@3 %** and **P@10 %**
(px²/char needed for CER ≤ 3 % / 10 %). Headline ratio: `P(subst) / min(P(latin), P(latin_condensed))`.
"Best Latin" = min over all Latin baselines. Note: Glyph's genetic rendering search chose a *wide* font (Verdana 9px, no condensing) — `latin` (DejaVu Sans, a Verdana-like humanist sans) covers that end; condensed may lose for Claude despite winning on width.
Also report each scheme's natural − scrambled CER gap as "prior credit" (DeepSeek-OCR loses 35–70 pts on random-letter text), since symbol streams get no language prior zero-shot; this is why K3 (Phase 2), not K2, is decisive.

CER = Levenshtein(normalise(ref), normalise(decode(output))) / len(ref); normalise collapses whitespace.

## 3. Phases

### Phase 0 — harness (foundation done; WPs below)
* `vbpe/layout.py` shared layout engine (units → wrapped lines → largest em that fits W×H).
* `vbpe/schemes/*` — one class per rendering scheme, common interface (`base.py`).
* Corpus/dictionary, Claude client, sweep runner, CRNN trainer.
* **K1 (analytic, free):** report width ratio `width(subst_N encoding) / width(condensed Latin)` at equal em on held-out text for N ∈ {100, 300, 1000, 3000}.

### Phase 1 — zero-shot Claude sweep
* Passages: ~3 000-char, sentence-shuffled from WikiText-103 **test** split (6 natural + 3 word-scrambled).
* Schemes (main grid): `latin`, `latin_condensed`, `subst300`, `subst1000`, `quad3` (+ `quad4`, `subst300_legend` small side runs).
* Budgets: px²/char ∈ ~{9, 13, 18, 27, 40, 60, 90}, square canvases (final grid chosen by a pilot). Prior art (docs/background/prior-art.md): published legible regimes are ~13–100 px²/char (Glyph's tuned setting ≈45–55; LensVLM 15× ≈13 and mostly illegible), so <9 is likely uninformative.
* Pilot first (1 passage, all schemes, wide grid) → report cost/call, then main sweep.
* Est. ~360 calls. Cost per call measured in pilot (probe: $0.013 for a tiny image; expect ~$0.03–0.10 with ~800 output tokens).

### Phase 2 — CRNN reader trained from scratch (CPU)
* One small CNN+BiLSTM+CTC model per (scheme, em) on single rendered lines from WikiText-103 train.
* Same architecture, steps, data for every scheme; held-out CER; px²/char = pitch × line width / chars.
* em grid ≈ {5, 6, 7, 8, 10, 12} px; total budget ≈ 4–6 CPU-hours.

### Phase 3 — GPU LoRA (only if Phases 1–2 pass) — scripts written later, run by the user on RunPod.

## 4. Pre-registered predictions & kill criteria

Predictions: (P1) quadrant needs ≥ 30 % *more* px²/char than best Latin. (P2) subst300 within ±20 % of
`latin_condensed`. (P3) Latin's advantage shrinks on word-scrambled text.

* **K1** – if subst1000 saves < 15 % line width vs condensed Latin at equal em → strong prior of failure.
* **K2 (Phase 1)** – subst never achieves ≥ 20 % lower P@10 % than best Latin (natural *or* scrambled) → zero-shot negative.
* **K3 (Phase 2, decisive)** – subst P@3 % not ≥ 30 % below best Latin (the concept doc's own bar) → **kill**. **Outcome (2026-11-03): KILLED** — converged plain DejaVu Latin beats subst300/subst1000 at every budget (docs/glyphs/RESULTS.md).
* Quadrant killed if its Phase-2 P@3 % ≥ best Latin.
* Context for any positive result: text-side compression (mean pooling, learned compressors, C3 at 98% @20×) already matches or beats optical compression, so RESULTS.md must state the win relative to that ceiling.
* If subst passes but quadrant fails → idea reduces to "bigger alphabet"; next step would be *learned* glyphs (end-to-end optimised against the reader), not hand-designed ones.

## 5. Interfaces (contract for all work packages)

```python
# vbpe/layout.py  (owner: lead — do not edit; report needed changes)
class Unit: source: str; labels: tuple[str,...]; is_space: bool; advance(em)->px; draw(d, x, baseline, em)
GlyphUnit(char, font_path, source=None, labels=None)
LayoutParams(margin=4, line_spacing=1.2, supersample=1)
render_page(units, W, H, params=None, em=None) -> (PIL.Image 'L', {"em","n_lines","fit"})
render_line(units, em, params=None) -> PIL.Image          # Phase 2
fit_em(units, W, H, params=None) -> float
# Ink must stay within [baseline-0.85em, baseline+0.25em]. Breaks only after units whose source ends in whitespace.

# vbpe/schemes/base.py
class Scheme: name; encode(text)->list[Unit]; decode(transcription)->str;
    transcription_target(text)->str   # ''.join(all unit labels)
    vlm_system_prompt()->str; vlm_prompt()->str; vlm_key_images()->list[PIL.Image]
get_scheme(name) in vbpe/schemes/__init__.py

# vbpe/corpus.py (WP-A)
load_passages(variant: "natural"|"scrambled") -> list[{"id": str, "text": str}]
iter_lines(split: "train"|"val", min_chars=30, max_chars=70, seed=0) -> Iterator[str]   # Phase 2
# vbpe/dictionary.py (WP-A)
class Dictionary: tokens: list[str]  (ordered by priority); segment(text)->list[str]  # each = token or 1 char
    @classmethod load(n: int) -> Dictionary      # reads data/dict_{n}.json, builds on demand

# vbpe/claude_eval.py (WP-E1)
transcribe(images: list[PIL.Image], system: str, prompt: str, model: str = DEFAULT_MODEL) -> dict
   # keys: text, model, input_tokens, output_tokens, cost_usd, latency_s, cached(bool), error(str|None)
transcribe_many(jobs: list[dict], workers=8) -> list[dict]   # job = kwargs of transcribe (+ any passthrough "meta")

# vbpe/metrics.py: normalize, cer, px_per_char, vision_tokens, text_tokens_est
```

All text is ASCII after corpus cleaning. Data in `data/` (git-ignored), outputs in `results/`.

## 6. Work packages (parallel)

| WP | Owner files | Depends on |
|---|---|---|
| A corpus + dictionary + K1 report | `vbpe/corpus.py`, `vbpe/dictionary.py`, `scripts/build_data.py`, `tests/test_corpus_dict.py`, `docs/glyphs/dictionary-report.md` | — |
| C substitution scheme | `vbpe/schemes/substitution.py`, `assets/cjk_symbols.txt`, `tests/test_substitution.py` | Dictionary interface (stub until A lands) |
| D quadrant scheme | `vbpe/schemes/quadrant.py`, `tests/test_quadrant.py` | layout |
| E1 Claude client + vision checks | `vbpe/claude_eval.py`, `scripts/check_claude_vision.py`, `tests/test_claude_eval.py`, `docs/claude/claude-eval-notes.md` | — |
| E2 Phase-1 runner + analysis | `experiments/phases/phase1_sweep.py`, `vbpe/analysis.py`, `tests/test_analysis.py` | E1 interface (fake transcriber for dry-run) |
| F Phase-2 CRNN | `vbpe/crnn/*`, `experiments/phases/phase2_train.py`, `tests/test_crnn.py` | layout, schemes, corpus interface |
| G prior art | `docs/background/prior-art.md` | — |

Integration (lead): merge, full tests, pilot → main Phase-1 sweep → Phase-2 grid → `docs/glyphs/RESULTS.md`.
