# Learned glyph alphabet (end-to-end) — plan

Hand-designed Visual BPE (CJK proxy glyphs, quadrant packing) looks weak (see ../PLAN.md, ../AGENTS.md,
../docs/prior-art.md). The remaining credible version of the idea: **learn the glyphs themselves**,
jointly with a reader, so each symbol (letter or frequent chain) gets a shape optimised to survive
low-resolution rendering. This sub-project tests whether that beats ordinary typography per pixel.

## Framing: a communication channel

```
text --segment--> symbol ids --GlyphBank--> ink cells (4x supersampled) --compose--> line canvas
     --Channel (random sub-pixel phase, area-downsample, blur, contrast, noise, 8-bit)--> line image
     --CRNN reader (vbpe.crnn.model.CRNN, CTC)--> symbol ids --decode--> text      (CER vs source)
```
Budget metric is the same as the parent project: **px²/char = line pitch (px) × line width (px) / source chars**.
Primary result: P@3% (px²/char needed for CER ≤ 3%), compared with Phase-2 Latin (≈ 10–15 px²/char).

## Main risk: channel exploitation
With a too-clean channel a learned bank degenerates into a barcode (bits in pixels) and "wins" for
reasons irrelevant to real rendering / VLM encoders. Guards (all mandatory):
1. **Calibration (E0):** the *fixed* DejaVu bank pushed through our channel must reproduce the Phase-2
   Latin CER curve (PIL rendering, results/phase2/phase2_main) within ~15% in P@3%. If our channel is
   easier than real rendering, harden it until it matches.
2. **Harsher channel check:** every headline result is re-run with the `harsh` preset; gains that vanish are exploitation.
3. **Transfer (E2):** freeze a learned bank, train a FRESH reader from scratch (same budget as Phase 2).
   Measures intrinsic legibility, not reader/glyph co-adaptation.
4. **Bridge (E3):** export a learned bank as a vbpe `Scheme` (bitmap font) and run it through the unchanged
   Phase-2 pipeline (`experiments/phase2_train.py`), so it is scored exactly like latin/subst.

## Experiments
| ID | What | Purpose |
|---|---|---|
| E0 | fixed DejaVu bank through channel presets, heights 4–8 px | calibrate channel vs Phase 2 |
| E1 | 2×2: {fixed font, learned} × {chars only (N=0), chars + N=300 tokens}, pitch h ∈ {4,5,6,8} px | separates "better font" from "more symbols" (the Visual-BPE effect) |
| E1b | fixed CJK subst300 bank (same symbols as ../vbpe/schemes/substitution.py) | cross-check vs Phase-2 subst300 |
| E2 | transfer: fresh reader on frozen learned banks | intrinsic legibility |
| E3 | bridge best banks into Phase-2 pipeline | apples-to-apples with Phase 2 |
| E4 | if promising: N=1000, width policies, harsh channel, more steps | robustness / scaling |

Width policy (default, configurable): char cell width = round(0.45·h) px, token cell width = round(0.9·h) px
(h = pitch). Fixed-font banks use each glyph's real advance (proportional), like Phase 2.

## Decision criteria
* **Font effect** S1 = P@3%(learned, N=0) / P@3%(fixed Latin, N=0).
* **Visual-BPE effect** S2 = P@3%(learned, N=300) / P@3%(learned, N=0).
* **GO** for GPU/VLM work only if learned+tokens needs ≥30% fewer px²/char than best fixed Latin
  **in transfer (E2) and in the bridge (E3)**, and the gain survives the harsh channel.
* If S1 is large but S2 ≈ 1, the finding is "optimised low-res fonts", not Visual BPE.

## Compute
CPU only (4 cores, shared with the parent project's Phase-2 grid until it finishes). Every run: 1 torch
thread. Target ≤ 40 min per run. Smoke tests during development: ≤ 5 CPU-min per agent.


## Amendments from prior art (learned_glyphs/docs/prior-art.md)
* **Channel realism is about geometry more than noise** (StegaStamp: no spatial distortions → chance on real
  captures). `harsh` preset: random sub-pixel shift, random non-integer scale 0.8–1.25 with random resize filter
  (box/bilinear/bicubic/lanczos), blur σ 0.3–1.2 px, gamma/contrast/offset, noise σ ≤ 0.1, differentiable JPEG
  q 40–90 (or a straight-through approximation). Plus a **test-only** held-out degradation set never used in training
  (real PIL JPEG, unseen resize filter) for every headline number.
* **Rule out pixel-alignment / grey-level codes by construction**: hard-binarised banks only; minimum feature size
  ≥ 1 output pixel (penalise sub-pixel detail, e.g. smoothness/erosion penalty at ss res); zero-ink guard band at
  cell edges; ink coverage within ~10–45%.
* **Report bits/px²** (empirical entropy of the symbol stream × (1 − symbol error)/area) next to px²/char.
  Sanity ceiling: per-char gains > ~3× over Phase-2 Latin are treated as channel exploitation until proven otherwise.
* **Extra baselines**: frozen random binary codebook; hand-made 3×2 block code (smartfonts); Tom Thumb-style 4×6
  pixel font.
* **Co-adaptation countermeasures**: channel strength ramp from clean; periodic reader re-initialisation or 2–3
  readers; oversample rare tokens; monitor glyph-pair confusions/duplicates. Transfer (E2) remains the real test.
* Cheap extra arm (E5): optimise glyphs against a FROZEN pretrained reader (e.g. the Phase-2 latin CRNN, or PP-OCR
  recogniser) — no co-adaptation possible.
* Naming: "Visual BPE" collides with ICLR-2025 BPE-on-visual-tokens work; use "learned glyph alphabet" in write-ups.

## Interfaces (contract between work packages)

```python
# lg/alphabet.py  (WP-LG2)
class Alphabet:
    symbols: list[str]          # 95 printable ASCII chars (32..126) first, then N dictionary tokens (priority order)
    n_chars: int                # = 95
    @classmethod
    def build(cls, n_tokens: int) -> "Alphabet"      # tokens = vbpe.dictionary.Dictionary.load(3000).prefix(n_tokens)
    def encode(self, text: str) -> list[int]          # Dictionary.segment (min-width DP) -> symbol ids; N=0 -> per char
    def decode(self, ids: list[int]) -> str
    def is_token(self, i: int) -> bool
# Reader classes: CTC blank = 0, symbol i -> class i+1 (n_classes = len(symbols)+1).

# lg/glyphs.py  (WP-LG1)
class GlyphBank(torch.nn.Module):
    ss: int; height_ss: int; widths_ss: torch.LongTensor  # per-symbol cell width at supersampled res
    def __init__(self, n_symbols, height_px, widths_px, ss=4, init="random", learnable=True)
    def forward(self, tau: float = 1.0, hard: bool = False) -> torch.Tensor
        # (n_symbols, height_ss, max_width_ss) ink in [0,1], zero outside each cell's width;
        # sigmoid(logits/tau); hard=True -> straight-through binarisation
    @classmethod
    def from_font(cls, symbols, font_path, height_px, ss=4, cjk_map: dict | None = None) -> "GlyphBank"
        # fixed bank: PIL renders each symbol at ss res, cell = its advance; tokens via cjk_map (token->ideograph) or rendered as text
    def save(self, path) / @classmethod load(path); def atlas_png(self, path, symbols)

# lg/compose.py  (WP-LG1)
def compose(ink, ids_batch: list[list[int]], widths_ss) -> tuple[torch.Tensor, torch.Tensor]
    # canvas_ss (B,1,height_ss,Wmax_ss) ink in [0,1], line widths_ss (B,)

# lg/channel.py  (WP-LG1)
class Channel:
    def __init__(self, preset: str = "default", ss: int = 4)   # presets: clean | default | harsh
    def __call__(self, canvas_ss, widths_ss, generator=None, train=True) -> tuple[torch.Tensor, torch.Tensor]
        # -> images (B,1,H,W) float32, ink=1 bg=0 (same as vbpe.crnn.data.collate), widths_px (B,)
        # differentiable w.r.t. canvas; eval (train=False) uses fixed mid-range degradation, seeded

# lg/train.py  (WP-LG2)
@dataclass class JointConfig: n_tokens, height_px, char_w, token_w, learn_glyphs, init, channel,
    steps, batch, lr_reader, lr_glyph, tau schedule, hard_from, seed, threads, run_name, ...
def train_joint(cfg) -> list[dict]     # rows (schema below), saves bank.pt + atlas.png + curves
# lg/transfer.py  (WP-LG3)
def train_reader_on_bank(bank_path, cfg) -> list[dict]     # fresh CRNN, frozen bank
# lg/scheme.py  (WP-LG3)
class LearnedGlyphScheme(vbpe.schemes.base.Scheme)          # bitmap-font Scheme for the Phase-2 pipeline
```

Rows (`learned_glyphs/results/<run>/runs.jsonl`) use the parent schema so `python3 -m vbpe.analysis` works:
`phase="lg", scheme (e.g. lg_learned_n300 / lg_fixed_n0), em (= h/1.2 for comparability), height_px,
px_per_char, cer, symbol_cer, steps, train_mode ("joint"|"transfer"|"bridge"), channel, n_tokens, learn_glyphs, train_time_s`.

## Work packages
| WP | Owns | Notes |
|---|---|---|
| LG1 renderer + channel + calibration | `lg/glyphs.py`, `lg/compose.py`, `lg/channel.py`, `experiments/calibrate_channel.py`, `tests/test_render.py` | E0 tooling |
| LG2 alphabet + joint trainer | `lg/alphabet.py`, `lg/train.py`, `experiments/train_joint.py`, `tests/test_train.py` | E1 |
| LG3 transfer + bridge + report | `lg/transfer.py`, `lg/scheme.py`, `lg/report.py`, `experiments/transfer_eval.py`, `tests/test_transfer.py` | E2, E3, atlases/confusions |
| LG4 prior art (learned/optimised fonts for OCR) | `docs/prior-art.md` | novelty + known pitfalls |
