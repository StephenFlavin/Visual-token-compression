"""Joint training of a glyph bank and a CRNN reader through the rendering channel (WP-LG2, E1/E5).

    text -> Alphabet.encode -> ids -> GlyphBank(tau, hard) -> compose -> Channel(train) -> CRNN -> CTC

Conventions mirrored from Phase 2 (``vbpe/crnn``):
  * Source text: test = first ``n_test`` lines of ``iter_lines('val', 30, 70, seed=data_seed)``,
    val (checkpoint selection) = the next ``n_val`` lines (``vbpe.crnn.data.source_lines``, so the
    lines and their order are identical to Phase 2). Train lines are streamed from
    ``iter_lines('train', 30, 70, seed=data_seed)`` and bucketed by rendered width.
  * Reader: ``vbpe.crnn.model.CRNN`` (same architecture), col_stride = ``col_stride_for_em(em)``
    (2 px), expand E = ``choose_expand`` over val + test + the first ``n_expand_probe`` train lines
    (T >= 2L+1), em = height_px / 1.2. CTC blank = 0, symbol i -> class i + 1.
  * Optimiser: AdamW + OneCycleLR (pct_start 0.15), grad clipping, eval every 250 steps on val,
    best checkpoint (bank + reader) by val CER, then one test row.

Glyph-specific:
  * separate param groups (``lr_reader``, ``lr_glyph`` on the bank logits, no weight decay on
    logits by default), separate grad clipping;
  * temperature tau annealed geometrically ``tau_start -> tau_end`` over the run;
  * straight-through hard binarisation from ``hard_from`` x steps onward;
  * evaluation (val and test) of learned banks ALWAYS uses hard (binarised) glyphs, and the channel
    in eval mode (seeded). Fixed font banks use LG1's stored PIL coverage at ss resolution (as
    Phase 2 renders), or a 0.5 threshold of it with ``fixed_hard``.
  * held-out test degradations: if the Channel supports a test-only mode (LG1:
    ``Channel(preset, test_only=True)``: PIL LANCZOS round trip + real JPEG q50 on top of the
    preset's eval degradation) the test set is also scored through it ->
    ``cer_heldout`` / ``symbol_cer_heldout`` (None if unsupported).

Anti-degeneracy (learned banks only; PLAN amendments):
  * ``guard_ss`` (default 1): hard zero-ink guard band of that many ss px at the cell edges. Passed
    to LG1's ``GlyphBank(guard_ss=...)`` (built into its mask, left/right AND top/bottom, saved with
    the bank); for banks without that kwarg a local left/right mask is used (logits there set to
    -30, so it is baked into the saved bank too);
  * ``min_feature_weight`` (default 1.0) x min-feature penalty (LG1 ``bank.min_feature_penalty``
    when present, else the local equivalent): mean |ink - opening(ink)| + |closing(ink) - ink|
    with an ss x ss (= 1 output px) square: penalises strokes/dots/gaps thinner than one output px;
  * ``coverage_weight`` (default 1.0) x ``coverage_penalty`` (local: LG1's squared hinge is ~100x
    weaker at these scales and does not skip the space): mean over non-space glyphs of the LINEAR
    hinge distance of each glyph's ink coverage from [coverage_lo, coverage_hi] = [0.10, 0.45];
  * ``ink_penalty`` / ``entropy_penalty`` (off): mean ink / mean binary entropy of soft glyphs;
  * all penalty weights ramp linearly 0 -> 1 over the first ``penalty_ramp`` (0.3) x steps, so CTC
    shapes the glyphs before the morphology terms act (with full weight from step 1 the
    min-feature term erased most random-init ink in the smoke: mean ink 0.05, many blank glyphs).

Co-adaptation countermeasures:
  * ``channel_ramp`` (default 0.3): channel strength ramps linearly from the ``clean`` preset to
    the target preset over the first 30% of steps (8 discrete levels, every range-valued
    ChannelParams field interpolated; JPEG quality from 100 down to the target range, off at
    level 0); 0 disables;
  * ``reader_resets`` K (default 0): re-initialise a reader (weights + optimiser state) K times,
    evenly spaced up to ``reset_until`` x steps; glyphs are kept; checkpoint selection restarts
    after each reset;
  * ``n_readers`` R (default 1): glyphs trained against R independently initialised readers
    (mean CTC loss); resets cycle through the readers; val/test use the reader with the best val CER;
  * ``oversample_alpha`` (default 0 = off): train lines are drawn from a 3x larger pool with
    probability ~ max over the line's tokens of clip((median token count / count) ** alpha, 1, 20)
    (counts from the probe train lines);
  * ``frozen_reader`` PATH (E5): optimise glyphs only against a FROZEN pretrained Phase-2 CRNN
    checkpoint (``results/phase2/<run>/checkpoints/latin_em*.pt``, character vocab -> N=0 only;
    its em should match: height_px = round(1.2 em)). train_mode = "frozen_reader".

Metrics (row keys beyond the PLAN schema):
  * px_per_char of a line = height_px (line pitch) x composed content width (px, margins excluded)
    / len(normalize(text)); the row reports the mean over test lines (as Phase 2);
  * ``bits_per_char`` = H1 x (symbols / chars), H1 = empirical unigram entropy (bits) of the symbol
    stream on the probe train lines; ``bits_per_px2`` = bits_per_char x (1 - cer) / px_per_char,
    ``bits_per_px2_sym`` = same with (1 - symbol_cer) (both clipped at >= 0);
  * glyph report (``glyph_report/<tag>.json``): pairwise Hamming distances of binarised glyphs at
    ss res (min, median nearest-neighbour, #pairs < ss^2 bits apart = less than one output pixel)
    and L1 at 1x (area-downsampled), 20 nearest pairs, 30 most frequent test substitutions
    (symbol-level edit ops); headline numbers copied into the row.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path

import numpy as np

from vbpe import config
from vbpe.metrics import normalize

from .alphabet import Alphabet

LG_ROOT = Path(__file__).resolve().parent.parent
RESULTS = LG_ROOT / "results"
FONTS = {"latin": config.LATIN_FONT, "mono": config.LATIN_MONO_FONT,
         "condensed": config.CONDENSED_FONT}
RAMP_LEVELS = 8
GUARD_LOGIT = -30.0

ROW_KEYS = ("phase", "run", "train_mode", "scheme", "em", "height_px", "px_per_char", "cer",
            "symbol_cer", "steps", "channel", "n_tokens", "learn_glyphs", "train_time_s")


@dataclass
class JointConfig:
    # alphabet / bank
    n_tokens: int = 0
    height_px: int = 7                    # line pitch (px) = round(1.2 * em)
    char_w: int | None = None             # learned banks: char cell width px (default round(0.45 h))
    token_w: int | None = None            # learned banks: token cell width px (default round(0.9 h))
    learn_glyphs: bool = True
    init: str = "random"                  # GlyphBank init (learned banks): random | font
    font: str = "latin"                   # fixed banks / font init: latin | condensed | mono | path
    token_glyphs: str = "cjk"             # fixed banks with tokens: cjk (subst ideographs) | text
    fixed_hard: bool = False              # fixed banks: threshold PIL coverage at 0.5 (default:
                                          # keep antialiased ss coverage = Phase-2 rendering)
    ss: int = 4
    channel: str = "default"              # Channel preset: clean | default | harsh
    pad_px: int = 0                       # extra zero padding (px) each side before the channel
    heldout_eval: bool = True             # also score test through the channel's held-out mode
    # optimisation
    steps: int = 3500
    batch: int = 32
    lr_reader: float = 2e-3
    lr_glyph: float = 0.05
    weight_decay: float = 1e-2
    glyph_weight_decay: float = 0.0
    pct_start: float = 0.15
    grad_clip: float = 5.0
    glyph_grad_clip: float = 5.0
    tau_start: float = 1.0
    tau_end: float = 0.2
    hard_from: float = 0.6                # fraction of steps after which glyphs are binarised (ST)
    # anti-degeneracy (learned banks)
    guard_ss: int = 1
    pixel_block: bool = False             # learned glyphs = whole output pixels (logits tied per ss x ss block)
    min_feature_weight: float = 1.0
    coverage_weight: float = 1.0
    coverage_lo: float = 0.10
    coverage_hi: float = 0.45
    ink_penalty: float = 0.0
    entropy_penalty: float = 0.0
    penalty_ramp: float = 0.3             # penalty weights ramp 0 -> 1 over this fraction of steps
    # co-adaptation countermeasures
    channel_ramp: float = 0.3
    reader_resets: int = 0
    reset_until: float = 0.5
    n_readers: int = 1
    oversample_alpha: float = 0.0
    frozen_reader: str | None = None
    # eval / data
    eval_every: int = 250
    eval_soft: bool = False               # also log val CER with soft glyphs (diagnostic)
    n_val: int = 300
    n_test: int = 1000
    n_expand_probe: int = 4000
    bucket_mult: int = 20
    n_samples: int = 20
    # misc
    seed: int = 0
    data_seed: int = 0
    threads: int = 1
    run_name: str = "joint"
    out_root: str | None = None           # default learned_glyphs/results
    scheme_suffix: str = ""
    model: dict = field(default_factory=dict)     # CRNN kwargs override

    # ---- derived ----
    @property
    def em(self) -> float:
        return self.height_px / 1.2

    @property
    def char_w_px(self) -> int:
        return int(self.char_w) if self.char_w else max(1, int(round(0.45 * self.height_px)))

    @property
    def token_w_px(self) -> int:
        return int(self.token_w) if self.token_w else max(1, int(round(0.9 * self.height_px)))

    @property
    def train_mode(self) -> str:
        return "frozen_reader" if self.frozen_reader else "joint"

    @property
    def scheme(self) -> str:
        s = f"lg_{'learned' if self.learn_glyphs else 'fixed'}_n{self.n_tokens}"
        if not self.learn_glyphs and self.n_tokens and self.token_glyphs != "cjk":
            s += f"_{self.token_glyphs}"
        if self.frozen_reader:
            s += "_frozenreader"
        if self.reader_resets and not self.frozen_reader:
            s += f"_rr{self.reader_resets}"
        if self.n_readers > 1 and not self.frozen_reader:
            s += f"_nr{self.n_readers}"
        return s + self.scheme_suffix

    @property
    def tag(self) -> str:
        t = f"{self.scheme}_h{self.height_px}"
        if self.channel != "default":
            t += f"_{self.channel}"
        if self.seed:
            t += f"_s{self.seed}"
        return t

    def out_dir(self) -> Path:
        return Path(self.out_root or RESULTS) / self.run_name

    @classmethod
    def from_dict(cls, d: dict) -> "JointConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# ============================================================================ LG1 modules
def lg1():
    """(GlyphBank, compose, Channel) from the sibling LG1 modules (imported lazily so tests can
    install stand-ins in ``sys.modules``)."""
    pkg = __package__ or "lg"
    g = importlib.import_module(f"{pkg}.glyphs")
    c = importlib.import_module(f"{pkg}.compose")
    ch = importlib.import_module(f"{pkg}.channel")
    return g.GlyphBank, c.compose, ch.Channel


def _cjk_map(cfg: JointConfig, alphabet: Alphabet):
    if alphabet.n_tokens and cfg.token_glyphs == "cjk":
        from vbpe.schemes.substitution import load_symbols
        return dict(zip(alphabet.tokens, load_symbols()[: alphabet.n_tokens]))
    return None


def build_bank(cfg: JointConfig, alphabet: Alphabet):
    GlyphBank, _, _ = lg1()
    font = FONTS.get(cfg.font, cfg.font)
    cjk_map = _cjk_map(cfg, alphabet)
    if cfg.learn_glyphs:
        widths = [cfg.char_w_px] * alphabet.n_chars + [cfg.token_w_px] * alphabet.n_tokens
        kw = {}
        if not getattr(GlyphBank, "STUB", False):     # LG1 extras: init="font" source, init seed
            kw = dict(symbols=alphabet.symbols, font_path=str(font), cjk_map=cjk_map,
                      seed=cfg.seed)
        if bank_has_guard(GlyphBank):
            kw["guard_ss"] = int(cfg.guard_ss)
        if cfg.pixel_block:
            kw["pixel_block"] = True
        return GlyphBank(len(alphabet), cfg.height_px, widths, ss=cfg.ss, init=cfg.init,
                         learnable=True, **kw)
    return GlyphBank.from_font(alphabet.symbols, str(font), cfg.height_px, ss=cfg.ss,
                               cjk_map=cjk_map)


def bank_has_guard(GlyphBank) -> bool:
    """LG1's GlyphBank builds the guard band into its mask (``guard_ss`` kwarg)."""
    try:
        return "guard_ss" in inspect.signature(GlyphBank.__init__).parameters
    except (TypeError, ValueError):
        return False


def glyph_params(bank) -> list:
    return [p for p in bank.parameters() if p.requires_grad]


def cell_mask(bank, widths_ss_t, shape):
    import torch
    m = getattr(bank, "mask", None)
    if m is not None and tuple(m.shape) == tuple(shape):
        return m.float()
    cols = torch.arange(shape[2])
    return (cols[None, None, :] < widths_ss_t[:, None, None]).float().expand(*shape)


def guard_mask(widths_ss_t, shape, g: int):
    """1 inside each cell except ``g`` ss columns at its left and right edges."""
    import torch
    cols = torch.arange(shape[2])[None, None, :]
    w = widths_ss_t[:, None, None]
    return ((cols >= g) & (cols < w - g)).float().expand(*shape)


# ============================================================================ penalties
def _morph(x, k: int, op: str, first: bool):
    """k x k min/max filter, stride 1, same size, replicate padding. ``first`` selects the
    padding split so that an erosion/dilation pair (first, second) cancels its offset."""
    import torch.nn.functional as F
    a, b = k // 2, k - 1 - k // 2
    pad = (a, b, a, b) if first else (b, a, b, a)
    if op == "max":
        return F.max_pool2d(F.pad(x, pad, mode="replicate"), k, 1)
    return -F.max_pool2d(F.pad(-x, pad, mode="replicate"), k, 1)


def min_feature_penalty(ink, mask, k: int):
    """Mean |ink - opening(ink)| + |closing(ink) - ink| over ``mask`` pixels (k x k square):
    strokes and gaps thinner than k ss pixels (= 1 output px for k = ss) are penalised."""
    x = ink[:, None]
    opened = _morph(_morph(x, k, "min", True), k, "max", False)
    closed = _morph(_morph(x, k, "max", True), k, "min", False)
    m = mask[:, None]
    d = ((x - opened).abs() + (closed - x).abs()) * m
    return d.sum() / m.sum().clamp_min(1.0)


def coverage_penalty(ink, mask, lo: float, hi: float, skip=(0,)):
    """Mean hinge distance of per-glyph ink coverage from [lo, hi] (``skip``: e.g. the space)."""
    import torch
    cov = ink.sum((1, 2)) / mask.sum((1, 2)).clamp_min(1.0)
    keep = torch.ones_like(cov, dtype=torch.bool)
    keep[list(skip)] = False
    pen = (lo - cov).clamp_min(0) + (cov - hi).clamp_min(0)
    return pen[keep].mean() if keep.any() else pen.sum() * 0


def feature_penalty(bank, ink, mask, ss: int):
    """LG1's ``GlyphBank.min_feature_penalty`` if present (same definition, background padding),
    else the local one."""
    h = getattr(bank, "min_feature_penalty", None)
    if callable(h):
        return h(ink)
    return min_feature_penalty(ink, mask, ss)


# ============================================================================ schedules
def tau_at(cfg: JointConfig, step: int) -> float:
    frac = (step - 1) / max(1, cfg.steps - 1)
    return float(cfg.tau_start * (cfg.tau_end / cfg.tau_start) ** frac)


def hard_at(cfg: JointConfig, step: int) -> bool:
    return (step - 1) >= cfg.hard_from * cfg.steps


def reset_steps(cfg: JointConfig) -> list[int]:
    K = 0 if cfg.frozen_reader else int(cfg.reader_resets)
    return [max(1, int(round(cfg.steps * cfg.reset_until * (k + 1) / K))) for k in range(K)]


# ============================================================================ channel helpers
def _interp(a, b, s: float, name: str = ""):
    if name == "jpeg" and tuple(a) == (0.0, 0.0) and tuple(b) != (0.0, 0.0):
        # quality ramps from ~lossless (100) down to the target range; 0 = off at level 0
        return (0.0, 0.0) if s == 0 else tuple(float(100 + (y - 100) * s) for y in b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return type(b)(a + (b - a) * s) if isinstance(b, float) else (a if s < 0.5 else b)
    if (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in a + b)):
        return tuple(float(x + (y - x) * s) for x, y in zip(a, b))
    return a if s < 0.5 else b


def make_channel_ramp(cfg: JointConfig, Channel, log=print) -> list:
    """[level 0 (clean) ... level L-1 (= target preset)]; [target] if ramping is off/unsupported."""
    target = Channel(cfg.channel, ss=cfg.ss)
    if cfg.channel_ramp <= 0 or cfg.channel == "clean":
        return [target]
    try:
        clean = Channel("clean", ss=cfg.ss)
        tp, cp = target.params, clean.params
        assert is_dataclass(tp) and type(tp) is type(cp)
        out = []
        for s in np.linspace(0.0, 1.0, RAMP_LEVELS)[:-1]:
            ov = {f.name: _interp(getattr(cp, f.name), getattr(tp, f.name), float(s), f.name)
                  for f in fields(tp)}
            out.append(Channel(cfg.channel, ss=cfg.ss, **ov))
        return out + [target]
    except Exception as e:  # unsupported channel API -> no ramp
        log(f"[train] channel ramp unavailable ({e!r}); training on the target preset")
        return [target]


def channel_at(cfg: JointConfig, ramp: list, step: int):
    if len(ramp) == 1:
        return ramp[0]
    frac = (step - 1) / max(1.0, cfg.channel_ramp * cfg.steps)
    return ramp[min(len(ramp) - 1, int(frac * (len(ramp) - 1)))]


def heldout_channel(cfg: JointConfig, Channel):
    """-> (channel, call kwargs) for the held-out test-only degradations, or None."""
    try:
        if "test_only" in inspect.signature(Channel.__init__).parameters:     # LG1 API
            return Channel(cfg.channel, ss=cfg.ss, test_only=True), {}
        mod = importlib.import_module(Channel.__module__)
        presets = getattr(mod, "PRESETS", {})
        for name in ("heldout", "test_only"):
            if name in presets:
                return Channel(name, ss=cfg.ss), {}
        ch = Channel(cfg.channel, ss=cfg.ss)
        params = inspect.signature(ch.__call__).parameters
        for k in ("test_only", "heldout"):
            if k in params:
                return ch, {k: True}
        if callable(getattr(ch, "heldout", None)):
            return ch.heldout(), {}
    except Exception:
        pass
    return None


# ============================================================================ data
def load_eval_lines(cfg: JointConfig) -> tuple[list[str], list[str]]:
    """(val, test) lines exactly as Phase 2 (test first in the 'val' stream, then val)."""
    from vbpe.crnn.data import source_lines
    test, _ = source_lines("val", cfg.n_test, skip=0, seed=cfg.data_seed)
    val, _ = source_lines("val", cfg.n_val, skip=cfg.n_test, seed=cfg.data_seed)
    return val, test


class TrainStream:
    """Infinite batches of (texts, ids) from ``iter_lines('train')``; within a buffer of
    batch*bucket_mult lines sort by rendered width, cut into batches, shuffle batch order.
    ``line_weight`` (symbol id -> weight array): lines drawn from a 3x pool with probability
    proportional to the max weight of their symbols (rare-token oversampling)."""

    POOL = 3

    def __init__(self, alphabet: Alphabet, widths_ss: np.ndarray, batch: int, bucket_mult: int = 20,
                 seed: int = 0, data_seed: int = 0, lines=None, line_weight: np.ndarray | None = None):
        from vbpe.corpus import iter_lines
        self.alphabet, self.wss, self.batch, self.mult = alphabet, widths_ss, batch, bucket_mult
        self.it = iter(lines) if lines is not None else iter_lines("train", 30, 70, seed=data_seed)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.weight = line_weight
        self.queue: list[list[tuple[str, list[int]]]] = []

    def _refill(self) -> None:
        n = self.batch * self.mult
        buf = []
        for _ in range(n * (self.POOL if self.weight is not None else 1)):
            t = normalize(next(self.it))
            ids = self.alphabet.encode(t)
            if ids:
                buf.append((t, ids, int(self.wss[ids].sum())))
        if self.weight is not None and len(buf) > n:
            sc = np.array([float(self.weight[r[1]].max()) for r in buf])
            pick = self.np_rng.choice(len(buf), size=n, replace=False, p=sc / sc.sum())
            buf = [buf[i] for i in sorted(pick)]
        buf.sort(key=lambda r: r[2])
        bs = [buf[i:i + self.batch] for i in range(0, len(buf), self.batch)]
        bs = [b for b in bs if len(b) == self.batch] or bs
        self.rng.shuffle(bs)
        self.queue = [[(t, ids) for t, ids, _ in b] for b in bs]

    def __next__(self):
        if not self.queue:
            self._refill()
        b = self.queue.pop()
        return [t for t, _ in b], [ids for _, ids in b]

    def __iter__(self):
        return self


def rare_weights(counts: np.ndarray, n_chars: int, alpha: float, cap: float = 20.0) -> np.ndarray:
    w = np.ones(len(counts))
    tok = counts[n_chars:].astype(float) + 1.0
    if len(tok) and alpha > 0:
        w[n_chars:] = np.clip((np.median(tok) / tok) ** alpha, 1.0, cap)
    return w


def channel_padding(channel, height_ss: int, ss: int) -> int:
    """Extra px the channel adds to a line's width (probed in eval mode on blank canvases)."""
    import torch
    pads = []
    for w_ss in (8 * ss, 40 * ss + 2):
        canvas = torch.zeros(1, 1, height_ss, w_ss)
        _, wpx = channel(canvas, torch.tensor([w_ss]), generator=torch.Generator().manual_seed(0),
                         train=False)
        pads.append(int(wpx[0]) - (-(-w_ss // ss)))
    return max(0, min(pads))


def render(ink, ids_batch, widths_ss_t, compose, channel, cfg, generator, train: bool,
           chan_kw: dict | None = None):
    import torch.nn.functional as F
    canvas, wss = compose(ink, ids_batch, widths_ss_t)
    p = cfg.pad_px * cfg.ss
    if p > 0:
        canvas, wss = F.pad(canvas, (p, p)), wss + 2 * p
    return channel(canvas, wss, generator=generator, train=train, **(chan_kw or {}))


# ============================================================================ classes
class ClassMap:
    """Symbol id <-> reader class. Default: blank 0, symbol i -> class i + 1. For a frozen Phase-2
    reader: classes = Vocab(itos = <blank>, <unk>, labels...), chars outside its vocab -> <unk>."""

    def __init__(self, alphabet: Alphabet, vocab: list[str] | None = None):
        if vocab is None:
            self.sym2cls = np.arange(len(alphabet)) + 1
            self.n_classes = len(alphabet) + 1
        else:
            stoi = {s: i + 2 for i, s in enumerate(vocab)}
            self.sym2cls = np.array([stoi.get(s, 1) for s in alphabet.symbols])
            self.n_classes = len(vocab) + 2
        self.cls2sym = {int(c): i for i, c in enumerate(self.sym2cls) if c > 1 or vocab is None}

    def targets(self, ids_batch):
        import torch
        tg = torch.tensor([int(self.sym2cls[i]) for s in ids_batch for i in s], dtype=torch.long)
        tl = torch.tensor([len(s) for s in ids_batch], dtype=torch.long)
        return tg, tl

    def decode(self, classes) -> list[int]:
        return [self.cls2sym[c] for c in classes if c in self.cls2sym]


def _ctc_targets(ids_batch):
    """Default mapping (symbol i -> class i + 1)."""
    import torch
    tg = torch.tensor([i + 1 for s in ids_batch for i in s], dtype=torch.long)
    tl = torch.tensor([len(s) for s in ids_batch], dtype=torch.long)
    return tg, tl


def load_frozen_reader(path: str, alphabet: Alphabet):
    import torch
    from vbpe.crnn.model import CRNN
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vocab = ck["vocab"]
    model_kw = (ck.get("config") or {}).get("model") or {}
    reader = CRNN(len(vocab) + 2, col_stride=ck["col_stride"], expand=ck["expand"], **model_kw)
    reader.load_state_dict(ck["state"])
    for p in reader.parameters():
        p.requires_grad_(False)
    reader.eval()
    em = (ck.get("config") or {}).get("em")
    return reader, ClassMap(alphabet, vocab), ck["col_stride"], ck["expand"], em


# ============================================================================ eval
def evaluate(ink, reader, alphabet: Alphabet, texts: list[str], ids: list[list[int]], widths_ss_t,
             compose, channel, cfg: JointConfig, batch: int = 64, seed: int = 12345,
             return_preds: bool = False, cmap: ClassMap | None = None,
             chan_kw: dict | None = None) -> dict:
    import torch
    from rapidfuzz.distance import Levenshtein
    from vbpe.crnn.model import greedy_decode

    cmap = cmap or ClassMap(alphabet)
    was_training = reader.training
    reader.eval()
    w = np.array([int(widths_ss_t[i].sum()) if len(i) else 0 for i in ids])
    order = np.argsort(w, kind="stable")
    edits = chars = sym_edits = sym_n = 0
    preds: list[str | None] = [None] * len(texts)
    pred_ids: list[list[int] | None] = [None] * len(texts)
    with torch.no_grad():
        for b, s in enumerate(range(0, len(order), batch)):
            idx = order[s:s + batch]
            gen = torch.Generator().manual_seed(seed * 7919 + b)
            x, wpx = render(ink, [ids[i] for i in idx], widths_ss_t, compose, channel, cfg, gen,
                            train=False, chan_kw=chan_kw)
            lp, lens = reader(x, wpx)
            for i, seq in zip(idx, greedy_decode(lp, lens, blank=0)):
                hyp_ids = cmap.decode(seq)
                hyp = normalize(alphabet.decode(hyp_ids))
                ref = normalize(texts[i])
                edits += Levenshtein.distance(ref, hyp)
                chars += len(ref)
                sym_edits += Levenshtein.distance(ids[i], hyp_ids)
                sym_n += len(ids[i])
                preds[i], pred_ids[i] = hyp, hyp_ids
    reader.train(was_training)
    out = {"cer": edits / max(1, chars), "symbol_cer": sym_edits / max(1, sym_n), "n": len(texts)}
    if return_preds:
        out["preds"], out["pred_ids"] = preds, pred_ids
    return out


def glyph_report(ink, widths_ss_t, ss: int, alphabet: Alphabet, ref_ids=None, hyp_ids=None,
                 n_pairs: int = 20, n_conf: int = 30) -> dict:
    """Near-duplicate glyphs (binarised Hamming at ss res, L1 at 1x) + test substitutions."""
    import torch
    import torch.nn.functional as F
    from rapidfuzz.distance import Levenshtein

    with torch.no_grad():
        b = (ink > 0.5).float()
        n = b.shape[0]
        flat = b.reshape(n, -1)
        ham = flat @ (1 - flat).T + (1 - flat) @ flat.T
        Hs, Wm = b.shape[1:]
        Wp = -(-Wm // ss) * ss
        low = F.avg_pool2d(F.pad(ink[:, None].float(), (0, Wp - Wm)), ss).reshape(n, -1)
        l1 = torch.cdist(low, low, p=1)
        eye = torch.eye(n, dtype=torch.bool)
        ham = ham.masked_fill(eye, float("inf"))
        l1 = l1.masked_fill(eye, float("inf"))
    iu = torch.triu_indices(n, n, 1)
    hv = ham[iu[0], iu[1]]
    order = torch.argsort(hv)[:n_pairs]
    sym = alphabet.symbols
    rep = {
        "min_hamming_ss": float(hv.min()) if hv.numel() else None,
        "median_nn_hamming_ss": float(ham.min(1).values.median()) if n > 1 else None,
        "n_pairs_lt_1px": int((hv < ss * ss).sum()),
        "min_l1_1x": float(l1.min()) if n > 1 else None,
        "median_nn_l1_1x": float(l1.min(1).values.median()) if n > 1 else None,
        "nearest_pairs": [[sym[int(iu[0, k])], sym[int(iu[1, k])], float(hv[k])] for k in order],
    }
    if ref_ids is not None and hyp_ids is not None:
        conf: Counter = Counter()
        for r, h in zip(ref_ids, hyp_ids):
            for op in Levenshtein.editops(r, h or []):
                if op.tag == "replace":
                    conf[(r[op.src_pos], h[op.dest_pos])] += 1
                elif op.tag == "delete":
                    conf[(r[op.src_pos], -1)] += 1
        rep["top_confusions"] = [[sym[a], sym[b] if b >= 0 else "<del>", c]
                                 for (a, b), c in conf.most_common(n_conf)]
    return rep


def entropy_bits_per_char(counts: np.ndarray, n_chars_text: int) -> float:
    c = counts[counts > 0].astype(float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum() * c.sum() / max(1, n_chars_text))


# ============================================================================ train
def train_joint(cfg: JointConfig, log=print, train_lines=None, eval_lines=None,
                save: bool = True) -> list[dict]:
    """Train bank + reader(s) jointly (or glyphs only against a frozen reader); returns [row]
    and (if ``save``) appends it to ``<out_dir>/runs.jsonl`` and writes banks/, atlas/, readers/,
    curves/, samples/, glyph_report/ files.

    ``train_lines`` / ``eval_lines=(val, test)`` override the corpus (tests)."""
    import torch
    from torch import nn
    from vbpe.crnn import data as D
    from vbpe.crnn.model import CRNN, n_params

    torch.set_num_threads(cfg.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    _, compose, Channel = lg1()
    tag = cfg.tag
    out_dir = cfg.out_dir()
    if cfg.frozen_reader and not cfg.learn_glyphs:
        raise ValueError("frozen_reader with a fixed bank: nothing to train")

    alphabet = Alphabet.build(cfg.n_tokens)
    bank = build_bank(cfg, alphabet)
    ramp = make_channel_ramp(cfg, Channel, log)
    channel = ramp[-1]
    held = heldout_channel(cfg, Channel) if cfg.heldout_eval else None
    widths_ss_t = torch.as_tensor(bank.widths_ss, dtype=torch.long)
    wss_np = widths_ss_t.numpy().astype(np.int64)
    ss = cfg.ss
    height_ss = int(getattr(bank, "height_ss", cfg.height_px * ss))

    # ---- data
    val_t, test_t = eval_lines if eval_lines is not None else load_eval_lines(cfg)
    val_ids = [alphabet.encode(t) for t in val_t]
    test_ids = [alphabet.encode(t) for t in test_t]
    if train_lines is None:
        from itertools import islice
        from vbpe.corpus import iter_lines
        probe = [normalize(t) for t in islice(iter_lines("train", 30, 70, seed=cfg.data_seed),
                                              cfg.n_expand_probe)]
    else:
        probe = list(train_lines[: cfg.n_expand_probe])
    probe_ids = [alphabet.encode(t) for t in probe]
    counts = np.bincount(np.concatenate([np.asarray(i, dtype=np.int64) for i in probe_ids if i]),
                         minlength=len(alphabet))
    bits_per_char = entropy_bits_per_char(counts, sum(len(t) for t in probe))
    lw = rare_weights(counts, alphabet.n_chars, cfg.oversample_alpha) \
        if cfg.oversample_alpha > 0 and alphabet.n_tokens else None
    stream = TrainStream(alphabet, wss_np, cfg.batch, cfg.bucket_mult, seed=cfg.seed,
                         data_seed=cfg.data_seed, lines=train_lines, line_weight=lw)

    # ---- reader(s) (Phase-2 geometry rule) or frozen reader
    pad = channel_padding(channel, height_ss, ss) + 2 * cfg.pad_px
    all_ids = val_ids + test_ids + probe_ids
    w_px = np.array([-(-int(wss_np[i].sum()) // ss) + pad for i in all_ids])
    L = np.array([len(i) for i in all_ids])
    if cfg.frozen_reader:
        if cfg.n_tokens:
            raise ValueError("frozen Phase-2 readers have a character vocab: use n_tokens=0")
        reader0, cmap, stride, expand, fr_em = load_frozen_reader(cfg.frozen_reader, alphabet)
        if fr_em is not None and int(round(1.2 * fr_em)) != cfg.height_px:
            log(f"[{tag}] WARNING frozen reader em={fr_em} expects height_px="
                f"{int(round(1.2 * fr_em))}, got {cfg.height_px}")
        readers = [reader0]
        bad = int((D.n_frames(w_px, stride, expand) < 2 * L + 1).sum())
        if bad:
            log(f"[{tag}] WARNING {bad} probe/eval lines violate T >= 2L+1 for the frozen reader")
    else:
        cmap = ClassMap(alphabet)
        stride = D.col_stride_for_em(cfg.em)
        expand = D.choose_expand(w_px, L, stride)

        def new_reader():
            return CRNN(cmap.n_classes, col_stride=stride, expand=expand, **cfg.model)
        readers = [new_reader() for _ in range(max(1, cfg.n_readers))]

    # ---- glyph-side masks / penalties
    gparams = glyph_params(bank) if cfg.learn_glyphs else []
    with torch.no_grad():
        shape = tuple(bank(tau=1.0, hard=False).shape)
    cmask = cell_mask(bank, widths_ss_t, shape)
    gmask = None
    if cfg.learn_glyphs and cfg.guard_ss > 0 and not bank_has_guard(type(bank)):
        gmask = guard_mask(widths_ss_t, shape, cfg.guard_ss)
        lg = getattr(bank, "logits", None)
        if lg is not None:
            with torch.no_grad():
                lg.masked_fill_(gmask == 0, GUARD_LOGIT)

    # ---- optimiser
    groups, max_lr = [], []
    if not cfg.frozen_reader:
        for r in readers:
            groups.append({"params": list(r.parameters()), "lr": cfg.lr_reader,
                           "weight_decay": cfg.weight_decay})
            max_lr.append(cfg.lr_reader)
    if gparams:
        groups.append({"params": gparams, "lr": cfg.lr_glyph, "weight_decay": cfg.glyph_weight_decay})
        max_lr.append(cfg.lr_glyph)
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=cfg.steps,
                                                pct_start=cfg.pct_start)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    resets = reset_steps(cfg)

    files = {}
    if save:
        for sub in ("curves", "banks", "atlas", "readers", "samples", "glyph_report"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        files["curve"] = open(out_dir / "curves" / f"{tag}.jsonl", "w")
    log(f"[{tag}] symbols={len(alphabet)} classes={cmap.n_classes} learn={cfg.learn_glyphs} "
        f"mode={cfg.train_mode} h={cfg.height_px} cell_w={cfg.char_w_px}/{cfg.token_w_px} "
        f"col_stride={stride} expand={expand} pad={pad} readers={len(readers)} "
        f"reader_params={n_params(readers[0])} glyph_params={sum(p.numel() for p in gparams)} "
        f"channel={cfg.channel} ramp_levels={len(ramp)} heldout={held is not None} "
        f"resets={resets} bits/char={bits_per_char:.3f} oversample={cfg.oversample_alpha}")

    def bank_ink(step: int | None, train: bool):
        """Glyph ink for a train step (tau/hard schedule) or for eval (hard)."""
        if not cfg.learn_glyphs:
            ink = bank(tau=1.0, hard=False)
            return (ink > 0.5).to(ink.dtype) if cfg.fixed_hard else ink
        if not train:
            ink = bank(tau=cfg.tau_end, hard=True)
        else:
            ink = bank(tau=tau_at(cfg, step), hard=hard_at(cfg, step))
        return ink * gmask if gmask is not None else ink

    fixed_ink = None
    if not cfg.learn_glyphs:
        with torch.no_grad():
            fixed_ink = bank_ink(None, False).detach()

    def snapshot(cer, step, ri):
        return {"cer": cer, "step": step, "reader_idx": ri,
                "reader": copy.deepcopy(readers[ri].state_dict()),
                "bank": copy.deepcopy(bank.state_dict())}

    best = {"cer": math.inf, "step": 0, "reader_idx": 0, "reader": None, "bank": None}
    loss_acc = pen_acc = n_acc = 0
    skipped = 0
    n_resets_done = 0
    t0, c0 = time.time(), time.process_time()
    eval_time = 0.0
    for r in readers:
        r.train(not cfg.frozen_reader)
    for step in range(1, cfg.steps + 1):
        if step in resets:
            ri = n_resets_done % len(readers)
            fresh = CRNN(cmap.n_classes, col_stride=stride, expand=expand, **cfg.model)
            readers[ri].load_state_dict(fresh.state_dict())
            for p in readers[ri].parameters():
                opt.state.pop(p, None)
            n_resets_done += 1
            best = {"cer": math.inf, "step": 0, "reader_idx": 0, "reader": None, "bank": None}
            log(f"[{tag}] step {step}: reader {ri} re-initialised ({n_resets_done}/{len(resets)})")
        texts, ids_b = next(stream)
        ink = fixed_ink if fixed_ink is not None else bank_ink(step, True)
        x, wpx = render(ink, ids_b, widths_ss_t, compose, channel_at(cfg, ramp, step), cfg, gen,
                        train=True)
        tg, tl = cmap.targets(ids_b)
        loss = 0.0
        for r in readers:
            lp, lens = r(x, wpx)
            loss = loss + ctc(lp, tg, lens, tl)
        loss = loss / len(readers)
        skipped += int((lens < 2 * tl + 1).sum())
        pen = torch.zeros(())
        pw = min(1.0, step / max(1.0, cfg.penalty_ramp * cfg.steps)) if cfg.penalty_ramp > 0 else 1.0
        if gparams:
            if cfg.min_feature_weight:
                pen = pen + cfg.min_feature_weight * feature_penalty(
                    bank, ink, gmask if gmask is not None else cmask, ss)
            if cfg.coverage_weight:
                pen = pen + cfg.coverage_weight * coverage_penalty(
                    ink, cmask, cfg.coverage_lo, cfg.coverage_hi)
            area = cmask.sum().clamp_min(1.0)
            if cfg.ink_penalty:
                pen = pen + cfg.ink_penalty * ink.sum() / area
            if cfg.entropy_penalty:
                p = bank(tau=tau_at(cfg, step), hard=False).clamp(1e-6, 1 - 1e-6)
                ent = -(p * p.log() + (1 - p) * (1 - p).log()) * cmask
                pen = pen + cfg.entropy_penalty * ent.sum() / area
        pen = pw * pen
        total = loss + pen
        opt.zero_grad(set_to_none=True)
        total.backward()
        if not cfg.frozen_reader:
            for r in readers:
                nn.utils.clip_grad_norm_(r.parameters(), cfg.grad_clip)
        if gparams and cfg.glyph_grad_clip:
            nn.utils.clip_grad_norm_(gparams, cfg.glyph_grad_clip)
        opt.step()
        sched.step()
        loss_acc += float(loss.detach())
        pen_acc += float(pen.detach())
        n_acc += 1
        if step % cfg.eval_every == 0 or step == cfg.steps:
            te = time.time()
            with torch.no_grad():
                ink_e = bank_ink(None, False)
                evs = [evaluate(ink_e, r, alphabet, val_t, val_ids, widths_ss_t, compose, channel,
                                cfg, cmap=cmap) for r in readers]
                ri = int(np.argmin([e["cer"] for e in evs]))
                ev = evs[ri]
                ev_soft = None
                if cfg.eval_soft and cfg.learn_glyphs:
                    soft = bank(tau=tau_at(cfg, step), hard=False)
                    soft = soft * gmask if gmask is not None else soft
                    ev_soft = evaluate(soft, readers[ri], alphabet, val_t, val_ids, widths_ss_t,
                                       compose, channel, cfg, cmap=cmap)
                mean_ink = float((ink_e * cmask).sum() / cmask.sum().clamp_min(1.0))
            eval_time += time.time() - te
            lrs = sched.get_last_lr()
            pt = {"step": step, "train_loss": round(loss_acc / n_acc, 4),
                  "penalty": round(pen_acc / n_acc, 5),
                  "val_cer": round(ev["cer"], 5), "val_symbol_cer": round(ev["symbol_cer"], 5),
                  "val_cer_by_reader": [round(e["cer"], 5) for e in evs] if len(evs) > 1 else None,
                  "tau": round(tau_at(cfg, step), 4), "hard": hard_at(cfg, step),
                  "channel_level": ramp.index(channel_at(cfg, ramp, step)), "mean_ink": round(mean_ink, 4),
                  "lr_reader": None if cfg.frozen_reader else lrs[0],
                  "lr_glyph": lrs[-1] if gparams else None,
                  "ctc_infeasible": skipped, "elapsed_s": round(time.time() - t0, 1)}
            if ev_soft is not None:
                pt["val_cer_soft"] = round(ev_soft["cer"], 5)
            if "curve" in files:
                files["curve"].write(json.dumps(pt) + "\n")
                files["curve"].flush()
            log(f"[{tag}] step {step} loss {pt['train_loss']:.3f} pen {pt['penalty']:.4f} "
                f"val_cer {ev['cer']:.4f}"
                + (f" (soft {ev_soft['cer']:.4f})" if ev_soft else "")
                + (f" readers {pt['val_cer_by_reader']}" if len(evs) > 1 else "")
                + f" tau {pt['tau']:.3f} hard={int(pt['hard'])} chan={pt['channel_level']} "
                f"ink={mean_ink:.3f} ({pt['elapsed_s']}s)")
            loss_acc = pen_acc = n_acc = 0
            if ev["cer"] <= best["cer"]:          # ties -> later checkpoint
                best = snapshot(ev["cer"], step, ri)
    train_time = time.time() - t0
    train_cpu = time.process_time() - c0
    if "curve" in files:
        files["curve"].close()
    if best["reader"] is None:            # (only if a reset happened after the last eval)
        best = snapshot(math.inf, cfg.steps, 0)
    reader = readers[best["reader_idx"]]
    reader.load_state_dict(best["reader"])
    bank.load_state_dict(best["bank"])

    # ---- test
    with torch.no_grad():
        ink = bank_ink(None, False)
        ev = evaluate(ink, reader, alphabet, test_t, test_ids, widths_ss_t, compose, channel, cfg,
                      seed=54321, return_preds=True, cmap=cmap)
        ev_h = None
        if held is not None:
            ev_h = evaluate(ink, reader, alphabet, test_t, test_ids, widths_ss_t, compose, held[0],
                            cfg, seed=54321, cmap=cmap, chan_kw=held[1])
        mean_ink = float((ink * cmask).sum() / cmask.sum().clamp_min(1.0))
        grep = glyph_report(ink.detach(), widths_ss_t, ss, alphabet, test_ids, ev["pred_ids"])
    ppc = float(np.mean([cfg.height_px * (wss_np[i].sum() / ss) / max(1, len(normalize(t)))
                         for t, i in zip(test_t, test_ids)]))
    n_sym = sum(len(i) for i in test_ids)
    n_chr = sum(len(normalize(t)) for t in test_t)
    row = {
        "phase": "lg", "run": cfg.run_name, "train_mode": cfg.train_mode, "scheme": cfg.scheme,
        "em": cfg.em, "height_px": cfg.height_px, "px_per_char": ppc,
        "cer": ev["cer"], "symbol_cer": ev["symbol_cer"], "steps": cfg.steps,
        "channel": cfg.channel, "n_tokens": cfg.n_tokens, "learn_glyphs": cfg.learn_glyphs,
        "train_time_s": round(train_time, 1),
        "cer_heldout": ev_h["cer"] if ev_h else None,
        "symbol_cer_heldout": ev_h["symbol_cer"] if ev_h else None,
        "bits_per_char": bits_per_char,
        "bits_per_px2": bits_per_char * max(0.0, 1 - ev["cer"]) / ppc,
        "bits_per_px2_sym": bits_per_char * max(0.0, 1 - ev["symbol_cer"]) / ppc,
        # diagnostics
        "tag": tag, "best_step": best["step"], "best_val_cer": best["cer"],
        "best_reader": best["reader_idx"], "n_readers": len(readers), "reader_resets": len(resets),
        "frozen_reader": cfg.frozen_reader,
        "n_params": n_params(reader), "glyph_params": sum(p.numel() for p in gparams),
        "n_symbols": len(alphabet), "col_stride": stride, "expand": expand, "channel_pad_px": pad,
        "char_w": cfg.char_w_px if cfg.learn_glyphs else None,
        "token_w": cfg.token_w_px if cfg.learn_glyphs else None,
        "seg_ratio": n_sym / max(1, n_chr), "mean_ink": mean_ink,
        "min_hamming_ss": grep["min_hamming_ss"], "n_pairs_lt_1px": grep["n_pairs_lt_1px"],
        "min_l1_1x": grep["min_l1_1x"],
        "s_per_step": round((train_time - eval_time) / cfg.steps, 4),
        "train_cpu_s": round(train_cpu, 1), "eval_time_s": round(eval_time, 1),
        "ctc_infeasible": skipped, "n_val": len(val_t), "n_test": len(test_t),
        "threads": cfg.threads, "seed": cfg.seed, "config": asdict(cfg),
    }
    log(f"[{tag}] TEST cer={row['cer']:.4f} symbol_cer={row['symbol_cer']:.4f} "
        f"heldout={row['cer_heldout']} px/char={ppc:.2f} bits/px2={row['bits_per_px2']:.4f} "
        f"min_ham={grep['min_hamming_ss']} best_step={best['step']} time={train_time:.0f}s "
        f"s/step={row['s_per_step']}")
    if save:
        bp = out_dir / "banks" / f"{tag}.pt"
        bank.save(bp)
        row["bank_path"] = str(bp)
        if hasattr(bank, "atlas_png"):
            try:
                try:
                    bank.atlas_png(out_dir / "atlas" / f"{tag}.png", alphabet.symbols,
                                   tau=cfg.tau_end, hard=cfg.learn_glyphs)
                except TypeError:
                    bank.atlas_png(out_dir / "atlas" / f"{tag}.png", alphabet.symbols)
            except Exception as e:  # atlas is a diagnostic; never lose a run over it
                log(f"[{tag}] atlas_png failed: {e!r}")
        torch.save({"state": best["reader"], "n_classes": cmap.n_classes, "col_stride": stride,
                    "expand": expand, "model": cfg.model, "config": asdict(cfg),
                    "frozen_reader": cfg.frozen_reader}, out_dir / "readers" / f"{tag}.pt")
        (out_dir / "glyph_report" / f"{tag}.json").write_text(json.dumps(grep, indent=1))
        with open(out_dir / "samples" / f"{tag}.txt", "w") as fh:
            for i in range(min(cfg.n_samples, len(test_t))):
                fh.write(f"REF {test_t[i]}\nHYP {ev['preds'][i]}\n\n")
        with open(out_dir / "runs.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return [row]


def run_job(cfg_dict: dict) -> list[dict]:
    """Process-pool entry point: train one config, logging to <out_dir>/logs/<tag>.log."""
    cfg = JointConfig.from_dict(cfg_dict)
    log_path = cfg.out_dir() / "logs" / f"{cfg.tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        def log(msg: str) -> None:
            fh.write(msg + "\n")
            fh.flush()
            print(msg, flush=True)
        return train_joint(cfg, log=log)
