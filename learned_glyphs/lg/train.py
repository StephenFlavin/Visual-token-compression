"""Joint training of a glyph bank and a CRNN reader through the rendering channel (WP-LG2, E1).

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
  * optional regularisers (off by default): ``ink_penalty`` x mean ink, ``entropy_penalty`` x mean
    binary entropy of the soft glyphs (pushes towards binary).

px_per_char of a line = height_px (line pitch) x composed content width (px, margins/padding
excluded) / len(normalize(text)); the row reports the mean over test lines (as Phase 2).
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from vbpe import config
from vbpe.metrics import normalize

from .alphabet import Alphabet

LG_ROOT = Path(__file__).resolve().parent.parent
RESULTS = LG_ROOT / "results"
FONTS = {"latin": config.LATIN_FONT, "mono": config.LATIN_MONO_FONT,
         "condensed": config.CONDENSED_FONT}

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
    init: str = "random"                  # GlyphBank init (learned banks)
    font: str = "latin"                   # fixed banks: latin | condensed | mono | <font path>
    token_glyphs: str = "cjk"             # fixed banks with tokens: cjk (subst ideographs) | text
    fixed_hard: bool = False              # fixed banks: threshold PIL coverage at 0.5 (default:
                                          # keep antialiased ss coverage = Phase-2 rendering)
    ss: int = 4
    channel: str = "default"              # Channel preset: clean | default | harsh
    pad_px: int = 0                       # extra zero padding (px) each side before the channel
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
    ink_penalty: float = 0.0
    entropy_penalty: float = 0.0
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
    def scheme(self) -> str:
        s = f"lg_{'learned' if self.learn_glyphs else 'fixed'}_n{self.n_tokens}"
        if not self.learn_glyphs and self.n_tokens and self.token_glyphs != "cjk":
            s += f"_{self.token_glyphs}"
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


def build_bank(cfg: JointConfig, alphabet: Alphabet):
    GlyphBank, _, _ = lg1()
    font = FONTS.get(cfg.font, cfg.font)
    cjk_map = None
    if alphabet.n_tokens and cfg.token_glyphs == "cjk":
        from vbpe.schemes.substitution import load_symbols
        pool = load_symbols()
        cjk_map = dict(zip(alphabet.tokens, pool[: alphabet.n_tokens]))
    if cfg.learn_glyphs:
        widths = [cfg.char_w_px] * alphabet.n_chars + [cfg.token_w_px] * alphabet.n_tokens
        kw = {}
        if not getattr(GlyphBank, "STUB", False):     # LG1 extras: init="font" source, init seed
            kw = dict(symbols=alphabet.symbols, font_path=str(font), cjk_map=cjk_map,
                      seed=cfg.seed)
        return GlyphBank(len(alphabet), cfg.height_px, widths, ss=cfg.ss, init=cfg.init,
                         learnable=True, **kw)
    return GlyphBank.from_font(alphabet.symbols, str(font), cfg.height_px, ss=cfg.ss,
                               cjk_map=cjk_map)


def glyph_params(bank) -> list:
    return [p for p in bank.parameters() if p.requires_grad]


# ============================================================================ schedules
def tau_at(cfg: JointConfig, step: int) -> float:
    frac = (step - 1) / max(1, cfg.steps - 1)
    return float(cfg.tau_start * (cfg.tau_end / cfg.tau_start) ** frac)


def hard_at(cfg: JointConfig, step: int) -> bool:
    return (step - 1) >= cfg.hard_from * cfg.steps


# ============================================================================ data
def load_eval_lines(cfg: JointConfig) -> tuple[list[str], list[str]]:
    """(val, test) lines exactly as Phase 2 (test first in the 'val' stream, then val)."""
    from vbpe.crnn.data import source_lines
    test, _ = source_lines("val", cfg.n_test, skip=0, seed=cfg.data_seed)
    val, _ = source_lines("val", cfg.n_val, skip=cfg.n_test, seed=cfg.data_seed)
    return val, test


class TrainStream:
    """Infinite batches of (texts, ids) from ``iter_lines('train')``; within a buffer of
    batch*bucket_mult lines sort by rendered width, cut into batches, shuffle batch order."""

    def __init__(self, alphabet: Alphabet, widths_ss: np.ndarray, batch: int, bucket_mult: int = 20,
                 seed: int = 0, data_seed: int = 0, lines=None):
        from vbpe.corpus import iter_lines
        self.alphabet, self.wss, self.batch, self.mult = alphabet, widths_ss, batch, bucket_mult
        self.it = iter(lines) if lines is not None else iter_lines("train", 30, 70, seed=data_seed)
        self.rng = random.Random(seed)
        self.queue: list[list[tuple[str, list[int]]]] = []

    def _refill(self) -> None:
        buf = []
        for _ in range(self.batch * self.mult):
            t = normalize(next(self.it))
            ids = self.alphabet.encode(t)
            if ids:
                buf.append((t, ids, int(self.wss[ids].sum())))
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


def _pad(canvas, wss, pad_ss: int):
    import torch.nn.functional as F
    if pad_ss <= 0:
        return canvas, wss
    return F.pad(canvas, (pad_ss, pad_ss)), wss + 2 * pad_ss


def render(ink, ids_batch, widths_ss_t, compose, channel, cfg, generator, train: bool):
    canvas, wss = compose(ink, ids_batch, widths_ss_t)
    canvas, wss = _pad(canvas, wss, cfg.pad_px * cfg.ss)
    return channel(canvas, wss, generator=generator, train=train)


# ============================================================================ eval
def evaluate(ink, reader, alphabet: Alphabet, texts: list[str], ids: list[list[int]], widths_ss_t,
             compose, channel, cfg: JointConfig, batch: int = 64, seed: int = 12345,
             return_preds: bool = False) -> dict:
    import torch
    from rapidfuzz.distance import Levenshtein
    from vbpe.crnn.model import greedy_decode

    was_training = reader.training
    reader.eval()
    w = np.array([int(widths_ss_t[i].sum()) if len(i) else 0 for i in ids])
    order = np.argsort(w, kind="stable")
    edits = chars = sym_edits = sym_n = 0
    preds: list[str | None] = [None] * len(texts)
    with torch.no_grad():
        for b, s in enumerate(range(0, len(order), batch)):
            idx = order[s:s + batch]
            gen = torch.Generator().manual_seed(seed * 7919 + b)
            x, wpx = render(ink, [ids[i] for i in idx], widths_ss_t, compose, channel, cfg, gen,
                            train=False)
            lp, lens = reader(x, wpx)
            for i, seq in zip(idx, greedy_decode(lp, lens, blank=0)):
                hyp_ids = Alphabet.from_classes(seq)
                hyp = normalize(alphabet.decode(hyp_ids))
                ref = normalize(texts[i])
                edits += Levenshtein.distance(ref, hyp)
                chars += len(ref)
                sym_edits += Levenshtein.distance(ids[i], hyp_ids)
                sym_n += len(ids[i])
                preds[i] = hyp
    reader.train(was_training)
    out = {"cer": edits / max(1, chars), "symbol_cer": sym_edits / max(1, sym_n), "n": len(texts)}
    if return_preds:
        out["preds"] = preds
    return out


# ============================================================================ train
def _ctc_targets(ids_batch):
    import torch
    tg = torch.tensor([i + 1 for s in ids_batch for i in s], dtype=torch.long)
    tl = torch.tensor([len(s) for s in ids_batch], dtype=torch.long)
    return tg, tl


def train_joint(cfg: JointConfig, log=print, train_lines=None, eval_lines=None,
                save: bool = True) -> list[dict]:
    """Train bank + reader jointly; returns [row] and (if ``save``) appends it to
    ``<out_dir>/runs.jsonl`` and writes banks/, atlas/, readers/, curves/, samples/ files.

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

    alphabet = Alphabet.build(cfg.n_tokens)
    bank = build_bank(cfg, alphabet)
    channel = Channel(cfg.channel, ss=cfg.ss)
    widths_ss_t = torch.as_tensor(bank.widths_ss, dtype=torch.long)
    wss_np = widths_ss_t.numpy().astype(np.int64)
    ss = cfg.ss

    # ---- data
    val_t, test_t = eval_lines if eval_lines is not None else load_eval_lines(cfg)
    val_ids = [alphabet.encode(t) for t in val_t]
    test_ids = [alphabet.encode(t) for t in test_t]
    stream = TrainStream(alphabet, wss_np, cfg.batch, cfg.bucket_mult, seed=cfg.seed,
                         data_seed=cfg.data_seed, lines=train_lines)
    if train_lines is None:
        from itertools import islice
        from vbpe.corpus import iter_lines
        probe = [normalize(t) for t in islice(iter_lines("train", 30, 70, seed=cfg.data_seed),
                                              cfg.n_expand_probe)]
    else:
        probe = list(train_lines[: cfg.n_expand_probe])
    probe_ids = [alphabet.encode(t) for t in probe]

    # ---- reader geometry (Phase-2 rule)
    height_ss = int(getattr(bank, "height_ss", cfg.height_px * ss))
    pad = channel_padding(channel, height_ss, ss) + 2 * cfg.pad_px
    all_ids = val_ids + test_ids + probe_ids
    w_px = np.array([-(-int(wss_np[i].sum()) // ss) + pad for i in all_ids])
    L = np.array([len(i) for i in all_ids])
    stride = D.col_stride_for_em(cfg.em)
    expand = D.choose_expand(w_px, L, stride)
    reader = CRNN(alphabet.n_classes, col_stride=stride, expand=expand, **cfg.model)

    # ---- optimiser
    gparams = glyph_params(bank) if cfg.learn_glyphs else []
    groups = [{"params": list(reader.parameters()), "lr": cfg.lr_reader,
               "weight_decay": cfg.weight_decay}]
    max_lr = [cfg.lr_reader]
    if gparams:
        groups.append({"params": gparams, "lr": cfg.lr_glyph, "weight_decay": cfg.glyph_weight_decay})
        max_lr.append(cfg.lr_glyph)
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=cfg.steps,
                                                pct_start=cfg.pct_start)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    files = {}
    if save:
        for sub in ("curves", "banks", "atlas", "readers", "samples"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        files["curve"] = open(out_dir / "curves" / f"{tag}.jsonl", "w")
    log(f"[{tag}] symbols={len(alphabet)} classes={alphabet.n_classes} learn={cfg.learn_glyphs} "
        f"h={cfg.height_px} cell_w={cfg.char_w_px}/{cfg.token_w_px} col_stride={stride} "
        f"expand={expand} pad={pad} reader_params={n_params(reader)} "
        f"glyph_params={sum(p.numel() for p in gparams)} channel={cfg.channel}")

    def bank_ink(step: int | None, train: bool):
        """Glyph ink for a train step (tau/hard schedule) or for eval (hard)."""
        if not cfg.learn_glyphs:
            ink = bank(tau=1.0, hard=False)
            return (ink > 0.5).to(ink.dtype) if cfg.fixed_hard else ink
        if not train:
            return bank(tau=cfg.tau_end, hard=True)
        return bank(tau=tau_at(cfg, step), hard=hard_at(cfg, step))

    fixed_ink = None
    if not cfg.learn_glyphs:
        with torch.no_grad():
            fixed_ink = bank_ink(None, False).detach()

    best = {"cer": math.inf, "step": 0, "reader": None, "bank": None}
    loss_acc = n_acc = 0
    skipped = 0
    t0, c0 = time.time(), time.process_time()
    eval_time = 0.0
    reader.train()
    for step in range(1, cfg.steps + 1):
        texts, ids_b = next(stream)
        if fixed_ink is not None:
            ink = fixed_ink
        else:
            ink = bank_ink(step, True)
        x, wpx = render(ink, ids_b, widths_ss_t, compose, channel, cfg, gen, train=True)
        lp, lens = reader(x, wpx)
        tg, tl = _ctc_targets(ids_b)
        skipped += int((lens < 2 * tl + 1).sum())
        loss = ctc(lp, tg, lens, tl)
        total = loss
        if gparams and (cfg.ink_penalty or cfg.entropy_penalty):
            area = float(height_ss * wss_np.sum())
            if cfg.ink_penalty:
                total = total + cfg.ink_penalty * ink.sum() / area
            if cfg.entropy_penalty:
                p = bank(tau=tau_at(cfg, step), hard=False).clamp(1e-6, 1 - 1e-6)
                ent = -(p * p.log() + (1 - p) * (1 - p).log())
                total = total + cfg.entropy_penalty * ent.sum() / area
        opt.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(reader.parameters(), cfg.grad_clip)
        if gparams and cfg.glyph_grad_clip:
            nn.utils.clip_grad_norm_(gparams, cfg.glyph_grad_clip)
        opt.step()
        sched.step()
        loss_acc += loss.item()
        n_acc += 1
        if step % cfg.eval_every == 0 or step == cfg.steps:
            te = time.time()
            with torch.no_grad():
                ev = evaluate(bank_ink(None, False), reader, alphabet, val_t, val_ids, widths_ss_t,
                              compose, channel, cfg)
                ev_soft = None
                if cfg.eval_soft and cfg.learn_glyphs:
                    ev_soft = evaluate(bank(tau=tau_at(cfg, step), hard=False), reader, alphabet,
                                       val_t, val_ids, widths_ss_t, compose, channel, cfg)
            eval_time += time.time() - te
            lrs = sched.get_last_lr()
            pt = {"step": step, "train_loss": round(loss_acc / n_acc, 4),
                  "val_cer": round(ev["cer"], 5), "val_symbol_cer": round(ev["symbol_cer"], 5),
                  "tau": round(tau_at(cfg, step), 4), "hard": hard_at(cfg, step),
                  "lr_reader": lrs[0], "lr_glyph": lrs[1] if len(lrs) > 1 else None,
                  "ctc_infeasible": skipped, "elapsed_s": round(time.time() - t0, 1)}
            if ev_soft is not None:
                pt["val_cer_soft"] = round(ev_soft["cer"], 5)
            if "curve" in files:
                files["curve"].write(json.dumps(pt) + "\n")
                files["curve"].flush()
            log(f"[{tag}] step {step} loss {pt['train_loss']:.3f} val_cer {ev['cer']:.4f}"
                + (f" (soft {ev_soft['cer']:.4f})" if ev_soft else "")
                + f" tau {pt['tau']:.3f} hard={int(pt['hard'])} ({pt['elapsed_s']}s)")
            loss_acc = n_acc = 0
            if ev["cer"] < best["cer"]:
                best = {"cer": ev["cer"], "step": step, "reader": copy.deepcopy(reader.state_dict()),
                        "bank": copy.deepcopy(bank.state_dict())}
    train_time = time.time() - t0
    train_cpu = time.process_time() - c0
    if "curve" in files:
        files["curve"].close()
    reader.load_state_dict(best["reader"])
    bank.load_state_dict(best["bank"])

    # ---- test
    with torch.no_grad():
        ink = bank_ink(None, False)
        ev = evaluate(ink, reader, alphabet, test_t, test_ids, widths_ss_t, compose, channel, cfg,
                      seed=54321, return_preds=True)
        mean_ink = float(ink.sum()) / float(height_ss * wss_np.sum())
    ppc = [cfg.height_px * (wss_np[i].sum() / ss) / max(1, len(normalize(t)))
           for t, i in zip(test_t, test_ids)]
    n_sym = sum(len(i) for i in test_ids)
    n_chr = sum(len(normalize(t)) for t in test_t)
    row = {
        "phase": "lg", "run": cfg.run_name, "train_mode": "joint", "scheme": cfg.scheme,
        "em": cfg.em, "height_px": cfg.height_px, "px_per_char": float(np.mean(ppc)),
        "cer": ev["cer"], "symbol_cer": ev["symbol_cer"], "steps": cfg.steps,
        "channel": cfg.channel, "n_tokens": cfg.n_tokens, "learn_glyphs": cfg.learn_glyphs,
        "train_time_s": round(train_time, 1),
        # diagnostics
        "tag": tag, "best_step": best["step"], "best_val_cer": best["cer"],
        "n_params": n_params(reader), "glyph_params": sum(p.numel() for p in gparams),
        "n_symbols": len(alphabet), "col_stride": stride, "expand": expand, "channel_pad_px": pad,
        "char_w": cfg.char_w_px if cfg.learn_glyphs else None,
        "token_w": cfg.token_w_px if cfg.learn_glyphs else None,
        "seg_ratio": n_sym / max(1, n_chr), "mean_ink": mean_ink,
        "s_per_step": round((train_time - eval_time) / cfg.steps, 4),
        "train_cpu_s": round(train_cpu, 1), "eval_time_s": round(eval_time, 1),
        "ctc_infeasible": skipped, "n_val": len(val_t), "n_test": len(test_t),
        "threads": cfg.threads, "seed": cfg.seed, "config": asdict(cfg),
    }
    log(f"[{tag}] TEST cer={row['cer']:.4f} symbol_cer={row['symbol_cer']:.4f} "
        f"px/char={row['px_per_char']:.2f} best_step={best['step']} time={train_time:.0f}s "
        f"s/step={row['s_per_step']}")
    if save:
        bank.save(out_dir / "banks" / f"{tag}.pt")
        row["bank_path"] = str((out_dir / "banks" / f"{tag}.pt").relative_to(LG_ROOT.parent)) \
            if out_dir.resolve().is_relative_to(LG_ROOT.parent) else str(out_dir / "banks" / f"{tag}.pt")
        if hasattr(bank, "atlas_png"):
            try:
                bank.atlas_png(out_dir / "atlas" / f"{tag}.png", alphabet.symbols)
            except Exception as e:  # atlas is a diagnostic; never lose a run over it
                log(f"[{tag}] atlas_png failed: {e!r}")
        torch.save({"state": best["reader"], "n_classes": alphabet.n_classes, "col_stride": stride,
                    "expand": expand, "model": cfg.model, "config": asdict(cfg)},
                   out_dir / "readers" / f"{tag}.pt")
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
