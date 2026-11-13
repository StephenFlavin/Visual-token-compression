"""Transfer (E2): a FRESH CRNN reader trained from scratch on a FROZEN glyph bank (WP-LG3).

    text -> Alphabet.encode -> ids -> frozen bank (hard glyphs) -> compose -> Channel(train) -> CRNN

Measures the bank's intrinsic legibility (no reader/glyph co-adaptation): the joint run's reader is
discarded, the bank's binarised glyphs are fixed and a new reader starts from random init with the
Phase-2 budget (``vtc.crnn.train.TrainConfig`` defaults: 3500 steps, batch 32, AdamW lr 2e-3,
wd 1e-2, OneCycle pct_start 0.15, grad clip 5, eval every 250, best-val checkpoint).

Reuse vs. duplication: ``lg.train.train_joint`` has no way to plug in a saved bank (with
``learn_glyphs=False`` it builds a *font* bank), so this module has its own ~60-line reader-only
loop but reuses every LG2 building block -- ``TrainStream`` (train lines/bucketing),
``load_eval_lines`` (Phase-2 val/test lines), ``render`` / ``evaluate`` (compose + Channel, eval with
the seeded eval channel and the same test seed 54321), ``channel_padding``, ``_ctc_targets`` -- so
data, channel, metrics and the reader geometry rule are identical to joint runs. If LG2 adds a
``bank_path`` option, ``train_reader_on_bank`` can become a thin wrapper.

Rows: ``learned_glyphs/results/<run_name>/runs.jsonl`` with the PLAN schema, train_mode="transfer",
scheme = the source bank's scheme (e.g. lg_learned_n300), plus bank_path / bank_sha8 / source_tag.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from vtc.metrics import normalize

LG_ROOT = Path(__file__).resolve().parent.parent
RESULTS = LG_ROOT / "results"


@dataclass
class TransferConfig:
    bank_path: str = ""
    channel: str = "default"
    hard: bool = True                      # binarised glyphs (learned banks); fixed banks: as stored
    # Phase-2 budget (vtc.crnn.train.TrainConfig defaults)
    steps: int = 3500
    batch: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-2
    pct_start: float = 0.15
    grad_clip: float = 5.0
    eval_every: int = 250
    # data (same lines as LG2 / Phase 2)
    n_val: int = 300
    n_test: int = 1000
    n_expand_probe: int = 4000
    bucket_mult: int = 20
    n_samples: int = 20
    pad_px: int = 0
    # misc
    seed: int = 1                          # reader init / batch order (joint runs use 0)
    data_seed: int = 0
    threads: int = 1
    run_name: str = "transfer"
    out_root: str | None = None
    scheme: str | None = None              # default: from the source run / bank metadata
    model: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "TransferConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def out_dir(self) -> Path:
        return Path(self.out_root or RESULTS) / self.run_name


# ------------------------------------------------------------------------------ bank metadata
def resolve_bank_path(p: str | Path) -> Path:
    p = Path(p)
    if not p.is_absolute() and not p.exists():
        for base in (LG_ROOT.parent, LG_ROOT, RESULTS):
            if (base / p).exists():
                return (base / p).resolve()
    return p.resolve()


def source_row(bank_path: Path) -> dict | None:
    """The joint-run row that produced ``bank_path`` (``<run>/banks/<tag>.pt`` -> <run>/runs.jsonl)."""
    run_dir = bank_path.parent.parent if bank_path.parent.name == "banks" else bank_path.parent
    f = run_dir / "runs.jsonl"
    if not f.exists():
        return None
    hit = None
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        bp = r.get("bank_path")
        if r.get("train_mode", "joint") == "joint" and (
                r.get("tag") == bank_path.stem or (bp and Path(bp).name == bank_path.name)):
            hit = r
    return hit


def bank_scheme_name(bank, n_tokens: int, src: dict | None) -> str:
    if src and src.get("scheme"):
        return src["scheme"]
    learned = getattr(bank, "logits", None) is not None
    return f"lg_{'learned' if learned else 'fixed'}_n{n_tokens}"


def load_frozen_bank(path):
    from .glyphs import GlyphBank
    bank = GlyphBank.load(str(path), learnable=False)
    for p in bank.parameters():
        p.requires_grad_(False)
    bank.eval()
    return bank


def alphabet_for_bank(bank):
    from .alphabet import N_CHARS, Alphabet
    syms = getattr(bank, "symbols", None)
    if syms is None:
        return Alphabet.build(int(bank.n_symbols) - N_CHARS)
    a = Alphabet.build(len(syms) - N_CHARS)
    return a if a.symbols == list(syms) else Alphabet(list(syms)[N_CHARS:])


# ------------------------------------------------------------------------------ training
def train_reader_on_bank(bank_path, cfg: TransferConfig | None = None, log=print,
                         train_lines=None, eval_lines=None, save: bool = True) -> list[dict]:
    """Fresh reader on a frozen bank. Returns [row]; with ``save`` appends it to
    ``<out_dir>/runs.jsonl`` and writes curves/, checkpoints/ (reader, git-ignored), samples/ files
    (tag = transfer_<bank stem>).

    ``train_lines`` / ``eval_lines=(val, test)`` override the corpus (tests / smoke)."""
    import torch
    from torch import nn
    from vtc.crnn import data as D
    from vtc.crnn.model import CRNN, n_params

    from . import train as T
    from .alphabet import Alphabet

    cfg = cfg or TransferConfig()
    bank_path = resolve_bank_path(bank_path or cfg.bank_path)
    cfg.bank_path = str(bank_path)
    torch.set_num_threads(cfg.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    _, compose, Channel = T.lg1()

    bank = load_frozen_bank(bank_path)
    alphabet = alphabet_for_bank(bank)
    if len(alphabet) != int(bank.n_symbols):
        raise ValueError(f"alphabet {len(alphabet)} != bank {bank.n_symbols} symbols")
    src = source_row(bank_path)
    scheme = cfg.scheme or bank_scheme_name(bank, alphabet.n_tokens, src)
    ss = int(bank.ss)
    height_px = int(bank.height_px)
    em = float(getattr(bank, "em", height_px / 1.2))
    tag = f"transfer_{bank_path.stem}" + (f"_{cfg.channel}" if cfg.channel != "default" and
                                          cfg.channel not in bank_path.stem else "")
    out_dir = cfg.out_dir()
    rcfg = _RenderCfg(ss=ss, pad_px=cfg.pad_px)          # what lg.train.render needs

    channel = Channel(cfg.channel, ss=ss)
    widths_ss_t = torch.as_tensor(bank.widths_ss, dtype=torch.long)
    wss_np = widths_ss_t.numpy().astype(np.int64)
    with torch.no_grad():
        ink = bank(tau=1.0, hard=cfg.hard).detach()        # frozen: computed once

    # ---- data (identical to LG2 / Phase 2)
    val_t, test_t = eval_lines if eval_lines is not None else T.load_eval_lines(
        _EvalCfg(cfg.n_val, cfg.n_test, cfg.data_seed))
    val_ids = [alphabet.encode(t) for t in val_t]
    test_ids = [alphabet.encode(t) for t in test_t]
    stream = T.TrainStream(alphabet, wss_np, cfg.batch, cfg.bucket_mult, seed=cfg.seed,
                           data_seed=cfg.data_seed, lines=train_lines)
    if train_lines is None:
        from itertools import islice
        from vtc.corpus import iter_lines
        probe = [normalize(t) for t in islice(iter_lines("train", 30, 70, seed=cfg.data_seed),
                                              cfg.n_expand_probe)]
    else:
        probe = list(train_lines[: cfg.n_expand_probe])
    probe_ids = [alphabet.encode(t) for t in probe]

    # ---- fresh reader, Phase-2 geometry rule
    pad = T.channel_padding(channel, int(bank.height_ss), ss) + 2 * cfg.pad_px
    all_ids = val_ids + test_ids + probe_ids
    w_px = np.array([int(wss_np[i].sum()) // ss + pad for i in all_ids])
    L = np.array([len(i) for i in all_ids])
    stride = D.col_stride_for_em(em)
    expand = D.choose_expand(w_px, L, stride)
    reader = CRNN(alphabet.n_classes, col_stride=stride, expand=expand, **cfg.model)
    opt = torch.optim.AdamW(reader.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps,
                                                pct_start=cfg.pct_start)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    curve_f = None
    if save:
        for sub in ("curves", "checkpoints", "samples"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        curve_f = open(out_dir / "curves" / f"{tag}.jsonl", "w")
    log(f"[{tag}] bank={bank_path.name} scheme={scheme} symbols={len(alphabet)} h={height_px} "
        f"col_stride={stride} expand={expand} pad={pad} reader_params={n_params(reader)} "
        f"channel={cfg.channel}")

    best = {"cer": math.inf, "step": 0, "reader": None}
    loss_acc = n_acc = skipped = 0
    t0, c0 = time.time(), time.process_time()
    eval_time = 0.0
    reader.train()
    for step in range(1, cfg.steps + 1):
        _, ids_b = next(stream)
        x, wpx = T.render(ink, ids_b, widths_ss_t, compose, channel, rcfg, gen, train=True)
        lp, lens = reader(x, wpx)
        tg, tl = T._ctc_targets(ids_b)
        skipped += int((lens < 2 * tl + 1).sum())
        loss = ctc(lp, tg, lens, tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(reader.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        loss_acc += loss.item()
        n_acc += 1
        if step % cfg.eval_every == 0 or step == cfg.steps:
            te = time.time()
            ev = T.evaluate(ink, reader, alphabet, val_t, val_ids, widths_ss_t, compose, channel, rcfg)
            eval_time += time.time() - te
            pt = {"step": step, "train_loss": round(loss_acc / n_acc, 4), "val_cer": round(ev["cer"], 5),
                  "val_symbol_cer": round(ev["symbol_cer"], 5), "lr": sched.get_last_lr()[0],
                  "ctc_infeasible": skipped, "elapsed_s": round(time.time() - t0, 1)}
            if curve_f:
                curve_f.write(json.dumps(pt) + "\n")
                curve_f.flush()
            log(f"[{tag}] step {step} loss {pt['train_loss']:.3f} val_cer {ev['cer']:.4f} "
                f"({pt['elapsed_s']}s)")
            loss_acc = n_acc = 0
            if ev["cer"] < best["cer"]:
                best = {"cer": ev["cer"], "step": step, "reader": copy.deepcopy(reader.state_dict())}
    train_time = time.time() - t0
    train_cpu = time.process_time() - c0
    if curve_f:
        curve_f.close()
    reader.load_state_dict(best["reader"])

    ev = T.evaluate(ink, reader, alphabet, test_t, test_ids, widths_ss_t, compose, channel, rcfg,
                    seed=54321, return_preds=True)
    ppc = [height_px * (wss_np[i].sum() / ss) / max(1, len(normalize(t)))
           for t, i in zip(test_t, test_ids)]
    n_sym = sum(len(i) for i in test_ids)
    n_chr = sum(len(normalize(t)) for t in test_t)
    from .scheme import bank_sha8
    row = {
        "phase": "lg", "run": cfg.run_name, "train_mode": "transfer", "scheme": scheme,
        "em": height_px / 1.2, "height_px": height_px, "px_per_char": float(np.mean(ppc)),
        "cer": ev["cer"], "symbol_cer": ev["symbol_cer"], "steps": cfg.steps,
        "channel": cfg.channel, "n_tokens": alphabet.n_tokens,
        "learn_glyphs": bool(src.get("learn_glyphs")) if src else getattr(bank, "logits", None) is not None,
        "train_time_s": round(train_time, 1),
        # diagnostics
        "tag": tag, "source_tag": bank_path.stem, "bank_path": _rel(bank_path),
        "bank_sha8": bank_sha8(bank_path), "source_run": src.get("run") if src else None,
        "source_cer": src.get("cer") if src else None,
        "source_channel": src.get("channel") if src else None,
        "best_step": best["step"], "best_val_cer": best["cer"], "n_params": n_params(reader),
        "n_symbols": len(alphabet), "col_stride": stride, "expand": expand, "channel_pad_px": pad,
        "seg_ratio": n_sym / max(1, n_chr), "mean_ink": float(ink.sum()) / float(bank.height_ss * wss_np.sum()),
        "s_per_step": round((train_time - eval_time) / cfg.steps, 4),
        "train_cpu_s": round(train_cpu, 1), "eval_time_s": round(eval_time, 1),
        "ctc_infeasible": skipped, "n_val": len(val_t), "n_test": len(test_t), "threads": cfg.threads,
        "seed": cfg.seed, "config": asdict(cfg),
    }
    log(f"[{tag}] TEST cer={row['cer']:.4f} symbol_cer={row['symbol_cer']:.4f} "
        f"px/char={row['px_per_char']:.2f} best_step={best['step']} time={train_time:.0f}s"
        + (f" (joint cer {src['cer']:.4f})" if src and src.get("cer") is not None else ""))
    if save:
        torch.save({"state": best["reader"], "n_classes": alphabet.n_classes, "col_stride": stride,
                    "expand": expand, "model": cfg.model, "config": asdict(cfg)},
                   out_dir / "checkpoints" / f"{tag}.pt")
        with open(out_dir / "samples" / f"{tag}.txt", "w") as fh:
            for i in range(min(cfg.n_samples, len(test_t))):
                fh.write(f"REF {test_t[i]}\nHYP {ev['preds'][i]}\n\n")
        with open(out_dir / "runs.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return [row]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(LG_ROOT.parent))
    except ValueError:
        return str(p)


@dataclass
class _RenderCfg:
    ss: int = 4
    pad_px: int = 0


@dataclass
class _EvalCfg:
    n_val: int
    n_test: int
    data_seed: int


def run_job(args: tuple[str, dict]) -> list[dict]:
    """Process-pool entry point: (bank_path, TransferConfig dict) -> rows; logs to
    <out_dir>/logs/<tag>.log."""
    bank_path, cfg_dict = args
    cfg = TransferConfig.from_dict(cfg_dict)
    p = resolve_bank_path(bank_path)
    log_path = cfg.out_dir() / "logs" / f"transfer_{p.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        def log(msg: str) -> None:
            fh.write(msg + "\n")
            fh.flush()
            print(msg, flush=True)
        return train_reader_on_bank(p, cfg, log=log)


# ------------------------------------------------------------------------------ bridge (E3)
def bridge_phase2(bank_path, em: float, *, steps: int = 3500, run_name: str = "bridge",
                  out_root: str | Path | None = None, n_train: int = 40_000, n_val: int = 300,
                  n_test: int = 1000, batch: int = 32, eval_every: int = 250, threads: int = 1,
                  seed: int = 0, render_workers: int = 1, allow_fallback: bool = False,
                  cache_dir: str | Path | None = None, log=print) -> list[dict]:
    """E3: score a bank with the UNCHANGED Phase-2 pipeline (vtc.crnn.data rendering via
    vtc.layout + vtc.crnn.train.train_model) at layout em ``em``.

    The bank is registered as scheme ``lg:<key>@<sha8>`` via the runtime hook (no parent edit).
    Phase-2 data are built exactly as ``vtc.crnn.train.load_data`` does, except that the render
    cache lives in ``<out>/cache`` (git-ignored). Writes the raw Phase-2 row (phase=2) to
    ``<out>/phase2/runs.jsonl`` and a PLAN-schema row (phase="lg", train_mode="bridge",
    scheme = the bank's lg scheme name) to ``<out>/runs.jsonl``. Returns [lg row].
    """
    from vtc.crnn import data as D
    from vtc.crnn.train import TrainConfig, train_model

    from .scheme import bank_sha8, install_phase2_hook, make_lg_scheme, scheme_name_for

    install_phase2_hook()
    bank_path = resolve_bank_path(bank_path)
    name = scheme_name_for(bank_path)
    sch = make_lg_scheme(name)
    out = Path(out_root or RESULTS) / run_name
    cache = Path(cache_dir) if cache_dir else out / "cache"
    cfg = TrainConfig(scheme=name, em=float(em), steps=steps, batch=batch, eval_every=eval_every,
                      n_train=n_train, n_val=n_val, n_test=n_test, threads=threads, seed=seed,
                      render_workers=render_workers, allow_fallback=allow_fallback,
                      render={"supersample": 4})
    kw = dict(seed=cfg.data_seed, rp=cfg.render_params(), workers=render_workers,
              allow_fallback=allow_fallback, cache_dir=cache)
    train = D.build_lineset(name, cfg.em, "train", n_train, train=True, tag="train", **kw)
    test = D.build_lineset(name, cfg.em, "val", n_test, tag="test", **kw)
    val = D.build_lineset(name, cfg.em, "val", n_val, skip=n_test, tag="val", **kw)
    p2 = train_model(cfg, out / "phase2", data=(train, {cfg.em: val}, {cfg.em: test}), log=log)[0]
    bank = load_frozen_bank(bank_path)
    src = source_row(bank_path)
    row = {
        "phase": "lg", "run": run_name, "train_mode": "bridge",
        "scheme": bank_scheme_name(bank, sch.alphabet.n_tokens, src), "em": float(em),
        "height_px": 1.2 * float(em), "px_per_char": p2["px_per_char"], "cer": p2["cer"],
        "symbol_cer": p2["symbol_cer"], "steps": steps, "channel": "phase2",
        "n_tokens": sch.alphabet.n_tokens,
        "learn_glyphs": bool(src.get("learn_glyphs")) if src else getattr(bank, "logits", None) is not None,
        "train_time_s": p2["train_time_s"],
        "phase2_scheme": name, "bank_path": _rel(bank_path), "bank_sha8": bank_sha8(bank_path),
        "source_tag": bank_path.stem, "bank_native_em": sch.native_em, "bank_height_px": sch.height_px,
        "best_step": p2["best_step"], "n_params": p2["n_params"], "vocab_size": p2["vocab_size"],
        "expand": p2["expand"], "corpus": p2["corpus"], "n_test": p2["n_test"], "seed": seed,
    }
    with open(out / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    log(f"[bridge {bank_path.stem} em={em:g}] cer={row['cer']:.4f} px/char={row['px_per_char']:.2f}")
    return [row]


def bridge_job(args: tuple[str, float, dict]) -> list[dict]:
    """Process-pool entry point for ``bridge_phase2`` (logs to <out>/logs/bridge_<stem>_em<em>.log)."""
    bank_path, em, kw = args
    out = Path(kw.get("out_root") or RESULTS) / kw.get("run_name", "bridge")
    log_path = out / "logs" / f"bridge_{Path(bank_path).stem}_em{em:g}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        def log(msg: str) -> None:
            fh.write(msg + "\n")
            fh.flush()
            print(msg, flush=True)
        return bridge_phase2(bank_path, em, log=log, **kw)
