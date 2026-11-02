"""Train one CRNN from scratch for one (scheme, em) and write a Phase-2 row.

Row keys (same as Phase 1 where they overlap, consumed by ``vbpe/analysis.py``):
  phase=2, scheme, em, px_per_char (mean over test lines), cer (micro: total edits / total
  chars, on scheme.decode(prediction) vs source), symbol_cer (micro, on label sequences),
  steps, train_time_s, n_params  (+ diagnostics: best_step, best_val_cer, frame_factor, ...).
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from vbpe import config
from vbpe.crnn import data as D
from vbpe.metrics import normalize


@dataclass
class TrainConfig:
    scheme: str
    em: float
    steps: int = 3000
    batch: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-2
    pct_start: float = 0.15
    grad_clip: float = 5.0
    eval_every: int = 250
    n_train: int = 40_000
    n_val: int = 300
    n_test: int = 1000
    threads: int = 1
    seed: int = 0
    data_seed: int = 0
    render_workers: int = 2
    allow_fallback: bool = False
    model: dict = field(default_factory=dict)       # CRNN kwargs override
    render: dict = field(default_factory=dict)      # RenderParams override

    def render_params(self) -> D.RenderParams:
        return D.RenderParams(**self.render)


def load_data(cfg: TrainConfig, workers: int | None = None):
    """(train, val, test) LineSets; test = first n_test 'val'-split lines, val = the next n_val."""
    kw = dict(seed=cfg.data_seed, rp=cfg.render_params(), workers=workers or cfg.render_workers,
              allow_fallback=cfg.allow_fallback)
    train = D.build_lineset(cfg.scheme, cfg.em, "train", cfg.n_train, train=True, tag="train", **kw)
    test = D.build_lineset(cfg.scheme, cfg.em, "val", cfg.n_test, tag="test", **kw)
    val = D.build_lineset(cfg.scheme, cfg.em, "val", cfg.n_val, skip=cfg.n_test, tag="val", **kw)
    return train, val, test


def evaluate(model, ls: D.LineSet, vocab: D.Vocab, scheme, batch: int = 64,
             return_preds: bool = False) -> dict:
    import torch
    from vbpe.crnn.model import greedy_decode

    model.eval()
    edits = chars = sym_edits = sym_n = 0
    preds = [None] * len(ls)
    with torch.no_grad():
        for idx in D.eval_batches(ls.widths, batch):
            x, w = D.collate(ls, idx)
            lp, lens = model(x, w)
            for i, seq in zip(idx, greedy_decode(lp, lens)):
                labs = vocab.decode(seq)
                hyp = normalize(scheme.decode("".join(labs)))
                ref = normalize(ls.texts[int(i)])
                edits += Levenshtein.distance(ref, hyp)
                chars += len(ref)
                sym_edits += Levenshtein.distance(list(ls.labels[int(i)]), labs)
                sym_n += len(ls.labels[int(i)])
                if return_preds:
                    preds[int(i)] = hyp
    model.train()
    out = {"cer": edits / max(1, chars), "symbol_cer": sym_edits / max(1, sym_n), "n": len(ls)}
    if return_preds:
        out["preds"] = preds
    return out


def train_run(cfg: TrainConfig, out_dir: Path | None = None, data=None, log=print) -> dict:
    """Train from scratch, select best checkpoint by val CER, evaluate on test, append row."""
    import torch
    from torch import nn
    from vbpe.crnn.model import CRNN, n_params

    torch.set_num_threads(cfg.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    train, val, test = data if data is not None else load_data(cfg)
    scheme = D.resolve_scheme(cfg.scheme)
    vocab = D.Vocab.from_labels(train.labels)
    stride = D.col_stride_for_em(cfg.em)
    expand = D.choose_expand(np.concatenate([s.widths for s in (train, val, test)]),
                             np.concatenate([s.label_lens for s in (train, val, test)]), stride)
    ff = expand / stride                               # CTC frames per input px
    model = CRNN(len(vocab), col_stride=stride, expand=expand, **cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps,
                                                pct_start=cfg.pct_start)
    ctc = nn.CTCLoss(blank=D.Vocab.BLANK, zero_infinity=True)
    tag = f"{cfg.scheme}_em{cfg.em:g}"
    curve_f = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "curves").mkdir(exist_ok=True)
        curve_f = open(out_dir / "curves" / f"{tag}.jsonl", "w")
    log(f"[{tag}] vocab={len(vocab)} col_stride={stride} expand={expand} params={n_params(model)} "
        f"train={len(train)} H={train.height} meanW={train.widths.mean():.0f}")

    batches = D.bucket_batches(train.widths, cfg.batch, rng)
    best = {"cer": math.inf, "step": 0, "state": None}
    curve, loss_acc, n_acc = [], 0.0, 0
    t0 = time.time()
    eval_time = 0.0
    model.train()
    for step in range(1, cfg.steps + 1):
        x, w, tg, tl = D.collate(train, next(batches), vocab)
        x = D.augment_batch(x, gen)
        lp, lens = model(x, w)
        loss = ctc(lp, tg, lens, tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        loss_acc += loss.item()
        n_acc += 1
        if step % cfg.eval_every == 0 or step == cfg.steps:
            te = time.time()
            ev = evaluate(model, val, vocab, scheme)
            eval_time += time.time() - te
            pt = {"step": step, "train_loss": round(loss_acc / n_acc, 4), "val_cer": round(ev["cer"], 5),
                  "val_symbol_cer": round(ev["symbol_cer"], 5), "lr": sched.get_last_lr()[0],
                  "elapsed_s": round(time.time() - t0, 1)}
            curve.append(pt)
            if curve_f:
                curve_f.write(json.dumps(pt) + "\n")
                curve_f.flush()
            log(f"[{tag}] step {step} loss {pt['train_loss']:.3f} val_cer {pt['val_cer']:.4f} "
                f"({pt['elapsed_s']}s)")
            loss_acc, n_acc = 0.0, 0
            if ev["cer"] < best["cer"]:
                best = {"cer": ev["cer"], "step": step, "state": copy.deepcopy(model.state_dict())}
    train_time = time.time() - t0
    if curve_f:
        curve_f.close()
    model.load_state_dict(best["state"])
    te = evaluate(model, test, vocab, scheme, return_preds=True)
    row = {
        "phase": 2, "scheme": cfg.scheme, "em": float(cfg.em),
        "px_per_char": float(np.mean(test.ppc)), "cer": te["cer"], "symbol_cer": te["symbol_cer"],
        "steps": cfg.steps, "train_time_s": round(train_time, 1), "n_params": n_params(model),
        "best_step": best["step"], "best_val_cer": best["cer"], "frame_factor": ff,
        "col_stride": stride, "expand": expand,
        "vocab_size": len(vocab), "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "batch": cfg.batch, "threads": cfg.threads, "s_per_step": round((train_time - eval_time) / cfg.steps, 4),
        "render_time_s": sum(s.info.get("render_time_s", 0) for s in (train, val, test)),
        "corpus": train.info.get("corpus"), "render": train.info.get("render"),
        "input_height": train.height, "seed": cfg.seed, "config": asdict(cfg),
    }
    if out_dir is not None:
        ck = out_dir / "checkpoints"
        ck.mkdir(exist_ok=True)
        torch.save({"state": best["state"], "vocab": vocab.to_json(), "col_stride": stride, "expand": expand,
                    "config": asdict(cfg)}, ck / f"{tag}.pt")
        (out_dir / "samples").mkdir(exist_ok=True)
        with open(out_dir / "samples" / f"{tag}.txt", "w") as fh:
            for i in range(min(20, len(test))):
                fh.write(f"REF {test.texts[i]}\nHYP {te['preds'][i]}\n\n")
        with open(out_dir / "runs.jsonl", "a") as fh:      # single small write: atomic append
            fh.write(json.dumps(row) + "\n")
    log(f"[{tag}] TEST cer={row['cer']:.4f} symbol_cer={row['symbol_cer']:.4f} "
        f"px/char={row['px_per_char']:.1f} best_step={best['step']} time={train_time:.0f}s")
    return row


def default_out(run_name: str) -> Path:
    return config.RESULTS / "phase2" / run_name
