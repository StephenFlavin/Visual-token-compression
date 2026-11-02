"""Train CRNNs from scratch and write Phase-2 rows.

Two modes (same model, optimiser, augmentation and decoding):
  * per-em  (``TrainConfig.mixed_ems is None``): one model per (scheme, em); one row.
  * mixed   (``mixed_ems=[4, 5, ...]``): ONE model per scheme trained on lines whose em is drawn
    uniformly from the list; checkpoint selected on the MEAN val CER over the ems; the same test
    lines are rendered at each em -> one row per (scheme, em) with ``train_mode="mixed"``.

Row keys (same as Phase 1 where they overlap, consumed by ``vbpe/analysis.py``):
  phase=2, scheme, em, px_per_char (mean over test lines), cer (micro: total edits / total
  chars, on scheme.decode(prediction) vs source), symbol_cer (micro, on label sequences),
  steps, train_time_s, n_params  (+ diagnostics: train_mode, best_step, best_val_cer, ...).
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
    em: float | None = None                 # per-em mode
    mixed_ems: list[float] | None = None    # mixed mode (em ignored)
    steps: int = 3500
    batch: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-2
    pct_start: float = 0.15
    grad_clip: float = 5.0
    eval_every: int = 250
    n_train: int = 40_000
    n_val: int = 300                        # per em
    n_test: int = 1000                      # per em
    threads: int = 1
    seed: int = 0
    data_seed: int = 0
    render_workers: int = 2
    allow_fallback: bool = False
    model: dict = field(default_factory=dict)       # CRNN kwargs override
    render: dict = field(default_factory=dict)      # RenderParams override

    def render_params(self) -> D.RenderParams:
        return D.RenderParams(**self.render)

    @property
    def ems(self) -> list[float]:
        return [float(e) for e in self.mixed_ems] if self.mixed_ems else [float(self.em)]

    @property
    def tag(self) -> str:
        return f"{self.scheme}_{D._em_name(tuple(self.mixed_ems))}" if self.mixed_ems \
            else f"{self.scheme}_em{self.em:g}"


def load_data(cfg: TrainConfig, workers: int | None = None):
    """-> (train, {em: val}, {em: test}). test = first n_test 'val'-split lines, val = the next
    n_val, rendered at each em; train = n_train 'train'-split lines (em drawn per line if mixed)."""
    kw = dict(seed=cfg.data_seed, rp=cfg.render_params(), workers=workers or cfg.render_workers,
              allow_fallback=cfg.allow_fallback)
    train_em = tuple(cfg.ems) if cfg.mixed_ems else cfg.em
    train = D.build_lineset(cfg.scheme, train_em, "train", cfg.n_train, train=True, tag="train", **kw)
    tests = {em: D.build_lineset(cfg.scheme, em, "val", cfg.n_test, tag="test", **kw) for em in cfg.ems}
    vals = {em: D.build_lineset(cfg.scheme, em, "val", cfg.n_val, skip=cfg.n_test, tag="val", **kw)
            for em in cfg.ems}
    return train, vals, tests


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


def _as_dict(sets, ems) -> dict:
    return sets if isinstance(sets, dict) else {ems[0]: sets}


def train_model(cfg: TrainConfig, out_dir: Path | None = None, data=None, log=print) -> list[dict]:
    """Train from scratch, select best checkpoint by (mean) val CER, evaluate on each em's test
    set, append one row per em to ``out_dir/runs.jsonl``. Returns the rows."""
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

    train, vals, tests = data if data is not None else load_data(cfg)
    ems = cfg.ems
    vals, tests = _as_dict(vals, ems), _as_dict(tests, ems)
    scheme = D.resolve_scheme(cfg.scheme)
    vocab = D.Vocab.from_labels(train.labels)
    stride = D.col_stride_for_em(min(ems))
    all_sets = [train, *vals.values(), *tests.values()]
    expand = D.choose_expand(np.concatenate([s.widths for s in all_sets]),
                             np.concatenate([s.label_lens for s in all_sets]), stride)
    model = CRNN(len(vocab), col_stride=stride, expand=expand, **cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps,
                                                pct_start=cfg.pct_start)
    ctc = nn.CTCLoss(blank=D.Vocab.BLANK, zero_infinity=True)
    tag = cfg.tag
    curve_f = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "curves").mkdir(exist_ok=True)
        curve_f = open(out_dir / "curves" / f"{tag}.jsonl", "w")
    log(f"[{tag}] vocab={len(vocab)} col_stride={stride} expand={expand} params={n_params(model)} "
        f"train={len(train)} H={train.height} meanW={train.widths.mean():.0f}")

    batches = D.bucket_batches(train.widths, cfg.batch, rng)
    best = {"cer": math.inf, "step": 0, "state": None}
    loss_acc, n_acc = 0.0, 0
    t0, c0 = time.time(), time.process_time()
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
            evs = {em: evaluate(model, vals[em], vocab, scheme) for em in ems}
            mean_cer = float(np.mean([e["cer"] for e in evs.values()]))
            eval_time += time.time() - te
            pt = {"step": step, "train_loss": round(loss_acc / n_acc, 4), "val_cer": round(mean_cer, 5),
                  "val_cer_by_em": {f"{em:g}": round(e["cer"], 5) for em, e in evs.items()},
                  "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
            if curve_f:
                curve_f.write(json.dumps(pt) + "\n")
                curve_f.flush()
            by_em = " ".join(f"{k}:{v:.4f}" for k, v in pt["val_cer_by_em"].items()) if len(ems) > 1 else ""
            log(f"[{tag}] step {step} loss {pt['train_loss']:.3f} val_cer {mean_cer:.4f} {by_em} "
                f"({pt['elapsed_s']}s)")
            loss_acc, n_acc = 0.0, 0
            if mean_cer < best["cer"]:
                best = {"cer": mean_cer, "step": step, "state": copy.deepcopy(model.state_dict())}
    train_time = time.time() - t0
    train_cpu = time.process_time() - c0
    if curve_f:
        curve_f.close()
    model.load_state_dict(best["state"])

    common = {
        "phase": 2, "run": out_dir.name if out_dir is not None else None,
        "train_mode": "mixed" if cfg.mixed_ems else "per_em", "scheme": cfg.scheme,
        "steps": cfg.steps, "train_time_s": round(train_time, 1), "n_params": n_params(model),
        "best_step": best["step"], "best_val_cer": best["cer"], "frame_factor": expand / stride,
        "col_stride": stride, "expand": expand, "vocab_size": len(vocab), "n_train": len(train),
        "batch": cfg.batch, "threads": cfg.threads,
        "s_per_step": round((train_time - eval_time) / cfg.steps, 4), "train_cpu_s": round(train_cpu, 1),
        "render_time_s": round(sum(s.info.get("render_time_s", 0) for s in all_sets), 1),
        "corpus": train.info.get("corpus"), "render": train.info.get("render"),
        "mixed_ems": cfg.mixed_ems, "seed": cfg.seed, "config": asdict(cfg),
    }
    rows = []
    for em in ems:
        test = tests[em]
        ev = evaluate(model, test, vocab, scheme, return_preds=True)
        row = {**common, "em": float(em), "px_per_char": float(np.mean(test.ppc)), "cer": ev["cer"],
               "symbol_cer": ev["symbol_cer"], "n_val": len(vals[em]), "n_test": len(test),
               "input_height": int(test.heights.max())}
        # keep the Phase-1-shared keys first for readability
        row = {k: row[k] for k in ("phase", "run", "train_mode", "scheme", "em", "px_per_char", "cer",
                                   "symbol_cer", "steps", "train_time_s", "n_params")} | row
        rows.append(row)
        log(f"[{tag}] TEST em={em:g} cer={row['cer']:.4f} symbol_cer={row['symbol_cer']:.4f} "
            f"px/char={row['px_per_char']:.1f} best_step={best['step']} time={train_time:.0f}s")
        if out_dir is not None:
            (out_dir / "samples").mkdir(exist_ok=True)
            with open(out_dir / "samples" / f"{cfg.scheme}_em{em:g}.txt", "w") as fh:
                for i in range(min(20, len(test))):
                    fh.write(f"REF {test.texts[i]}\nHYP {ev['preds'][i]}\n\n")
    if out_dir is not None:
        ck = out_dir / "checkpoints"
        ck.mkdir(exist_ok=True)
        torch.save({"state": best["state"], "vocab": vocab.to_json(), "col_stride": stride,
                    "expand": expand, "config": asdict(cfg)}, ck / f"{tag}.pt")
        with open(out_dir / "runs.jsonl", "a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    return rows


def train_run(cfg: TrainConfig, out_dir: Path | None = None, data=None, log=print) -> dict:
    """Per-em mode: train one model for (scheme, em) and return its single row."""
    assert not cfg.mixed_ems, "use train_model for mixed-em runs"
    return train_model(cfg, out_dir, data, log)[0]


def default_out(run_name: str) -> Path:
    return config.RESULTS / "phase2" / run_name
