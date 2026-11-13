"""E0: calibrate the learned-glyph Channel against Phase-2 PIL rendering (WP-LG1).

For each em (Phase-2 ems 4..8 -> pitch h = round(1.2 em)) a FIXED DejaVu bank
(``GlyphBank.from_font(..., em=em)``) composes real 'val' lines (the first n lines = the Phase-2
test lines) which go through each Channel preset, and are compared with the Phase-2 rendering of
the same lines (``vtc.crnn.data.render_sample`` with the Phase-2 ``RenderParams``):

  stats   pixel statistics per (em, source): mean ink, ink histogram, peak ink (p99), edge sharpness
          (mean |dx|+|dy|), background noise; pixel-aligned relative MAD (phase forced to 0 on both
          sides, eval degradation) on the drift-free first 12 chars and on the full line, with the
          PIL-vs-PIL quarter-pixel-phase MAD as yardstick. Sources: pil_eval (Phase-2 test images),
          pil_train (Phase-2 train rendering + augment_batch), <preset>_eval, <preset>_train.
  strips  side-by-side PNGs (PIL vs each preset, eval and train mode), nearest-upscaled.
  --train train a vtc CRNN reader (same model/optimiser/schedule/eval/checkpoint rule as
          vtc.crnn.train.train_model) on bank+channel lines per (em, preset) and report test CER
          (+ P@3% per preset vs Phase-2 latin from results/phase2/phase2_main/runs.jsonl).
          ``--pil-ref`` also trains the unchanged Phase-2 trainer for the same steps (use when
          --steps differs from Phase 2's 3500).

Usage
-----
  quick smoke (~1-2 CPU-min):   python3 learned_glyphs/experiments/calibrate_channel.py --quick
  full E0 (stats + 15 runs):    nohup python3 learned_glyphs/experiments/calibrate_channel.py \
                                    --train --parallel 3 > learned_glyphs/results/calibration/e0.log 2>&1 &
Outputs go to learned_glyphs/results/calibration/ (quick: results/calibration_quick/).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vtc import config  # noqa: E402
from vtc.crnn import data as D  # noqa: E402

LG = ROOT / "learned_glyphs"
OUT = LG / "results" / "calibration"
PHASE2_RUNS = config.RESULTS / "phase2" / "phase2_main" / "runs.jsonl"
SYMBOLS = [chr(c) for c in range(32, 127)]
S2I = {s: i for i, s in enumerate(SYMBOLS)}
HIST_EDGES = [0.0, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98, 1.0001]
PRESETS = ["clean", "default", "harsh"]
STAT_PRESETS = PRESETS + ["harsh+heldout"]      # test-only channel: eval stats only
SCHEME = "lg_fixed_n0"


def _torch(threads: int = 1):
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch


def height_for_em(em: float) -> int:
    return int(round(1.2 * em))


def encode(line: str) -> list[int]:
    return [S2I[c] for c in line if c in S2I]


def make_bank(em: float):
    from learned_glyphs.lg.glyphs import GlyphBank
    return GlyphBank.from_font(SYMBOLS, config.LATIN_FONT, height_for_em(em), ss=4, em=em)


def bank_lines(bank, channel, lines, *, train: bool, gen=None, phase=None):
    """-> (x (B,1,H,W) float tensor, widths_px (B,)) for a list of text lines."""
    from learned_glyphs.lg.compose import compose
    import torch
    with torch.no_grad():
        cv, w = compose(bank(), [encode(t) for t in lines], bank.widths_ss)
        return channel(cv, w, generator=gen, train=train, phase=phase)


def pil_lines(lines, em, *, train: bool, rp=D.RenderParams(), seed: int = 0, tag: str = "test",
              augment_gen=None):
    """Phase-2 rendering of the same lines -> (x (B,1,H,W) float32 numpy, widths)."""
    scheme = D.resolve_scheme("latin")
    imgs = [D.render_sample(scheme, t, em, rp, D.line_rng(seed, tag, i), train)[0]
            for i, t in enumerate(lines)]
    H = max(a.shape[0] for a in imgs)
    W = max(a.shape[1] for a in imgs)
    x = np.zeros((len(imgs), 1, H, W), np.float32)
    for b, a in enumerate(imgs):
        x[b, 0, :a.shape[0], :a.shape[1]] = (255 - a.astype(np.float32)) / 255.0
    widths = np.array([a.shape[1] for a in imgs])
    if augment_gen is not None:
        import torch
        x = D.augment_batch(torch.from_numpy(x), augment_gen).numpy()
    return x, widths


# ------------------------------------------------------------------------------------- stats
def pixel_stats(x, widths) -> dict:
    x = np.asarray(x, np.float32)[:, 0]
    widths = np.asarray(widths)
    B, H, W = x.shape
    valid = np.arange(W)[None, :] < widths[:, None]                 # (B, W)
    m = np.broadcast_to(valid[:, None, :], x.shape)
    vals = x[m]
    hist = np.histogram(vals, bins=HIST_EDGES)[0] / max(1, vals.size)
    gx = np.abs(np.diff(x, axis=2))[np.broadcast_to((valid[:, 1:] & valid[:, :-1])[:, None, :], (B, H, W - 1))]
    gy = np.abs(np.diff(x, axis=1))[np.broadcast_to(valid[:, None, :], (B, H - 1, W))]
    sharp = float(gx.mean() + gy.mean())
    ink = float(vals.mean())
    return {"mean_ink": ink, "p99_ink": float(np.quantile(vals, 0.99)),
            "frac_full_ink": float((vals > 0.9).mean()), "sharpness": sharp,
            "sharpness_per_ink": sharp / max(ink, 1e-9), "bg_std": float(x[:, 0, :][valid].std()),
            "hist": [round(float(h), 5) for h in hist]}


def stat_distance(s: dict, ref: dict) -> float:
    """hist L1 + relative |d mean ink| + relative |d sharpness| (0 = identical)."""
    return (float(np.abs(np.array(s["hist"]) - np.array(ref["hist"])).sum())
            + abs(s["mean_ink"] - ref["mean_ink"]) / ref["mean_ink"]
            + abs(s["sharpness"] - ref["sharpness"]) / ref["sharpness"])


def aligned_mad(bank, channel, lines, em, prefix_chars: int = 12) -> dict:
    """Relative MAD (sum|a-b| / sum a) vs the PIL phase-0 rendering, both at phase 0 (channel in
    eval mode); 'prefix' = first ``prefix_chars`` chars (free of cumulative advance-rounding drift)."""
    from vtc.layout import render_line
    rp = D.RenderParams(phase=False)
    scheme = D.resolve_scheme("latin")
    xb, wb = bank_lines(bank, channel, lines, train=False, phase=0)
    xb = xb.numpy()[:, 0]
    full, pre, q_full, q_pre = [], [], [], []
    for i, t in enumerate(lines):
        units = scheme.encode(t)
        a = (255 - np.asarray(render_line(units, em, rp.layout()), np.float32)) / 255
        q = (255 - np.asarray(render_line([D.PhaseOffset(0.25, em)] + units, em, rp.layout()),
                              np.float32)) / 255
        b = xb[i]
        H = min(a.shape[0], b.shape[0])
        W = min(a.shape[1], int(wb[i]), q.shape[1])
        Wp = min(W, rp.margin + int(math.ceil(sum(int(bank.widths_ss[S2I[c]]) for c in t[:prefix_chars]) / bank.ss)))
        den, denp = a[:H, :W].sum(), a[:H, :Wp].sum()
        full.append(np.abs(a[:H, :W] - b[:H, :W]).sum() / den)
        pre.append(np.abs(a[:H, :Wp] - b[:H, :Wp]).sum() / denp)
        q_full.append(np.abs(a[:H, :W] - q[:H, :W]).sum() / den)
        q_pre.append(np.abs(a[:H, :Wp] - q[:H, :Wp]).sum() / denp)
    return {"rel_mad_prefix": float(np.mean(pre)), "rel_mad_full": float(np.mean(full)),
            "pil_quarter_phase_rel_mad_prefix": float(np.mean(q_pre)),
            "pil_quarter_phase_rel_mad_full": float(np.mean(q_full))}


def save_strip(path: Path, rows: list[tuple[str, np.ndarray]], zoom: int = 4) -> None:
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(config.LATIN_FONT), 12)
    lab_w = 120
    W = lab_w + max(a.shape[1] for _, a in rows) * zoom
    Ht = sum(a.shape[0] * zoom + 4 for _, a in rows)
    img = Image.new("L", (W, Ht), 255)
    d = ImageDraw.Draw(img)
    y = 0
    for lab, a in rows:
        im = Image.fromarray((255 - np.clip(a, 0, 1) * 255).round().astype(np.uint8))
        im = im.resize((a.shape[1] * zoom, a.shape[0] * zoom), Image.Resampling.NEAREST)
        img.paste(im, (lab_w, y))
        d.text((4, y + a.shape[0] * zoom // 2 - 7), lab, fill=0, font=font)
        y += a.shape[0] * zoom + 4
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def run_stats(args, out: Path) -> dict:
    torch = _torch(args.threads)
    from learned_glyphs.lg.channel import Channel
    lines, corpus = D.source_lines("val", args.n_lines)
    results = {"corpus": corpus, "n_lines": len(lines), "ems": {}, "presets": {}}
    for p in args.presets:
        results["presets"][p] = Channel(p).describe()
    for em in args.ems:
        t0 = time.time()
        bank = make_bank(em)
        if em == max(args.ems):
            bank.atlas_png(out / f"atlas_dejavu_em{em:g}.png")
        r = {"height_px": height_for_em(em), "max_clip_frac": max(bank.clip_frac),
             "mean_clip_frac": float(np.mean(bank.clip_frac))}
        x_pe, w_pe = pil_lines(lines, em, train=False)
        x_pt, w_pt = pil_lines(lines, em, train=True, tag="train",
                               augment_gen=torch.Generator().manual_seed(1))
        stats = {"pil_eval": pixel_stats(x_pe, w_pe), "pil_train": pixel_stats(x_pt, w_pt)}
        imgs = {"pil_eval": (x_pe, w_pe), "pil_train": (x_pt, w_pt)}
        mads = {}
        for p in args.presets:
            ch = Channel(p)
            xe, we = bank_lines(bank, ch, lines, train=False)
            stats[f"{p}_eval"] = pixel_stats(xe.numpy(), we.numpy())
            imgs[f"{p}_eval"] = (xe.numpy(), we.numpy())
            if not ch.test_only:
                xt, wt = bank_lines(bank, ch, lines, train=True, gen=torch.Generator().manual_seed(1))
                stats[f"{p}_train"] = pixel_stats(xt.numpy(), wt.numpy())
                imgs[f"{p}_train"] = (xt.numpy(), wt.numpy())
            mads[p] = aligned_mad(bank, ch, lines[: args.n_mad], em)
        for k, s in stats.items():
            if k.startswith("pil"):
                continue
            ref = stats["pil_eval" if k.endswith("_eval") else "pil_train"]
            s["dist_vs_pil"] = stat_distance(s, ref)
        r["stats"], r["aligned"] = stats, mads
        # px^2/char: pitch x content width / chars (bank) vs Phase-2 (1.2 em x content / chars)
        from learned_glyphs.lg.compose import line_widths_ss
        cw = line_widths_ss([encode(t) for t in lines], bank.widths_ss).numpy() / bank.ss
        nch = np.array([len(t) for t in lines])
        r["px_per_char_bank"] = float(np.mean(height_for_em(em) * cw / nch))
        r["px_per_char_phase2_formula"] = float(np.mean(1.2 * em * cw / nch))
        results["ems"][f"{em:g}"] = r
        for li in range(args.n_strip):
            rows = []
            for k in ["pil_eval"] + [f"{p}_eval" for p in args.presets] + ["pil_train"] + \
                     [f"{p}_train" for p in args.presets]:
                if k not in imgs:
                    continue
                x, w = imgs[k]
                rows.append((k, x[li, 0, :, : int(w[li])]))
            save_strip(out / "strips" / f"em{em:g}_line{li}.png", rows)
        print(f"[stats] em {em:g} h={r['height_px']} done in {time.time() - t0:.1f}s", flush=True)
    (out / "stats.json").write_text(json.dumps(results, indent=1))
    (out / "stats.md").write_text(stats_markdown(results, args.presets))
    print(stats_markdown(results, args.presets))
    return results


def stats_markdown(res: dict, presets) -> str:
    L = [f"# E0 pixel statistics ({res['n_lines']} val lines, corpus {res['corpus']})", "",
         "dist = hist L1 + |d mean ink|/ref + |d sharpness|/ref vs the PIL source of the same mode "
         "(eval vs Phase-2 test images; train vs Phase-2 train render + augment_batch).", "",
         "| em | h | source | mean ink | p99 | sharp | sharp/ink | bg std | dist |", "|---|---|---|---|---|---|---|---|---|"]
    for em, r in res["ems"].items():
        for k, s in r["stats"].items():
            L.append(f"| {em} | {r['height_px']} | {k} | {s['mean_ink']:.4f} | {s['p99_ink']:.3f} | "
                     f"{s['sharpness']:.4f} | {s['sharpness_per_ink']:.3f} | {s['bg_std']:.4f} | "
                     f"{s.get('dist_vs_pil', 0):.3f} |")
    L += ["", "Pixel-aligned relative MAD (phase 0 both sides, eval degradation); yardstick = PIL vs "
          "PIL shifted by 1/4 px.", "",
          "| em | preset | prefix (12 ch) | full line | PIL 1/4-px prefix | PIL 1/4-px full | clip max |",
          "|---|---|---|---|---|---|---|"]
    for em, r in res["ems"].items():
        for p, a in r["aligned"].items():
            L.append(f"| {em} | {p} | {a['rel_mad_prefix']:.3f} | {a['rel_mad_full']:.3f} | "
                     f"{a['pil_quarter_phase_rel_mad_prefix']:.3f} | {a['pil_quarter_phase_rel_mad_full']:.3f} | "
                     f"{r['max_clip_frac']:.3f} |")
    L += ["", "| em | px2/char bank (h x W / chars) | 1.2 em x W / chars |", "|---|---|---|"]
    for em, r in res["ems"].items():
        L.append(f"| {em} | {r['px_per_char_bank']:.2f} | {r['px_per_char_phase2_formula']:.2f} |")
    L += ["", "Mean dist per preset (lower = closer to PIL):", ""]
    for mode in ("eval", "train"):
        for p in presets:
            v = [r["stats"][f"{p}_{mode}"]["dist_vs_pil"] for r in res["ems"].values()
                 if f"{p}_{mode}" in r["stats"]]
            if v:
                L.append(f"* {p}_{mode}: {np.mean(v):.3f}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------- training
class BankLines:
    """Text lines + symbol ids + px widths for one split (rendered on the fly through the channel)."""

    def __init__(self, lines, bank, ss: int = 4, margin: int = 2):
        from learned_glyphs.lg.compose import line_widths_ss
        self.texts = lines
        self.ids = [encode(t) for t in lines]
        self.labels = [tuple(t) for t in lines]
        self.wss = line_widths_ss(self.ids, bank.widths_ss).numpy()
        self.widths = -(-self.wss // ss) + 2 * margin
        self.label_lens = np.array([len(x) for x in self.labels])
        self.ppc = bank.height_px * (self.wss / ss) / np.maximum(1, [len(t) for t in lines])


def _evaluate(model, sets_x, data: BankLines, vocab) -> dict:
    import torch
    from rapidfuzz.distance import Levenshtein
    from vtc.crnn.model import greedy_decode
    model.eval()
    edits = chars = 0
    preds = [None] * len(data.texts)
    with torch.no_grad():
        for idx, x, w in sets_x:
            lp, lens = model(x, w)
            for i, seq in zip(idx, greedy_decode(lp, lens)):
                hyp = "".join(vocab.decode(seq))
                ref = data.texts[int(i)]
                edits += Levenshtein.distance(ref, hyp)
                chars += len(ref)
                preds[int(i)] = hyp
    model.train()
    return {"cer": edits / max(1, chars), "preds": preds}


def _prerender(bank, channel, data: BankLines, batch: int = 64):
    """Eval-mode images (deterministic: one seeded generator over the whole set, width-sorted)."""
    import torch
    from learned_glyphs.lg.compose import compose
    gen = torch.Generator().manual_seed(channel.eval_seed)
    out = []
    with torch.no_grad():
        ink = bank()
        for idx in D.eval_batches(data.widths, batch):
            cv, wss = compose(ink, [data.ids[int(i)] for i in idx], bank.widths_ss)
            x, w = channel(cv, wss, generator=gen, train=False)
            out.append((idx, x, w))
    return out


def train_job(job: dict) -> dict:
    """Train a CRNN on fixed-bank + channel lines at one em; returns (and appends) a row."""
    torch = _torch(job.get("threads", 1))
    from torch import nn
    from learned_glyphs.lg.channel import Channel
    from learned_glyphs.lg.compose import compose
    from vtc.crnn.model import CRNN, n_params

    em, preset, steps = float(job["em"]), job["preset"], int(job["steps"])
    out = Path(job["out"])
    tag = f"{SCHEME}_{preset}_em{em:g}"
    logf = open(out / "logs" / f"{tag}.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    seed = job.get("seed", 0)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    bank = make_bank(em)
    ch_train = Channel(preset)
    ch_eval = Channel(job.get("eval_preset") or preset)
    tr_lines, corpus = D.source_lines("train", job["n_train"])
    te_lines, _ = D.source_lines("val", job["n_test"])
    va_lines, _ = D.source_lines("val", job["n_val"], skip=job["n_test"])
    train, val, test = (BankLines(x, bank) for x in (tr_lines, va_lines, te_lines))
    vocab = D.Vocab.from_labels(train.labels)
    stride = D.col_stride_for_em(em)
    sets = (train, val, test)
    expand = D.choose_expand(np.concatenate([s.widths for s in sets]),
                             np.concatenate([s.label_lens for s in sets]), stride)
    model = CRNN(len(vocab), col_stride=stride, expand=expand)
    opt = torch.optim.AdamW(model.parameters(), lr=job.get("lr", 2e-3), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=job.get("lr", 2e-3), total_steps=steps,
                                                pct_start=0.15)
    ctc = nn.CTCLoss(blank=D.Vocab.BLANK, zero_infinity=True)
    val_x = _prerender(bank, ch_eval, val)
    log(f"[{tag}] vocab={len(vocab)} expand={expand} params={n_params(model)} train={len(train.texts)} "
        f"h={bank.height_px} meanW={train.widths.mean():.0f} channel={ch_train.describe()}")
    batches = D.bucket_batches(train.widths, job["batch"], rng)
    ink = bank().detach()
    best = {"cer": math.inf, "step": 0, "state": None}
    curve_f = open(out / "curves" / f"{tag}.jsonl", "w")
    t0, eval_time, loss_acc, n_acc = time.time(), 0.0, 0.0, 0
    model.train()
    for step in range(1, steps + 1):
        idx = next(batches)
        cv, wss = compose(ink, [train.ids[int(i)] for i in idx], bank.widths_ss)
        x, w = ch_train(cv, wss, generator=gen, train=True)
        tg = [vocab.encode(train.labels[int(i)]) for i in idx]
        tgt = torch.tensor([t for s in tg for t in s], dtype=torch.long)
        tl = torch.tensor([len(s) for s in tg], dtype=torch.long)
        lp, lens = model(x, w)
        loss = ctc(lp, tgt, lens, tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        loss_acc += loss.item()
        n_acc += 1
        if step % job["eval_every"] == 0 or step == steps:
            te = time.time()
            ev = _evaluate(model, val_x, val, vocab)
            eval_time += time.time() - te
            pt = {"step": step, "train_loss": round(loss_acc / n_acc, 4), "val_cer": round(ev["cer"], 5),
                  "elapsed_s": round(time.time() - t0, 1)}
            curve_f.write(json.dumps(pt) + "\n")
            curve_f.flush()
            log(f"[{tag}] step {step} loss {pt['train_loss']:.3f} val_cer {ev['cer']:.4f} ({pt['elapsed_s']}s)")
            loss_acc, n_acc = 0.0, 0
            if ev["cer"] < best["cer"]:
                best = {"cer": ev["cer"], "step": step, "state": copy.deepcopy(model.state_dict())}
    train_time = time.time() - t0
    curve_f.close()
    model.load_state_dict(best["state"])
    ev = _evaluate(model, _prerender(bank, ch_eval, test), test, vocab)
    extra = {}
    for pe in job.get("extra_eval") or []:
        extra[pe] = _evaluate(model, _prerender(bank, Channel(pe), test), test, vocab)["cer"]
        log(f"[{tag}] TEST eval-channel {pe}: cer={extra[pe]:.4f}")
    ref = phase2_latin_cer().get(em)
    row = {"phase": "lg", "run": out.name, "train_mode": "calib", "scheme": SCHEME, "em": em,
           "px_per_char": float(np.mean(test.ppc)), "cer": ev["cer"], "symbol_cer": ev["cer"],
           "steps": steps, "train_time_s": round(train_time, 1), "n_params": n_params(model),
           "height_px": bank.height_px, "channel": preset, "eval_channel": ch_eval.preset,
           "n_tokens": 0, "learn_glyphs": False, "best_step": best["step"], "best_val_cer": best["cer"],
           "s_per_step": round((train_time - eval_time) / steps, 4), "expand": expand,
           "col_stride": stride, "n_train": len(train.texts), "n_test": len(test.texts),
           "batch": job["batch"], "seed": seed, "corpus": corpus, "channel_params": ch_train.describe(),
           "phase2_latin_cer": ref, "extra_eval_cer": extra}
    log(f"[{tag}] TEST cer={ev['cer']:.4f} (phase2 latin em{em:g}: {ref}) px/char={row['px_per_char']:.2f} "
        f"best_step={best['step']} time={train_time:.0f}s")
    with open(out / "samples" / f"{tag}.txt", "w") as fh:
        for i in range(min(20, len(test.texts))):
            fh.write(f"REF {test.texts[i]}\nHYP {ev['preds'][i]}\n\n")
    with open(out / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    logf.close()
    return row


def pil_ref_job(job: dict) -> dict:
    """Unchanged Phase-2 trainer (latin, same steps) for a same-budget reference."""
    _torch(1)
    from vtc.crnn.train import TrainConfig, train_run
    cfg = TrainConfig(scheme="latin", em=float(job["em"]), steps=int(job["steps"]), batch=job["batch"],
                      eval_every=job["eval_every"], n_train=job["n_train"], n_val=job["n_val"],
                      n_test=job["n_test"], threads=1, render_workers=1)
    row = train_run(cfg, Path(job["out"]) / "pil_ref")
    return {k: row[k] for k in ("scheme", "em", "px_per_char", "cer", "steps")}


def _run_job(job):
    try:
        return pil_ref_job(job) if job["kind"] == "pil" else train_job(job)
    except Exception as e:  # keep the pool alive; report
        import traceback
        traceback.print_exc()
        return {"error": repr(e), **{k: job[k] for k in ("kind", "em", "preset")}}


def phase2_latin_cer(path: Path = PHASE2_RUNS) -> dict[float, float]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("scheme") == "latin" and r.get("train_mode", "per_em") == "per_em":
                out[float(r["em"])] = r["cer"]
    return out


def summarise_training(out: Path) -> str:
    from vtc.analysis import p_at
    rows = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines() if x.strip()]
    p2 = [json.loads(x) for x in PHASE2_RUNS.read_text().splitlines()] if PHASE2_RUNS.exists() else []
    p2 = [r for r in p2 if r.get("scheme") == "latin"]
    L = ["# E0 training calibration", "", "| channel | em | h | px2/char | CER | CER via other eval channels | Phase-2 latin CER | steps |",
         "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["channel"], r["em"])):
        ex = " ".join(f"{k}={v:.4f}" for k, v in (r.get("extra_eval_cer") or {}).items())
        L.append(f"| {r['channel']} | {r['em']:g} | {r['height_px']} | {r['px_per_char']:.2f} | "
                 f"{r['cer']:.4f} | {ex} | {r.get('phase2_latin_cer')} | {r['steps']} |")
    L += ["", "P@3% (px2/char at CER 3%, isotonic + log-linear interpolation, vtc.analysis.p_at):", ""]
    ref = p_at(p2, "latin", 0.03) if p2 else None
    L.append(f"* Phase-2 latin (phase2_main): {ref}")
    for ch in sorted({r["channel"] for r in rows}):
        sub = [dict(r, scheme=SCHEME) for r in rows if r["channel"] == ch]
        p = p_at(sub, SCHEME, 0.03) if len({r["em"] for r in sub}) > 1 else None
        rel = f" ({p / ref:.2f}x Phase 2)" if p and ref else ""
        L.append(f"* train {ch} / eval {sub[0].get('eval_channel', ch)}: {p}{rel}")
        for pe in sorted({k for r in sub for k in (r.get("extra_eval_cer") or {})}):
            se = [dict(r, cer=r["extra_eval_cer"][pe]) for r in sub if pe in (r.get("extra_eval_cer") or {})]
            p = p_at(se, SCHEME, 0.03) if len({r["em"] for r in se}) > 1 else None
            rel = f" ({p / ref:.2f}x Phase 2)" if p and ref else ""
            L.append(f"  * eval through {pe}: {p}{rel}")
    L += ["", "Calibration target (PLAN E0): a preset within ~15% of Phase-2 latin P@3%."]
    return "\n".join(L) + "\n"


def run_training(args, out: Path) -> None:
    for d in ("logs", "curves", "samples"):
        (out / d).mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "runs.jsonl").exists() and not args.no_resume:
        for x in (out / "runs.jsonl").read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                done.add((r["channel"], float(r["em"]), r["steps"]))
    base = {"steps": args.steps, "batch": args.batch, "n_train": args.n_train, "n_val": args.n_val,
            "n_test": args.n_test, "eval_every": args.eval_every, "out": str(out), "threads": 1,
            "eval_preset": args.eval_preset, "extra_eval": args.extra_eval}
    jobs = [dict(base, kind="bank", em=em, preset=p)
            for em in sorted(args.train_ems, reverse=True) for p in args.train_presets
            if (p, float(em), args.steps) not in done]
    if args.pil_ref:
        jobs += [dict(base, kind="pil", em=em, preset="pil") for em in args.train_ems]
    print(f"[train] {len(jobs)} jobs ({len(done)} done), parallel={args.parallel}", flush=True)
    if args.parallel > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.parallel, maxtasksperchild=1) as pool:
            res = pool.map(_run_job, jobs, chunksize=1)
    else:
        res = [_run_job(j) for j in jobs]
    for r in res:
        print("[train] result", {k: r.get(k) for k in ("channel", "scheme", "em", "cer", "px_per_char", "error")})
    if (out / "runs.jsonl").exists():
        s = summarise_training(out)
        (out / "e0_training.md").write_text(s)
        print(s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ems", type=float, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--presets", nargs="+", default=STAT_PRESETS, help="presets for pixel stats")
    ap.add_argument("--n-lines", type=int, default=256, help="lines for pixel stats")
    ap.add_argument("--n-mad", type=int, default=64, help="lines for the aligned MAD")
    ap.add_argument("--n-strip", type=int, default=3)
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--train", action="store_true", help="train CRNN readers (E0 CER calibration)")
    ap.add_argument("--train-ems", type=float, nargs="+", default=None, help="default: --ems")
    ap.add_argument("--train-presets", nargs="+", default=PRESETS)
    ap.add_argument("--eval-preset", default=None, help="evaluate with this preset (default: same)")
    ap.add_argument("--extra-eval", nargs="*", default=["clean", "harsh+heldout"],
                    help="also report test CER through these (eval-only) channels")
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=40_000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--pil-ref", action="store_true", help="also run the Phase-2 trainer at --steps")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="smoke: ems 5 8, 64 stat lines; with --train: em 5, default preset, 300 steps")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.quick:
        args.ems = [5, 8] if args.ems == [4, 5, 6, 7, 8] else args.ems
        args.n_lines, args.n_mad, args.n_strip = 64, 32, 2
        if args.steps == 3500:
            args.steps = 300
        args.n_train, args.n_val, args.n_test, args.eval_every = 5000, 100, 200, 100
        args.train_ems = args.train_ems or [5]
        if args.train_presets == PRESETS:
            args.train_presets = ["default"]
    args.train_ems = args.train_ems or args.ems
    out = args.out or (OUT.with_name("calibration_quick") if args.quick else OUT)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                                               for k, v in vars(args).items()}, indent=1))
    if not args.no_stats:
        run_stats(args, out)
    if args.train:
        run_training(args, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
