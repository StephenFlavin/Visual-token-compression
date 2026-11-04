"""Quick test Q1 (2026-11-04): layer-wise linear readout of frozen Qwen2.5-VL ViT features -> page pixels.

    python3 encoder_probe/experiments/quick_linear_readout.py --threads 2            # all encodings, Qwen + DeepEncoder
    python3 encoder_probe/experiments/quick_linear_readout.py --summarize            # summary.md from rows.jsonl

Does the frozen vision encoder keep the pixels of the dense codes, and at which depth are they lost? For each
encoding (448-px pages, 64 train / 16 held-out test pages) and each feature:
  raw        the 14x14x3 patch pixels themselves (calibration, R^2 = 1)
  embed      patch embedding (Conv3d, before block 0; a linear map of the pixels: checks the raster reordering)
  b8 ... b28 per-patch hidden states after blocks 0..k-1 (k = 8, 16, 24, 28; 1280-d, 14-px patches)
  final      after all 32 blocks, before the merger
  merger     the patch-merger tokens the LLM receives (2048-d, 28-px tokens)
a closed-form ridge regression maps the feature (plus a bias) to the clean pixels it covers (588 values per
patch, 2352 per merged token). Lambda is picked on train pages 48-63 (fit on 0-47), then refit on all 64 train
pages; test pages are only scored. Inside the blocks the tower keeps patches in attention-window order: hidden states
are put back in raster order with the window index (merger output rebuilt from the final states is checked against
the module's pooler_output). Metrics on the test pages: R^2 of the pixels (1 - SSE/SST, SST around the test mean per
output value) and ink F1 per channel (truth and reconstruction thresholded at the channel's ink midpoint
(paper + ink) / 2 of the true test pages; ink = the dark side). Calibration: DeepEncoder projector tokens (1280-d
per 64-px token at 512-px pages; 1280 dims cannot hold 12288 pixels linearly) for latin_p8 and rgb3_lg_pb_h5.
A linear readout is a lower bound on what the features hold.
Rows: encoder_probe/results/quick_q1/rows.jsonl; panels: docs/img/q1_readout_<encoding>.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from encoder_probe.ep.geom import get_encoding_s, make_pages_s  # noqa: E402

OUT = ROOT / "encoder_probe" / "results" / "quick_q1"
ENCODINGS = ["latin_p16", "latin_p8", "col_b4", "rgb3_p8", "lg_pb_h5", "rgb3_lg_pb_h5"]
QWEN_LAYERS = ["raw", "embed", "b8", "b16", "b24", "b28", "final", "merger"]
ALPHAS = [10.0 ** e for e in range(-7, 2)]
N_FIT, N_TRAIN, N_TEST = 48, 64, 16


def patch_pixels(imgs: np.ndarray, P: int) -> np.ndarray:
    """(B,S,S,3) uint8 -> (B*G*G, P*P*3) float in [0,1], tokens row-major, values (dy, dx, ch)."""
    B, S = imgs.shape[:2]
    G = S // P
    x = imgs.astype(np.float32) / 255.0
    return x.reshape(B, G, P, G, P, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B * G * G, P * P * 3)


def unpatch(y: np.ndarray, B: int, S: int, P: int) -> np.ndarray:
    G = S // P
    return y.reshape(B, G, G, P, P, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, S, S, 3)


def window_to_raster(h: torch.Tensor, reverse: torch.Tensor, B: int, g: int) -> torch.Tensor:
    """Packed window-ordered patch states (B*g*g, d) -> (B*g*g, d) raster patch order per image.

    Window order permutes 2x2 merge groups (reverse = argsort(window_index) restores group-major order); inside a
    group the 4 patches are (dy, dx). Group-major (gy, gx, dy, dx) -> raster (2gy+dy, 2gx+dx)."""
    d = h.shape[-1]
    x = h.reshape(-1, 4, d)[reverse]                                      # group-major, row-major over groups
    x = x.reshape(B, g // 2, g // 2, 2, 2, d).permute(0, 1, 3, 2, 4, 5)   # B, gy, dy, gx, dx, d
    return x.reshape(B * g * g, d)


class QwenLayers:
    """Frozen Qwen2.5-VL-3B vision tower -> dict of raster-ordered features per layer (fp32 on the CPU)."""

    def __init__(self):
        from encoder_probe.ep.vlm_encoders import QwenVision
        self.v = QwenVision()
        self.m = self.v.model
        self.max_abs_check = 0.0

    @torch.no_grad()
    def __call__(self, imgs: np.ndarray) -> dict[str, np.ndarray]:
        from encoder_probe.ep import qwen_split as QS
        from encoder_probe.ep.vlm_encoders import qwen_pixel_values
        B, S = imgs.shape[:2]
        g = S // 14
        m = self.m
        pv = torch.from_numpy(qwen_pixel_values(imgs)).to(self.v.dtype)
        thw = torch.tensor([[1, g, g]] * B)
        meta = QS.vit_meta(m, thw)
        h = m.patch_embed(pv)                                             # processor order (group-major)
        out = {"embed": h}
        out["embed"] = h.reshape(B, g // 2, g // 2, 2, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(B * g * g, -1)
        h = m.permute_input_for_window_attn(h, meta["window_index"], seq_len=h.shape[0])
        pe = QS._pos_emb(m, meta, h)
        prev = 0
        for k, name in ((8, "b8"), (16, "b16"), (24, "b24"), (28, "b28"), (32, "final")):
            h = QS.run_blocks(m, h, meta, prev, k, pos_emb=pe)
            prev = k
            out[name] = window_to_raster(h, meta["reverse"], B, g)
        merged = m.merger(h)[meta["reverse"]]
        ref = m(hidden_states=pv, grid_thw=thw).pooler_output if not hasattr(self, "_checked") else None
        if ref is not None:                                               # once: rebuilt merger == pooler_output
            self.max_abs_check = (merged - ref).abs().max().item()
            assert torch.allclose(merged, ref, atol=1e-4, rtol=1e-4), self.max_abs_check
            self._checked = True
        out["merger"] = merged
        return {k: v.float().numpy() for k, v in out.items()}


class DeepLayers:
    def __init__(self):
        from encoder_probe.ep.vlm_encoders import DeepEncoderVision
        self.v = DeepEncoderVision()

    def __call__(self, imgs):
        return {"deepencoder": self.v(imgs).float().reshape(-1, 1280).numpy()}


class Ridge:
    """Sufficient statistics for ridge regression with an unpenalised bias, fit / val / all."""

    def __init__(self):
        self.s = {}

    def add(self, part: str, X: np.ndarray, Y: np.ndarray):
        X = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1).astype(np.float64)
        Y = Y.astype(np.float64)
        st = self.s.setdefault(part, [0.0, 0.0, 0.0])
        st[0] = st[0] + X.T @ X
        st[1] = st[1] + X.T @ Y
        st[2] = st[2] + float((Y * Y).sum())

    @staticmethod
    def solve(XtX, XtY, lam):
        A = XtX.copy()
        i = np.arange(len(A) - 1)
        A[i, i] += lam
        return np.linalg.solve(A, XtY)

    def fit(self):
        XtX_f, XtY_f, _ = self.s["fit"]
        XtX_v, XtY_v, YtY_v = self.s["val"]
        scale = np.trace(XtX_f[:-1, :-1]) / (len(XtX_f) - 1)
        best = None
        for a in ALPHAS:
            W = self.solve(XtX_f, XtY_f, a * scale)
            sse = YtY_v - 2 * np.sum(W * XtY_v) + np.sum(W * (XtX_v @ W))
            if best is None or sse < best[0]:
                best = (sse, a)
        XtX = XtX_f + XtX_v
        XtY = XtY_f + XtY_v
        lam = best[1] * np.trace(XtX[:-1, :-1]) / (len(XtX) - 1)
        return self.solve(XtX, XtY, lam).astype(np.float32), best[1]


def ink_f1(truth: np.ndarray, rec: np.ndarray) -> list[float]:
    """Per channel F1 of ink pixels (dark side of the channel's (paper + ink) / 2 midpoint)."""
    f = []
    for c in range(3):
        t, r = truth[..., c], rec[..., c]
        mid = (t.max() + t.min()) / 2
        ti, ri = t < mid, r < mid
        tp = np.sum(ti & ri)
        f.append(float(2 * tp / max(1, ti.sum() + ri.sum())))
    return f


def run_encoding(model, encoding: str, S: int, P_of: dict, log=print) -> list[dict]:
    enc = get_encoding_s(encoding, S)
    t0 = time.time()
    tr = make_pages_s(enc, "train", N_TRAIN)
    te = make_pages_s(enc, "test", N_TEST, allow_fewer=True)
    log(f"[q1] {encoding} S={S}: pages {time.time() - t0:.0f}s", flush=True)
    ridges, test_x = {}, {}
    t1 = time.time()
    for split, pages in (("train", tr), ("test", te)):
        for i in range(0, len(pages), 2):
            imgs = np.stack([p.img for p in pages[i:i + 2]])
            feats = model(imgs)
            if "merger" in feats:
                feats = {"raw": patch_pixels(imgs, 14), **feats}
            for name, X in feats.items():
                if split == "test":
                    test_x.setdefault(name, []).append(X.astype(np.float32))
                    continue
                Y = patch_pixels(imgs, P_of[name])
                ridges.setdefault(name, Ridge()).add("fit" if i < N_FIT else "val", X, Y)
        log(f"[q1] {encoding} {split} features done ({time.time() - t1:.0f}s)", flush=True)
    truth = np.stack([p.img for p in te]).astype(np.float32) / 255.0
    rows, recs = [], {}
    for name, rd in ridges.items():
        W, alpha = rd.fit()
        X = np.concatenate(test_x[name])
        Yhat = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1) @ W
        rec = np.clip(unpatch(Yhat, len(te), S, P_of[name]), 0, 1)
        ytrue = truth
        sse = float(((rec - ytrue) ** 2).sum())
        Yt = patch_pixels((truth * 255).round().astype(np.uint8), P_of[name])
        sst = float(((Yt - Yt.mean(0)) ** 2).sum())
        f1 = ink_f1(ytrue, rec)
        chars = sum(p.n_chars for p in te)
        rows.append({"encoding": encoding, "feature": name, "S": S, "P": P_of[name], "d": int(X.shape[1]),
                     "targets": int(W.shape[1]), "r2": 1 - sse / sst, "ink_f1": f1, "ink_f1_mean": float(np.mean(f1)),
                     "alpha": alpha, "train_pages": len(tr), "test_pages": len(te), "fit_pages": N_FIT,
                     "chars_per_qwen_token": chars / (len(te) * (S // 28) ** 2) if S == 448 else None,
                     "chars_per_vlm_token": chars / (len(te) * (S // P_of[name]) ** 2),
                     "check_merger_vs_pooler_max_abs": getattr(model, "max_abs_check", None)})
        recs[name] = (rec[0] * 255).round().astype(np.uint8)
        log(f"[q1] {encoding:14s} {name:11s} R2 {rows[-1]['r2']:.4f} inkF1 "
            f"{' '.join(f'{v:.3f}' for v in f1)} alpha {alpha:g}", flush=True)
    recs["original"] = te[0].img
    return rows, recs


def panel(encoding: str, recs: dict, names: list[str]) -> Path:
    from PIL import Image, ImageDraw
    ims = [("original", recs["original"])] + [(n, recs[n]) for n in names if n in recs]
    S = ims[0][1].shape[0]
    cols = 4
    rows = (len(ims) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (S + 8), rows * (S + 24)), "white")
    d = ImageDraw.Draw(canvas)
    for k, (n, im) in enumerate(ims):
        x, y = (k % cols) * (S + 8), (k // cols) * (S + 24)
        canvas.paste(Image.fromarray(im), (x, y + 20))
        d.text((x + 4, y + 4), n if n == "original" else f"linear readout from {n}", fill="black")
    path = ROOT / "docs" / "img" / f"q1_readout_{encoding}.png"
    canvas.save(path)
    return path


def summarize() -> Path:
    rows = [json.loads(l) for l in (OUT / "rows.jsonl").read_text().splitlines() if l.strip()]
    last = {}
    for r in rows:
        last[(r["encoding"], r["feature"], r["S"])] = r
    feats = QWEN_LAYERS
    L = ["# Quick test Q1: linear readout of frozen Qwen2.5-VL ViT features (2026-11-04)", "",
         "Closed-form ridge from each feature to the clean pixels it covers (14-px patches; merger: 28-px tokens), "
         f"{N_TRAIN} train pages (lambda picked on pages {N_FIT}-{N_TRAIN - 1}), {N_TEST} held-out test pages "
         "(WikiText test/val), 448-px pages. bN = after blocks 0..N-1; final = after block 31, before the merger. "
         "A linear readout is a lower bound on what a feature holds.", "",
         "## R^2 of the test-page pixels", "",
         "| encoding | chars/Qwen token | " + " | ".join(feats) + " |", "|---|---:|" + "---:|" * len(feats)]
    encs = [e for e in ENCODINGS if any(k[0] == e and k[2] == 448 for k in last)]
    for e in encs:
        cpt = next(r["chars_per_qwen_token"] for k, r in last.items() if k[0] == e and k[2] == 448)
        L.append(f"| {e} | {cpt:.1f} | " + " | ".join(
            f"{last[(e, f, 448)]['r2']:.3f}" if (e, f, 448) in last else "-" for f in feats) + " |")
    L += ["", "## Ink F1 (mean over R, G, B; rgb3 codes per channel R/G/B)", "",
          "| encoding | " + " | ".join(feats) + " |", "|---|" + "---:|" * len(feats)]
    for e in encs:
        cells = []
        for f in feats:
            r = last.get((e, f, 448))
            if r is None:
                cells.append("-")
            elif e.startswith("rgb3"):
                cells.append("/".join(f"{v:.2f}" for v in r["ink_f1"]))
            else:
                cells.append(f"{r['ink_f1_mean']:.3f}")
        L.append(f"| {e} | " + " | ".join(cells) + " |")
    de = [r for k, r in last.items() if k[1] == "deepencoder"]
    if de:
        L += ["", "## Calibration: DeepEncoder projector tokens (512-px pages, 64-px tokens, 1280-d -> 12288 pixels)",
              "", "| encoding | chars/DeepEncoder token | R^2 | ink F1 (R/G/B) |", "|---|---:|---:|---|"]
        for r in de:
            L.append(f"| {r['encoding']} | {r['chars_per_vlm_token']:.1f} | {r['r2']:.3f} | "
                     + "/".join(f"{v:.2f}" for v in r["ink_f1"]) + " |")
    chk = [r["check_merger_vs_pooler_max_abs"] for r in last.values() if r.get("check_merger_vs_pooler_max_abs")
           is not None and r["feature"] == "merger"]
    if chk:
        L += ["", f"Check: merger output rebuilt from the raster/window bookkeeping vs the module's pooler_output, "
                  f"max |diff| {max(chk):.2e} (fp32)."]
    p = OUT / "summary.md"
    p.write_text("\n".join(L) + "\n")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encodings", nargs="+", default=ENCODINGS)
    ap.add_argument("--deep-encodings", nargs="*", default=["latin_p8", "rgb3_lg_pb_h5"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--panels", nargs="*", default=["latin_p8", "rgb3_lg_pb_h5"])
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(argv)
    if a.summarize:
        print(summarize().read_text())
        return 0
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT / "rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {(json.loads(l)["encoding"], json.loads(l)["S"], json.loads(l)["feature"] == "deepencoder")
                for l in rows_path.read_text().splitlines() if l.strip()}
    pq = {n: 14 for n in QWEN_LAYERS}
    pq["merger"] = 28
    for vlm, encs, S, P_of in (("qwen", a.encodings, 448, pq), ("deep", a.deep_encodings, 512, {"deepencoder": 64})):
        todo = [e for e in encs if (e, S, vlm == "deep") not in done]
        if not todo:
            continue
        t0 = time.time()
        model = QwenLayers() if vlm == "qwen" else DeepLayers()
        print(f"[q1] {vlm} loaded ({time.time() - t0:.0f}s)", flush=True)
        for e in todo:
            rows, recs = run_encoding(model, e, S, P_of)
            with open(rows_path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            if vlm == "qwen" and e in a.panels:
                print(f"[q1] panel {panel(e, recs, ['embed', 'b8', 'b16', 'b24', 'b28', 'final', 'merger'])}")
        del model
    print(summarize().read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
