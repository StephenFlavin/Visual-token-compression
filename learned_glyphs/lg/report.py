"""Reports for the learned-glyph sub-project (WP-LG3): atlases, confusions, summary + decision.

(a) ``atlas_comparison``: fixed DejaVu vs learned banks for the same symbols, each glyph shown at
    native supersampled resolution and at 1x (area-downsampled, = what the reader sees before
    blur/noise), both nearest-upscaled.
(b) ``confusions`` / ``confusions_from_samples`` / ``test_pass``: top confused symbol pairs from
    REF/HYP pairs (``samples/*.txt`` written by joint / transfer / Phase-2 runs, or a fresh test pass
    of a saved reader). Alignment = Levenshtein edit ops on symbol sequences (chars by default;
    with an Alphabet the strings are segmented with ``Alphabet.encode`` first).
(c) ``write_report``: rows from ``learned_glyphs/results/*/runs.jsonl`` (+ the parent's
    ``results/phase2/phase2_main/runs.jsonl``) -> ``vtc.analysis.write_summary`` (P@x, CIs,
    ratios vs best Latin, cer_curves.png) on display names ``<scheme>/<train_mode>[/<channel>]``,
    plus the PLAN decision section (S1, S2, GO) prepended to summary.md.

CLI: ``python3 -m learned_glyphs.lg.report [--out DIR] [--include-smoke] [--atlas BANK ...]``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

LG_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = LG_ROOT.parent
RESULTS = LG_ROOT / "results"
PHASE2_MAIN = REPO_ROOT / "results" / "phase2" / "phase2_main" / "runs.jsonl"
GO_RATIO = 0.70                      # >= 30 % fewer px^2/char
THRESHOLD = 0.03

DEFAULT_ATLAS_SYMBOLS = list("aeosnrilcdmhgu") + list("AEST") + list("0123") + list(".,;'")


# ============================================================================ (a) atlases
def _bank_cells(bank, hard: bool = True) -> tuple[dict[str, np.ndarray], int, int]:
    from .scheme import bank_bitmaps
    maps, hs, ss, meta = bank_bitmaps(bank, hard=hard)
    syms = meta.get("symbols") or [str(i) for i in range(len(maps))]
    return dict(zip(syms, maps)), hs, ss


def _load_any_bank(b):
    from .glyphs import GlyphBank
    return GlyphBank.load(str(b), learnable=False) if isinstance(b, (str, Path)) else b


def atlas_comparison(banks: list[tuple[str, object]], path: str | Path, symbols=None,
                     add_dejavu: bool = True, zoom_px: int = 56, hard: bool = True) -> Path:
    """Figure: one block per bank (label, GlyphBank or path); per block two rows - glyphs at ss res
    and at 1x - for the same ``symbols``. ``add_dejavu`` prepends a fixed DejaVu bank at the first
    bank's line pitch (the Phase-2 Latin reference) unless one is given."""
    from PIL import Image, ImageDraw, ImageFont

    from vtc import config

    banks = [(lab, _load_any_bank(b)) for lab, b in banks]
    if not banks:
        raise ValueError("no banks")
    symbols = list(symbols or DEFAULT_ATLAS_SYMBOLS)
    if add_dejavu and not any("dejavu" in lab.lower() for lab, _ in banks):
        from .glyphs import GlyphBank
        b0 = banks[0][1]
        banks.insert(0, (f"fixed DejaVu h={int(b0.height_px)}",
                         GlyphBank.from_font(symbols, config.LATIN_FONT, int(b0.height_px), ss=int(b0.ss))))
    try:
        font = ImageFont.truetype(str(config.LATIN_FONT), 13)
        small = ImageFont.truetype(str(config.LATIN_FONT), 11)
    except OSError:
        font = small = ImageFont.load_default()
    blocks = []
    z = max(1, zoom_px // max(int(b.height_ss) for _, b in banks))   # common scale: sizes comparable
    for lab, bank in banks:
        cells, hs, ss = _bank_cells(bank, hard=hard)
        tiles_hi, tiles_lo = [], []
        for s in symbols:
            c = cells.get(s)
            if c is None:
                c = np.zeros((hs, max(1, hs // 2)))
            w = c.shape[1]
            wp = int(math.ceil(w / ss))
            pad = np.zeros((hs, wp * ss))
            pad[:, :w] = c
            lo = pad.reshape(hs // ss, ss, wp, ss).mean(axis=(1, 3))
            hi_img = Image.fromarray((255 - 255 * c).round().astype(np.uint8))
            lo_img = Image.fromarray((255 - 255 * lo).round().astype(np.uint8))
            tiles_hi.append(hi_img.resize((w * z, hs * z), Image.Resampling.NEAREST))
            tiles_lo.append(lo_img.resize((wp * ss * z, hs * z), Image.Resampling.NEAREST))
        blocks.append((lab, tiles_hi, tiles_lo, hs * z))
    gap, lab_h, left = 6, 18, 6
    col_w = [max(max(b[1][i].width, b[2][i].width) for b in blocks) for i in range(len(symbols))]
    W = left + sum(w + gap for w in col_w) + left
    H = 22 + sum(lab_h + 2 * h + 3 * gap for _, _, _, h in blocks) + 16
    img = Image.new("L", (max(W, 420), H), 255)
    d = ImageDraw.Draw(img)
    d.text((left, 3), "top row: supersampled cell (ss res); bottom row: 1x area-downsampled",
           fill=90, font=small)
    y = 22
    for lab, hi, lo, h in blocks:
        d.text((left, y), lab, fill=0, font=font)
        y += lab_h
        x = left
        for i, w in enumerate(col_w):
            img.paste(hi[i], (x, y))
            img.paste(lo[i], (x, y + h + gap))
            d.rectangle([x - 1, y - 1, x + hi[i].width, y + h], outline=215)
            x += w + gap
        y += 2 * h + 3 * gap
    x = left
    for s, w in zip(symbols, col_w):
        d.text((x, y - 4), s.replace(" ", "␣")[:6], fill=60, font=small)
        x += w + gap
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ============================================================================ (b) confusions
def read_samples(path: str | Path) -> list[tuple[str, str]]:
    """REF/HYP pairs from a samples/*.txt file."""
    pairs, ref = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith("REF "):
            ref = line[4:]
        elif line.startswith("HYP ") and ref is not None:
            pairs.append((ref, line[4:]))
            ref = None
        elif line == "REF" or line == "HYP":   # empty strings
            if line == "REF":
                ref = ""
            elif ref is not None:
                pairs.append((ref, ""))
                ref = None
    return pairs


def confusions(pairs, alphabet=None, top: int = 20) -> dict:
    """Edit-op statistics over (ref, hyp) pairs. Symbols = chars, or Alphabet segments.

    -> {"subs": [((ref_sym, hyp_sym), n)...], "dels": [(sym, n)], "ins": [(sym, n)], "n_ref": N,
        "n_edits": E, "rate": E/N}; pairs are unordered-merged in "pairs" ((a, b) with a <= b)."""
    from rapidfuzz.distance import Levenshtein

    subs, dels, ins, upairs = Counter(), Counter(), Counter(), Counter()
    n_ref = 0
    for ref, hyp in pairs:
        if alphabet is not None:
            r = [alphabet.symbols[i] for i in alphabet.encode(ref)]
            h = [alphabet.symbols[i] for i in alphabet.encode(hyp)]
        else:
            r, h = list(ref), list(hyp)
        n_ref += len(r)
        for op in Levenshtein.editops(r, h):
            if op.tag == "replace":
                a, b = r[op.src_pos], h[op.dest_pos]
                subs[(a, b)] += 1
                upairs[tuple(sorted((a, b)))] += 1
            elif op.tag == "delete":
                dels[r[op.src_pos]] += 1
            elif op.tag == "insert":
                ins[h[op.dest_pos]] += 1
    n_ed = sum(subs.values()) + sum(dels.values()) + sum(ins.values())
    return {"subs": subs.most_common(top), "pairs": upairs.most_common(top),
            "dels": dels.most_common(top), "ins": ins.most_common(top), "n_ref": n_ref,
            "n_edits": n_ed, "rate": n_ed / max(1, n_ref)}


def confusions_from_samples(paths, alphabet=None, top: int = 20) -> dict:
    pairs = [p for f in paths for p in read_samples(f)]
    out = confusions(pairs, alphabet, top)
    out["n_lines"] = len(pairs)
    return out


def test_pass(bank_path, reader_path, n_test: int = 1000, channel: str | None = None,
              data_seed: int = 0, eval_lines=None) -> list[tuple[str, str]]:
    """Re-run a saved reader (joint: readers/<tag>.pt, transfer: checkpoints/<tag>.pt) on its bank's test
    lines -> (ref, hyp) pairs for ``confusions``."""
    import torch
    from vtc.crnn.model import CRNN

    from . import train as T
    from .transfer import _EvalCfg, _RenderCfg, alphabet_for_bank, load_frozen_bank

    ck = torch.load(str(reader_path), map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    bank = load_frozen_bank(bank_path)
    alphabet = alphabet_for_bank(bank)
    _, compose, Channel = T.lg1()
    ch = Channel(channel or cfg.get("channel", "default"), ss=int(bank.ss))
    reader = CRNN(ck["n_classes"], col_stride=ck["col_stride"], expand=ck["expand"], **(ck.get("model") or {}))
    reader.load_state_dict(ck["state"])
    if eval_lines is None:
        _, test = T.load_eval_lines(_EvalCfg(0, n_test, data_seed))
    else:
        test = list(eval_lines)
    ids = [alphabet.encode(t) for t in test]
    w = torch.as_tensor(bank.widths_ss, dtype=torch.long)
    with torch.no_grad():                   # eval glyphs are always hard (fixed banks: as stored)
        ink = bank(tau=1.0, hard=bool(cfg.get("hard", True)))
    ev = T.evaluate(ink, reader, alphabet, test, ids, w, compose, ch,
                    _RenderCfg(ss=int(bank.ss), pad_px=int(cfg.get("pad_px", 0))), seed=54321,
                    return_preds=True)
    return list(zip(test, ev["preds"]))


def confusion_md(conf: dict, title: str) -> list[str]:
    def s(x):
        return repr(x)[1:-1] if isinstance(x, str) else str(x)
    md = [f"### {title}", "",
          f"{conf.get('n_lines', '?')} lines, {conf['n_ref']} ref symbols, {conf['n_edits']} edits "
          f"(rate {conf['rate']:.2%}).", "", "| ref -> hyp | n |", "|---|---|"]
    md += [f"| `{s(a)}` -> `{s(b)}` | {n} |" for (a, b), n in conf["subs"][:12]]
    if conf["dels"]:
        md += ["", "deleted: " + ", ".join(f"`{s(a)}`×{n}" for a, n in conf["dels"][:10])]
    if conf["ins"]:
        md += ["", "inserted: " + ", ".join(f"`{s(a)}`×{n}" for a, n in conf["ins"][:10])]
    return md + [""]


# ============================================================================ (c) summary
def collect_rows(include_smoke: bool = False, phase2: bool = True, extra=()) -> list[dict]:
    """All lg rows (learned_glyphs/results/*/runs.jsonl, smoke*/ dirs only if asked) + Phase-2 rows."""
    files = sorted(RESULTS.glob("*/runs.jsonl"))
    if not include_smoke:
        files = [f for f in files if not f.parent.name.startswith("smoke")]
    files = [f for f in files if f.parent.name != "summary"]
    rows = []
    for f in list(files) + [Path(p) for p in extra]:
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                r.setdefault("run", f.parent.name)
                rows.append(r)
    if phase2 and PHASE2_MAIN.exists():
        for line in open(PHASE2_MAIN):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def display_name(r: dict) -> str:
    """lg rows -> '<scheme>/<train_mode>[/<channel>]'; Phase-2 rows keep their scheme name."""
    if r.get("phase") != "lg":
        return r["scheme"]
    mode = r.get("train_mode", "joint")
    n = f"{r['scheme']}/{mode}"
    ch = r.get("channel") or "default"
    return n if ch == "default" or mode == "bridge" else f"{n}/{ch}"


def analysis_rows(rows: list[dict]) -> list[dict]:
    """Copies with scheme = display name and phase = 2 (so vtc.analysis treats them uniformly)."""
    out = []
    for r in rows:
        if r.get("cer") is None or r.get("px_per_char") is None or r.get("em") is None:
            continue
        rr = dict(r)
        rr["scheme"] = display_name(r)
        rr["phase"] = 2
        rr.pop("variant", None)
        out.append(rr)
    return out


def p3(rows: list[dict], name: str, thr: float = THRESHOLD) -> tuple[float | None, str | None]:
    from vtc.analysis import CurveTable, crossing
    t = CurveTable(rows, name)
    if t.empty:
        return None, "missing"
    return crossing(t.x, t.mean(), thr, t.counts())


def _fmt_p(p, cens) -> str:
    if cens == "missing":
        return "missing"
    if p is None:
        return "> grid"
    return ("≤ " if cens == "left" else "") + f"{p:.1f}"


def _ratio(a, b):
    (pa, ca), (pb, cb) = a, b
    if ca == "missing" or cb == "missing":
        return None, "missing"
    if pa is None and pb is None:
        return None, "neither crosses"
    if pa is None:
        return math.inf, "learned never crosses"
    if pb is None:
        return 0.0, "reference never crosses (ratio is an upper bound)"
    note = None
    if ca == "left" and cb == "left":
        note = "both left-censored (undetermined)"
    elif ca == "left":
        note = "upper bound"
    elif cb == "left":
        note = "lower bound"
    return pa / pb, note


def _fmt_r(r) -> str:
    v, note = r
    if v is None:
        return f"n/a ({note})"
    s = "∞" if math.isinf(v) else f"{v:.2f}"
    return s + (f" ({note})" if note else "")


def decision(rows: list[dict], thr: float = THRESHOLD, n_tok: int | None = None) -> dict:
    """PLAN decision quantities on analysis rows (display names).

    S1 = P(learned N=0) / P(fixed Latin N=0) and S2 = P(learned N=k) / P(learned N=0), per train
    mode and channel. Fixed banks are frozen by construction, so their 'joint' rows ARE transfer
    rows (fresh reader, frozen glyphs) and serve as the fixed reference for both joint and transfer.
    GO: learned+tokens <= 0.70 x best fixed Latin in transfer (vs lg_fixed_n0 through the same
    channel, and vs Phase-2 best Latin for the default channel) AND in the bridge (vs Phase-2 best
    Latin), and the transfer gain also holds in the harsh channel.
    """
    names = {r["scheme"] for r in rows}
    toks = sorted({int(m.group(1)) for s in names if (m := re.match(r"lg_learned_n(\d+)/", s))} - {0})
    k = n_tok if n_tok is not None else (300 if 300 in toks else (toks[-1] if toks else 300))
    lat = [p3(rows, s, thr) for s in ("latin", "latin_condensed")]
    lat_ok = [x for x in lat if x[1] != "missing"]
    fin = [x for x in lat_ok if x[0] is not None]
    best_lat = min(fin, key=lambda x: x[0]) if fin else (lat_ok[0] if lat_ok else (None, "missing"))
    out = {"threshold": thr, "n_tokens": k, "best_phase2_latin": best_lat, "modes": {}}
    for ch in ("default", "harsh"):
        sfx = "" if ch == "default" else f"/{ch}"
        fixed0 = p3(rows, f"lg_fixed_n0/joint{sfx}", thr)
        if fixed0[1] == "missing":
            fixed0 = p3(rows, f"lg_fixed_n0/transfer{sfx}", thr)
        for mode in ("joint", "transfer", "bridge"):
            if mode == "bridge" and ch != "default":
                continue
            l0 = p3(rows, f"lg_learned_n0/{mode}{sfx}", thr)
            lk = p3(rows, f"lg_learned_n{k}/{mode}{sfx}", thr)
            ref = best_lat if mode == "bridge" else fixed0
            out["modes"][f"{mode}{sfx}"] = {
                "P_learned_n0": l0, f"P_learned_n{k}": lk, "P_fixed_ref": ref,
                "S1": _ratio(l0, ref), "S2": _ratio(lk, l0), "R_tokens_vs_fixed": _ratio(lk, ref),
                "R_tokens_vs_phase2_latin": _ratio(lk, best_lat) if ch == "default" else (None, "n/a"),
            }

    def passes(r):
        return r[0] is not None and r[0] <= GO_RATIO and r[1] not in ("lower bound",
                                                                      "both left-censored (undetermined)")
    m = out["modes"]
    crit = {
        "transfer vs fixed (same channel)": passes(m["transfer"]["R_tokens_vs_fixed"]),
        "transfer vs Phase-2 best Latin": passes(m["transfer"]["R_tokens_vs_phase2_latin"]),
        "bridge vs Phase-2 best Latin": passes(m["bridge"]["R_tokens_vs_fixed"]),
        "harsh transfer vs harsh fixed": passes(m["transfer/harsh"]["R_tokens_vs_fixed"]),
    }
    out["criteria"] = crit
    out["GO"] = all(crit.values())
    return out


def decision_md(dec: dict) -> list[str]:
    k, thr = dec["n_tokens"], dec["threshold"]
    md = ["## Decision (PLAN criteria)", "",
          f"P@{thr:.0%} in px²/char ('≤' = already below threshold at the smallest grid point; "
          "'> grid' = never reaches it). Fixed reference for joint/transfer = `lg_fixed_n0/joint` (else `/transfer`) "
          "through the same channel (fixed banks are frozen, so their runs are transfer runs); for "
          "the bridge = best Phase-2 Latin "
          f"(**{_fmt_p(*dec['best_phase2_latin'])}**).", "",
          f"| mode | P learned N=0 | P learned N={k} | P fixed ref | S1 = L0/F0 | S2 = L{k}/L0 | "
          f"L{k}/fixed | L{k}/Phase-2 Latin |", "|---|---|---|---|---|---|---|---|"]
    for mode, d in dec["modes"].items():
        md.append(f"| {mode} | {_fmt_p(*d['P_learned_n0'])} | {_fmt_p(*d[f'P_learned_n{k}'])} | "
                  f"{_fmt_p(*d['P_fixed_ref'])} | {_fmt_r(d['S1'])} | {_fmt_r(d['S2'])} | "
                  f"{_fmt_r(d['R_tokens_vs_fixed'])} | {_fmt_r(d['R_tokens_vs_phase2_latin'])} |")
    md += ["", f"GO needs learned+tokens ≤ {GO_RATIO:.2f} × fixed in every criterion:", ""]
    md += [f"* {'PASS' if v else 'fail'}: {c}" for c, v in dec["criteria"].items()]
    md += ["", f"**Verdict: {'GO' if dec['GO'] else 'NO-GO (or incomplete data)'}**", ""]
    s1 = dec["modes"]["transfer"]["S1"][0]
    s2 = dec["modes"]["transfer"]["S2"][0]
    if s1 is not None and s2 is not None and not math.isinf(s1) and not math.isinf(s2):
        if s1 < 0.85 and 0.9 <= s2 <= 1.1:
            md += ["Reading: S1 large, S2 ≈ 1 -> 'optimised low-res fonts', not Visual BPE.", ""]
    return md


def write_report(out_dir: str | Path = RESULTS / "summary", rows: list[dict] | None = None,
                 include_smoke: bool = False, n_boot: int = 1000, samples: bool = True,
                 title: str = "Learned glyphs vs Phase-2 typography") -> dict:
    """summary.md / summary.json / cer_curves.png / p_at_bars.png (+ decision.json)."""
    from vtc import analysis

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(include_smoke) if rows is None else rows
    arows = analysis_rows(rows)
    if not arows:
        (out / "summary.md").write_text(f"# {title}\n\nNo usable rows yet.\n")
        return {"rows": 0}
    summ = analysis.write_summary(arows, out, n_boot=n_boot, title=title)
    dec = decision(arows)
    (out / "decision.json").write_text(json.dumps(dec, indent=1, default=str))
    md = decision_md(dec)
    if samples:
        files = sorted(f for f in RESULTS.glob("*/samples/*.txt")
                       if include_smoke or not f.parent.parent.name.startswith("smoke"))
        if files:
            md += ["## Top confusions (char level, from samples/)", ""]
            for f in files[:12]:
                c = confusions_from_samples([f])
                if c["n_ref"]:
                    md += confusion_md(c, f"{f.parent.parent.name}/{f.stem}")
    body = (out / "summary.md").read_text().split("\n")
    head, rest = body[:1], body[1:]
    (out / "summary.md").write_text("\n".join(head + [""] + md + rest))
    return {"rows": len(arows), "summary": summ, "decision": dec}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "summary"))
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--atlas", nargs="*", default=None, help="bank paths/globs for an atlas figure")
    ap.add_argument("--symbols", default=None, help="atlas symbols, '|'-separated")
    a = ap.parse_args(argv)
    if a.atlas:
        paths = [p for g in a.atlas for p in (sorted(glob.glob(g)) or [g])]
        syms = a.symbols.split("|") if a.symbols else None
        p = atlas_comparison([(Path(x).stem, x) for x in paths], Path(a.out) / "atlas_comparison.png", syms)
        print(f"wrote {p}")
    r = write_report(a.out, include_smoke=a.include_smoke, n_boot=a.n_boot)
    print((Path(a.out) / "summary.md").read_text())
    return 0 if r["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
