"""Round 3 (R3-2): long-context QA over a document given as text, Latin page images or dense-code page images.

    python longctx_bench.py --inputs text latin_p8 --lengths 10000 --tasks needle --depths 0.5 --run-name round3_lc_smoke
    python longctx_bench.py --inputs text latin_p8 col_b4 --adapter encoder_probe/cache/r3_adapters/<file>.pt

Plan: encoder_probe/ROUND3.md (R3-2). Runbook (RTX 3090): docs/gpu/LOCAL_GPU.md, "Round 3 on the 3090".

* Documents: concatenated held-out WikiText (data/test.txt + data/val.txt, 2.06M chars, never used for training),
  ``--lengths`` chars from a seeded random offset, cut at word boundaries.
* Inputs (``--inputs``): ``text`` = the document as plain text tokens; any page-encoding name (latin_p8, col_b4,
  rgb3_lg_pb_h5, ...) = the document rendered onto consecutive 448-px pages by the round-2 renderers
  (get_encoding_s), one image (256 positions) per page, then the question as text.
* Model: the base Qwen2.5-VL-3B-Instruct, or with ``--adapter`` a finetune_qwen.py result (LoRA merged into the
  weights). bf16 on CUDA.
* Tasks (short answers only, never a transcription, so the answer does not refill the KV cache):
  ``needle`` = 'The secret number of <name> is <6 digits>.' inserted at each of ``--depths`` (0-1 of the document),
  question 'What is the secret number of <name>?'; ``cloze`` = 'which word comes right after "<6 words>"' for a
  context unique in the document (``--n-cloze`` per document).
* Measured per item: correct, context positions (= KV entries), M-RoPE span, KV-cache bytes summed over the
  cache's key/value tensors after prefill (and the analytic layers x 2 x kv_heads x head_dim x positions x bytes),
  peak GPU memory, vision-encoder time and LLM prefill time. Rows (one per input x length x task, items inside):
  encoder_probe/results/<run>/runs.jsonl. An input whose positions exceed ``--max-positions`` (default 32768,
  the native window) is not run: its row has ``skipped: true`` and the reason. ``--rope-scaling`` (JSON merged
  into the text config's rope_parameters, e.g. YaRN) is recorded in every row, as is ``max_positions``.
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
from encoder_probe.ep import qwen_ft as Q  # noqa: E402
from encoder_probe.ep import r3_tasks as T  # noqa: E402
from encoder_probe.ep.encoders import degrade, resolve_device  # noqa: E402
from encoder_probe.ep.geom import get_encoding_s, text_stream_s  # noqa: E402

S = 448
DOC_HEAD_TEXT = "Read the document below, then answer the question.\n\n<document>\n"
DOC_TAIL_TEXT = "\n</document>\n\n"
IMG_HEAD = "The images above are the pages of one document, in order. "


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", default=["text", "latin_p8"], help="'text' and/or page encodings")
    ap.add_argument("--lengths", nargs="+", type=int, default=[10_000, 30_000, 100_000, 300_000, 1_000_000])
    ap.add_argument("--tasks", nargs="+", default=["needle", "cloze"], choices=["needle", "cloze"])
    ap.add_argument("--depths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n-cloze", type=int, default=5, help="cloze questions per document")
    ap.add_argument("--cond", default="clean", help="degradation of the page images (encoders.degrade)")
    ap.add_argument("--adapter", default=None, help="finetune_qwen.py adapter file (LoRA + vision); default: base")
    ap.add_argument("--max-positions", type=int, default=32768, help="skip inputs longer than this (KV entries)")
    ap.add_argument("--rope-scaling", default=None,
                    help='JSON merged into rope_parameters, e.g. \'{"rope_type":"yarn","factor":4.0,'
                         '"original_max_position_embeddings":32768}\' (record it; raise --max-positions with it)')
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--vision-batch", type=int, default=16, help="pages per vision-encoder call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto = cuda if available")
    ap.add_argument("--run-name", default="round3_longctx")
    ap.add_argument("--tiny", action="store_true", help="random-init tiny model + byte tokenizer (tests)")
    ap.add_argument("--out-root", default=str(ROOT / "encoder_probe" / "results"))
    return ap.parse_args(argv)


def render_document(enc, doc: str) -> list[np.ndarray]:
    """The whole document on consecutive pages of ``enc`` (the last page may be partly empty)."""
    imgs, pos = [], 0
    while pos < len(doc) and doc[pos:].strip():
        pg, used = enc.render(doc, pos)
        if used <= 0:
            raise RuntimeError(f"{enc.name}: renderer consumed no text at {pos}")
        imgs.append(pg.img)
        pos += used
    return imgs


def make_sample(kind: str, doc: str, question: str, enc, cond: str) -> dict:
    if kind == "text":
        return {"imgs": [], "doc_text": DOC_HEAD_TEXT + doc + DOC_TAIL_TEXT, "prompt": question}
    return {"imgs": [degrade(im, cond) for im in render_document(enc, doc)], "prompt": IMG_HEAD + question}


def n_positions(tok, sample: dict) -> int:
    im = sample["imgs"]
    ids, _, _ = Q.chat_ids(tok, sample["prompt"], len(im), Q.image_tokens(S) if im else 0,
                           doc_text=sample.get("doc_text"))
    return len(ids)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def answer(model, tok, sample: dict, dev, max_new: int, vision_batch: int = 16) -> dict:
    """Prefill (vision encoder in chunks, then one LLM forward with logits for the last position only), read the
    KV cache, then greedy-decode a short answer from the cache."""
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    m = Q.hf_model(model)
    m.model.rope_deltas = None
    b = Q.collate([{**sample, "answer": None}], tok, dev)
    kw = {"input_ids": b["input_ids"], "mm_token_type_ids": b["mm_token_type_ids"]}
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sync(dev)
    t0 = time.time()
    imgs = sample["imgs"]
    if imgs:
        feats = []
        for i in range(0, len(imgs), vision_batch):
            pv = Q.pixel_inputs(imgs[i:i + vision_batch], dev)
            feats += list(m.model.get_image_features(pv["pixel_values"], pv["image_grid_thw"]).pooler_output)
        kw["image_grid_thw"] = Q.pixel_inputs(imgs[:1], dev)["image_grid_thw"].repeat(len(imgs), 1)
        kw["mm_encoder_outputs"] = {"image": BaseModelOutputWithPooling(pooler_output=tuple(feats))}
    sync(dev)
    t_vis = time.time() - t0
    out = m(**kw, use_cache=True, logits_to_keep=1)
    sync(dev)
    t_pre = time.time() - t0 - t_vis
    cache = out.past_key_values
    kvb, kv_pos = Q.kv_bytes(cache)
    rope_span = int(kv_pos + (m.model.rope_deltas.max().item() if m.model.rope_deltas is not None else 0))
    peak_prefill = torch.cuda.max_memory_allocated() if dev.type == "cuda" else None
    nxt = out.logits[:, -1].argmax(-1)
    gen = []
    for _ in range(max_new):
        t = int(nxt.item())
        if t in (tok.im_end, tok.eos):
            break
        gen.append(t)
        o = m(input_ids=nxt[:, None], past_key_values=cache, use_cache=True)
        nxt = o.logits[:, -1].argmax(-1)
    sync(dev)
    res = {"hyp": tok.decode(gen), "gen_ids": gen, "positions": int(b["input_ids"].shape[1]), "kv_positions": kv_pos,
           "kv_bytes": kvb, "kv_bytes_analytic": Q.kv_bytes_analytic(m, kv_pos,
                                                                     cache.layers[0].keys.element_size()),
           "rope_span": rope_span, "vision_s": t_vis, "prefill_s": t_pre, "decode_s": time.time() - t0 - t_vis - t_pre,
           "peak_gpu_bytes": torch.cuda.max_memory_allocated() if dev.type == "cuda" else None,
           "peak_gpu_bytes_prefill": peak_prefill, "n_pages": len(imgs)}
    m.model.rope_deltas = None
    del out, cache
    return res


def load(a, dev):
    rope = json.loads(a.rope_scaling) if a.rope_scaling else None
    model, tok = Q.load_model(dev, tiny=a.tiny, rope_scaling=rope, seed=a.seed)
    tag, info = "base", None
    if a.adapter:
        st = torch.load(a.adapter, map_location="cpu", weights_only=False)
        lc = st["lora"]
        pm = Q.make_trainable(model, lc["r"], lc["alpha"], 0.0, lc["vision_blocks"], lc.get("train_merger", True))
        Q.load_trainable_state(pm, st["trainable"])
        model = pm.merge_and_unload()
        tag = Path(a.adapter).stem
        info = {"adapter": str(a.adapter), "adapter_encoding": st.get("encoding"), "adapter_steps": st.get("steps"),
                "adapter_fair_hash": st.get("fair_hash")}
    if dev.type == "cuda":
        model = model.to(torch.bfloat16)
    model.eval()
    return model, tok, tag, info, rope


def main(argv=None) -> int:
    a = parse(argv)
    dev = resolve_device(a.device)
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    out = Path(a.out_root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "runs.jsonl"
    done = set()
    if rows_path.exists():
        for l in rows_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done.add((r["input"], r["length_chars"], r["task"], r["model"], r["cond"], json.dumps(r["rope_scaling"])))
    model, tok, tag, ad_info, rope = load(a, dev)
    rinfo = Q.rope_info(model)
    held = text_stream_s("test", max(2_100_000, max(a.lengths) + 10_000))   # all of test + val
    print(f"[lc] model={tag} device={dev.type} held-out chars={len(held)} rope={rinfo} max_positions={a.max_positions}",
          flush=True)
    cfg = {k: v for k, v in vars(a).items() if k != "out_root"}
    for length in a.lengths:
        base_doc = T.make_document(held, length, np.random.default_rng([a.seed, length]))
        for task in a.tasks:
            items = []
            if task == "needle":
                for j, d in enumerate(a.depths):
                    doc, q, ans, at = T.insert_needle(base_doc, d, np.random.default_rng([a.seed, length, j]))
                    items.append({"doc": doc, "q": q, "ref": ans, "depth": d, "char_pos": at})
            else:
                r = np.random.default_rng([a.seed, length, 99])
                for _ in range(a.n_cloze):
                    s = T.doc_cloze(base_doc, r)
                    if s is not None:
                        items.append({"doc": base_doc, "q": s[0], "ref": s[1], **s[2]})
            for inp in a.inputs:
                key = (inp, length, task, tag, a.cond, json.dumps(rope))
                if key in done:
                    print(f"[lc] skip, done: {key}")
                    continue
                enc = None if inp == "text" else get_encoding_s(inp, S)
                row = {"phase": "longctx", "round": 3, "input": inp, "encoding": None if enc is None else inp,
                       "family": "text" if enc is None else enc.family, "length_chars": length, "task": task,
                       "model": tag, "cond": a.cond if enc is not None else "text", "rope_scaling": rope,
                       "rope": rinfo, "max_positions": a.max_positions, "device": dev.type, "tiny": a.tiny,
                       "base": None if a.tiny else f"{Q.QWEN_REPO}@{Q.QWEN_REV}", **(ad_info or {}),
                       "config": cfg, "smoke": "smoke" in a.run_name}
                if enc is not None:
                    row["cond"] = a.cond
                res, skip = [], None
                for it in items:
                    smp = make_sample(inp, it["doc"], it["q"], enc, a.cond)
                    npos = n_positions(tok, smp)
                    if npos > a.max_positions:
                        skip = (f"input needs {npos} positions > max_positions {a.max_positions} "
                                f"({len(smp['imgs'])} pages)" if smp["imgs"] else
                                f"input needs {npos} positions > max_positions {a.max_positions}")
                        row["positions_needed"] = npos
                        break
                    r = answer(model, tok, smp, dev, a.max_new_tokens, a.vision_batch)
                    r.update(ref=it["ref"], correct=T.answer_correct("needle" if task == "needle" else "cloze",
                                                                     it["ref"], r["hyp"]),
                             depth=it.get("depth"), char_pos=it.get("char_pos"), doc_chars=len(it["doc"]))
                    r.pop("gen_ids")
                    res.append(r)
                    print(f"[lc] {inp} len={length} {task} depth={it.get('depth')} pos={r['positions']} "
                          f"kv={r['kv_bytes'] / 1e6:.1f}MB prefill={r['prefill_s']:.2f}s ref={it['ref']!r} "
                          f"hyp={r['hyp'][:40]!r} {'OK' if r['correct'] else 'x'}", flush=True)
                if skip:
                    row.update(skipped=True, skip_reason=skip, n_items=0, acc=None)
                    print(f"[lc] SKIP {inp} len={length} {task}: {skip}", flush=True)
                else:
                    pos = [r["positions"] for r in res]
                    kv = [r["kv_bytes"] for r in res]
                    peaks = [r["peak_gpu_bytes"] for r in res if r["peak_gpu_bytes"] is not None]
                    row.update(skipped=False, n_items=len(res),
                               acc=float(np.mean([r["correct"] for r in res])) if res else None,
                               positions=int(np.mean(pos)) if pos else None, positions_max=max(pos, default=None),
                               kv_bytes=int(np.mean(kv)) if kv else None,
                               kv_bytes_per_position=(sum(kv) / sum(r["kv_positions"] for r in res)) if res else None,
                               kv_bytes_analytic=int(np.mean([r["kv_bytes_analytic"] for r in res])) if res else None,
                               chars_per_position=(length / np.mean(pos)) if pos else None,
                               n_pages=res[0]["n_pages"] if res else None,
                               rope_span_max=max((r["rope_span"] for r in res), default=None),
                               peak_gpu_bytes=max(peaks) if peaks else None,
                               prefill_s=float(np.mean([r["prefill_s"] for r in res])) if res else None,
                               vision_s=float(np.mean([r["vision_s"] for r in res])) if res else None,
                               items=res)
                    print(f"[lc] ROW {inp} len={length} {task} acc={row['acc']} positions={row['positions']} "
                          f"kv={row['kv_bytes'] / 1e9 if row['kv_bytes'] else 0:.3f}GB", flush=True)
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
