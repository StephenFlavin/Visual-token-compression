"""Colour gist, CPU track (docs/COLOUR_GIST.md "S1"): a small VLM whose vision encoder is trained FROM SCRATCH.

Why: in Qwen2.5-VL the pretrained ViT keeps both colour streams to block 16, then its pretrained letter/word features
blend them (A1-cpu diagnostics). An encoder with no pretrained text features is the fairest test of the overlay
itself, and it trains on a CPU.

* Pages: the A1 stream encodings (rgb<k>s_p<p>, one Latin page per colour channel) at S = 224 px: 64 vision tokens
  of 28 x 28 px (the same token size as Qwen2.5-VL, so chars per token compare). 16 px: ~6.5 chars/token per stream.
* Encoder (~4M params, fp32, random init): conv stem 7x7/7 -> 3x3/2 -> 3x3/2 (224 -> 32 -> 16 -> 8 x 8 tokens),
  learned 2-D positions, 4 pre-norm transformer layers (d 256), then LayerNorm + Linear to the LLM width and a gain
  matched to the token-embedding scale.
* LLM: Qwen2.5-0.5B-Instruct, frozen, LoRA r = 16 on every attention/MLP projection. Chat template:
  <|im_start|>user\\n [64 vision embeddings] \\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>.
* Tasks: ep/r3_tasks.py stream tasks (per-stream transcription, cloze, MC with the other stream's phrase as a lure,
  copy); loss on answer tokens only.
* Val (fixed, same targets in the real and the blind run): one whole-stream transcription per test page and stream;
  ``val_per_sample`` = [(loss sum, tokens)] -> analyse_colour.py (info per vision token, R(k) with a bootstrap CI).
* Eval: per stream MC (+ lure rate), cloze, greedy transcription CER.
* --blind-images: page images deranged (train and test), same texts: the arm's blind control (run name *_blind).

    python3 encoder_probe/experiments/small_vlm.py --encoding rgb2s_p16 --smoke            # timing
    python3 encoder_probe/experiments/small_vlm.py --encoding rgb2s_p16 --steps 2000 --run-name s1
    python3 encoder_probe/experiments/analyse_colour.py --real s1 --pitch 16 --equal-density rgb2s_p16 rgb1s_p12
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from encoder_probe.ep import qwen_ft as Q  # noqa: E402
from encoder_probe.ep import r3_tasks as T  # noqa: E402
from encoder_probe.ep.encoders import resolve_device  # noqa: E402
from encoder_probe.ep.geom import get_encoding_s, make_pages_s  # noqa: E402

LLM = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_TARGETS = r".*layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
P = 28
FAIR_KEYS = ["S", "steps", "batch_size", "lr", "lr_enc", "warmup", "lora_r", "enc_dim", "enc_layers", "train_pages",
             "test_pages", "task_mix", "max_target_chars", "eval_mc", "eval_cloze", "seed", "tiny", "gist",
             "gist_words"]


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--S", type=int, default=224, help="page side (multiple of 28)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA")
    ap.add_argument("--lr-enc", type=float, default=1e-3, help="the from-scratch encoder")
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--enc-dim", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--train-pages", type=int, default=3000)
    ap.add_argument("--test-pages", type=int, default=24)
    ap.add_argument("--task-mix", type=float, default=0.4, help="share of transcription samples (S1 amendment 1: "
                    "0.4, gist-weighted)")
    ap.add_argument("--no-gist", dest="gist", action="store_false",
                    help="drop the gist word-presence task from training (questions: cloze, mc, copy only)")
    ap.add_argument("--gist-words", type=int, default=8, help="eval: present / absent / lure words per page+stream")
    ap.add_argument("--max-target-chars", type=int, default=500)
    ap.add_argument("--eval-mc", type=int, default=4, help="MC questions per test page and stream")
    ap.add_argument("--eval-cloze", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind-images", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run-name", default="s1")
    ap.add_argument("--out-root", default=str(ROOT / "encoder_probe" / "results"))
    ap.add_argument("--cache-root", default=str(ROOT / "encoder_probe" / "cache"))
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="tiny random LLM + byte tokenizer (CPU tests)")
    ap.add_argument("--smoke", action="store_true", help="30 steps, small eval: timing")
    a = ap.parse_args(argv)
    if a.smoke:
        a.steps, a.train_pages, a.test_pages, a.eval_every, a.ckpt_every = 30, 64, 4, 30, 0
        a.eval_mc, a.eval_cloze = 1, 1
        a.run_name += "_smoke"
    if a.blind_images:
        a.run_name += "_blind"
    return a


# ------------------------------------------------------------------------------------------------- models

class PageEncoder(nn.Module):
    """224 x 224 x 3 page -> (S/28)^2 tokens in the LLM's embedding space. Random init: no pretrained features."""

    def __init__(self, S: int, d: int, layers: int, out_dim: int):
        super().__init__()
        assert S % P == 0
        self.g = S // P
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 7), nn.GELU(), nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(128, d, 3, 2, 1))
        self.pos = nn.Parameter(torch.randn(1, self.g * self.g, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim))
        self.gain = nn.Parameter(torch.ones(()))

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:          # imgs uint8 (B, S, S, 3)
        x = imgs.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        x = self.stem(x).flatten(2).transpose(1, 2)                    # (B, g*g, d)
        assert x.shape[1] == self.g * self.g, x.shape
        return self.out(self.tf(x + self.pos)) * self.gain


def load_llm(dev, tiny: bool, seed: int):
    if tiny:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        tok = Q.ByteTok()
        g = torch.random.get_rng_state()
        torch.manual_seed(seed)
        m = Qwen2ForCausalLM(Qwen2Config(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128,
                                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                         max_position_embeddings=4096, tie_word_embeddings=True,
                                         initializer_range=0.3, attn_implementation="sdpa"))
        torch.random.set_rng_state(g)
        return m.float().to(dev), tok
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = AutoTokenizer.from_pretrained(LLM)
    tok = Q.HFTok(hf)
    tok.name = LLM
    dtype = torch.bfloat16 if dev.type == "cuda" or Q_cpu_bf16() else torch.float32
    return AutoModelForCausalLM.from_pretrained(LLM, dtype=dtype, attn_implementation="sdpa").to(dev), tok


def Q_cpu_bf16() -> bool:
    from encoder_probe.ep.vlm_encoders import cpu_has_bf16
    return cpu_has_bf16()


class SmallVLM:
    def __init__(self, a, dev):
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(a.seed)
        llm, self.tok = load_llm(dev, a.tiny, a.seed)
        for p in llm.parameters():
            p.requires_grad_(False)
        self.lm = get_peft_model(llm, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                                                 target_modules=LORA_TARGETS, bias="none"))
        for p in self.lm.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.lm.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.emb = self.lm.get_input_embeddings()
        self.head = self.lm.get_output_embeddings()
        width = self.emb.weight.shape[1]
        self.enc = PageEncoder(a.S, a.enc_dim, a.enc_layers, width).to(dev)
        with torch.no_grad():   # vision tokens start at the scale of real token embeddings (per-vector RMS)
            self.enc.gain.fill_(float(self.emb.weight.float().pow(2).mean(1).sqrt().mean()))
        self.dev, self.n_vis = dev, (a.S // P) ** 2
        self.autocast = dev.type == "cuda" or (not a.tiny and Q_cpu_bf16())

    def ac(self):
        return torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.autocast)

    def ids(self, prompt: str, answer: str | None):
        t, e = self.tok, self.tok.encode
        pre = [t.im_start] + e("user\n")
        post = e("\n" + prompt) + [t.im_end] + e("\n") + [t.im_start] + e("assistant\n")
        ans = (e(answer) + [t.im_end]) if answer is not None else []
        ids = pre + [t.image_pad] * self.n_vis + post + ans
        labels = [-100] * (len(pre) + self.n_vis + len(post)) + ans
        return ids, labels, len(pre)

    def embed(self, imgs: list[np.ndarray], seqs: list[list[int]], vis_at: list[int], left: bool = False):
        L = max(len(s) for s in seqs)
        B = len(seqs)
        ids = torch.full((B, L), self.tok.pad, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        offs = []
        for i, s in enumerate(seqs):
            o = L - len(s) if left else 0
            ids[i, o:o + len(s)] = torch.tensor(s)
            att[i, o:o + len(s)] = 1
            offs.append(o)
        ids, att = ids.to(self.dev), att.to(self.dev)
        with self.ac():
            x = self.emb(ids)
            v = self.enc(torch.from_numpy(np.stack(imgs)).to(self.dev)).to(x.dtype)
            rows = []
            for i in range(B):
                a0 = offs[i] + vis_at[i]
                rows.append(torch.cat([x[i, :a0], v[i], x[i, a0 + self.n_vis:]], 0))
            x = torch.stack(rows)
        return x, att, offs

    def loss(self, samples: list[dict]) -> tuple[torch.Tensor, int, list[tuple[float, int]]]:
        """Mean CE over answer tokens; also per-sample (loss sum, tokens)."""
        built = [self.ids(s["prompt"], s["answer"]) for s in samples]
        x, att, _ = self.embed([s["img"] for s in samples], [b[0] for b in built], [b[2] for b in built])
        L = x.shape[1]
        lab = torch.full((len(built), L), -100, dtype=torch.long)
        for i, b in enumerate(built):
            lab[i, :len(b[1])] = torch.tensor(b[1])
        lab = lab.to(self.dev)
        with self.ac():
            h = self.lm.base_model.model.model(inputs_embeds=x, attention_mask=att, use_cache=False).last_hidden_state
        h, y = h[:, :-1], lab[:, 1:]
        sel = y != -100
        ce = F.cross_entropy(self.head(h[sel]).float(), y[sel], reduction="none")
        per, k = [], 0
        for i in range(len(built)):
            n = int(sel[i].sum())
            per.append((float(ce[k:k + n].detach().sum()), n))
            k += n
        return ce.mean(), int(sel.sum()), per

    @torch.no_grad()
    def generate(self, img: np.ndarray, prompt: str, max_new: int) -> str:
        ids, _, at = self.ids(prompt, None)
        x, att, _ = self.embed([img], [ids], [at])
        out = []
        with self.ac():
            o = self.lm.base_model.model(inputs_embeds=x, attention_mask=att, use_cache=True)
            past, logits = o.past_key_values, o.logits[:, -1]
            for _ in range(max_new):
                nxt = int(logits.float().argmax(-1))
                if nxt in (self.tok.im_end, self.tok.eos):
                    break
                out.append(nxt)
                o = self.lm.base_model.model(input_ids=torch.tensor([[nxt]], device=self.dev), past_key_values=past,
                                             use_cache=True)
                past, logits = o.past_key_values, o.logits[:, -1]
        return self.tok.decode(out)

    @torch.no_grad()
    def yes_no(self, samples: list[dict]) -> list[float]:
        """log p(yes) - log p(no) at the first answer position (gist word presence, no generation)."""
        y, n = self.tok.encode("yes")[0], self.tok.encode("no")[0]
        built = [self.ids(s["prompt"], None) for s in samples]
        x, att, _ = self.embed([s["img"] for s in samples], [b[0] for b in built], [b[2] for b in built], left=True)
        with self.ac():
            h = self.lm.base_model.model.model(inputs_embeds=x, attention_mask=att, use_cache=False).last_hidden_state
        lp = torch.log_softmax(self.head(h[:, -1]).float(), -1)
        return (lp[:, y] - lp[:, n]).tolist()

    def params(self):
        return [p for p in self.lm.parameters() if p.requires_grad], list(self.enc.parameters())


# ------------------------------------------------------------------------------------------------ helpers

def derange(pages, rng):
    n = len(pages)
    while True:
        perm = rng.permutation(n)
        if n < 2 or not np.any(perm == np.arange(n)):
            break
    imgs = [pages[j].img for j in perm]
    for pg, im in zip(pages, imgs):
        pg.img = im
    return perm.tolist()


def val_set(te, max_chars: int) -> list[dict]:
    out = []
    for pg in te:
        for s, lines in enumerate(pg.streams):
            lo, hi = 0, len(lines)
            if len("\n".join(lines)) > max_chars:
                lo, hi = T.window_from(lines, 0, max_chars)
            out.append({"img": pg.img, "prompt": T.stream_transcribe_prompt(pg.colours[s], lo, hi, len(lines)),
                        "answer": "\n".join(lines[lo:hi]), "stream": s})
    return out


def lr_lambda(steps: int, warmup: float):
    w = max(1, int(round(warmup * steps)))
    return lambda i: (i + 1) / w if i < w else 0.5 * (1 + math.cos(math.pi * min(1.0, (i - w) / max(1, steps - w))))


def main(argv=None) -> int:
    a = parse(argv)
    dev = resolve_device(a.device)
    torch.set_num_threads(a.threads)
    torch.set_flush_denormal(True)
    out = Path(a.out_root) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    rows_f = out / "runs.jsonl"
    if rows_f.exists() and any(json.loads(x).get("encoding") == a.encoding and json.loads(x).get("steps") == a.steps
                               for x in rows_f.read_text().splitlines() if x.strip()):
        print(f"[s1] skip, done: {a.encoding} {a.steps}")
        return 0
    cfg = {k: v for k, v in vars(a).items() if k not in ("out_root", "cache_root", "stop_after")}
    fair_hash = Q.config_hash({k: cfg[k] for k in FAIR_KEYS})
    enc = get_encoding_s(a.encoding, a.S)
    tr = make_pages_s(enc, "train", a.train_pages)
    te = make_pages_s(enc, "test", a.test_pages, allow_fewer=True)
    chars_te = sum(p.n_chars for p in te)
    cpt = chars_te / (len(te) * (a.S // P) ** 2)
    blind = None
    if a.blind_images:
        brng = np.random.default_rng([a.seed, 7919])
        blind = {"perm_test": derange(te, brng), "perm_train_head": derange(tr, brng)[:20]}
    flats_tr = [" ".join(p.refs) for p in tr]
    flats_te = [" ".join(p.refs) for p in te]
    m = SmallVLM(a, dev)
    lora_p, enc_p = m.params()
    opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr}, {"params": enc_p, "lr": a.lr_enc}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.steps, a.warmup))
    rng = np.random.default_rng(a.seed)
    val = val_set(te, a.max_target_chars)
    print(f"[s1] {a.encoding} S={a.S} chars/token={cpt:.1f} train={len(tr)} test={len(te)} val={len(val)} "
          f"lora={sum(p.numel() for p in lora_p)} enc={sum(p.numel() for p in enc_p)} device={dev.type} "
          f"bf16={m.autocast} fair_hash={fair_hash} blind={a.blind_images}", flush=True)

    # eval-time blind: every val item gets the image of the NEXT test page (same targets), for the learnability
    # check and an interim R; the proper blind is the separately trained --blind-images run
    pages_of = [next(k for k, pg in enumerate(te) if pg.img is v["img"]) for v in val]
    val_swap = [{**v, "img": te[(k + 1) % len(te)].img} for v, k in zip(val, pages_of)]

    def val_loss(per_sample=False, swap=False):
        m.lm.eval()
        m.enc.eval()
        res = []
        vs = val_swap if swap else val
        with torch.no_grad():
            for i in range(0, len(vs), 8):
                res += m.loss(vs[i:i + 8])[2]
        m.lm.train()
        m.enc.train()
        tot = sum(x for x, _ in res) / max(1, sum(n for _, n in res))
        return (tot, res) if per_sample else tot

    ck = Path(a.cache_root) / "ckpt" / f"s1__{a.run_name}__{a.encoding}__{a.steps}.pt"
    curve, start, t_prev, ema = [], 1, 0.0, None
    params = lora_p + enc_p
    if a.ckpt_every and ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        for p, v in zip(params, st["params"]):
            p.data.copy_(v.to(p.device, p.dtype))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        curve, start, t_prev, ema = st["curve"], st["step"] + 1, st["t_train"], st["ema"]
        print(f"[s1] resumed from step {st['step']}", flush=True)
    t0 = time.time() - t_prev
    for step in range(start, a.steps + 1):
        batch = []
        for k in rng.choice(len(tr), a.batch_size, replace=len(tr) < a.batch_size):
            prompt, ans, _ = T.stream_page_sample(tr[k], flats_tr, rng, a.max_target_chars, a.task_mix,
                                                  ("cloze", "mc", "copy", "word") if a.gist else ("cloze", "mc", "copy"))
            batch.append({"img": tr[k].img, "prompt": prompt, "answer": ans})
        opt.zero_grad(set_to_none=True)
        loss, _, _ = m.loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        lv = loss.item()
        ema = lv if ema is None else 0.95 * ema + 0.05 * lv
        if step % a.log_every == 0 or step == 1:
            print(f"[s1] step {step}/{a.steps} loss {lv:.4f} ema {ema:.4f} ({(time.time() - t0) / step:.2f} s/step)",
                  flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            vl = val_loss()
            curve.append((step, round(ema, 5), round(vl, 5)))
            print(f"[s1] step {step} val_loss {vl:.4f}", flush=True)
        if a.ckpt_every and step % a.ckpt_every == 0 and step < a.steps:
            ck.parent.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            torch.save({"params": [p.detach().cpu() for p in params], "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "curve": curve, "ema": ema, "step": step,
                        "t_train": time.time() - t0}, tmp)
            tmp.replace(ck)
        if a.stop_after and step >= a.stop_after:
            return 0
    train_s = time.time() - t0
    fin = Path(a.cache_root) / "s1_models" / f"{a.run_name}__{a.encoding}__{a.steps}.pt"
    fin.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora": [p.detach().cpu() for p in lora_p], "encoder": m.enc.state_dict(), "config": cfg}, fin)
    vl, vps = val_loss(per_sample=True)
    vsl, vsps = val_loss(per_sample=True, swap=True)
    print(f"[s1] val {vl:.4f} vs swapped images {vsl:.4f} (gap {vsl - vl:.3f} nats/token)", flush=True)

    # eval: per-stream MC (+ lure), cloze, greedy transcription CER
    from rapidfuzz.distance import Levenshtein
    m.lm.eval()
    m.enc.eval()
    per = {}
    t1 = time.time()
    for k, pg in enumerate(te):
        r = np.random.default_rng([a.seed, 1000, k])
        others = flats_te[:k] + flats_te[k + 1:]
        for s in range(len(pg.streams)):
            c = per.setdefault(s, {"mc": [], "lure": [], "cloze": [], "errs": 0, "chars": 0})
            for _ in range(a.eval_mc):
                q = T.stream_mc_sample(pg, s, others, r)
                if q:
                    h = m.generate(pg.img, q[0], 4)
                    c["mc"].append(T.answer_correct("mc", q[1], h))
                    c["lure"].append(T.answer_correct("mc", q[2]["lure"], h))
            for _ in range(a.eval_cloze):
                q = T.stream_cloze_sample(pg, s, r)
                if q:
                    c["cloze"].append(T.answer_correct("cloze", q[1], m.generate(pg.img, q[0], 8)))
            v = next(x for x in val if x["img"] is pg.img and x["stream"] == s)
            h = m.generate(pg.img, v["prompt"], int(1.4 * len(m.tok.encode(v["answer"]))) + 8)
            c["errs"] += Levenshtein.distance(v["answer"], h)
            c["chars"] += len(v["answer"])
    # gist: word presence per page and stream (scores = log p(yes) - log p(no)); analysed in analyse_colour.py
    gist_pages = []
    for k, pg in enumerate(te):
        others = flats_te[:k] + flats_te[k + 1:]
        for s in range(len(pg.streams)):
            r = np.random.default_rng([a.seed, 2000, k, s])
            entry = {"page": k, "stream": s, "n_cw": len(T.content_words(T.stream_flat(pg, s)))}
            for kind in ("present", "absent") + (("lure",) if len(pg.streams) > 1 else ()):
                qs = [q for q in (T.stream_word_sample(pg, s, others, r, kind) for _ in range(a.gist_words)) if q]
                entry[kind] = m.yes_no([{"img": pg.img, "prompt": q[0]} for q in qs]) if qs else []
            gist_pages.append(entry)
    per_stream = [{"colour": te[0].colours[s], "cer": c["errs"] / max(1, c["chars"]), "n_chars": c["chars"],
                   "mc_acc": float(np.mean(c["mc"])) if c["mc"] else None, "n_mc": len(c["mc"]),
                   "lure_rate": float(np.mean(c["lure"])) if c["lure"] else None,
                   "cloze_acc": float(np.mean(c["cloze"])) if c["cloze"] else None, "n_cloze": len(c["cloze"])}
                  for s, c in sorted(per.items())]
    allmc = [x for c in per.values() for x in c["mc"]]
    row = {"phase": "small_vlm", "encoding": a.encoding, "cond": "clean", "steps": a.steps, "S": a.S,
           "n_tokens": (a.S // P) ** 2, "chars_per_token": cpt, "train_pages": len(tr), "test_pages": len(te),
           "cer": sum(c["errs"] for c in per.values()) / max(1, sum(c["chars"] for c in per.values())),
           "mc_acc": float(np.mean(allmc)) if allmc else None, "per_stream": per_stream, "val_loss": vl,
           "val_per_sample": vps, "val_swap_loss": vsl,
           "val_swap_per_sample": vsps, "gist_pages": gist_pages, "curve": curve, "s_per_step": train_s / max(1, a.steps - start + 1),
           "train_time_s": train_s, "eval_time_s": time.time() - t1, "device": str(dev), "bf16": m.autocast,
           "llm": "tiny" if a.tiny else LLM, "config": cfg, "fair_hash": fair_hash, "blind_images": blind,
           "smoke": a.smoke, "tiny": a.tiny}
    with open(rows_f, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[s1] TEST {a.encoding}: val {vl:.4f} cer {row['cer']:.3f} mc {row['mc_acc']} per stream "
          f"{[(p['colour'], p['mc_acc'], p['lure_rate']) for p in per_stream]}", flush=True)
    if ck.exists():
        ck.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
