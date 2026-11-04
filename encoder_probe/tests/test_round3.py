"""Round 3 CPU checks: fine-tuning (finetune_qwen.py) and long-context (longctx_bench.py) code on a TINY random-init
Qwen2.5-VL (2 LLM layers, 6 vision blocks, byte tokenizer). No downloads, no GPU; ~1 min at 1 thread."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "encoder_probe" / "experiments"))

from encoder_probe.ep import qwen_ft as Q  # noqa: E402
from encoder_probe.ep import r3_tasks as T  # noqa: E402
from encoder_probe.ep.geom import get_encoding_s, make_pages_s  # noqa: E402

CPU = torch.device("cpu")
torch.set_num_threads(1)


@pytest.fixture(scope="module")
def pages():
    out = {}
    for name in ("latin_p24", "col_b4", "rgb3_p8"):
        enc = get_encoding_s(name, 448)
        out[name] = (enc, make_pages_s(enc, "test", 3))
    return out


@pytest.fixture(scope="module")
def tiny():
    return Q.load_model(CPU, tiny=True)


# ------------------------------------------------------------------------------------------ data / answers

def test_windows_cover_every_line_within_budget(pages):
    for enc, pgs in pages.values():
        for pg in pgs:
            for mx in (60, 300, 1500):
                w = T.windows(pg.refs, mx)
                assert w[0][0] == 0 and w[-1][1] == len(pg.refs)
                assert all(a < b and b == c for (a, b), (c, _) in zip(w, w[1:]))
                for a, b in w:
                    assert b - a == 1 or len(T.transcript(pg, a, b)) <= mx
                ev = T.eval_windows(pg, mx, 3)
                assert len(ev) == min(3, len(w)) and set(ev) <= set(w)


def test_answers_are_decoded_plain_text_not_class_labels(pages):
    enc, pgs = pages["rgb3_p8"]
    pg = pgs[0]
    assert pg.labels[0][0][0] in "RGB" and len(pg.labels[0][0]) == 2       # channel-tagged CTC classes
    p, ans, meta = T.transcribe_sample(pg, np.random.default_rng(0), 10 ** 6)
    assert p == T.TRANSCRIBE_ALL and ans == "\n".join(pg.refs)               # the original characters
    assert ans.replace("\n", "") == T.flat_text(enc, pg)                     # raw chunks: refs are source slices
    enc_l, pl = pages["latin_p24"]
    assert not T.is_raw_chunks(enc_l) and T.is_raw_chunks(pages["col_b4"][0])
    assert T.flat_text(enc_l, pl[0]) == " ".join(pl[0].refs)


def test_task_mix_and_question_answers(pages):
    rng = np.random.default_rng(1)
    kinds = {"transcribe": 0, "cloze": 0, "mc": 0, "copy": 0}
    for name in ("latin_p24", "col_b4"):
        enc, pgs = pages[name]
        flats = [T.flat_text(enc, p) for p in pgs]
        for i in range(300):
            k = i % len(pgs)
            prompt, ans, meta = T.page_sample(pgs[k], flats[k], flats, rng, 400, 0.6)
            kinds[meta["task"]] += 1
            flat = flats[k]
            if meta["task"] == "transcribe":
                a, b = meta["lines"]
                assert ans == T.transcript(pgs[k], a, b) and (b - a == 1 or len(ans) <= 400)
                assert prompt == T.transcribe_prompt(a, b, len(pgs[k].refs))
            elif meta["task"] == "cloze":
                ctx = meta["context"]
                assert flat.count(ctx) == 1 and flat[flat.index(ctx) + len(ctx):].split()[0] == ans
            elif meta["task"] == "copy":
                ctx = meta["context"]
                assert flat.count(ctx) == 1 and len(ans.split()) == meta["k"]
                assert " ".join(flat[flat.index(ctx) + len(ctx):].split()[:meta["k"]]) == ans
            else:
                opts = meta["options"]
                right = opts[T.LETTERS.index(ans)]
                assert right in flat and all(o not in flat for o in opts if o != right)
    n = sum(kinds.values())
    assert 0.5 < kinds["transcribe"] / n < 0.7, kinds
    assert min(kinds["cloze"], kinds["mc"], kinds["copy"]) > 0.08 * n, kinds


def test_answer_scoring():
    assert T.answer_correct("cloze", "Factory", "factory.") and not T.answer_correct("cloze", "in", "on in")
    assert T.answer_correct("mc", "C", " C) blah") and not T.answer_correct("mc", "C", "B")
    assert T.answer_correct("needle", "123456", "It is 123456.") and not T.answer_correct("needle", "123456", "1234567")


def test_needle_and_document_cloze():
    from encoder_probe.ep.geom import text_stream_s
    held = text_stream_s("test", 50_000)
    doc = T.make_document(held, 20_000, np.random.default_rng(0))
    assert 19_900 <= len(doc) <= 20_000 and doc in held
    for d in (0.0, 0.5, 1.0):
        out, q, ans, pos = T.insert_needle(doc, d, np.random.default_rng(3))
        assert out.count(f"is {ans}.") == 1 and ans in out and "secret number" in q
        at = out.index("The secret number")
        assert abs(at / len(out) - d) < 0.02
    q, w, meta = T.doc_cloze(doc, np.random.default_rng(0))
    assert doc.count(meta["context"]) == 1 and doc[meta["char_pos"] + len(meta["context"]):].split()[0] == w


def test_chat_ids_labels_and_collate(tiny):
    _, tok = tiny
    ids, lab, typ = Q.chat_ids(tok, "Read it.", n_images=2, n_img_tokens=256, answer="abc")
    assert sum(typ) == 512 and ids.count(tok.image_pad) == 512 and ids.count(tok.vision_start) == 2
    assert all(t == tok.image_pad for t, m in zip(ids, typ) if m == 1)
    ans = [t for t, l in zip(ids, lab) if l != -100]
    assert ans == tok.encode("abc") + [tok.im_end] and ids[-len(ans):] == ans
    assert tok.decode(ids[:-len(ans)]).endswith("Read it.\nassistant\n")
    img = np.full((448, 448, 3), 255, np.uint8)
    b = Q.collate([{"imgs": [img], "prompt": "a", "answer": "xy"}, {"imgs": [img], "prompt": "b"}], tok, CPU,
                  left_pad=True)
    assert b["pixel_values"].shape == (2 * 1024, 3 * 2 * 14 * 14) and b["image_grid_thw"].tolist() == [[1, 32, 32]] * 2
    L = b["input_ids"].shape[1]
    assert b["attention_mask"][1, -1] == 1 and b["attention_mask"][1, 0] == 0          # left padding
    assert (b["mm_token_type_ids"].sum(1) == 256).all() and b["input_ids"].shape == (2, L)


# --------------------------------------------------------------------------------------------- model / LoRA

def test_lora_wiring_only_intended_params_train(tiny):
    model, tok = Q.load_model(CPU, tiny=True)
    depth = len(model.model.visual.blocks)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    pm = Q.make_trainable(model, r=4, alpha=8, dropout=0.0, vision_blocks=4)
    tr = [n for n, p in pm.named_parameters() if p.requires_grad]
    lora = [n for n in tr if "lora_" in n]
    other = [n for n in tr if "lora_" not in n]
    layers = model.config.text_config.num_hidden_layers
    mods = Q.lora_module_names(pm)
    assert len(mods) == layers * 7 and len(lora) == 2 * len(mods)
    assert all(".language_model.layers." in m for m in mods)
    assert {m.rsplit(".", 1)[1] for m in mods} == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                   "down_proj"}
    keep = tuple(f"visual.blocks.{i}." for i in range(depth - 4, depth)) + ("visual.merger.",)
    assert other and all(any(k in n for k in keep) for n in other)
    vis_names = [n for n, _ in pm.named_parameters() if any(k in n for k in keep)]
    assert sorted(vis_names) == sorted(other)                                 # all of them, nothing else
    assert all(pm.get_parameter(n).dtype == torch.float32 for n in tr)
    # one optimizer step changes only trainable tensors
    img = (np.random.default_rng(0).random((448, 448, 3)) * 255).astype(np.uint8)
    b = Q.collate([{"imgs": [img], "prompt": "Transcribe.", "answer": "hello"}], tok, CPU)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-2)
    pm.train()
    loss, n = Q.lm_loss(pm, b)
    loss.backward()
    assert all(p.grad is not None for p in pm.parameters() if p.requires_grad)
    assert all(p.grad is None for p in pm.parameters() if not p.requires_grad)
    opt.step()
    after = dict(Q.hf_model(pm).named_parameters())
    for n, p0 in before.items():
        key = n.replace(".weight", ".base_layer.weight") if n.replace(".weight", ".base_layer.weight") in after else n
        if key in after and not any(k in key for k in keep):
            assert torch.equal(after[key].detach(), p0), key


def test_loss_falls_in_20_steps():
    model, tok = Q.load_model(CPU, tiny=True)
    pm = Q.make_trainable(model)
    Q.hf_model(pm).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc = get_encoding_s("latin_p24", 448)
    pg = make_pages_s(enc, "train", 1)[0]
    b = Q.collate([{"imgs": [pg.img], "prompt": T.TRANSCRIBE_ALL, "answer": pg.refs[0][:40]},
                   {"imgs": [pg.img], "prompt": "What word follows?", "answer": "the"}], tok, CPU)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-2)
    pm.train()
    losses = []
    for _ in range(20):
        loss, _ = Q.lm_loss(pm, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.2 * losses[0] and losses[-1] < min(losses[:5]), losses


def _spiky(model):
    """Larger output weights so a random-init model makes non-degenerate greedy choices."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        e = Q.hf_model(model).model.language_model.embed_tokens.weight
        e.copy_(torch.randn(e.shape, generator=g) * 2.0)
    return model


def test_generate_and_cached_decode_match_full_recompute():
    from longctx_bench import answer
    model, tok = Q.load_model(CPU, tiny=True)
    _spiky(model).eval()
    enc = get_encoding_s("col_b4", 448)
    pgs = make_pages_s(enc, "test", 2)
    smp = [{"imgs": [pgs[0].img], "prompt": "Transcribe."}, {"imgs": [pgs[1].img], "prompt": "Q: what is a b c?"}]
    gen = Q.generate(model, tok, smp, CPU, 6)                                   # batched, left padded
    for s, g in zip(smp, gen):
        # reference: no cache, full forward per new token (positions computed from scratch every time)
        b = Q.collate([{**s, "answer": None}], tok, CPU)
        ids = b["input_ids"]
        typ = b["mm_token_type_ids"]
        out = []
        for _ in range(6):
            model.model.rope_deltas = None
            with torch.no_grad():
                lg = model(input_ids=ids, mm_token_type_ids=typ, pixel_values=b["pixel_values"],
                           image_grid_thw=b["image_grid_thw"], use_cache=False).logits[0, -1]
            t = int(lg.argmax())
            if t in (tok.im_end, tok.eos):
                break
            out.append(t)
            ids = torch.cat([ids, torch.tensor([[t]])], 1)
            typ = torch.cat([typ, torch.zeros((1, 1), dtype=typ.dtype)], 1)
        assert g == tok.decode(out)
        r = answer(model, tok, s, CPU, 6)                                       # longctx manual prefill + decode
        assert r["gen_ids"] == out and len(set(out)) > 1


# ------------------------------------------------------------------------------------------------ KV bytes

def test_kv_bytes_match_analytic_and_real_config():
    from longctx_bench import answer
    model, tok = Q.load_model(CPU, tiny=True)
    model.eval()
    enc = get_encoding_s("latin_p24", 448)
    pgs = make_pages_s(enc, "test", 2)
    for s in ({"imgs": [p.img for p in pgs], "prompt": "Which word?"},
              {"imgs": [], "doc_text": "some document text " * 20, "prompt": "Which word?"}):
        r = answer(model, tok, s, CPU, 3)
        t = model.config.text_config
        hd = t.hidden_size // t.num_attention_heads
        assert r["kv_positions"] == r["positions"]
        assert r["kv_bytes"] == t.num_hidden_layers * 2 * t.num_key_value_heads * hd * r["positions"] * 4
        assert r["kv_bytes"] == r["kv_bytes_analytic"]
        if s["imgs"]:
            assert r["positions"] > 512 and r["rope_span"] < r["positions"]   # M-RoPE: 16 rope steps per image
    # Qwen2.5-VL-3B-Instruct (config.json at the pinned revision): 36 layers, 2 KV heads, head_dim 2048/16 = 128
    from transformers import Qwen2_5_VLConfig
    real = Qwen2_5_VLConfig(text_config=dict(num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2,
                                             hidden_size=2048, intermediate_size=11008))
    assert Q.kv_bytes_analytic(real, 1, bytes_per=2) == 36_864                  # bf16 bytes per position
    assert abs(Q.kv_bytes_analytic(real, 32_768, bytes_per=2) / 1e9 - 1.21) < 0.01


# ------------------------------------------------------------------------------ scripts end to end (tiny)

FT = ["--tiny", "--device", "cpu", "--steps", "4", "--train-pages", "3", "--test-pages", "2", "--batch-size", "2",
      "--max-target-chars", "80", "--max-new-tokens", "24", "--eval-windows", "1", "--eval-cloze", "1",
      "--eval-mc", "1", "--val-samples", "2", "--eval-every", "2", "--ckpt-every", "2", "--log-every", "100"]


def _ft(tmp, run, enc, *extra):
    from finetune_qwen import main
    return main(["--encoding", enc, *FT, "--run-name", run, "--out-root", str(tmp / "res"),
                 "--cache-root", str(tmp / "cache"), *extra])


def test_finetune_resume_is_exact_and_rows(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _ft(a, "r", "latin_p24") == 0
    assert _ft(b, "r", "latin_p24", "--stop-after", "2") == 0                  # "crash" after the step-2 checkpoint
    ck = b / "cache" / "ckpt" / "r3ft__r__latin_p24__4.pt"
    assert ck.exists() and not (b / "res" / "r" / "runs.jsonl").exists()
    assert _ft(b, "r", "latin_p24") == 0                                       # resumes from step 2
    assert not ck.exists()
    sa = torch.load(a / "cache" / "r3_adapters" / "r__latin_p24__4.pt", weights_only=False)
    sb = torch.load(b / "cache" / "r3_adapters" / "r__latin_p24__4.pt", weights_only=False)
    assert sa["trainable"].keys() == sb["trainable"].keys()
    for k in sa["trainable"]:
        assert torch.equal(sa["trainable"][k], sb["trainable"][k]), k
    ra = [json.loads(l) for l in (a / "res" / "r" / "runs.jsonl").read_text().splitlines()]
    rb = [json.loads(l) for l in (b / "res" / "r" / "runs.jsonl").read_text().splitlines()]
    assert [r["cond"] for r in ra] == ["clean", "jpeg75"]
    assert [(r["cer"], r["curve"]) for r in ra] == [(r["cer"], r["curve"]) for r in rb]
    for r in ra:
        for k in ("encoding", "cond", "steps", "phase", "round", "family", "S", "P", "G", "n_tokens", "px_per_char",
                  "budget", "chars_per_token", "chars_per_vlm_token", "cer", "n_eval_chars", "cloze_acc", "mc_acc",
                  "device", "config", "fair", "fair_hash", "params", "train_pages", "test_pages", "digest_test",
                  "curve", "smoke", "adapter"):
            assert k in r, k
        assert r["phase"] == "finetune" and r["round"] == 3 and r["device"] == "cpu" and r["n_tokens"] == 256
        assert abs(r["chars_per_token"] * r["px_per_char"] - 784) < 1e-6        # 28 x 28 px per Qwen token
        assert r["budget"] == round(r["px_per_char"], 3) and 0 <= r["cer"] and r["n_eval_chars"] > 0
        assert r["config"]["encoding"] == "latin_p24" and r["fair"]["lora_r"] == 16
        assert r["fair"]["vision_blocks"] == 4 and r["params"]["lora"] > 0
    assert _ft(a, "r", "latin_p24") == 0                                       # done -> skipped, no new rows
    assert len((a / "res" / "r" / "runs.jsonl").read_text().splitlines()) == 2
    # fairness guard: a second arm with identical settings gets the identical fair config / hash
    assert _ft(a, "r", "col_b4", "--eval-conds", "clean") == 0
    rows = [json.loads(l) for l in (a / "res" / "r" / "runs.jsonl").read_text().splitlines()]
    assert rows[-1]["encoding"] == "col_b4" and rows[-1]["fair"] == ra[0]["fair"]
    assert rows[-1]["fair_hash"] == ra[0]["fair_hash"] and rows[-1]["chars_per_token"] > 10 * ra[0]["chars_per_token"]
    # the bench loads the adapter (LoRA merged) and runs
    from longctx_bench import main as lc
    assert lc(["--tiny", "--device", "cpu", "--inputs", "latin_p24", "--lengths", "500", "--tasks", "needle",
               "--depths", "0.5", "--adapter", str(a / "cache" / "r3_adapters" / "r__latin_p24__4.pt"),
               "--run-name", "lc", "--out-root", str(a / "res")]) == 0
    row = json.loads((a / "res" / "lc" / "runs.jsonl").read_text())
    assert row["model"] == "r__latin_p24__4" and row["adapter_fair_hash"] == ra[0]["fair_hash"]


def test_longctx_rows_skip_and_rope_setting(tmp_path):
    from longctx_bench import main
    rope = '{"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 2048}'
    assert main(["--tiny", "--device", "cpu", "--inputs", "text", "latin_p24", "col_b4", "--lengths", "600", "3000",
                 "--depths", "0", "1", "--n-cloze", "2", "--max-positions", "2500", "--rope-scaling", rope,
                 "--run-name", "lc", "--out-root", str(tmp_path)]) == 0
    rows = [json.loads(l) for l in (tmp_path / "lc" / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 2 * 3
    for r in rows:
        assert r["rope_scaling"] == json.loads(rope) and r["rope"]["rope_parameters"]["rope_type"] == "yarn"
        assert r["max_positions"] == 2500 and r["phase"] == "longctx" and r["round"] == 3
        if r["input"] == "text" and r["length_chars"] == 3000:
            assert r["skipped"] and "max_positions 2500" in r["skip_reason"] and r["positions_needed"] > 2500
            continue
        assert not r["skipped"] and r["n_items"] == 2 and 0 <= r["acc"] <= 1
        assert r["kv_bytes"] == r["kv_bytes_analytic"] and r["kv_bytes_per_position"] == 2 * 2 * 2 * 16 * 4
        assert r["positions"] <= 2500 and r["chars_per_position"] > 0 and r["prefill_s"] >= 0
        for it in r["items"]:
            assert {"ref", "hyp", "correct", "positions", "kv_bytes", "rope_span", "prefill_s"} <= set(it)
        if r["input"] != "text":
            assert r["n_pages"] >= 1 and r["cond"] == "clean"
    byk = {(r["input"], r["length_chars"], r["task"]): r for r in rows}
    # at equal document length the dense code needs far fewer positions than Latin pages and text
    assert byk[("col_b4", 3000, "needle")]["positions"] < byk[("latin_p24", 3000, "needle")]["positions"]
    assert byk[("latin_p24", 600, "needle")]["positions"] < byk[("text", 600, "needle")]["positions"]
