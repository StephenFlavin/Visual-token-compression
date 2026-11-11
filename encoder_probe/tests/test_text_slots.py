"""Test B (text_slots.py) on the TINY random-init Qwen2.5-VL: CPU, no downloads."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "encoder_probe" / "experiments"))
import analyse_slots as AN  # noqa: E402
import text_slots as TS  # noqa: E402

CPU = torch.device("cpu")
torch.set_num_threads(1)


def setup(ratio=4, decoder="lora", layers=1):
    a = TS.parse(["--ratio", str(ratio), "--tiny", "--device", "cpu", "--seq-tokens", "32", "--decoder", decoder,
                  "--enc-layers", str(layers)])
    pm, tok, sw = TS.build(a, CPU)
    rd = TS.Reader(pm, tok, 32 // ratio, CPU, False, sw)
    torch.manual_seed(0)
    enc = TS.LocalSlots(ratio, rd.emb.weight.shape[1], layers, 32)
    ids = torch.randint(0, 256, (2, 32))
    return pm, tok, sw, rd, enc, ids


def test_ratio_1_slots_are_the_token_embeddings():
    pm, tok, sw, rd, enc, ids = setup(ratio=1)
    with torch.no_grad():
        assert torch.equal(rd.encode(ids, enc), rd.emb(ids).float())


def test_initial_slots_are_chunk_means_and_local():
    pm, tok, sw, rd, enc, ids = setup(ratio=4, layers=0)
    with torch.no_grad():
        mem = rd.encode(ids, enc)
        assert torch.allclose(mem, rd.emb(ids).float().reshape(2, 8, 4, -1).mean(2), atol=1e-6)
        ids2 = ids.clone()
        ids2[:, 5] = (ids2[:, 5] + 1) % 256            # a token of chunk 1 only
        changed = (rd.encode(ids2, enc) - mem).abs().sum(-1) > 0
    assert changed[:, 1].all() and not changed[:, [0, 2, 3, 4, 5, 6, 7]].any()


def test_warm_start_is_the_mean_of_adjacent_old_slots():
    pm, tok, sw, rd, old, ids = setup(ratio=2)
    with torch.no_grad():
        old.pool.weight.add_(torch.randn_like(old.pool.weight) * 0.05)
        old.out.weight.normal_(0, 0.05)
        new = TS.LocalSlots(4, old.d, 1, 32)
        new.load_from(old.state_dict(), 2)
        want = rd.encode(ids, old).reshape(2, 8, 2, -1).mean(2)
        assert torch.allclose(rd.encode(ids, new), want, atol=1e-5)


def test_reader_uses_lora_and_prior_does_not():
    pm, tok, sw, rd, enc, ids = setup()
    pm.eval()
    with torch.no_grad():
        mem = rd.encode(ids, enc)
        base = rd.window_losses(ids, mem), rd.window_losses(ids, rd.empty(2), lora_off=True)
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
        moved = rd.window_losses(ids, mem), rd.window_losses(ids, rd.empty(2), lora_off=True)
    assert not np.allclose(base[0], moved[0], atol=1e-4)      # the reader's LoRA is on
    assert np.allclose(base[1], moved[1], atol=1e-6)          # the prior is the unmodified LLM
    assert rd.dec_off is False and sw.off is False


def test_frozen_decoder_trains_only_the_encoder():
    pm, tok, sw, rd, enc, ids = setup(decoder="frozen")
    assert not any(p.requires_grad for p in pm.parameters())
    loss, _ = rd.recon_loss(ids, rd.encode(ids, enc))
    loss.backward()
    assert enc.pool.weight.grad.abs().sum() > 0


def test_slots_learn_a_fixed_batch():
    pm, tok, sw, rd, enc, ids = setup()
    pm.eval()
    params = [p for p in pm.parameters() if p.requires_grad] + list(enc.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    first = None
    for _ in range(40):
        opt.zero_grad()
        loss, _ = rd.recon_loss(ids, rd.encode(ids, enc))
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.8 * first


def test_shuffled_window_keeps_the_words():
    tok = TS.Q.ByteTok()
    src = np.asarray(tok.encode("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"))
    out = TS.shuffled_ids(tok, src, 40, np.random.default_rng(0))
    assert len(out) == 40
    full = tok.encode(" ".join(sorted(tok.decode(src.tolist()).split())))
    assert sorted(tok.decode(TS.shuffled_ids(tok, src, len(full), np.random.default_rng(0)).tolist()).split()) \
        == sorted(tok.decode(src.tolist()).split())


def test_main_ladder_resume_skip_and_analysis(tmp_path):
    common = ["--tiny", "--device", "cpu", "--seq-tokens", "32", "--batch-size", "2", "--eval-windows", "4",
              "--val-windows", "2", "--eval-every", "3", "--ckpt-every", "3", "--gen-batch", "2", "--threads", "1",
              "--enc-layers", "1", "--out-root", str(tmp_path / "res"), "--cache-root", str(tmp_path / "cache"),
              "--data-dir", str(ROOT / "data")]
    r2 = ["--ratio", "2", "--steps", "6"]
    assert TS.main(common + ["--ratio", "1", "--steps", "0", "--run-name", "a"]) == 0
    assert TS.main(common + r2 + ["--run-name", "a"]) == 0
    assert TS.main(common + r2 + ["--run-name", "b", "--stop-after", "3"]) == 0
    assert TS.main(common + r2 + ["--run-name", "b"]) == 0          # resumes from the step-3 checkpoint
    warm = str(TS.final_path(tmp_path / "cache", "a", 2, "shuffled", "lora"))
    assert TS.main(common + ["--ratio", "4", "--steps", "4", "--run-name", "a", "--init-from", warm]) == 0
    assert TS.main(common + ["--ratio", "4", "--steps", "4", "--run-name", "a", "--text", "natural"]) == 0
    assert TS.main(common + r2 + ["--run-name", "a"]) == 0          # done -> skip, no second row
    rows = [json.loads(x) for x in (tmp_path / "res/a/runs.jsonl").read_text().splitlines()]
    rb = json.loads((tmp_path / "res/b/runs.jsonl").read_text())
    assert [(r["ratio"], r["text"]) for r in rows] == [(1, "shuffled"), (2, "shuffled"), (4, "shuffled"),
                                                       (4, "natural")]
    assert rows[1]["curve"] == rb["curve"]
    assert np.isclose(rows[1]["eval"]["shuffled"]["cer"], rb["eval"]["shuffled"]["cer"])
    assert rows[0]["cum_samples"] == 0 and rows[1]["cum_samples"] == 12 and rows[2]["cum_samples"] == 20
    assert rows[2]["chain"] == [[2, 6, 2], [4, 4, 2]] and rows[3]["chain"] == [[4, 4, 2]]
    assert rows[1]["mem_slots"] == 16 and set(rows[1]["eval"]) == {"natural", "shuffled"}
    for r in rows:
        for e in r["eval"].values():
            w = e["window_losses"]
            want = (np.minimum(w["blind"], w["prior"]) - np.asarray(w["real"])).mean() * 33 / r["mem_slots"] / np.log(2)
            assert len(w["real"]) == 4 and np.isclose(e["bits_per_slot"], want)
    table = AN.table(rows)
    assert "| 4 | natural | lora | 4 | 8 | shuffled |" in table and len(AN.gates(rows)) == 2
    assert AN.main([str(tmp_path / "res/a"), "--image", "x:1.0:-:0.5"]) == 0
    assert "vs **x**" in (tmp_path / "res/a/summary.md").read_text()


def test_bootstrap_interval_contains_the_mean():
    rng = np.random.default_rng(0)
    w = {"real": rng.uniform(0.1, 0.3, 32).tolist(), "blind": rng.uniform(2, 3, 32).tolist(),
         "prior": rng.uniform(2, 3, 32).tolist()}
    row = {"eval": {"natural": {"window_losses": w}}, "seq_tokens": 480, "mem_slots": 60}
    mean, lo, hi = AN.bits_ci(row)
    assert lo < mean < hi and hi - lo < 5
