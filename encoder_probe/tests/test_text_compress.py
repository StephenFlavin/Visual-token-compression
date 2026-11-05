"""Stage 0 (text_compress.py) on the TINY random-init Qwen2.5-VL: CPU, no downloads, ~1 min."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "encoder_probe" / "experiments"))
import text_compress as S0  # noqa: E402

CPU = torch.device("cpu")
torch.set_num_threads(1)


def setup(grad_ckpt=True, m=4):
    a = S0.parse(["--ratio", "8", "--tiny", "--device", "cpu", "--seq-tokens", "32"]
                 + ([] if grad_ckpt else ["--no-grad-ckpt"]))
    pm, tok, sw = S0.build(a, CPU)
    comp = S0.Compressor(pm, tok, m, CPU, False, sw)
    slots = S0.Slots(m, comp.emb.weight.shape[1], 0.3, torch.Generator().manual_seed(0))
    torch.manual_seed(1)
    with torch.no_grad():   # make the adapter matter: LoRA B starts at 0
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)
    ids = torch.randint(0, 256, (2, 32))
    return pm, tok, sw, comp, slots, ids


def test_decoder_is_the_unmodified_llm_and_encoder_uses_lora():
    pm, tok, sw, comp, slots, ids = setup()
    pm.eval()
    with torch.no_grad():
        mem = comp.encode(ids, slots)
        l_switch, _ = comp.recon_loss(ids, mem)
        with sw.state(True):
            mem_off = comp.encode(ids, slots)
        for n, p in pm.named_parameters():   # ground truth: zero adapter = the unmodified LLM
            if "lora_B" in n:
                p.zero_()
        l_plain, _ = comp.recon_loss(ids, mem)
    assert torch.allclose(l_switch, l_plain, atol=1e-5)
    assert not torch.allclose(mem, mem_off, atol=1e-3)   # the encoder pass does use the adapter
    assert sw.off is False


def test_grad_checkpointing_gives_the_same_gradients():
    grads = []
    for ck in (True, False):
        pm, tok, sw, comp, slots, ids = setup(grad_ckpt=ck)
        pm.train()
        pm.eval()   # no dropout, so both runs are deterministic
        loss, _ = comp.recon_loss(ids, comp.encode(ids, slots))
        loss.backward()
        ps = S0.trainables(pm, slots)
        grads.append(torch.cat([p.grad.flatten() for p in ps if p.grad is not None]))
        assert sum(p.grad is not None and p.grad.abs().sum() > 0 for p in ps) >= len(ps) - 1
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-6)


def test_greedy_matches_teacher_forced_argmax():
    pm, tok, sw, comp, slots, ids = setup()
    pm.eval()
    with torch.no_grad():
        mem = comp.encode(ids, slots)
        out = comp.greedy(mem, 12)
        with sw.state(True):
            pre = comp.prefix(mem)
            for b, row in enumerate(out):
                seq = torch.tensor(row[:6], dtype=torch.long)
                inp = torch.cat([pre[b:b + 1], comp.emb(seq)[None]], 1)
                h = comp.lm(inputs_embeds=inp).last_hidden_state[0, pre.shape[1] - 1:]
                assert comp.head(h).argmax(-1)[:len(seq)].tolist() == seq.tolist()


def test_slots_learn_a_fixed_batch():
    pm, tok, sw, comp, slots, ids = setup()
    pm.eval()
    params = S0.trainables(pm, slots)
    opt = torch.optim.Adam(params, lr=3e-3)
    first = None
    for _ in range(40):
        opt.zero_grad()
        loss, _ = comp.recon_loss(ids, comp.encode(ids, slots))
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.8 * first


def test_main_writes_row_resumes_and_skips(tmp_path):
    common = ["--ratio", "8", "--tiny", "--device", "cpu", "--seq-tokens", "32", "--steps", "6",
              "--eval-windows", "4", "--val-windows", "2", "--eval-every", "3", "--ckpt-every", "3",
              "--gen-batch", "2", "--threads", "1", "--out-root", str(tmp_path / "res"),
              "--cache-root", str(tmp_path / "cache"), "--data-dir", str(ROOT / "data")]
    assert S0.main(common + ["--run-name", "a"]) == 0
    assert S0.main(common + ["--run-name", "b", "--stop-after", "3"]) == 0
    assert S0.main(common + ["--run-name", "b"]) == 0        # resumes from the step-3 checkpoint
    ra = json.loads((tmp_path / "res/a/runs.jsonl").read_text())
    rb = json.loads((tmp_path / "res/b/runs.jsonl").read_text())
    assert ra["curve"] == rb["curve"] and np.isclose(ra["cer"], rb["cer"]) and ra["fair_hash"] == rb["fair_hash"]
    assert ra["mem_slots"] == 4 and ra["blind_cer"] > 0 and "prefix_frac" in ra["ops"]
    assert S0.main(common + ["--run-name", "a"]) == 0        # done -> skip, no second row
    assert len((tmp_path / "res/a/runs.jsonl").read_text().splitlines()) == 1
