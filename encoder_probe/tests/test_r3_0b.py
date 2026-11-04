"""R3-0b checks: resumable chunked token builds, the Qwen ViT split, and the partial-ViT-tuning probe (tiny, CPU)."""
import numpy as np
import pytest
import torch

from encoder_probe.ep import token_cache
from encoder_probe.ep.geom import get_encoding_s, make_pages_s, pages_digest


class FakeSplitVLM:
    """(B,S,S,3) uint8 -> tokens (B, 16, 8) bf16, or with split=k a dict that also has "h<k>" (B, 64, 4)."""

    def __init__(self, fail_after=None):
        self.calls, self.fail_after, self.pages = 0, fail_after, 0

    def __call__(self, imgs, split=None):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise KeyboardInterrupt("killed")
        self.calls += 1
        self.pages += len(imgs)
        m = torch.from_numpy(imgs.reshape(len(imgs), -1).mean(1, dtype=np.float64)).float()
        tok = (m[:, None, None] + torch.arange(8.0)[None, None] + torch.zeros(len(imgs), 16, 8)).to(torch.bfloat16)
        if split is None:
            return tok
        return {"tok": tok, f"h{split}": (m[:, None, None] * split + torch.zeros(len(imgs), 64, 4)).to(torch.bfloat16)}


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(token_cache, "CACHE", tmp_path)
    monkeypatch.setattr(token_cache, "LOCK", tmp_path / ".vlm.lock")
    monkeypatch.setattr(token_cache, "ROOT", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def pages():
    enc = get_encoding_s("col_b4", 448)
    return make_pages_s(enc, "train", 7), make_pages_s(enc, "test", 3)


def _quiet(*a, **k):
    pass


def test_chunked_build_resumes_and_matches_one_shot(tmp_cache, pages):
    tr, te = pages
    ref = token_cache.build(FakeSplitVLM(), "qwen25vl", "col_b4", "clean", 448, 7, 3, tr, te, log=_quiet, chunk=100)
    one = torch.load(ref, weights_only=False)
    ref.unlink()
    killed = FakeSplitVLM(fail_after=2)                     # dies in the 3rd chunk (train pages 4-5)
    with pytest.raises(KeyboardInterrupt):
        token_cache.build(killed, "qwen25vl", "col_b4", "clean", 448, 7, 3, tr, te, log=_quiet, chunk=2)
    assert not ref.exists() and len(list(ref.with_name(ref.name + ".parts").iterdir())) == 2
    again = FakeSplitVLM()
    token_cache.build(again, "qwen25vl", "col_b4", "clean", 448, 7, 3, tr, te, log=_quiet, chunk=2)
    assert again.pages == 10 - 4                            # only the missing chunks were extracted
    d = torch.load(ref, weights_only=False)
    assert list(d)[:2] == ["train", "test"] and set(one) == set(d)
    assert torch.equal(d["train"], one["train"]) and torch.equal(d["test"], one["test"])
    assert d["digest_train"] == pages_digest(tr) and d["digest_test"] == pages_digest(te)
    assert not ref.with_name(ref.name + ".parts").exists()
    got, _ = token_cache.get_tokens("qwen25vl", "col_b4", "clean", 448, 7, 3, tr, te, poll=0.01)
    assert torch.equal(got["train"], one["train"])


def test_chunk_with_changed_pages_is_recomputed(tmp_cache, pages):
    tr, te = pages
    with pytest.raises(KeyboardInterrupt):
        token_cache.build(FakeSplitVLM(fail_after=1), "qwen25vl", "col_b4", "clean", 448, 7, 3, tr, te,
                          log=_quiet, chunk=2)
    fake = FakeSplitVLM()
    p = token_cache.build(fake, "qwen25vl", "col_b4", "clean", 448, 7, 3, tr[::-1], te, log=_quiet, chunk=2)
    assert fake.pages == 10 and torch.load(p, weights_only=False)["digest_train"] == pages_digest(tr[::-1])


def test_split_build_writes_tokens_and_hidden_caches(tmp_cache, pages):
    tr, te = pages
    p = token_cache.build(FakeSplitVLM(), "qwen25vl", "col_b4", "clean", 448, 7, 3, tr, te, log=_quiet, chunk=3,
                          split=28)
    h = token_cache.cache_path(token_cache.hidden_encoder("qwen25vl", 28), "clean", "col_b4", 7, 3)
    assert p.exists() and h.exists() and h.parent.parent.name == "qwen25vl_h28"
    d, dh = torch.load(p, weights_only=False), torch.load(h, weights_only=False)
    assert d["train"].shape == (7, 16, 8) and dh["train"].shape == (7, 64, 4) and dh["test"].shape == (3, 64, 4)
    assert dh["split_block"] == 28 and dh["order"] == "window" and dh["digest_train"] == d["digest_train"]
    assert torch.allclose(dh["train"][:, 0, 0].float() / 28, d["train"][:, 0, 0].float(), rtol=1e-2)
    fake = FakeSplitVLM()                                    # both outputs exist: nothing to do
    token_cache.build(fake, "qwen25vl", "col_b4", "clean", 448, 7, 3, tr, te, log=_quiet, split=28)
    assert fake.calls == 0


def test_probe_waits_while_a_parallel_builder_holds_the_cache_lock(tmp_cache, pages):
    import threading
    import time
    from encoder_probe.ep import flock
    tr, te = pages
    path = token_cache.cache_path("qwen25vl", "clean", "col_b4", 7, 3)
    path.parent.mkdir(parents=True)
    lk = open(path.with_name(path.name + ".lock"), "a")
    assert flock.try_lock(lk) and token_cache.build_locked(path)

    def builder():
        time.sleep(0.4)
        flock.unlock(lk)
        token_cache.build(FakeSplitVLM(), "qwen25vl", "col_b4", "clean", 448, 7, 3, tr, te, log=_quiet)
    th = threading.Thread(target=builder)
    th.start()
    d, waited = token_cache.get_tokens("qwen25vl", "col_b4", "clean", 448, 7, 3, tr, te, poll=0.05, log=_quiet)
    th.join()
    assert waited >= 0.3 and d["train"].shape == (7, 16, 8) and not token_cache.build_locked(path)


# ------------------------------------------------------------------------------ Qwen ViT split (tiny tower)

@pytest.fixture(scope="module")
def tower():
    from encoder_probe.ep import qwen_ft
    m, _ = qwen_ft.load_model(torch.device("cpu"), tiny=True)
    return m.model.visual.eval()


@pytest.mark.parametrize("k", [0, 2, 3, 5, 6])
def test_split_head_tail_reproduces_tower(tower, k):
    from encoder_probe.ep import qwen_split as QS
    from encoder_probe.ep.vlm_encoders import qwen_pixel_values
    imgs = np.random.default_rng(k).integers(0, 255, (3, 448, 448, 3), dtype=np.uint8)
    pv = torch.from_numpy(qwen_pixel_values(imgs))
    thw = torch.tensor([[1, 32, 32]] * 3)
    with torch.no_grad():
        ref = tower(hidden_states=pv, grid_thw=thw).pooler_output
        h = QS.head(tower, pv, thw, k)
        assert torch.allclose(QS.tail(tower, h, thw, k), ref, atol=1e-6, rtol=1e-5)
        # per-page heads, concatenated = the batch's packed sequence (window order stays within each page)
        hs = torch.cat([QS.head(tower, torch.from_numpy(qwen_pixel_values(imgs[i:i + 1])), thw[:1], k)
                        for i in range(3)])
        assert torch.allclose(hs, h, atol=1e-6, rtol=1e-5)
        out = QS.QwenViTTail(tower, k, 448)(h.reshape(3, 1024, -1))
        assert out.shape == (3, 256, ref.shape[-1]) and torch.allclose(out.reshape(-1, ref.shape[-1]), ref,
                                                                         atol=1e-6, rtol=1e-5)
