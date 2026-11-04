"""Fast checks for round 2 (sized pages, grid probe, token cache, Qwen pixel layout). No downloads, no VLM."""
import threading
import time

import numpy as np
import pytest
import torch

from encoder_probe.ep import token_cache
from encoder_probe.ep.geom import (SizedRGB3, get_encoding_s, make_pages_s, mono_ink_s, pages_digest)
from encoder_probe.ep.pages import get_encoding, make_pages
from encoder_probe.ep.probe import Probe
from encoder_probe.ep.probe_grid import GridProbe, unpatchify_g
from encoder_probe.ep.rgb_overlay import split_streams
from encoder_probe.ep.vlm_encoders import GEOMS, patchify_s, qwen_pixel_values

ROUND2 = ["latin_p5", "latin_p6", "latin_p8", "latin_p10", "latin_p16", "col_b1", "col_b2", "col_b4",
          "lg_free_h4", "lg_pb_h5", "rgb3_p6", "rgb3_p8", "mono_p6", "mono_p8", "latin_p24", "latin_p32"]


@pytest.mark.parametrize("name", ["latin_p6", "col_b2", "rgb3_p6", "mono_p8", "lg_free_h4"])
def test_sized_224_reproduces_round1_first_page(name):
    a = make_pages(get_encoding(name), "test", 1)[0]
    b = make_pages_s(get_encoding_s(name, 224), "test", 1)[0]
    assert (a.img == b.img).all() and a.refs == b.refs and a.labels == b.labels


@pytest.mark.parametrize("S", [448, 512])
@pytest.mark.parametrize("name", ROUND2)
def test_sized_pages_full_distinct_and_decodable(name, S):
    enc = get_encoding_s(name, S)
    pages = make_pages_s(enc, "test", 2)
    assert len({tuple(p.refs) for p in pages}) == 2
    for pg in pages:
        assert pg.img.shape == (S, S, 3) and pg.img.dtype == np.uint8
        assert len(pg.refs) == len(pg.labels) == S // enc.pitch
        assert all(pg.refs), "every line of a page is filled (text never runs out mid-page)"
        for lab, r in zip(pg.labels, pg.refs):
            assert enc.clean_hyp(enc.decode("".join(lab))) == r
            assert 2 * len(lab) + 1 <= 3 * S                         # CTC frame budget (3 frames per column)
        assert pg.n_chars <= enc.page_chars_max


def test_dense_test_pages_use_val_and_never_repeat():
    enc = get_encoding_s("col_b1", 448)
    with pytest.raises(ValueError):
        make_pages_s(enc, "test", 16)
    pages = make_pages_s(enc, "test", 16, allow_fewer=True)
    assert 4 <= len(pages) < 16 and len({tuple(p.refs) for p in pages}) == len(pages)
    assert sum(p.n_chars for p in pages) > 1_093_347               # beyond data/test.txt: val.txt is used too


def test_rgb3_448_channels_are_mono_streams():
    enc = get_encoding_s("rgb3_p8", 448)
    assert isinstance(enc, SizedRGB3) and enc.n_cells > 100
    pg = make_pages_s(enc, "test", 1)[0]
    streams = [split_streams(r, enc.n_cells) for r in pg.refs]
    for c in range(3):
        assert (mono_ink_s([s[c] for s in streams], 8, 448) == pg.img[:, :, c]).all()


def test_grid_probe_equals_round1_probe_at_14x16():
    torch.manual_seed(0)
    old = Probe(pitch=6, n_classes=9, hid=32)
    new = GridProbe(pitch=6, n_classes=9, d=768, G=14, P=16, hid=32)
    sd = {k: v for k, v in old.state_dict().items() if k != "gate"}
    assert {k: v.shape for k, v in sd.items()} == {k: v.shape for k, v in new.state_dict().items()}
    new.load_state_dict(sd)
    old.eval(), new.eval()
    tok = torch.randn(2, 196, 768)
    with torch.no_grad():
        fo, fn = old.features(tok), new.features(tok)
        assert torch.allclose(fo, fn)
        pi, li = torch.tensor([0, 1]), torch.tensor([3, 30])
        assert torch.allclose(old.read(fo, pi, li), new.read(fn, pi, li))


@pytest.mark.parametrize("enc_name", list(GEOMS))
def test_patchify_unpatchify_inverse_and_identity_skip(enc_name):
    S, P = GEOMS[enc_name]
    G = S // P
    x = np.random.rand(1, S, S, 3).astype(np.float32)
    tok = torch.from_numpy(patchify_s(x, P))
    assert tok.shape == (1, G * G, P * P * 3)
    g = tok.transpose(1, 2).reshape(1, P, P, 3, G, G).permute(0, 3, 1, 2, 4, 5).reshape(1, 3 * P * P, G, G)
    assert torch.allclose(unpatchify_g(g, 3, P, G), torch.from_numpy(x).permute(0, 3, 1, 2))
    if P <= 32:   # the 1x1 skip spans the identity map: an exact permutation reproduces the page
        pr = GridProbe(pitch=4, n_classes=3, d=P * P * 3, G=G, P=P, hid=4)
        w = torch.zeros(3 * P * P, P * P * 3)
        idx = torch.arange(P * P * 3).reshape(P, P, 3).permute(2, 0, 1).reshape(-1)  # (ch,dy,dx) -> (dy,dx,ch)
        w[torch.arange(3 * P * P), idx] = 1
        with torch.no_grad():
            pr.skip.weight.copy_(w[:, :, None, None]); pr.skip.bias.zero_()
            for m in pr.grid:
                if isinstance(m, torch.nn.Conv2d):
                    m.weight.zero_(); m.bias.zero_()
            assert torch.allclose(pr.features(tok), torch.from_numpy(x).permute(0, 3, 1, 2), atol=1e-6)


@pytest.mark.parametrize("enc_name", list(GEOMS))
def test_grid_probe_shapes(enc_name):
    S, P = GEOMS[enc_name]
    G = S // P
    pr = GridProbe(pitch=5, n_classes=7, d=16, G=G, P=P, hid=8)
    feat = pr.features(torch.randn(2, G * G, 16))
    assert feat.shape == (2, 3, S, S)
    lp = pr.read(feat, torch.tensor([0, 1]), torch.tensor([0, S // 5 - 1]))
    assert lp.shape == (3 * S, 2, 8)


def test_qwen_pixel_values_match_processor():
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    proc = Qwen2VLImageProcessorPil(min_pixels=3136, max_pixels=12845056, patch_size=14, temporal_patch_size=2,
                                    merge_size=2, image_mean=[0.48145466, 0.4578275, 0.40821073],
                                    image_std=[0.26862954, 0.26130258, 0.27577711])
    imgs = np.random.randint(0, 256, (2, 448, 448, 3), dtype=np.uint8)
    o = proc(images=list(imgs), return_tensors="np")
    assert (o["image_grid_thw"] == [[1, 32, 32]] * 2).all()           # no resizing at 448: 1024 patches -> 256 tokens
    assert np.abs(o["pixel_values"] - qwen_pixel_values(imgs)).max() < 1e-5


class FakeVLM:
    """Stands in for a VLM encoder: (B,S,S,3) uint8 -> (B, G*G, d) bf16."""

    def __init__(self, G=4, d=8):
        self.G, self.d, self.calls = G, d, 0

    def __call__(self, imgs):
        self.calls += 1
        m = torch.from_numpy(imgs.reshape(len(imgs), -1).mean(1, dtype=np.float64)).float()
        return (m[:, None, None] + torch.zeros(len(imgs), self.G * self.G, self.d)).to(torch.bfloat16)


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(token_cache, "CACHE", tmp_path)
    monkeypatch.setattr(token_cache, "LOCK", tmp_path / ".vlm.lock")
    monkeypatch.setattr(token_cache, "ROOT", tmp_path)
    return tmp_path


def _pages(n=3):
    enc = get_encoding_s("col_b4", 448)
    return make_pages_s(enc, "train", n), make_pages_s(enc, "test", 2)


def test_cache_build_and_digest_check(tmp_cache):
    tr, te = _pages()
    p = token_cache.build(FakeVLM(), "qwen25vl", "col_b4", "clean", 448, 3, 2, tr, te, log=lambda *a, **k: None)
    d, waited = token_cache.get_tokens("qwen25vl", "col_b4", "clean", 448, 3, 2, tr, te, poll=0.01)
    assert p.exists() and d["train"].shape == (3, 16, 8) and d["test"].shape == (2, 16, 8) and waited < 1
    assert d["digest_train"] == pages_digest(tr)
    with pytest.raises(RuntimeError):
        token_cache.get_tokens("qwen25vl", "col_b4", "clean", 448, 3, 2, tr[::-1], te, poll=0.01)


def test_cache_waits_for_lock_holder_then_loads(tmp_cache):
    tr, te = _pages()
    holder = token_cache.VLMLock()
    holder.acquire()                                       # stands in for the builder process
    fake = FakeVLM()

    def builder():
        time.sleep(0.5)
        token_cache.build(fake, "qwen25vl", "col_b4", "clean", 448, 3, 2, tr, te, log=lambda *a, **k: None)
        holder.release()
    th = threading.Thread(target=builder)
    th.start()
    d, waited = token_cache.get_tokens("qwen25vl", "col_b4", "clean", 448, 3, 2, tr, te, poll=0.05)
    th.join()
    assert waited >= 0.4 and fake.calls == 2 and d["test"].shape[0] == 2


def test_cache_self_builds_when_no_builder(tmp_cache, monkeypatch):
    from encoder_probe.ep import vlm_encoders
    tr, te = _pages()
    fake = FakeVLM()
    monkeypatch.setattr(vlm_encoders, "load_vlm", lambda name, device="cpu": fake)
    d, _ = token_cache.get_tokens("deepencoder", "col_b4", "jpeg75", 448, 3, 2, tr, te, poll=0.01,
                                  log=lambda *a, **k: None)
    assert fake.calls == 2 and token_cache.cache_path("deepencoder", "jpeg75", "col_b4", 3, 2).exists()


def test_queue_order_and_anchors():
    from encoder_probe.ep.vlm_encoders import encodings_for
    assert encodings_for("qwen25vl") == encodings_for("identity28") and "latin_p24" in encodings_for("qwen25vl")
    assert encodings_for("deepencoder")[-2:] == ["latin_p24", "latin_p32"]
