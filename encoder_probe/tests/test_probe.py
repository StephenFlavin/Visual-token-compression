"""Fast checks for the encoder probe (no model downloads)."""
import numpy as np
import torch

from encoder_probe.ep.encoders import Encoder, degrade, patchify
from encoder_probe.ep.pages import ColourEncoding, get_encoding, make_pages, palette
from encoder_probe.ep.probe import GRID, PATCH, Probe, greedy, unpatchify


def test_colour_page_roundtrip():
    enc = ColourEncoding(2)
    pg = make_pages(enc, "test", 1)[0]
    assert pg.img.shape == (224, 224, 3) and len(pg.refs) == 112
    pal = palette()
    ch = pg.refs[3][5]
    assert (pg.img[3 * 2, 5 * 2] == pal[enc.idx[ch]]).all()
    assert pg.n_chars == 112 * 112


def test_latin_page_lines_fit():
    enc = get_encoding("latin_p6")
    pg = make_pages(enc, "test", 1)[0]
    assert len(pg.refs) == 224 // 6
    assert all(len(lab) == len(r) for lab, r in zip(pg.labels, pg.refs))


def test_patchify_unpatchify_inverse():
    x = np.random.rand(2, 224, 224, 3).astype(np.float32)
    tok = torch.from_numpy(patchify(x))                       # (2,196,768), per-token order (dy,dx,ch)
    g = tok.transpose(1, 2).reshape(2, PATCH, PATCH, 3, GRID, GRID)
    g = g.permute(0, 3, 1, 2, 4, 5).reshape(2, 3 * PATCH * PATCH, GRID, GRID)   # -> (ch,dy,dx)
    assert torch.allclose(unpatchify(g, 3), torch.from_numpy(x).permute(0, 3, 1, 2))


def test_probe_shapes_and_greedy():
    for pitch, lines in ((5, [0, 3, 43]), (1, [0, 17, 223])):
        pr = Probe(pitch=pitch, n_classes=7, hid=32)
        feat = pr.features(torch.randn(2, 196, 768))
        assert feat.shape == (2, pr.c, 224, 224)
        lp = pr.read(feat, torch.tensor([0, 1, 1]), torch.tensor(lines))
        assert lp.shape == (224 * pr.expand, 3, 8)
        assert len(greedy(lp)) == 3
        assert pr.recon_loss(feat, torch.zeros(2, 3, 224, 224)).ndim == 0


def test_probe_skip_can_reconstruct_identity_pixels():
    """The 1x1 skip spans the identity token->pixel map: a least-squares fit reproduces the page exactly."""
    x = np.random.rand(1, 224, 224, 3).astype(np.float32)
    tok = torch.from_numpy(patchify(x))
    pr = Probe(pitch=4, n_classes=7, hid=8, c=3)
    w = torch.zeros(3 * PATCH * PATCH, 768)
    for ch in range(3):
        for dy in range(PATCH):
            for dx in range(PATCH):
                w[ch * 256 + dy * 16 + dx, (dy * 16 + dx) * 3 + ch] = 1
    with torch.no_grad():
        pr.skip.weight.copy_(w[:, :, None, None]); pr.skip.bias.zero_()
        for m in pr.grid:
            if isinstance(m, torch.nn.Conv2d):
                m.weight.zero_(); m.bias.zero_()
        feat = pr.features(tok)
    assert torch.allclose(feat, torch.from_numpy(x).permute(0, 3, 1, 2), atol=1e-6)


def test_identity_encoder_and_jpeg():
    img = make_pages(ColourEncoding(1), "test", 1)[0].img
    assert Encoder("identity")(img[None]).shape == (1, 196, 768)
    j = degrade(img, "jpeg75")
    assert j.shape == img.shape and np.abs(j.astype(int) - img).mean() > 5


def test_colour_hypothesis_not_stripped():
    """Colour refs are raw chunks (may start/end with a space); a perfect read must score 0 CER."""
    enc = ColourEncoding(3)
    pg = make_pages(enc, "test", 2)
    refs = [r for p in pg for r in p.refs]
    assert any(r != r.strip() for r in refs)
    assert all(enc.clean_hyp(r) == r for r in refs)
    assert get_encoding("latin_p6").clean_hyp("  a  b ") == "a b"
