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
    pr = Probe(pitch=5, n_classes=7)
    feat = pr.features(torch.randn(2, 196, 768))
    assert feat.shape == (2, pr.c, 224, 224)
    lp = pr.read(feat, torch.tensor([0, 1, 1]), torch.tensor([0, 3, 43]))
    assert lp.shape == (224 * pr.expand, 3, 8)
    assert len(greedy(lp)) == 3


def test_identity_encoder_and_jpeg():
    img = make_pages(ColourEncoding(1), "test", 1)[0].img
    assert Encoder("identity")(img[None]).shape == (1, 196, 768)
    j = degrade(img, "jpeg75")
    assert j.shape == img.shape and np.abs(j.astype(int) - img).mean() > 5
