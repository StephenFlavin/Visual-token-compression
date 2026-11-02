"""Zero-shot pretrained OCR reader (RapidOCR = PaddleOCR PP-OCR ch+en models on ONNX, CPU).

Used as a non-chat zero-shot reader for Phase 1b: it knows both Latin and CJK glyphs and has no
safety layer. Its Chinese/English model drops inter-word spaces and emits full-width punctuation,
so it is scored with ``cer_nospace`` (whitespace removed, punctuation normalised) for ALL schemes.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache

import numpy as np
from rapidfuzz.distance import Levenshtein


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def read(img) -> str:
    """OCR a page; returns lines joined by spaces in reading order."""
    res, _ = _engine()(np.array(img.convert("RGB")))
    if not res:
        return ""
    items = sorted((sum(p[1] for p in b) / 4, min(p[0] for p in b), t,
                    max(p[1] for p in b) - min(p[1] for p in b)) for b, t, _ in res)
    lines: list[list] = []
    for y, x, t, h in items:
        if lines and abs(lines[-1][0] - y) < 0.5 * h:
            lines[-1][1].append((x, t))
        else:
            lines.append([y, [(x, t)]])
    return " ".join(" ".join(t for _, t in sorted(l[1])) for l in lines)


_PUNCT = str.maketrans({"，": ",", "。": ".", "：": ":", "；": ";", "“": '"', "”": '"', "‘": "'",
                        "’": "'", "（": "(", "）": ")", "！": "!", "？": "?", "、": ","})


def canon(s: str) -> str:
    """NFKC (full-width -> ASCII), CJK punctuation -> ASCII, drop all whitespace."""
    s = unicodedata.normalize("NFKC", s).translate(_PUNCT)
    return "".join(s.split())


def cer_nospace(ref: str, hyp: str) -> float:
    r, h = canon(ref), canon(hyp)
    return Levenshtein.distance(r, h) / max(1, len(r))


def read_lines(img, em: float, n_lines: int, params=None) -> str:
    """Recognition-only OCR on the known line strips of a page rendered by vbpe.layout.

    Calls the PP-OCR recognition model directly (rapidocr_onnxruntime 1.2.x ignores use_det and
    re-applies a 0.5 score filter when kwargs are passed). Detection and reading-order assembly are
    skipped, so the score reflects glyph legibility at the page's own resolution. Each strip is
    exactly one line pitch tall: [margin + i*pitch, margin + (i+1)*pitch).
    """
    from vbpe.layout import LayoutParams
    p = params or LayoutParams()
    pitch = p.pitch(em)
    W, H = img.size
    rgb = img.convert("RGB")
    crops = []
    for i in range(n_lines):
        top = int(round(p.margin + i * pitch))
        bot = min(H, int(round(p.margin + (i + 1) * pitch)))
        crops.append(np.array(rgb.crop((0, top, W, bot)))[:, :, ::-1].copy())  # BGR like cv2
    rec_res, _ = _engine().text_recognizer(crops)
    return " ".join(t for t, _score in rec_res)
