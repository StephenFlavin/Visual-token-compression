"""Scoring and budget helpers shared by Phase 1 and Phase 2."""
from __future__ import annotations

import math
import re

from rapidfuzz.distance import Levenshtein

from vbpe import config

_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Collapse all whitespace runs to one space and strip. Case and punctuation are kept."""
    return _WS.sub(" ", s).strip()


def cer(ref: str, hyp: str) -> float:
    """Character error rate on normalised strings: edit distance / len(ref). Can exceed 1."""
    r, h = normalize(ref), normalize(hyp)
    return Levenshtein.distance(r, h) / max(1, len(r))


def px_per_char(W: int, H: int, text: str) -> float:
    """The core budget metric: canvas pixels per normalised source character."""
    return W * H / max(1, len(normalize(text)))


def vision_tokens(W: int, H: int) -> float:
    """Measured Claude image-token count: 28x28 patches (+~3 overhead)."""
    P = config.CLAUDE_PATCH
    return math.ceil(W / P) * math.ceil(H / P) + 3


def text_tokens_est(text: str) -> float:
    """Crude proxy (~4 chars/token for English). Replace with a real count when available."""
    return len(normalize(text)) / 4.0
