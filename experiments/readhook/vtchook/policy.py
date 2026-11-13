"""When is an image cheaper than text? Token estimates for both sides of one tool result."""
from __future__ import annotations

import os

MARGIN = float(os.environ.get("VTCHOOK_MARGIN", "0.8"))   # image only if it costs <= MARGIN x the text tokens
READ_PREFIX_TOKENS = 2.5                                   # Read's "<n>\t" prefix per line (~7 chars of digits/space)


def text_tokens(text: str, cpt: float, read_format: bool = False) -> float:
    t = len(text) / cpt
    if read_format:
        t += READ_PREFIX_TOKENS * (text.count("\n") + 1)
    return t


def use_image(image_tokens: int, text_tok: float, margin: float = MARGIN) -> bool:
    return image_tokens <= margin * text_tok
