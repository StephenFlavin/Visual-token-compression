"""Scheme interface. Every rendering scheme (latin, substitution, quadrant, ...) implements this."""
from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from vbpe.layout import Unit

BASE_OCR_SYSTEM_PROMPT = (
    "You are a precise OCR engine. Transcribe exactly what is written in the image, in reading "
    "order (left to right, top to bottom). Join lines with single spaces. Do not correct, "
    "complete, summarise or comment. Output only the transcription."
)


class Scheme(ABC):
    name: str = "base"

    @abstractmethod
    def encode(self, text: str) -> list[Unit]:
        """Source text -> units in reading order. Must be deterministic."""

    def decode(self, transcription: str) -> str:
        """Reader output (concatenated labels, or VLM output) -> source-text estimate."""
        return transcription

    def transcription_target(self, text: str) -> str:
        """What a perfect reader would output for ``text``: ''.join of all unit labels."""
        return "".join("".join(u.labels) for u in self.encode(text))

    # ---- zero-shot VLM evaluation (Phase 1) ----
    def vlm_system_prompt(self) -> str:
        return BASE_OCR_SYSTEM_PROMPT

    def vlm_prompt(self) -> str:
        """User-turn instruction placed after the image(s)."""
        return "Transcribe all text in the image exactly."

    def vlm_key_images(self) -> list[Image.Image]:
        """Optional instruction images (legend/examples) sent BEFORE the target image.

        These are treated like a system prompt: amortisable, NOT counted in the pixel budget.
        """
        return []
