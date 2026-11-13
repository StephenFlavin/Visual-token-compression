"""Symbol alphabet for the learned-glyph channel (WP-LG2).

``symbols`` = the 95 printable ASCII characters (32..126, in code-point order) followed by the first
``n_tokens`` tokens of the vtc master dictionary (``Dictionary.load(3000)``, priority order), so
``Alphabet.build(n)`` for n < 3000 is a prefix of every larger alphabet (nested experiments) and the
token set equals Phase-2 ``subst{n}``.

Encoding uses ``Dictionary.segment`` (min-width DP with the condensed-font widths and 1-em symbols),
exactly the segmentation Phase-2 substitution uses, so symbol sequences are comparable. With
n_tokens = 0 the encoding is per character. Characters outside printable ASCII are dropped (the
cleaned corpus is pure printable ASCII already).

Reader convention: CTC blank = 0, symbol i -> class i + 1 (``n_classes = len(symbols) + 1``).
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Iterable, Sequence

N_CHARS = 95
CHARS = [chr(c) for c in range(32, 127)]
MASTER_N = 3000


@lru_cache(maxsize=1)
def _master():
    from vtc.dictionary import Dictionary
    return Dictionary.load(MASTER_N)


@lru_cache(maxsize=16)
def _prefix_dict(n: int):
    return _master().prefix(n)


class Alphabet:
    n_chars: int = N_CHARS

    def __init__(self, tokens: Sequence[str] = (), dictionary=None):
        tokens = list(tokens)
        if any(len(t) < 2 for t in tokens):
            raise ValueError("tokens must have length >= 2")
        if len(set(tokens)) != len(tokens):
            raise ValueError("duplicate tokens")
        self.symbols: list[str] = CHARS + tokens
        self.n_tokens = len(tokens)
        self.index = {s: i for i, s in enumerate(self.symbols)}
        self._dict = dictionary
        if self.n_tokens and dictionary is None:
            from vtc.dictionary import Dictionary
            self._dict = Dictionary(tokens)

    @classmethod
    def build(cls, n_tokens: int) -> "Alphabet":
        n = int(n_tokens)
        if n <= 0:
            return cls()
        d = _prefix_dict(n)
        return cls(d.tokens, dictionary=d)

    # ------------------------------------------------------------------ sizes / classes
    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def n_classes(self) -> int:
        """CTC classes: blank + one per symbol."""
        return len(self.symbols) + 1

    @property
    def tokens(self) -> list[str]:
        return self.symbols[N_CHARS:]

    def is_token(self, i: int) -> bool:
        return int(i) >= N_CHARS

    @staticmethod
    def to_classes(ids: Iterable[int]) -> list[int]:
        return [int(i) + 1 for i in ids]

    @staticmethod
    def from_classes(classes: Iterable[int]) -> list[int]:
        return [int(c) - 1 for c in classes if int(c) > 0]

    # ------------------------------------------------------------------ codec
    @staticmethod
    def clean(text: str) -> str:
        """Drop characters outside printable ASCII."""
        return "".join(c for c in text if 32 <= ord(c) < 127)

    def segments(self, text: str) -> list[str]:
        text = self.clean(text)
        if not self.n_tokens:
            return list(text)
        return self._dict.segment(text)

    def encode(self, text: str) -> list[int]:
        idx = self.index
        out: list[int] = []
        for s in self.segments(text):
            i = idx.get(s)
            if i is None:                 # cannot happen with the default dictionary; spell out
                out.extend(idx[c] for c in s)
            else:
                out.append(i)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        sy, n = self.symbols, len(self.symbols)
        return "".join(sy[i] for i in ids if 0 <= i < n)

    # ------------------------------------------------------------------ stats
    def stats(self, lines: Iterable[str]) -> dict:
        """Symbol-frequency statistics over ``lines`` (e.g. a train sample).

        -> chars, symbols, seg_ratio (symbols/char), token_coverage (fraction of chars inside
        tokens), counts (list per symbol id), unused_tokens, top (20 most frequent symbols)."""
        cnt: Counter = Counter()
        chars = syms = covered = 0
        for line in lines:
            ids = self.encode(line)
            chars += len(self.clean(line))
            syms += len(ids)
            cnt.update(ids)
            covered += sum(len(self.symbols[i]) for i in ids if i >= N_CHARS)
        counts = [cnt.get(i, 0) for i in range(len(self.symbols))]
        return {"n_tokens": self.n_tokens, "n_symbols": len(self.symbols), "chars": chars,
                "symbols": syms, "seg_ratio": syms / max(1, chars),
                "token_coverage": covered / max(1, chars), "counts": counts,
                "unused_tokens": sum(1 for i in range(N_CHARS, len(self.symbols)) if counts[i] == 0),
                "unused_chars": sum(1 for i in range(N_CHARS) if counts[i] == 0),
                "top": [(self.symbols[i], c) for i, c in cnt.most_common(20)]}
