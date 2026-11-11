"""Small text statistics used by the reporting job."""
from __future__ import annotations

import re

WORD = re.compile(r"[a-z']+")
STOP = {"the", "a", "an", "of", "and", "to", "in"}


def tokenize(text: str) -> list[str]:
    """Lower-cased words made of letters and apostrophes."""
    return WORD.findall(text.lower())


def word_count(text: str, skip_stop: bool = False) -> int:
    words = tokenize(text)
    if skip_stop:
        words = [w for w in words if w not in STOP]
    return len(words)


def longest_words(text: str, k: int = 3) -> list[str]:
    """The k longest distinct words; ties keep the order of first appearance."""
    seen: dict[str, int] = {}
    for i, w in enumerate(tokenize(text)):
        seen.setdefault(w, i)
    return sorted(seen, key=lambda w: (-len(w), seen[w]))[:k]


def sentence_lengths(text: str) -> list[int]:
    """Number of words in each sentence (split on . ! ?), empty sentences skipped."""
    out = []
    for s in re.split(r"[.!?]+", text):
        n = len(tokenize(s))
        if n:
            out.append(n)
    return out


def top_bigrams(text: str, k: int = 10) -> list[tuple[tuple[str, str], int]]:
    """The k most frequent word bigrams with their counts.

    Ties are broken by first occurrence: the bigram that appears earlier in the text comes first.
    """
    words = tokenize(text)
    pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    uniq = []
    for p in pairs:
        if p not in uniq:
            uniq.append(p)
    scored = [(pairs.count(p), -uniq.index(p), p) for p in uniq]
    scored.sort(reverse=True)
    return [(p, c) for c, _, p in scored[:k]]


def type_token_ratio(text: str) -> float:
    words = tokenize(text)
    return round(len(set(words)) / len(words), 4) if words else 0.0
