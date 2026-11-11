"""Shared contract for the vbhook minifiers (read SPEC.md first).

A minifier turns text (a source file, a config file, command output) into a *flow*: one long string that the page
renderer lays out like prose, with no line structure of its own. Conventions inside a flow:

* ``§<n> ``  marks the start of original line n (1-based). Put it before every line that starts a statement or
  record, so a reader can find line numbers (tracebacks, compiler errors, Read offsets). Continuation lines may be
  joined without a tag.
* ``»``      one level of *semantic* indentation (Python blocks, YAML nesting, Makefile recipes). Only where
  indentation carries meaning; never for cosmetic indentation.

The invariant every minifier must keep (checked by ``preserves``): removing tags, level markers and all whitespace
from the flow gives exactly the original with all whitespace removed (after the minifier's declared ``normalise``,
e.g. ANSI colour codes stripped). In other words: only whitespace may be removed or changed, and the only characters
added are tags and markers. A minifier that restructures (e.g. grep output with the path factored out) must also give
``expand(flow)`` that restores a text satisfying the invariant against the original.

Register a minifier with ``@register(kind, exts=[...], names=[...], cpt=<measured chars per Claude token of the
ORIGINAL text>)``. ``detect`` picks one by path, then by content sniffing (``sniff`` callables).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

TAG_RE = re.compile(r"§\d+ ?")
LEVEL = "»"
MARKS = ("§", "»")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def tag(n: int) -> str:
    return f"§{n} "


def strip_marks(flow: str) -> str:
    return TAG_RE.sub("", flow).replace(LEVEL, "")


def nonws(s: str) -> str:
    return re.sub(r"\s+", "", s)


@dataclass
class Minifier:
    kind: str
    fn: Callable[[str], str]
    exts: tuple = ()
    names: tuple = ()
    cpt: float = 2.5                       # chars per Claude token of the original text (measured where possible)
    sniff: Callable[[str], bool] | None = None
    expand: Callable[[str], str] | None = None
    normalise: Callable[[str], str] = field(default=lambda s: s)


REGISTRY: dict[str, Minifier] = {}


def register(kind: str, exts=(), names=(), cpt: float = 2.5, sniff=None, expand=None, normalise=None):
    def deco(fn):
        REGISTRY[kind] = Minifier(kind, fn, tuple(e.lower() for e in exts), tuple(names), cpt, sniff, expand,
                                  normalise or (lambda s: s))
        return fn
    return deco


def preserves(m: Minifier, original: str, flow: str) -> bool:
    restored = m.expand(flow) if m.expand else strip_marks(flow)
    return nonws(m.normalise(original)) == nonws(restored)


def detect(path: str | None = None, text: str | None = None, hint: str | None = None) -> Minifier | None:
    if hint and hint in REGISTRY:
        return REGISTRY[hint]
    if path:
        base = path.rsplit("/", 1)[-1]
        low = base.lower()
        for m in REGISTRY.values():
            if base in m.names or any(low.endswith(e) for e in m.exts):
                return m
    if text is not None:
        for m in REGISTRY.values():
            if m.sniff and m.sniff(text):
                return m
    return None


def minify(text: str, path: str | None = None, hint: str | None = None) -> tuple[str, Minifier] | None:
    """-> (flow, minifier), or None when no minifier applies, the text contains our marker characters, the
    minifier fails, or the invariant does not hold (callers then fall back to plain text)."""
    if any(c in text for c in MARKS):
        return None
    m = detect(path, text, hint)
    if m is None:
        return None
    try:
        flow = m.fn(text)
    except Exception:  # noqa: BLE001 - a minifier bug must never break a tool call
        return None
    if not flow or not preserves(m, text, flow):
        return None
    return flow, m
