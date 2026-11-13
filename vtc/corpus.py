"""WikiText-103 corpus: cleaning to plain English, evaluation passages, and line streams.

Source: ``datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")``.

Cleaning (``clean_paragraph`` / ``split_sentences``):
  * heading lines (`` = Title = ``, `` = = Section = = ``) and blank lines are dropped;
  * wikitext tokenisation is undone: `` @-@ `` -> ``-``, `` @,@ `` -> ``,``, `` @.@ `` -> ``.``,
    no space before ``, . ; : ! ? % ) ]`` or after ``( [ $``, clitics (`` 's``, `` n't`` ...),
    plural possessives (``players ' `` -> ``players' ``), and paired double quotes are re-attached;
  * a few typographic Unicode characters are transliterated (dashes, curly quotes, nbsp);
  * paragraphs are split into sentences; a sentence is kept only if it is pure printable ASCII,
    20-400 chars long and has balanced double quotes / parentheses.

Everything cached under ``config.DATA``:
  * ``passages_{natural,scrambled}.jsonl`` - rows ``{"id", "text"}``;
  * ``{train,val,test}.txt`` - cleaned text, one paragraph per line (kept sentences joined by a
    single space). ``train`` is capped at ~``TRAIN_CAP_CHARS`` chars (taken from the start of the
    WikiText train split).
"""
from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from vtc import config

HF_NAME = ("Salesforce/wikitext", "wikitext-103-raw-v1")
SPLITS = {"train": "train", "val": "validation", "test": "test"}
TRAIN_CAP_CHARS = 40_000_000

PASSAGE_SEED = 1234
N_NATURAL = 12
N_SCRAMBLED = 6
PASSAGE_MIN, PASSAGE_MAX = 2800, 3200
SENT_MIN, SENT_MAX = 20, 400

# ----------------------------------------------------------------------------- cleaning

_TRANSLIT = str.maketrans({
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"', " ": " ",
    "…": "...",
})

_SUBS = [
    (re.compile(r" @-@ "), "-"),
    (re.compile(r" @,@ "), ","),
    (re.compile(r" @\.@ "), "."),
    (re.compile(r" @(.)@ "), r"\1"),                      # any other @x@ join marker
    (re.compile(r" (n't|'t|'s|'re|'ve|'ll|'d|'m|'S)\b"), r"\1"),
    (re.compile(r"(?<=[sSzZ]) ' (?=\w|$)"), "' "),        # plural possessive: players ' X
    (re.compile(r"(?<=[sSzZ]) '$"), "'"),
    (re.compile(r" ([,.;:!?%)\]}])"), r"\1"),
    (re.compile(r"([(\[{$]) "), r"\1"),
    (re.compile(r"(?<=\d) (st|nd|rd|th)\b"), r"\1"),
    (re.compile(r" {2,}"), " "),
]
_HEADING = re.compile(r"^\s*(=\s*)+[^=].*?(\s*=)+\s*$")
_PRINTABLE = re.compile(r"^[\x20-\x7e]+$")


def _fix_double_quotes(s: str) -> str:
    """Attach paired standalone ``"`` tokens: opening quote hugs the next word, closing the previous."""
    toks = s.split(" ")
    out: list[str] = []
    opening = True
    glue_next = False
    for t in toks:
        if t == '"':
            if opening:
                glue_next = True
            elif out:
                out[-1] += '"'
            else:
                out.append('"')
            opening = not opening
            continue
        if glue_next:
            t = '"' + t
            glue_next = False
        out.append(t)
    if glue_next:
        out.append('"')
    return " ".join(out)


def clean_paragraph(line: str) -> str:
    """Undo wikitext tokenisation for one raw line. Returns '' for headings / blank lines."""
    if not line.strip() or _HEADING.match(line):
        return ""
    s = " " + line.strip().translate(_TRANSLIT) + " "
    s = re.sub(r" {2,}", " ", s)
    # quotes at line start/end also need the ' " ' pattern
    s = _fix_double_quotes(s)
    s = s.strip()
    for pat, rep in _SUBS:
        s = pat.sub(rep, s)
    return s.strip()


_ABBREV = {
    "Mr", "Mrs", "Ms", "Dr", "St", "Sr", "Jr", "Prof", "Gen", "Col", "Lt", "Capt", "Sgt", "Maj",
    "Rev", "Gov", "Sen", "Rep", "Adm", "Cpl", "Pvt", "Hon", "Fr", "Mt", "Ft", "Co", "Corp", "Inc",
    "Ltd", "Bros", "vs", "No", "Nos", "Vol", "pp", "ca", "c", "approx", "al", "etc", "Jan", "Feb",
    "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec", "U.S", "U.K", "e.g",
    "i.e", "a.m", "p.m", "op", "Op", "cf", "Ave", "Blvd", "Rd", "Ch", "fl", "b", "d",
}
_SENT_END = re.compile(r'(?<=[.!?])(["\')]?)\s+(?=["\'(]?[A-Z0-9])')


def split_sentences(par: str) -> list[str]:
    """Split a cleaned paragraph into sentences (regex + abbreviation / initial guard)."""
    pieces: list[str] = []
    start = 0
    for m in _SENT_END.finditer(par):
        head = par[start:m.end(1)]
        # last word before the boundary, without the terminal punctuation
        lastword = head.rstrip("\"')").rstrip(".!?").rsplit(" ", 1)[-1].lstrip("\"'(")
        if head.rstrip("\"')").endswith(".") and (
                lastword in _ABBREV or (len(lastword) == 1 and lastword.isupper())
                or re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", lastword)):
            continue
        pieces.append(head.strip())
        start = m.end()
    pieces.append(par[start:].strip())
    return [p for p in pieces if p]


def keep_sentence(s: str) -> bool:
    return (SENT_MIN <= len(s) <= SENT_MAX and bool(_PRINTABLE.match(s))
            and s.count('"') % 2 == 0 and s.count("(") == s.count(")")
            and s.count("[") == s.count("]") and "@" not in s and " ' " not in s
            and not s[0].islower() and s[-1] in '.!?"\')')


def clean_sentences(par_raw: str) -> list[str]:
    return [s for s in split_sentences(clean_paragraph(par_raw)) if keep_sentence(s)]


# ----------------------------------------------------------------------------- split files

def _raw_split(split: str):
    import datasets  # heavy import, only when building

    return datasets.load_dataset(*HF_NAME, split=SPLITS[split])["text"]


def split_path(split: str) -> Path:
    return config.DATA / f"{split}.txt"


def build_split(split: str, cap: int | None = None) -> Path:
    """Clean a WikiText split into data/{split}.txt (one paragraph of kept sentences per line)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    cap = cap if cap is not None else (TRAIN_CAP_CHARS if split == "train" else None)
    path = split_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for raw in _raw_split(split):
            sents = clean_sentences(raw)
            if not sents:
                continue
            par = " ".join(sents)
            f.write(par + "\n")
            total += len(par) + 1
            if cap and total >= cap:
                break
    tmp.replace(path)
    return path


def split_text_path(split: str) -> Path:
    """Path to the cleaned split text, building it on first use."""
    p = split_path(split)
    if not p.exists():
        build_split(split)
    return p


@lru_cache(maxsize=4)
def read_paragraphs(split: str) -> tuple[str, ...]:
    with split_text_path(split).open() as f:
        return tuple(line.rstrip("\n") for line in f if line.strip())


# ----------------------------------------------------------------------------- passages

def _passage_path(variant: str) -> Path:
    return config.DATA / f"passages_{variant}.jsonl"


def make_natural_passages(sentences: list[str], n: int = N_NATURAL, seed: int = PASSAGE_SEED,
                          lo: int = PASSAGE_MIN, hi: int = PASSAGE_MAX) -> list[dict]:
    """Concatenate globally shuffled sentences into passages of lo..hi chars."""
    pool = list(sentences)
    random.Random(seed).shuffle(pool)
    out: list[dict] = []
    cur: list[str] = []
    length = 0
    deferred: list[str] = []
    it = iter(pool)
    while len(out) < n:
        s = deferred.pop(0) if deferred else next(it)
        add = len(s) + (1 if cur else 0)
        if length + add > hi:
            deferred.append(s)          # does not fit: try it in the next passage
            if len(deferred) > 50:      # pathological; drop the oldest
                deferred.pop(0)
            if length >= lo:
                out.append({"id": f"nat{len(out):02d}", "text": " ".join(cur)})
                cur, length = [], 0
            continue
        cur.append(s)
        length += add
        if length >= lo:
            out.append({"id": f"nat{len(out):02d}", "text": " ".join(cur)})
            cur, length = [], 0
    return out


def scramble(text: str, seed: int) -> str:
    """Shuffle the space-separated words of ``text`` (punctuation stays attached)."""
    words = text.split(" ")
    random.Random(seed).shuffle(words)
    return " ".join(words)


def make_scrambled_passages(natural: list[dict], n: int = N_SCRAMBLED,
                            seed: int = PASSAGE_SEED) -> list[dict]:
    return [{"id": "scr" + p["id"][3:], "text": scramble(p["text"], seed + i)}
            for i, p in enumerate(natural[:n])]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_passages() -> None:
    sents = [s for par in read_paragraphs("test") for s in split_sentences(par)]
    nat = make_natural_passages(sents)
    _write_jsonl(_passage_path("natural"), nat)
    _write_jsonl(_passage_path("scrambled"), make_scrambled_passages(nat))


def load_passages(variant: str = "natural") -> list[dict]:
    """12 natural (sentence-shuffled WikiText-103 test) or 6 word-scrambled ~3000-char passages."""
    if variant not in ("natural", "scrambled"):
        raise ValueError(f"unknown variant {variant!r}")
    path = _passage_path(variant)
    if not path.exists():
        build_passages()
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------- line stream

def iter_lines(split: str = "train", min_chars: int = 30, max_chars: int = 70,
               seed: int = 0) -> Iterator[str]:
    """Infinite deterministic stream of word-boundary chunks of cleaned text.

    Each epoch shuffles paragraph order with ``Random(seed + epoch)``; each paragraph is cut
    into consecutive chunks whose target length is drawn uniformly from [min_chars, max_chars].
    Chunks are whole words joined by single spaces, no leading/trailing space, and always
    ``min_chars <= len <= max_chars``. Same arguments -> same sequence. Take a finite prefix
    (e.g. ``itertools.islice``) for a fixed validation set.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    pars = read_paragraphs(split)
    epoch = 0
    while True:
        rng = random.Random(seed * 1_000_003 + epoch)
        order = list(range(len(pars)))
        rng.shuffle(order)
        for pi in order:
            words = pars[pi].split(" ")
            i = 0
            while i < len(words):
                target = rng.randint(min_chars, max_chars)
                chunk = words[i]
                i += 1
                while i < len(words) and len(chunk) + 1 + len(words[i]) <= target:
                    chunk += " " + words[i]
                    i += 1
                if min_chars <= len(chunk) <= max_chars:
                    yield chunk
        epoch += 1
