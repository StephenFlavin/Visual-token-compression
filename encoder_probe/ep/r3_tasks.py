"""Round 3 tasks: prompts and short answers built from rendered pages (R3-1) and long documents (R3-2).

Everything here is plain Python on strings and numpy RNGs, so it is shared by finetune_qwen.py, longctx_bench.py
and the CPU tests. The answer text is always the decoded plain text (for codes: the original characters, never
class labels).

Page text
  * ``page.refs`` = one ref per line, exactly what rounds 1-2 scored (normalised source text for Latin and
    learned-glyph banks; raw fixed-length chunks for colour / mono / rgb3, which may start or end with a space).
  * transcription target of lines [a, b) = "\\n".join(refs[a:b]).
  * ``flat_text`` = the page's source text: refs joined by "" for raw-chunk encodings (consecutive slices of the
    source), by " " for word-wrapped ones (lines break at whitespace).

R3-1 task mix per sample (``task_mix`` = share of transcription, default 0.6; the rest split evenly):
  transcribe  the whole page, or a window of whole lines with at most ``max_chars`` target chars
              ("Transcribe lines a-b of the n lines on this page."), so dense pages stay short sequences
  cloze       'What word follows "w1 w2 w3" on this page?' -> the next word (context unique on the page)
  mc          'Which of these phrases appears on this page? A) .. D)' -> the letter (distractors from other pages)
  copy        'Copy the k words that follow "w1 w2 w3" on this page.' -> those k words
"""
from __future__ import annotations

import re

import numpy as np

TRANSCRIBE_ALL = "Transcribe all the text on this page."
LETTERS = "ABCD"
_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


def is_raw_chunks(enc) -> bool:
    """True if the encoding's refs are raw fixed-length chunks (its clean_hyp keeps boundary spaces)."""
    return enc.clean_hyp(" a ") == " a "


def flat_text(enc, page) -> str:
    return ("" if is_raw_chunks(enc) else " ").join(page.refs)


def transcript(page, a: int = 0, b: int | None = None) -> str:
    return "\n".join(page.refs[a:b])


def clean_transcript(enc, hyp: str) -> str:
    """Hypothesis post-processing per line, matching how refs were made (rounds 1-2 ``clean_hyp``)."""
    return "\n".join(enc.clean_hyp(ln) for ln in hyp.strip("\n").split("\n"))


def windows(lines: list[str], max_chars: int) -> list[tuple[int, int]]:
    """Greedy cover of all lines by consecutive windows [a, b) of at most ``max_chars`` target chars
    ("\\n" counted); a single longer line is its own window."""
    out, a = [], 0
    while a < len(lines):
        b, n = a + 1, len(lines[a])
        while b < len(lines) and n + 1 + len(lines[b]) <= max_chars:
            n += 1 + len(lines[b])
            b += 1
        out.append((a, b))
        a = b
    return out


def window_from(lines: list[str], a: int, max_chars: int) -> tuple[int, int]:
    b, n = a + 1, len(lines[a])
    while b < len(lines) and n + 1 + len(lines[b]) <= max_chars:
        n += 1 + len(lines[b])
        b += 1
    return a, b


def transcribe_prompt(a: int, b: int, n: int) -> str:
    if a == 0 and b == n:
        return TRANSCRIBE_ALL
    return f"Transcribe lines {a + 1}-{b} of the {n} lines on this page."


def transcribe_sample(page, rng: np.random.Generator, max_chars: int) -> tuple[str, str, dict]:
    n = len(page.refs)
    if len(transcript(page)) <= max_chars:
        a, b = 0, n
    else:
        a, b = window_from(page.refs, int(rng.integers(n)), max_chars)
    return transcribe_prompt(a, b, n), transcript(page, a, b), {"task": "transcribe", "lines": (a, b)}


def eval_windows(page, max_chars: int, k: int, lines: list[str] | None = None) -> list[tuple[int, int]]:
    """``k`` deterministic windows spread evenly over the page's lines, or over ``lines`` (one stream)."""
    w = windows(page.refs if lines is None else lines, max_chars)
    if len(w) <= k:
        return w
    return [w[i] for i in sorted({int(round(x)) for x in np.linspace(0, len(w) - 1, k)})]


def norm_word(w: str) -> str:
    return _PUNCT.sub("", w).lower()


def _unique_context(words: list[str], flat: str, rng, n_ctx: int, need_after: int, tries: int = 60):
    """-> index i such that words[i-n_ctx:i] occurs exactly once in ``flat`` and i+need_after <= len(words)."""
    lo, hi = n_ctx, len(words) - need_after
    if hi < lo:
        return None
    for _ in range(tries):
        i = int(rng.integers(lo, hi + 1))
        ctx = " ".join(words[i - n_ctx:i])
        if flat.count(ctx) == 1 and norm_word(words[i]):
            return i
    return None


def cloze_sample(flat: str, rng, n_ctx: int = 3):
    words = flat.split()
    i = _unique_context(words, flat, rng, n_ctx, 1)
    if i is None:
        return None
    ctx = " ".join(words[i - n_ctx:i])
    return (f'What word follows "{ctx}" on this page? Answer with that word only.', words[i],
            {"task": "cloze", "context": ctx})


def copy_sample(flat: str, rng, n_ctx: int = 3, k_range=(5, 20)):
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    words = flat.split()
    i = _unique_context(words, flat, rng, n_ctx, k)
    if i is None:
        return None
    ctx = " ".join(words[i - n_ctx:i])
    return (f'Copy the {k} words that follow "{ctx}" on this page.', " ".join(words[i:i + k]),
            {"task": "copy", "context": ctx, "k": k})


def mc_sample(flat: str, others: list[str], rng, n_words: int = 4, tries: int = 60):
    """Multiple choice: one 4-word phrase of the page and 3 from ``others`` (other pages) absent from it."""
    words = flat.split()
    if len(words) < n_words or not others:
        return None
    i = int(rng.integers(0, len(words) - n_words + 1))
    right = " ".join(words[i:i + n_words])
    wrong: list[str] = []
    for _ in range(tries):
        if len(wrong) == 3:
            break
        ow = others[int(rng.integers(len(others)))].split()
        if len(ow) < n_words:
            continue
        j = int(rng.integers(0, len(ow) - n_words + 1))
        ph = " ".join(ow[j:j + n_words])
        if ph not in flat and ph != right and ph not in wrong:
            wrong.append(ph)
    if len(wrong) < 3:
        return None
    pos = int(rng.integers(4))
    opts = wrong[:pos] + [right] + wrong[pos:]
    q = "Which of these phrases appears on this page? " + " ".join(
        f'{LETTERS[k]}) "{o}"' for k, o in enumerate(opts)) + " Answer with the letter only."
    return q, LETTERS[pos], {"task": "mc", "options": opts}


def page_sample(page, flat: str, others: list[str], rng, max_chars: int, task_mix: float = 0.6):
    """One (prompt, answer, meta) for a training page: transcription with prob ``task_mix``, else a question
    (cloze / mc / copy, equally likely; falls back to transcription if the page has no usable context)."""
    if getattr(page, "streams", None):
        return stream_page_sample(page, others, rng, max_chars, task_mix)
    if rng.random() >= task_mix:
        kind = ("cloze", "mc", "copy")[int(rng.integers(3))]
        s = (cloze_sample(flat, rng) if kind == "cloze" else mc_sample(flat, others, rng) if kind == "mc"
             else copy_sample(flat, rng))
        if s is not None:
            return s
    return transcribe_sample(page, rng, max_chars)


# ----------------------------------------------------------------- colour-gist streams (docs/gpu/COLOUR_GIST.md)
# A StreamPage (geom.SizedStreams) holds k text streams, each in its own colour. Every question names the stream
# by its ink colour; the MC "lure" is a phrase from ANOTHER stream on the same page (for 1 stream: from the text
# that follows the page), so a model that cannot tell the streams apart picks it.

def stream_flat(page, s: int) -> str:
    return " ".join(page.streams[s])


def _all_flat(page) -> str:
    return " | ".join(stream_flat(page, t) for t in range(len(page.streams)))


def stream_transcribe_prompt(colour: str, a: int, b: int, n: int) -> str:
    if a == 0 and b == n:
        return f"Transcribe all the {colour} text on this page."
    return f"Transcribe lines {a + 1}-{b} of the {n} lines of {colour} text on this page."


def stream_transcribe_sample(page, s: int, rng, max_chars: int):
    lines = page.streams[s]
    n = len(lines)
    if len("\n".join(lines)) <= max_chars:
        a, b = 0, n
    else:
        a, b = window_from(lines, int(rng.integers(n)), max_chars)
    return (stream_transcribe_prompt(page.colours[s], a, b, n), "\n".join(lines[a:b]),
            {"task": "transcribe", "stream": s, "lines": (a, b)})


def stream_cloze_sample(page, s: int, rng, n_ctx: int = 3):
    words = stream_flat(page, s).split()
    i = _unique_context(words, _all_flat(page), rng, n_ctx, 1)   # context unique across ALL streams
    if i is None:
        return None
    ctx = " ".join(words[i - n_ctx:i])
    return (f'In the {page.colours[s]} text, what word follows "{ctx}"? Answer with that word only.', words[i],
            {"task": "cloze", "stream": s, "context": ctx})


def stream_copy_sample(page, s: int, rng, n_ctx: int = 3, k_range=(5, 20)):
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    words = stream_flat(page, s).split()
    i = _unique_context(words, _all_flat(page), rng, n_ctx, k)
    if i is None:
        return None
    ctx = " ".join(words[i - n_ctx:i])
    return (f'In the {page.colours[s]} text, copy the {k} words that follow "{ctx}".', " ".join(words[i:i + k]),
            {"task": "copy", "stream": s, "context": ctx, "k": k})


def _phrase(text: str, rng, n_words: int, avoid: str, tries: int = 60):
    w = text.split()
    if len(w) < n_words:
        return None
    for _ in range(tries):
        j = int(rng.integers(0, len(w) - n_words + 1))
        ph = " ".join(w[j:j + n_words])
        if ph not in avoid:
            return ph
    return None


def stream_mc_sample(page, s: int, others: list[str], rng, n_words: int = 4):
    """One phrase of stream s, a LURE (another stream on the page; 1 stream: the text after the page) and 2
    phrases of other pages. meta["lure"] = the lure's letter."""
    flat, allf = stream_flat(page, s), _all_flat(page)
    right = _phrase(flat, rng, n_words, "")
    if right is None or not others:
        return None
    k = len(page.streams)
    lure_src = stream_flat(page, (s + 1 + int(rng.integers(k - 1))) % k) if k > 1 else page.after
    lure = _phrase(lure_src, rng, n_words, flat)
    wrong = []
    for _ in range(60):
        if len(wrong) == 2:
            break
        ph = _phrase(others[int(rng.integers(len(others)))], rng, n_words, allf + " " + page.after)
        if ph and ph != right and ph != lure and ph not in wrong:
            wrong.append(ph)
    if lure is None or len(wrong) < 2:
        return None
    opts = [right, lure] + wrong
    order = rng.permutation(4)
    q = (f"Which of these phrases appears in the {page.colours[s]} text? "
         + " ".join(f'{LETTERS[j]}) "{opts[o]}"' for j, o in enumerate(order)) + " Answer with the letter only.")
    pos = int(np.where(order == 0)[0][0])
    return q, LETTERS[pos], {"task": "mc", "stream": s, "options": [opts[o] for o in order],
                             "lure": LETTERS[int(np.where(order == 1)[0][0])]}


# gist: word presence (order-free, spelling-light): does content word w occur in stream s?
_STOP = set("""about above after again against also among around because been before being below between both
could does doing down during each from further have having here into itself just more most much must never once
only other over same should since some such than that their theirs them then there these they this those through
under until upon very were what when where which while whom whose will with within without would your also later
first second third following during including made became became""".split())
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")


def content_words(text: str) -> set[str]:
    """Distinct lower-cased words of >= 4 letters that are not stop words."""
    return {w.lower().strip("'-") for w in _WORD.findall(text)} - _STOP


def stream_word_sample(page, s: int, others: list[str], rng, kind: str | None = None):
    """'Does the word "w" appear in the <colour> text? Answer yes or no.' kind: present (a content word of stream
    s), absent (a content word of another page, not on this page) or lure (on the page, other stream only; k > 1).
    Default for training: present 1/2, absent 1/3, lure 1/6 (absent if k = 1)."""
    here = content_words(stream_flat(page, s))
    page_all = content_words(_all_flat(page))
    if kind is None:
        x = rng.random()
        kind = "present" if x < 0.5 else "absent" if x < 5 / 6 or len(page.streams) == 1 else "lure"
    if kind == "present":
        pool = sorted(here)
    elif kind == "lure":
        pool = sorted(page_all - here)
    else:
        pool = sorted({w for o in others for w in content_words(o)} - page_all)
    if not pool:
        return None
    w = pool[int(rng.integers(len(pool)))]
    return (f'Does the word "{w}" appear in the {page.colours[s]} text? Answer yes or no.',
            "yes" if kind == "present" else "no", {"task": "word", "stream": s, "kind": kind, "word": w})


def stream_page_sample(page, others: list[str], rng, max_chars: int, task_mix: float = 0.6,
                       kinds: tuple = ("cloze", "mc", "copy")):
    """page_sample for a StreamPage: pick a stream uniformly, then a task as in page_sample (``kinds`` = the
    question types, equally likely; "word" = gist word presence)."""
    s = int(rng.integers(len(page.streams)))
    if rng.random() >= task_mix:
        kind = kinds[int(rng.integers(len(kinds)))]
        out = (stream_cloze_sample(page, s, rng) if kind == "cloze" else
               stream_mc_sample(page, s, others, rng) if kind == "mc" else
               stream_word_sample(page, s, others, rng) if kind == "word" else stream_copy_sample(page, s, rng))
        if out is not None:
            return out
    return stream_transcribe_sample(page, s, rng, max_chars)


def answer_correct(task: str, ref: str, hyp: str) -> bool:
    """Short-answer scoring: cloze = first word (punctuation and case ignored); mc = first letter;
    needle = the digits appear; copy = exact after whitespace normalisation."""
    h = hyp.strip()
    if task == "cloze":
        w = h.split()
        return bool(w) and norm_word(w[0]) == norm_word(ref)
    if task == "mc":
        m = re.search(r"[A-D]", h)
        return bool(m) and m.group(0) == ref
    if task == "needle":
        return ref in re.sub(r"[^0-9]", " ", h).split()
    return " ".join(h.split()) == " ".join(ref.split())


# ------------------------------------------------------------------------------------------- R3-2 documents

NEEDLE_NAMES = ["the blue heron", "the copper kettle", "the north tower", "the silver fox", "the old mill",
                "the red lantern", "the glass harbour", "the quiet orchard", "the iron bridge", "the green comet"]


def make_document(text: str, length: int, rng) -> str:
    """``length`` chars (+/- one word) of ``text`` from a random offset, cut at word boundaries."""
    if len(text) < length + 2:
        raise ValueError(f"held-out text has {len(text)} chars, need {length}")
    off = int(rng.integers(0, len(text) - length))
    s = text.find(" ", off) + 1 if off else 0
    e = text.rfind(" ", s, s + length)
    return text[s:e if e > s else s + length]


def insert_needle(doc: str, depth: float, rng) -> tuple[str, str, str, int]:
    """Insert 'The secret number of <name> is <6 digits>.' at the sentence boundary nearest depth*len(doc).

    -> (document, question, answer digits, insert position)."""
    name = NEEDLE_NAMES[int(rng.integers(len(NEEDLE_NAMES)))]
    num = str(int(rng.integers(100000, 1000000)))
    sent = f"The secret number of {name} is {num}."
    target = int(round(depth * len(doc)))
    cands = [m.end() for m in re.finditer(r"\. ", doc)] or [0]
    pos = 0 if depth <= 0 else len(doc) if depth >= 1 else min([0, len(doc)] + cands, key=lambda p: abs(p - target))
    if pos == 0:
        out = sent + " " + doc
    elif pos >= len(doc):
        out = doc.rstrip() + " " + sent
    else:
        out = doc[:pos] + sent + " " + doc[pos:]
    q = f"What is the secret number of {name}? Answer with the number only."
    return out, q, num, pos


def doc_cloze(doc: str, rng, n_ctx: int = 6, avoid: tuple[int, int] | None = None):
    """'In the document, which word comes right after "<6 words>"?' with a context unique in the document."""
    words = doc.split()
    for _ in range(100):
        i = int(rng.integers(n_ctx, len(words)))
        ctx = " ".join(words[i - n_ctx:i])
        if doc.count(ctx) != 1 or not norm_word(words[i]):
            continue
        at = doc.find(ctx)
        if avoid and avoid[0] - 200 <= at <= avoid[1] + 200:
            continue
        return (f'In the document, which word comes right after "{ctx}"? Answer with that word only.',
                words[i], {"context": ctx, "char_pos": at, "depth": at / max(1, len(doc))})
    return None
