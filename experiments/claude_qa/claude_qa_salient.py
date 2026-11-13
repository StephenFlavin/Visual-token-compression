"""Content-chosen mixed resolution: long-document Q&A with the facts drawn large and the filler drawn small.

The user's idea (2026-11-13): instead of grading resolution by position (``claude_qa_gradient.py``), mix two type
sizes throughout every page, chosen by content: what must be read exactly (numbers, names, words that flip meaning)
large and safe, the filler around it small and cheap, with the choice made by a cheap rule-based detector (regular
expressions and fixed word lists, no model call). Same v2 suite as the uniform arms (3 documents, 120 questions,
same system prompt and prompt, Opus 5.5 ``--effort high``, pages <= 1932 px), at the token count of uniform 58.8
(4.2x fewer tokens than text, where uniform pages lose 8 points).

Arms:
* ``salient_large``  salient words at the em of the 117.6 px²/char setting, filler at the em that makes the total
                     image tokens equal uniform 58.8's (solved per document);
* ``salient_small``  control: salient words at that small em, filler at the em re-solved to the same total;
* ``salient_large_small_words`` (only if the cap allows): ``salient_large`` on Test A's document with its 40 extra
                     "small-word" questions (``claude_qa_prune.build_doc``), all 160 questions in one call.

    python3 experiments/claude_qa/claude_qa_salient.py check            # free: detector coverage, sizes, tokens, sample
    python3 experiments/claude_qa/claude_qa_salient.py run --yes        # the calls (resumable; rows appended per call)
    python3 experiments/claude_qa/claude_qa_salient.py analyse          # works on whatever rows exist
Pre-registration and results: results/claude_qa/salient/README.md.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "claude_qa"))
import analyse_claude_qa as A  # noqa: E402
import claude_qa_gradient as G  # noqa: E402
import claude_qa_render as R  # noqa: E402
from vbpe import config  # noqa: E402
from vbpe.layout import get_font, glyph_advance  # noqa: E402

OUT = config.RESULTS / "claude_qa" / "salient"
QA = config.RESULTS / "claude_qa"
FONT = str(config.LATIN_FONT)            # DejaVu Sans, the font of every Latin page in this repo
PAGE_W = 1932
PAGE_MAX_H = 1932
PATCH = 28
MARGIN = 4                               # vbpe.layout.LayoutParams defaults
LINE_SPACING = 1.2
EM_LARGE = 13.76                         # em of the 117.6 px²/char pages: median over the 67 full pages of d0-d2
SAFE_PPC = 117.6
TARGET = 58.8                            # equal tokens to uniform 58.8
TOL = 0.02
ARMS = ("salient_large", "salient_small")
SW_ARM = "salient_large_small_words"
REPEATS = [0, 1]
CAP_USD = 12.0
SW_MIN_LEFT = 4.0                        # the small-word arm runs only if at least $4 of the cap is left
EST_PER_CALL = 1.0
LOOKUP = G.LOOKUP
SUBS = G.SUBS
REFUSAL = G.REFUSAL


def salt(rep: int) -> str | None:
    return None if rep == 0 else f"salient{rep}"


# =========================================================================== detector (frozen in the README)
# 1. numbers: any word containing a digit; a word made only of %, $, £, €, ¥; spelled-out number words below.
NUMBER_WORDS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million
billion trillion hundreds thousands millions billions dozen dozens half quarter percent first second third fourth
fifth sixth seventh eighth ninth tenth eleventh twelfth twentieth hundredth thousandth once twice thrice double
triple""".split())
SYMBOL_WORD = re.compile(r"^[%$£€¥]+$")
# 3. words that flip meaning (single words; one line of justification each is in the README)
FLIP_WORDS = {"not", "no", "never", "none", "neither", "nor", "without", "except", "unless", "only", "fewer",
              "less", "before", "after", "instead"}
# multi-word phrases: every word of the phrase is marked
FLIP_PHRASES = [("all", "but"), ("at", "most"), ("at", "least"), ("more", "than"), ("less", "than"),
                ("fewer", "than"), ("rather", "than"), ("instead", "of"), ("no", "longer")]
# "by" / "from" / "to" only after a verb of payment or transfer, within the next 8 words of the same sentence
TRANSFER_VERBS = {"paid", "pay", "pays", "paying", "sold", "sell", "sells", "bought", "buy", "buys", "gave", "give",
                  "given", "gives", "sent", "send", "sends", "lent", "lend", "loaned", "owed", "owe", "owes",
                  "shipped", "ship", "transferred", "transfer", "donated", "donate", "received", "receive", "traded",
                  "trade"}
TRANSFER_PREPS = {"by", "from", "to"}
TRANSFER_WINDOW = 8
NUMBER_PAD = 1                           # numbers are widened by one word on each side; names and flip words are not

WORD = re.compile(r"\S+")
CORE = re.compile(r"^[^\w]*(.*?)[^\w]*$", re.S)        # strip leading / trailing punctuation
SENT_END = re.compile(r"[.!?][\"'”’)\]]*$")


def core(w: str) -> str:
    c = CORE.match(w).group(1)
    return re.sub(r"(?:'s|’s)$", "", c)


def is_cap(w: str) -> bool:
    c = core(w)
    return bool(c) and c[0].isupper()


def detect_words(doc: str) -> tuple[list[tuple[int, int]], list[set]]:
    """-> (word spans, per word the set of reasons it is salient: 'number', 'number_pad', 'name', 'flip')."""
    spans = [m.span() for m in WORD.finditer(doc)]
    words = [doc[s:e] for s, e in spans]
    cores = [core(w) for w in words]
    low = [c.lower() for c in cores]
    n = len(words)
    start = [i == 0 or bool(SENT_END.search(words[i - 1])) for i in range(n)]
    why: list[set] = [set() for _ in range(n)]
    # 1. numbers
    for i, (w, lc) in enumerate(zip(words, low)):
        parts = [p for p in re.split(r"[-–]", lc) if p]
        if any(ch.isdigit() for ch in w) or SYMBOL_WORD.match(w) or any(p in NUMBER_WORDS for p in parts):
            why[i].add("number")
    for i in range(n):           # padding never crosses a sentence end
        if "number" in why[i]:
            for k in range(1, NUMBER_PAD + 1):
                if i - k >= 0 and not SENT_END.search(words[i - k]):
                    why[i - k].add("number_pad")
                if i + k < n and not SENT_END.search(words[i + k - 1]):
                    why[i + k].add("number_pad")
    # 2. names: capitalised words not at a sentence start; at a sentence start only if the same word occurs
    #    capitalised mid-sentence elsewhere and never in lower case in the document
    mid_cap = {cores[i] for i in range(n) if not start[i] and is_cap(words[i])}
    lower_forms = {c for c in cores if c and c[0].islower()}
    for i in range(n):
        if not is_cap(words[i]) or cores[i] == "I":
            continue
        if not start[i] or (cores[i] in mid_cap and cores[i].lower() not in lower_forms):
            why[i].add("name")
    # 3. flip words, phrases, and by/from/to after a transfer verb
    for i, lc in enumerate(low):
        if lc in FLIP_WORDS or lc.endswith("n't") or lc.endswith("n’t"):
            why[i].add("flip")
    for i in range(n - 1):
        for ph in FLIP_PHRASES:
            if tuple(low[i:i + len(ph)]) == ph:
                for j in range(i, i + len(ph)):
                    why[j].add("flip")
    for i, lc in enumerate(low):
        if lc not in TRANSFER_VERBS:
            continue
        for j in range(i + 1, min(n, i + 1 + TRANSFER_WINDOW)):
            if low[j] in TRANSFER_PREPS:
                why[j].add("flip")
            if SENT_END.search(words[j]):
                break
    return spans, why


def salient_mask(doc: str) -> np.ndarray:
    """Boolean per character of ``doc``: True inside a salient word (whole words; spaces between two salient words
    are salient too)."""
    spans, why = detect_words(doc)
    m = np.zeros(len(doc), dtype=bool)
    prev_end, prev_sal = None, False
    for (s, e), w in zip(spans, why):
        sal = bool(w)
        if sal:
            m[s:e] = True
            if prev_sal and prev_end is not None:
                m[prev_end:s] = True
        prev_end, prev_sal = e, sal
    return m


def coverage(doc: str, mask: np.ndarray, strings: list[str]) -> float:
    """Share of the non-space characters of the given strings (each located once in doc) that are salient."""
    tot = hit = 0
    for a in strings:
        s = doc.find(a)
        assert s >= 0, a
        for k, ch in enumerate(a):
            if not ch.isspace():
                tot += 1
                hit += bool(mask[s + k])
    return hit / tot if tot else float("nan")


# =========================================================================== two-size renderer
class Flow:
    """The document as words with a size class each (True = salient). A space between two salient words is drawn at
    the salient size, every other space at the filler size."""

    def __init__(self, doc: str, mask: np.ndarray):
        self.words, self.big = [], []
        for m in WORD.finditer(doc):
            s, e = m.span()
            self.words.append(m.group())
            self.big.append(bool(mask[s:e].any()))

    def widths(self, em_sal: float, em_fill: float) -> tuple[list[float], list[float], list[float]]:
        """-> (word width, em of the word, width of the space after it)."""
        cache: dict = {}

        def w(text: str, em: float) -> float:
            k = (text, em)
            if k not in cache:
                cache[k] = sum(glyph_advance(FONT, em, ch) for ch in text)
            return cache[k]
        em_sal, em_fill = round(em_sal, 3), round(em_fill, 3)
        ems = [em_sal if b else em_fill for b in self.big]
        ww = [w(t, e) for t, e in zip(self.words, ems)]
        sp_s, sp_f = glyph_advance(FONT, em_sal, " "), glyph_advance(FONT, em_fill, " ")
        n = len(self.words)
        sp = [sp_s if (self.big[i] and i + 1 < n and self.big[i + 1]) else sp_f for i in range(n)]
        return ww, ems, sp


def layout(flow: Flow, em_sal: float, em_fill: float, W: int = PAGE_W, H: int = PAGE_MAX_H) -> list[dict]:
    """Greedy word wrap at mixed sizes, then greedy pagination. Each line's pitch = 1.2 x the largest em on it;
    its baseline sits 0.9 x that em below the line's top (as vbpe.layout places a uniform line). A page takes lines
    while margin + pitches + the last line's ink box (0.1 + 0.8 + 0.25 em) + margin <= H.
    -> pages: {'lines': [(first word, end word, em_max, baseline y)], 'H': height (multiple of 28)}."""
    ww, ems, sp = flow.widths(em_sal, em_fill)
    maxw = W - 2 * MARGIN
    lines, i, n = [], 0, len(ww)
    while i < n:
        j, x, m = i, 0.0, 0.0
        while j < n:
            add = ww[j] if j == i else sp[j - 1] + ww[j]
            if j > i and x + add > maxw:
                break
            x += add
            m = max(m, ems[j])
            j += 1
        lines.append((i, j, m))
        i = j
    pages, cur, top = [], [], MARGIN
    for (a, b, m) in lines:
        need = top + 1.15 * m + MARGIN
        if cur and need > H:
            pages.append(cur)
            cur, top = [], MARGIN
        cur.append((a, b, m, top + 0.9 * m))
        top += LINE_SPACING * m
    if cur:
        pages.append(cur)
    out = []
    for k, pl in enumerate(pages):
        a, b, m, base = pl[-1]
        used = base + 0.25 * m + MARGIN
        h = H if k < len(pages) - 1 else min(H, max(2 * PATCH, math.ceil(used / PATCH) * PATCH))
        out.append({"lines": pl, "H": h})
    return out


def tokens_of(pages: list[dict], W: int = PAGE_W) -> int:
    return sum(math.ceil(W / PATCH) * math.ceil(p["H"] / PATCH) + 3 for p in pages)


def draw(flow: Flow, pages: list[dict], em_sal: float, em_fill: float, W: int = PAGE_W) -> list[Image.Image]:
    ww, ems, sp = flow.widths(em_sal, em_fill)
    out = []
    for p in pages:
        img = Image.new("L", (W, p["H"]), 255)
        d = ImageDraw.Draw(img)
        for a, b, m, base in p["lines"]:
            x = float(MARGIN)
            for j in range(a, b):
                f = get_font(FONT, round(ems[j], 3))
                for ch in flow.words[j]:
                    d.text((x, base), ch, font=f, fill=0, anchor="ls")
                    x += glyph_advance(FONT, round(ems[j], 3), ch)
                if j < b - 1:
                    x += sp[j]
        out.append(img)
    return out


def solve_em(flow: Flow, target: int, fixed: float, which: str, lo: float = 1.0, hi: float = 24.0) -> float:
    """The largest em (to 0.01 px) for the free size class (``which`` = 'fill' or 'sal') at which the document's
    image tokens stay <= target; the other class is drawn at ``fixed``."""
    def tok(em):
        return tokens_of(layout(flow, fixed, em) if which == "fill" else layout(flow, em, fixed))
    if tok(lo) > target:
        return lo
    for _ in range(40):
        mid = (lo + hi) / 2
        if tok(mid) <= target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.005:
            break
    em = math.floor(lo * 100) / 100
    while tok(em) > target:          # tokens are not strictly monotone in em: step down until under the target
        em = round(em - 0.01, 2)
    return em


def filler_density(flow: Flow, pages: list[dict], em_sal: float, em_fill: float) -> dict:
    """px² per character as laid out, split by size class: each character is charged its advance x its line's pitch
    (spaces charged to the class they are drawn in)."""
    ww, ems, sp = flow.widths(em_sal, em_fill)
    area = {True: 0.0, False: 0.0}
    chars = {True: 0, False: 0}
    for p in pages:
        for a, b, m, _ in p["lines"]:
            pitch = LINE_SPACING * m
            for j in range(a, b):
                area[flow.big[j]] += ww[j] * pitch
                chars[flow.big[j]] += len(flow.words[j])
                if j < b - 1:
                    sb = flow.big[j] and flow.big[j + 1]
                    area[sb] += sp[j] * pitch
                    chars[sb] += 1
    return {"salient": area[True] / max(1, chars[True]), "filler": area[False] / max(1, chars[False]),
            "salient_chars": chars[True], "filler_chars": chars[False]}


def uniform_density(em: float) -> float:
    """px²/char of a uniform page at this em with this renderer: 1.2 em line pitch x DejaVu Sans' mean advance on
    the measured character mix (0.51 em/char) -- the density a whole page in this size would have."""
    return LINE_SPACING * em * 0.5108 * em


# =========================================================================== questions: what each one needs
def needed_numbers(doc: str, q: dict) -> list[tuple[int, int]]:
    """Character spans of the numbers a question's answer is read from (none for counts)."""
    if q.get("sub") == "count":
        return []
    spans = G.anchor_spans(doc, q)
    nums = [(m.start(), m.end(), R.norm_num(m.group())) for s, e in spans for m in R.NUM.finditer(doc, s, e)]
    if q.get("sub") in ("sum", "max"):
        return [(s, e) for s, e, _ in nums]
    hit = [(s, e) for s, e, v in nums if v == q["answer"]]
    return hit or [(s, e) for s, e, _ in nums]


def annotate(doc: str, qs: list[dict], mask: np.ndarray) -> list[dict]:
    """Per question: share of the non-space characters of its anchor facts that are salient (anchor_cov), and of the
    digits of the numbers its answer is read from (num_cov; None for counts)."""
    out = []
    for q in qs:
        sp = G.anchor_spans(doc, q)
        ns = [k for s, e in sp for k in range(s, e) if not doc[k].isspace()]
        digits = [k for s, e in needed_numbers(doc, q) for k in range(s, e) if doc[k].isdigit()]
        out.append({**q, "anchor_cov": float(np.mean(mask[ns])) if ns else None,
                    "num_cov": float(np.mean(mask[digits])) if digits else None})
    return out


# =========================================================================== calls
def load_doc(d: int, small_words: bool) -> tuple[str, list[dict]]:
    if small_words:
        import claude_qa_prune as P
        return P.build_doc(d)
    return G.build_doc(d)


def target_tokens(d: int, small_words: bool) -> int:
    """Image tokens of the uniform 58.8 pages of the same document (v2 row; Test A's page59 row for small words)."""
    if small_words:
        p = QA / "prune" / f"d{d}" / "runs.jsonl"
        return next(json.loads(x)["image_tokens_formula"] for x in p.read_text().splitlines()
                    if x.strip() and json.loads(x)["arm"] == "page59")
    return G.v2_rows(d)[58.8]["image_tokens_formula"]


def prepare(d: int, small_words: bool = False, cache: dict | None = None) -> dict:
    """Document, mask, questions, flows and both arms' sizes for document d (memoised in ``cache``)."""
    cache = cache if cache is not None else {}
    key = (d, small_words)
    if key in cache:
        return cache[key]
    doc, qs = load_doc(d, small_words)
    mask = salient_mask(doc)
    flow = Flow(doc, mask)
    tgt = target_tokens(d, small_words)
    em_s = solve_em(flow, tgt, EM_LARGE, "fill")
    em_f2 = solve_em(flow, tgt, em_s, "fill")          # control: salient at em_s, filler re-solved
    sizes = {"salient_large": (EM_LARGE, em_s), "salient_small": (em_s, em_f2)}
    c = {"d": d, "small_words": small_words, "doc": doc, "qs": annotate(doc, qs, mask), "mask": mask, "flow": flow,
         "target": tgt, "sizes": sizes}
    cache[key] = c
    return c


def call_spec(d: int, arm: str, cache: dict | None = None, images: bool = True) -> dict:
    sw = arm == SW_ARM
    c = prepare(d, sw, cache)
    em_sal, em_fill = c["sizes"]["salient_large" if sw else arm]
    pages = layout(c["flow"], em_sal, em_fill)
    imgs = draw(c["flow"], pages, em_sal, em_fill) if images else []
    if sw:
        import claude_qa_prune as P
        prompt = P.prompt("page59", c["doc"], c["qs"], len(pages))
    else:
        prompt = R.prompt_for("image", c["doc"], c["qs"], len(pages))
    return {"d": d, "arm": arm, "doc": c["doc"], "qs": c["qs"], "pages": imgs, "layout": pages, "em_sal": em_sal,
            "em_fill": em_fill, "vt": tokens_of(pages), "target": c["target"], "system": R.SYSTEM, "prompt": prompt,
            "mask": c["mask"], "flow": c["flow"]}


def parse(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*\**([NSCW]\d{2})\**\s*[:\-]\s*(.+)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def load_rows() -> list[dict]:
    rows = {}
    p = OUT / "runs.jsonl"
    for x in p.read_text().splitlines() if p.exists() else []:
        if x.strip():
            r = json.loads(x)
            rows[(r["doc"], r["arm"], r["repeat"])] = r
    return [rows[k] for k in sorted(rows)]


def spent() -> float:
    p = OUT / "spend.jsonl"
    return sum(json.loads(x)["cost_usd"] for x in p.read_text().splitlines() if x.strip()) if p.exists() else 0.0


def est_per_call() -> float:
    p = OUT / "spend.jsonl"
    costs = [json.loads(x)["cost_usd"] for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return max([EST_PER_CALL] + [1.3 * c for c in costs])


def make_row(spec: dict, rep: int, rec: dict) -> dict:
    qs = spec["qs"]
    ans = parse(rec.get("text", ""))
    per = [{"id": q["id"], "kind": q["kind"], "sub": q.get("sub"), "pos": q["pos"], "answer": q["answer"],
            "lure": q.get("lure"), "anchor_cov": q["anchor_cov"], "num_cov": q["num_cov"], "given": ans.get(q["id"]),
            "correct": ans.get(q["id"]) is not None and R.norm_num(ans[q["id"]]) == q["answer"]} for q in qs]
    incomplete = bool(rec.get("input_tokens") is not None and rec["input_tokens"] < 0.97 * spec["vt"])
    refused = bool(rec.get("stop_reason") == "refusal" or (rec.get("error") and REFUSAL.search(rec["error"]))
                   or (not ans and REFUSAL.search(rec.get("text", ""))))
    lay = spec["layout"]
    row = {"arm": spec["arm"], "doc": spec["d"], "repeat": rep, "cache_salt": salt(rep), "em_salient": spec["em_sal"],
           "em_filler": spec["em_fill"], "chars": len(spec["doc"]), "salient_share": float(spec["mask"].mean()),
           "n_pages": len(lay), "page_heights": [p["H"] for p in lay], "image_tokens_formula": spec["vt"],
           "target_tokens": spec["target"], "effort": "high", "model": rec.get("model"),
           "input_tokens": rec.get("input_tokens"), "output_tokens": rec.get("output_tokens"), "usage": rec.get("usage"),
           "cost_usd": rec.get("cost_usd"), "latency_s": rec.get("latency_s"), "cached": rec.get("cached"),
           "error": rec.get("error"), "stop_reason": rec.get("stop_reason"), "cache_key": rec.get("cache_key"),
           "incomplete": incomplete, "refused": refused, "page_px": PAGE_W, "n_questions": len(qs),
           "acc_all": float(np.mean([p["correct"] for p in per])),
           "acc": {k: float(np.mean([p["correct"] for p in per if p["kind"] == k])) for k in sorted({q["kind"] for q in qs})},
           "per_question": per, "raw": rec.get("text", "")}
    row["unanswered"], row["disengaged"] = A.engagement(row)
    return row


def todo_jobs(rows: list[dict], docs, small_words: bool = False) -> list[tuple[int, str, int]]:
    have = {(r["doc"], r["arm"], r["repeat"]) for r in rows}
    if small_words:
        return [(d, SW_ARM, 0) for d in docs if (d, SW_ARM, 0) not in have]
    return [(d, arm, rep) for rep in REPEATS for d in docs for arm in ARMS if (d, arm, rep) not in have]


# --------------------------------------------------------------------------- check (free)
def make_sample(c: dict, path: Path, width: int = 940, zoom: int = 2) -> None:
    """A strip of document 0 around its first planted fact, drawn by both arms at that document's sizes and the
    same column width (the real pages are 1932 px wide), at ``zoom`` x with nearest-neighbour scaling, so the pixels
    the model receives are visible."""
    doc, mask = c["doc"], c["mask"]
    at = doc.find(c["qs"][0]["fact"])
    s = doc.rfind(". ", 0, at - 600) + 2
    e = doc.find(". ", at + 700) + 1
    ex = doc[s:e]
    flow = Flow(ex, mask[s:e])
    strips = []
    for arm in ARMS:
        em_sal, em_fill = c["sizes"][arm]
        lay = layout(flow, em_sal, em_fill, W=width, H=4000)
        img = draw(flow, lay, em_sal, em_fill, W=width)[0]
        img = img.crop((0, 0, width, min(img.height, 150)))
        strips.append((arm, em_sal, em_fill, img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)))
    lab_h, gap = 34, 14
    W = width * zoom
    out = Image.new("L", (W, sum(lab_h + im.height + gap for *_, im in strips)), 255)
    d = ImageDraw.Draw(out)
    f = get_font(FONT, 20)
    y = 0
    for arm, es, ef, im in strips:
        d.text((4, y + 6), f"{arm}: salient words at em {es:.2f} px, filler at em {ef:.2f} px "
                           f"(document 0; shown at {zoom}x, nearest-neighbour)", font=f, fill=60)
        y += lab_h
        out.paste(im, (0, y))
        d.rectangle((0, y, W - 1, y + im.height - 1), outline=170)
        y += im.height + gap
    out.save(path, optimize=True)


def cmd_check(a) -> None:
    cache: dict = {}
    res = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for d in a.docs:
        stored = json.loads((QA / f"qav2_d{d}_high_b88" / "questions.json").read_text())
        c = prepare(d, False, cache)
        assert [(q["id"], q["question"], q["answer"]) for q in c["qs"]] == \
               [(q["id"], q["question"], q["answer"]) for q in stored], f"d{d}: questions differ from v2"
        assert len(c["doc"]) == G.v2_rows(d)[88.2]["chars"], "document differs from v2"
        doc, mask = c["doc"], c["mask"]
        ns = np.array([not ch.isspace() for ch in doc])
        _, why = detect_words(doc)
        reasons = {k: sum(k in w for w in why) for k in ("number", "number_pad", "name", "flip")}
        x = {"chars": len(doc), "salient_share_nonspace": float(mask[ns].mean()), "salient_share_all": float(mask.mean()),
             "words": len(why), "salient_words": sum(bool(w) for w in why), "words_by_reason": reasons,
             "target_tokens": c["target"]}
        by = {}
        for kind in ("needle", "number", "two_hop", "three_hop", "sum", "max", "count"):
            qq = [q for q in c["qs"] if (q.get("sub") or q["kind"]) == kind]
            nc = [q["num_cov"] for q in qq if q["num_cov"] is not None]
            by[kind] = {"n": len(qq), "answer_numbers_salient": float(np.mean(nc)) if nc else None,
                        "answer_numbers_fully_salient": sum(v == 1.0 for v in nc),
                        "anchor_chars_salient": float(np.mean([q["anchor_cov"] for q in qq]))}
        x["coverage_by_type"] = by
        for arm in ARMS:
            s = call_spec(d, arm, cache, images=(d == a.docs[0]))
            dens = filler_density(s["flow"], s["layout"], s["em_sal"], s["em_fill"])
            x[arm] = {"em_salient": s["em_sal"], "em_filler": s["em_fill"], "pages": len(s["layout"]),
                      "tokens": s["vt"], "target": s["target"], "ratio": s["vt"] / s["target"],
                      "laid_out_px2_per_char": dens,
                      "uniform_px2_per_char_of_small": uniform_density(min(s["em_sal"], s["em_fill"])),
                      "uniform_px2_per_char_of_large": uniform_density(max(s["em_sal"], s["em_fill"]))}
            assert abs(s["vt"] / s["target"] - 1) <= TOL, f"d{d} {arm}: tokens {s['vt']} vs {s['target']}"
            print(f"[check] d{d} {arm}: em salient {s['em_sal']:.2f} filler {s['em_fill']:.2f}; "
                  f"{len(s['layout'])} pages, {s['vt']:,} image tokens (uniform 58.8: {s['target']:,}; "
                  f"ratio {s['vt'] / s['target']:.3f}); laid out px²/char salient {dens['salient']:.1f}, filler "
                  f"{dens['filler']:.1f}; uniform px²/char of the small size {x[arm]['uniform_px2_per_char_of_small']:.1f}",
                  flush=True)
            if s["pages"]:
                sd = OUT / "images"          # git-ignored
                sd.mkdir(exist_ok=True)
                for k in (0, len(s["pages"]) // 2):
                    s["pages"][k].save(sd / f"d{d}_{arm}_p{k:02d}.png")
        print(f"[check] d{d}: salient {100 * x['salient_share_nonspace']:.1f}% of non-space characters "
              f"({x['salient_words']:,} of {x['words']:,} words; {reasons}); answer-number coverage by type "
              + ", ".join(f"{k} {v['answer_numbers_salient'] if v['answer_numbers_salient'] is None else round(100 * v['answer_numbers_salient'], 1)}"
                          for k, v in by.items()), flush=True)
        res[f"d{d}"] = x
        if d == 0:
            make_sample(c, ROOT / "docs" / "img" / "claude_salient_sample.png")
    if a.small_words:
        for d in a.docs:
            s = call_spec(d, SW_ARM, cache, images=False)
            c = prepare(d, True, cache)
            sw = [q for q in c["qs"] if q["kind"] == "small"]
            crit = []
            for q in sw:
                st = c["doc"].find(q["anchors"][0])
                words = [(m.start() + st, m.group()) for m in WORD.finditer(q["anchors"][0])]
                cw = [ (pos, w) for pos, w in words if core(w).lower() in [x.lower() for x in q["critical"]]]
                crit.append(float(np.mean([c["mask"][pos] for pos, _ in cw])) if cw else None)
            x = {"em_salient": s["em_sal"], "em_filler": s["em_fill"], "pages": len(s["layout"]), "tokens": s["vt"],
                 "target": s["target"], "ratio": s["vt"] / s["target"],
                 "small_word_numbers_salient": float(np.mean([q["num_cov"] for q in sw if q["num_cov"] is not None])),
                 "small_word_critical_words_salient": float(np.mean([v for v in crit if v is not None])),
                 "small_word_anchor_chars_salient": float(np.mean([q["anchor_cov"] for q in sw]))}
            assert abs(x["ratio"] - 1) <= TOL
            res[f"d{d} {SW_ARM}"] = x
            print(f"[check] d{d} {SW_ARM}: em {s['em_sal']:.2f} / {s['em_fill']:.2f}, {x['pages']} pages, "
                  f"{s['vt']:,} tokens (page59 {s['target']:,}); critical words salient "
                  f"{100 * x['small_word_critical_words_salient']:.0f}%, numbers {100 * x['small_word_numbers_salient']:.0f}%",
                  flush=True)
    jobs = todo_jobs(load_rows(), a.docs)
    print(f"[check] {len(jobs)} calls to make, at most ${est_per_call():.2f} each = ${len(jobs) * est_per_call():.2f}; "
          f"spent ${spent():.2f}; cap ${CAP_USD:.0f}", flush=True)
    (OUT / "dry_run.json").write_text(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- run
def cmd_run(a) -> None:
    from vbpe import claude_eval
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = todo_jobs(load_rows(), a.docs, a.small_words)
    if a.limit:
        jobs = jobs[:a.limit]
    e = est_per_call()
    left = CAP_USD - spent()
    print(f"[run] {len(jobs)} calls, at most ${e:.2f} each; spent ${spent():.2f}; cap ${CAP_USD:.0f}", flush=True)
    if a.small_words and left < SW_MIN_LEFT:
        print(f"[run] small-word arm not run: ${left:.2f} of the cap left (< ${SW_MIN_LEFT:.0f})", flush=True)
        return
    if not a.yes or not jobs:
        print("[run] add --yes to make the calls" if jobs else "[run] nothing to do", flush=True)
        return
    cache: dict = {}
    specs = {(d, arm): call_spec(d, arm, cache) for d, arm, _ in jobs}
    lock = threading.Lock()
    state = {"inflight": 0.0, "stop": False}

    def one(job):
        d, arm, rep = job
        with lock:
            if state["stop"]:
                return
            if spent() + state["inflight"] + e > CAP_USD:
                print(f"[run] d{d} {arm} r{rep}: skipped, could cross the ${CAP_USD:.0f} cap", flush=True)
                return
            state["inflight"] += e
        spec = specs[(d, arm)]
        rec = claude_eval.transcribe(spec["pages"], spec["system"], spec["prompt"], effort="high", timeout=1800.0,
                                     hires=True, cache_salt=salt(rep))
        row = make_row(spec, rep, rec)
        with lock:
            state["inflight"] -= e
            if not rec.get("cached"):
                with open(OUT / "spend.jsonl", "a") as f:
                    f.write(json.dumps({"what": f"d{d} {arm} r{rep}", "cost_usd": rec.get("cost_usd", 0.0),
                                        "input_tokens": rec.get("input_tokens"),
                                        "output_tokens": rec.get("output_tokens"), "error": rec.get("error")}) + "\n")
            if row["error"] and not row["refused"]:
                if re.search(r"usage limit|limit reached|quota", row["error"], re.I):
                    state["stop"] = True
                print(f"[run] d{d} {arm} r{rep}: ERROR (not stored, rerun): {row['error'][:300]}", flush=True)
                return
            with open(OUT / "runs.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"[run] d{d} {arm} r{rep}: all {row['acc_all']:.3f} {row['acc']}; in {row['input_tokens']:,} "
              f"(formula {row['image_tokens_formula']:,}) out {row['output_tokens']:,} ${row['cost_usd']:.2f} "
              f"{row['latency_s']:.0f}s incomplete {row['incomplete']} refused {row['refused']} "
              f"unanswered {row['unanswered']} disengaged {row['disengaged']}; spent ${spent():.2f}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, jobs))
    print(f"[run] done; spent ${spent():.2f}", flush=True)


# --------------------------------------------------------------------------- analysis
valid = G.valid
acc = G.acc
fmt = G.fmt
KINDS = ("needle", "number")


def arm_stats(entries: list[tuple[int, dict]], subs=SUBS) -> dict:
    per = [p for _, r in entries for p in r["per_question"]]
    return {"calls": len(entries), "docs": sorted({d for d, _ in entries}),
            "input_tokens_mean": float(np.mean([r["input_tokens"] for _, r in entries])),
            "cost_mean": float(np.mean([r["cost_usd"] for _, r in entries])),
            "per_call": [r["acc_all"] for _, r in entries],
            "all": acc([p["correct"] for p in per]),
            "lookup": acc([p["correct"] for p in per if p["kind"] in LOOKUP]),
            "complex": acc([p["correct"] for p in per if p["kind"] == "complex"]),
            **{k: acc([p["correct"] for p in per if p["kind"] == k]) for k in KINDS},
            "complex_by_sub": {s: acc([p["correct"] for p in per if p.get("sub") == s]) for s in subs}}


def paired(rows: list[dict], refs: dict, reps, ids=None) -> list:
    """(doc, ref correct, arm correct) for every question of the given rows; refs: {doc: reference row}."""
    out = []
    for r in rows:
        if r["repeat"] not in reps or r["doc"] not in refs:
            continue
        rq = {p["id"]: p["correct"] for p in refs[r["doc"]]["per_question"]}
        out += [(r["doc"], bool(rq[p["id"]]), bool(p["correct"])) for p in r["per_question"]
                if p["id"] in rq and (ids is None or ids(p))]
    return out


def test(pr: list) -> dict:
    diff = A.paired_diff_ci(pr)
    n10, n01, pm = A.mcnemar([x[1] for x in pr], [x[2] for x in pr])
    return {"n": len(pr), "arm_minus_ref": -diff[0], "lo": -diff[2], "hi": -diff[1], "ref_only": n10, "arm_only": n01,
            "p": pm}


def wrong_answers(rows: list[dict]) -> dict:
    """Wrong answers by type, and whether the numbers they are read from were drawn in salient (large) words."""
    out = {}
    for r in rows:
        for p in r["per_question"]:
            if p["correct"]:
                continue
            t = p.get("sub") or p["kind"]
            x = out.setdefault(t, {"wrong": 0, "number_salient": 0, "number_not_salient": 0, "no_number": 0,
                                   "anchor_cov": []})
            x["wrong"] += 1
            nc = p.get("num_cov")
            x["no_number" if nc is None else ("number_salient" if nc == 1.0 else "number_not_salient")] += 1
            x["anchor_cov"].append(p["anchor_cov"])
    for x in out.values():
        x["anchor_cov_mean"] = float(np.mean(x.pop("anchor_cov")))
    return out


def baseline_sw() -> dict:
    """Test A (prune) and repeat rows: {'page118': [(doc, row)], 'page59': [...], 'text': [...]} (all calls)."""
    out: dict = {}
    for d in range(3):
        p = QA / "prune" / f"d{d}" / "runs.jsonl"
        for x in p.read_text().splitlines() if p.exists() else []:
            if x.strip():
                r = json.loads(x)
                if r["arm"] in ("page118", "page59", "page88", "text"):
                    out.setdefault(r["arm"], []).append((d, {**r, "repeat": 0}))
    p = QA / "repeat" / "runs.jsonl"
    for x in p.read_text().splitlines() if p.exists() else []:
        if x.strip():
            r = json.loads(x)
            out.setdefault(r["arm"], []).append((r["doc"], r))
    return out


def sw_stats(entries) -> dict:
    per = [p for _, r in entries for p in r["per_question"]]
    sm = [p for p in per if p["kind"] == "small"]
    subs = sorted({p["sub"] for p in sm})
    return {"calls": len(entries), "input_tokens_mean": float(np.mean([r["input_tokens"] for _, r in entries])),
            "cost_mean": float(np.mean([r["cost_usd"] for _, r in entries])),
            "small": acc([p["correct"] for p in sm]), "original": acc([p["correct"] for p in per if p["kind"] != "small"]),
            "small_lure": acc([p.get("given") is not None and R.norm_num(p["given"]) == p.get("lure") for p in sm]),
            "small_by_sub": {s: acc([p["correct"] for p in sm if p["sub"] == s]) for s in subs},
            "small_per_call": [float(np.mean([p["correct"] for p in r["per_question"] if p["kind"] == "small"]))
                               for _, r in entries]}


def analyse(rows: list[dict], v2: dict, sw_base: dict | None = None) -> dict:
    """rows: this test's rows; v2: {doc: G.v2_rows(doc)}; sw_base: baseline_sw() (for the small-word arm)."""
    main = [r for r in rows if r["arm"] in ARMS]
    good = [r for r in main if valid(r)]
    res = {"invalid": [(r["doc"], r["arm"], r["repeat"]) for r in rows if not valid(r)], "arms": {}, "tests": {},
           "calls": [{"doc": r["doc"], "arm": r["arm"], "repeat": r["repeat"], "acc_all": r["acc_all"],
                      "em_salient": r["em_salient"], "em_filler": r["em_filler"],
                      "input_tokens": r["input_tokens"], "image_tokens_formula": r["image_tokens_formula"],
                      "target_tokens": r["target_tokens"], "output_tokens": r["output_tokens"],
                      "cost_usd": r["cost_usd"], "latency_s": r["latency_s"], "valid": valid(r)} for r in rows]}
    pool: dict = {}
    for d in sorted(v2):
        for k in ("text", 117.6, 88.2, 58.8):
            if k in v2[d]:
                pool.setdefault("text" if k == "text" else f"uniform {k}", []).append((d, v2[d][k]))
    for r in good:
        pool.setdefault(r["arm"], []).append((r["doc"], r))
    for name, entries in pool.items():
        res["arms"][name] = arm_stats(entries)
    for arm in ARMS:
        if arm in res["arms"]:
            res["arms"][arm]["wrong"] = wrong_answers([r for r in good if r["arm"] == arm])
    for arm in ARMS:
        for ref in ("uniform 58.8", "uniform 88.2", "uniform 117.6", "text"):
            k = "text" if ref == "text" else float(ref.split()[1])
            refs = {d: v2[d][k] for d in v2 if k in v2[d]}
            for lab, reps in (("first_call", (0,)), ("all_calls", tuple(REPEATS))):
                pr = paired([r for r in good if r["arm"] == arm], refs, reps)
                if pr:
                    res["tests"][f"{arm} vs {ref} ({lab})"] = test(pr)
    pr = []
    for d in sorted(v2):
        a = {r["repeat"]: r for r in good if r["arm"] == "salient_large" and r["doc"] == d}
        b = {r["repeat"]: r for r in good if r["arm"] == "salient_small" and r["doc"] == d}
        if 0 in a and 0 in b:
            pr += paired([a[0]], {d: b[0]}, (0,))
    if pr:
        res["tests"]["salient_large vs salient_small (first_call)"] = test(pr)
    ar = res["arms"]
    pred = {}
    if all(a in ar for a in ARMS) and all(f"uniform {b}" in ar for b in (58.8, 88.2)):
        sl, ss = ar["salient_large"], ar["salient_small"]
        d1 = sl["all"]["acc"] - ar["uniform 58.8"]["all"]["acc"]
        pred["M1"] = {"pass": d1 >= 0.03, "salient_large": sl["all"]["acc"], "uniform58.8": ar["uniform 58.8"]["all"]["acc"],
                      "diff": d1}
        d2 = sl["complex"]["acc"] - ar["uniform 88.2"]["complex"]["acc"]
        pred["M2"] = {"pass": sl["lookup"]["acc"] >= 0.99 and d2 <= -0.03, "lookups": sl["lookup"]["acc"],
                      "multi_step": sl["complex"]["acc"], "uniform88.2_multi_step": ar["uniform 88.2"]["complex"]["acc"],
                      "multi_step_diff": d2}
        d3 = sl["all"]["acc"] - ss["all"]["acc"]
        pred["M3"] = {"pass": d3 >= 0.05, "salient_large": sl["all"]["acc"], "salient_small": ss["all"]["acc"], "diff": d3}
    sw = [r for r in rows if r["arm"] == SW_ARM and valid(r)]
    if sw and sw_base:
        x = {"salient_large": sw_stats([(r["doc"], r) for r in sw])}
        for k in ("page118", "page88", "page59", "text"):
            if k in sw_base:
                x[k] = sw_stats(sw_base[k])
        res["small_words"] = x
        first = {k: {d: r for d, r in sw_base.get(k, []) if r.get("source") != "repeat" and r["repeat"] == 0}
                 for k in ("page118", "page59", "text")}
        for k, refs in first.items():
            pr = paired(sw, refs, (0,), ids=lambda p: p["kind"] == "small")
            if pr:
                res["tests"][f"{SW_ARM} vs Test A {k} (small-word questions)"] = test(pr)
        if "page118" in x:
            pred["M4"] = {"pass": x["salient_large"]["small"]["acc"] <= x["page118"]["small"]["acc"],
                          "salient_large": x["salient_large"]["small"]["acc"], "uniform117.6": x["page118"]["small"]["acc"],
                          "uniform117.6_calls": x["page118"]["calls"]}
        res["small_words_wrong"] = wrong_answers([{**r, "per_question": [p for p in r["per_question"]
                                                                         if p["kind"] == "small"]} for r in sw])
    res["predictions"] = pred
    return res


def summary_md(res: dict) -> str:
    ar = res["arms"]
    order = [a for a in ("text", "uniform 117.6", "uniform 88.2", "uniform 58.8", *ARMS) if a in ar]
    md = ["# Content-chosen mixed resolution: summary (generated by `claude_qa_salient.py analyse`)", "",
          "Pooled over every valid call of each arm (uniform arms and text: the v2 rows, one call per document).", "",
          "| arm | calls | input tokens / call | cost / call | all | lookups | needle | number | multi-step |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in order:
        x = ar[a]
        md.append(f"| {a} | {x['calls']} | {x['input_tokens_mean']:,.0f} | ${x['cost_mean']:.2f} | {fmt(x['all'])} | "
                  f"{fmt(x['lookup'])} | {fmt(x['needle'])} | {fmt(x['number'])} | {fmt(x['complex'])} |")
    md += ["", "Multi-step questions by type:", "", "| arm | " + " | ".join(SUBS) + " |", "|---|" + "---|" * len(SUBS)]
    for a in order:
        md.append(f"| {a} | " + " | ".join(fmt(ar[a]["complex_by_sub"][s]) for s in SUBS) + " |")
    md += ["", "Wrong answers by type, and whether the numbers the answer is read from were drawn large (all digits "
           "salient). 'anchor' = mean share of the facts' characters that are salient:", "",
           "| arm | type | wrong | number large | number small | no number (counts) | anchor salient |",
           "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for t, w in sorted(ar.get(a, {}).get("wrong", {}).items()):
            md.append(f"| {a} | {t} | {w['wrong']} | {w['number_salient']} | {w['number_not_salient']} | "
                      f"{w['no_number']} | {100 * w['anchor_cov_mean']:.0f}% |")
    if "small_words" in res:
        x = res["small_words"]
        subs = sorted(x["salient_large"]["small_by_sub"])
        md += ["", "Small-word questions (Test A's document; 40 per document; baselines: Test A and the repeat calls):", "",
               "| arm | calls | input tokens / call | small-word | answers = lure | original 120 | " + " | ".join(subs) + " |",
               "|---|---|---|---|---|---|" + "---|" * len(subs)]
        for k, v in x.items():
            md.append(f"| {k} | {v['calls']} | {v['input_tokens_mean']:,.0f} | {fmt(v['small'])} | {fmt(v['small_lure'])} | "
                      f"{fmt(v['original'])} | " + " | ".join(fmt(v['small_by_sub'].get(s, {'n': 0})) if s in v['small_by_sub']
                                                         else "-" for s in subs) + " |")
        ww = "; ".join(f"{t}: {w['wrong']} (number large {w['number_salient']}, small {w['number_not_salient']})"
                       for t, w in sorted(res["small_words_wrong"].items()))
        md += ["", "Small-word wrong answers: " + (ww or "none")]
    md += ["", "Paired tests (arm - reference, points; Newcombe 95% interval; exact McNemar on discordant pairs):", "",
           "| comparison | pairs | arm - ref | 95% interval | ref-only right | arm-only right | p |", "|---|---|---|---|---|---|---|"]
    for k, t in res["tests"].items():
        md.append(f"| {k} | {t['n']} | {100 * t['arm_minus_ref']:+.1f} | {100 * t['lo']:+.1f} to {100 * t['hi']:+.1f} | "
                  f"{t['ref_only']} | {t['arm_only']} | {t['p']:.3g} |")
    md += ["", "Calls:", "", "| doc | arm | repeat | em salient / filler | all | input tokens | image-token formula | "
           "uniform 58.8 formula | output | cost | s | valid |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["calls"]:
        md.append(f"| d{c['doc']} | {c['arm']} | {c['repeat']} | {c['em_salient']:.2f} / {c['em_filler']:.2f} | "
                  f"{100 * c['acc_all']:.1f}% | {c['input_tokens']:,} | {c['image_tokens_formula']:,} | "
                  f"{c['target_tokens']:,} | {c['output_tokens']:,} | ${c['cost_usd']:.2f} | {c['latency_s']:.0f} | "
                  f"{c['valid']} |")
    md += ["", f"Invalid calls: {res['invalid'] or 'none'}", "", "## Pre-registered predictions", ""]
    p = res["predictions"]
    if "M1" in p:
        md.append(f"* **M1 {'PASS' if p['M1']['pass'] else 'FAIL'}** (salient_large >= uniform 58.8 + 3 points overall): "
                  f"{100 * p['M1']['salient_large']:.1f} vs {100 * p['M1']['uniform58.8']:.1f} ({100 * p['M1']['diff']:+.1f})")
        md.append(f"* **M2 {'PASS' if p['M2']['pass'] else 'FAIL'}** (lookups >= 99% and multi-step <= uniform 88.2 - 3): "
                  f"lookups {100 * p['M2']['lookups']:.1f}; multi-step {100 * p['M2']['multi_step']:.1f} vs "
                  f"{100 * p['M2']['uniform88.2_multi_step']:.1f} ({100 * p['M2']['multi_step_diff']:+.1f})")
        md.append(f"* **M3 {'PASS' if p['M3']['pass'] else 'FAIL'}** (salient_small <= salient_large - 5 points): "
                  f"{100 * p['M3']['salient_small']:.1f} vs {100 * p['M3']['salient_large']:.1f} ({100 * p['M3']['diff']:+.1f})")
    else:
        md.append("M1-M3 not judged yet: rows for both arms are needed.")
    if "M4" in p:
        md.append(f"* **M4 {'PASS' if p['M4']['pass'] else 'FAIL'}** (small-word: salient_large no better than uniform "
                  f"117.6): {100 * p['M4']['salient_large']:.1f} vs {100 * p['M4']['uniform117.6']:.1f} "
                  f"({p['M4']['uniform117.6_calls']} calls)")
    else:
        md.append("* M4: not run (or no rows yet).")
    return "\n".join(md) + "\n"


def cmd_analyse(a) -> None:
    rows = load_rows()
    v2 = {d: G.v2_rows(d) for d in a.docs}
    res = analyse(rows, v2, baseline_sw())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    md = summary_md(res)
    (OUT / "summary.md").write_text(md)
    print(md)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--docs", type=int, nargs="+", default=[0, 1, 2])
        if name in ("check", "run"):
            p.add_argument("--small-words", action="store_true", help="the small-word arm (Test A's document)")
        if name == "run":
            p.add_argument("--yes", action="store_true")
            p.add_argument("--workers", type=int, default=6)
            p.add_argument("--limit", type=int, default=0, help="make at most this many calls")
    a = ap.parse_args(argv)
    {"check": cmd_check, "run": cmd_run, "analyse": cmd_analyse}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
