"""Shared pieces for the EnterpriseRAG-Bench runs (docs/claude/ERB_PLAN.md).

Data: the benchmark's GitHub repository cloned to data/erb/repo (git-ignored, ~3.8 GB without .git; commit
d36685e2, 2026-05-07) and the leaderboard's published BM25 baseline files in data/erb/leaderboard/. Setup:

    git clone --depth 1 https://github.com/onyx-dot-app/EnterpriseRAG-Bench data/erb/repo && rm -rf data/erb/repo/.git
    python3 experiments/erb/fetch_leaderboard.py          # official BM25 top-10 per question + its judged scores

Document text, the context format and the answer prompt are copied from the benchmark's own code (MIT licence):
src/utils/document_content.py, src/utils/retrieval.py and src/prompts/vector_search_answer_gen.py, so our
text arm sees exactly what the official "BM25 + GPT-5.4" baseline's model saw.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ERB = Path(os.environ.get("ERB_REPO", ROOT / "data" / "erb" / "repo"))
LB = ROOT / "data" / "erb" / "leaderboard"
SOURCES = ERB / "generated_data" / "sources"
UUID_INDEX = ERB / "generated_data" / "uuid_index.json"
QUESTIONS = ERB / "questions.jsonl"
OUT = ROOT / "results" / "erb"

# src/prompts/vector_search_answer_gen.py, verbatim.
ANSWER_GEN_PROMPT = """
You are a helpful and precise assistant that generates answers based on the provided documents. The documents came from a retrieval system which is imperfect. \
Base your answer purely on the documents and do not make up any information. Many of the documents provided are likely to be irrelevant. \
Be concise and only provide information directly relevant to the query.

## Context Documents
{context_documents}

## Question
{question}

## Answer
Output your answer below, do not include any additional text or formatting:
"""


@lru_cache(maxsize=1)
def uuid_index() -> dict[str, str]:
    return json.loads(UUID_INDEX.read_text())


def questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]


def extract_document_content(d: dict) -> tuple[str, str]:
    """src/utils/document_content.py::extract_document_content (validation errors raise KeyError/ValueError)."""
    title = str(d[d["title_field_name"]])
    names = d["content_field_names"]
    if not isinstance(names, list) or not names:
        raise ValueError("bad content_field_names")
    if len(names) == 1:
        content = str(d[names[0]])
    else:
        parts = []
        for n in names:
            v = d[n]
            if isinstance(v, list):
                v = "\n".join(str(x) for x in v)
            parts.append(f"{n}:\n{v}")
        content = "\n\n".join(parts)
    return title, content


def load_doc(uuid: str) -> tuple[str, str]:
    return extract_document_content(json.loads((SOURCES / uuid_index()[uuid]).read_text()))


def format_doc(i: int, uuid: str) -> str:
    """One entry of src/utils/retrieval.py::format_context_documents."""
    try:
        title, content = load_doc(uuid)
        return f"--- Document {i} (ID: {uuid}) ---\nTitle: {title}\n\n{content}"
    except Exception as e:                                         # same fallback text as the original
        return f"--- Document {i} (ID: {uuid}) ---\n[Error loading document: {e}]"


def format_context_documents(uuids: list[str], start: int = 1) -> str:
    return "\n\n".join(format_doc(i, u) for i, u in enumerate(uuids, start))


def official_top10() -> dict[str, list[str]]:
    """The leaderboard's published BM25 baseline retrieval (answers_bm25.jsonl): 10 document ids per question."""
    return {r["question_id"]: r["document_ids"]
            for r in map(json.loads, (LB / "answers_bm25.jsonl").read_text().splitlines()) if r}
