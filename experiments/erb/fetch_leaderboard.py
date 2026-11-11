"""Download the EnterpriseRAG-Bench leaderboard's published files for the official BM25 baseline (no login needed).

answers_bm25.jsonl: the baseline's answers and its 10 retrieved document ids per question (OpenSearch BM25, standard
analyzer, title + content in one field); results_bm25.json: Onyx's judged scores per question; leaderboard.csv and
systems.yaml: the board itself. Saved to data/erb/leaderboard/.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from erb_common import LB  # noqa: E402

BASE = "https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard/resolve/main/data"
FILES = ["raw_data/answers_bm25.jsonl", "raw_data/results_bm25.json", "raw_data/questions_updated_bm25.jsonl",
         "raw_data/questions.jsonl", "raw_data/systems.yaml", "final_display_data/leaderboard.csv"]


def main() -> int:
    LB.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = LB / Path(f).name
        with urllib.request.urlopen(f"{BASE}/{f}", timeout=120) as r:
            dst.write_bytes(r.read())
        print(f"{dst} {dst.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
