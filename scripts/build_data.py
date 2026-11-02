"""Build all WP-A data: cleaned split texts, evaluation passages, Visual-BPE dictionaries.

    python3 scripts/build_data.py            # build whatever is missing
    python3 scripts/build_data.py --force    # rebuild everything

Outputs (data/, git-ignored): train.txt val.txt test.txt, passages_{natural,scrambled}.jsonl,
dict_3000.json (master) and its prefixes dict_{100,300,1000}.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vbpe import config, corpus, dictionary  # noqa: E402

SIZES = (100, 300, 1000, 3000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()
    config.DATA.mkdir(parents=True, exist_ok=True)

    for split in ("test", "val", "train"):
        p = corpus.split_path(split)
        if args.force or not p.exists():
            print(f"building {p} ...", flush=True)
            corpus.build_split(split)
        print(f"{p}: {p.stat().st_size:,} bytes")

    if args.force or not all((config.DATA / f"passages_{v}.jsonl").exists()
                             for v in ("natural", "scrambled")):
        corpus.build_passages()
    for v in ("natural", "scrambled"):
        ps = corpus.load_passages(v)
        print(f"passages_{v}: {len(ps)} passages, lengths {[len(p['text']) for p in ps]}")

    master_p = dictionary.dict_path(dictionary.MASTER_N)
    if args.force or not master_p.exists():
        master = dictionary.build(dictionary.MASTER_N)
    else:
        master = dictionary.Dictionary.from_file(master_p)
    for n in SIZES:
        if n != dictionary.MASTER_N:
            master.prefix(n).save(dictionary.dict_path(n))
        print(f"dict_{n}.json: first tokens {master.tokens[:min(n, 8)]}")


if __name__ == "__main__":
    main()
