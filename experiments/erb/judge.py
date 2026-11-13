"""Provisional judge: the benchmark's own evaluator (metrics_based_eval.py), with its LLM calls sent to the Claude Code
CLI instead of an API key (docs/claude/ERB_PLAN.md).

Everything but the transport is theirs: prompts, citation stripping, correctness, per-fact completeness, document
recall and the three-judge gold-document correction. Their Anthropic class is replaced by `CLILLM` before the
evaluator imports it; each call is a single-shot CLI call (no tools), cached on disk by vtc.claude_eval (re-runs and
--resume are free) and logged with its usage to results/erb/judge/usage.jsonl.

Judge model: $ERB_JUDGE_MODEL (default claude-sonnet-4-6, the evaluator's own Anthropic default) for every call the
evaluator makes with its main model; $ERB_JUDGE_CHEAP_MODEL (default claude-haiku-4-5) where it asks for its cheap one.

    python3 experiments/erb/judge.py --answers results/erb/answers_T10.jsonl --name T10 [--parallelism 8] [--resume]
    python3 experiments/erb/judge.py --answers data/erb/leaderboard/answers_bm25.jsonl --name official_bm25 \\
        --only-pilot          # calibration: our judge on GPT-5.4's published answers vs Onyx's scores
    python3 experiments/erb/judge.py --fake ...   # no Claude calls: canned replies, to test the plumbing
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import erb_common as E  # noqa: E402

JUDGE = E.OUT / "judge"
_lock = threading.Lock()


class DetRandom:
    """Stands in for the `random` module in the evaluator: its 20-candidate sample and the per-run candidate shuffles
    of the three-judge consensus become functions of the document set (and of how many times that set was drawn),
    so a re-run, or another arm with the same documents (T10, P10 and the official baseline share their top-10),
    builds byte-identical judge prompts and reuses the cached judgments. Each consensus run still sees its own
    order."""

    def __init__(self):
        self.n = {}

    def _rng(self, op: str, pop) -> random.Random:
        key = zlib.crc32("\n".join(map(str, pop)).encode())
        with _lock:
            k = self.n.get((op, key), 0)
            self.n[(op, key)] = k + 1
        return random.Random(f"{op}-{key}-{k}")

    def sample(self, pop, k):
        return self._rng("sample", sorted(pop)).sample(list(pop), k)

    def shuffle(self, x):
        self._rng("shuffle", sorted(x)).shuffle(x)

    def __getattr__(self, name):                                      # anything else: the real module
        return getattr(random, name)


_seen: dict[str, int] = {}


def occurrence(key: str) -> int:
    """How many times this exact judge call was already made in this process: repeated identical prompts (e.g. a
    consensus run over a single candidate) get separate cache slots, so they stay independent judgments."""
    with _lock:
        k = _seen.get(key, 0)
        _seen[key] = k + 1
    return k


def make_cli_llm(fake: bool):
    sys.path[:0] = [str(E.ERB), str(E.ROOT)]
    from src.llm.interface import LLMInterface                      # the benchmark's interface
    from vtc import claude_eval as CE

    main_model = os.environ.get("ERB_JUDGE_MODEL", "claude-sonnet-4-6")
    cheap_model = os.environ.get("ERB_JUDGE_CHEAP_MODEL", "claude-haiku-4-5")
    cheap_names = {os.environ.get("CHEAP_LLM_MODEL_NAME", "claude-haiku-4-5")}
    effort = {"low": "low", "medium": "medium", "high": "high"}

    class CLILLM(LLMInterface):
        def __init__(self, model=None, tools=None, quiet=False, reasoning_level="medium", **_):
            if tools:
                raise ValueError("the judge adapter supports tool-free calls only")
            self.model = cheap_model if (model and (model in cheap_names or "haiku" in model)) else main_model
            self.effort = effort.get(reasoning_level or "", None)

        def generate(self, messages):
            system = "\n\n".join(m.content for m in messages if m.role == "system") or "You are a helpful assistant."
            convo = [m for m in messages if m.role != "system"]
            prompt = convo[0].content if len(convo) == 1 else "\n\n".join(
                f"[{m.role}]\n{m.content}" for m in convo)
            if fake:                                                  # sizes only, for the cost estimate
                with _lock, open(JUDGE / "fake_prompts.jsonl", "a") as f:
                    f.write(json.dumps({"model": self.model, "chars": len(system) + len(prompt),
                                        "head": prompt[:60]}) + "\n")
                yield _fake_reply(prompt)
                return
            k = occurrence(CE.cache_key(self.model, system, prompt, [], self.effort))
            r = CE.transcribe([], system, prompt, model=self.model, effort=self.effort, max_retries=3,
                              cache_dir=CE.CACHE_DIR / "erb_judge" / f"occ{k}")
            if r.get("error"):
                raise RuntimeError(f"judge call failed: {str(r['error'])[:300]}")
            with _lock, open(JUDGE / "usage.jsonl", "a") as f:
                f.write(json.dumps({"model": self.model, "cached": r.get("cached"), "usage": r.get("usage"),
                                    "cost_usd": r.get("cost_usd")}) + "\n")
            yield r.get("text") or ""

    return CLILLM


def _fake_reply(prompt: str) -> str:
    """Plausible JSON for every evaluator prompt shape, so the plumbing runs end to end without calls."""
    return json.dumps({"answer_correct": True, "correct": True, "is_correct": True, "reasoning": "fake",
                       "fact_supported": True, "supported": True, "valid": True, "classification": "required",
                       "documents": [], "answer": "fake", "stripped_answer": "fake", "result": "required"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--parallelism", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only-pilot", action="store_true", help="keep only the pilot questions' answers")
    ap.add_argument("--no-correction", action="store_true")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    JUDGE.mkdir(parents=True, exist_ok=True)
    ans = Path(a.answers).resolve()
    if a.only_pilot:
        keep = set(json.loads((E.OUT / "pilot_qids.json").read_text()))
        rows = [l for l in ans.read_text().splitlines() if l.strip() and json.loads(l)["question_id"] in keep]
        ans = JUDGE / f"answers_{a.name}_pilot.jsonl"
        ans.write_text("\n".join(rows) + "\n")
    os.environ.setdefault("LLM_PROVIDER", "anthropic")
    os.environ.setdefault("LLM_API_KEY", "unused-by-the-cli-adapter")
    cli = make_cli_llm(a.fake)
    import types                                                      # factory imports AnthropicLLM from here:
    stub = types.ModuleType("src.llm.anthropic_llm")                  # a stub module, so no SDK is needed
    stub.AnthropicLLM = cli
    sys.modules["src.llm.anthropic_llm"] = stub
    os.chdir(E.ERB)                                                   # the evaluator uses repo-relative paths
    from src.scripts.answer_evaluation import metrics_based_eval as M
    from src.utils import eval_utils as U
    M.random = U.random = DetRandom()
    tag = a.name + ("_fake" if a.fake else "")
    sys.argv = ["metrics_based_eval", "--answers-file", str(ans), "--questions-file", str(E.QUESTIONS),
                "--results-file", str(JUDGE / f"results_{tag}.json"),
                "--updated-questions-file", str(JUDGE / f"questions_updated_{tag}.jsonl"),
                "--parallelism", str(a.parallelism)]
    sys.argv += ["--resume"] * a.resume + ["--no-correction"] * a.no_correction
    M.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
