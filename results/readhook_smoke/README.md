# Read-as-image smoke test (2026-11-11)

6 refactor-for-performance tasks (`experiments/readhook/smoke/tasks/`), Opus 5.5 at high effort, Claude Code
headless with Read, Edit, Write and `python3 -m pytest` only. Arms: `text` (normal Read) and `image` (Read of a .py
file returns minified Python rendered at 117.6 px²/char; Edit matched ignoring whitespace and re-indented, both by
hooks in `experiments/readhook/smoke/`). Scored outside the agent: visible tests, every public function against the
original on random inputs (`checks.py`), speedup of the target function on a large input, tokens, cost.

Predictions, written before the run (a smoke test, not a powered comparison):
1. text arm: 6/6 equivalent and faster.
2. image arm: at least 5/6 equivalent; speedups comparable to the text arm.
3. image arm: at most 2 failed Edit calls in total.
4. file-read tokens: images below text on every task (small files: one partial page each).
