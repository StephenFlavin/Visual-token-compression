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

## Results (2026-11-11): 36 runs (6 tasks x 2 arms x 3), $2.38 at list price

| task | text: equivalent | image: equivalent | file-read tokens per run, text (est.) | image (mean) | per read, text / image | speedup, text / image (median) |
|---|---|---|---|---|---|---|
| orders | 3/3 | 3/3 | 1325 | 740 | 1325 / 555 | 55x / 65x |
| text_stats | 3/3 | 3/3 | 830 | 487 | 830 / 365 | 265x / 293x |
| graph_hops | 3/3 | 3/3 | 694 | 580 | 694 / 348 | 73x / 83x |
| intervals | 3/3 | 3/3 | 875 | 348 | 525 / 348 | 181x / 287x |
| moving | 3/3 | 3/3 | 615 | 279 | 615 / 279 | 35x / 44x |
| scheduler | 3/3 | 3/3 | 881 | 695 | 660 / 417 | 158x / 159x |

* Both arms: 18/18 runs equivalent to the original on every public function (random inputs) and all visible tests pass.
* Image arm: every Edit (24) matched the real file exactly; the model restored the real indentation from the `»` markers itself, so the whitespace-insensitive matcher was never needed. No failed Edit or Write in either arm.
* File-read tokens: 9,387 for the image arm against ~15,660 for the text arm (1.67x fewer). Text-arm figures are estimates (Python in Read format: 2.65 chars/token, measured); image figures are exact (pages are not downscaled). The image arm re-read the file after editing in 6 of 18 runs, so on graph_hops two runs used 696 tokens against 694.
* Every run in both arms tried one non-pytest shell command first and was denied by the guard; pytest output stayed short (all runs passed; at most 439 characters), so no code reached the image arm as text. Text-arm Read errors were attempts to read the directory.
* Speedups differ by run and algorithm (single timing); they are not a measure of reading.

Predictions: 1 (text 6/6) PASS; 2 (image >= 5/6, comparable speedups) PASS; 3 (<= 2 failed Edits) PASS (0); 4 (image read tokens below text on every task) PASS per read, but not per run on graph_hops (re-reads).

Limits: small single-page files (~2-4k chars), so this tests reading and editing from pages, not long-context savings; 3 runs per arm; Opus 5.5 only.
