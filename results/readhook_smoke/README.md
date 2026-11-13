# Read-as-image smoke test (2026-11-11)

> **In brief (added in the clean-up, 2026-11-12).** A first check that Claude Code can refactor code it only sees
> as page images. On 6 small Python tasks, 3 runs each, every run in both arms produced code equivalent to the
> original (18 of 18 with text; 18 of 18 with images in each of the hook versions tried). The second half of this
> file is the engineering log of the full hook set (`experiments/readhook/vbhook/`): three versions, each fixing what
> the one before exposed. The larger test is [results/readhook_swe](../readhook_swe/README.md); the summary is
> [docs/claude/SWE_IMAGES.md](../../docs/claude/SWE_IMAGES.md).

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

## Full hook set at 88.2 px²/char (2.9x), 2026-11-11

`experiments/readhook/vbhook/` (minifiers for 40+ file and output kinds; Read, Edit/Write and Bash hooks; installer).
Same 6 tasks x 3 runs, Opus 5.5 high effort; Bash no longer restricted to pytest (file dumps and long output are
converted to images instead of blocked). Three iterations, each fixing what the previous run exposed:

| arm | equivalent | failed Edits | image views | image tokens vs the same content as text | cost |
|---|---|---|---|---|---|
| v1: first full set | 18/18 | 0 | 7 | 1,656 vs 3,800 (2.29x fewer) | $1.88 |
| v2: compound commands, output sniffing fixed | 18/18 | 2 | 34 | 10,695 vs 27,408 (2.56x fewer) | $2.02 |
| v3: edit-safe code flows | 18/18 | 1 | 32 | 10,350 vs 27,231 (2.63x fewer) | $2.09 |

* v1: models read code with compound commands (`ls && cat a.py && cat test*.py`) whose output had no minifier and stayed text, and Python output was taken for a Dockerfile; only 7 views were images.
* v2: every run saw its code as images (file contents inside compound output are minified by their own kind). 2 Edits failed because the dense flow drops spaces around operators and the model copied that spacing; Claude Code checks old_string before any hook runs, so the whitespace-insensitive matcher cannot help.
* v3: edit-safe flows (spaces kept inside lines, every line tagged; +6.8% page tokens). The one failed Edit was a real misread (`iv`/`ov` swapped in a symmetric condition), recovered after the model checked the lines; 7 Write calls were refused by Claude Code because the file had only been printed by a shell command (a note now tells the model to Read first).
* Failed calls: PostToolUse does not run for a failed Bash call, so a command that exits non-zero leaked its output as text in v1. `bash_pre_hook.py` appends `echo "[exit code $?]"` so the call succeeds (code 0 is stripped again); the installer allows only that exact line, so permission rules still apply to the command itself.
* Reads that stayed text (14 in v3) were small partial reads where text was cheaper (the policy works as intended).

Conclusion: at 2.9x density with the full hook set, 54/54 runs across the three iterations produced equivalent refactors; the content seen as images cost 2.6x fewer tokens than the same content as text (small files; larger files gain more). One content misread in 24 v3 Edits vs none in 24 at 2.2x (117.6).
