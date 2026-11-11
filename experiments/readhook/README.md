# Read-as-image hook (prototype, 2026-11-11)

A Claude Code PreToolUse hook on `Read`: renders the requested lines of a text file as one PNG page (monospace,
line numbers, header "lines a-b of N", long lines wrapped) and rewrites Read's `file_path` to the PNG, so the model
sees code as an image. `additionalContext` tells it which `offset` to read next.

Install for one project in `.claude/settings.local.json` and run headless with `--setting-sources project,local`:

```json
{"hooks": {"PreToolUse": [{"matcher": "Read", "hooks": [{"type": "command",
  "command": "READHOOK_LOG=/path/hook.log READHOOK_CACHE=/path/cache python3 /path/experiments/readhook/hook.py"}]}]}}
```

Checked with Haiku 5.5 (Claude Code 2.1.322): the hook fires in `claude -p`; Read returns the image; the model read a
constant's value correctly from it and its Edit of the real file succeeded (read-before-edit is not broken); a full
1932x1876 page costs 4,505 more input tokens than a one-line image (4,485 expected), so it is not downscaled.

Token economics (estimate over this repo's 112 Python files; text = 2.26 chars/token measured + Read's line numbers):
13.8 px font (prose-safe) saves at most 1.2x (3 columns of 60); 11.9 px saves 1.6x (3 x 80); 9.8 px saves 2.2x
(4 x 60). Code's short lines leave most of a page blank. Not covered: Grep and Bash (`cat`, `sed -n`) still return
text.

## Minified Java (2026-11-11)

`javamin.py` reflows Java for rendering: indentation and line breaks dropped, lines joined and tagged `§<line>`,
literals left intact; optionally comments stripped. Measured on 4 real files (13,923 lines: commons-lang
StringUtils and DateUtils, Guava ImmutableList, Spring StringUtils); text tokens counted with Haiku 5.5, image tokens
from rendering with the prose renderer on 1932-px pages:

| form | text tokens | pages at 117.6 px²/char (prose 2.2x) | pages at 88.2 |
|---|---|---|---|
| Read text (original, with line numbers) | 275,717 (1.00x) | - | - |
| minified, comments kept | 247,529 (1.11x) | 85,065 tokens (3.24x) | 63,867 (4.32x) |
| minified, comments stripped | 70,103 (3.93x) | 23,478 (11.74x) | 17,607 (15.66x) |

Minifying the text alone saves little (whitespace is cheap for the tokenizer); reflowed pages fill like prose, and
Java tokenizes worse than prose (2.4 vs 2.9 chars/token), so pages save more than they do on prose. Pages vs the same
minified text: 2.9-3.0x either way. Stripping comments is a separate, lossy text-level saving (these files are
mostly Javadoc). Not tested: accuracy of reading and editing from minified pages; Edit needs a whitespace-insensitive
match to the real file.

## Minified Python (2026-11-11)

`pymin.py`: each logical line becomes `§<line> <depth><code>`, the block depth from Python's tokenizer shown as one
`»` per level; continuation lines (inside brackets, after a backslash, inside multi-line strings) are joined with a
space; blank lines dropped; comments and docstrings optionally stripped. `reconstruct()` rebuilds code with 4 spaces
per level, and `same_ast()` checks it: all 119 files tried (this repo's 115 + requests `sessions.py`, Django
`query.py`, Flask `app.py`, pandas `groupby.py`) rebuild to the same AST from the comment-stripped form (whitespace
inside strings collapsed on both sides). With comments kept the form is for reading only: an end-of-line comment
inside brackets swallows the joined continuation if rebuilt.

Measured on the 4 library files (11,725 lines), as for Java:

| form | text tokens | pages at 117.6 px²/char | pages at 88.2 |
|---|---|---|---|
| Read text (original, with line numbers) | 187,867 (1.00x) | - | - |
| minified, comments kept | 147,188 (1.28x) | 51,717 (3.63x) | 38,736 (4.85x) |
| minified, comments stripped | 135,271 (1.39x) | 46,332 (4.05x) | 34,731 (5.41x) |
| minified, comments + docstrings stripped | 83,838 (2.24x) | 27,273 (6.89x) | 20,439 (9.19x) |

## Command output as images (2026-11-11)

`bash_output_hook.py` (PostToolUse on Bash) replaces long command output with a rendered image through
`updatedToolOutput` (`stdout` = `data:image/png;base64,...`, `isImage: true`). Checked with Haiku 5.5: the tool result
reached the model as an image and it read a printed value and a numbered line correctly. Rewriting the command in a
PreToolUse hook to a wrapper that prints the data URI works too. Output under 800 characters stays text (an image
costs at least ~140 tokens).

## vbhook: the full hook set (2026-11-11)

`vbhook/` supersedes the prototypes above. Install for a project with
`python3 experiments/readhook/vbhook/install.py [--budget 88.2] > .claude/settings.local.json` (hooks: Read and
Edit/Write PreToolUse; Bash PreToolUse (exit-code line) and PostToolUse (output to image); plus one allow rule for
that exact `echo` line). Contract and rules for minifiers: `vbhook/SPEC.md`. Families, each checked on real files:

| family | kinds | savings as pages at 88.2 vs text |
|---|---|---|
| `python_family.py` | Python, shell, Makefile, Dockerfile | 3.5-4.8x (Python ~3.25x in the default edit-safe mode) |
| `brace.py` | JS, TS/TSX, Java, Kotlin, Scala, C, C++, C#, Go, Rust, Swift, PHP, Dart | 3.6-4.4x dense, ~3.6x overall edit-safe |
| `config_docs.py` | YAML, TOML, INI, JSON(L), XML/SVG/HTML, Markdown/RST/text, CSV/TSV, SQL; lock files and digit/hash-heavy files stay text | 2.7-5.5x |
| `outputs.py` | pytest, tracebacks, compiler/linter diagnostics, logs, git diff/log/status, grep, find, listings | 3.8-8x on large outputs; text below ~300 chars |

Policy (`policy.py`): an image only when it costs at most 0.8x the text tokens of the same lines or output. Tests:
`cd experiments/readhook && python3 -m pytest -q vbhook/tests` (333 tests). Results with Opus:
`results/readhook_smoke/README.md`.

## Results with Opus 5.5 (2026-11-11)

Smoke tests (6 refactors x 3 runs per arm): [results/readhook_smoke](../../results/readhook_smoke/README.md).
SWE-bench Verified, Django: [results/readhook_swe](../../results/readhook_swe/README.md); SWE-bench Multilingual,
Rust and Java: [results/readhook_swe_ml](../../results/readhook_swe_ml/README.md); harnesses in `swe/`. Summary,
cost anatomy and remaining levers: [docs/SWE_IMAGES.md](../../docs/SWE_IMAGES.md).
