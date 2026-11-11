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
