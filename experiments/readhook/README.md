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
