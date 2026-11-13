# vtchook minifiers: spec for contributors

Goal: Claude reads code, config, docs and command output as **page images** (rendered at 88.2 px² per character on
1932-px pages, ~4,600 tokens per full page) instead of text, when that costs fewer tokens. A minifier removes the
whitespace that wastes page area (indentation, line breaks, blank lines, alignment) while keeping everything a reader
needs. See `minify/base.py` for the contract and invariant; `experiments/readhook/pymin.py` is the reference
(Python; 119/119 files rebuild to the same AST).

Two things were learnt after this spec was first written (2026-11-11) and are now the default:

* **Code minifiers are edit-safe by default.** Claude Code checks an Edit's `old_string` against the file before any
  hook runs, so the Edit hook's whitespace-insensitive match (rule 1) cannot rescue a copied string whose spacing
  differs. The Python and brace-language minifiers therefore keep the spaces inside a line and tag every line; rule
  5's tighter spacing applies only to the dense variant (`VTCHOOK_TIGHT=1`) and to kinds that are read, not edited
  (command output).
* **The Python minifier refuses a file that itself contains the marker characters `§` or `»`** (rule 6); the caller
  then falls back to a plainer form or to text.

Terms: a *flow* is the minified text as one long string, laid out by the renderer like prose; *cpt* is characters
per Claude text token.

Rules
1. Never change or drop a non-whitespace character (except a declared `normalise`, e.g. ANSI codes). Readers copy
   exact text from the image into Edit calls; the Edit hook matches ignoring whitespace only.
2. `§<n> ` before each line that starts a statement/record (original 1-based line numbers). Continuation lines may be
   joined untagged. Measure whether tags on every line are worth their cost; never drop them where line numbers are
   referenced (errors, tracebacks).
3. `»` per level only where indentation is semantic. Cosmetic indentation is dropped.
4. Whitespace *inside* tokens that matter (string literals, heredocs, code blocks, `<pre>`) is kept as is.
5. Spaces between tokens may be removed where no ambiguity is created for a human reader (`a + b` -> `a+b` is fine;
   `a - -b` -> `a--b` is not; identifiers/keywords/numbers keep one space between them).
6. Return the flow; if the input cannot be handled safely, raise or return "" (the caller falls back to text).
7. Register with the measured `cpt` (chars per Claude token of the ORIGINAL text, from `count_tokens()` below) so the
   hook can decide image vs text.
8. Tests in `vtchook/tests/test_<family>.py` (pytest, < 60 s, no network, no Claude calls): the invariant on real
   files, semantic round-trips where a parser exists (ast, json, yaml, tomllib, a tokenizer), edge cases.

Measuring tokens (costs < $0.01 per 100k chars; cap your spend at $0.50): use
`vtc.claude_eval.transcribe([], "Reply with OK.", "OK?\n<code>\n" + chunk + "\n</code>", model="claude-haiku-5-5",
effort="low")["input_tokens"]` minus the same call without the code, chunks <= 60k chars.
Image tokens: `vtchook.render.page_tokens(flow, budget=88.2)`.
