# Plan-quality test: an orchestrator that reads the atlas once (design, queued)

*2026-11-12. Approved by the user and queued behind the two running atlas tests (hard fresh tasks, C-to-Rust
migration). Not run. An Opus subagent turns this design into the pre-registration in
`results/plan_quality/README.md` before its first scored call. Spend cap: $45 at list price.*

## The use case

A backlog of tickets against one service. One orchestrator session loads the repo atlas
(`experiments/repo_atlas.py`), reads every ticket and writes a brief per ticket. Worker subagents get their brief,
not the atlas, and run in parallel; after a batch merges the atlas is rebuilt and the next phase is planned. Cost
reasoning and what earlier tests showed: [CLAUDE_QA.md](CLAUDE_QA.md), "Proposed, not tested".

**Question:** are briefs written from the atlas good enough to replace scouting, and do they size the work well
enough to pick the right worker (model and reasoning level) per ticket?

## The brief (fixed structure, asked for in the prompt)

```
TICKET: <id>
SUMMARY: <one sentence>
FILES: <path> : <what changes>            (at most 6, most important first)
FUNCTIONS: <names>
APPROACH: <at most 3 sentences>
RISKS: <at most 2>
COLLIDES_WITH: <ticket ids that touch the same files, or none>
ORDER: <tickets that should land first, or none>
SIZE: S | M | L
WORKER: <haiku | sonnet | opus> / <low | medium | high>
CONFIDENCE: low | medium | high
```

Sizes, defined in the prompt: **S** one file, a local change, no design choice; **M** two or three files, or one
decision that is not obvious; **L** four or more files, or a change across modules. Default worker per size:
S = Haiku 5.5 at low effort, M = Sonnet 5.5 at medium, L = Opus 5.5 at high; the planner may choose otherwise and
must then say why in RISKS.

## Tasks

Fresh tickets with a known fix, from the benchmark the fresh-task test uses (SWE-rebench; its selection, clones
and validation records are in `results/repo_atlas_swe/`), in repositories whose atlas fits and not located by a
bare file list. They are not newer than the model (created 2025-07 to 2026-05; nothing later exists in the
benchmark), so the bare-file-list screen is the guard against memorised tickets. Layers 1 and 2 need only the real fix, so they use
every such ticket (target 40 or more). Layer 3 needs tests that run here, so it uses the tickets that test
validated and compared (16; its environments were deleted and must be rebuilt from its selection records).

## Planners (Opus 5.5, high effort; the same instruction and brief format)

1. `tree`: the file list only;
2. `map`: the repo map (signatures per file);
3. `atlas`: the page images and their index, one call;
4. `scout`: Claude Code with read-only tools on the checkout (what a worker does today before it starts).

## Layer 1: the brief against the real fix (no execution)

* files: share of the fix's source files named; share of named files that are not in the fix;
* functions: share of the functions the fix changes that are named;
* **things that do not exist:** named paths and symbols absent from the repository (what a misread page would
  produce);
* approach: judged blind (planner names removed, order shuffled) against the real fix by a fixed rubric: right
  cause, right place, would lead to the fixed behaviour; one call per brief;
* **sizing:** predicted SIZE against the real fix's size (files, functions and lines changed), rank correlation
  and a confusion table;
* cost and seconds per brief.

## Layer 2: tickets that collide (what only an orchestrator can do)

Groups of three or more tickets in one repository, planned together on one commit. The planner fills
COLLIDES_WITH and ORDER. Scored against the overlap of the real fixes' files (pairs found, pairs wrongly flagged).
If the benchmark has fewer than three usable groups, this layer is reported as not evaluable.

## Layer 3: does the brief help the worker, and is the worker right-sized?

Workers are Claude Code with edit tools, graded by the ticket's own tests (the fresh-task runner).

* **Brief or scouting:** an Opus high-effort worker with the `atlas` brief, with the `tree` brief, and with no
  brief (rows from the fresh-task test): resolved, turns, cost, seconds.
* **Right-sizing:** with the `atlas` brief, each ticket is also run on Haiku at low effort and Sonnet at medium.
  The cheapest tier that resolves a ticket is its measured tier. Scores: the planner's WORKER against the measured
  tier (right, too small, too large); and three policies compared on tickets resolved and total cost: always Opus
  at high effort, always the cheapest tier, and the planner's choice.

## Predictions (before any row)

1. The `atlas` brief names more of the fix's files than the `tree` brief (paired, at least 5 points).
2. Things that do not exist: at most 2% of what the `atlas` brief names, and not more than 2 points above `scout`.
3. `scout` names at least as many of the fix's files as `atlas` (within 5 points either way).
4. Collisions: the `atlas` planner finds more true colliding pairs than `tree` at no more false ones.
5. Sizing: the `atlas` planner's SIZE rises with the real fix's size (rank correlation at least 0.3).
6. Right-sizing: following the planner's WORKER choice resolves within 1 ticket of always-Opus-high at no more
   than 70% of its cost.
7. A worker with the `atlas` brief uses fewer turns than one with no brief (paired median) and resolves within 2
   tickets of it.

Predictions 1, 4 and 6 are the uncertain ones: on every earlier test the atlas alone did not beat a file list.

## Budget and order (estimates)

Layers 1-2: ~$20 (atlas briefs are the main cost). Layer 3: ~$14 for the two Opus brief arms and ~$4 for the
Haiku and Sonnet runs. Cap $45. Order: layer 1 pilot on 6 tickets, then layers 1-2 in full, then layer 3 while
whole tickets fit under the cap. With 17 tickets layer 3 can only show large differences; say so in the write-up.
