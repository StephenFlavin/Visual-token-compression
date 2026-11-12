# Repo atlas: does it help find a security flaw?

*Pre-registered 2026-11-12, before any scored call. Code: `experiments/repo_atlas_sec.py` (tests
`tests/test_repo_atlas_sec.py`). Plan of the atlas: [docs/claude/REPO_ATLAS_PLAN.md](../../docs/claude/REPO_ATLAS_PLAN.md).
Results are added below this pre-registration once the calls are in; nothing above the "Results" heading changes
after the first call, except dated amendments.*

## Question

Does giving Claude a whole repository as page images (the "atlas") help it find where a security flaw is? Against
two baselines: the file list alone (what names and memory give) and Claude Code searching the code itself.

This is defensive detection on published, already fixed flaws in open-source code. Every arm reports locations and
weakness classes only. No arm is asked for exploit code, payloads or proofs of concept.

## Why this design

The two earlier atlas tests failed for one reason: on well-known code the model answers from memory (a bare file
list located SWE-bench fixes; with no code at all it scored 75.6 of 100 on SWE-QA). So here the flaws are newer
than the model's training data (knowledge cutoff June 2026): advisories published 2026-11-09 to 2026-11-11, fix
commits from 2026-07-01 on. The `closed` arm measures what memory and file names still give.

VulnGym could not be used: none of its measured repositories fits an atlas (smallest 246k image tokens, limit ~150k).
So the tasks come straight from the GitHub Advisory Database, in small libraries.

## Selection rule (`select`, fixed before any call)

Source: `github.com/github/advisory-database`, `advisories/github-reviewed/2026/` (blobless sparse clone,
2026-11-12). An advisory is an OSV JSON; its fix commit is a GitHub commit URL among its references.

1. Reviewed advisories published on or after 2026-07-01, newest first (ties: id); not withdrawn.
2. Exactly one distinct fix commit on GitHub among the references (`/commit/<sha>`, `/commits/<sha>`,
   `/pull/<n>/commits/<sha>`; a short sha that prefixes a long one is the same commit). The vulnerable version is its
   first parent. **Added to the brief:** the fix commit itself must be committed on or after 2026-07-01, so the fix
   was not public before the model's training cutoff.
3. The fix changes 1 to 5 counted source files: not tests, docs, changelogs, lock or vendored files
   (`repo_atlas_eval.is_gold`) and in a programming language. Config and data files (json, yaml, toml, ...) are not
   counted. Every counted file must have a `repo_atlas` minifier (a fix touching Ruby, Lua, Vue, ... fails).
4. At most 2 advisories per repository.
5. At least one gold file exists at the parent, and **every** gold file that does is in the atlas's file set at the
   parent (`repo_atlas.collect`). **Added to the brief** after a free check: `repo_atlas` leaves out files that
   contain its marker characters (`§`, `▶`, ...), and one compose-lint advisory had 3 of its 4 gold files left out
   that way. Without this the atlas arms could not see the flaw.
6. The atlas at the parent is 15k to 150k image tokens (a path-count prefilter rejects parents with more than 2,500
   atlas-candidate files without downloading them).
7. Stop at 14 tasks or after 300 advisories examined.

**Gold** = the counted source files of the fix that exist at the parent, and the functions the fix changes in them
(`gold_functions`: for each changed line of the parent's file, its enclosing function, method or class declaration;
plus names declared on added lines; approximate by design).

### Funnel

142 advisories examined (published 2026-11-09 18:47 to 2026-11-11 00:47 UTC); 14 selected.

| Reason (in the order applied) | Advisories |
|---|---|
| withdrawn | 13 |
| no fix commit on GitHub | 15 |
| more than one fix commit on GitHub | 51 |
| fix commit before 2026-07-01 | 11 |
| no source file changed | 0 |
| more than 5 source files changed | 9 |
| a changed source file has no minifier | 4 |
| repository already has 2 tasks | 7 |
| a gold file is outside the atlas's file set | 3 |
| atlas too large (path-count prefilter) | 8 |
| atlas too large (measured) | 7 |
| atlas too small | 0 |
| fix commit not fetchable | 0 |
| **selected** | **14** |

Per advisory: `selection.json` (`funnel`), with repository, fix and parent commits, CWE ids and changed paths.

## Tasks (in run order)

Run order (fixed now, `run_order`): round-robin over repositories so a capped budget covers as many repositories as
it can; within a round, ascending atlas size. The advisory text, id, title and fixed version are never shown to any
arm; the table gives no advisory text either.

| # | Repository | Vulnerable commit (parent of fix) | CWE (family) | Published | Fix committed | Gold files | Gold functions | Atlas pages / image tokens |
|---|---|---|---|---|---|---|---|---|
| 1 | filegator/filegator | `c6715ff7` | CWE-330, CWE-862 (access, crypto) | 2026-11-09 | 2026-08-12 | `backend/Controllers/DownloadController.php`, `backend/Services/Archiver/Adapters/ZipArchiver.php` | batchDownloadCreate, batchDownloadStart, rememberBatchArchive, ownsBatchArchive, createArchive | 6 / 27,255 |
| 2 | zeroturnaround/zt-zip | `0a9d0e1f` | CWE-22 (path) | 2026-11-09 | 2026-07-01 | `src/main/java/org/zeroturnaround/zip/ZipUtil.java`, `src/main/java/org/zeroturnaround/zip/Zips.java` | checkDestinationFileForTraversal, process | 8 / 34,500 |
| 3 | nuxt/content | `d8fed0c3` | CWE-400, CWE-770 (dos) | 2026-11-10 | 2026-07-23 | `src/runtime/internal/security.ts` | assertSafeQuery, cleanupQuery, strippingFence | 8 / 36,432 |
| 4 | naturalintelligence/fast-xml-parser | `56805655` | CWE-407 (dos) | 2026-11-10 | 2026-11-03 | `src/xmlparser/OrderedObjParser.js` | readStopNodeData, sanitizeName | 11 / 47,541 |
| 5 | gradleup/shadow | `e822f993` | CWE-22 (path) | 2026-11-09 | 2026-07-09 | `src/main/kotlin/com/github/jengelman/gradle/plugins/shadow/internal/Utils.kt` | zipEntry | 12 / 51,405 |
| 6 | agamnentzar/ag-psd | `e64f0c96` | CWE-770 (dos) | 2026-11-09 | 2026-07-01 | `src/psdReader.ts` | decodeLayerImageData | 15 / 68,586 |
| 7 | tmatens/compose-lint | `c43204ad` | CWE-117, CWE-150 (injection, validation) | 2026-11-10 | 2026-09-26 | `src/compose_lint/_output.py` | _escape, sanitize, sanitize_line, _defuse_commands | 18 / 83,214 |
| 8 | kiwitcms/kiwi | `494385fc` | CWE-601 (redirect) | 2026-11-09 | 2026-07-20 | `tcms/kiwi_auth/views.py` | get_redirect_url | 22 / 101,223 |
| 9 | urwid/urwid | `567e3006` | CWE-116, CWE-248, CWE-775 (dos, injection) | 2026-11-09 | 2026-07-24 | `urwid/display/web.py` | set_input_timeouts, is_web_request, handle_short_request, _request_charset | 24 / 109,917 |
| 10 | digitalbazaar/forge | `2bd5146c` | CWE-347 (crypto) | 2026-11-10 | 2026-11-03 | `lib/ed25519.js` | crypto_sign_open, _isCanonicalSignatureScalar, _isSmallOrder | 25 / 116,196 |
| 11 | filegator/filegator | `d876ac38` | CWE-862 (access) | 2026-11-09 | 2026-08-13 | `backend/Controllers/UploadController.php` | upload | 6 / 27,393 |
| 12 | zeroturnaround/zt-zip | `780a2fd9` | CWE-23 (path) | 2026-11-09 | 2026-07-01 | `src/main/java/org/zeroturnaround/zip/ZipUtil.java` | checkDestinationFileForTraversal, process, destinationIsOutputDir, isSafeDescendantEntryName | 8 / 35,259 |
| 13 | tmatens/compose-lint | `8e4fac95` | CWE-200, CWE-532 (info) | 2026-11-10 | 2026-10-03 | `src/compose_lint/engine.py` | FindingLimitError, run_rules | 19 / 86,250 |
| 14 | digitalbazaar/forge | `326a4d09` | CWE-20, CWE-248 (validation) | 2026-11-10 | 2026-11-03 | `lib/tls.js` | parseHelloMessage | 26 / 118,818 |

Context sizes: `contexts.json`. Atlas indexes: `index/<task>.txt`.

## Contexts and isolation

Each parent commit is fetched at depth 1 from GitHub. The atlas (`repo_atlas.build`, 88.2 px² per character,
1932-px pages at the high-resolution tier) and the file list come from that checkout. The agents work in a
`git archive` export of it (no `.git`, so no history and no fix). The fix commits and patches are deleted from or
kept outside the work tree before the first call. The agents run with Claude Code's default permissions, so reads
outside the export are denied; every tool call with a path outside it (absolute or `..`) is recorded.

## Arms

All arms: Opus 5.5 (`claude-opus-5-5`), `--effort high`, one call or run per task and arm. The same instruction
ends every prompt:

> This repository, at this commit, contains at least one security vulnerability that was fixed later. Audit it and
> report up to 5 findings, most likely first, each on one line as `FINDING: <file path> | <function or line range> |
> <weakness class, CWE-style> | <one sentence why>`, with file paths exactly as they appear in the repository.
> Report locations and classes only: no exploit code, no proof of concept.

1. `closed`: "Repository <owner/repo> at commit <sha7>. Its code is not shown: only the paths of its N source
   files", then the paths (the atlas's file set). One call, no tools. System prompt `SYSTEM` (an auditor that reports
   where flaws are, in the asked format).
2. `atlas`: the page images, then the atlas note (format of the minified code) and `index.txt`, then the
   instruction. One call, no tools, same system prompt.
3. `agent`: Claude Code headless (`claude -p`, stream-json input) in the export, tools `Read,Grep,Glob` only (no
   Bash, edits, web or MCP; no settings sources), Claude Code's own system prompt, `--max-turns 40`, and a run-away
   guard `--max-budget-usd 4` (recorded if hit). First message: "The current directory is a checkout of the
   <owner/repo> repository at commit <sha7>. Use your tools to look at the code as much as you need; do not change
   anything." and the instruction.
4. `agent_atlas`: the same agent; its first message holds the page images, the atlas note and index, the agent
   sentence and: "The page images show the repository's source; use them for orientation and read the files when
   exact text matters." Then the instruction.

Agents: the final answer is the result text; if a run stops on its turn or budget limit, the last assistant text
with FINDING lines.

## Scores (objective, no judge)

Per task and arm, from the parsed `FINDING:` lines (at most 5; paths normalised; a path not in the repository at the
parent counts as wrong and is reported):

* **primary: a gold file is named in the top 5 findings;**
* a gold file is the first finding;
* a gold function is named (whole token) in the location or reason of a finding that names a gold file;
* class match: the first finding that names a gold file gives a weakness class in a family of the advisory's CWE
  ids. The mapping (`FAMILY_CWES`: CWE id -> family; `FAMILY_WORDS`: words -> family when no CWE id is given) is in
  the code and committed with this file;
* number of findings; cost (list price, the CLI's `total_cost_usd`), input tokens, seconds, turns.

Paired exact McNemar tests on the primary score: `atlas` vs `closed`, `agent_atlas` vs `agent`, `agent` vs `atlas`.
Turns: paired median of `agent_atlas` − `agent`.

**What ~14 tasks can and cannot show.** With n paired tasks, McNemar needs at least 6 discordant tasks all in one
direction to reach p < 0.05. So only a large effect can be significant; a tie or a small difference says little.
The pass/fail predictions below are counts, not significance tests. Known biases: a finding can point to a real flaw
that is not the advisory's (another advisory of the same repository, or an unknown one) and is scored as a miss;
gold functions are approximate; class families are coarse.

## Checks

* First atlas call and first `agent_atlas` run: first-turn input tokens at least 97% of the image-token formula
  (sum of ceil(W/28) x ceil(H/28)); every atlas row records it, and a row below is invalid (pages dropped).
* A refusal or a cut-off by a safety classifier is the result for that task and arm (no findings); nothing is
  reworded or retried with another prompt. Errors (API, network) are retried and never recorded as results. A
  usage-limit error stops the run; rows so far are committed.

## Predictions (the coordinator's)

* **V1** `closed` names a gold file in its top 5 on at most 30% of tasks.
* **V2** `atlas` beats `closed` by at least 3 tasks on the primary score.
* **V3** `agent` is at least as good as `atlas` (primary score).
* **V4** `agent_atlas` and `agent` differ by at most 1 task on the primary score, and `agent_atlas` uses fewer turns
  (paired median).

## Stages and cost (list price; hard cap $25)

* Pilot: tasks 1-3 x 4 arms, then a cost re-estimate (written below as an amendment).
* Then the remaining tasks in run order, starting a task only if spend so far plus a conservative bound for the whole
  task (`task_bound`: the largest cost per arm so far, scaled by atlas size for the atlas arms, x 1.15) stays under
  $25. At most 2 calls or runs at once.
* Estimate before the pilot: `closed` ~$0.1, `atlas` ~$0.15 + $5 per million image tokens (~$0.3-0.8), `agent`
  ~$1-2 (40 turns at most), `agent_atlas` ~$1.5-3 (the pages are re-read from the cache each turn). About $3-5 per
  task, so the cap most likely covers **5 to 8 tasks, not 14**. That makes the McNemar tests weaker still.

## Amendment 1 (2026-11-12, after the pilot; nothing above changed)

Pilot (tasks 1-3, 12 rows): $2.86 at list price, about $1 per task (closed $0.03, atlas $0.22-0.25, agent
$0.11-0.63, agent_atlas $0.27-0.53). Far below the $3-5 estimate. Re-estimate for tasks 4-14 (atlases 35k-119k image
tokens): ~$1-2 per task, ~$13-20 in all, so all 14 tasks should fit under the $25 cap; `task_bound` still guards
each start. All atlas pages arrived (first-turn input 104-122% of the formula). No tool call left the export; no
permission denials. No refusals.

## Amendment 2 (2026-11-12, 11:05 UTC): closed on 12 tasks

At 10:18 UTC the subscription's 5-hour usage limit stopped the run (stated reset 14:10 UTC) during task 13
(tmatens/compose-lint `8e4fac95`): its `agent` and `agent_atlas` runs failed with the usage-limit error and wrote no
rows; their partial cost ($1.76) stays booked. Spend stood at $20.70 of $25. Task 13's pre-registered bound for all
four arms was $4.04, which leaves $0.26 of margin; that is not a clear fit, so the test closes on the **12 tasks with
all four arms** (tasks 1-12 of the run order). Tasks 13 and 14 were not run. Nothing else changed.

## Results (12 tasks, 48 rows, $20.70 at list price)

**Verdict: no, the atlas did not help find the flaw.** The file list alone already named a gold file in its top 5
on 10 of 12 tasks; the one-shot atlas did so on 9. On code the model cannot remember, file and directory names
still point at the flaw (`security.ts`, `ZipUtil.java`, `kiwi_auth/views.py`, `display/web.py`), and 5 guesses
among a few dozen to a few hundred files is enough. The primary score is near its ceiling for every arm, so this
test cannot show an atlas benefit even if one exists. One direction is worth a follow-up: the agent **with** the
atlas found a gold file on 10 tasks against 7 for the plain agent (3 to 0 discordant, exact p = 0.25), in about half
the turns (median 8 vs 18), at 1.5 times the cost ($0.67 vs $0.46 per task). With 12 tasks that is not significant.

### Per arm

| arm | tasks | gold file in top 5 | gold file first | gold function named | class match | mean input tokens | mean cost | median turns |
|---|---|---|---|---|---|---|---|---|
| closed (file list) | 12 | 10 | 6 | 5 | 6 | 2,695 | $0.032 | 1 |
| atlas (pages, one call) | 12 | 9 | 5 | 5 | 5 | 64,839 | $0.417 | 1 |
| agent (Read/Grep/Glob) | 12 | 7 | 4 | 6 | 5 | 500,901 | $0.458 | 18 |
| agent_atlas | 12 | 10 | 6 | 7 | 7 | 630,873 | $0.672 | 8 |

"Gold function named" counts only functions named in a finding about a gold file. Class match: the first finding
on a gold file gives a weakness class in the advisory's CWE family (`FAMILY_CWES`). Mean findings: 5.0, 5.0, 4.4,
5.0. Wrong paths (not in the repository): closed 1, others 0.

### Paired tests (primary score)

| pair | first only | second only | exact McNemar p |
|---|---|---|---|
| atlas vs closed | 1 | 2 | 1.000 |
| agent_atlas vs agent | 3 | 0 | 0.250 |
| agent vs atlas | 1 | 3 | 0.625 |
| agent vs closed | 0 | 3 | 0.250 |
| agent_atlas vs closed | 1 | 1 | 1.000 |

Turns, agent_atlas − agent (paired): median −8.5; fewer on 11 of 12 tasks.

### Per task

`#n` = rank of the first finding on a gold file; `fn` = a gold function named; `cls` = class match.

| task | repository | CWE (family) | gold files | atlas tokens | closed | atlas | agent | agent_atlas |
|---|---|---|---|---|---|---|---|---|
| digitalbazaar__forge@2bd5146c | digitalbazaar/forge | CWE-347 (crypto) | 1 | 117,852 | #4 cls | miss | no findings (turn limit) | miss |
| digitalbazaar__forge@326a4d09 | digitalbazaar/forge | CWE-20, CWE-248 (other,validation) | 1 | 120,474 | not run | not run | not run | not run |
| naturalintelligence__fast-xml-parser@56805655 | naturalintelligence/fast-xml-parser | CWE-407 (dos) | 1 | 48,852 | #3 cls | #4 cls | #2 fn cls | #1 fn cls |
| tmatens__compose-lint@8e4fac95 | tmatens/compose-lint | CWE-200, CWE-532 (info) | 1 | 87,492 | not run | not run | not run | not run |
| nuxt__content@d8fed0c3 | nuxt/content | CWE-400, CWE-770 (dos) | 1 | 36,915 | #1 fn | #1 fn | #1 fn | #1 fn |
| tmatens__compose-lint@c43204ad | tmatens/compose-lint | CWE-117, CWE-150 (injection,validation) | 1 | 84,387 | #1 cls | #4 fn cls | miss | #4 cls |
| kiwitcms__kiwi@494385fc | kiwitcms/kiwi | CWE-601 (redirect) | 1 | 102,603 | #5 | #1 fn cls | #1 fn cls | #1 fn cls |
| filegator__filegator@d876ac38 | filegator/filegator | CWE-862 (access) | 1 | 27,738 | #1 fn | miss | #1 fn cls | #1 fn cls |
| filegator__filegator@c6715ff7 | filegator/filegator | CWE-330, CWE-862 (access,crypto) | 2 | 27,600 | miss | #2 | miss | #2 fn cls |
| urwid__urwid@567e3006 | urwid/urwid | CWE-116, CWE-248, CWE-775 (dos,injection,other) | 1 | 111,504 | #1 fn | #5 fn | miss | #4 |
| gradleup__shadow@e822f993 | gradleup/shadow | CWE-22 (path) | 1 | 53,475 | miss | miss | miss | miss |
| agamnentzar__ag-psd@e64f0c96 | agamnentzar/ag-psd | CWE-770 (dos) | 1 | 69,552 | #3 cls | #1 cls | #3 cls | #1 cls |
| zeroturnaround__zt-zip@780a2fd9 | zeroturnaround/zt-zip | CWE-23 (path) | 1 | 35,673 | #1 fn cls | #1 | #2 fn | #2 fn |
| zeroturnaround__zt-zip@0a9d0e1f | zeroturnaround/zt-zip | CWE-22 (path) | 2 | 34,983 | #1 fn cls | #1 fn cls | #1 fn cls | #1 fn cls |

### Predictions

* **V1** (`closed` gold file in top 5 on at most 30% of tasks): **FAIL.** 10 of 12 (83%).
* **V2** (`atlas` beats `closed` by at least 3 tasks): **FAIL.** atlas 9, closed 10.
* **V3** (`agent` at least as good as `atlas`): **FAIL.** agent 7, atlas 9 (1 to 3 discordant, p = 0.63).
* **V4** (`agent_atlas` and `agent` within 1 task, and `agent_atlas` fewer turns): **FAIL** on the first part (10 vs
  7), PASS on the second (median −8.5 turns, fewer on 11 of 12).

### What 12 tasks can and cannot show

None of the paired tests is significant; the largest split (3 to 0) gives p = 0.25. With the file-list floor at 10
of 12, the primary score leaves almost no room above it, so "the atlas adds nothing over the file list" is the
plain reading, not "the atlas is worse". The stricter scores (gold file first, gold function, class) are just as
flat. The `agent_atlas` lead over `agent` is a direction for a larger test, not a finding.

### Where the plain agent lost

* forge `2bd5146c` (ed25519 signature check): the agent read `lib/ed25519.js` among 14 files, ran into the 40-turn
  limit while still auditing, and wrote no FINDING lines (scored as no findings).
* filegator `c6715ff7`: it reported the upload flaw that is the *other* filegator advisory (task 11's gold), scored a
  miss as pre-registered. Several arms reported the sibling advisory's file in the two-task repositories
  (agent_atlas named `lib/tls.js`, task 14's gold, on forge `2bd5146c`).
* compose-lint `c43204ad` and urwid: it looked elsewhere; the atlas arms named the gold file at rank 4-5.

### Cost and checks

| arm | spent |
|---|---|
| closed | $0.39 |
| atlas | $5.00 |
| agent | $6.20 (incl. $0.71 of the failed task-13 run) |
| agent_atlas | $9.11 (incl. $1.05 of the failed task-13 run) |
| **total** | **$20.70 of $25** |

* All 24 atlas and agent_atlas rows: pages arrived (first-turn input 102-127% of the image-token formula).
* No refusals, no safety cut-offs, no permission denials. One agent Grep used `urwid/widget/../vterm.py`, inside its
  export. No tool call reached outside the export.
* One agent run (forge) hit the turn limit; no run hit the $4 budget guard.

### Problems and limits

* The primary score is too easy: names give the file away even on post-cutoff flaws. A sharper test would score the
  function or line range, or pick flaws whose file name says nothing about security.
* Gold functions are approximate (enclosing declarations of changed lines) and class families are coarse.
* Two tasks per repository share code: a finding on the sibling advisory's flaw counts as a miss.
* `repo_atlas.collect` drops files that contain its marker characters (`§`, `▶`): one candidate advisory lost 3 of 4
  gold files that way (rule 5 excluded it).
