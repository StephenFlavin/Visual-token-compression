# Repo atlas: does it help find a security flaw?

*Pre-registered 2026-11-12, before any scored call. Code: `experiments/repo_atlas_sec.py` (tests
`tests/test_repo_atlas_sec.py`). Plan of the atlas: [docs/REPO_ATLAS_PLAN.md](../../docs/REPO_ATLAS_PLAN.md).
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

## Results

*(added after the runs)*
