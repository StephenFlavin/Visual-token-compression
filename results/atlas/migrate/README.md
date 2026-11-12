# Repo atlas, migration test: does the C project as page images help Claude Code migrate it to Rust?

*Pre-registration, 2026-11-12, written and pushed before the first agent run. Runner:
`experiments/atlas/repo_atlas_migrate.py` (tests `tests/test_repo_atlas_migrate.py`). Plan and earlier atlas results:
[docs/claude/REPO_ATLAS_PLAN.md](../../../docs/claude/REPO_ATLAS_PLAN.md), [results/atlas/pilot/README.md](../pilot/README.md).*

## Results (2026-11-12)

*14 C projects (2 large: 3,759-3,889 C lines; 12 small: 806-1,083) x 3 arms = 42 Claude Code runs, Opus 5.5 high
effort. $24.01 at list price of the $30 cap ($0.02 smoke, $12.72 pilot, $11.27 smaller end). Rows `runs.jsonl`,
tables `summary.md` (`repo_atlas_migrate.py report`). The pre-registration and its three amendments follow below.*

**Verdict: the atlas neither helps nor hurts a migration, and here the pages could stand in for the source, but this
test cannot show much.** All three arms have the same outcome on every one of the 14 projects: 12 pass every test in
every arm, lib2bit fails its one test in every arm, and Genetic passes 22 of 23 in every arm (the 23rd runs 50,000 GA
generations and times out everywhere). Turns, cost and time differ by noise. The `atlas` agents still opened the C
files in all 14 runs, usually with one `cat` of everything, so the pages saved them about one tool call. `atlas_only`
agents never saw the C files and did just as well, so the tests could not tell pages from source. Two reasons limit
what this shows. First, the task was easy where it was cheap: the small projects were done in 4-9 turns and about a
minute, at the ceiling. Second, the Rust interfaces and tests specify much of the behaviour, and CRUST-Bench and its C
projects have been public since 2025. So "pages replaced the source" means only that nothing here needed the source
beyond what the pages, the interfaces and memory gave. How sure: the arms tie exactly, so there is no sign of an
effect, but 14 projects at the ceiling cannot rule out a few-point difference on harder migrations.

| Arm | Projects | All tests pass | Mean share passing | Builds | Median turns | Mean cost | Median seconds (agent) | Median output tokens | `unsafe` |
|---|---|---|---|---|---|---|---|---|---|
| plain | 14 | 12 | 92.5% | 14 | 6.5 | $0.54 | 72 | 8,210 | 0 |
| atlas | 14 | 12 | 92.5% | 14 | 6.0 | $0.57 | 63 | 7,139 | 0 |
| atlas_only | 14 | 12 | 92.5% | 14 | 6.5 | $0.60 | 80 | 9,525 | 0 |

Large end (pilot, 2 projects): each arm 1 of 2 all-pass, 97.8% of tests. Smaller end (12): each arm 11 of 12, 91.7%.
First 10 projects (the planned size): each arm 8 of 10, 89.6%. No arm left an `unimplemented!()`.

| Project | C lines | Tests | plain: passed / turns / $ / s | atlas | atlas_only |
|---|---|---|---|---|---|
| Genetic_neural_network_for_simple_control | 3,889 | 23 | 22 / 26 / 1.72 / 488 | 22 / 28 / 2.58 / 991 | 22 / 31 / 2.14 / 746 |
| recordManager | 3,759 | 3 | 3 all / 17 / 1.79 / 428 | 3 all / 19 / 1.97 / 468 | 3 all / 18 / 2.51 / 636 |
| SimpleXML | 806 | 3 | 3 all / 7 / 0.29 / 65 | 3 all / 5 / 0.22 / 51 | 3 all / 6 / 0.29 / 78 |
| utf8 | 806 | 25 | 25 all / 6 / 0.27 / 58 | 25 all / 7 / 0.26 / 53 | 25 all / 4 / 0.22 / 49 |
| clhash | 824 | 5 | 5 all / 6 / 0.25 / 56 | 5 all / 5 / 0.26 / 55 | 5 all / 6 / 0.40 / 103 |
| Simple_Sparsehash | 845 | 12 | 12 all / 6 / 0.30 / 76 | 12 all / 8 / 0.28 / 75 | 12 all / 5 / 0.23 / 63 |
| proj_2DPartInt | 855 | 6 | 6 all / 7 / 0.35 / 67 | 6 all / 6 / 0.38 / 70 | 6 all / 7 / 0.41 / 97 |
| Phills_DHT | 875 | 29 | 29 all / 4 / 0.11 / 23 | 29 all / 5 / 0.12 / 23 | 29 all / 5 / 0.11 / 23 |
| lib2bit | 911 | 1 | 0 / 9 / 0.56 / 143 | 0 / 6 / 0.46 / 111 | 0 / 7 / 0.58 / 164 |
| rubiksolver | 923 | 6 | 6 all / 7 / 0.29 / 54 | 6 all / 5 / 0.26 / 48 | 6 all / 7 / 0.31 / 74 |
| c_string | 929 | 15 | 15 all / 4 / 0.21 / 50 | 15 all / 4 / 0.19 / 39 | 15 all / 4 / 0.20 / 48 |
| simple_lang | 944 | 3 | 3 all / 6 / 0.45 / 92 | 3 all / 5 / 0.28 / 55 | 3 all / 7 / 0.33 / 81 |
| clog | 1,060 | 1 | 1 all / 8 / 0.59 / 154 | 1 all / 7 / 0.31 / 79 | 1 all / 4 / 0.24 / 60 |
| roaring_bitmap | 1,083 | 15 | 15 all / 6 / 0.42 / 111 | 15 all / 9 / 0.43 / 109 | 15 all / 8 / 0.43 / 122 |

**Paired tests** (14 projects): all-tests-pass is identical in every pair (McNemar: 0 discordant pairs, p = 1).
atlas vs plain: turns 7 fewer / 6 more / 1 tie (median -0.5, sign test p = 1.0); cost 8 lower / 6 higher (median
-$0.01, p = 0.79); agent seconds 10 faster / 3 slower (median -3.5 s, p = 0.09); output tokens median -517 (p = 0.42).
atlas_only vs plain: turns 5 / 5 / 4 ties (p = 1.0), cost 7 / 7 (p = 1.0), seconds median +12 s (p = 0.58), output
tokens median +1,440 (p = 0.79).

**Predictions:**
* **M1** (plain and atlas differ by at most 1 project on all-tests-pass): **PASS** (they differ by 0).
* **M2** (atlas uses fewer turns than plain, paired median): **PASS on the letter only**: median -0.5 turns, but 7
  fewer against 6 more (p = 1.0). There is no real difference.
* **M3** (atlas_only at least 2 projects or 10 points worse than plain): **FAIL**: identical on every project (12 of 14,
  92.5% both).
* **M4** (atlas cost within 20% of plain): **PASS**: mean $0.57 vs $0.54 (+5%); paired median -$0.01.

**Funnel:** 100 projects -> 100 matched to their Rust crate -> 55 with 800-4,000 C lines -> 52 with at least 3 C files
-> 50 validate. Run: the 2 largest (pilot), then (amendment 2) the 12 smallest of the 50.

**Cost:** the large projects cost $1.72-2.58 per run ($12.72 for 6 runs), the small ones $0.11-0.59 ($11.27 for 36).
At the pilot's cost only 4 large projects would have fit, so the pre-registered switch to the smaller end applied.
$5.99 of the cap is unspent. Amendment 3 bounded the extension at the 12 queued projects, and more projects at the
ceiling would add nothing.

**Checks and problems (all reported, none changes a score):**
* Pages arrived on all 28 atlas runs (first-turn input 1.5 to 4.5 times the image-token formula; the turn also holds
  the system prompt and the tool definitions).
* No run used `unsafe`, added a dependency or changed an interface type (two flags were artifacts: a parameter rename
  in utf8 and a trailing comma removed by rustfmt in proj_2DPartInt).
* All three roaring_bitmap agents rewrote `src/bin/benchmark.rs` (a benchmark binary next to the tests); it was
  restored before grading, and the 15 tests passed with the original. The Genetic `atlas` and `atlas_only` agents
  added `opt-level = 3` to Cargo.toml to speed up the slow test; it still timed out.
* No `atlas_only` run touched anything outside its work directory: no network, no filesystem-wide search, no other
  run's files (audit flags read by hand: Rust `use` lines and string literals). **One contamination, between other
  arms:** Claude Code's auto-memory (`~/.claude/projects/<cwd>/memory/`) is visible to every run. The recordManager
  `atlas` agent read the Genetic `atlas` agent's notes there (`cat ~/.claude/projects/*/memory/...`: generic porting
  notes about another project, such as stdin defaults and a faster build profile). From the smaller-end runs on, each
  run's store was deleted when it ended, and the audit shows no further reads. Next time, turn auto-memory off.
* Harness faults fixed during the test (amendments 1-3): per-binary grading (one hours-long test hid the other
  binaries' results); tests keyed by binary; turns, time and usage summed over the result events that a background
  task's wake-up adds. The pilot rows were re-graded from the kept crates (the old scores are in each row as
  `grade_v1`). Every fix applies to all arms alike.
* lib2bit's single test compares output with an `expected` file that no arm matched (the agents saw the failure and
  stopped). It may be a defect in the benchmark's expected output; it is the same in every arm.
* Two runs (Genetic `atlas`, `atlas_only`) were killed at the 30-minute wall clock after they had answered, because a
  long test they had started in the background was still running. They are graded as left, and "seconds" is the agent's
  own time.
* Usage limit: the stop at about 10:18 UTC hit this session, not a child run (every child run ended with its own
  result). No run was lost or retried.

## Why this test

In the bug-fix pilot the atlas had nothing to add: Claude Code found the right files by searching, for a few cents.
A migration is different: every line of the C code has to be read and carried over. So if pages that hold the whole
project ever help a coding agent, it should be here. Against that: pages lose about 3 points where one small word or
symbol decides ([docs/claude/CLAUDE_QA.md](../../../docs/claude/CLAUDE_QA.md), repeat calls), and a translation needs every constant
and operator exact.

## The benchmark (what is actually there)

CRUST-Bench (`github.com/anirudhkhatry/CRUST-bench`, commit `4b47025`, cloned 2026-11-12): C-to-safe-Rust over whole
repositories.

* `datasets/CRUST_bench.zip` holds `CBench/` (100 C projects, each with the upstream project's `.git`) and `RBench/`
  (100 Rust crates, hand-written). A crate has `src/interfaces/*.rs` (types and function signatures, bodies
  `unimplemented!()`; a few helper bodies such as constructors are filled in), `src/bin/test_*.rs` (the tests, as
  `#[test]` functions in bin targets), `src/lib.rs` and `Cargo.toml` (18 of the 52 candidates depend on crates such as
  `libc` or `rand`).
* **How the benchmark grades** (`src/check_benchmarks/check_build.py`, `src/compile_projects.py`,
  `src/utils/compile_rust_utils.py`): the interface files are moved from `src/interfaces/` to `src/`, then
  `cargo test` runs with a 60 s timeout; the score counts `... ok` and `... FAILED` in its output, and a project
  passes when there is no failure and at least one ok.
* **Reference solutions: none published.** No branch of the repository tracks a Rust implementation (the only `.rs`
  file outside the zip is a stub `main.rs`), and none of the 23 versions of the zip in its history holds anything but
  skeletons and tests. The `additional-datasets` branch mentions "the reference port" in its mutation-testing tools,
  but that port is not in the repository. The C projects' `.git` histories are the upstream C projects' own; they are
  dropped from the work directories.

**Local grading** (the benchmark's criterion, made stricter in two places): the same layout step; `cargo test
--no-run` (20 min limit), then `cargo test --no-fail-fast` (5 min limit). `--no-fail-fast` because plain `cargo test`
stops at the first failing test binary, and a binary that crashes prints no `FAILED` line, so the benchmark's count can
call a partial run a pass. Each test is keyed `<bin>::<test>` and checked against the list of tests that the untouched
skeleton runs; a test missing from the output counts as failing. The benchmark's own `ok`/`FAILED` counts are kept in
every row too.

## Selection rule and funnel (fixed before any agent run)

C size = physical lines of all `.c` and `.h` files of the C project. A project is a candidate if it has 800 to 4,000
such lines in at least 3 `.c`/`.h` files. It validates if the untouched skeleton's tests compile and do not all pass
(they must fail before any implementation). Candidates are taken by C lines, largest first; the first 10 that validate
are the test set.

| Step | Projects |
|---|---|
| CRUST-Bench | 100 |
| C project matched to its Rust crate | 100 |
| 800 to 4,000 C lines | 55 |
| and at least 3 `.c`/`.h` files | 52 |
| tests compile and fail on the skeleton | 50 (impcheck: the test run hangs past 5 min; libtinyfseq: its 2 tests pass unimplemented) |

(A first validation pass used `cargo --offline` and wrongly failed 18 projects whose crates have dependencies; it was
redone online before anything else. Record of the first pass kept in the scratch directory only.)

**Test set, in order** (`selection.json`; pages and image tokens are the atlas):

| # | Project | C lines | C files | Tests | Atlas pages | Image tokens | Source as text (est. tokens) |
|---|---|---|---|---|---|---|---|
| 1 | Genetic_neural_network_for_simple_control | 3,889 | 50 | 23 | 3 | 13,386 | 54,790 |
| 2 | recordManager | 3,759 | 18 | 3 | 3 | 11,316 | 48,380 |
| 3 | worsp | 3,227 | 5 | 4 | 3 | 12,075 | 52,604 |
| 4 | libpgn | 3,142 | 35 | 21 | 3 | 12,144 | 51,647 |
| 5 | XOpt | 3,087 | 5 | 2 | 2 | 9,384 | 38,772 |
| 6 | fslib | 2,885 | 27 | 50 | 2 | 7,383 | 34,973 |
| 7 | libutf | 2,858 | 45 | 11 | 2 | 8,832 | 35,963 |
| 8 | skp | 2,338 | 5 | 3 | 2 | 7,107 | 31,136 |
| 9 | razz_simulation | 2,232 | 6 | 1 | 2 | 6,486 | 27,603 |
| 10 | c_aces | 2,227 | 21 | 11 | 2 | 7,245 | 29,996 |

The pilot is projects 1 and 2. Several projects have very few tests (1 to 4), so "all tests pass" there is a coarse
check.

**Smaller-end order** (used only under the amendment rule below): the same 50 validated projects ordered by C lines,
smallest first: SimpleXML (806), utf8 (806), clhash, Simple_Sparsehash, proj_2DPartInt, Phills_DHT, lib2bit,
rubiksolver, c_string, simple_lang, ...

## The atlas

Every `.c` and `.h` file of the C project (all directories, the C project's own tests included, since the `plain` arm
sees them too), minified and rendered by `experiments/atlas/repo_atlas.py`'s minifiers and renderer at 88.2 px² per
character, 1932 px wide, a `▶path◀` header per file and a line number on every 10th line, plus `index.txt` (file ->
pages). `repo_atlas.collect` would have skipped C files under `test/`, `examples/` or `build/`-like directories, so this
test collects the files itself (`build_atlas`); the minifier covers `.c` and `.h` (brace family), and no C file went
uncovered. Makefiles, READMEs and data files are not in the atlas. The test set's atlases are 2 or 3 pages, 6-13k image
tokens, about 4x fewer tokens than the same source as text.

## Arms

Claude Code headless (`claude -p`, CLI 2.1.322), model `claude-opus-5-5`, `--effort high`, tools Bash, Read, Edit,
Write (WebFetch and WebSearch off; `curl`, `wget` and `git clone` denied), no settings files, no MCP, no slash
commands, no session saved, a clean environment (the `claude_eval` whitelist), `--max-turns 60`, a 30 min wall clock
(the process group is killed; the work directory is graded as left) and `--max-budget-usd 5` per run (the same for
every arm; a run that hits it is graded as left). The first message goes in as stream-json, so images arrive
unresized.

1. `plain`: the work directory holds `c_src/` (the C project, without `.git`) and `rust/` (the formatted skeleton).
2. `atlas`: the same work directory, plus the atlas pages and `index.txt` in the first message with: "The page images
   above show the same C source ... Use them for orientation and read the files when exact text matters."
3. `atlas_only`: `rust/` only. The first message holds the pages and says the C source is not reachable and the pages
   are the only view of it.

The instruction is otherwise the same in every arm (`COMMON` in the runner): implement the interfaces as a faithful
migration of the C library in safe Rust so that the provided tests pass; do not edit the tests or the interface
signatures; no `unsafe`; no dependencies beyond `Cargo.toml`; work only with what is in the directory, do not look for
the project, its C source or other ports elsewhere or online; reply DONE.

**Keeping the C source away from `atlas_only`:** outside the work directories of running `plain`/`atlas` runs the C
sources exist only as one XOR-scrambled tar in the scratch directory (the clone and the zip are deleted), so `find` or
`grep` cannot come across them; an `atlas_only` run never runs next to a `plain` or `atlas` run of the same project
(the scheduler waits); every transcript is audited for absolute paths outside the work directory, filesystem-wide
searches and network commands, and any hit marks the run `suspect`, to be read by hand. A run that did reach the C
source or a port is reported as invalid, and the comparisons are given with and without its project.

## Scores (per project and arm)

* **All tests pass** (primary): the crate builds and every test of the skeleton's list passes.
* Share of tests passing; builds or not; `unsafe` blocks, functions and impls in the library code; `unimplemented!()`
  left; turns; cost (the CLI's `total_cost_usd`, list price); seconds; output tokens.
* **Integrity:** tests (`src/bin/`) must be byte-identical and no test file may be added: if not, they are restored
  before grading and the row says so. Every public signature and type of every interface file must still be there:
  if not, the row says so (graded as left, since restoring would delete the implementation). Cargo.toml changes are
  recorded.
* **Pages arrived:** on every atlas run, the first turn's input tokens must be at least 97% of the image-token formula.

## Analysis

Paired by project: exact McNemar on all-tests-pass (atlas vs plain, atlas_only vs plain, atlas_only vs atlas); paired
medians with exact sign tests for turns, cost, seconds, output tokens and share of tests passing.

What 6-10 projects can show: only large effects. With 10 pairs, McNemar reaches p < 0.05 only if at least 6 projects
are discordant and all in one direction. A tie on all-tests-pass says the arms are within a few projects of each
other, not that they are equal. Turns and cost differences of 20-30% are visible in a sign test only if they are
consistent across nearly all projects.

**Training data caveat:** CRUST-Bench has been public since April 2025 (paper and repository), and most C projects
are public GitHub repositories, so the model may have seen the C code, the skeletons and possibly ports. This would
help every arm, `atlas_only` most of all.

## Predictions (the coordinator's)

* **M1** `plain` and `atlas` differ by at most 1 project on all-tests-pass.
* **M2** `atlas` uses fewer turns than `plain` (paired median).
* **M3** `atlas_only` does worse than `plain`: at least 2 projects fewer with all tests passing, or a mean share of
  passing tests at least 10 points lower.
* **M4** the cost of `atlas` is within 20% of `plain` (mean over projects; output dominates a migration).

## Cost estimate and stopping rule

Hard cap $30 at list price for every Claude call of this test (a $0.017 smoke call of the stream-json agent setup is
already in `spend.jsonl`). Opus 5.5 list prices (fitted on earlier rows): input $4, cache write $5, cache read $0.20,
output $20 per million tokens. A 60-turn migration at high effort with a context growing to ~100k tokens: ~3M cache
reads ($0.6), ~150k cache writes ($0.75), 40-90k output ($0.8-1.8): **about $2-4 per run, $6-12 per project**. The
atlas adds 6-13k image tokens read on every turn: about $0.15 per run. So $30 buys about 2.5-5 projects at the large
end.

* **Pilot:** projects 1 and 2, all three arms (6 runs, at most $30 by the per-run cap; expected $12-24).
* **Then:** the next project starts only if all three of its runs fit under the cap at the measured mean cost per run
  (x1.2), after the runs already committed; every started project is finished in all three arms. A run's
  `--max-budget-usd` is lowered to what is left under the cap if needed (recorded as `budget_usd`), so the cap holds.
* **Amendment rule (set now):** if the pilot shows that fewer than 6 projects fit in total, the remaining budget goes
  to the smaller end by the same rule (validated projects ordered by C lines, smallest first), as a dated amendment
  written here before those runs.

At most 2 agent runs at once (two other atlas tests share the container).

## Amendment 1 (2026-11-12, 09:45 UTC, after the first pilot row, before any comparison)

Two harness faults found on the first row (Genetic_neural_network_for_simple_control, `plain`), fixed for every arm
alike:

* **Grading.** One test (`test_full_run::testNNRun`, 50,000 generations) runs for hours. Under the single 5-minute
  `cargo test` call, the binaries queued after it never ran and their tests counted as failing (3/23). Also, cargo
  prints its "Running <binary>" lines to stderr and the test lines to stdout, so tests were keyed by name only, and
  same-named tests in different binaries collapsed (fslib: 50 keys for 53 tests). **Now:** each test target (the
  library's unit tests, then every `src/bin/*.rs`) runs on its own, `cargo test --no-fail-fast --bin <name>`, with a
  5-minute limit per binary (the process group is killed on timeout; tests that finished before count). Baselines were
  recomputed the same way (`validate_v2.jsonl`; all 22 projects of the test set and the smaller-end list still
  validate). Rows graded the old way are re-graded from the crate the agent left (kept in scratch), the old scores
  kept in the row as `grade_v1`. The selection is unchanged.
* **A leak path.** Claude Code keeps per-directory auto-memory and spilled tool outputs (which can hold C source text)
  under `~/.claude/projects/<cwd>`, and the `plain` agent listed `~/.claude/projects/*/memory/` at its start. An
  `atlas_only` agent could have found another arm's notes there. **Now:** each run's directory there is deleted as
  soon as the run ends (a watcher for the runs already in flight, then in the runner), before any `atlas_only` run of
  the same project can start. The audit also flags any read of another run's directory there. Its path rule was
  refined: `//` comments and the run's own CLI store no longer count as outside paths.

## Pilot cost and amendment 2 (2026-11-12, 13:05 UTC, before any run after the pilot)

**Pilot spend:** 6 runs, $12.71 (plus the $0.02 smoke call): $1.72-2.58 per run, mean $2.12, so about $6.40 per
project, $7.60 at the runner's 1.2 margin. $17.27 is left under the cap: at this cost two more projects of the large
end fit, four in all, fewer than 6. **So the pre-registered switch applies:** the remaining budget goes to the
validated projects in smaller-end order (SimpleXML 806 C lines, utf8 806, clhash 824, Simple_Sparsehash 845,
proj_2DPartInt 855, Phills_DHT 875, lib2bit 911, rubiksolver 923, ...), each started only while its three runs fit
under the cap at the measured mean cost per run (the mean updates as runs come in), each finished in all three arms.
The two pilot projects stay in the analysis; the large/small split is reported.

**Two more harness faults, found in the pilot's transcripts (bookkeeping only, every arm alike):** an agent can start
a long test in the background (Genetic's 50,000-generation test), end its turn, and then be woken by the task's
notification; each wake writes its own result event. Cost there is cumulative (the last result is the run's
total, so the spend above is right), but turns, duration and token usage are per wake. The runner had taken only
the last result (Genetic `atlas_only` showed 2 turns instead of 31). Now turns, agent time and usage are summed over
all result events, and the answer is the first result's text. "Seconds" in the tables is the agent's own time (sum of
`duration_ms`): the wall clock also counts a background test left running until the 30-minute kill (Genetic
`atlas` and `atlas_only`: 1800 s wall vs 991 s and 746 s agent time). Wall seconds stay in the rows (`wall_s`).

## Amendment 3 (2026-11-12, 13:45 UTC, after 4 smaller-end runs, before any project past the 10th)

* The smaller-end projects cost about $0.25 per run (5-7 turns, about a minute), so the budget no longer binds at 10
  projects. The runner was started with the first 12 projects of the smaller-end order; it runs them all, each in all
  three arms, as long as the next one fits under the cap. That makes up to 14 projects instead of 10. This does not
  favour any arm (every project is paired), and the cap rule is unchanged. Results are also given for the first 10
  projects (the 2 pilot projects and the first 8 of the smaller end) to show the extension changes nothing.
* The interface check counted a parameter rename as a signature change (utf8: `_owned_str` -> `owned_str`).
  Parameter names are not part of a Rust function's type, so the check now compares parameter types only. Integrity is
  recomputed from the kept crates when the report is written.
