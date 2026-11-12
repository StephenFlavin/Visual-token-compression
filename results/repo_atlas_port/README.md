# Repo atlas, Q2: does a whole-repository port (Go to Rust) go better with the source as page images?

*Pre-registration, 2026-11-12, written and pushed before the first scored agent run. Runner:
`experiments/repo_atlas_port.py` (tests `tests/test_repo_atlas_port.py`). Plan: [docs/claude/REPO_ATLAS_PLAN.md](../../docs/claude/REPO_ATLAS_PLAN.md)
(Queued, Q2). Earlier migration test: [results/repo_atlas_migrate](../repo_atlas_migrate/README.md).*

**One run per arm on one project. This is an anecdote, not a measurement:** a difference of a few tests or turns
between two single runs says nothing about the arms. The write-up will say so.

## Results (2026-11-12)

*2 Claude Code runs (Opus 5.5, high effort), $23.14 of the $50 cap ($0.62 smoke, $11.32 plain, $11.20 atlas). Rows
`runs.jsonl` (agent side), `grades.jsonl` (tests), table `summary.md` (`repo_atlas_port.py analyse`), ports
`ports/<arm>/`, commands `commands/<arm>.txt`, transcripts `transcripts/<arm>.jsonl.gz` (token deltas and images
removed). The pre-registration and amendment 1 follow below.*

**Verdict: both arms ported gitleaks completely, so this anecdote cannot separate them.** Each Rust port builds
from clean and passes all 35 hidden tests (the Go original passes 35 here too). Turns, cost and time are nearly the
same (atlas 99 turns, $11.20, 0.59 h; plain 102, $11.32, 0.69 h). Both agents used the same method: build the Go
original into `/tmp` as a local oracle and compare their port's output with it on many inputs. That is allowed (the
prompt forbids only calling the original from the port), but it makes the hidden tests close to a formality. Neither
port is a concurrent scanner, although the Go original scans files through a 40-wide semaphore group
(`semgroup`, goroutines, mutexes). The atlas did not make the agent read less source: the atlas agent also read the
Go files and the Go module cache (`~/go/pkg/mod`: go-gitdiff, zerolog, lipgloss, filetype, archives), as plain did.
One run per arm, a well-known project (public since 2018) and a ceiling score: this shows that a 22k-line Go-to-Rust
port is within reach of one Claude Code session at about $11. It says nothing about the atlas.

| Arm | Tests passed | Builds | Concurrent scan | `unsafe` blocks | Rust lines (files) | Deps | Turns | Cost | Hours (wall) | Output tokens | Cut |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain | 35 / 35 | yes | no | 3 | 9,431 (22) | 18 | 102 | $11.32 | 0.69 | 234,160 | no (resumed once, amendment 1) |
| atlas | 35 / 35 | yes | no | 2 | 8,902 (16) | 3 | 99 | $11.20 | 0.59 | 201,480 | no (resumed once, amendment 1) |

* **Concurrency.** plain: scans sequentially. Its only thread reads `git log`'s stderr while the main thread parses
  the diff stream (the Go source does the same for that pipe). `Mutex` and atomics hold the global log level, the
  config state and an output lock. atlas: single-threaded, with no thread, rayon, channel or async anywhere. `Arc`
  only shares commit and remote metadata between findings, and atomics hold the log level. The automatic counter marks
  plain "concurrent" because of the stderr thread; by hand, neither is.
* **`unsafe`:** plain has 3 blocks (`libc::isatty`, a SIGINT handler that writes and `_exit`s, `libc::signal`
  setup); atlas has 2 (in logging and in `gitdiff.rs`). These are by hand and by the corrected counter. The counter
  first reported 1 and 0, because a `'"'` char literal opened a fake string; it was fixed and the rows were recounted.
* **Dependencies differ for an incidental reason.** The atlas agent tested the network with `curl` (denied) and then
  assumed no crates could be fetched. It built with `--offline` from the crates already in the cache (`regex`,
  `aho-corasick`, `libc`) and hand-wrote its TOML, JSON and CLI parsing and its DEFLATE, LZMA/xz/7z and zstd decoders.
  The plain agent checked with `cargo fetch` (which worked) and used 18 crates (`toml`, `serde_json`, `zip`, `tar`,
  `flate2`, `bzip2`, `lzma-rs`, `ruzstd`, `sevenz-rust`, `chrono`, ...). The atlas port is still 529 lines shorter.
  This difference comes from one agent's guess about the network, not from the pages.
* Turns count tool calls plus closing answers, without the atlas's "OK". The CLI's own `num_turns` covers only the
  resumed segments (4 and 6), because the first segments were stopped before their result events; API calls were
  82 (plain) and 97 (atlas).

**Predictions:**
* **B1** (neither arm passes every test): **FAIL**. Both pass 35 of 35.
* **B2** (shares differ by < 15 points): **PASS**, with a difference of 0 points.
* **B3** (atlas uses fewer turns): **PASS on the letter only**: 99 vs 102. That is one run per arm and a 3% difference,
  so it means nothing.

**Checks:**
* **Pages arrived:** the atlas run's first call had 94,350 input tokens, against 81,834 by the image-token formula.
* **Leaks: none found.** Every hit of the download pattern was read by hand. Most were reads of the Go module cache
  (paths with `github.com` in them: the Go dependencies' source, not a Rust port). plain also ran `cargo fetch` of
  `regex` in a scratch crate, to test the network. atlas ran `curl -sI https://index.crates.io/config.json`, which
  was denied, and one `find / -type d -name 'archives@*'` to look for a Go module. Neither ran `cargo add`, `cargo
  install`, `git clone` or `go install`, or fetched anything gitleaks-like. No dependency in `Cargo.toml` or
  `Cargo.lock` is a secret-scanner crate (the only flag is the port's own package name, `gitleaks`).
* **Outside paths:** the audit flags are `/tmp` scratch files (the Go oracle in `/tmp/ref` or `/tmp/goref` and test
  inputs), `~/go/pkg/mod`, and false positives from regexes and format strings (`/blob/`, `/commit/`, `/testsuite`).
  Neither run read `~/.claude/`, the repository, the other run's directory or the stash.
* **Memory:** auto-memory was off. Each run's store held only its session file, no `memory/` directory, and was
  deleted after `finish`.
* **Interruption:** the first segment of each run was stopped early by the mispriced guard (amendment 1). Both were
  resumed once, and both finished within 2 minutes of the resume. plain's resume used 1.3 min and $0.43; atlas's
  used 2.2 min and $0.59. Both were close to done when they were stopped, so the resumes changed little.
* **Second project:** jmespath did not run. After gitleaks, $26.86 of the cap was left, under the $30 the
  pre-registered rule needs (two runs at the full $15 guard).
* Q1's re-run was running agent runs in the same container at the same time.

## Question

The C-to-Rust test (14 small CRUST-Bench projects, atlases of 2-13k image tokens) tied on every project. Does the
atlas (the whole source repository as page images in Claude Code's first message) help when the port is large
(a 22k-line Go repository), has no Rust skeleton or interfaces to lean on, and the original is concurrent
(goroutines, wait groups, mutexes)?

## Benchmark and project

RepoMod-Bench (KDD 2026, `github.com/Modelcode-ai/mcode-benchmark`, commit `73093fc`): 21 whole-repository ports,
graded by pytest suites that drive the built CLI and so do not depend on the implementation language. The tests stay
hidden from the agent (`tests/` is outside `workspace/`); the agent gets `workspace/src/` (the original),
an empty `workspace/dst/` and `workspace/prompt.md`.

* **Project: `gitleaks`, Go to Rust** (benchmark id 007): a secret scanner (a defensive tool), 22,358 lines by the
  benchmark's count, 35 tests (`test_gitleaks.py` 16, `test_gitleaks_generated.py` 19: directory and stdin scans,
  `git` history scan, JSON / CSV / SARIF / JUnit reports, allow comments, `.gitleaksignore`, baseline, redaction,
  custom config, exit codes). Source: `github.com/gitleaks/gitleaks` at `ca20267` (the benchmark's
  `metadata.yml` `source_commit`; v8.30.0 is its version label). Build `cargo build --release`, run
  `./target/release/gitleaks`.
* The workspace's `src/` is the repository at that commit without `.git`, with the Go repository's own unit tests
  and `testdata/` left in (RepoMod-Bench does not publish its workspaces; its line count includes the Go tests).
* **A second project only if both its arms fit:** `jmespath` (Go to Rust, 888 tests) runs only if at least $30 of
  the cap is left after gitleaks (two runs at the full $15 guard) and the gitleaks runs finished cleanly. Caveat if it
  runs: the benchmark's own Rust target is `jmespath.rs`, a public crate named `jmespath`; a port that depends on it
  is invalid.

**Reference check (done, 2026-11-12):** the Go original, built here as the benchmark builds it (`go build
-buildvcs=false -o gitleaks`, Go 1.24.7), passes **35 of 35** tests (`reference.json`). **No test is excluded.**

## Arms

Claude Code headless (`claude -p`, CLI 2.1.322), model `claude-opus-5-5`, `--effort high`, tools Bash, Read, Edit,
Write (WebFetch and WebSearch off; `curl`, `wget`, `git clone` and `cargo install` denied), no settings files, no MCP,
no slash commands, a clean environment (the `vbpe/claude_eval.py` whitelist, own `TMPDIR`), auto-memory off
(`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`), `CARGO_BUILD_JOBS=2`, cwd = the run's own `workspace/` in an empty run
directory. The two arms run **in parallel** (same time of day, same machine load).

1. **`plain`**: one user message, the benchmark's own prompt (`templates/v3/prompt_cli.md.j2` rendered by its
   `generate_prompts.py` for gitleaks), with the container path `/workspace` replaced by the run's workspace path.
2. **`atlas`**: message 1 = the 18 atlas pages + a description of them + the index (file -> page) + "Reply only OK."
   (a context is read from the prompt cache only when it is its own first message); message 2 = the same prompt as
   `plain`. The pages show the source at the start; the text tells the agent to use them for orientation and read
   the files when exact text matters. The Go repository is in `src/` in both arms.

Neither arm sees the benchmark's tests or expected outputs: they exist on the machine only as one XOR-scrambled tar
(`stash/bench.bin` in the scratch directory; the benchmark clone was deleted) and are unpacked only to grade, after
both agents have ended. Go is installed, so either agent may build and run the Go original to compare outputs (the
prompt forbids only calling it from the port).

**Guards:** each run has a **$15 budget guard** (the runner reads the stream as it arrives, prices every API message
from its usage, and kills the process group when the run's cost reaches $15; the CLI's own `--max-budget-usd 15`
stops it between turns as a second line) and a **2.5 h wall clock**. A killed run is graded on what it left on disk
and marked `cut`. No turn limit (`--max-turns 500`).

## The atlas

`experiments/repo_atlas.py build` with **default settings, no exclusions** (the atlas fits): 88.2 px² per character,
pages 1932 px wide, `▶path◀` header per file, a line number on every 10th line; tests, `testdata/`, vendored and
build directories, files over 200 kB and lock/data files left out by the tool's defaults. **199 files, 18 pages,
81,834 image tokens** (limit 150k), about 4x fewer than the same files as text (~331k tokens). It holds every Go
source file (185, the rule generators under `cmd/generate/` included), the default rule table
`config/gitleaks.toml` (98k characters), the README and other docs, and `.gitleaksignore` (91k characters of
fingerprints: noise, but the default rule keeps it). Not covered by a minifier, so not in it: `go.mod`, `go.sum`,
`.gitignore` and the `.tmpl` report templates. Sizes and index: `atlas.json`.

## Grading (no Docker here)

`docker` cannot reach a daemon on this runner (checked once), so the benchmark's `test.py` steps run directly: copy
the port's source (no `target/`) to a fresh directory, `cargo build --release` (40 min limit), then the hidden
pytest suite with `--work-dir=<that dir> --run-cmd=./target/release/gitleaks`, plus `--timeout=120` per test
(`pytest-timeout`, in the benchmark's `tests/requirements.txt`; a hung test counts as failing) and a junit report
for per-test outcomes. Nothing in the tests is changed, skipped or weakened.

## Scores (per arm)

* **Tests passed of 35** (share); builds or not.
* **Concurrent or not:** counts in the Rust source (comments and strings stripped) of thread spawns / scopes, rayon,
  channels, `Mutex`, `RwLock`, `Arc`, atomics and async; a port is concurrent if it spawns threads, uses rayon or
  async. Plus one sentence, written after reading the code, on what the port does concurrently.
* `unsafe` blocks / functions / impls; lines of Rust (non-blank, all `.rs`); Rust files; dependencies.
* **Turns**: tool calls + closing answers of the task (what the CLI's `num_turns` counts: checked equal on all three
  smoke runs; also defined for a killed run); the atlas arm's "OK" turn is excluded. CLI `num_turns` and API calls
  are reported too.
* Cost (the CLI's `total_cost_usd`, list price; a killed stretch is priced from its usage), tokens, wall-clock hours,
  agent seconds (all summed over every result event and segment of a run).
* **Pages arrived** (atlas): the first API call's input tokens must be at least 97% of the image-token formula.

## Checks (reported, read by hand)

* **Leaks:** every Bash command is logged (`commands/<arm>.txt`); commands that download or fetch anything (`cargo
  add/install/search`, `git clone/fetch`, `curl`, `wget`, `go install/get`, URLs, crates.io) are listed and read by
  hand for an existing Rust port of gitleaks or its releases; `Cargo.toml` and `Cargo.lock` package names are checked
  for secret-scanner crates. The audit also lists paths outside the run's directory (toolchains and system dirs
  allowed), filesystem-wide searches and reads of `~/.claude/`.
* **Memory:** the run's store under `~/.claude/projects/<cwd>` is deleted before the run, checked for a `memory/`
  directory after it, and deleted after it.

## Predictions (from the design, unchanged)

* **B1:** neither arm passes every test on gitleaks.
* **B2:** the two arms' test shares differ by less than 15 points.
* **B3:** the atlas arm uses fewer turns.

## Spend and stop rules

* **Hard cap $50 at list price** for every Claude call of this test (`spend.jsonl`). Spent before the scored runs:
  **$0.62** (three smoke calls, below). A run starts only if the spend so far plus its full $15 guard stays under the
  cap. Expected: $5-15 per run (an atlas run reads 82k more cached tokens per API call).
* **Usage limit:** a run that ends on a usage-limit or API error is not booked as a result; the runner stops, the
  rows and state are committed, and what remains is reported (the run can be resumed later with `--resume`).
* **Restarts:** the stream goes to disk line by line, the CLI session id is fixed up front (`--session-id`), and the
  run state is in the scratch directory. If a restart kills a run, `run` resumes that session once with
  `claude -p --resume <id>` in the same directory ("The session was interrupted by a machine restart. Continue..."),
  counting its cost against the same $15 guard and its time against the same 2.5 h; a second death leaves it cut.
  `finish`, `grade` and `analyse` are standalone commands.

## Harness smoke (2026-11-12, $0.62, before this pre-registration)

* Atlas-arm smoke (two-message input, toy task): pages arrived (first call 94,318 input tokens vs 81,834 by the
  formula), the "OK" turn cost $0.47 (cache write of the pages), the task turns $0.08; the fixed session id was used.
* **Stream accounting:** the `assistant` events of the stream carry the output-token count at the start of each
  message (38 output tokens where the result reports 601), so the runner adds `--include-partial-messages` and reads
  each message's final usage from its `message_delta` event (checked: 175 = 175). Priced with cache reads at $0.40/M
  the live estimate runs 5-10% above the CLI's figure (the CLI's figures fit $0.20/M), so the guard errs early.
* Guard smoke (no pages, guard $0.05): the runner stopped the run at $0.066 (the CLI's own budget stop fired in the
  same second); 10 files written; turns 11 = CLI `num_turns` 11 (10 parallel tool calls in 2 API calls).
* No `memory/` directory was written in any smoke run; each run's store held only its session file and was deleted.

## Notes

* gitleaks has been public since 2018 and the model knows it well; RepoMod-Bench is from 2026. Both arms get that
  prior alike.
* Another test (Q1's re-run) is running agent runs in the same container at the same time; both arms share that load.

## Amendment 1 (2026-11-12, 20:25 UTC, after both first segments, before any grading)

**The guard stopped both runs early.** The live estimator priced cache reads at $0.40/M (the brief's 0.1x of $4);
the CLI's `total_cost_usd` prices them at $0.20/M (fitted exactly on the smoke runs, noted above, but the guard was
left at the conservative rate). A port reads a large context on every call (atlas 22.7M cache-read tokens in 91 API
calls, plain 16.2M in 67), so cache reads dominate and the estimate ran about 40% high. Checked from the
transcripts (usage per assistant message, de-duplicated by message id, final output counts from `message_delta`;
the de-duplication was right, only the price was off):

| Arm | Stopped at (v1 estimate) | At the CLI's prices | Time | Turns | Cache reads | Cache writes | Output |
|---|---|---|---|---|---|---|---|
| atlas | $15.15 | $10.61 | 0.55 h | 93 | 22.70M | 0.42M | 198,481 |
| plain | $15.06 | $10.90 | 0.67 h | 87 (at kill) | ~16.2M+ | ~0.41M | ~223k+ |

The coordinator asked for a check before either was graded as cut. **Fix, the same for both arms:** the estimator
now prices cache reads at $0.20/M; each run's first segment is repriced in `spend.jsonl` (a correction row), and the
run is resumed once (`claude -p --resume <session id>`, same directory, message "The session was interrupted.
Continue the task from where you left off; the files you wrote are still on disk.") with the rest of its $15 and its
2.5 h. The resume is the one allowed by the restart rule; a run stopped again is cut. The plain arm reached the old
guard about 7 minutes after the atlas arm, before the fixed runner could take over its live process, so both arms
had exactly the same interruption (atlas resumed at 20:19, plain at 20:24 UTC). The interruption itself (one extra
user message, cache possibly re-written) is a deviation from a single uninterrupted run in both arms.
