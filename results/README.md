# results/

Rows from every experiment are `runs.jsonl` files (one JSON object per run). Summaries, plots and the conclusions
drawn from them are in [docs/glyphs/RESULTS.md](../docs/glyphs/RESULTS.md), [encoder_probe/RESULTS.md](../encoder_probe/RESULTS.md),
[encoder_probe/ROUND3.md](../encoder_probe/ROUND3.md) and [learned_glyphs/RESULTS.md](../learned_glyphs/RESULTS.md).
Images, render caches, checkpoints and live logs are git-ignored and regenerate from the commands in
[docs/RUNNING.md](../docs/RUNNING.md) (setup: [AGENTS.md](../AGENTS.md)). The Claude tests each have a `README.md`
in their folder with the pre-registration and the result; the findings in order are in
[docs/FINDINGS.md](../docs/FINDINGS.md).

## Where each stage's rows live

| Stage | Rows | Key runs |
|---|---|---|
| Phase 1: Claude reads natural Latin | `results/phase1/<run>/runs.jsonl` | `latin_natural`, `latin_scrambled`, `pilot`; `dryrun*` use a fake reader |
| Phase 1b: PP-OCR zero-shot | `results/phase1b/<run>/runs.jsonl` | `ocr_natural`, `ocr_scrambled`; tables in `summary/`, `natural_only/` |
| Phase 2: CRNNs trained per scheme/em (K3) | `results/phase2/<run>/runs.jsonl` | `phase2_main`, `phase2_long`; **`merged/`** = longest run per (scheme, em), the one to cite |
| Learned glyph alphabet | `learned_glyphs/results/<run>/` | `e1long` (banks), `e3_bridge_long` (bridge), `e2_transfer_long` |
| Frozen-encoder probe, rounds 1-2 | `encoder_probe/results/<run>/runs.jsonl` | `main`, `controls*` (SigLIP); `round2*`, `round2_gpu_tp*` (DeepEncoder) |
| Round 3 (Qwen2.5-VL) | `encoder_probe/results/<run>/runs.jsonl` | `round3_cpu_tp200` (G0), `quick_q1` (Q1), `g2lite*` (G2-lite), `round3_ft_gpu3h{,_blind}` (R3-1) |
| Claude QA: page images vs text on long documents | `results/claude_qa/<run>/runs.jsonl` | `qav2*` (pre-registered v2; pooled in `analysis_v2*/`), [`prune/`](claude_qa/prune/README.md) (Test A), [`repeat/`](claude_qa/repeat/README.md) (repeat calls), `research/`; `summary.md` |
| RAG and EnterpriseRAG-Bench with page images | [`results/rag/`](rag/README.md), [`results/erb/`](erb/README.md) | `rag/runs.jsonl`; `erb/` is prep only (retrieval, pilot question ids) |
| Compaction: archived context as page images | [`results/compaction/`](compaction/README.md), [`results/compaction_scale/`](compaction_scale/README.md) | `runs.jsonl`, `summary.json` |
| Coding agents reading page images (vbhook) | [`results/readhook_smoke/`](readhook_smoke/README.md), [`results/readhook_swe/`](readhook_swe/README.md), [`results/readhook_swe_ml/`](readhook_swe_ml/README.md) | smoke tasks; SWE-bench Django; SWE-bench Multilingual (Rust, Java) |
| Repo atlas and plan quality (Claude) | `results/atlas/<test>/` | see the next table |

## The atlas tests (`results/atlas/`)

A whole repository as page images, read up front. Runners are in `experiments/atlas/`; each folder's `README.md`
holds the pre-registration, tables and caveats. Before the 2026-11-12 restructure these folders were
`results/repo_atlas*` and `results/plan_quality`.

| Folder (was) | Test | Runner | Rows |
|---|---|---|---|
| [`atlas/pilot/`](atlas/pilot/README.md) (`repo_atlas/`) | Stage 1 pilot: where does the change go? 8 SWE-bench tasks | `repo_atlas_eval.py` | `runs.jsonl`, `selection.json`, `contexts.json`, `index/` |
| [`atlas/qa/`](atlas/qa/README.md) (`repo_atlas_qa/`) | SWE-QA: questions about a repository | `repo_atlas_qa.py` | `answers.jsonl`, `judged.jsonl`, `calls.jsonl`, `summary.{md,json}` |
| [`atlas/sec/`](atlas/sec/README.md) (`repo_atlas_sec/`) | Finding a security flaw, 12 advisories | `repo_atlas_sec.py` | `runs.jsonl`, `selection.json`, `summary.json` |
| [`atlas/swe/`](atlas/swe/README.md) (`repo_atlas_swe/`) | Hard coding tasks, 16 SWE-rebench tasks, one run each | `repo_atlas_swe.py` | `funnel.jsonl`, `validate.jsonl`, `runs.jsonl`, `summary.md` |
| [`atlas/swe_rerun/`](atlas/swe_rerun/README.md) (`repo_atlas_swe_rerun/`) | Q1: the same tasks, two clean runs per arm, auto-memory off | `repo_atlas_swe_rerun.py` | `runs.jsonl`, `summary.md` |
| [`atlas/swe_sonnet/`](atlas/swe_sonnet/README.md) (`repo_atlas_swe_sonnet/`) | Q3: the same tasks on Sonnet | `repo_atlas_swe_sonnet.py` | `gate.jsonl`, `runs.jsonl`, `summary.md` |
| [`atlas/migrate/`](atlas/migrate/README.md) (`repo_atlas_migrate/`) | C-to-Rust migration, 14 CRUST-Bench projects | `repo_atlas_migrate.py` | `runs.jsonl`, `validate*.jsonl`, `summary.md` |
| [`atlas/port/`](atlas/port/README.md) (`repo_atlas_port/`) | Q2: a larger port, gitleaks Go to Rust | `repo_atlas_port.py` | `runs.jsonl`, `grades.jsonl`, `ports/`, `transcripts/`, `summary.md` |
| [`atlas/plan_quality/`](atlas/plan_quality/README.md) (`plan_quality/`) | Briefs written from the atlas, right-sized workers | `plan_quality.py` | `briefs.jsonl`, `judge.jsonl`, `work.jsonl`, `summary.md` |

Every folder also has `spend.jsonl` where the runner logged cost.

## What each folder here holds

| Folder | Contents |
|---|---|
| `phase1/` | Claude sweeps: `config.json`, `runs.jsonl`, `summary.{md,json}`, CER-curve plots; page images git-ignored |
| `phase1b/` | PP-OCR rows and their analysis summaries |
| `phase2/` | CRNN grid: rows, `args.json`, training curves and logs (force-added once a grid finished), `analysis/`; `diag_*` and `smoke_*` are diagnostics |
| `claude_checks/` | `scripts/check_claude_vision.py` output: token formula, legibility, effort, determinism, hidden calls; sample images in `img/` |
| `samples/` | Example renders of the substitution, quadrant and Phase-2 schemes |
| `claude_qa/`, `rag/`, `erb/`, `compaction/`, `compaction_scale/`, `readhook_smoke/`, `readhook_swe/`, `readhook_swe_ml/` | The Claude-reader tests listed above |
| `atlas/` | The nine atlas tests listed above |
| `local_qa/` | Output folder of `experiments/claude_qa/local_qa.py` (local Qwen test; not run yet, nothing tracked) |
| `r12_logs/` | Logs of the PLAN_3090 Runs 1-2 driver (`frozen.log`, `smoke.log`) |
| `queue_logs/<name>/` | Per-job logs from `scripts/job_queue.py` (git-ignored; the GPU rounds' logs are force-added) |
| `archive/` | Finished rounds' job-queue inputs (`queue_*.txt`) and top-level logs (`pages_*.log`, `smoke_*.log`, `phase2_driver.log`, ...) |
| `archive/logs/` | Wrapper scripts and logs that used to sit at the top of `results/`: `run_a1cpu.sh`, `run_s1*.sh`, `s1*.log` (colour gist S1), `stage0*.log` (own encoder Stage 0) |
| `cache/`, `memlog/` | Claude response cache, memory logs: git-ignored |

## Note on archived paths

Older docs (`encoder_probe/ROUND2.md`, `encoder_probe/ROUND3.md`, `docs/background/history/`) name files such
as `results/queue_probe_c.txt` or `results/queue_gpu3h.txt`. Those finished queue files and logs now live in
`results/archive/` under the same names. New queue files can still be written to `results/` while a round runs;
move them to `archive/` when it finishes.
