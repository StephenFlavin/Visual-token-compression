# results/

Rows from every experiment are `runs.jsonl` files (one JSON object per run). Summaries, plots and the conclusions
drawn from them are in [docs/glyphs/RESULTS.md](../docs/glyphs/RESULTS.md), [encoder_probe/RESULTS.md](../encoder_probe/RESULTS.md),
[encoder_probe/ROUND3.md](../encoder_probe/ROUND3.md) and [learned_glyphs/RESULTS.md](../learned_glyphs/RESULTS.md).
Images, render caches, checkpoints and live logs are git-ignored and regenerate from the commands in
[AGENTS.md](../AGENTS.md).

## Where each stage's rows live

| Stage | Rows | Key runs |
|---|---|---|
| Phase 1: Claude reads natural Latin | `results/phase1/<run>/runs.jsonl` | `latin_natural`, `latin_scrambled`, `pilot`; `dryrun*` use a fake reader |
| Phase 1b: PP-OCR zero-shot | `results/phase1b/<run>/runs.jsonl` | `ocr_natural`, `ocr_scrambled`; tables in `summary/`, `natural_only/` |
| Phase 2: CRNNs trained per scheme/em (K3) | `results/phase2/<run>/runs.jsonl` | `phase2_main`, `phase2_long`; **`merged/`** = longest run per (scheme, em), the one to cite |
| Learned glyph alphabet | `learned_glyphs/results/<run>/` | `e1long` (banks), `e3_bridge_long` (bridge), `e2_transfer_long` |
| Frozen-encoder probe, rounds 1-2 | `encoder_probe/results/<run>/runs.jsonl` | `main`, `controls*` (SigLIP); `round2*`, `round2_gpu_tp*` (DeepEncoder) |
| Round 3 (Qwen2.5-VL) | `encoder_probe/results/<run>/runs.jsonl` | `round3_cpu_tp200` (G0), `quick_q1` (Q1), `g2lite*` (G2-lite), `round3_ft_gpu3h{,_blind}` (R3-1) |

## What each folder here holds

| Folder | Contents |
|---|---|
| `phase1/` | Claude sweeps: `config.json`, `runs.jsonl`, `summary.{md,json}`, CER-curve plots; page images git-ignored |
| `phase1b/` | PP-OCR rows and their analysis summaries |
| `phase2/` | CRNN grid: rows, `args.json`, training curves and logs (force-added once a grid finished), `analysis/`; `diag_*` and `smoke_*` are diagnostics |
| `claude_checks/` | `scripts/check_claude_vision.py` output: token formula, legibility, effort, determinism, hidden calls; sample images in `img/` |
| `samples/` | Example renders of the substitution, quadrant and Phase-2 schemes |
| `queue_logs/<name>/` | Per-job logs from `scripts/job_queue.py` (git-ignored; the GPU rounds' logs are force-added) |
| `archive/` | Finished rounds' job-queue inputs (`queue_*.txt`) and top-level logs (`pages_*.log`, `smoke_*.log`, `phase2_driver.log`, ...) |
| `cache/`, `memlog/` | Claude response cache, memory logs: git-ignored |

## Note on archived paths

Older docs (`encoder_probe/ROUND2.md`, `encoder_probe/ROUND3.md`, `docs/background/history/`) name files such
as `results/queue_probe_c.txt` or `results/queue_gpu3h.txt`. Those finished queue files and logs now live in
`results/archive/` under the same names. New queue files can still be written to `results/` while a round runs;
move them to `archive/` when it finishes.
