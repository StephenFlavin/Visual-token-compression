# AGENTS.md: how to work in this repo (visual-text-compression)

**State (2026-11-12): a finished study, wound down.** The story and results are in [README.md](README.md). One run
is left, on the user's local GPU.

## The one thing left to run

**[docs/HANDOFF.md](docs/HANDOFF.md) starts with the prompt to paste into a session on the RTX 3090 PC.** It runs
the Claude page-image test in small on a local Qwen3.8-27B (plan and predictions:
[docs/gpu/LOCAL_QWEN.md](docs/gpu/LOCAL_QWEN.md); runbook: [docs/gpu/LOCAL_GPU.md](docs/gpu/LOCAL_GPU.md)).
Everything else is parked or closed; do not restart parked runs unless the user asks.

## Where things are

| Need | Go to |
|---|---|
| What was found, in story order | [README.md](README.md) |
| Every finding with its numbers, in date order | [docs/FINDINGS.md](docs/FINDINGS.md) |
| Index of all write-ups | [docs/README.md](docs/README.md) |
| The command for any experiment | [docs/RUNNING.md](docs/RUNNING.md) |
| Things that go wrong (network, headless Claude, OCR, training, cloud runner, git) | [docs/GOTCHAS.md](docs/GOTCHAS.md): read the section for the area you touch |
| Where result rows live | [results/README.md](results/README.md) |

## Setup (cloud CPU runner: 4 CPU, ~15.7 GB RAM, no GPU)

```bash
pip install -r requirements.txt                       # if the torch line fails, use the next line
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install rapidocr-onnxruntime
scripts/fetch_fonts.sh
python3 scripts/build_data.py                         # WikiText-103 passages and dictionaries (~5 min)
OMP_NUM_THREADS=1 python3 -m pytest -q tests encoder_probe/tests learned_glyphs/tests
python3 scripts/check_links.py                        # every markdown link and path must resolve
```

Start cloud sessions in the `visual-bpe` environment (it keeps the repo's old name): the Default one cannot
reach Hugging Face.
`data/`, fonts, caches, checkpoints and adapters are git-ignored and regenerate from these commands.

## Rules

1. **Validation runs go to an Opus subagent** (standing instruction from the user, 2026-11-12). Any test that checks
   a claim is delegated with a written brief, a pre-registration committed before the first scored call, and a
   hard spend cap. The main session designs, checks the returned rows itself and updates the docs.
2. **Every trained or probed arm needs its own blind control**, and one run per task settles nothing: three
   separate "3 to 0" hints in this repo vanished or stayed unrepeated.
3. **Never reword or disguise a prompt to get past a refusal or a safety classifier.** Claude is measured on
   natural text only.
4. **Never weaken, skip or edit a test to get a pass.**
5. **Keep the tree clean:** a stop hook requires everything committed and pushed; long jobs snapshot-commit their
   rows (`WIP:`). The container is suspended when idle and can restart, so long jobs must resume from their rows.
6. **When you add a result:** put the rows and a README under `results/`, add one entry to
   [docs/FINDINGS.md](docs/FINDINGS.md), update the table in the write-up it belongs to, and add the command to
   [docs/RUNNING.md](docs/RUNNING.md). New pitfalls go in [docs/GOTCHAS.md](docs/GOTCHAS.md). Keep this file short.
