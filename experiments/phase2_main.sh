#!/usr/bin/env bash
# Phase-2 main grid: per-em CRNN, 3500 steps, ems concentrated around each scheme's expected 3% crossing.
set -u
cd "$(dirname "$0")/.."
R="--steps 3500 --parallel 3 --threads 1 --run-name phase2_main"
python3 experiments/phase2_train.py --schemes latin --ems 4 5 6 7 8 $R
python3 experiments/phase2_train.py --schemes latin_condensed subst300 subst1000 --ems 5 6 7 8 10 $R
python3 experiments/phase2_train.py --schemes quad3 quad4 --ems 8 10 12 $R
