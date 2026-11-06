#!/bin/bash
# Colour gist S1 (docs/COLOUR_GIST.md): small VLM, encoder from scratch, CPU. Rerunnable: finished runs are
# skipped, the current one resumes from its checkpoint. Stops after the first arm if the learnability gate fails.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
R="python3 encoder_probe/experiments/small_vlm.py --steps 1500 --device cpu --run-name s1"
$R --encoding rgb1s_p16
python3 - <<'PY'
import json, sys
r = [json.loads(l) for l in open("encoder_probe/results/s1/runs.jsonl") if '"rgb1s_p16"' in l][-1]
gap = r["val_swap_loss"] - r["val_loss"]
print(f"S1 learnability: rgb1s_p16 val {r['val_loss']:.3f}, swapped images {r['val_swap_loss']:.3f}, gap {gap:.3f}")
sys.exit(0 if gap >= 0.3 else 3)
PY
$R --encoding rgb2s_p16
python3 encoder_probe/experiments/analyse_colour.py --real s1 --pitch 16 --swap-blind --equal-density rgb2s_p16 rgb1s_p12 || true
$R --encoding rgb1s_p16 --blind-images
$R --encoding rgb2s_p16 --blind-images
python3 encoder_probe/experiments/analyse_colour.py --real s1 --pitch 16 --equal-density rgb2s_p16 rgb1s_p12 || true
$R --encoding rgb1s_p12
$R --encoding rgb1s_p12 --blind-images
python3 encoder_probe/experiments/analyse_colour.py --real s1 --pitch 16 --equal-density rgb2s_p16 rgb1s_p12
echo S1_DONE
