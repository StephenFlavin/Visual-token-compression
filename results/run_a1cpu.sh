#!/bin/bash
# Colour-gist A1-cpu (docs/COLOUR_GIST.md, amendment 2). Rerunnable: finished steps are skipped, the fine-tune
# resumes from its last checkpoint.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
E=encoder_probe/experiments
if [ ! -s encoder_probe/results/z0_info_w400/runs.jsonl ]; then
  rm -rf encoder_probe/results/z0_info_w400
  python3 $E/zeroshot_info.py --device cpu --pitches 16 --streams 1 2 --pages 24 --max-target-chars 400 --run-name z0_info_w400
fi
python3 $E/finetune_qwen.py --encoding rgb2s_p16 --device cpu --bf16 on --steps 500 --batch-size 2 --vision-blocks 0 \
  --no-train-merger --max-target-chars 400 --train-pages 1000 --eval-windows 2 --eval-mc 4 --eval-cloze 4 \
  --val-samples 32 --eval-every 250 --ckpt-every 50 --eval-conds clean --threads 4 --run-name colour_a1cpu
if [ ! -s encoder_probe/results/a1cpu_info/runs.jsonl ]; then
  rm -rf encoder_probe/results/a1cpu_info
  python3 $E/zeroshot_info.py --device cpu --pitches 16 --streams 2 --pages 24 --max-target-chars 400 \
    --adapter encoder_probe/cache/r3_adapters/colour_a1cpu__rgb2s_p16__500.pt \
    --blind-ref encoder_probe/results/z0_info_w400/runs.jsonl --run-name a1cpu_info
fi
echo A1CPU_DONE
