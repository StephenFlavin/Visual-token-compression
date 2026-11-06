#!/bin/bash
# Colour gist S1 amendment 3 (docs/COLOUR_GIST.md): the s1r recipe (reconstruction loss) plus a LLaVA-style alignment
# stage: steps 1-500 the LLM is frozen and the encoder learns from <= 120-char transcriptions. Alignment gate at step
# 500: val gap vs swapped images >= 0.05 nats/token, else exit 4 (S1 closed as inconclusive).
# Rerunnable: finished runs are skipped, the current one resumes from its checkpoint.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
# jemalloc: glibc malloc fragmentation grew RSS to the 15.7 GB cgroup limit (OOM at step ~235)
JE=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2; [ -f $JE ] && export LD_PRELOAD=$JE
# Steps slow from 6 to ~10.5 s within one process (~470k page faults/s; jemalloc without decay grew to 11.8 GB and
# was OOM-killed): each run restarts in a fresh process every 300 steps (exit 5) and resumes exactly from its checkpoint
PY="python3 encoder_probe/experiments/small_vlm.py --steps 1500 --device cpu --run-name s1a --session-steps 300 --align-steps 500 --align-chars 120"
R() { local rc; while :; do rc=0; $PY "$@" || rc=$?; [ $rc -eq 5 ] || return $rc; done; }
CK=encoder_probe/cache/ckpt/s1__s1a__rgb1s_p16__1500.pt
if ! grep -qs '"rgb1s_p16"' encoder_probe/results/s1a/runs.jsonl; then
  R --encoding rgb1s_p16 --stop-after 500
  python3 - "$CK" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["curve"][-1]
print(f"S1a alignment gate (step {c[0]}): val {c[2]:.3f}, swapped {c[3]:.3f}, gap {c[3] - c[2]:.3f}, "
      f"image spread {c[4]:.3f}", flush=True)
sys.exit(0 if c[3] - c[2] >= 0.05 else 4)
PY
fi
R --encoding rgb1s_p16
python3 - <<'PY'
import json, sys
r = [json.loads(l) for l in open("encoder_probe/results/s1a/runs.jsonl") if '"rgb1s_p16"' in l][-1]
sys.path.insert(0, "encoder_probe/experiments")
import analyse_colour as AC
gap = r["val_swap_loss"] - r["val_loss"]
a = AC.gist_value(r)[1][0][0]
print(f"S1 learnability: rgb1s_p16 val {r['val_loss']:.3f}, swapped images {r['val_swap_loss']:.3f}, gap {gap:.3f}; "
      f"gist AUC {a:.3f}")
sys.exit(0 if gap >= 0.3 or a >= 0.65 else 3)
PY
R --encoding rgb2s_p16
python3 encoder_probe/experiments/analyse_colour.py --real s1a --pitch 16 --swap-blind --equal-density rgb2s_p16 rgb1s_p12 || true
R --encoding rgb1s_p16 --blind-images
R --encoding rgb2s_p16 --blind-images
python3 encoder_probe/experiments/analyse_colour.py --real s1a --pitch 16 --equal-density rgb2s_p16 rgb1s_p12 || true
R --encoding rgb1s_p12
R --encoding rgb1s_p12 --blind-images
python3 encoder_probe/experiments/analyse_colour.py --real s1a --pitch 16 --equal-density rgb2s_p16 rgb1s_p12
echo S1_DONE
