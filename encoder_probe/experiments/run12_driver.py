"""Resumable driver for docs/HANDOFF.md "Prompt for the 3090" steps 3-6 (PLAN_3090 Runs 1-2, amendments 3-6).

Runs the pre-registered commands one at a time, in order, and records each finished step in
results/r12_logs/done.txt, so a restart (the PC is on only in 6-7 h stretches) skips finished steps;
finetune_qwen.py / channel_capacity.py resume from their --ckpt-every checkpoints and capacity_curve.py per page.
Mechanics only: no flag differs from the prompt except amendment 7's --allow-fewer. Stops (status file STOP) on any failed command or failed sanity check.

    python encoder_probe/experiments/run12_driver.py --steps S [--micro-batch 4] --stages A chan B C
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "results" / "r12_logs"
DONE = LOGS / "done.txt"
RES = "encoder_probe/results"
ADP = "encoder_probe/cache/r3_adapters"
PY = sys.executable
R = ("--lora-r 0 --vision-blocks 16 --lr-vision 2e-5 --task-mix 1.0 --text shuffled --top-margin 7 "
     "--train-pages 12000 --test-pages 16 --batch-size 8 --max-target-chars 2000 --eval-conds clean --device cuda").split()
STAGES = {"A": ["latin_p8", "lg_n0_pb_c4_h7", "lg_n0_pb_c7_h7"], "B": ["latin_p6", "latin_p5"],
          "C": ["lg_n0_pb_c2_h7md", "lg_pb_c2t4_h7md"]}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGS / "driver.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done() -> set[str]:
    return set(DONE.read_text().split()) if DONE.exists() else set()


def stop(msg: str) -> None:
    (LOGS / "STOP").write_text(msg + "\n")
    log("STOP: " + msg)
    sys.exit(1)


def run(key: str, cmd: list[str]) -> None:
    if key in done():
        return
    import shutil
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < 10:   # a resume checkpoint is 4.0 GB and is written next to the old one; adapters are ~1.4 GB
        stop(f"only {free:.1f} GB disk free before {key}")
    log(f"start {key} ({free:.0f} GB free): {' '.join(cmd)}")
    with open(LOGS / f"{key}.log", "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        stop(f"{key} exited {rc} (see results/r12_logs/{key}.log)")
    with open(DONE, "a") as f:
        f.write(key + "\n")
    log(f"done {key}")


def git_push(msg: str) -> None:
    dirs = [f"{RES}/{p.name}" for p in (ROOT / RES).glob("r12*")] + [f"{RES}/analysis_run12"]
    cmds = [["git", "add", "-A"] + dirs,
            ["git", "add", "-f", "results/r12_logs"],
            ["git", "commit", "-q", "-m", msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
            ["git", "pull", "-q", "--rebase", "origin", "claude/validation-harness"],
            ["git", "push", "-q", "origin", "HEAD:claude/validation-harness"]]
    for c in cmds:
        r = subprocess.run(c, cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0 and c[1] != "commit":
            log(f"git warning ({' '.join(c[:3])}): {r.stdout.strip()} {r.stderr.strip()}")


def analyse() -> dict:
    args = [PY, "encoder_probe/experiments/analyse_run12.py", "--frozen", f"{RES}/r12_frozen",
            "--trained", f"{RES}/r12_cap_trained", "--blind", f"{RES}/r12_cap_blind"]
    for flag, d in (("--sweep", "r12_sweep"), ("--chan", "r12_chan")):
        if (ROOT / RES / d).exists():
            args += [flag, f"{RES}/{d}"]
    with open(LOGS / "analyse.log", "a", encoding="utf-8") as f:
        rc = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        stop(f"analyse_run12 exited {rc}")
    return json.loads((ROOT / RES / "analysis_run12" / "summary.json").read_text())


def sanity(res: dict, enc: str) -> None:
    """Amendment 3: blind bits/token on shuffled text <= 1.0; trained not below frozen by more than its CI."""
    for p in res["points"]:
        if p["encoding"] == enc and p["text"] == "shuffled" and "blind" in p and p["blind"][0] > 1.0:
            stop(f"sanity: {enc} blind shuffled bits/token {p['blind'][0]:.2f} > 1.0")
    for r in res["run1"]:
        if r["encoding"] == enc and r["trained_minus_frozen"][2] < 0:
            stop(f"sanity: {enc}/{r['text']} trained below frozen beyond CI {r['trained_minus_frozen']}")
    log(f"sanity OK for {enc}")


def arm(enc: str, S: int, extra: list[str]) -> None:
    ft = [PY, "encoder_probe/experiments/finetune_qwen.py", "--encoding", enc] + R + extra + ["--steps", str(S)]
    run(f"ft__{enc}", ft + ["--run-name", "r12"])
    run(f"ftblind__{enc}", ft + ["--run-name", "r12", "--blind-images"])
    # both training runs done: the 12000-page train cache (~1.2 GB) is not needed again (capacity curves use test)
    for f in (ROOT / "encoder_probe" / "cache" / "pages").glob(f"ps_{enc}_shuffled_train_n12000_*.pkl.gz"):
        f.unlink()
        log(f"deleted train page cache {f.name}")
    cap = [PY, "encoder_probe/experiments/capacity_curve.py", "--device", "cuda", "--encodings", enc,
           "--texts", "natural", "shuffled"]
    main = ["--pages", "200", "--allow-fewer", "--shifts", "7", "--cer-pages", "4"]   # amendment 7
    run(f"captrained__{enc}", cap + main + ["--adapter", f"{ADP}/r12__{enc}__{S}.pt", "--run-name", "r12_cap_trained"])
    run(f"capblind__{enc}", cap + main + ["--adapter", f"{ADP}/r12_blind__{enc}__{S}.pt",
                                          "--run-name", "r12_cap_blind"])
    run(f"sweep__{enc}", cap + ["--pages", "64", "--shifts", "7", "14", "21", "28", "--shift-dx", "7",
                                "--shift-margin", "28", "--arms", "real", "blank", "--cer-pages", "0",
                                "--adapter", f"{ADP}/r12__{enc}__{S}.pt", "--run-name", "r12_sweep"])
    if f"pushed__{enc}" in done():
        return
    res = analyse()
    sanity(res, enc)
    git_push(f"Run 1-2 arm {enc}: trained, blind, capacity curves, dy sweep, analysis")
    with open(DONE, "a") as f:
        f.write(f"pushed__{enc}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--stages", nargs="+", default=["A", "chan", "B", "C"])
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    if (LOGS / "STOP").exists():
        log("STOP file present; remove it to continue")
        return 1
    extra = ["--micro-batch", str(a.micro_batch)] if a.micro_batch else []
    for st in a.stages:
        if st == "chan":
            # step 4b smoke: loss must fall below ln 8 = 2.079 nats/digit; peak VRAM < ~21 GB in nvidia-smi
            # (~3.3 GB above torch's peak allocated in the Run 1 smoke, so allocated < 17.5 GB)
            run("chansmoke", [PY, "encoder_probe/experiments/channel_capacity.py", "--cell", "4", "--smoke",
                              "--device", "cuda"] + extra)
            if "chansmoke_ok" not in done():
                import re
                emas = [float(x) for x in re.findall(r"ema ([\d.]+)", (LOGS / "chansmoke.log").read_text())]
                rows = [json.loads(x) for x in (ROOT / RES / "r12_chan_smoke" / "runs.jsonl").read_text().splitlines()
                        if x.strip()]
                peak = rows[-1].get("peak_gpu_bytes") or 0
                log(f"chan smoke: final ema {emas[-1] if emas else None} nats/digit, peak allocated {peak / 1e9:.1f} GB, "
                    f"{rows[-1].get('s_per_step')} s/step")
                if not emas or emas[-1] >= 2.079:
                    stop("chan smoke: loss did not fall below 2.079 nats/digit")
                if peak > 17.5e9:
                    stop("chan smoke: peak VRAM too high; rerun the chan stage with --micro-batch 4")
                with open(DONE, "a") as f:
                    f.write("chansmoke_ok\n")
            for c in (7, 4, 2):
                run(f"chan__{c}", [PY, "encoder_probe/experiments/channel_capacity.py", "--cell", str(c),
                                   "--steps", str(a.steps), "--device", "cuda", "--run-name", "r12_chan"] + extra)
            if "pushed__chan" not in done():
                res = analyse()
                for row in res["chan"]:
                    hi = row["trained"]["swap_recovered_bits_per_token"][2]
                    if hi > 0.5:
                        stop(f"sanity: chan cell {row['cell']} swapped upper CI {hi:.2f} > 0.5")
                log("sanity OK for chan (every swapped upper CI <= 0.5)")
                git_push("Run 1-2 channel capacity (amendment 6): cells 7/4/2, analysis")
                with open(DONE, "a") as f:
                    f.write("pushed__chan\n")
            continue
        if st == "C":
            res = analyse()
            if not (res.get("gate_A1") == "PASS" and res.get("gate_A2") == "PASS"):
                log(f"stage C skipped: A1 {res.get('gate_A1')}, A2 {res.get('gate_A2')} (amendment 4)")
                continue
        for enc in STAGES[st]:
            arm(enc, a.steps, extra)
        if st == "A":
            res = analyse()
            log(f"stage A gates: A1 {res.get('gate_A1')} ({res.get('gate_A1_value')}), "
                f"A2 {res.get('gate_A2')} ({res.get('gate_A2_ratio')})")
    log("all requested stages done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
