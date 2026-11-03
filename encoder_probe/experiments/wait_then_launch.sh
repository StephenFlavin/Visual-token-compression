#!/usr/bin/env bash
# Wait until no process matches a pgrep -f pattern, then start a command in the background (repo root).
#
#   nohup bash encoder_probe/experiments/wait_then_launch.sh "^python3 scripts/job_queue.py results/queue_probe_b" \
#       results/queue_probe_c.log python3 scripts/job_queue.py results/queue_probe_c.txt \
#       --log-dir results/queue_logs/probe_c > results/queue_probe_c_waiter.log 2>&1 &
#
# Anchor the pattern with ^ so it cannot match this waiter (its command line starts with "bash").
# WAIT_POLL (seconds, default 120) sets the polling interval.
set -u
pat="$1"; log="$2"; shift 2
cd "$(dirname "$0")/../.." || exit 1
case "$pat" in ^*) ;; *) echo "[waiter] pattern must start with ^ (else it can match this script)"; exit 2;; esac
echo "[waiter] $(date -u +%FT%TZ) pid $$ waiting while pgrep -f '$pat' matches: $(pgrep -f "$pat" | tr '\n' ' ')"
while pgrep -f "$pat" >/dev/null; do sleep "${WAIT_POLL:-120}"; done
echo "[waiter] $(date -u +%FT%TZ) no match; launching: $* (log: $log)"
nohup "$@" > "$log" 2>&1 &
echo "[waiter] launched pid $!"
