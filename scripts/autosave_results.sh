#!/usr/bin/env bash
# Commit + push experiment result rows (runs.jsonl) every N seconds while long jobs run, so a
# container reclaim loses at most N seconds of finished runs. Usage: nohup scripts/autosave_results.sh 600 &
set -u
cd "$(dirname "$0")/.."
INTERVAL=${1:-600}
while true; do
  sleep "$INTERVAL"
  if [ -n "$(git status --porcelain -- 'results/**/runs.jsonl' results/*/*/runs.jsonl)" ]; then
    git add results/*/*/runs.jsonl
    git commit -q -m "Autosave: experiment result rows

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WVzSJ934fYhDkwG8gwoyMQ" -- results/*/*/runs.jsonl && \
    for d in 2 4 8 16; do git push -q origin HEAD 2>/dev/null && break; sleep $d; done
  fi
done
