#!/bin/bash
# Kill programs started from the Rust build dirs that run too long (the built tool itself: 3 minutes; test binaries in
# target/*/deps: 15 minutes): an orphaned `target/debug/bat` held a `cargo test | grep` pipe open and hung agent runs in both arms.
# Same rule for every arm; each kill is logged.
ROOT=/tmp/claude-0/-home-user-visual-token-compaction/9e274458-7352-5bb1-b191-52e96364b3f6/scratchpad/swe/ml
LOG=$ROOT/watchdog.log
while true; do
  for pid in $(pgrep -f "$ROOT/[^ ]*/target/"); do
    exe=$(readlink /proc/$pid/exe 2>/dev/null) || continue
    case "$exe" in "$ROOT"/*/target/*) ;; *) continue ;; esac
    age=$(ps -o etimes= -p $pid 2>/dev/null | tr -d ' ')
    case "$exe" in */target/*/deps/*) limit=900 ;; *) limit=180 ;; esac   # test binaries vs the built tool itself
    if [ -n "$age" ] && [ "$age" -gt "$limit" ]; then
      echo "$(date -u +%FT%TZ) kill $pid ($age s) $exe" >> $LOG
      kill $pid 2>/dev/null; sleep 2; kill -9 $pid 2>/dev/null
    fi
  done
  sleep 60
done
