#!/bin/bash
# Poll a detached Modal app's logs into a file until a terminal marker appears.
#
# `modal app logs` EXITS once it catches up rather than following, so a single
# invocation returns a snapshot and looks like a finished job. This re-invokes it
# on an interval and rewrites the file, then appends POLLER-DONE so a watcher can
# tell "finished" from "still going".
#
#   scripts/poll_modal_app.sh <app-id> <logfile> [deadline-seconds]
set -uo pipefail
APP="${1:?app id}"
LOG="${2:?log path}"
DEADLINE_S="${3:-7800}"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
deadline=$(( $(date +%s) + DEADLINE_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  uv run --with modal modal app logs "$APP" > "$LOG.new" 2>&1 || true
  if [ -s "$LOG.new" ]; then mv "$LOG.new" "$LOG"; fi
  # Terminal markers: our own summaries, plus the failure signatures. Silence is
  # not success — a crashloop must break the loop too, not just a happy path.
  if grep -aqE "SUMMARY|adapter saved|Traceback|Killed|OOM|exceeded|Terminating" "$LOG"; then
    sleep 25
    uv run --with modal modal app logs "$APP" > "$LOG" 2>&1 || true
    break
  fi
  sleep 30
done
echo "POLLER-DONE" >> "$LOG"
