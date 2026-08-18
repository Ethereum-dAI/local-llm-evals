#!/usr/bin/env bash
# Score the remaining models over the 1000-case benchmark, one after another.
#
# SEQUENTIAL on purpose: every provider here is a local llama.cpp process on one
# Metal device, so two at once contend for the GPU instead of going faster — and
# promptfoo writes each run to a SQLite DB per config dir, which races under
# concurrency (SQLITE_BUSY kills runs mid-flight; see CLAUDE.md).
#
# Resumable at chunk granularity: run_chunked.sh skips any chunk whose export
# already exists, so re-running this script after an interruption picks up where it
# stopped rather than repeating ~2 h of work.
#
# Launch DETACHED, or the harness reaps it: a Bash tool call caps its timeout at
# 10 minutes, and a ~2 h job started inside one is killed partway with no error in
# any log (observed: chunk 3 died at 110/250 having written no export).
#
#   uv run python scripts/launch_detached.py scripts/run_all_four.sh
#
# Usage: scripts/run_all_four.sh            (waits for any in-flight run first)
set -uo pipefail
cd "$(dirname "$0")/.."

GEMMA_CFG=promptfooconfig.v4-vs-base.yaml
QWEN_CFG=promptfooconfig.qwen-v4-vs-base.yaml

# Don't stampede a run that is still going.
# Serialise on a LOCKFILE, not on `pgrep`. Matching a process by command line is
# unreliable in both directions here: any watcher script whose own argv contains
# the pattern text counts as a match (that deadlocked this driver twice — it waited
# forever on a monitor that was waiting on it), and the bracket trick only stops
# pgrep matching ITSELF, not other processes carrying the same literal string.
#
# The lock records a pid, so a stale lock from a killed run is detected and cleared
# rather than blocking forever.
LOCK="runs/.eval.lock"
mkdir -p runs
if [ -f "$LOCK" ]; then
  other=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
    echo "[driver] another run holds $LOCK (pid $other) — waiting"
    while kill -0 "$other" 2>/dev/null; do sleep 30; done
  else
    echo "[driver] clearing stale $LOCK (pid ${other:-none} is gone)"
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

run() {
  local label="$1" cfg="$2"
  echo "=============================================================="
  echo "[driver] $label  via $cfg  start $(date -u +%FT%TZ)"
  echo "=============================================================="
  scripts/run_chunked.sh "$label" "$cfg"
  local n
  n=$(ls runs/"$label".part*.out.json 2>/dev/null | wc -l | tr -d ' ')
  echo "[driver] $label done $(date -u +%FT%TZ) — $n/4 chunk exports"
}

# Which models to run, in order. Defaults to all four; pass labels to run a
# subset, which is how the Gemma pair runs locally while the Qwen pair runs on
# Modal (different hardware, so they do not contend and the wall clock is the
# max of the two rather than the sum).
#
#   scripts/run_all_four.sh gemma4-e4b-base gemma4-e4b-ft-v4-appprompt
#
# run_chunked.sh skips chunks whose export already exists, so re-running a
# partially-finished model resumes it rather than repeating work.
cfg_for() {
  case "$1" in
    qwen*) echo "$QWEN_CFG" ;;
    *)     echo "$GEMMA_CFG" ;;
  esac
}

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(gemma4-e4b-base gemma4-e4b-ft-v4-appprompt qwen3-8b-base qwen3-8b-ft-v4)
fi

for label in "${MODELS[@]}"; do
  run "$label" "$(cfg_for "$label")"
done

echo "[driver] ALL RUNS ATTEMPTED $(date -u +%FT%TZ)"
for l in "${MODELS[@]}"; do
  printf "  %-32s %s chunk export(s)\n" "$l" \
    "$(ls runs/$l.part*.out.json 2>/dev/null | wc -l | tr -d ' ')"
done
