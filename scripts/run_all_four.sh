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
# Usage: scripts/run_all_four.sh            (waits for any in-flight run first)
set -uo pipefail
cd "$(dirname "$0")/.."

GEMMA_CFG=promptfooconfig.v4-vs-base.yaml
QWEN_CFG=promptfooconfig.qwen-v4-vs-base.yaml

# Don't stampede a run that is still going.
while pgrep -f "promptfoo eval" >/dev/null 2>&1; do
  echo "[driver] waiting for the in-flight eval to finish ($(date -u +%H:%M:%SZ))"
  sleep 60
done

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

run gemma4-e4b-ft-v4-appprompt "$GEMMA_CFG"
run qwen3-8b-base              "$QWEN_CFG"
run qwen3-8b-ft-v4             "$QWEN_CFG"

echo "[driver] ALL RUNS ATTEMPTED $(date -u +%FT%TZ)"
for l in gemma4-e4b-base gemma4-e4b-ft-v4-appprompt qwen3-8b-base qwen3-8b-ft-v4; do
  printf "  %-32s %s chunk export(s)\n" "$l" \
    "$(ls runs/$l.part*.out.json 2>/dev/null | wc -l | tr -d ' ')"
done
