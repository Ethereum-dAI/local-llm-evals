#!/usr/bin/env bash
# Run the four-way gemma4 comparison ONE PROVIDER PER INVOCATION, then merge.
#
# Why not a single four-provider run: promptfoo's -j controls CONCURRENCY, not how
# many models stay RESIDENT. With four local GGUF providers in one config it
# interleaves them per test case, so all four ~5 GB models end up loaded at once.
# On a 36 GB host that produced `RuntimeError: llama_decode returned -3` on
# 466/480, 466/480 and 468/480 cases for the three fine-tunes (the base provider,
# which loaded first, was clean at 0). Same per-runtime memory blowup CLAUDE.md
# documents for the local-llm test suite.
#
# One provider per process = one model resident. Slower in wall clock, but it
# actually completes.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=promptfooconfig.gemma4-fourway.yaml
declare -a LABELS=(
  gemma4-e4b-base
  gemma4-e4b-ft-old-shipped
  gemma4-e4b-ft-v1-appcontract
  gemma4-e4b-ft-v2-appcontract
)
declare -a OUTS=(
  gemma4.base.out.json
  gemma4.ftold.out.json
  gemma4.ftv1.out.json
  gemma4.ftv2.out.json
)

for i in "${!LABELS[@]}"; do
  L="${LABELS[$i]}"; O="${OUTS[$i]}"
  # Resumable: a provider that already produced a non-empty export is skipped, so
  # an interrupted run does not repay 20+ GPU-minutes per completed provider.
  if [ -s "$O" ]; then
    echo "=== [$((i+1))/4] $L SKIPPED — $O already exists ($(stat -f %z "$O") bytes) ==="
    continue
  fi
  echo "=== [$((i+1))/4] $L -> $O  ($(date +%H:%M:%S)) ==="
  # `|| true`: promptfoo exits NON-ZERO whenever any assertion fails, which is the
  # normal case for an eval — under `set -e` that killed the loop after the first
  # provider even though it had completed 560/560 cleanly. The real success test is
  # whether the export was written, checked immediately below.
  # Anchored regex: the v1/v2 labels are prefixes of each other's shape.
  scripts/eval.sh -c "$CONFIG" --filter-providers "^${L}\$" -o "$O" -j 1 || true
  if [ ! -s "$O" ]; then
    echo "=== [$((i+1))/4] $L FAILED — no output written; stopping before the merge ==="
    exit 1
  fi
  echo "=== [$((i+1))/4] $L DONE ($(date +%H:%M:%S)) ==="
done

echo "=== merging ==="
uv run python scripts/merge_evals.py "${OUTS[@]}" -o gemma4.fourway.out.json
echo "=== ALL DONE ==="
