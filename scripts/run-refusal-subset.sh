#!/usr/bin/env bash
# Run the 49-case refusal slice for a set of providers, ONE MODEL PER PROCESS.
#
# Same constraint as run-fourway-sequential.sh: promptfoo's -j caps concurrency,
# not resident models, and four local GGUFs at once produced
# `RuntimeError: llama_decode returned -3` on 466/480 cases. One provider per
# invocation keeps exactly one ~5 GB model in memory.
#
# 49 cases instead of 569 makes this a ~2 min/model loop rather than ~25.
#
#   scripts/run-refusal-subset.sh gemma4-e4b-ft-v2-appcontract gemma4-e4b-ft-v3-refusals
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG=promptfooconfig.gemma4-fourway.yaml
for L in "$@"; do
  O="refusals.${L}.out.json"
  if [ -s "$O" ]; then echo "=== $L SKIPPED — $O exists ==="; continue; fi
  echo "=== $L -> $O ($(date +%H:%M:%S)) ==="
  # `|| true`: promptfoo exits non-zero whenever any assertion fails, which is
  # normal for an eval; the real success test is whether the export was written.
  EVAL_DATASET=pf/tests.refusals.yaml \
    scripts/eval.sh -c "$CONFIG" --filter-providers "^${L}\$" -o "$O" -j 1 || true
  if [ ! -s "$O" ]; then echo "=== $L FAILED — no output ==="; exit 1; fi
  echo "=== $L DONE ($(date +%H:%M:%S)) ==="
done
