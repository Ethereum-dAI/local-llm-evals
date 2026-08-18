#!/bin/bash
# Both slices of the base-model prompt A/B, sequentially.
#
# SEQUENTIAL on purpose, twice over: two llama.cpp processes would contend for the
# one Metal device (hence -j 1), and two concurrent promptfoo runs race for the same
# SQLite result DB (SQLITE_BUSY, which can still write a truncated -o export).
#
# NOT wrapped in `set -e`: promptfoo exits non-zero whenever any test case FAILS,
# which is a normal outcome here. Under `set -e` that aborts the script after the
# first slice while still looking like a clean finish — the regression that cost a
# deadline earlier in this work.
cd "$(dirname "$0")/.."
mkdir -p runs
for slice in dev dev.safety; do
  ds="pf/tests.${slice}.yaml"
  out="runs/ab-${slice}.out.json"
  echo "== ${slice}: starting $(date -u +%H:%M:%SZ) ($(uv run python -c "
import yaml;print(len(yaml.safe_load(open('${ds}').read())))") cases x 2 arms)"
  if timeout 4h env PROMPTFOO_CONFIG_DIR=.promptfoo-ab EVAL_DATASET="$ds" \
      scripts/eval.sh -c promptfooconfig.prompt-ab.yaml -j 1 --no-cache -o "$out" \
      > "runs/ab-${slice}.log" 2>&1; then rc=0; else rc=$?; fi
  echo "== ${slice}: done rc=$rc $(date -u +%H:%M:%SZ) -> $out"
done
echo "AB-ALL-DONE"
