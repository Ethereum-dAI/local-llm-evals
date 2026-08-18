#!/bin/bash
# Three-arm safety prompt A/B, both slices, sequentially.
# Same reasoning as run_prompt_ab.sh: one Metal device (-j 1), one SQLite DB, and
# NOT under `set -e` because promptfoo exits non-zero whenever a case fails.
cd "$(dirname "$0")/.."
mkdir -p runs
for slice in dev dev.safety; do
  ds="pf/tests.${slice}.yaml"
  out="runs/safety-${slice}.out.json"
  echo "== ${slice}: starting $(date -u +%H:%M:%SZ)"
  if timeout 5h env PROMPTFOO_CONFIG_DIR=.promptfoo-safety EVAL_DATASET="$ds" \
      scripts/eval.sh -c promptfooconfig.safety-ab.yaml -j 1 --no-cache -o "$out" \
      > "runs/safety-${slice}.log" 2>&1; then rc=0; else rc=$?; fi
  echo "== ${slice}: done rc=$rc $(date -u +%H:%M:%SZ) -> $out"
done
echo "SAFETY-AB-ALL-DONE"
