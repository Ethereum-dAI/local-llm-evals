#!/usr/bin/env bash
# Fine-tune gate for the winning safety clause. NOT under `set -e`: promptfoo exits
# non-zero whenever any case fails, which is the normal outcome of an eval.
set -uo pipefail
cd "$(dirname "$0")/.."
export PROMPTFOO_CONFIG_DIR=.promptfoo-ftsafety
for ds in pf/tests.dev.yaml pf/tests.dev.safety.yaml; do
  tag=$(basename "$ds" .yaml | sed 's/tests\.//')
  echo "== ft $tag: starting $(date -u +%H:%M:%SZ)"
  EVAL_DATASET="$ds" scripts/eval.sh -c promptfooconfig.safety-ab-ft.yaml \
      -j 1 --no-cache -o "runs/ftsafety-$tag.out.json"
  echo "== ft $tag: done rc=$? $(date -u +%H:%M:%SZ) -> runs/ftsafety-$tag.out.json"
done
echo "FT-SAFETY-ALL-DONE"
