#!/usr/bin/env bash
# ACT_NOT_ASK A/B, both dev slices. NOT under `set -e`: promptfoo exits non-zero
# whenever any case fails, which is the normal outcome of an eval measuring failures.
set -uo pipefail
cd "$(dirname "$0")/.."
export PROMPTFOO_CONFIG_DIR="${PROMPTFOO_CONFIG_DIR:-.promptfoo-act}"
JOBS="${JOBS:-1}"
CONFIG="${CONFIG:-promptfooconfig.act-ab.yaml}"
TAGPREFIX="${TAGPREFIX:-act}"
for ds in pf/tests.dev.yaml pf/tests.dev.safety.yaml; do
  tag=$(basename "$ds" .yaml | sed 's/tests\.//')
  echo "== $TAGPREFIX $tag: starting $(date -u +%H:%M:%SZ)"
  EVAL_DATASET="$ds" scripts/eval.sh -c "$CONFIG" -j "$JOBS" --no-cache \
      -o "runs/$TAGPREFIX-$tag.out.json"
  echo "== $TAGPREFIX $tag: done rc=$? $(date -u +%H:%M:%SZ) -> runs/$TAGPREFIX-$tag.out.json"
done
echo "$TAGPREFIX-AB-ALL-DONE"
