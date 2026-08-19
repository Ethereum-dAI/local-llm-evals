#!/usr/bin/env bash
# THE DELIVERABLE RUN: v5 (alpha chosen on dev) vs base on the FROZEN 1000-case set.
#
# Both arms on rented GPUs so the comparison is device-controlled, both Q4_K_M, both at
# the sampling of the runs that produced base 90.7% and ft-v4 78.7%. The frozen set is
# scored ONCE here — alpha was already selected on pf/tests.dev.yaml, and re-running this
# with a second candidate would make the test set a hyperparameter.
#
# Not under `set -e`: a failed arm must not skip the pod teardown in the trap.
set -u
cd "$(dirname "$0")/.."

V5_REPO="${V5_REPO:-ef-dai-team/gemma-4-E4B-wallet-ft-v5}"
V5_GGUF="${V5_GGUF:-gemma-4-E4B-wallet-ft-a075.Q4_K_M.gguf}"
V5_SHA="${V5_SHA:-40332b62f282336d92a94dc4147ecc44e83c0e11496ac2e5738d8ef342d1b09c}"
# The EXACT GGUF the wallet ships, at the pinned revision: Q4_K_M was deleted from the
# repo's main branch and exists only at this commit.
BASE_REPO="${BASE_REPO:-ggml-org/gemma-4-E4B-it-GGUF}"
BASE_GGUF="${BASE_GGUF:-gemma-4-E4B-it-Q4_K_M.gguf}"
BASE_REV="${BASE_REV:-1762c8e8713f}"
DATASET="${DATASET:-pf/tests.combined.yaml}"
JOBS="${JOBS:-12}"
PODS=""

cleanup() {
  for pid in $PODS; do
    echo "[run] terminating pod $pid"
    uv run --with runpod python scripts/runpod_serve_gguf.py down --pod-id "$pid"
  done
}
trap cleanup EXIT INT TERM

launch() {  # name repo revision gguf sha extra…
  local name="$1" repo="$2" rev="$3" gguf="$4" sha="$5"; shift 5
  echo "[run] launching $name"
  local out
  out=$(uv run --with runpod python scripts/runpod_serve_gguf.py up \
        --repo "$repo" --revision "$rev" --gguf "$gguf" --expect-sha "$sha" \
        --n-ctx 4096 --parallel 8 --disk 40 --wait 2400 "$@" 2>&1 | tee /dev/stderr)
  LAST_POD=$(echo "$out" | sed -n 's/^RUNPOD_POD_ID=//p' | tail -1)
  LAST_URL=$(echo "$out" | sed -n 's/^RUNPOD_LLAMA_URL=//p' | tail -1)
  [ -n "$LAST_POD" ] && PODS="$PODS $LAST_POD"
}

launch v5 "$V5_REPO" main "$V5_GGUF" "$V5_SHA" --private
V5_URL="$LAST_URL"
launch base "$BASE_REPO" "$BASE_REV" "$BASE_GGUF" ""
BASE_URL="$LAST_URL"

if [ -z "$V5_URL" ] || [ -z "$BASE_URL" ]; then
  echo "[run] an arm failed to come up (v5='$V5_URL' base='$BASE_URL') — refusing to"
  echo "[run] score a one-armed comparison as a head-to-head"
  exit 1
fi

export RUNPOD_V5_URL="$V5_URL" RUNPOD_BASE_URL="$BASE_URL"
echo "[run] scoring $DATASET  v5=$V5_URL  base=$BASE_URL"
EVAL_DATASET="$DATASET" PROMPTFOO_CONFIG_DIR=.promptfoo-v5final scripts/eval.sh \
    -c promptfooconfig.v5-vs-base.remote.yaml -j "$JOBS" --no-cache \
    -o runs/v5-vs-base.out.json

echo "[run] ---- report ----"
uv run python scripts/report_1000.py runs/v5-vs-base.out.json runs/gpt5-1000.out.json \
    --label gpt-5 --json runs/v5-final-report.json
