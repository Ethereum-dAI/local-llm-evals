#!/usr/bin/env bash
# v5 with and without SAFETY_FULL, both arms on ONE pod so the weights are identical.
# Not under `set -e`: a failed eval must not skip the teardown in the trap.
set -u
cd "$(dirname "$0")/.."

REPO="${REPO:-ef-dai-team/gemma-4-E4B-wallet-ft-v5}"
GGUF="${GGUF:-gemma-4-E4B-wallet-ft-a075.Q4_K_M.gguf}"
SHA="${SHA:-40332b62f282336d92a94dc4147ecc44e83c0e11496ac2e5738d8ef342d1b09c}"
DATASET="${DATASET:-pf/tests.combined.yaml}"
OUT="${OUT:-runs/v5-safety-ab.out.json}"
JOBS="${JOBS:-12}"
POD=""

cleanup() {
  [ -n "$POD" ] && { echo "[safety] terminating pod $POD"
    uv run --with runpod python scripts/runpod_serve_gguf.py down --pod-id "$POD"; }
}
trap cleanup EXIT INT TERM

echo "[safety] launching a pod for $GGUF"
OUT_TXT=$(uv run --with runpod python scripts/runpod_serve_gguf.py up \
    --repo "$REPO" --revision main --gguf "$GGUF" --expect-sha "$SHA" --private \
    --n-ctx 4096 --parallel 8 --disk 40 --wait 2400 2>&1 | tee /dev/stderr)
POD=$(echo "$OUT_TXT" | sed -n 's/^RUNPOD_POD_ID=//p' | tail -1)
URL=$(echo "$OUT_TXT" | sed -n 's/^RUNPOD_LLAMA_URL=//p' | tail -1)
[ -z "$URL" ] && { echo "[safety] pod never became ready"; exit 1; }

export RUNPOD_V5_URL="$URL"
echo "[safety] scoring $DATASET at $URL"
EVAL_DATASET="$DATASET" PROMPTFOO_CONFIG_DIR=.promptfoo-v5safety scripts/eval.sh \
    -c promptfooconfig.v5-safety.remote.yaml -j "$JOBS" --no-cache -o "$OUT"

echo "[safety] ---- report ----"
uv run python scripts/report_1000.py "$OUT"
