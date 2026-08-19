#!/usr/bin/env bash
# Serve both v5 alpha variants on rented GPUs and score the 145-case OOD dev set.
#
# THE DEV SET IS THE SELECTOR and this is the one decision it makes. The frozen
# 1000-case benchmark is scored ONCE, afterwards, for the winner only — scoring both
# alphas there would make the test set a hyperparameter.
#
# Deliberately NOT under `set -e`: a failure in one arm must not skip the pod teardown
# below, because a forgotten pod is the only way this gets expensive.
#
#   scripts/run_v5_alpha_ab.sh                 # both arms, dev set
#   ALPHAS="a1" scripts/run_v5_alpha_ab.sh     # one arm only
set -u
cd "$(dirname "$0")/.."

REPO="${REPO:-ef-dai-team/gemma-4-E4B-wallet-ft-v5}"
ALPHAS="${ALPHAS:-a1 a075}"
DATASET="${DATASET:-pf/tests.dev.yaml}"
OUT="${OUT:-runs/v5-alpha-ab.out.json}"
JOBS="${JOBS:-12}"
PODS=""
declare -A URLS=()

cleanup() {
  for pid in $PODS; do
    echo "[ab] terminating pod $pid"
    uv run --with runpod python scripts/runpod_serve_gguf.py down --pod-id "$pid"
  done
}
trap cleanup EXIT INT TERM

for a in $ALPHAS; do
  echo "[ab] launching a pod for alpha=$a"
  # --private: the v5 repo is private, so the in-pod download needs a bearer token or
  # curl writes a zero-byte file and llama-server reports 'failed to load model'.
  OUTPUT=$(uv run --with runpod python scripts/runpod_serve_gguf.py up \
      --repo "$REPO" --revision main --private \
      --gguf "gemma-4-E4B-wallet-ft-${a}.Q4_K_M.gguf" \
      --n-ctx 4096 --parallel 8 --disk 40 --wait 2400 2>&1 | tee /dev/stderr)
  pid=$(echo "$OUTPUT" | sed -n 's/^RUNPOD_POD_ID=//p' | tail -1)
  url=$(echo "$OUTPUT" | sed -n 's/^RUNPOD_LLAMA_URL=//p' | tail -1)
  if [ -z "$url" ]; then echo "[ab] alpha=$a FAILED to come up"; continue; fi
  PODS="$PODS $pid"
  URLS[$a]="$url"
  echo "[ab] alpha=$a ready at $url"
done

if [ ${#URLS[@]} -eq 0 ]; then echo "[ab] no arm came up"; exit 1; fi

export RUNPOD_A10_URL="${URLS[a1]:-}"
export RUNPOD_A075_URL="${URLS[a075]:-}"
# Filter to the arms that actually came up, so a half-failed launch still produces a
# usable single-arm export instead of 145 provider errors on the missing one.
FILTER=""
[ -z "$RUNPOD_A10_URL" ]  && FILTER="v5-alpha0.75"
[ -z "$RUNPOD_A075_URL" ] && FILTER="v5-alpha1.0"

echo "[ab] scoring $DATASET"
set -- -c promptfooconfig.v5-alpha-ab.remote.yaml -j "$JOBS" --no-cache -o "$OUT"
[ -n "$FILTER" ] && set -- "$@" --filter-providers "$FILTER"
EVAL_DATASET="$DATASET" PROMPTFOO_CONFIG_DIR=.promptfoo-v5ab scripts/eval.sh "$@"

echo "[ab] ---- report ----"
uv run python scripts/report_1000.py "$OUT"
