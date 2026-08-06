#!/usr/bin/env bash
# Publish the eval report Space + its training-data dataset repo to Hugging Face.
#
#   space/deploy.sh [namespace]
#
# Deploys the STATIC report (space/static/) — free on any plan. The interactive
# Gradio app in space/ needs a Team plan on the org; once that exists, run
#   space/deploy.sh <ns> --gradio [flavor]
# to publish it alongside as a second Space.
#
# Prerequisites: `hf auth login` with a write token, and the org must exist
# (orgs can only be created at https://huggingface.co/organizations/new).
#
# Idempotent — re-run to push updates.
set -euo pipefail

NS="${1:-ef-dai-team}"
MODE="${2:-}"
FLAVOR="${3:-cpu-basic}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

DATASET="$NS/wallet-tool-calling-ft"
REPORT="$NS/wallet-tool-calling-eval"
PLAYGROUND="$NS/wallet-tool-call-playground"

echo "==> Logged in as"
hf auth whoami

echo "==> Rebuilding frozen data from the run artefacts"
(cd "$ROOT" && uv run python space/build_static.py && uv run python space/build_data.py)

echo "==> Dataset: $DATASET"
hf repos create "$DATASET" --repo-type dataset --private --exist-ok
hf upload "$DATASET" "$ROOT/dataset_repo" . --repo-type dataset \
    --exclude "**/__pycache__/**" \
    --commit-message "Wallet tool-calling SFT data + training scripts"

if [[ "$MODE" == "--gradio" ]]; then
    echo "==> Interactive Space: $PLAYGROUND (flavor: $FLAVOR)"
    hf repos create "$PLAYGROUND" --repo-type space --space-sdk gradio \
        --flavor "$FLAVOR" --private --exist-ok
    hf upload "$PLAYGROUND" "$HERE" . --repo-type space \
        --exclude "**/__pycache__/**" --exclude "static/**" \
        --exclude "deploy.sh" --exclude "build_data.py" --exclude "build_static.py" \
        --commit-message "Live playground over the local GGUFs"
    echo "Playground: https://huggingface.co/spaces/$PLAYGROUND"
    exit 0
fi

echo "==> Report Space: $REPORT (static)"
hf repos create "$REPORT" --repo-type space --space-sdk static --private --exist-ok
hf upload "$REPORT" "$HERE/static" . --repo-type space \
    --exclude "**/__pycache__/**" \
    --commit-message "Eval report: 307 cases x 7 models, exact-match scoring"

echo
echo "Report:  https://huggingface.co/spaces/$REPORT"
echo "Dataset: https://huggingface.co/datasets/$DATASET"
echo
echo "Logs: hf spaces logs $REPORT --build --follow"
