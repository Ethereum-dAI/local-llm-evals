#!/usr/bin/env bash
# Publish the eval report Space + its training-data dataset repo to Hugging Face.
#
#   space/deploy.sh [namespace]
#   space/deploy.sh [namespace] --gradio [flavor]
#
# Deploys the STATIC report (space/static/) — free on any plan. The interactive
# Gradio app needs a Team plan on the org; once that exists, `--gradio` publishes
# it alongside as a second Space.
#
# The Space and dataset trees are ASSEMBLED, not committed: `space/stage.py`
# copies each file from its one source in the repo into space/build/<target>.
# Nothing under space/ duplicates a file that already exists elsewhere, so
# editing pf/ or src/ can't leave a stale copy behind.
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

if [[ "$MODE" == "--gradio" ]]; then
    echo "==> Staging the Gradio tree"
    (cd "$ROOT" && uv run python space/stage.py gradio)
    echo "==> Interactive Space: $PLAYGROUND (flavor: $FLAVOR)"
    hf repos create "$PLAYGROUND" --repo-type space --space-sdk gradio \
        --flavor "$FLAVOR" --private --exist-ok
    hf upload "$PLAYGROUND" "$HERE/build/gradio" . --repo-type space \
        --exclude "**/__pycache__/**" \
        --commit-message "Live playground over the local GGUFs"
    echo "Playground: https://huggingface.co/spaces/$PLAYGROUND"
    exit 0
fi

echo "==> Staging the dataset tree"
(cd "$ROOT" && uv run python space/stage.py dataset)

echo "==> Dataset: $DATASET"
hf repos create "$DATASET" --repo-type dataset --private --exist-ok
hf upload "$DATASET" "$HERE/build/dataset" . --repo-type dataset \
    --exclude "**/__pycache__/**" \
    --commit-message "Wallet tool-calling SFT data + training scripts"

# space/static/ duplicates nothing, so it uploads straight from the tree.
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
