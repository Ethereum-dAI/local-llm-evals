#!/usr/bin/env bash
# Run the promptfoo eval with the project's uv-managed Python, so the
# `file://pf/assert.py` scorer can import `wallet_evals`.
#
# Without this, `npx promptfoo eval` spawns the system `python3` (which has no
# wallet_evals on its path) and EVERY case fails with
#   Error running Python script: ModuleNotFoundError: No module named 'wallet_evals'
# while still burning the API calls — a silent, expensive 0% run.
#
# Usage (args pass through to `promptfoo eval`):
#   scripts/eval.sh -o out.json
#   EVAL_DATASET=pf/tests.protocols.yaml scripts/eval.sh -o protocols.out.json
set -euo pipefail
cd "$(dirname "$0")/.."

# The venv interpreter has wallet_evals installed editable (see pyproject).
PROMPTFOO_PYTHON="$(uv run --quiet python -c 'import sys; print(sys.executable)')"
export PROMPTFOO_PYTHON

exec npx promptfoo eval "$@"
