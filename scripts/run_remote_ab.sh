#!/usr/bin/env bash
# Run the safety A/B against a RunPod-served GGUF, and time it against local.
#
# Deliberately NOT under `set -e`: promptfoo exits non-zero whenever ANY case fails,
# which is the normal outcome of an eval measuring failures. Under `set -e` the first
# arm would abort the script and the export for the remaining arms would never be
# written — a scoring run that looks like a crash.
#
# Usage: RUNPOD_LLAMA_URL=https://<pod>-8080.proxy.runpod.net scripts/run_remote_ab.sh
set -uo pipefail
cd "$(dirname "$0")/.."

: "${RUNPOD_LLAMA_URL:?set RUNPOD_LLAMA_URL (scripts/runpod_serve_gguf.py up prints it)}"
JOBS="${JOBS:-8}"
DATASET="${DATASET:-pf/tests.dev.safety.yaml}"
OUT="${OUT:-runs/remote-safety.out.json}"
# A dir of its own — promptfoo keeps one SQLite result DB per config dir and two
# concurrent runs sharing one deadlock on the write lock (SQLITE_BUSY).
export PROMPTFOO_CONFIG_DIR="${PROMPTFOO_CONFIG_DIR:-.promptfoo-safety-remote}"
export RUNPOD_LLAMA_URL

echo "== remote: $DATASET at -j $JOBS via $RUNPOD_LLAMA_URL"
START=$(date -u +%s)
EVAL_DATASET="$DATASET" scripts/eval.sh \
    -c promptfooconfig.safety-ab.remote.yaml -j "$JOBS" --no-cache -o "$OUT"
RC=$?
END=$(date -u +%s)
echo "== remote: done rc=$RC in $((END-START))s -> $OUT"
echo "REMOTE-AB-DONE elapsed_s=$((END-START)) rc=$RC"
