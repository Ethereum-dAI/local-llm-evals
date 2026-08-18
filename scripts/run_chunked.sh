#!/usr/bin/env bash
# Score one provider over the 1000-case benchmark in 250-case chunks.
#
# Why chunks: a 1000-case local-GGUF run takes ~3 h, and a run that dies at
# minute 48 writes NO `-o` export at all — promptfoo only writes it at the end,
# so the 234 completed cases were recoverable only out of its SQLite DB. Chunking
# means a death costs one chunk, and re-running the script SKIPS the chunks whose
# export already exists, so it resumes instead of starting over.
#
# Each chunk gets its OWN PROMPTFOO_CONFIG_DIR: promptfoo writes every run to one
# SQLite DB per config dir, and concurrent or repeated runs against one DB race
# for the write lock (SQLITE_BUSY kills them mid-run — see CLAUDE.md).
#
# Usage:
#   scripts/run_chunked.sh <provider-label> [config]
#
#   scripts/run_chunked.sh gemma4-e4b-base
#   scripts/run_chunked.sh gemma4-e4b-ft-v4-appprompt
#
# Outputs runs/<label>.part<N>.out.json, which scripts/report_1000.py reads as one
# run:  --run "base=runs/gemma4-e4b-base.part*.out.json"
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:?usage: run_chunked.sh <provider-label> [config]}"
CONFIG="${2:-promptfooconfig.v4-vs-base.yaml}"
CHUNKS=(runs/chunks/tests.part*.yaml)

echo "provider: $LABEL"
echo "config:   $CONFIG"
echo "chunks:   ${#CHUNKS[@]}"

for chunk in "${CHUNKS[@]}"; do
  part="$(basename "$chunk" .yaml)"; part="${part#tests.}"
  out="runs/${LABEL}.${part}.out.json"
  if [ -f "$out" ]; then
    echo "== $part: already done ($out) — skipping"
    continue
  fi
  echo "== $part: starting $(date -u +%H:%M:%SZ)"
  # A chunk that fails must not abort the remaining chunks: the point of chunking
  # is that one loss is one chunk. `|| true` plus the missing-export check below
  # reports it and moves on.
  EVAL_DATASET="$chunk" \
  PROMPTFOO_CONFIG_DIR=".promptfoo-${LABEL}-${part}" \
    scripts/eval.sh -c "$CONFIG" -j 1 --no-cache \
      --filter-providers "$LABEL" -o "$out" \
      > "runs/${LABEL}.${part}.log" 2>&1 || true
  if [ -f "$out" ]; then
    n=$(uv run --quiet python -c "
import json,sys
d=json.load(open('$out'))['results']['results']
print(len(d), sum(1 for r in d if r.get('success')),
      sum(1 for r in d if r.get('failureReason')==2))
")
    echo "== $part: done $(date -u +%H:%M:%SZ) -> cases/passed/errors: $n"
  else
    echo "== $part: FAILED — no export written (see runs/${LABEL}.${part}.log)"
  fi
done

echo "ALL CHUNKS ATTEMPTED for $LABEL"
ls -l runs/"${LABEL}".part*.out.json 2>/dev/null || echo "no exports!"
