# evals-local-llm

LLM eval harness for the local agent wallet. Scores how well a model turns
natural-language requests into correct on-chain intents — `executeTx` / `readTx`
calls (decoded contract calls) and `swap` (a synthetic, routing-free swap
intent). Scoring is binary (pass/fail), deterministic, and offline.

Design + plan live under `docs/` (gitignored).

## Setup

```bash
uv sync
```

## Inspect a dataset (no model needed)

```bash
uv run python scripts/preview.py                 # the 5-case tiny demo
uv run python scripts/preview.py datasets/cases.json
```

## Offline scoring smoke test (no API key)

```bash
uv run python scripts/smoke_score.py             # canned responses vs tiny.json
```

## Run the eval via OpenRouter

```bash
export OPENROUTER_API_KEY=sk-...
uv run python -m wallet_evals.cli \
  --backend openrouter \
  --model openai/gpt-4o-mini \
  --dataset datasets/cases.json \
  --repeats 1 \
  --json-out out.json
```

Prints overall accuracy plus per-slice breakdowns (level / protocol / query_type
/ difficulty / language / capability) and writes the full results to `out.json`.

## Regenerate the dataset from the legacy source

```bash
uv run python scripts/convert_recognition.py
```

Converts the Swift app's `recognition.json` into the unified schema (transfers →
`executeTx`, swaps → synthetic `swap`, ambiguous → no-call). Cases it can't
resolve mechanically (exact-output swaps, `"all"` amounts, unknown ENS/tokens)
are reported as needing manual authoring.

## Test

```bash
uv run pytest -q
```
