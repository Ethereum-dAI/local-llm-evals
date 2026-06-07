# evals-local-llm

LLM eval harness for the local agent wallet, run through
[promptfoo](https://www.promptfoo.dev/). Scores how well a model turns
natural-language requests into correct on-chain intents — `executeTx` / `readTx`
(decoded contract calls) and `swap` (a synthetic, routing-free swap intent).
Scoring is binary (pass/fail) and deterministic.

Design notes live under `docs/` (gitignored).

## Layout

| Path | Role |
| --- | --- |
| `pf/tests.yaml` | **single source of truth** for test cases (promptfoo-native; `vars.user_message` + `metadata.expected_calls`; multi-turn cases add `vars.history` of prior turns and `metadata.query_type: multi_turn`) |
| `pf/prompt.json` | static base of the system + user chat prompt |
| `pf/prompt.py` | promptfoo prompt function: renders the base plus the known-token table from `datasets/lookup.json` |
| `pf/tools.json` | the `executeTx` / `readTx` / `swap` tool schemas |
| `pf/assert.py` | python assertion — reuses `score_case` against the gold in each test's metadata |
| `promptfooconfig.yaml` | providers (cheap OpenRouter models), prompt, tools, assertion, tests |
| `src/wallet_evals/` | reused core: schema, tool-call parsing, binary scorer, tests loader |
| `scripts/convert_recognition.py` | regenerates `pf/tests.yaml` from the Swift app's `recognition.json` |

## Setup

```bash
uv sync                      # Python deps for the scorer/parser/loader
# Node is required for promptfoo (run via npx; no install needed)
echo "OPENROUTER_API_KEY=sk-..." > .env   # loaded automatically by promptfoo
```

## Run the eval

```bash
PROMPTFOO_PYTHON="$PWD/.venv/bin/python" npx promptfoo@latest eval
npx promptfoo@latest view          # interactive results (filter/group by metadata)
```

`PROMPTFOO_PYTHON` points promptfoo at the uv venv so the python assertion can
import `wallet_evals`. Useful flags: `--filter-first-n N` (subset of cases),
`--filter-providers <regex>` (subset of models), `-o results.json|html|csv`.

## Regenerate the dataset

```bash
uv run python scripts/convert_recognition.py
```

Rebuilds `pf/tests.yaml` from the Swift `recognition.json`: transfers/approvals →
`executeTx`, swaps → synthetic `swap`, ambiguous → no-call. Cases it can't resolve
mechanically (exact-output swaps, `"all"` amounts, unknown ENS/tokens) are
reported as needing manual authoring. Multi-turn cases (`query_type:
multi_turn`) are authored directly in `pf/tests.yaml` and carried across
regenerations untouched.

> Assumes the `local-wallet-mac` repo is checked out as a sibling directory
> (`../local-wallet-mac`); pass an explicit path as the first argument otherwise.

## Test (offline, no API key)

```bash
uv run pytest -q
```

Covers the parser, scorer, swap support, the converter, and a dataset-integrity
check that every gold case in `pf/tests.yaml` self-scores to 1.
