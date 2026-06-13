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
| `datasets/seeds.yaml` | hand-authored structured intents for the generator |
| `pf/tests.yaml` | curated recognition-derived test cases (promptfoo-native; `vars.user_message` + `metadata.expected_calls`) |
| `pf/tests.generated.yaml` | **generated** harder cases (noise, param diversity, ablation negatives, multi-turn) |
| `pf/prompt.py` | chat builder: renders `messages` (multi-turn) or `user_message` |
| `pf/tools.json` | the `executeTx` / `readTx` / `swap` tool schemas |
| `pf/assert.py` | python assertion — reuses `score_case` against the gold in each test's metadata |
| `promptfooconfig.yaml` | providers (cheap OpenRouter models), prompt, tools, assertion, tests |
| `src/wallet_evals/` | reused core: schema, tool-call parsing, binary scorer, tests loader |
| `src/wallet_evals/intents.py` | shared gold-builders (structured intent → `expected_calls`) |
| `scripts/convert_recognition.py` | regenerates `pf/tests.yaml` from the Swift app's `recognition.json` |
| `scripts/generate_cases.py` | regenerates `pf/tests.generated.yaml` from `datasets/seeds.yaml` |
| `datasets/protocols/*.fixtures.json` | frozen real protocol txs (decoded) for protocol evals |
| `src/wallet_evals/protocols/` | protocol modules (Safe owner-management; Aave later) → `executeTx` gold |
| `scripts/generate_protocol_cases.py` | builds `pf/tests.protocols.yaml` from protocol fixtures |
| `pf/tests.protocols.yaml` | generated protocol-transaction eval cases (Safe add/remove signer) |

## Setup

```bash
uv sync                      # Python deps for the scorer/parser/loader
# Node is required for promptfoo (run via npx; no install needed)
echo "OPENROUTER_API_KEY=sk-..." > .env   # loaded automatically by promptfoo
```

## Run the eval

```bash
# Default: the generated dataset (pf/tests.generated.yaml)
PROMPTFOO_PYTHON="$PWD/.venv/bin/python" npx promptfoo@latest eval

# Curated recognition-derived set instead:
EVAL_DATASET=pf/tests.yaml PROMPTFOO_PYTHON="$PWD/.venv/bin/python" \
  npx promptfoo@latest eval

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
reported as needing manual authoring.

> Assumes the `local-wallet-mac` repo is checked out as a sibling directory
> (`../local-wallet-mac`); pass an explicit path as the first argument otherwise.

## Generate harder cases

```bash
uv run python scripts/generate_cases.py
```

Expands `datasets/seeds.yaml` (structured intents with `{vary: [...]}` params)
into `pf/tests.generated.yaml`: many noisy surface phrasings per intent, diverse
amounts/addresses, single-turn ablation negatives ("address is missing"), and
scripted multi-turn cases. Gold is **computed** from each seed intent, so every
generated case self-scores to 1 (`tests/test_generated_integrity.py`). Output is
deterministic for a fixed seed.

## Protocol-transaction evals (Safe)

```bash
uv run python scripts/generate_protocol_cases.py
EVAL_DATASET=pf/tests.protocols.yaml \
  PROMPTFOO_PYTHON="$PWD/.venv/bin/python" npx promptfoo@latest eval
```

Safe add/remove-signer cases whose gold is a generic `executeTx`
(`addOwnerWithThreshold` / `removeOwner`), built from real mainnet transactions
frozen in `datasets/protocols/safe.fixtures.json`. The model is given a Safe
reference + the Safe's owners list (so `prevOwner` is derivable) via a gated
system addendum; non-protocol cases are unaffected. Expect low scores until
protocol tooling/context is added — the gap is the point. Refresh the fixtures
with `uv run --with web3 python scripts/fetch_safe_fixtures.py`.

## Test (offline, no API key)

```bash
uv run pytest -q
```

Covers the parser, scorer, swap support, the converter, and a dataset-integrity
check that every gold case in `pf/tests.yaml` self-scores to 1.
