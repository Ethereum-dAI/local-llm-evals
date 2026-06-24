# CLAUDE.md

Operational pointers for working in this repo. See `README.md` for the full layout.

## What this is

A **deterministic** promptfoo eval harness for the local wallet LLM. It scores
whether a model turns a natural-language request into the correct structured
tool call. It is built as a **discriminator**: a capable anchor (gpt-5) should
score very high, weaker models (gpt-4o-mini, gemma-4) lower. base-unit
arithmetic and exact arg encoding are the main capability separators.

## Running the eval — always via the wrapper

```bash
scripts/eval.sh                              # default dataset (pf/tests.generated.yaml)
EVAL_DATASET=pf/tests.protocols.yaml scripts/eval.sh -o protocols.out.json
```

**Never run a bare `npx promptfoo eval`.** It spawns the system `python3`, which
can't import `wallet_evals`, so the `pf/assert.py` scorer errors on *every* case
(`ModuleNotFoundError`) — a silent 0% run that still spends all the API calls.
`scripts/eval.sh` exports `PROMPTFOO_PYTHON` (the uv venv) to fix this. The API
key lives in `.env` (promptfoo auto-loads it); `*.out.json` is gitignored.

## Re-score frozen outputs instead of re-running

A full run is slow + costs money. promptfoo captures every model output in the
`-o` JSON, so to see the effect of a **scorer or gold change** (not a prompt
change), replay the frozen outputs through the real scorer offline — no API:

```python
import json, importlib
ga = importlib.import_module("pf.assert").get_assert   # 'assert' is a keyword → importlib
d = json.load(open("safety.out.json"))
for r in d["results"]["results"]:
    md = r["testCase"]["metadata"]; out = r["response"]["output"]
    ok = ga(out, {"test": {"metadata": md}, "providerResponse": r["response"]})["pass"]
```

This only works if the **prompt** is unchanged (outputs would differ otherwise).
For a prompt change, do a small A/B: run a subset, toggle the prompt, compare.

## Scoring rules (don't break these)

- Binary, deterministic. Gold = `metadata.expected_calls`, **computed** from a
  structured intent, never parsed from the surface. Every gold self-scores to 1
  (`tests/test_*_integrity.py`) — keep it that way.
- `scorer._norm_scalar`: lowercases 0x-addresses and coerces JSON numbers to
  decimal strings (`0` == `"0"`; both ABI-encode identically). Don't add
  normalization that erases a *real* capability gap — int-vs-string was a genuine
  false negative; loosening further would destroy discrimination.
- Refusal cases have `expected_calls == []` → pass iff the model makes **no** tool
  call. All models currently pass these (safety floor, not a discriminator).

## Datasets are generated — don't hand-edit

`pf/tests.generated.yaml` and `pf/tests.protocols.yaml` are byte-stable outputs of
seeded scripts. Edit the source, then regenerate:

```bash
uv run python scripts/generate_cases.py            # from datasets/seeds.yaml
uv run python scripts/generate_protocol_cases.py   # from datasets/protocols/*.fixtures.json
```

- Surface phrasings: `TRANSFER_TEMPLATES` / `SWAP_TEMPLATES` (+ narrative) in
  `src/wallet_evals/generation.py`.
- Safety refusals: `REFUSAL_SCENARIOS` + `build_refusal_case` (same file).
- Protocol modules: `src/wallet_evals/protocols/` — gold is a **generic
  `executeTx`** for all protocols (no per-protocol tools; scorer/schema/tools.json
  stay unchanged when adding one).

## Prompt (`pf/prompt.py`)

- `SYSTEM` is the wallet operating manual (token book, base-unit rule, swap
  defaults) **plus a global SAFETY clause** (refuse burn/zero-address sends,
  unknown-spender approvals, unverified-contract swaps). The clause is scoped so
  normal transfers/ENS still execute.
- `vars.protocol` gates a per-protocol reference block (Safe/Aave); legacy cases
  render unchanged.
- `vars.expected_summary` is a **read-only viewer column** — never emitted to the
  model, never scored.

## Conventions

- `uv run` for Python; `uv run --with web3` for the (non-suite) fixture fetchers.
- `uv run pytest -q` is offline (no API key).
- `docs/` is gitignored — specs/plans live on disk only, not committed.
- Commit/push only when asked; branch off `eval/protocol-harness`.
- Skills (SKILL.md) are **not** usable here — the models run as plain OpenRouter
  chat completions with no agent runtime. Gated context injection (`vars.protocol`)
  is the portable equivalent.
