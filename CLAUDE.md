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

Every `pf/tests.*.yaml` is a byte-stable output of a seeded script. Edit the
source, then regenerate:

```bash
uv run python scripts/generate_cases.py --extra-seeds datasets/seeds.arithmetic.yaml
uv run python scripts/generate_conversation_cases.py   # from datasets/seeds.conversations.yaml
uv run python scripts/build_combined_benchmark.py      # -> the 1000-case benchmark
uv run python scripts/generate_protocol_cases.py       # aave/safe, no longer in the benchmark
```

- Surface phrasings: `TRANSFER_TEMPLATES` / `SWAP_TEMPLATES` (+ narrative) in
  `src/wallet_evals/generation.py`.
- Safety refusals: `REFUSAL_SCENARIOS` + `build_refusal_case` (same file).
- RAILGUN keeps its own templates, refusals and multi-turn banks inside
  `protocols/railgun.py` (self-contained; it needs privacy-specific wording).
- `_PROTECTED_WORDS` in `generation.py` shields token symbols **and** the privacy
  verbs from `mutate_typos`: a typo'd "unshield" is unanswerable, not harder.
- Changing `pf/tools.json` invalidates the fine-tune JSONLs, which embed it
  verbatim (`tests/test_*finetune_integrity.py::test_tools_present` catches this).
  They're gitignored — just rerun `scripts/generate_finetune_data.py` and
  `scripts/generate_gemma4_finetune_data.py`.
- Protocol modules: `src/wallet_evals/protocols/` — gold is a **generic
  `executeTx`** for all protocols (no per-protocol tools; scorer/schema/tools.json
  stay unchanged when adding one). **One deliberate exception: `railgun`.** Shield's
  real ABI is nested note-ciphertext tuples and unshield is not a tx at all (Groth16
  proof + the wallet's own broadcaster), so it scores the app's own `shield`/
  `unshield` intent tools instead — see below. Prefer `executeTx` for anything new;
  only break the rule when there is genuinely no transaction to encode.

## The benchmark is `pf/tests.combined.yaml` — 1000 cases, two thirds multi-round

`scripts/build_combined_benchmark.py` concatenates exactly two generated files:

| Source | Cases | What it is |
| --- | --- | --- |
| `pf/tests.app-contract.yaml` | 429 | single-turn transfer/swap + arithmetic slice + refusals + 93 legacy 2-round cases |
| `pf/tests.conversations.yaml` | 571 | 2-6 round conversations, six mechanisms |

Round distribution (a round = one user turn + the assistant's reply; only the
model's reply to the LAST user turn is scored): **336 / 273 / 150 / 110 / 80 / 51**
for 1-6 rounds — 66.4% multi-round, 131 cases at 5+ rounds. **120 of the 1000 cases
(12.0%) expect NO tool call**, so a model that never acts scores 12% — always read
the call/no-call split (`scripts/report_1000.py` prints it) before a headline.
`scripts/dataset_census.py` prints the whole census (also `--csv` / `--cases-csv`).
Both the size and the distribution are **asserted** in
`tests/test_combined_benchmark_integrity.py`, so a source file that silently grows
or collapses the long conversations fails the suite rather than quietly changing
what a score means.

**Aave/Safe are no longer in the benchmark** — the wallet ships no lending or
multisig tool, so `executeTx` gold for Aave's Pool or a Safe self-call scored a
capability the product does not expose, and it was ~25% of the old 569-case
number. This is a removal from the BENCHMARK ONLY: `pf/tests.protocols.yaml`,
`scripts/generate_protocol_cases.py`, `src/wallet_evals/protocols/` and
`pf/prompt.py`'s `AAVE_REFERENCE`/`SAFE_REFERENCE` all still exist and still pass
`tests/test_protocol_integrity.py`. Run them directly:

```bash
EVAL_DATASET=pf/tests.protocols.yaml scripts/eval.sh -o protocols.out.json
```

**No published score is comparable to a run of this file.** It is 1000 cases, not
569, drops 140 aave/safe cases and adds 571 harder ones. The 429 app-contract
cases inside it are byte-identical to before (asserted against the frozen
base-unit dataset), so a per-case-id comparison on that subset is still honest —
a headline-number comparison is not. `space/build_static.py` needs a full new run
vintage before the report roster can move.

### The conversation slice — `src/wallet_evals/conversations.py`

Six mechanisms. The first four test conversational memory; the last two test the
CONTRACT BOUNDARY — what the wallet can and cannot execute:

- **`progressive`** (2-4 rounds) — the action's three fields revealed one per
  round in a seeded permutation. Capped at 4 rounds: there are only three fields.
- **`correction`** (2-6 rounds) — one field withheld so the assistant has a
  standing reason to keep asking, and every later round revises an
  already-stated field, naming the old value ("actually make it 7.77, not 6.02").
  Gold takes the LATEST value, so a model that keeps the first number fails.
- **`distractor`** (3-6 rounds) — non-actionable interruptions between the ask
  and the answer. Some canned replies carry a **number** ("around 12 gwei", "6
  decimals") that must not reach the call. Starts at 3 rounds: a 2-round
  distractor case has no distractor in it.
- **`switch`** (2-6 rounds) — a completed request, then abandonment for a
  different one. Gold is the final intent ALONE, so re-emitting the stale call or
  emitting both scores 0.
- **`exact_output`** (2-4 rounds, 32 cases) — the user answers "how much?" with an
  amount of the DESTINATION token. **Gold is NO CALL**, because the wallet does
  exact-input swaps only. See the `amount_side` section below.
- **`token_address`** (2-3 rounds, 24 cases) — the withheld token field arrives as
  a **0x contract address**, which `pf/tools.app.json` documents and
  `WalletTokenRegistry.token(matching:)` really does resolve. Gold carries the
  address **verbatim** — `APP_SYSTEM` has no token table, so translating it to a
  symbol would measure recall the wallet never needs. What it tests is 42-char hex
  pass-through across a turn boundary, a failure mode nothing else here exercises.
  It pairs with `safety-refusal-unverified-token-swap`, which names an address that
  is NOT in the registry and expects no call: known address → pass through,
  unknown → refuse.

Six constraints that are load-bearing, not stylistic:

1. **Every canned assistant turn is on-policy for `APP_SYSTEM`.** That prompt says
   to emit the call as soon as the values are known and never to ask for
   confirmation, so only three assistant moves are used: ask for a genuinely
   missing field, answer a non-actionable question in prose, or report a completed
   request as prepared. A scripted assistant stalling on a *complete* request
   would be showing the model an example of the behaviour we score it for not doing.
2. **A revision may never return a field to its original value.** With a
   4-symbol token bank, `USDC -> ETH -> USDC` is a plausible two-revision
   sequence whose gold equals the seed intent — a model that ignored every
   correction would score 1, and the case would measure nothing. `_revised_value`
   excludes the original as well as the current value;
   `test_correction_never_revises_a_field_back_to_its_original_value` sweeps seeds
   for it because it is probabilistic.
3. **The cancellation verb is never mutated.** A typo'd "Forget that one." →
   "Forgte that one." leaves nothing telling the model to abandon the first
   request, making the case unanswerable rather than harder — the same rule
   `_PROTECTED_WORDS` applies to the privacy verbs. It really fired before the fix.
4. **The `punctuation` mutator is excluded from this slice entirely.** It
   comma-groups amounts, so a conversation case could fail because the model
   reread "260,000.5" as "260.0005" rather than because it lost track of a value
   — and separator handling already has its own labelled slice
   (`arithmetic-*`). The other four mutators never touch digits.

5. **The exact-output demand and the address answer are never mutated either**,
   for the same reason as the cancellation verb. "exactly ... out" is the entire
   signal that a request is output-side; mangle it and the surface reads as an
   ordinary swap whose correct answer is a call, while gold still says no call.
6. **`_build_case`'s `expected_calls` override is only ever `[]`, only for
   `exact_output`.** Empty gold is a strong claim — it passes any model that stays
   silent — so `test_only_exact_output_cases_have_empty_gold` stops it spreading.

`ROUND_PLAN` in `scripts/generate_conversation_cases.py` fixes how many cases each
(rounds, mechanism) pair contributes, so the distribution is declared rather than
a by-product of pool sizes. The 56 contract-boundary cases are carved OUT of the
four memory mechanisms at the same round they land in, and
`EXPECTED_ROUND_TOTALS` asserts the result, so adding them left the slice at 571
and the benchmark's round mix unchanged. Amount literals in
`datasets/seeds.conversations.yaml` **and** the revision targets in `ALT_AMOUNTS`
are disjoint from `seeds.yaml`, `seeds.arithmetic.yaml` and
`finetune_seeds.yaml` — asserted on the GOLD amounts, so the slice stays honestly
held out.

### A gold field that cannot vary is a field that measures nothing

`tests/test_dataset_degeneracy.py` exists because two such fields shipped and no
test caught either; both were found by reading the dataset by hand.

| Field | Was | Why it mattered |
| --- | --- | --- |
| `swap.amount_side` | `"input"` in **436/436** golds | a model hardcoding the string scored it perfectly |
| `arithmetic/transfer.to` | `vitalik.eth` in **36/36** golds | and 51.5% of transfer golds overall |

The second is why those checks run **per slice** as well as globally: across the
whole dataset `to` had 63 distinct values and looked healthy, which is exactly how
the arithmetic slice's constant hid. Constants are still allowed — some are forced
by the contract, some deliberately hold a variable fixed — but only via
`ALLOWED_CONSTANTS`, keyed by (slice, tool, field) **with the reason recorded in
code**. A new constant fails loudly; a stale exemption also fails, because
`test_every_allowed_constant_is_still_actually_constant` checks the allowlist from
the other side.

`vitalik.eth` was also the **only** ENS name in `datasets/finetune_seeds.yaml`, so
"handles ENS" and "has memorised one string" were indistinguishable.
`generation.ENS_NAMES` is a 14-name bank that **excludes** it — disjoint from
training by construction — with varied shapes (plain, org-style, hyphenated,
subdomain, alphanumeric) because the capability is copying whatever `.eth` token
the user typed. The benchmark now has **13 distinct ENS names** and
`vitalik.eth` at 14% of transfer golds, all of them inside the untouched frozen
307. Expect a fine-tune's transfer accuracy to DROP against this: that drop is the
memorisation being measured, not a regression.

### Why `amount_side` has no `"output"` gold, and never should

The obvious fix — add cases whose gold is `amount_side: "output"` — is wrong, and
the wallet source says so. `pf/tools.app.json` pins the enum to `["input"]`, its
`amount` description says *"Do not use this tool when the user specifies only the
desired output amount"*, and **two independent guards reject anything else**:

- `ChatDashboardView.swift:3335` → `ChatIntentExecutionError.unsupportedSwapAmountSide`
- `SlashCommandParser.swift:66` → `malformedArgument("amount_side", "only input is supported")`

So `"output"` gold would score models on emitting a call the app throws on.
Dropping the field from scoring is also worse than it looks: today a model that
emits `"output"` correctly **fails**, and dropping it makes that silently pass.
The `exact_output` mechanism fixes the free point instead — a model that hardcodes
`"input"` and emits a swap now loses 32 cases, and one that emits `"output"` still
loses them. `scripts/convert.py` reached the same conclusion independently
(`test_convert_swap_exact_output_to_manual` refuses to auto-convert these).

### `pf/tools.app.json` promises two things the wallet cannot do — do not encode them

Audited against `ChatDashboardView.swift`. Filed as
[local-wallet-mac#92](https://github.com/Ethereum-dAI/local-wallet-mac/issues/92);
**#93** (the app's token registry is Sepolia-only, so `tokens(on: 1)` is empty)
and **#94** (`PortedAppEncoding.swift` claims to port that table verbatim but has
17 entries against the app's 7) came out of the same audit.

| Schema says | Executor does | Encode it? |
| --- | --- | --- |
| `token`: "or a 0x-prefixed contract address" | `token(matching:)` matches `contractAddress` case-insensitively | **yes** — the `token_address` mechanism |
| `amount`: 'use the literal `"all"`' | `guard rawAmount.lowercased() != "all"` throws (`:3267`, `:3354`, `:2257`) | **no** |
| `to`: "or a contact name" | `guard rawRecipient.contains(".")` throws (`:3288`) | **no** |

Whole-balance sends and contact-name recipients were both proposed as new slices
and **dropped for this reason**: gold must be a call the wallet can execute. Check
the executor, not just the schema, before adding a field to gold.

## RAILGUN shield/unshield — the human-unit exception

`shield`/`unshield` mirror `local-wallet-mac`'s `ToolDefinitions.swift` **verbatim**:
a human-decimal `amount` ("0.01"), an ETH-only `token`, and (unshield) a 0x `to`.
They are the ONE place base units do not apply — `RAILGUN_REFERENCE` in `pf/prompt.py`
states that override, and a wei-converted amount correctly scores 0. Two scoped
normalizations in the scorer, both erasing formatting rather than capability:
`token` folds case and defaults to ETH when omitted (the app's own
`SlashCommandParser` fills it in), and `amount` compares numerically so "0.010" ==
"0.01". Gold builders live in `intents.py` and refuse a non-ETH token or an
unresolved ENS recipient — the app cannot resolve ENS for unshield yet, so no gold
may encode one.

Because `pf/tools.json` now offers 5 tools to **every** case, run numbers from
before this change aren't directly comparable — the tool list in the request grew,
so a re-run is needed rather than a re-score of frozen outputs.

**This slice does not discriminate — treat it as a regression test.** E4B-base
scores ~97% here vs 9.8% on the generated set, because human-unit amounts remove
the base-unit arithmetic that separates models. What's left is tool selection plus
copying a number. Don't read a high railgun score as capability.

### OPEN FINDING: E4B under-refuses the zero address (not fixed here)

The refusal cases exposed a real safety gap, left **unfixed on purpose** — the fix
is a prompt change, which belongs in its own PR, not a dataset one. E4B-base
refuses burn-address (`0x…dEaD`) unshields 10/10 but the **zero** address only
**3/10**. Likely cause: `SYSTEM` teaches that the zero address *is* native ETH for
swaps, so the model holds a strong positive association for `0x0` and none for
`dEaD`.

Two hard-won constraints for whoever fixes it:

1. **A single pass cannot tell a fix from noise.** At temperature 0.2 over 4
   refusal cases, one attempt looked like it "moved" failures between cases and was
   actually flat. Use reps (4 cases × 5) and report a rate.
2. **A/B every model in the config, not just one.** Strengthening this wording so
   base hit 100% cost the *fine-tune* 7 shield cases: the sentence that fixed base
   named `swap` and `0x0` together inside the railgun block, and the swap-heavy
   fine-tune then emitted `swap` with `currencyIn=0x0` for plain shield requests.
   Removing the mention recovered the fine-tune and dropped base back to 75%. The
   two models want opposite wording, so it is a real tradeoff, not a wording bug.

Also rejected: a mechanical rule ("refuse if `to` starts with 4+ zeros"). It
passes all 4 cases but real addresses can begin with zeros, so it wins the eval by
shipping a false-positive heuristic.

## Prompt (`pf/prompt.py`)

- `SYSTEM` is the wallet operating manual (token book, base-unit rule, swap
  defaults) **plus a global SAFETY clause** (refuse burn/zero-address sends,
  unknown-spender approvals, unverified-contract swaps). The clause is scoped so
  normal transfers/ENS still execute.
- `vars.protocol` gates a per-protocol reference block (Safe/Aave); legacy cases
  render unchanged.
- `vars.expected_summary` is a **read-only viewer column** — never emitted to the
  model, never scored.

## Published artifacts (Hugging Face, `ef-dai-team`)

Everything lives under the org now — the models were **moved** out of
`gabrielfior/` (old ids still redirect, but don't write new ones).

| Repo | Vis. | Notes |
| --- | --- | --- |
| [`functiongemma-270m-wallet-ft`](https://huggingface.co/ef-dai-team/functiongemma-270m-wallet-ft) | public | `license: gemma` (inherited). The failed fine-tune. |
| [`gemma-4-E4B-wallet-ft`](https://huggingface.co/ef-dai-team/gemma-4-E4B-wallet-ft) | public | `license: apache-2.0`. The 80.1% one. |
| [`qwen3-8b-wallet-ft`](https://huggingface.co/ef-dai-team/qwen3-8b-wallet-ft) | public | `license: apache-2.0`. The 86.0% one — best on-device. |
| [`wallet-tool-calling-ft`](https://huggingface.co/datasets/ef-dai-team/wallet-tool-calling-ft) | private | Both training JSONLs + the Modal jobs. Apache-2.0. |
| [`wallet-tool-calling-eval`](https://huggingface.co/spaces/ef-dai-team/wallet-tool-calling-eval) | private | Static report Space. |

`space/` holds both Space builds. Deploy with `space/deploy.sh ef-dai-team`:

- `space/static/` — the **shipped** report. One tick per case per model over all
  307 cases, plus a browser showing every model's recorded output and the
  scorer's verdict. `build_static.py` bakes `data.json` from the `*.out.json`
  runs, so **`space/static/data.json` is the only committed record of those runs**
  (the `*.out.json` files themselves are gitignored) — don't ignore it. Every
  column is drawn from ONE run vintage (the 2026-08-10/11 relaunch, the first runs
  made with the 5-tool `pf/tools.json`), which is why the 2026-07-09 columns —
  gpt-4o-mini, Gemma-4 26B-A4B and both FunctionGemma-270M models — are no longer
  on it. Adding a model to `build_static.py:MODELS` means having a run of the same
  vintage, not just any export.
- `space/app.py` — a Gradio playground doing live inference over the local
  GGUFs, reusing the harness's own prompt/tools/scorer so it scores identically.
  **Not deployed:** Gradio and Docker Spaces are 402-gated behind a Team plan for
  orgs (and PRO for personal accounts) — Static is the only free SDK. Ship it
  with `space/deploy.sh ef-dai-team --gradio` once the org is upgraded.

### Never commit a second copy of anything

HF repos must be flat and self-contained, so the Space needs `prompt.py` and
`wallet_evals/` beside `app.py`, and the dataset needs the training scripts
beside the data. **Do not vendor them.** `space/stage.py` assembles those trees
on demand from one source each, into the gitignored `space/build/`:

```bash
uv run python space/stage.py gradio     # pf/ + src/wallet_evals/ + space/app.py
uv run python space/stage.py dataset    # generators + seeds + finetune/ + the data
uv run python space/stage.py static     # the report + its charts, from charts/
```

**The report tree is staged too, and that is load-bearing.** `hf upload` runs with
`--delete "*"`, so anything living only in the deployed Space is destroyed by the
next deploy. That already happened once: `overall-accuracy.jpg` was uploaded
straight to the Space, existed nowhere in this repo, and a later redeploy of
`space/static/` removed it along with the section that displayed it. Both charts
are now manifest entries pointing at `charts/`, which is also where
`scripts/export_chart_images.py` writes them — one source, and a redeploy
reproduces them.

Adding an import to `space/app.py` means adding the module to
`WALLET_EVALS_MODULES` in `stage.py` — nowhere else. `tests/test_space_staging.py`
fails if any tracked file becomes a byte-for-byte copy of another, if a staged
file drifts from its source, or if the staged Space can't import what it needs.
An earlier revision of this work shipped 16 such duplicates; the test exists so
that can't recur.

### The dataset repo must reproduce itself, and must not leak the eval set

It ships the generators, their seeds, `datasets/lookup.json`, the traced module
closure and a `pyproject.toml`, so a downloader can regenerate both JSONLs with
no access to this repo. `test_published_dataset_regenerates_its_own_data` runs
the staged generators and fails unless the output is byte-identical — it has
already caught two dependencies an import trace cannot see (a missing module,
and `lookup.json`, which `intents.py` reads at import time).

Two constraints when touching `DATASET_FILES`:

- **Keep the `src/` prefix.** `intents.py` resolves `datasets/lookup.json` from
  its own grandparent, so flattening the package misses by one level.
- **Never add anything in `EVAL_SET_FILES`.** The 307 held-out cases stay off the
  Hub; `test_dataset_never_publishes_the_eval_set` checks the manifest by path
  *and* by content hash, so a rename doesn't slip through.

`finetune/_bundled.py` lets the Modal jobs resolve their data from either layout
(`data_for_finetune/` here, `data/` there) — they all silently died at
`add_local_file` in the published repo before it existed. `deploy.sh` uploads
with `--delete "*"` so a file that moves in the manifest doesn't linger.

## The Qwen3-8B fine-tune

Third fine-tune, and the best on-device score so far: **86.0%** (vs 41.0% untuned
and 83.7% for the Gemma-4 fine-tune re-measured the same day). Published at
[`ef-dai-team/qwen3-8b-wallet-ft`](https://huggingface.co/ef-dai-team/qwen3-8b-wallet-ft).

```bash
uv run python scripts/generate_qwen_finetune_data.py     # 1739 rows, Hermes JSON
uv run --with modal modal run finetune/modal_finetune_qwen.py
uv run --with modal modal run finetune/modal_export_qwen.py            # merge -> Q4_K_M
uv run --with modal modal run finetune/modal_export_qwen.py --no-do-export --do-upload
uv run --with modal modal volume get qwen-ft-outputs \
    qwen3-8b-wallet-ft.Q4_K_M.gguf models/
PROMPTFOO_CONFIG_DIR=.promptfoo-qft scripts/eval.sh -c promptfooconfig.qwen3-ft.yaml \
    -j 1 --no-cache -o relaunch/qwen3-ft.out.json
uv run python scripts/compare_all_models.py              # the table
```

It trains on the SAME 1739 rows as the Gemma-4 set, only re-encoded — asserted by
`test_same_rows_as_the_gemma4_set`, because if the rows drift the comparison
measures data rather than model. Targets are Hermes
`<tool_call>{"name":…,"arguments":{…}}</tool_call>`, read back by
`wallet_evals/json_tool_calls.py` (which decodes *wrappers* only — it will not
repair malformed JSON, since that would erase a real failure).

Four things that will burn an A100 hour if you touch this recipe:

- **Import unsloth BEFORE trl.** unsloth patches TRL at import; alphabetise those
  lines and the names bind the unpatched classes, and the run dies with
  `eos_token '<EOS_TOKEN>' not found in vocabulary`. Three symptom-level fixes
  failed before the import order turned out to be the cause.
- **Never template the whole conversation for Qwen3.** Its chat template splits
  assistant messages on `</think>` and re-emits only what follows, silently
  deleting the target. Render `messages[:-1]` with `add_generation_prompt=True`
  and append the target — which also makes training byte-identical to inference.
  The all-masked-rows guard catches this; keep it.
- **TRL renames things**: `max_seq_length`→`max_length`,
  `tokenizer`→`processing_class`. The script picks whichever the installed
  version declares rather than pinning.
- **`modal volume get` can corrupt a 5 GB GGUF**: same byte count, different
  sha256, and it loads and runs without error. `finetune/modal_hash_gguf.py`
  hashes the file where it lives; check that before believing a benchmark of a
  downloaded model.

The fine-tune is **effectively deterministic despite T=0.6** — two full runs gave
307/307 identical verdicts, and repeated probes give identical outputs. Its loss
is 0.036, so the distribution is peaked enough that sampling rarely changes the
token. Do not explain differences between its runs as sampling noise.

## Where it still loses

The remaining failures are **base-unit arithmetic**, not a reasoning-to-emission
gap: the emitted call faithfully carries whatever the `<think>` trace computed.
Most are a single decimal place out on large amounts, and they repeat identically
across samples rather than drifting — see the "known weaknesses" section of the
published model card. Safety refusals sit at 5/7, unchanged from the Gemma-4
fine-tune and for the same reason: training holds ~1 example per safety category.

## A local GGUF run dies silently after ~150 cases without an explicit KV clear

`Llama.reset()` does NOT free the llama.cpp KV cache. Read its source: it sets
`n_tokens = 0` and only calls `llama_memory_clear` when the model `_is_recurrent`
or `_is_hybrid`. Gemma-4 is a plain transformer, so cells accumulate across
`create_completion` calls on the Llama object `pf/provider_functiongemma.py`
caches for the whole run (`_llms`), until no KV slot can be allocated. Then
`llama_decode` returns -3 — and **every remaining case fails the same way**,
because the poisoned object is reused.

Measured on the 1000-case benchmark with gemma4-e4b-base: **156 cases scored
cleanly, case 157 failed, and all 844 after it failed.** The export still held
1000 rows and reported a plausible-looking 14.7%. Three things about that make it
dangerous:

- **The case count check does not catch it.** `report_relaunch.py`'s
  `<< only N cases!` guard fires on a short export; this one was full length.
  What catches it is counting `failureReason == 2` (provider raised, case never
  scored) — `scripts/report_1000.py` reports those separately and excludes them
  from the denominator.
- **It is not a context-size problem, so raising `n_ctx` is the wrong fix.** The
  longest prompt in the 1000-case set is **1133 tokens** against `n_ctx: 4096`
  (measured with the model's own tokenizer; a chars/4 estimate is not good enough
  here — scrambled letter case and 42-char hex addresses tokenize badly, ~0.29
  tok/char).
- **It looks like a model weakness.** 844 empty outputs read as "the model
  stopped emitting tool calls on long conversations", which is exactly the claim
  this benchmark was built to test.

`_clear_kv(llm)` calls `reset()` **and** `llm._ctx.kv_cache_clear()` before every
case, plus once more on a failed decode before retrying once. Verified
verdict-identical on a 12-case probe (10 pass / 2 fail both ways), so it is state
hygiene, not a scoring change. It costs the shared-prefix reuse: **~12.3 s/case
instead of ~7.7 s** on Metal, i.e. ~3 h for 1000 cases. Worth it — a case's score
must not depend on which case ran before it, and the alternative (clear only on
failure, keeping prefix reuse) leaves the cascade one unvalidated retry away.

## Two evals at once need two result DBs

promptfoo writes every run to one SQLite DB under its config dir. Start a second
`promptfoo eval` while one is running and both race for the write lock:
`SQLITE_BUSY: database is locked` kills them mid-run — and the crash can still
write an `-o` export **missing the cases it never reached** (one such export held
289 of 307 and reported a plausible-looking 11.1%). Always check the case count;
`report_relaunch.py` flags it with `<< only N cases!`.

```bash
PROMPTFOO_CONFIG_DIR=.promptfoo-a scripts/eval.sh -c … -o relaunch/a.json &
PROMPTFOO_CONFIG_DIR=.promptfoo-b scripts/eval.sh -c … -o relaunch/b.json &
```

## Conventions

- `uv run` for Python; `uv run --with web3` for the (non-suite) fixture fetchers.
- `uv run pytest -q` is offline (no API key).
- `docs/` is gitignored — specs/plans live on disk only, not committed.
- Commit/push only when asked; branch off `main` and target PRs at `main` (it is
  the repo's default branch as of 2026-07-31). `feat/eval-harness` was the old
  default and is a superseded line of development — don't branch from it.
- Skills (SKILL.md) are **not** usable here — the models are plain chat
  completions with no agent runtime, whether hosted (OpenRouter) or local GGUF
  (`pf/provider_functiongemma.py` via llama-cpp-python). Gated context injection
  (`vars.protocol`) is the portable equivalent.
