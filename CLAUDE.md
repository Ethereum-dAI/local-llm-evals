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
| `pf/tests.app-contract.yaml` | 429 | single-turn transfer/swap + arithmetic slice + refusals + 92 legacy 2-round cases |
| `pf/tests.conversations.yaml` | 571 | 2-6 round conversations, four mechanisms |

Round distribution (a round = one user turn + the assistant's reply; only the
model's reply to the LAST user turn is scored): **337 / 272 / 150 / 110 / 80 / 51**
for 1-6 rounds — 66.3% multi-round, 131 cases at 5+ rounds.
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

Four mechanisms, each a distinct failure mode with computed gold:

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

Four constraints that are load-bearing, not stylistic:

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

`ROUND_PLAN` in `scripts/generate_conversation_cases.py` fixes how many cases each
(rounds, mechanism) pair contributes, so the distribution is declared rather than
a by-product of pool sizes. Amount literals in
`datasets/seeds.conversations.yaml` **and** the revision targets in `ALT_AMOUNTS`
are disjoint from `seeds.yaml`, `seeds.arithmetic.yaml` and
`finetune_seeds.yaml` — asserted on the GOLD amounts, so the slice stays honestly
held out.

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
| [`wallet-tool-calling-ft`](https://huggingface.co/datasets/ef-dai-team/wallet-tool-calling-ft) | private | Both training JSONLs + the Modal jobs. Apache-2.0. |
| [`wallet-tool-calling-eval`](https://huggingface.co/spaces/ef-dai-team/wallet-tool-calling-eval) | private | Static report Space. |

`space/` holds both Space builds. Deploy with `space/deploy.sh ef-dai-team`:

- `space/static/` — the **shipped** report. One tick per case per model over all
  307 cases, plus a browser showing every model's recorded output and the
  scorer's verdict. `build_static.py` bakes `data.json` from the `*.out.json`
  runs, so **`space/static/data.json` is the only committed record of those runs**
  (the `*.out.json` files themselves are gitignored) — don't ignore it.
- `space/app.py` — a Gradio playground doing live inference over the three local
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
```

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
