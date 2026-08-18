---
license: apache-2.0
pretty_name: Wallet Tool-Calling SFT (app contract)
language:
  - en
task_categories:
  - text-generation
tags:
  - function-calling
  - tool-use
  - ethereum
  - wallet
  - gemma
  - qwen
size_categories:
  - 1K<n<10K
configs:
  - config_name: gemma4
    data_files: data/gemma4_train.jsonl
  - config_name: gemma4-with-protocol
    data_files: data/gemma4_train.with-protocol.jsonl
  - config_name: qwen
    data_files: data/qwen_train.jsonl
  - config_name: qwen-with-protocol
    data_files: data/qwen_train.with-protocol.jsonl
  - config_name: functiongemma
    data_files: data/functiongemma_train.jsonl
---

# Wallet tool-calling SFT data — the app contract

Supervised fine-tuning data that teaches a small local model to turn a
natural-language wallet request into the **exact structured tool call the macOS
[Local Wallet](https://github.com/Ethereum-dAI/local-wallet-mac) app can
execute**:

> *"Send 0.1 ETH to vitalik.eth"*
>
> ```
> transfer{to:"vitalik.eth", amount:"0.1", token:"ETH"}
> ```

## Read this first: the contract changed

Earlier versions of this dataset trained `executeTx(chainId, to, value, function,
args)` with **base-unit amounts** and **ENS pre-resolved** to `0x…`. **The wallet
never registers that tool.** Grep `executeTx` in `WalletToolLayer/ToolDefinitions.swift`
and you get nothing: the app exposes `transfer(to, amount, token)` and
`swap(from_token, to_token, amount, amount_side)`, takes **human decimal
amounts**, and resolves ENS itself.

That mismatch was not academic. The fine-tune trained against the old contract
scored 80.1% on the benchmark it was built for while producing a signable
UserOperation only **71.9%** of the time inside the app.

Everything here is the app contract. If you have a local copy of this dataset
from before 2026-08-18 with `executeTx` in its gold, it is superseded — discard it.

## Four files: two row-mixes × two encodings

| File | Rows | Encoding | Base model |
| --- | --: | --- | --- |
| `data/gemma4_train.jsonl` | **1768** | Gemma DSL | `google/gemma-4-E4B-it` |
| `data/gemma4_train.with-protocol.jsonl` | **1863** | Gemma DSL | same |
| `data/qwen_train.jsonl` | **1768** | Hermes JSON | `Qwen/Qwen3-8B` |
| `data/qwen_train.with-protocol.jsonl` | **1863** | Hermes JSON | same |
| `data/functiongemma_train.jsonl` | 1768 | FunctionGemma DSL | `unsloth/functiongemma-270m-it` (negative result — see below) |

**Which one do you want?**

- **`*_train.jsonl` (1768) is the default.** Wallet rows only.
- **`*_train.with-protocol.jsonl` (1863)** adds 95 Aave/Safe transaction-builder
  rows. These teach a *second* tool contract (`executeTx`, base units, resolved
  addresses) opposed to the app's own, which is why they are no longer the
  default. They are published because **this is what `gemma-4-E4B-wallet-ft-v4`
  and `qwen3-8b-wallet-ft-v4` actually trained on** — without them neither
  published model is reproducible.

The gemma and qwen files are the **same rows with different targets**: identical
row ids, identical gold, identical user turns, differing only in how the
assistant turn is serialised (`<|tool_call>call:transfer{…}` vs
`<tool_call>{"name":"transfer",…}</tool_call>`). 1725 of 1863 targets differ; the
138 that match are refusal rows, whose target is prose either way.

## Row format

```json
{
  "id": "ft-gen-transfer-pos-0137",
  "category": "generated-transfer-pos",
  "protocol": "transfer",
  "messages": [
    {"role": "system",    "content": "<the app's verbatim 533-char prompt>"},
    {"role": "user",      "content": "Send 0.000002 USDC to vitalik.eth"},
    {"role": "assistant", "content": "<|tool_call>call:transfer{…}<tool_call|>"}
  ],
  "tools": [ "…pf/tools.app.json, verbatim…" ],
  "expected_calls": [ "…gold, for validation only — never fed to the model…" ]
}
```

The system turn is **not** written by hand. It is the app's own prompt, extracted
from the app via `wallet-eval prompt-dump` and shipped here as
`pf/app_contract_reference.json`. Training asserts byte-parity against it before
spending a GPU hour, because a fine-tune trained on a prompt the app does not send
is measuring the wrong thing — two earlier defects were found exactly that way
(`enable_thinking` never passed, so no model had trained with the `<|think|>`
marker the app emits; and the harness rendering 2925 characters where the app
sends 2935).

`messages` minus the final assistant turn is *exactly* what the eval harness feeds
at inference. Every target decodes and scores back to 1.0 through the harness's
unchanged scorer.

## Known weaknesses — read before you retrain

A 1000-case benchmark (2-6 round conversations, held out) was run against
`gemma-4-E4B-wallet-ft-v4` and its untuned base. The fine-tune scored **78.7%**
against base's **90.7%** — *worse overall* — while scoring **96.3%** on the 349
cases that match the older shallow benchmark. It is **overfit to its training
distribution**, and the holes are visible in this data:

| This data contains | Consequence measured |
| --- | --- |
| 85.9% one-turn, 14.1% two-turn, **0% three-plus** | 95.8% at 1 round → **49.0% at 6**; base is flat (~86-94%) |
| **one** ENS name (`vitalik.eth`, ×420) | novel ENS names **−16.9pt** vs base; falsely refused as "not a valid Ethereum address" |
| `amount_side: "input"` in all 792 swaps, no output-side rows | exact-output requests: **0/32** (base 32/32) |
| **zero** token fields given as a 0x contract address | token-as-address: **25%** (base 91.7%) |
| no interruption/distractor rows | distractor conversations **29.7%** (base 76.2%) |

It did buy real improvements — safety refusal 61.2% → **93.9%**, ablation 85.7% →
**100%**, single-turn 92.4% → **96.4%** — so it is a trade, not a pure regression.

**If you fine-tune on this data, add a validation split containing capabilities it
lacks.** The original recipe had none at all (3 epochs, LR 2e-4, no
`eval_dataset`), so nothing could detect the collapse. A 10% holdout of *these*
rows would not help either: it is 86% single-turn, so its loss falls happily while
multi-round accuracy halves.

## Disjoint from the eval set — by construction

Training rows come from **separate sources under a different seed** to the
evaluation sets: `datasets/finetune_seeds.yaml` rather than `datasets/seeds.yaml`,
with disjoint amount banks. Verified by exact user-turn comparison: **0 of 1739
training rows collide** with the 1000-case benchmark. An integrity test in the
harness enforces it on every commit.

The evaluation sets are deliberately **not** published here — keeping them off the
Hub is what stops them leaking into training corpora and is the reason any of these
numbers mean anything.

## Reproduce

Self-contained: the generators, their seeds, the modules they import, and the
training jobs all ship with the data. `tests/test_space_staging.py` in the harness
reruns every generator **from this published tree** on each commit and fails unless
the output matches these files bit for bit — including the with-protocol variants.

```bash
uv run python scripts/generate_gemma4_finetune_data.py --out data/gemma4_train.jsonl
uv run python scripts/generate_gemma4_finetune_data.py --include-protocol-rows \
    --out data/gemma4_train.with-protocol.jsonl
uv run python scripts/generate_qwen_finetune_data.py   --out data/qwen_train.jsonl
uv run python scripts/generate_qwen_finetune_data.py   --include-protocol-rows \
    --out data/qwen_train.with-protocol.jsonl
uv run python scripts/generate_finetune_data.py        --out data/functiongemma_train.jsonl
```

The bundled `pyproject.toml` puts `src/` on the path; the only third-party
dependency is PyYAML.

**Train.** Unsloth does not run on macOS, so training happens on a CUDA box.
Launch with `--detach`: these are ephemeral Modal apps, and without it the app is
stopped when the local entrypoint returns, silently training nothing.

```bash
modal run --detach scripts/modal_finetune_gemma4.py   # LoRA adapter (E4B)
modal run --detach scripts/modal_export_gemma4.py     # merge + Q4_K_M GGUF
```

They resolve their inputs from `data/` here or `data_for_finetune/` in the harness,
whichever exists (`scripts/_bundled.py`).

## The 270M row is a negative result

`data/functiongemma_train.jsonl` targets FunctionGemma-270M, which reached 8.8% on
the old 307-case set — and that came entirely from ablation and refusal cases,
where the correct answer is *not* to emit a call. On transfers, swaps and
multi-turn it scored 0%. The same data lifted Gemma-4 E4B from 9.8% to 80.1%, so
the binding constraint was model capacity, not the data. Kept for the record, not
to deploy.

## Models trained from this data

- [`gemma-4-E4B-wallet-ft-v4`](https://huggingface.co/ef-dai-team/gemma-4-E4B-wallet-ft-v4) — app contract, 1863 rows
- [`qwen3-8b-wallet-ft-v4`](https://huggingface.co/ef-dai-team/qwen3-8b-wallet-ft-v4) — same 1863 rows, different base
- [`gemma-4-E4B-wallet-ft`](https://huggingface.co/ef-dai-team/gemma-4-E4B-wallet-ft) — v1, OLD base-unit contract (superseded)
- [`functiongemma-270m-wallet-ft`](https://huggingface.co/ef-dai-team/functiongemma-270m-wallet-ft) — negative result

Benchmarks: [`wallet-eval-benchmark`](https://huggingface.co/datasets/ef-dai-team/wallet-eval-benchmark).

## License

Apache-2.0. Every row is generated deterministically by the seeded scripts shipped
alongside it — no model output and no third-party corpus is involved.

Models fine-tuned from it carry their own base-model licenses:
`functiongemma-270m-wallet-ft` inherits the Gemma Terms of Use;
`gemma-4-E4B-wallet-ft*` is Apache-2.0; Qwen3-8B is Apache-2.0.
