---
license: apache-2.0
pretty_name: Wallet Tool-Calling SFT
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
size_categories:
  - 1K<n<10K
configs:
  - config_name: functiongemma
    data_files: data/functiongemma_train.jsonl
  - config_name: gemma4
    data_files: data/gemma4_train.jsonl
---

# Wallet tool-calling SFT data

Supervised fine-tuning data that teaches a small local model to turn a
natural-language wallet request into the **exact structured tool call** a macOS
Ethereum wallet can execute:

> *"Send 0.1 ETH to vitalik.eth"*
>
> ```
> executeTx{chainId:"1", to:"0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
>           value:"100000000000000000", function:null, args:[]}
> ```

Getting that right requires resolving an ENS name, selecting the right tool from
five, and converting a human amount to base units. Base-unit arithmetic is the
capability that separates models on this task.

> **The dataset viewer is blank on purpose.** Hugging Face only renders the
> viewer for private datasets on PRO or Enterprise plans, and `ef-dai-team` is on
> the free tier. The `configs` in the card above are correct and the viewer will
> light up if the repo is made public or the org is upgraded. Downloads work
> either way — see *Reproduce* below.

## Two configs, one recipe

| File | Target model | Rows |
| --- | --- | --- |
| `data/functiongemma_train.jsonl` | `unsloth/functiongemma-270m-it` | 1739 |
| `data/gemma4_train.jsonl` | `google/gemma-4-E4B-it` | 1739 |

Same intents, same gold. They differ only in serialisation: FunctionGemma needs
the system turn remapped to the **`developer`** role (that is what activates its
function calling) and emits its own DSL delimiters; Gemma-4 uses `system` and a
different delimiter dialect.

## Row format

```json
{
  "id": "ft-gen-transfer-pos-0137",
  "category": "generated-transfer-pos",
  "protocol": "transfer",
  "messages": [
    {"role": "developer", "content": "<wallet operating manual>"},
    {"role": "user", "content": "Send 0.000002 USDC to vitalik.eth"},
    {"role": "assistant", "content": "<start_function_call>call:executeTx{...}<end_function_call>"}
  ],
  "tools": [ "…tools.json, verbatim…" ],
  "expected_calls": [ "…gold, for validation only — never fed to the model…" ]
}
```

`messages` minus the final assistant turn is *exactly* what the evaluation
harness feeds at inference time, so training and eval see the same prompt shape.
The assistant target is a tool call for gold-call cases, a clarifying question
for ablation (missing-field) cases, or a safety warning for refusals. Every
target decodes and scores back to 1.0 through the harness's unchanged scorer.

`tools.json` at the repo root is the tool schema embedded verbatim in every row:
`executeTx`, `readTx`, `shield`, `unshield`, `swap`.

## Composition

| Category | Rows |
| --- | --- |
| `generated-transfer-pos` | 650 |
| `generated-swap-pos` | 650 |
| `multiturn-*` (amount / recipient / to_token / token) | 250 |
| `ablation-*` (missing field → clarifying question) | 90 |
| `safe-*` (add / remove signer) | 40 |
| `aave-*` (supply / withdraw / borrow / repay) | 55 |
| `safety-refusal-*` (burn address, zero address, unknown spender, unverified token) | 4 |

Weighted toward transfer and swap, which are the core task.

## Disjoint from the eval set — by construction

Training rows are generated from **separate sources under a different seed** to
the 307-case evaluation set: `datasets/finetune_seeds.yaml` rather than
`datasets/seeds.yaml`, and separate protocol fixtures. No training conversation
appears in the eval set, and an integrity test in the harness enforces it.

That is what makes the benchmark numbers below meaningful rather than
memorisation.

## Results using this data

Fine-tuned with LoRA (r=16, α=16) via the Modal jobs in `scripts/`, then
quantised to GGUF and scored on the 307-case eval set:

| Model | Before | After |
| --- | --- | --- |
| FunctionGemma-270M | 8.1% | **8.8%** |
| Gemma-4 E4B | 9.8% | **80.1%** |

**The 270M fine-tune did not work.** Its 8.8% comes entirely from ablation and
refusal cases — categories where the correct answer is *not* to emit a call. On
transfers, swaps and multi-turn it scores 0%. The same data lifts Gemma-4 E4B
from 9.8% to 80.1%, so the binding constraint is model capacity, not the
dataset. Treat the 270M row as a negative result worth knowing, not a model to
deploy.

(An earlier v1 of this dataset had ~90 examples and the 270M fine-tune collapsed
to 0% on the core task. v2 scaled it ~20×. That helped, but not enough.)

## Reproduce

**Regenerate the data.** Both JSONLs are byte-stable outputs of seeded scripts in
the `evals-local-llm` harness — verified: regenerating at the current HEAD
reproduces these files bit for bit.

```bash
uv run python scripts/generate_finetune_data.py            # FunctionGemma-270M
uv run python scripts/generate_gemma4_finetune_data.py     # Gemma-4 E4B
```

Add `--reasoning` to prepend a deterministic `<think>…</think>` block with the
base-unit arithmetic before each call. Off by default; A/B it on the eval set
rather than assuming it helps.

**Train.** Unsloth does not run on macOS, so training happens on a CUDA box. The
Modal jobs used for the runs above are in `scripts/`:

```bash
modal run scripts/modal_finetune.py          # LoRA adapter (270M)
modal run scripts/modal_export.py            # merge + Q8_0 GGUF
modal run scripts/modal_finetune_gemma4.py   # LoRA adapter (E4B)
modal run scripts/modal_export_gemma4.py     # merge + Q4_K_M GGUF
```

`scripts/train_functiongemma.py` is the equivalent single-box / Colab script.

**Score.** Point the harness's local GGUF provider at your export and run the
eval. The eval set is untouched by training, so the number is comparable to the
table above.

## Models trained from this data

- [`ef-dai-team/functiongemma-270m-wallet-ft`](https://huggingface.co/ef-dai-team/functiongemma-270m-wallet-ft)
- [`ef-dai-team/gemma-4-E4B-wallet-ft`](https://huggingface.co/ef-dai-team/gemma-4-E4B-wallet-ft)

Both are playable side by side in the demo Space.

## License

Apache-2.0. Every row is generated deterministically by the seeded scripts in the `evals-local-llm` harness — no model output and no third-party corpus is involved — so the data and the training scripts shipped alongside it are covered by the same permissive license.

Models fine-tuned from it carry their own base-model licenses: `functiongemma-270m-wallet-ft` inherits the Gemma Terms of Use, `gemma-4-E4B-wallet-ft` is Apache-2.0.
