---
title: Wallet Tool-Call Playground
emoji: 🔐
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
python_version: "3.12"
short_description: Local SLMs turning wallet requests into tool calls
startup_duration_timeout: 1h
models:
  - ef-dai-team/functiongemma-270m-wallet-ft
  - ef-dai-team/gemma-4-E4B-wallet-ft
  - unsloth/functiongemma-270m-it-GGUF
---

# Wallet tool-call playground

Does a small, local language model turn *"send 0.1 ETH to vitalik.eth"* into the
exact structured call a macOS Ethereum wallet can execute?

```json
[{"name": "executeTx",
  "arguments": {"chainId": "1",
                "to": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
                "value": "100000000000000000",
                "function": null, "args": []}}]
```

Getting that right means resolving an ENS name, picking the right tool, and
converting a human amount to base units — the last of which is the capability
that separates models.

## What's in here

| Tab | What it does |
| --- | --- |
| **Playground** | Free-form request → decoded tool call, raw model output, and the exact prompt sent. |
| **Scored eval cases** | Replay a real eval case and score it with the harness's own deterministic binary scorer. |
| **Benchmark** | The frozen scoreboard from real promptfoo runs — read this one. |
| **Fine-tune it yourself** | The full recipe: dataset, generators, training jobs. |

## Honest result

The 270M fine-tune this Space showcases (`ef-dai-team/functiongemma-270m-wallet-ft`)
scores **8.8%** on the 307-case eval set, against **8.1%** for the untuned base —
and **0% on every category that requires emitting a tool call**. All of its
passing cases are ones where the correct answer is *not* to call a tool.

The same data and recipe applied to `gemma-4-E4B` reaches **80.1%**. The binding
constraint is model capacity, not the dataset. Both models are selectable in the
playground so the gap is visible live rather than asserted.

## Fidelity to the harness

`prompt.py`, `tools.json`, `wallet_evals/` and `scoring.py` are copied verbatim
from the [`evals-local-llm`](https://github.com/Ethereum-dAI) harness, and the
models are served by the same `llama-cpp-python` code path the eval uses. A case
that passes here passes there.

Regenerate the frozen data with `uv run python space/build_data.py` from the
harness repo root.

## Training data

Published as a separate dataset repo — see the **Fine-tune it yourself** tab.
It is disjoint from the eval set by construction, which is what makes the
scoreboard meaningful.
