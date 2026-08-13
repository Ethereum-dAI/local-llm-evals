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
  - ef-dai-team/qwen3-8b-wallet-ft
  - ef-dai-team/gemma-4-E4B-wallet-ft
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

The same 1739 synthetic rows and the same LoRA recipe were applied to two bases.
`gemma-4-E4B` goes from **12.7% to 80.1%** on the 307-case eval set;
`Qwen3-8B` goes from **41.0% to 86.0%**, against a **95.8%** gpt-5 anchor. The
weaker base gains far more (+67.4 pp vs +45.0 pp) and still finishes behind — so
the dataset is not the binding constraint, the capability the base brought with
it is.

Both fine-tunes are selectable in the playground, so the remaining gap to the
anchor is visible live rather than asserted. Both are also still weak in the same
place: safety refusals, 5 of 7, from roughly one training example per refusal
category.

## Fidelity to the harness

`prompt.py`, `tools.json`, `wallet_evals/` and `scoring.py` are not written for
this Space — they are the [`evals-local-llm`](https://github.com/Ethereum-dAI)
harness's own files, copied in at build time by `space/stage.py` from a single
source each (`pf/prompt.py`, `pf/tools.json`, `pf/assert.py`,
`src/wallet_evals/`). There is no second copy to drift, and a test enforces it.
Models are served by the same `llama-cpp-python` code path the eval uses, so a
case that passes here passes there.

To run it from the harness repo root:

```bash
uv run python space/build_data.py            # refresh the frozen scoreboard
uv run python space/stage.py gradio          # assemble space/build/gradio
cd space/build/gradio && python app.py
```

## Training data

Published as a separate dataset repo — see the **Fine-tune it yourself** tab.
It is disjoint from the eval set by construction, which is what makes the
scoreboard meaningful.
