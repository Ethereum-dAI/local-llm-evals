---
title: Wallet Tool-Calling Eval
emoji: 🧮
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: 307 wallet requests, 5 models, exact-match scoring
models:
  - ef-dai-team/qwen3-8b-wallet-ft
  - ef-dai-team/gemma-4-E4B-wallet-ft
datasets:
  - ef-dai-team/wallet-tool-calling-ft
---

# Exact or nothing

An internal report on whether a small local model can turn a natural-language
wallet request into the byte-exact tool call a macOS Ethereum wallet would
execute. 307 cases, 5 models, deterministic binary scoring.

The headline: **fine-tuning works, and the base model sets the ceiling.** The
same 1739 synthetic rows and the same LoRA recipe take Gemma-4 E4B from 12.7% to
80.1% and Qwen3-8B from 41.0% to 86.0%, against a 95.8% gpt-5 anchor. The weaker
base gains far more and still finishes behind.

## What you're looking at

`index.html` renders one tick per case per model, in dataset order — filled for
an exact match, hollow for anything else. Cases are grouped transfer → swap →
multi-turn → ablation → refusal, so a model that only ever passes by staying
silent shows up as a cluster at the far right — 35 of the 307 cases want no tool
call, which is why a headline percentage alone cannot be read as capability.

Hover a tick to read the case; click it to open the full record below, where
every model's recorded output appears verbatim next to the scorer's verdict.

## One run vintage

Every column comes from the 2026-08-10/11 relaunch — the first runs made after
the tool list in the request grew from 3 tools to 5. A model shown 5 tools is
answering a different question from one shown 3, so earlier runs (gpt-4o-mini,
Gemma-4 26B-A4B, and both FunctionGemma-270M columns) are not shown here rather
than being silently mixed in. Bringing them back means re-running them.

## Provenance

`data.json` is built from the promptfoo run artefacts in the harness repo:

```bash
uv run python space/build_static.py
```

Nothing on this page is recomputed at view time and nothing is estimated. Each
output is the text a model actually produced during a real run, and each verdict
comes from the same deterministic scorer that produced the strips.

## Why this is static

Hosting a Gradio Space requires a Team plan on `ef-dai-team` (verified: the API
returns 402 for both `cpu-basic` and `zero-a10g`). The interactive playground —
live inference over the local GGUFs with the same prompt, tools and scorer — is
written and lives in `space/` in the harness repo. Assemble it with
`uv run python space/stage.py gradio`, run it with `python app.py` from
`space/build/gradio`, and it deploys unchanged once the org is upgraded.
