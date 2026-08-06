---
title: Wallet Tool-Calling Eval
emoji: 🧮
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: 307 wallet requests, 7 models, exact-match scoring
models:
  - ef-dai-team/functiongemma-270m-wallet-ft
  - ef-dai-team/gemma-4-E4B-wallet-ft
datasets:
  - ef-dai-team/wallet-tool-calling-ft
---

# Exact or nothing

An internal report on whether a small local model can turn a natural-language
wallet request into the byte-exact tool call a macOS Ethereum wallet would
execute. 307 cases, 7 models, deterministic binary scoring.

The headline: **the FunctionGemma-270M fine-tune did not work.** It scores 8.8%
against 8.1% for the untuned base, and 0% on every category that requires
emitting a call. The same data and recipe take Gemma-4 E4B from 9.8% to 80.1%.

## What you're looking at

`index.html` renders one tick per case per model, in dataset order — filled for
an exact match, hollow for anything else. Cases are grouped transfer → swap →
multi-turn → ablation → refusal, so a model that only ever passes by staying
silent shows up as a cluster at the far right. That is exactly what the 270M
fine-tune does.

Hover a tick to read the case; click it to open the full record below, where
every model's recorded output appears verbatim next to the scorer's verdict.

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
live inference over all three local GGUFs with the same prompt, tools and
scorer — is written and lives in `space/` in the harness repo. It runs locally
with `python app.py` and deploys unchanged once the org is upgraded.
