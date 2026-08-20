---
title: Wallet Tool-Call Benchmark
emoji: 🧮
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: 1000 wallet requests, 7 configurations, exact-match scoring
models:
  - ef-dai-team/gemma-4-E4B-wallet-ft-v5
datasets:
  - ef-dai-team/wallet-eval-benchmark
---

# Wallet tool-call benchmark

1000 natural-language requests, scored on whether a model emits the exact
`transfer`/`swap` call a macOS Ethereum wallet would execute. The contract is the
app's own: human decimal amounts (`50.25 USDC`, not base units), a token symbol,
and the recipient copied verbatim. A case scores 1 only if every field of every
call equals gold.

571 cases are multi-round conversations (2–6 turns) testing which turn's value
should win. 80 are an arithmetic slice. 120 have no correct call at all, 49 of them
adversarial refusals.

| Configuration | Overall | Wants a call (880) | No call (120) | Refusals (49) |
| --- | --: | --: | --: | --: |
| wallet-ft — shipped until 2026-08-20 | 68.4% | 69.8% | 58.3% | 49.0% |
| Gemma-4 E4B base | 90.3% | 91.7% | 80.0% | 61.2% |
| Gemma-4 E4B base + clause | 91.0% | 90.6% | 94.2% | 91.8% |
| gpt-5 | 92.8% | 94.1% | 83.3% | 69.4% |
| gpt-5 + clause | 93.7% | 93.5% | 95.0% | **95.9%** |
| ft-v5 | 94.9% | **95.2%** | 92.5% | 81.6% |
| **ft-v5 + clause** | **95.1%** | 94.7% | **98.3%** | **95.9%** |

The `+ clause` rows append a 1577-character refusal contract to the system turn;
nothing else differs.

Three results:

- **The fine-tune the wallet shipped scored 21.7 points below the untuned base.**
  Its card claimed 80.1%, measured on a retired benchmark whose amounts were base
  units. The contract moved to human decimals and nobody re-measured.
- **ft-v5's gain is one bucket:** "expected a call, produced none" went 45 → 1. An
  act-not-ask clause, retry-on-no-call and few-shot exemplars all failed to move
  it; gpt-5 has 51.
- **The refusal win is the prompt, not the fine-tune.** ft-v5's 47/49 against
  gpt-5's 34/49 looked like a model difference. Given the same clause, gpt-5 also
  scores 47/49.

**Discard differences under about six cases.** Three runs of the same ft-v5 GGUF on
different rented GPUs scored 949, 952 and 944. The three clause-off columns come
from one run so that comparison is device-controlled; each clause-on column comes
from its own paired A/B with both arms on one GPU.

## Contents

`index.html` renders one mark per case per model — filled for an exact match. Click
a mark to read the case and every model's recorded output next to the scorer's
verdict. Nothing is recomputed at view time. `index.standalone.html` is the same
report with the data inlined, for handing over as one file.

`data.json` is built from the promptfoo run artefacts in the harness repo:

```bash
uv run python space/build_static.py
```

## Reproduce

The benchmark is public: [`ef-dai-team/wallet-eval-benchmark`](https://huggingface.co/datasets/ef-dai-team/wallet-eval-benchmark)
ships all 1000 cases plus the scorer, prompt and tool schemas. So is the model:
[`ef-dai-team/gemma-4-E4B-wallet-ft-v5`](https://huggingface.co/ef-dai-team/gemma-4-E4B-wallet-ft-v5).

The training rows are not published, which is what makes a score here comparable.
If you train on this benchmark, say so.

## No live demo

A Gradio Space needs a Team plan on this org (the API returns 402 for both
`cpu-basic` and `zero-a10g`). The playground — live inference over the same GGUFs,
prompt and scorer — is written and deploys unchanged once that exists.
