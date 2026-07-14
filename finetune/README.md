# Fine-tuning FunctionGemma-270m (Modal → HuggingFace → promptfoo)

End-to-end pipeline for fine-tuning the local wallet model and evaluating it with
the same discriminator harness the hosted models run through. Everything here is
**additive** — the scorer, `pf/tools.json`, `pf/assert.py`, and the datasets are
unchanged; a fine-tuned model is just another provider.

```
datasets/finetune_seeds.yaml ─┐
protocols/*.finetune.fixtures ─┤ scripts/generate_finetune_data.py
                               └─▶ data_for_finetune/functiongemma_train.jsonl  (gitignored, ~1740 rows)
                                        │
                     Modal T4 GPU  ─────┤ finetune/modal_finetune.py     (LoRA train → adapter + GGUF)
                                        ├ finetune/modal_diagnose.py     (isolate HF vs GGUF issues)
                                        ├ finetune/modal_eval.py         (score adapter on the eval set)
                                        └ finetune/modal_export.py       (bf16 merge → GGUF q8_0 → HF upload)
                                                 │
                       huggingface.co/gabrielfior/functiongemma-270m-wallet-ft  (public: GGUF + adapter)
                                                 │
                    promptfooconfig.functiongemma*.yaml → pf/provider_functiongemma.py (llama-cpp)
```

The training set is a **disjoint synthetic** set (same intent builders as the eval,
different seed/amounts), so no training conversation leaks into the eval — enforced
by `tests/test_finetune_integrity.py`. See `data_for_finetune/README.md` for the
dataset format and `finetune/README_colab.md` for the Colab alternative.

## Prerequisites

- **Modal** account + CLI: `uv tool install modal` then `modal token new` (writes
  `~/.modal.toml`). We invoke it as `uv run --with modal modal ...`.
- **HuggingFace** token with **write** scope for uploads: `hf auth login` (writes
  `~/.cache/huggingface/token`).
- Local eval only needs `llama-cpp-python`: `uv sync --group local`.

## Why Modal (not Colab)

Colab free-tier reclaims idle GPU sessions minutes after a job ends and drops the
`exec` websocket mid-run — we lost trained GGUFs twice. Modal is scripted,
reproducible, and uses **persistent Volumes** so the base weights download once:

- `functiongemma-hf-cache` → mounted at `/root/.cache/huggingface` (`HF_HOME`).
  The base model is fetched on run #1 and reused after.
- `functiongemma-ft-outputs` → mounted at `/outputs`. Checkpoints, merged model,
  and GGUF persist here across runs and can be pulled with `modal volume get`.

Each script is a `modal.App` with one GPU function; run with `modal run`. The
functions self-install unsloth in the image (`.pip_install("unsloth")`).

### Two Modal gotchas we hit (and fixed)

1. **Never read local files at module top-level.** Modal re-imports the script
   *inside the container* to hydrate the function, where your local files don't
   exist. Read the HF token only in `@app.local_entrypoint()` and pass it as a
   function argument (see `modal_export.py`).
2. **Unsloth's GGUF export builds llama.cpp** and, if system packages are missing,
   prompts on stdin → `EOFError` in the non-interactive container. Pre-install
   `build-essential cmake libssl-dev libcurl4-openssl-dev curl` in the image.

## Run the pipeline

```bash
# 1. generate the disjoint training set (local, deterministic)
uv run python scripts/generate_finetune_data.py

# 2. train (LoRA) on a Modal T4 → adapter + (unsloth) GGUF in the outputs volume
uv run --with modal modal run finetune/modal_finetune.py

# 3. score the fine-tuned ADAPTER on pf/tests.generated.yaml (returns pass rates)
uv run --with modal modal run finetune/modal_eval.py

# 4. export a correct GGUF and upload it (+ adapter) to HuggingFace
uv run --with modal modal run finetune/modal_export.py
```

### The gemma3 merge trap (important)

Training converges fine, but **unsloth's LoRA→16-bit merge corrupts gemma3**
(fp16 overflow — you'll see `Using float16 precision for gemma3 won't work`), so
the merged model and its GGUF emit token-garbage while the **unmerged adapter is
correct**. Always validate the unmerged adapter (`modal_diagnose.py`) before
trusting a merged/GGUF artifact. `modal_export.py` sidesteps it by merging in
**bf16** with plain `peft` (`merge_and_unload()`), converting to GGUF q8_0 via
`llama.cpp/convert_hf_to_gguf.py`, and **sanity-gating** on a real tool-call
generation before it uploads.

## HuggingFace upload

`modal_export.py` uploads from the Modal container using `HfApi(token=...)`:

- Repo: **[gabrielfior/functiongemma-270m-wallet-ft](https://huggingface.co/gabrielfior/functiongemma-270m-wallet-ft)** (public)
- `functiongemma-270m-wallet-ft.Q8_0.gguf` — deployable, ~292 MB
- `adapter/` — the LoRA adapter (base = `unsloth/functiongemma-270m-it`)

The token is read from `~/.cache/huggingface/token` locally and passed to the
remote function (never committed, never printed). Read-only tokens 403 on upload.

## How promptfoo calls the HuggingFace model

`pf/provider_functiongemma.py` is a promptfoo Python provider serving a **local
GGUF in-process via `llama-cpp-python`** (FunctionGemma isn't on OpenRouter). It
accepts either source in the config:

```yaml
providers:
  - id: file://pf/provider_functiongemma.py:call_api
    label: functiongemma-ft
    config:
      repo_id: gabrielfior/functiongemma-270m-wallet-ft   # pull from HF (Llama.from_pretrained)
      filename: "*.Q8_0.gguf"                             # glob within the repo
      # model_path: models/foo.Q8_0.gguf                  # …or a local file instead
      n_ctx: 4096
      temperature: 0.2
      max_tokens: 1024
```

`repo_id`+`filename` → `Llama.from_pretrained(repo_id, filename)` downloads and
caches the GGUF from HF (no token needed for public repos). The provider then:
remaps `system`→`developer` (the role FunctionGemma needs for tool-calling),
calls `create_chat_completion(messages, tools=pf/tools.json)`, and translates the
model's plain-text Gemma DSL back into the OpenAI-shaped tool calls that the
unchanged `pf/assert.py` scores. Model loads once per worker and is reused.

Run the eval (base vs fine-tuned, or fine-tuned only), local, no API cost:

```bash
uv sync --group local   # once — installs llama-cpp-python
scripts/eval.sh -c promptfooconfig.functiongemma.yaml    -o functiongemma.out.json   # base + ft
scripts/eval.sh -c promptfooconfig.functiongemma-ft.yaml -o functiongemma.ft.out.json # ft only
```

CPU inference of a 270M Q8_0 works but is slow (~15–25 min for 307 cases); a fast,
GPU-batched alternative that scores the adapter directly is `modal_eval.py`.

## Current results

Fine-tuned adapter on `pf/tests.generated.yaml` (307 cases, `wallet_evals` scorer):

| Slice | Pass rate |
|---|---|
| ablation (ask when info missing) | 96% |
| generated-swap-pos | 36% |
| generated-transfer-pos | 30% |
| multi-turn | 25% |
| safety-refusal | ~14% ⚠️ (under-represented in training) |
| **overall** | **36.5%** |

Known limitations / next levers: refusals regressed (rebalance the training mix);
exact base-unit arithmetic + 42-char address reproduction is the 270M's ceiling
(the harness's main discriminator); base-270m baseline not yet measured, so the
absolute lift over stock FunctionGemma is unquantified.
