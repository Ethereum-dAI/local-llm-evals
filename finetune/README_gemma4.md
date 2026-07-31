# Fine-tuning Gemma 4 E4B (Modal → HuggingFace → promptfoo)

Fine-tunes **the exact model local-wallet-mac ships on-device** — Gemma 4 E4B —
and evaluates it with the same discriminator harness the hosted models run
through. Everything is **additive**: the scorer, `pf/tools.json`, `pf/assert.py`,
the `GEMMA4` DSL dialect, and the datasets are unchanged; the fine-tuned model is
just another local provider. This mirrors the FunctionGemma pipeline
(`finetune/README.md`) — same shape, bigger model, Gemma-4 dialect.

```
(same disjoint sources + seed as FunctionGemma)
scripts/generate_gemma4_finetune_data.py
   └─▶ data_for_finetune/gemma4_train.jsonl  (gitignored, 1739 rows, GEMMA4 dialect, <think> traces)
            │
   Modal A100  ──┤ finetune/modal_finetune_gemma4.py  (LoRA train → adapter @ /outputs/adapter)
                 ├ finetune/modal_eval_gemma4.py       (score adapter on the eval set — fast checkpoint)
                 └ finetune/modal_export_gemma4.py     (bf16 merge → f16 GGUF → llama-quantize Q4_K_M → HF)
                          │
        huggingface.co/gabrielfior/gemma-4-E4B-wallet-ft  (public: Q4_K_M GGUF + adapter)
                          │
        promptfooconfig.gemma4-ft.yaml → pf/provider_functiongemma.py (llama-cpp-python)
```

## What model, exactly

The wallet content-pins its GGUF by SHA256 (`OnboardingSettingsStore.swift`,
`scripts/package-macos-demo.sh`):

| | value |
|---|---|
| Deployed GGUF | `ggml-org/gemma-4-E4B-it-GGUF` → `gemma-4-E4B-it-Q4_K_M.gguf` (5.34 GB) |
| SHA256 pin | `90ce98129eb3e8cc57e62433d500c97c624b1e3af1fcc85dd3b55ad7e0313e9f` |
| HF revision | **`1762c8e8713f`** — Q4_K_M was deleted from `main` on 2026-07-16; this pinned commit still carries the byte-identical file |
| Upstream weights | `google/gemma-4-E4B-it` (gated) — we train from `unsloth/gemma-4-E4B-it`, the ungated mirror of the same weights |

The base provider in `promptfooconfig.gemma4-ft.yaml` pulls that exact pinned file
(`revision: 1762c8e8713f`), so the benchmark's baseline **is** the wallet's model,
byte-for-byte. The fine-tuned side is quantized to the same **Q4_K_M** so the
comparison is apples-to-apples at the quant the wallet runs.

## Dataset — the same one FunctionGemma used

`scripts/generate_gemma4_finetune_data.py` reuses FunctionGemma's own
`_collect`/`_select`/`_reasoning_text` under the same `SEED`, so the two training
sets share a single source of truth (identical IDs, user turns, gold, tools,
distribution — verified). Only the **encoding** differs:

- targets use the **GEMMA4** dialect `<|tool_call>call:NAME{key:<|"|>v<|"|>,…}<tool_call|>`
  (the shape the wallet's `Gemma4FallbackParser` reads), not FunctionGemma's `<start_function_call>…<escape>`;
- the **`system`** role is kept (Gemma-4's template has a native `<|turn>system`);
  FunctionGemma remapped it to `developer`;
- **`<think>` base-unit-arithmetic traces are ON** by default (Gemma-4 E4B is
  thinking-capable; arithmetic is the eval's discriminator). The eval decoder
  strips `<think>…</think>` before scoring.

Disjoint from the eval set (no leakage) and every target self-scores to 1 through
the unchanged scorer — enforced by `tests/test_gemma4_finetune_integrity.py`.

## Prerequisites

Same as FunctionGemma (`finetune/README.md`): a Modal account (`~/.modal.toml`), a
**write**-scoped HF token (`~/.cache/huggingface/token`), and `uv sync --group
local` for local eval. Persistent Modal Volumes: `gemma4-hf-cache` (base weights,
downloaded once) and `gemma4-ft-outputs` (the adapter + GGUF).

## Run the pipeline

```bash
# 1. generate the training set (local, deterministic; same data as FunctionGemma)
uv run python scripts/generate_gemma4_finetune_data.py

# 2. LoRA-train on a Modal A100 → adapter at gemma4-ft-outputs:/outputs/adapter
uv run --with modal modal run finetune/modal_finetune_gemma4.py

# 3. score the ADAPTER on pf/tests.generated.yaml (fast GPU checkpoint / fallback)
uv run --with modal modal run finetune/modal_eval_gemma4.py

# 4. export a correct Q4_K_M GGUF and upload it (+ adapter) to HuggingFace
uv run --with modal modal run finetune/modal_export_gemma4.py

# 5. base (pinned wallet GGUF) vs fine-tuned, local, no API cost
uv sync --group local
scripts/eval.sh -c promptfooconfig.gemma4-ft.yaml -o gemma4.ft.out.json
```

### Differences from the FunctionGemma scripts

- **GPU:** A100-40GB (E4B LoRA ≈17 GB VRAM — a T4 is too small).
- **Model class:** unsloth `FastModel` (E4B is a multimodal-capable checkpoint),
  not `FastLanguageModel`.
- **Response masking:** Gemma-4 turn markers `<|turn>user\n` / `<|turn>model\n`
  (verified against the rendered template). The trainer aborts if >50 % of rows
  end up fully masked — a wrong-marker guard so a no-op run fails fast.
- **No in-training GGUF.** unsloth's LoRA→16-bit merge corrupts gemma3-family
  weights (the "merge trap", see `finetune/README.md`). Training only saves the
  adapter; the correct GGUF comes from the bf16 export.
- **Q4_K_M export is two steps.** `convert_hf_to_gguf.py` can't emit k-quants
  directly, so `modal_export_gemma4.py` converts to an f16 GGUF, then runs
  `llama-quantize … Q4_K_M`. It sanity-gates on a real tool-call generation before
  uploading.

## How promptfoo calls the pinned base + the fine-tune

`pf/provider_functiongemma.py` is the single local-GGUF provider (via
`llama-cpp-python`) shared with FunctionGemma; `config.dialect: gemma4` +
`config.system_role: system` select this model's conventions. When `revision` is
set it resolves the exact file with `hf_hub_download` (llama's `from_pretrained`
globs `main`, where the wallet's Q4_K_M no longer exists). It calls
`create_chat_completion(messages, tools=…)` and translates the plain-text GEMMA4
DSL (stripping `<think>`) — or native `tool_calls`, if the chat template surfaces
them — into the OpenAI-shaped calls the unchanged `pf/assert.py` scores.

## Current results

**Deployment-faithful checkpoint** — base vs fine-tuned, both as **Q4_K_M GGUF**
(the exact quant the wallet ships), scored via `promptfooconfig.gemma4-ft.yaml`
(llama-cpp-python, `pf/tests.generated.yaml`, 307 cases, `wallet_evals` scorer).
Base = the SHA-pinned stock wallet GGUF; fine-tuned = our upload.

| category | base (stock wallet) | fine-tuned |
|---|---|---|
| generated-transfer-pos | 9/101 | 67/101 |
| generated-swap-pos | 4/96 | 80/96 |
| multiturn-amount | 4/20 | 20/20 |
| multiturn-to_token | 2/22 | 19/22 |
| multiturn-recipient | 0/21 | 13/21 |
| multiturn-token | 0/12 | 6/12 |
| ablation (ask when info missing) | 9/28 | 26/28 |
| safety-refusal (4 kinds) | 2/7 | 5/7 |
| **overall** | **30/307 = 9.8%** | **236/307 = 76.9%** |

**A ~7.8× lift (9.8% → 76.9%)** from fine-tuning the on-device model on its own
tool-call DSL. The stock E4B mostly gets the format but botches exact base-unit
arithmetic / 42-char addresses (the harness's discriminators) and doesn't refuse
dangerous sends; the fine-tune fixes transfer/swap/multi-turn wholesale and adds
refusals (the one gap: `unverified-token-swap`, 0/2 both — it still swaps into a
raw unknown token address). ~14 cases per side hit the 5-min llama-cpp worker
timeout and count as fails; the gap dwarfs that noise.

> **The 76.9% understates the fine-tune.** A re-run on 2026-07-30 hit **zero**
> worker timeouts and scored **246/307 = 80.1%** — same GGUF, same prompt, same
> scorer. The 13 timeouts in the run above were counted as failures, so treat
> **80.1%** as the current figure and 76.9% as timeout-depressed.

Training: LoRA r=16, 3 epochs, lr 2e-4, A100-40GB, final train loss 0.0078,
0/1739 rows masked (turn markers verified). The GPU-scored **adapter** (bf16, via
`modal_eval_gemma4.py`) reads 190/307 = 61.9% — lower only because that eval caps
generation at 256 tokens, truncating some `<think>`+call outputs; the Q4_K_M
promptfoo run (1024-token budget) is the truer figure.

## Same-day head-to-head vs gpt-5 (2026-07-30)

Both halves run fresh on the same day, same prompt, same 307 cases, `--no-cache`.
Split into two evals on purpose: gpt-5 is network-bound and parallelises at `-j 8`,
the local GGUF is CPU-bound and must stay at `-j 1` (the llama-cpp worker is not
reentrant). In one serialized eval each half waited on the other — gpt-5's average
latency was 37 s there vs 11 s apart. `scripts/merge_evals.py` stitches the two
exports back into a single eval so the UI shows a real side-by-side.

```bash
scripts/eval.sh -c promptfooconfig.gpt5-only.yaml     -j 8 -o gpt5.fresh.out.json
scripts/eval.sh -c promptfooconfig.gemma4ft-only.yaml -j 1 -o gemma4ft.fresh.out.json
uv run python scripts/merge_evals.py -o merged.out.json gpt5.fresh.out.json gemma4ft.fresh.out.json
npx promptfoo import merged.out.json --new-id
```

| category | gpt-5 | fine-tuned E4B (on-device) |
|---|---|---|
| generated-transfer-pos | 95/101 | 72/101 |
| generated-swap-pos | 88/96 | 80/96 |
| multi-turn (4 kinds) | 75/75 | 61/75 |
| ablation (ask when info missing) | 28/28 | 28/28 |
| safety-refusal (4 kinds) | 7/7 | 5/7 |
| **overall** | **293/307 = 95.4%** | **246/307 = 80.1%** |

Cost $3.26, ~1 h wall-clock (both concurrent). Every result re-scores through
`pf/assert.py` with 0 mismatches against promptfoo's own verdicts.

**The ~15-point gap is almost entirely base-unit arithmetic.** 50 of the
fine-tune's 61 failures are amount errors (48 off by a power of 10, 2
digit-transposition) vs 4 of gpt-5's 14. Discount amount errors on both sides and
the two converge to **96.4% vs 96.7% — a one-case gap**. Against the *stock*
model on the same basis the lift still reads **49.5% → 92.2%**, so fine-tuning's
value (DSL compliance, arg completeness, refusals) does not depend on arithmetic.

Failure taxonomy, fine-tuned (61): 48 amount off-by-10^k · 7 malformed address
length · 2 digit-transposition · 2 acted-when-must-refuse · 1 wrong tool · 1 wrong
`currencyIn`. The amount errors are zero-counting, not token-decimals confusion —
12 of 13 token groupings involve only 18-decimal tokens, and the direction is
near-symmetric (26 overshoot / 24 undershoot). **The 7 malformed addresses are the
dangerous class**: all are length errors (5×41 hex, 2×39), which would produce an
unsendable or wrong-recipient transaction. A `^0x[0-9a-fA-F]{40}$` check at the
tool boundary catches 100% of them.

Failure taxonomy, gpt-5 (14): **10 are over-refusal, not capability** — it answers
"I'm sorry, but I cannot assist with that request" to heavily case-mangled asks
(`hOnEstLy eTh haS JUSt been siTTinG…`), i.e. the surface mutators read as
adversarial obfuscation. Some are arguably *correct* refusals, because a typo'd
**verb** destroys intent (`send 0.000003 USDC to becmoe WETH`). Token symbols are
already typo-protected; the action verb is not. So the anchor's ceiling is
suppressed here and the true frontier gap is wider than 15 points.

gpt-5 also re-scores **96.7% (Jul 9) vs 95.4% (Jul 30)** on a byte-identical prompt
and dataset — ±1.3 pts of run-to-run nondeterminism at `temperature: 0.1`. Don't
read small deltas against the anchor as regressions.

**Uploaded:** [`gabrielfior/gemma-4-E4B-wallet-ft`](https://huggingface.co/gabrielfior/gemma-4-E4B-wallet-ft)
(public) — `gemma-4-E4B-wallet-ft.Q4_K_M.gguf` (5.34 GB, same footprint as the
stock wallet model) + `adapter/`. The bf16 merge sanity-gated correctly (emits
`<think>` + a well-formed call with right base-unit arithmetic).
