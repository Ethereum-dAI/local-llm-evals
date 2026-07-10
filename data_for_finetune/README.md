# FunctionGemma fine-tuning data

`functiongemma_train.jsonl` — a **disjoint** synthetic training set for
fine-tuning `unsloth/functiongemma-270m-it`, generated from the same intent
builders as the eval harness but from separate sources under a different seed, so
**no training conversation appears in the eval set** (`tests/test_finetune_integrity.py`
enforces this). 88 examples ≈ 20% of the eval set (447), stratified across
transfer / swap / multi-turn / ablation / Safe / Aave / refusal.

## Regenerate

```bash
uv run python scripts/generate_finetune_data.py              # plain targets
uv run python scripts/generate_finetune_data.py --reasoning  # <think> arithmetic trace variant
```

Sources (all disjoint from the eval inputs):
- `datasets/finetune_seeds.yaml` — transfer/swap intents (different amounts/recipients)
- `datasets/protocols/{safe,aave}.finetune.fixtures.json` — protocol fixtures
- refusal scenarios: `REFUSAL_SCENARIOS` in `scripts/generate_finetune_data.py`

## Row format

One JSON object per line:

```json
{
  "id": "ft-gen-transfer-pos-0137",
  "category": "generated-transfer-pos",
  "protocol": "transfer",
  "messages": [
    {"role": "developer", "content": "<system/operating manual>"},
    {"role": "user", "content": "Send 0.000002 USDC to vitalik.eth"},
    {"role": "assistant", "content": "<start_function_call>call:executeTx{...}<end_function_call>"}
  ],
  "tools": [ /* pf/tools.json, verbatim */ ],
  "expected_calls": [ /* gold, for validation/debugging — not fed to the model */ ]
}
```

- `system` roles are remapped to **`developer`** — the role FunctionGemma needs to
  activate function calling (matches `wallet_evals.functiongemma.decode_prompt`,
  the inference path). So `messages` (minus the final assistant turn) is exactly
  what the provider feeds at eval time.
- The assistant target is the FunctionGemma **DSL** call for gold-call cases, a
  clarifying question for ablation (missing-field) cases, or a safety warning for
  refusals. Every target decodes+scores back to 1.0 through the unchanged scorer.
- `--reasoning` prepends a deterministic `<think>…</think>` block with the
  base-unit arithmetic before the call (Unsloth's "reason before tool calling"
  style). Off by default — see the notes below.

## Training + export (deferred — needs a CUDA GPU)

Unsloth doesn't run on macOS; run steps 4–5 on Colab or a Linux GPU box.

1. Load `unsloth/functiongemma-270m-it` with `FastLanguageModel`, add LoRA.
2. Map each row to a training string. Prefer letting the tokenizer render the
   template so it matches what the model was trained with, then assert it agrees
   with our validated DSL before training:
   ```python
   text = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
   assert row["messages"][-1]["content"] in text   # our target survives templating
   ```
3. `SFTTrainer` over the `text` field (LoRA, a few epochs at this scale).
4. Export GGUF `Q8_0` and point the existing provider at the local file:
   ```yaml
   # promptfooconfig.yaml
   config:
     model_path: /path/to/functiongemma-270m-ft.Q8_0.gguf
   ```
5. Measure lift: `scripts/eval.sh` against the fine-tuned GGUF vs. the base
   FunctionGemma baseline. The eval set is untouched, so the number is honest.

## Reasoning: on or off?

The base 270M model is weak at multi-turn tool calling (~10–39% call-equivalence
in reported benchmarks) and fine-tuning lifts it well above 90%. Chain-of-thought
("reason before tool calling") specifically helps small models on the arithmetic
step — which is exactly this task's capability separator (base-unit conversion).

Recommendation: **train the plain variant first** (simplest, matches how the
scorer already reads output — the decoder ignores any `<think>` prefix anyway),
then A/B it against the `--reasoning` variant on the eval set and let the
discriminator decide. Reasoning costs output tokens/latency, which matters for a
270M edge model, so only keep it if the eval shows a real lift.
