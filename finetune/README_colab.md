# Step 4 (smoke): fine-tune FunctionGemma on 5 examples, from your terminal

Unsloth needs a CUDA GPU, so we drive a remote Colab runtime with Google's
official [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli).
`finetune/smoke_finetune.py` trains on the **first 5** rows of
`data_for_finetune/functiongemma_train.jsonl` — just enough to prove the pipeline
runs end-to-end before we scale it up.

## Run it (all from the repo root, on your Mac)

```bash
# 0. install the CLI once
uv tool install google-colab-cli

# 1. authenticate. The CLI's default strategy is `adc` (Application Default
#    Credentials), so set those up once with gcloud:
gcloud auth application-default login
#    (Alternative: `--auth oauth2` with an OAuth client config at
#    ~/.colab-cli-oauth-config.json via `-c PATH`. `colab auth -s NAME` is a
#    different thing — it authenticates the *VM* for GCP services, not the CLI.)

# 2. provision a GPU session (T4 is plenty for a 270M model)
colab new -s gemma-ft --gpu T4

# 3. install Unsloth on the runtime (pulls torch/transformers/trl/peft/bitsandbytes)
colab install -s gemma-ft unsloth

# 4. upload the training data (LOCAL -> REMOTE path, keep the same filename)
colab upload -s gemma-ft data_for_finetune/functiongemma_train.jsonl functiongemma_train.jsonl

# 5. run the smoke fine-tune (ships the script and executes it on the GPU)
colab exec -s gemma-ft -f finetune/smoke_finetune.py

# 6. (optional) pull the LoRA adapter back
colab exec -s gemma-ft -f - <<'PY'
import shutil; shutil.make_archive("lora", "gztar", "outputs/lora")
PY
colab download -s gemma-ft lora.tar.gz ./lora.tar.gz
```

## What "success" looks like

The exec output should show, in order:
- `[smoke] training on 5 examples: [...]`
- one rendered training example containing the `<start_of_turn>developer` framing,
  the tools, and the `<start_function_call>...<end_function_call>` target;
- a `final training loss:` line (any finite number — it should drop across steps);
- a `[smoke] ---- probe generation ----` block (garbled after 30 steps is fine —
  we're testing the *plumbing*, not quality);
- `[smoke] SMOKE OK`.

## If it breaks

The likely failure points are exactly what this smoke run exists to surface:
- **`apply_chat_template(..., tools=...)`** rejecting the args, or the `assert
  target in text` firing → the FunctionGemma template renders assistant tool calls
  differently than "raw DSL as content"; we'll adjust the encoder/renderer.
- **TRL/Unsloth version drift** on `SFTConfig`/`SFTTrainer` kwargs (e.g.
  `tokenizer=` vs `processing_class=`, `dataset_text_field` location).

Paste the traceback back to me and I'll do the expansion pass:
`LIMIT=None`, real epoch count, `train_on_responses_only` (mask the prompt),
`EXPORT_GGUF=True`, then wire the GGUF into `promptfooconfig.yaml` and re-run
`scripts/eval.sh` to measure lift.
