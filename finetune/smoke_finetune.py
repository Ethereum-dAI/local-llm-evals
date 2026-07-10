"""Smoke-test fine-tune of FunctionGemma-270m on 5 examples — runs on a Colab GPU.

PURPOSE: prove the whole pipeline runs end-to-end (load base model -> LoRA ->
apply the chat template -> a few train steps -> one inference -> save adapter)
BEFORE scaling to the full data_for_finetune/functiongemma_train.jsonl. Unsloth
needs CUDA, so this is driven from your terminal via google-colab-cli — see
finetune/README_colab.md for the exact commands.

EXPANSION (what to flip once this runs clean): LIMIT -> None (all examples),
MAX_STEPS -> a proper epoch count, EXPORT_GGUF -> True, and mask the prompt with
train_on_responses_only (see the TODO near the trainer).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ---- knobs (smoke defaults; the expansion pass edits these) -------------------
DATA_NAME = "functiongemma_train.jsonl"  # uploaded via `colab upload`
LIMIT = 5            # smoke: first 5 rows. Expansion: None -> use all 88.
MAX_STEPS = 30       # smoke: a handful of steps just to move the loss.
BASE_MODEL = "unsloth/functiongemma-270m-it"
MAX_SEQ_LEN = 2048
OUT = "outputs/lora"
EXPORT_GGUF = False  # smoke: skip. Expansion: True -> also save a Q8_0 GGUF.
# ------------------------------------------------------------------------------


def resolve_data() -> Path:
    """Find the uploaded JSONL. `colab upload` may land it outside the exec cwd
    (/content), so check the likely dirs and then glob for it."""
    env = os.environ.get("SMOKE_DATA")
    candidates = [
        Path(env) if env else None,
        Path(DATA_NAME),
        Path("/content") / DATA_NAME,
        Path("/root") / DATA_NAME,
        Path.home() / DATA_NAME,
    ]
    for c in candidates:
        if c and c.exists():
            return c
    for root in ("/content", "/root", str(Path.home())):
        hits = list(Path(root).rglob(DATA_NAME))
        if hits:
            return hits[0]
    raise SystemExit(
        f"{DATA_NAME} not found (cwd={Path.cwd()}). Upload it to /content, then verify:\n"
        "  colab upload -s gemma-ft data_for_finetune/functiongemma_train.jsonl "
        "/content/functiongemma_train.jsonl\n"
        "  colab ls -s gemma-ft /content"
    )


def load_examples() -> list[dict]:
    data = resolve_data()
    print(f"[smoke] using data at {data}")
    rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    return rows[:LIMIT] if LIMIT else rows


def main() -> None:
    from unsloth import FastLanguageModel  # noqa: E402  (heavy; import on-runtime)
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,      # 270M full-precision fits easily (~0.6 GB)
        full_finetuning=False,   # LoRA
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    examples = load_examples()
    print(f"[smoke] training on {len(examples)} examples: {[e['id'] for e in examples]}")

    def to_text(ex: dict) -> dict:
        # Let the model's own chat template render the developer/tool-declaration
        # framing; we only assert our validated DSL target survives templating.
        text = tokenizer.apply_chat_template(ex["messages"], tools=ex["tools"],
                                             tokenize=False)
        target = ex["messages"][-1]["content"]
        assert target in text, f"target vanished after templating: {ex['id']}"
        return {"text": text}

    ds = Dataset.from_list([to_text(ex) for ex in examples])
    print("[smoke] ---- one rendered training example (truncated) ----")
    print(ds[0]["text"][:900])
    print("[smoke] -------------------------------------------------")

    # TODO(expansion): wrap with unsloth.chat_templates.train_on_responses_only
    # so loss is computed on the assistant turn only, not the prompt.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=1,
            max_steps=MAX_STEPS,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )
    stats = trainer.train()
    print(f"[smoke] final training loss: {stats.training_loss:.4f}")

    # Inference sanity: feed the first example's prompt (minus the gold turn) and
    # see whether the model emits a <start_function_call> at all.
    FastLanguageModel.for_inference(model)
    probe = examples[0]["messages"][:-1]
    inputs = tokenizer.apply_chat_template(
        probe, tools=examples[0]["tools"], add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    gen = model.generate(input_ids=inputs, max_new_tokens=128,
                         temperature=1.0, top_p=0.95, top_k=64)
    print("[smoke] ---- probe generation ----")
    print(tokenizer.decode(gen[0][inputs.shape[1]:], skip_special_tokens=False))
    print("[smoke] ----------------------------")

    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    print(f"[smoke] saved LoRA adapter -> {OUT}")

    if EXPORT_GGUF:
        model.save_pretrained_gguf("outputs/gguf", tokenizer,
                                   quantization_method="q8_0")
        print("[smoke] saved GGUF -> outputs/gguf")

    print("[smoke] SMOKE OK")


if __name__ == "__main__":
    main()
