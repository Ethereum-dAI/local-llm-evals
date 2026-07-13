"""Full fine-tune of FunctionGemma-270m on the wallet tool-call dataset (Colab GPU).

Expansion of finetune/smoke_finetune.py once the smoke run passed:
- train on ALL rows of functiongemma_train.jsonl (not just 5),
- real epoch count instead of a fixed step budget,
- **response-only loss** (train_on_responses_only) so the model learns the tool
  call / clarification / refusal, not the developer+user framing,
- fixed generation (attention mask via return_dict, greedy) for a clean sanity
  check that it now emits real <start_function_call> calls,
- export a Q8_0 GGUF (falls back to a merged-16bit save if the llama.cpp build
  fails) that drops straight into pf/provider_functiongemma.py via `model_path`.

Run (data already uploaded to /content by the smoke step):
    colab exec -s gemma-ft -f finetune/train_functiongemma.py --timeout 3600
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ---- knobs --------------------------------------------------------------------
DATA_NAME = "functiongemma_train.jsonl"
EPOCHS = 8
BATCH = 8
BASE_MODEL = "unsloth/functiongemma-270m-it"
MAX_SEQ_LEN = 2048
# Absolute /content paths so `colab download` targets are deterministic and
# survive the exec's (unknown) cwd. The final GGUF is copied to FINAL_GGUF.
LORA_OUT = "/content/outputs/lora"
GGUF_OUT = "/content/outputs/gguf"
GGUF_QUANT = "q8_0"
FINAL_GGUF = "/content/functiongemma-270m-ft.Q8_0.gguf"
# ------------------------------------------------------------------------------


def ensure_deps() -> None:
    try:
        import unsloth  # noqa: F401
        return
    except Exception:
        import subprocess
        import sys
        print("[train] installing unsloth (one-time, ~minutes) ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"],
                       check=True)


def resolve_data() -> Path:
    env = os.environ.get("TRAIN_DATA")
    for c in [Path(env) if env else None, Path(DATA_NAME),
              Path("/content") / DATA_NAME, Path("/root") / DATA_NAME,
              Path.home() / DATA_NAME]:
        if c and c.exists():
            return c
    for root in ("/content", "/root", str(Path.home())):
        hits = list(Path(root).rglob(DATA_NAME))
        if hits:
            return hits[0]
    raise SystemExit(f"{DATA_NAME} not found (cwd={Path.cwd()}); upload it to /content.")


def main() -> None:
    ensure_deps()
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    rows = [json.loads(l) for l in resolve_data().read_text().splitlines() if l.strip()]
    print(f"[train] {len(rows)} examples")

    def to_text(ex: dict) -> dict:
        text = tokenizer.apply_chat_template(ex["messages"], tools=ex["tools"],
                                             tokenize=False)
        assert ex["messages"][-1]["content"] in text, f"target vanished: {ex['id']}"
        return {"text": text}

    ds = Dataset.from_list([to_text(r) for r in rows])

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=BATCH, gradient_accumulation_steps=1,
            warmup_ratio=0.05, num_train_epochs=EPOCHS, learning_rate=2e-4,
            logging_steps=5, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="linear", seed=3407, output_dir="outputs",
            report_to="none",
        ),
    )
    # Mask everything up to each model turn: loss only on the assistant response.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    stats = trainer.train()
    print(f"[train] final loss: {stats.training_loss:.4f}")

    # Sanity: greedy-decode a few examples' prompts and compare to gold.
    FastLanguageModel.for_inference(model)
    hits = 0
    for ex in rows[:6]:
        enc = tokenizer.apply_chat_template(
            ex["messages"][:-1], tools=ex["tools"], add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        gen = model.generate(**enc, max_new_tokens=200, do_sample=False)
        text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=False)
        gold = ex["messages"][-1]["content"]
        gold_call = "<start_function_call>" in gold
        got_call = "<start_function_call>" in text
        hits += int(gold_call == got_call)
        print(f"\n[{ex['id']}] gold_call={gold_call} got_call={got_call}")
        print("  GOLD:", gold[:130])
        print("  GEN :", text.split("<end_of_turn>")[0][:200])
    print(f"\n[train] call-presence agreement on 6 probes: {hits}/6")

    model.save_pretrained(LORA_OUT)
    tokenizer.save_pretrained(LORA_OUT)
    print(f"[train] saved LoRA -> {LORA_OUT}")

    import shutil
    try:
        model.save_pretrained_gguf(GGUF_OUT, tokenizer, quantization_method=GGUF_QUANT)
        # save_pretrained_gguf writes under a sibling *_gguf dir; find the Q8_0 and
        # copy it to a stable absolute path for `colab download`.
        q8 = [p for p in Path("/content").rglob("*.gguf")
              if "q8_0" in p.name.lower()]
        if q8:
            shutil.copy(q8[0], FINAL_GGUF)
            print(f"[train] GGUF ready: {FINAL_GGUF} "
                  f"({Path(FINAL_GGUF).stat().st_size / 1e6:.1f} MB)")
        else:
            print("[train] WARNING: no q8_0 gguf found under /content")
    except Exception as e:  # llama.cpp build can be flaky on Colab
        print(f"[train] GGUF export failed ({type(e).__name__}: {e}); "
              "saving merged 16-bit for local conversion instead.")
        model.save_pretrained_merged("/content/outputs/merged16", tokenizer,
                                     save_method="merged_16bit")
        print("[train] saved merged 16-bit -> /content/outputs/merged16")

    print("[train] TRAIN OK")


if __name__ == "__main__":
    main()
