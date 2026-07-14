"""GPU fine-tune of FunctionGemma-270m on Modal.

This mirrors finetune/train_functiongemma.py (the Colab recipe) exactly — same
base model, LoRA config, response-only loss, SFT hyperparameters and Q8_0 GGUF
export — but runs on a Modal T4 GPU instead of Colab, with two persistent
Volumes so the work is cheap and repeatable:

  * an HF-cache Volume mounted at /root/.cache/huggingface (HF_HOME) so the
    base weights download only ONCE and are reused across runs;
  * an outputs Volume mounted at /outputs so the produced GGUF persists and can
    be pulled down with `modal volume get`.

The Colab run writes models/functiongemma-270m-ft.Q8_0.gguf; this run is
retrieved locally as models/functiongemma-270m-ft-modal.Q8_0.gguf (the `-modal`
suffix keeps the two from colliding).

Run:
    uv run --with modal modal run finetune/modal_finetune.py

Retrieve (output volume is "functiongemma-ft-outputs"):
    modal volume get functiongemma-ft-outputs \\
        /functiongemma-270m-ft.Q8_0.gguf \\
        models/functiongemma-270m-ft-modal.Q8_0.gguf
"""
from __future__ import annotations

from pathlib import Path

import modal

# ---- knobs (identical recipe to finetune/train_functiongemma.py) --------------
BASE_MODEL = "unsloth/functiongemma-270m-it"
MAX_SEQ_LEN = 2048
EPOCHS = 3  # v2: ~1740 examples (20x v1) -> fewer epochs to avoid overfit/collapse
BATCH = 16  # 270M barely uses a T4; bigger batch -> fewer steps within the timeout
GGUF_QUANT = "q8_0"

HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/outputs"
DATA_REMOTE = "/data/functiongemma_train.jsonl"
FINAL_GGUF = f"{OUTPUTS_DIR}/functiongemma-270m-ft.Q8_0.gguf"

# Repo root, so `modal run` works regardless of cwd.
_REPO = Path(__file__).resolve().parent.parent
_DATA_LOCAL = _REPO / "data_for_finetune" / "functiongemma_train.jsonl"
# ------------------------------------------------------------------------------

# Persistent Volumes: base weights cached once, outputs survive the container.
hf_cache = modal.Volume.from_name("functiongemma-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("functiongemma-ft-outputs", create_if_missing=True)

# unsloth pulls torch/trl/peft/transformers/bitsandbytes. cmake + build-essential
# and libssl-dev/libcurl4-openssl-dev/curl are needed by unsloth's Q8_0 GGUF
# export, which builds llama.cpp on the fly — install them up front so the build
# never stops to prompt for packages (it errors with "EOF when reading a line"
# on Modal's non-interactive stdin otherwise).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake",
                 "libssl-dev", "libcurl4-openssl-dev", "curl")
    .pip_install("unsloth", "huggingface_hub")
    .env({"HF_HOME": HF_CACHE_DIR})
    .add_local_file(str(_DATA_LOCAL), DATA_REMOTE)
)

app = modal.App("functiongemma-finetune")


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={HF_CACHE_DIR: hf_cache, OUTPUTS_DIR: outputs},
)
def train() -> str:
    import json
    import shutil

    # Warm the HF cache first: Unsloth loads the base model under a forced
    # hf-offline context, which raises "no model.safetensors" if the weights
    # aren't already cached. With the Volume this only downloads on run #1.
    from huggingface_hub import snapshot_download
    for attempt in range(3):
        try:
            snapshot_download(BASE_MODEL)
            break
        except Exception as e:  # transient network hiccups
            print(f"[train] snapshot_download retry {attempt}: {e}", flush=True)
    hf_cache.commit()

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

    rows = [json.loads(l) for l in Path(DATA_REMOTE).read_text().splitlines()
            if l.strip()]
    print(f"[train] {len(rows)} examples", flush=True)

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
            lr_scheduler_type="linear", seed=3407, output_dir=OUTPUTS_DIR,
            report_to="none",
        ),
    )
    # Mask everything up to each model turn: loss only on the assistant response.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # Diagnostic: rows whose labels are entirely masked (-100) contribute no
    # gradient — v1 silently dropped 14/88. Report how many and in which
    # categories so a systematic masking bug is visible instead of silent.
    from collections import Counter
    dropped = [i for i, ex in enumerate(trainer.train_dataset)
               if all(t == -100 for t in ex["labels"])]
    print(f"[train] all-masked rows: {len(dropped)}/{len(rows)} "
          f"-> {dict(Counter(rows[i]['category'] for i in dropped))}", flush=True)

    stats = trainer.train()
    print(f"[train] final loss: {stats.training_loss:.4f}", flush=True)

    # Sanity: greedy-decode across DIVERSE categories (not just the first rows,
    # which all sort to one class) and compare call-presence to gold.
    FastLanguageModel.for_inference(model)
    seen: set[str] = set()
    probes = []
    for ex in rows:
        if ex["category"] not in seen:
            seen.add(ex["category"])
            probes.append(ex)
    hits = 0
    for ex in probes:
        enc = tokenizer.apply_chat_template(
            ex["messages"][:-1], tools=ex["tools"], add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        gen = model.generate(**enc, max_new_tokens=200, do_sample=False)
        text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=False)
        gold = ex["messages"][-1]["content"]
        ok = ("<start_function_call>" in gold) == ("<start_function_call>" in text)
        hits += int(ok)
        print(f"[probe] {ex['category']:24} ok={ok} got={text.split(chr(10))[0][:90]!r}",
              flush=True)
    print(f"[train] call-presence agreement: {hits}/{len(probes)} probes", flush=True)

    # Export Q8_0 GGUF (unsloth builds llama.cpp for the conversion).
    gguf_dir = f"{OUTPUTS_DIR}/gguf"
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=GGUF_QUANT)
    q8 = [p for p in Path(OUTPUTS_DIR).rglob("*.gguf") if "q8_0" in p.name.lower()]
    if not q8:
        raise SystemExit("no q8_0 gguf produced under /outputs")
    shutil.copy(q8[0], FINAL_GGUF)
    size_mb = Path(FINAL_GGUF).stat().st_size / 1e6
    print(f"[train] GGUF ready: {FINAL_GGUF} ({size_mb:.1f} MB)", flush=True)

    outputs.commit()
    return f"final_loss={stats.training_loss:.4f} gguf={FINAL_GGUF} size={size_mb:.1f}MB"


@app.local_entrypoint()
def main() -> None:
    print(train.remote())
