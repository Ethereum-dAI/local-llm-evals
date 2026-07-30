"""GPU fine-tune of Gemma-4 E4B on Modal — the on-device model shipped by
local-wallet-mac (google/gemma-4-E4B-it, run as a Q4_K_M GGUF via llama.cpp).

Mirrors finetune/modal_finetune.py (the FunctionGemma recipe) but for Gemma-4:

  * base = unsloth/gemma-4-E4B-it, loaded with unsloth's FastModel (E4B is a
    multimodal-capable checkpoint) — LoRA on the language+attention+MLP layers;
  * data = data_for_finetune/gemma4_train.jsonl (GEMMA4 dialect, `system` role,
    <think> reasoning traces) — see scripts/generate_gemma4_finetune_data.py;
  * response-only loss masked at Gemma-4's turn markers (<|turn>user / <|turn>model);
  * A100-40GB (E4B LoRA ≈17 GB VRAM — a T4 is too small);
  * NO in-training GGUF export — unsloth's LoRA→16bit merge corrupts gemma3-family
    weights (the "merge trap", see finetune/README.md). We only save the proven
    LoRA adapter here; finetune/modal_export_gemma4.py produces the correct GGUF.

Two persistent Volumes keep it cheap and repeatable: an HF cache (base weights
download once) and an outputs Volume (the adapter survives the container).

Run:
    uv run --with modal modal run finetune/modal_finetune_gemma4.py

The final adapter lands at gemma4-ft-outputs:/outputs/adapter (a stable path the
export/eval scripts read — no checkpoint-number guessing).
"""
from __future__ import annotations

from pathlib import Path

import modal

# ---- knobs --------------------------------------------------------------------
# The wallet's on-device model id is google/gemma-4-E4B-it (ggml-org quantizes it
# to the Q4_K_M GGUF it ships). We train from unsloth's ungated mirror of those
# exact weights — same model, no google/* license gate on Modal, unsloth-optimized.
BASE_MODEL = "unsloth/gemma-4-E4B-it"
MAX_SEQ_LEN = 2048
EPOCHS = 3
# E4B is ~15x FunctionGemma-270m: small per-device batch + accumulation to reach
# an effective batch of 16 without exceeding 40 GB.
BATCH = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4

HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/outputs"
DATA_REMOTE = "/data/gemma4_train.jsonl"
ADAPTER_OUT = f"{OUTPUTS_DIR}/adapter"

# Gemma-4 chat-template turn markers — response-only loss masks everything up to
# each model turn. If these are wrong the trainer masks every row (zero gradient),
# so we assert on the all-masked count below and fail loudly instead of burning a
# GPU-hour on a no-op run.
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"

_REPO = Path(__file__).resolve().parent.parent
_DATA_LOCAL = _REPO / "data_for_finetune" / "gemma4_train.jsonl"
# ------------------------------------------------------------------------------

hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("gemma4-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake",
                 "libssl-dev", "libcurl4-openssl-dev", "curl")
    .pip_install("unsloth", "huggingface_hub")
    .env({"HF_HOME": HF_CACHE_DIR})
    .add_local_file(str(_DATA_LOCAL), DATA_REMOTE)
)

app = modal.App("gemma4-finetune")


@app.function(
    image=image,
    gpu="A100",
    timeout=10800,
    volumes={HF_CACHE_DIR: hf_cache, OUTPUTS_DIR: outputs},
)
def train() -> str:
    import json
    from collections import Counter

    # Warm the HF cache first so unsloth's forced hf-offline load finds weights.
    from huggingface_hub import snapshot_download
    for attempt in range(3):
        try:
            snapshot_download(BASE_MODEL)
            break
        except Exception as e:  # transient network hiccups
            print(f"[train] snapshot_download retry {attempt}: {e}", flush=True)
    hf_cache.commit()

    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
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
    # Eyeball the exact turn markers the template produced (so INSTRUCTION_PART /
    # RESPONSE_PART can be corrected if the template ever changes).
    print(f"[train] rendered sample head:\n{ds[0]['text'][:600]}", flush=True)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.05, num_train_epochs=EPOCHS, learning_rate=LEARNING_RATE,
            logging_steps=5, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="linear", seed=3407, output_dir=OUTPUTS_DIR,
            report_to="none",
        ),
    )
    # Mask everything up to each model turn: loss only on the assistant response.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART,
    )

    # Guard: rows whose labels are entirely masked (-100) contribute no gradient.
    # A handful can be legitimate truncation, but if most rows are masked the turn
    # markers are wrong — abort before wasting the GPU hour.
    dropped = [i for i, ex in enumerate(trainer.train_dataset)
               if all(t == -100 for t in ex["labels"])]
    print(f"[train] all-masked rows: {len(dropped)}/{len(rows)} "
          f"-> {dict(Counter(rows[i]['category'] for i in dropped))}", flush=True)
    if len(dropped) > 0.5 * len(rows):
        raise SystemExit(
            f"{len(dropped)}/{len(rows)} rows fully masked — response markers "
            f"({INSTRUCTION_PART!r}/{RESPONSE_PART!r}) do not match the template"
        )

    stats = trainer.train()
    print(f"[train] final loss: {stats.training_loss:.4f}", flush=True)

    # Save the LoRA adapter to a STABLE path FIRST (export/eval read this — no
    # checkpoint number to track), then commit — so nothing below can cost us the
    # trained weights. GGUF is produced separately by the bf16 export (merge trap).
    model.save_pretrained(ADAPTER_OUT)
    tokenizer.save_pretrained(ADAPTER_OUT)
    outputs.commit()
    print(f"[train] adapter saved -> {ADAPTER_OUT}", flush=True)

    # Sanity: greedy-decode one probe per category and compare call-presence to
    # gold. Best-effort — a tokenizer/generate quirk must never block the save.
    hits = probed = 0
    try:
        FastModel.for_inference(model)
        seen: set[str] = set()
        probes = [ex for ex in rows if not (ex["category"] in seen or seen.add(ex["category"]))]
        probed = len(probes)
        for ex in probes:
            enc = tokenizer.apply_chat_template(
                ex["messages"][:-1], tools=ex["tools"], add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            enc = {k: v.to(model.device) for k, v in enc.items() if hasattr(v, "to")}
            gen = model.generate(**enc, max_new_tokens=220, do_sample=False)
            text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=False)
            gold = ex["messages"][-1]["content"]
            ok = ("<|tool_call>" in gold) == ("<|tool_call>" in text)
            hits += int(ok)
            print(f"[probe] {ex['category']:24} ok={ok} got={text.split(chr(10))[0][:90]!r}",
                  flush=True)
        print(f"[train] call-presence agreement: {hits}/{probed} probes", flush=True)
    except Exception as e:  # diagnostics only — adapter is already saved
        print(f"[train] probe loop skipped ({type(e).__name__}: {e})", flush=True)

    return f"final_loss={stats.training_loss:.4f} probes={hits}/{probed} adapter={ADAPTER_OUT}"


@app.local_entrypoint()
def main() -> None:
    # spawn (not .remote): submit the job and return immediately so the run does
    # NOT depend on the local client's streaming connection staying alive. A
    # dropped connection was cancelling .remote()/--detach runs ~30 min in. The
    # function runs server-side to completion and commits the adapter to the
    # outputs Volume; poll `modal volume ls gemma4-ft-outputs /` for `adapter`.
    call = train.spawn()
    print(f"SPAWNED train call_id={call.object_id} — running detached on Modal.")
