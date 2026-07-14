"""Isolate WHERE the v2 fine-tune breaks: HF inference vs GGUF.

Training converged (loss ~0.02, no NaN) yet the exported Q8_0 GGUF emits garbled
token-salad locally. This loads the MERGED 16-bit HF model that was produced right
before GGUF conversion (functiongemma-ft-outputs volume at /outputs/gguf) and runs
greedy HF inference on a handful of TRAINING prompts, printing gold vs generated.

- If HF output reproduces the gold DSL calls  -> the weights are fine; the GGUF
  q8_0 conversion / chat-template is the culprit (try f16, or serve via HF).
- If HF output is ALSO garbage                -> it's the model/approach itself
  (exposure bias, DSL-as-content, template used at inference), not quantization.

Run:  uv run --with modal modal run finetune/modal_diagnose.py
"""
from __future__ import annotations

from pathlib import Path

import modal

# Test the UNMERGED LoRA adapter directly (base + checkpoint). If this generates
# clean DSL but the merged model (/outputs/gguf) is garbage, the LoRA->16bit merge
# is the culprit (gemma3 is dtype-sensitive; note unsloth's "float16 won't work"
# warning). ADAPTER holds adapter_model.safetensors + adapter_config.json.
ADAPTER = "/outputs/checkpoint-309"
DATA_REMOTE = "/data/diag_sample.jsonl"
MAX_SEQ_LEN = 2048

_REPO = Path(__file__).resolve().parent.parent
_DIAG_LOCAL = _REPO / "finetune" / "diag_sample.jsonl"

hf_cache = modal.Volume.from_name("functiongemma-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("functiongemma-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake",
                 "libssl-dev", "libcurl4-openssl-dev", "curl")
    .pip_install("unsloth", "huggingface_hub")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_file(str(_DIAG_LOCAL), DATA_REMOTE)
)

app = modal.App("functiongemma-diagnose")


@app.function(image=image, gpu="T4", timeout=1200,
              volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs})
def diagnose() -> None:
    import json

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, full_finetuning=False,
    )
    FastLanguageModel.for_inference(model)

    rows = [json.loads(l) for l in Path(DATA_REMOTE).read_text().splitlines() if l.strip()]
    for ex in rows:
        enc = tokenizer.apply_chat_template(
            ex["messages"][:-1], tools=ex["tools"], add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        gen = model.generate(**enc, max_new_tokens=200, do_sample=False)
        got = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=False)
        print(f"\n===== {ex['id']} ({ex['category']}) =====", flush=True)
        print("USER:", [m["content"] for m in ex["messages"] if m["role"] == "user"][-1][:80])
        print("GOLD:", ex["messages"][-1]["content"][:170])
        print("HF  :", got.split("<end_of_turn>")[0][:220], flush=True)


@app.local_entrypoint()
def main() -> None:
    diagnose.remote()
