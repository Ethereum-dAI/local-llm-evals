"""GPU fine-tune of Qwen3-8B on Modal — the strongest general base measured here.

Mirrors finetune/modal_finetune_gemma4.py deliberately, so the two fine-tunes
differ in base model and encoding ONLY, and the resulting scores are comparable:

  * base = unsloth/Qwen3-8B (unsloth's mirror of Qwen/Qwen3-8B, which scored
    41.0% untuned on the 307-case dev set);
  * data = data_for_finetune/qwen_train.jsonl — the SAME 1739 rows as the
    Gemma-4 set (asserted by tests/test_qwen_finetune_integrity.py), re-encoded
    as Hermes `<tool_call>{...}</tool_call>` JSON with <think> traces;
  * response-only loss masked at Qwen3's ChatML turn markers;
  * A100-40GB — 8B LoRA in bf16 does not fit a smaller card;
  * NO in-training GGUF export: the Gemma-4 run hit unsloth's LoRA->16bit "merge
    trap" (see finetune/README.md), so the proven pattern is save-adapter-only
    here and a separate export job.

Run:
    uv run --with modal modal run finetune/modal_finetune_qwen.py

The adapter lands at qwen-ft-outputs:/outputs/adapter (a stable path the export
and eval jobs read — no checkpoint-number guessing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _bundled import bundled  # noqa: E402  (needs the line above)
except ModuleNotFoundError:
    # Modal 1.5 dropped directory automounting, so inside the container this file
    # is the ONLY thing at /root and its sibling helper is gone — while the module
    # body (including the image definition) still re-executes there. `bundled`
    # only ever resolves a build-time path, so a no-op stand-in is correct: the
    # data file was already baked into the image by the local run.
    def bundled(repo: Path, *candidates: str) -> Path:  # type: ignore[misc]
        return repo / candidates[0]

# ---- knobs --------------------------------------------------------------------
BASE_MODEL = "unsloth/Qwen3-8B"
# Qwen3 targets carry a <think> trace plus the tool call, and the wallet system
# prompt is ~1.3k tokens; 2048 truncated a visible slice of the Gemma-4 rows, and
# a truncated row loses its target entirely (silently, as an all-masked row).
MAX_SEQ_LEN = 4096
EPOCHS = 3
# 8B at 4096 ctx: batch 2 x accum 8 keeps the effective batch at 16 (matching the
# Gemma-4 run) inside 40 GB.
BATCH = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4

HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/outputs"
DATA_REMOTE = "/data/qwen_train.jsonl"
ADAPTER_OUT = f"{OUTPUTS_DIR}/adapter"

# Qwen3 uses ChatML. If these markers are wrong every row is masked (zero
# gradient), so the run asserts on the all-masked count rather than burning an
# A100-hour on a no-op.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

_REPO = Path(__file__).resolve().parent.parent
_DATA_LOCAL = bundled(_REPO, "data_for_finetune/qwen_train.jsonl",
                             "data/qwen_train.jsonl")
# ------------------------------------------------------------------------------

hf_cache = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("qwen-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake",
                 "libssl-dev", "libcurl4-openssl-dev", "curl")
    .pip_install("unsloth", "huggingface_hub")
    .env({"HF_HOME": HF_CACHE_DIR})
    .add_local_file(str(_DATA_LOCAL), DATA_REMOTE)
)

app = modal.App("qwen-finetune")


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
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

    # IMPORT ORDER IS LOAD-BEARING — unsloth must come before trl.
    # unsloth patches TRL's SFTConfig/SFTTrainer when IT is imported. Import trl
    # first (e.g. by alphabetising these lines) and the names here bind the
    # unpatched classes, whose `eos_token` stays the literal sentinel
    # '<EOS_TOKEN>' and blows up at trainer construction against Qwen's
    # vocabulary. Three symptom-level fixes to that error all failed; this
    # ordering — the one modal_finetune_gemma4.py uses — is the actual cause.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    rows = [json.loads(l) for l in Path(DATA_REMOTE).read_text().splitlines()
            if l.strip()]
    print(f"[train] {len(rows)} examples", flush=True)

    def to_text(ex: dict) -> dict:
        """Render the PROMPT with the template, then append the target verbatim.

        Templating the full conversation (what the Gemma-4 job does) silently
        destroys these targets: Qwen3's template splits every assistant message on
        `</think>` and re-emits only the part after it, so the <think> arithmetic
        trace — the thing this fine-tune most needs to learn — is dropped from the
        training text. The guard below caught exactly that ("target vanished").

        Appending after `add_generation_prompt=True` also makes the training text
        byte-identical to what the model sees at inference, which templating the
        whole conversation does not guarantee.
        """
        prompt = tokenizer.apply_chat_template(
            ex["messages"][:-1], tools=ex["tools"], tokenize=False,
            add_generation_prompt=True)
        target = ex["messages"][-1]["content"]
        text = f"{prompt}{target}<|im_end|>"
        assert target in text, f"target vanished: {ex['id']}"
        assert RESPONSE_PART in text, f"no assistant marker: {ex['id']}"
        return {"text": text}

    ds = Dataset.from_list([to_text(r) for r in rows])
    print(f"[train] rendered sample head:\n{ds[0]['text'][:600]}", flush=True)

    # TRL renamed SFTConfig's sequence-length knob (`max_seq_length` -> `max_length`)
    # and the image is built from an unpinned `unsloth`, so the name depends on
    # whenever the image was last rebuilt. Pick whichever this TRL declares — the
    # wrong one is a TypeError at construction, i.e. an A100 sitting idle.
    import inspect

    cfg_kwargs = dict(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.05, num_train_epochs=EPOCHS, learning_rate=LEARNING_RATE,
        logging_steps=5, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=3407, output_dir=OUTPUTS_DIR,
        report_to="none",
    )
    params = inspect.signature(SFTConfig.__init__).parameters
    length_arg = "max_seq_length" if "max_seq_length" in params else "max_length"
    cfg_kwargs[length_arg] = MAX_SEQ_LEN

    # Same story for the trainer: `tokenizer` was renamed `processing_class`.
    tparams = inspect.signature(SFTTrainer.__init__).parameters
    tok_arg = "tokenizer" if "tokenizer" in tparams else "processing_class"
    print(f"[train] TRL api: {length_arg}={MAX_SEQ_LEN}, tokenizer arg={tok_arg}",
          flush=True)

    # This TRL carries an `eos_token` defaulting to the literal placeholder
    # '<EOS_TOKEN>', which is in no real vocabulary and raises inside
    # SFTTrainer.__init__. It must be passed to the CONSTRUCTOR: assigning it on
    # the instance afterwards does not survive (the trainer rebuilds the config
    # from the dataclass fields), and `inspect` can't detect support because
    # unsloth's patched SFTConfig has a (*args, **kwargs) signature. So: try it,
    # and fall back for a TRL old enough not to have the argument at all.
    # Qwen3 ends a turn with <|im_end|> — the token `to_text` appends to targets.
    try:
        sft_config = SFTConfig(eos_token=tokenizer.eos_token, **cfg_kwargs)
        print(f"[train] eos_token -> {tokenizer.eos_token!r}", flush=True)
    except TypeError:
        sft_config = SFTConfig(**cfg_kwargs)
        print("[train] SFTConfig has no eos_token argument; using its default",
              flush=True)

    trainer = SFTTrainer(model=model, train_dataset=ds,
                         args=sft_config, **{tok_arg: tokenizer})
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART,
    )

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

    model.save_pretrained(ADAPTER_OUT)
    tokenizer.save_pretrained(ADAPTER_OUT)
    outputs.commit()
    print(f"[train] adapter saved -> {ADAPTER_OUT}", flush=True)
    return f"loss={stats.training_loss:.4f} adapter={ADAPTER_OUT}"


@app.local_entrypoint()
def main():
    print(train.remote())
