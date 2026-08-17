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
from _bundled import bundled  # noqa: E402  (needs the line above)

# ---- knobs --------------------------------------------------------------------
BASE_MODEL = "unsloth/Qwen3-8B"
# Qwen3 targets carry a <think> trace plus the tool call, and the wallet system
# prompt is ~1.3k tokens; 2048 truncated a visible slice of the Gemma-4 rows, and
# a truncated row loses its target entirely (silently, as an all-masked row).
# 2048 since 2026-08-15. The app-contract prompt is far smaller than the old
# harness scaffold it replaced (two tools and a 533-char systemNudge instead of
# four tools and ~4 KB of REFERENCE DATA/CONVENTIONS/SAFETY), putting the worst
# rendered row near ~1.1k tokens. 2048 keeps ~2x headroom and roughly halves
# step time; the marker assertion below still backstops truncation.
MAX_SEQ_LEN = 2048
EPOCHS = 3
# 8B at 4096 ctx: batch 2 x accum 8 keeps the effective batch at 16 (matching the
# Gemma-4 run) inside 40 GB.
BATCH = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4

HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/outputs"
DATA_REMOTE = "/data/qwen_train.jsonl"
# The app's own prompt dump for a QWEN model. The wallet renders through each
# model's own chat template, so Qwen's app-prompt bytes are not Gemma's — this
# is produced by `wallet-eval prompt-dump --model <qwen gguf>`.
REFERENCE_REMOTE = "/data/app_contract_reference.qwen.json"
ADAPTER_OUT = f"{OUTPUTS_DIR}/adapter"

# Qwen3 uses ChatML. If these markers are wrong every row is masked (zero
# gradient), so the run asserts on the all-masked count rather than burning an
# A100-hour on a no-op.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

_REPO = Path(__file__).resolve().parent.parent
# ------------------------------------------------------------------------------

hf_cache = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("qwen-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake",
                 "libssl-dev", "libcurl4-openssl-dev", "curl")
    .pip_install("unsloth", "huggingface_hub")
    .env({"HF_HOME": HF_CACHE_DIR})
    .add_local_python_source("_bundled")
)

# `bundled()` is a LOCAL-only path helper (see modal_export_gemma4_local.py for
# the full explanation) — it resolves candidates relative to _REPO, which is
# only the repo root when this module is imported by the local `modal run`
# CLI. Modal re-imports this module INSIDE the container to find the app/
# function objects after the image is already built, and there `__file__` is
# `/root/modal_finetune_qwen.py`, so _REPO becomes `/` and bundled() would
# raise. Gate on modal.is_local() (False inside a Function/container, True
# everywhere else) so correctness doesn't depend on a path coincidence.
if modal.is_local():
    _DATA_LOCAL = bundled(_REPO, "data_for_finetune/qwen_train.jsonl",
                                 "data/qwen_train.jsonl")
    image = image.add_local_file(str(_DATA_LOCAL), DATA_REMOTE)
    image = image.add_local_file(
        str(_REPO / "pf" / "app_contract_reference.qwen.json"), REFERENCE_REMOTE)

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
        # enable_thinking mirrors SamplerOptions.enableThinking, which the app
        # leaves on. Qwen3's template branches on it, so omitting it trains on a
        # prompt the app never sends — the same defect found in the Gemma path.
        prompt = tokenizer.apply_chat_template(
            ex["messages"][:-1], tools=ex["tools"], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        target = ex["messages"][-1]["content"]
        text = f"{prompt}{target}<|im_end|>"
        assert target in text, f"target vanished: {ex['id']}"
        assert RESPONSE_PART in text, f"no assistant marker: {ex['id']}"
        return {"text": text}

    # Fail before GPU spend if training-time rendering has drifted from the bytes
    # the wallet app sends for THIS model family.
    reference = json.loads(Path(REFERENCE_REMOTE).read_text())
    ref_tools = json.loads(reference["toolsJSON"])
    for ref_case in reference["cases"]:
        rendered = tokenizer.apply_chat_template(
            ref_case["messages"], tools=ref_tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        assert rendered == ref_case["rendered"], (
            f"PROMPT PARITY FAILED for {ref_case['label']}: "
            f"{len(rendered)} vs {len(ref_case['rendered'])} chars")
    print(f"[train] prompt parity OK against the app renderer "
          f"({len(reference['cases'])} shapes)", flush=True)

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
def main() -> None:
    # spawn (not .remote()): submit the job and return immediately so the run
    # does NOT depend on the local client's streaming connection staying alive.
    # A dropped connection was cancelling .remote()/--detach runs ~30 min in
    # on the sibling Gemma-4 job (see modal_finetune_gemma4.py) — the same risk
    # applies here, and this run is long enough (3 epochs, A100) to hit it. The
    # function runs server-side to completion and commits the adapter to the
    # outputs Volume; poll `modal app logs <app-id>` or
    # `modal volume ls qwen-ft-outputs /` for `adapter`.
    call = train.spawn()
    print(f"SPAWNED train call_id={call.object_id} — running detached on Modal.")
