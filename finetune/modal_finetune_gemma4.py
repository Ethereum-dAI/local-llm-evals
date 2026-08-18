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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bundled import bundled  # noqa: E402  (needs the line above)

# ---- knobs --------------------------------------------------------------------
# The wallet's on-device model id is google/gemma-4-E4B-it (ggml-org quantizes it
# to the Q4_K_M GGUF it ships). We train from unsloth's ungated mirror of those
# exact weights — same model, no google/* license gate on Modal, unsloth-optimized.
BASE_MODEL = "unsloth/gemma-4-E4B-it"
# 2048 since 2026-08-15, down from 4096. The old figure was sized for a
# ~6.3k-char `tools` payload and a ~4k-char system scaffold, which put rendered
# examples at ~2.8k tokens median / ~3.1k worst case; 2048 truncated the
# `<|turn>model\n` response marker off nearly every row, masking all labels to
# -100. Both shrank when the prompt moved to the app's own contract: two tools
# instead of four (2.9k chars) and the app's 533-char systemNudge instead of the
# scaffold, putting the worst case near ~1.0k tokens. 2048 keeps 2x headroom and
# roughly halves step time. The all-masked guard below still backstops it.
MAX_SEQ_LEN = 2048
#: Defaults. Both are overridable per run from the entrypoint, because the Phase 1
#: sweep exists to test exactly these two: gemma-4-E4B-wallet-ft-v4 was trained at
#: 3 epochs / 2e-4 and scored 78.7% on the 1000-case benchmark against base's
#: 90.7%, collapsing from 95.8% at one round to 49.0% at six. Base is flat across
#: depth, so the capability was trained AWAY — which points at over-training on a
#: narrow set rather than at missing data.
EPOCHS = 3
# E4B is ~15x FunctionGemma-270m: small per-device batch + accumulation to reach
# an effective batch of 16 without exceeding 40 GB.
BATCH = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
#: Fraction of the training rows held back for the in-training `eval_loss` guard.
#: This split is IN-DISTRIBUTION and therefore weak on its own: the rows are 85.9%
#: single-turn and 0% three-plus, so falling eval_loss here is consistent with the
#: depth collapse above. It is the cheap inner signal only. The real checkpoint
#: selector is pf/tests.dev.yaml scored by the harness's own scorer
#: (finetune/modal_eval_gemma4.py), which is out-of-distribution by construction.
HOLDOUT_FRAC = 0.10
#: Stop if eval_loss has not improved for this many evaluations. Evaluation and
#: saving both run per EPOCH, so with EPOCHS=3 this rarely fires — that is
#: deliberate for the sweep, where the point is to keep every epoch's checkpoint
#: and score them all rather than to stop early.
EARLY_STOPPING_PATIENCE = 2

HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/outputs"
DATA_REMOTE = "/data/gemma4_train.jsonl"
# The app's own prompt dump, mounted so training can assert parity in-container
# before spending the GPU hour. Regenerate with `wallet-eval prompt-dump`.
REFERENCE_REMOTE = "/data/app_contract_reference.json"
ADAPTER_OUT = f"{OUTPUTS_DIR}/adapter"

# Gemma-4 chat-template turn markers — response-only loss masks everything up to
# each model turn. If these are wrong the trainer masks every row (zero gradient),
# so we assert on the all-masked count below and fail loudly instead of burning a
# GPU-hour on a no-op run.
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"

_REPO = Path(__file__).resolve().parent.parent
# ------------------------------------------------------------------------------

hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("gemma4-ft-outputs", create_if_missing=True)

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
# `/root/modal_finetune_gemma4.py`, so _REPO becomes `/` and bundled() would
# raise. This script only survived that so far by coincidence — DATA_REMOTE
# happens to equal bundled()'s second candidate path, so the in-container
# FileNotFoundError never actually triggered. Gate on modal.is_local() (False
# inside a Function/container, True everywhere else) so correctness no longer
# depends on that coincidence.
if modal.is_local():
    _DATA_LOCAL = bundled(_REPO, "data_for_finetune/gemma4_train.jsonl",
                                 "data/gemma4_train.jsonl")
    image = image.add_local_file(str(_DATA_LOCAL), DATA_REMOTE)
    image = image.add_local_file(
        str(_REPO / "pf" / "app_contract_reference.json"), REFERENCE_REMOTE)

app = modal.App("gemma4-finetune")


@app.function(
    image=image,
    gpu="A100",
    timeout=10800,
    volumes={HF_CACHE_DIR: hf_cache, OUTPUTS_DIR: outputs},
)
def train(epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE,
          tag: str = "") -> str:
    import json
    import random
    from collections import Counter

    # A non-empty tag writes to /outputs/adapter-<tag> so a sweep does not
    # overwrite itself. Empty keeps the historical /outputs/adapter path that
    # modal_export_gemma4.py and modal_eval_gemma4.py already read.
    adapter_out = f"{ADAPTER_OUT}-{tag}" if tag else ADAPTER_OUT
    run_dir = f"{OUTPUTS_DIR}/run-{tag}" if tag else OUTPUTS_DIR
    print(f"[train] epochs={epochs} lr={learning_rate} tag={tag or '(none)'} "
          f"-> adapter={adapter_out}", flush=True)

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
    from transformers import EarlyStoppingCallback
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

    # `enable_thinking=True` is NOT optional. The app renders with thinking on
    # (SamplerOptions.enableThinking defaults true), which puts a `<|think|>`
    # marker right after `<|turn>system`. Without this kwarg the template omits
    # it, and every fine-tune before 2026-08-15 was trained on a prompt the app
    # never sends. finetune/modal_verify_prompt_parity.py asserts the equality.
    def to_text(ex: dict) -> dict:
        text = tokenizer.apply_chat_template(ex["messages"], tools=ex["tools"],
                                             tokenize=False, enable_thinking=True)
        assert ex["messages"][-1]["content"] in text, f"target vanished: {ex['id']}"
        return {"text": text}

    # Fail before the GPU spend if training-time rendering has drifted from the
    # bytes the app sends. The reference is produced by the app's own renderer
    # (`wallet-eval prompt-dump`); see modal_verify_prompt_parity.py.
    reference = json.loads(Path(REFERENCE_REMOTE).read_text())
    ref_tools = json.loads(reference["toolsJSON"])
    for ref_case in reference["cases"]:
        rendered = tokenizer.apply_chat_template(
            ref_case["messages"], tools=ref_tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        if rendered.startswith("<bos>"):      # HF emits BOS as text, llama.cpp as a token
            rendered = rendered[len("<bos>"):]
        assert rendered == ref_case["rendered"], (
            f"PROMPT PARITY FAILED for {ref_case['label']}: training-time rendering "
            f"does not match what the wallet app sends "
            f"({len(rendered)} vs {len(ref_case['rendered'])} chars)"
        )
    print(f"[train] prompt parity OK against the app renderer "
          f"({len(reference['cases'])} shapes)", flush=True)

    # Seeded holdout for the in-training eval_loss guard. Shuffled before slicing
    # because the JSONL is grouped by category — a tail slice would hold out one
    # category entirely and measure something else.
    shuffled = list(rows)
    random.Random(3407).shuffle(shuffled)
    n_hold = max(16, int(len(shuffled) * HOLDOUT_FRAC))
    hold_rows, train_rows = shuffled[:n_hold], shuffled[n_hold:]
    print(f"[train] split: {len(train_rows)} train / {len(hold_rows)} holdout "
          f"({HOLDOUT_FRAC:.0%})", flush=True)

    ds = Dataset.from_list([to_text(r) for r in train_rows])
    eval_ds = Dataset.from_list([to_text(r) for r in hold_rows])
    # Eyeball the exact turn markers the template produced (so INSTRUCTION_PART /
    # RESPONSE_PART can be corrected if the template ever changes).
    print(f"[train] rendered sample head:\n{ds[0]['text'][:600]}", flush=True)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds, eval_dataset=eval_ds,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.05, num_train_epochs=epochs, learning_rate=learning_rate,
            logging_steps=5, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="linear", seed=3407, output_dir=run_dir,
            report_to="none",
            # Evaluate and save PER EPOCH, and keep every epoch. The sweep needs all
            # of them on disk: eval_loss picks one checkpoint, dev-set accuracy may
            # pick another, and that disagreement is itself the result we are after.
            # `load_best_model_at_end` additionally requires the two strategies to
            # match, which is why both are "epoch".
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=epochs,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Unsloth's documented settings for the eval loop, which OOMs readily:
            # keep the eval batch at 2 and accumulate. bf16 (not fp16) because this
            # is an A100.
            bf16_full_eval=True,
            per_device_eval_batch_size=2,
            eval_accumulation_steps=4,
        ),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE)],
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
    print(f"[train] all-masked rows: {len(dropped)}/{len(train_rows)} "
          f"-> {dict(Counter(train_rows[i]['category'] for i in dropped))}", flush=True)
    if len(dropped) > 0.5 * len(train_rows):
        raise SystemExit(
            f"{len(dropped)}/{len(train_rows)} rows fully masked — response markers "
            f"({INSTRUCTION_PART!r}/{RESPONSE_PART!r}) do not match the template"
        )

    stats = trainer.train()
    print(f"[train] final loss: {stats.training_loss:.4f}", flush=True)

    # PROVE eval_loss was actually produced. Unsloth issue #1019 ("No Validation
    # Loss logged (possibly related to train_on_responses_only?)") is still labelled
    # "fixed - pending confirmation", and this script does use
    # train_on_responses_only — so a silently absent eval_loss would leave
    # load_best_model_at_end and EarlyStoppingCallback selecting on nothing while
    # appearing to work. Fail loudly instead: a sweep whose selector is inert is
    # worse than no sweep, because the numbers still look meaningful.
    evals = [(e.get("epoch"), e["eval_loss"])
             for e in trainer.state.log_history if "eval_loss" in e]
    if not evals:
        raise SystemExit(
            "no eval_loss in trainer.state.log_history — the evaluation loop did "
            "not report. Do not trust load_best_model_at_end or early stopping "
            "until this is resolved (see unslothai/unsloth#1019)."
        )
    print("[train] eval_loss by epoch: "
          + "  ".join(f"e{ep:.0f}={loss:.4f}" for ep, loss in evals), flush=True)
    best = min(evals, key=lambda t: t[1])
    print(f"[train] best eval_loss epoch={best[0]:.0f} loss={best[1]:.4f} "
          f"(load_best_model_at_end restored this one)", flush=True)
    # NOTE: best-by-eval_loss is NOT necessarily best-by-task-accuracy. The holdout
    # is in-distribution (85.9% single-turn), so it cannot see a multi-round
    # collapse. finetune/modal_eval_gemma4.py scores every checkpoint below against
    # pf/tests.dev.yaml, and that is the selector of record.

    # Save the LoRA adapter to a STABLE path FIRST (export/eval read this — no
    # checkpoint number to track), then commit — so nothing below can cost us the
    # trained weights. GGUF is produced separately by the bf16 export (merge trap).
    model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    # Commit the per-epoch checkpoints too — modal_eval_gemma4.py scores each of
    # them against the dev set, and an uncommitted checkpoint dies with the
    # container.
    outputs.commit()
    ckpts = sorted(str(p) for p in Path(run_dir).glob("checkpoint-*"))
    print(f"[train] adapter saved -> {adapter_out}", flush=True)
    print(f"[train] per-epoch checkpoints: {ckpts or '(none)'}", flush=True)

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

    return (f"final_loss={stats.training_loss:.4f} "
            f"best_eval_loss=e{best[0]:.0f}/{best[1]:.4f} "
            f"probes={hits}/{probed} adapter={adapter_out}")


@app.local_entrypoint()
def main(epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE,
         tag: str = "") -> None:
    """Phase 1 sweep is one command per point, e.g.

        modal run finetune/modal_finetune_gemma4.py --epochs 3 --tag e3-lr2e4
        modal run finetune/modal_finetune_gemma4.py \
            --epochs 3 --learning-rate 5e-5 --tag e3-lr5e5

    `--tag` keeps each run's adapter and checkpoints separate; without it the path
    stays the historical /outputs/adapter that the export script reads.
    """
    # spawn() submits and returns immediately, so the run does not depend on the
    # local client's streaming connection — a dropped connection was cancelling
    # .remote() runs ~30 min in.
    #
    # BUT spawn() ALONE IS NOT ENOUGH. This is an EPHEMERAL app, and Modal stops an
    # ephemeral app when its local entrypoint returns — taking the spawned function
    # with it. Observed: a launch without --detach reached "Stopping app - local
    # entrypoint completed" and the app went to `stopped` with 0 tasks, having
    # trained nothing. It must be launched as:
    #
    #     uv run --with modal modal run --detach \
    #         finetune/modal_finetune_gemma4.py --epochs 3 --tag e3-lr2e4
    #
    # `modal app list` then shows "ephemeral (detached)" with 1 task, which is the
    # state to check for. Poll `modal volume ls gemma4-ft-outputs /` for the adapter.
    call = train.spawn(epochs=epochs, learning_rate=learning_rate, tag=tag)
    print(f"SPAWNED train call_id={call.object_id}")
    print("NOTE: this only survives if you launched with `modal run --detach`; "
          "an ephemeral app is stopped when this entrypoint returns. Verify with "
          "`modal app list` — the row must read 'ephemeral (detached)' with 1 task.")
