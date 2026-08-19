#!/usr/bin/env python3
"""Container-side job: train Gemma-4 v5 on a rented RunPod GPU and export GGUFs.

Runs UNATTENDED inside the pod. Ported from finetune/modal_finetune_gemma4.py +
finetune/modal_export_gemma4.py, which are the proven recipes — every constant here
that looks arbitrary was paid for, and the source comments explain why. Modal is out
(no credits), so the two Modal Volumes are replaced by:

  * `/workspace` for anything that must survive a step (weights, GGUFs);
  * a private HF repo for the job bundle IN and the artifacts OUT — the pod dies at
    the end of the run and nothing on its disk is recoverable afterwards.

Everything is logged to /workspace/logs/train.log, which the pod serves over HTTP for
the whole run. RunPod's REST API has no logs endpoint, so without that a failure is
just an unreachable pod.

Phases, each one skippable via --only so a late failure does not re-run the GPU hour:

  parity  assert training-time rendering matches the bytes the wallet app sends
  train   1 epoch LoRA (see EPOCHS: 3 epochs cost 14.5 points of OOD accuracy)
  merge   bf16 merge with plain peft, once per alpha scale
  gguf    convert to f16 then llama-quantize to Q4_K_M (what the wallet ships)
  upload  push adapter + GGUFs to the private HF repo
"""
from __future__ import annotations

# UNSLOTH MUST BE IMPORTED BEFORE TRL. unsloth patches TRL's classes at import time;
# with the order reversed the names bind the UNPATCHED classes and the run dies with
# `eos_token '<EOS_TOKEN>' not found in vocabulary`. Three symptom-level fixes failed
# before the import order turned out to be the cause — do not alphabetise these.
import unsloth  # noqa: F401  isort:skip
from unsloth import FastModel  # noqa: E402  isort:skip

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

BASE_MODEL = "unsloth/gemma-4-E4B-it"
MAX_SEQ_LEN = 2048
EPOCHS = 1
BATCH = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
HOLDOUT_FRAC = 0.10
RANK = 16
LORA_ALPHA = 16
SEED = 3407

# Gemma-4 chat-template turn markers. Wrong markers mask every label to -100 and the
# run silently learns nothing, so the all-masked count is asserted below.
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"

WORK = Path("/workspace")
JOB = WORK / "job"
OUT = WORK / "out"
LLAMA = Path("/llama.cpp")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kw) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------
def check_parity(tokenizer) -> None:
    """Fail BEFORE the GPU spend if rendering drifted from what the app sends.

    `enable_thinking=True` is not optional: the app renders with thinking on
    (SamplerOptions.enableThinking defaults true), which puts a `<|think|>` marker
    right after `<|turn>system`. Every fine-tune before 2026-08-15 was trained on a
    prompt the app never sends because this kwarg was missing.
    """
    ref = json.loads((JOB / "app_contract_reference.json").read_text())
    tools = json.loads(ref["toolsJSON"])
    for case in ref["cases"]:
        rendered = tokenizer.apply_chat_template(
            case["messages"], tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        if rendered.startswith("<bos>"):   # HF emits BOS as text, llama.cpp as a token
            rendered = rendered[len("<bos>"):]
        if rendered != case["rendered"]:
            raise SystemExit(
                f"PROMPT PARITY FAILED for {case['label']}: "
                f"{len(rendered)} chars vs the app's {len(case['rendered'])}"
            )
    log(f"parity OK against the app renderer ({len(ref['cases'])} shapes)")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def do_train(args) -> Path:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, full_finetuning=False,
    )
    check_parity(tokenizer)

    # `--mlp-only` drops the attention projections from the LoRA target set. Motive,
    # from results/v4-failure-anatomy.md: 21 of v4's 27 wrong recipients were a
    # single-character corruption of the correct 40-hex address (a dropped 'b', an
    # inserted space, an inserted 'm' that is not even hex) — a defect base does not
    # have. Verbatim copying is an attention behaviour, so an adapter that leaves
    # attention alone may keep it. Untested; that is what the dev set is for.
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=not args.mlp_only,
        finetune_mlp_modules=True,
        r=RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )

    rows = [json.loads(l) for l in (JOB / "gemma4_train.jsonl").read_text().splitlines()
            if l.strip()]
    log(f"{len(rows)} training rows")

    def to_text(ex: dict) -> dict:
        text = tokenizer.apply_chat_template(ex["messages"], tools=ex["tools"],
                                            tokenize=False, enable_thinking=True)
        assert ex["messages"][-1]["content"] in text, f"target vanished: {ex['id']}"
        return {"text": text}

    # Shuffle BEFORE slicing the holdout: the JSONL is grouped by category, so a tail
    # slice would hold out one category entirely and measure something else.
    shuffled = list(rows)
    random.Random(SEED).shuffle(shuffled)
    n_hold = max(16, int(len(shuffled) * HOLDOUT_FRAC))
    hold_rows, train_rows = shuffled[:n_hold], shuffled[n_hold:]
    log(f"split: {len(train_rows)} train / {len(hold_rows)} holdout")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=Dataset.from_list([to_text(r) for r in train_rows]),
        eval_dataset=Dataset.from_list([to_text(r) for r in hold_rows]),
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.05, num_train_epochs=args.epochs,
            learning_rate=args.lr, logging_steps=5, optim="adamw_8bit",
            weight_decay=0.01, lr_scheduler_type="linear", seed=SEED,
            output_dir=str(OUT / "run"), report_to="none",
            eval_strategy="epoch", save_strategy="no",
            bf16_full_eval=True, per_device_eval_batch_size=2,
            eval_accumulation_steps=4,
        ),
    )
    # Loss on the assistant response only.
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

    dropped = [i for i, ex in enumerate(trainer.train_dataset)
               if all(t == -100 for t in ex["labels"])]
    log(f"all-masked rows: {len(dropped)}/{len(train_rows)} "
        f"-> {dict(Counter(train_rows[i]['category'] for i in dropped))}")
    if len(dropped) > 0.5 * len(train_rows):
        raise SystemExit(
            f"{len(dropped)}/{len(train_rows)} rows fully masked — the response "
            f"markers ({INSTRUCTION_PART!r}/{RESPONSE_PART!r}) do not match the "
            f"template, so this run would learn nothing"
        )

    stats = trainer.train()
    log(f"final training loss: {stats.training_loss:.4f}")
    evals = [(e.get("epoch"), e["eval_loss"])
             for e in trainer.state.log_history if "eval_loss" in e]
    # eval_loss is a DIAGNOSTIC, never a selector: it is anti-correlated with OOD
    # accuracy here (0.0068 -> 67.6%, 0.0016 -> 53.1%). Its absence would still mean
    # the holdout was never scored, so demand it.
    if not evals:
        raise SystemExit("no eval_loss logged — the evaluation loop never ran "
                         "(unslothai/unsloth#1019)")
    log("eval_loss by epoch: " + "  ".join(f"e{ep:.0f}={v:.4f}" for ep, v in evals))

    adapter = OUT / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    log(f"adapter saved -> {adapter}")
    return adapter


# ---------------------------------------------------------------------------
# merge + gguf
# ---------------------------------------------------------------------------
def scaled_adapter(adapter: Path, scale: float) -> Path:
    """A copy of the adapter whose effective LoRA scaling is multiplied by `scale`.

    peft computes scaling = lora_alpha / r at LOAD time, so rewriting `lora_alpha` is
    exactly the alpha_scale knob that recovered +29.7 points on the v4 adapter with no
    retraining (results/dev-alpha.e1-lr2e4.md) — no need to touch ~294 layers by hand.
    """
    if scale == 1.0:
        return adapter
    dst = adapter.parent / f"adapter-a{scale:g}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(adapter, dst)
    cfg_path = dst / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["lora_alpha"] = cfg["lora_alpha"] * scale
    cfg_path.write_text(json.dumps(cfg, indent=2))
    log(f"alpha-scaled adapter {scale:g}: lora_alpha -> {cfg['lora_alpha']}")
    return dst


def do_export(adapter: Path, scale: float, tag: str) -> Path:
    """bf16 merge -> f16 GGUF -> Q4_K_M.

    THE MERGE TRAP: unsloth's LoRA->16bit merge corrupts gemma3-family weights (fp16
    overflow) while the adapter itself is fine, so the merge goes through plain peft
    in BF16 — same exponent range as fp32, no overflow. And convert_hf_to_gguf cannot
    emit a k-quant directly, hence the separate llama-quantize step.
    """
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    src = scaled_adapter(adapter, scale)
    merged_dir = OUT / f"merged-{tag}"
    gguf_path = OUT / f"gemma-4-E4B-wallet-ft-{tag}.Q4_K_M.gguf"
    f16_path = OUT / f"f16-{tag}.gguf"
    if gguf_path.exists():
        log(f"{gguf_path.name} already present — skipping export")
        return gguf_path

    # gemma4 is NOT a native transformers arch; unsloth registers it on import, so the
    # base must come through FastModel (AutoModel raises KeyError('gemma4')). Kwargs are
    # kept identical to the proven modal_export_gemma4.py call — no `dtype=` and no
    # post-merge `.to()`, both of which would be untested deviations on a path whose
    # failure mode is silently corrupted weights.
    base, _ = FastModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, full_finetuning=False,
    )
    merged = PeftModel.from_pretrained(base, str(src)).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(str(src))
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tok.save_pretrained(str(merged_dir))
    log(f"merged (bf16, alpha x{scale:g}) -> {merged_dir}")

    # SANITY-CHECK THE MERGE BEFORE SHIPPING IT. The merge trap produces weights that
    # load and run without error and emit garbage, so "it converted" is not evidence.
    # Generate from a REAL training row, greedily, and demand an actual tool call.
    rows = [json.loads(l) for l in
            (JOB / "gemma4_train.jsonl").read_text().splitlines() if l.strip()]
    ex = next(r for r in rows if "<|tool_call>" in r["messages"][-1]["content"])
    enc = tok.apply_chat_template(ex["messages"][:-1], tools=ex["tools"],
                                  add_generation_prompt=True, enable_thinking=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(merged.device) for k, v in enc.items() if hasattr(v, "to")}
    out = merged.generate(**enc, max_new_tokens=220, do_sample=False)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    log(f"sanity ({ex['id']}) -> {gen[:260]!r}")
    if "<|tool_call>" not in gen:
        raise SystemExit(f"merged model (alpha x{scale:g}) emitted no tool call — "
                         f"refusing to export it")

    del merged, base
    gc.collect()
    torch.cuda.empty_cache()

    run([sys.executable, LLAMA / "convert_hf_to_gguf.py", merged_dir,
         "--outfile", f16_path, "--outtype", "f16"])
    run([LLAMA / "build" / "bin" / "llama-quantize", f16_path, gguf_path, "Q4_K_M"])
    f16_path.unlink(missing_ok=True)     # 16 GB we do not need past this point
    shutil.rmtree(merged_dir, ignore_errors=True)
    size_gb = gguf_path.stat().st_size / 1e9
    sha = subprocess.run(["sha256sum", str(gguf_path)], capture_output=True,
                         text=True).stdout.split()[0]
    log(f"GGUF {gguf_path.name} {size_gb:.2f} GB sha256={sha}")
    return gguf_path


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
def do_upload(paths: list[Path], repo: str, adapter: Path | None) -> None:
    """Push artifacts to a PRIVATE HF repo — the only storage that outlives the pod.

    `modal volume get` once returned a GGUF with the same byte count and a different
    sha256, and it loaded and ran without error. So the sha256 is logged at the source
    (above) and must be re-checked wherever the file is consumed.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    if adapter is not None:
        api.upload_folder(folder_path=str(adapter), path_in_repo="adapter",
                          repo_id=repo, repo_type="model")
        log(f"uploaded adapter -> {repo}/adapter")
    for p in paths:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                        repo_id=repo, repo_type="model")
        log(f"uploaded {p.name} -> {repo}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ef-dai-team/gemma-4-E4B-wallet-ft-v5")
    ap.add_argument("--epochs", type=float, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--alphas", default="1.0,0.75",
                    help="LoRA alpha scales to export, comma separated")
    ap.add_argument("--mlp-only", action="store_true",
                    help="LoRA on the MLP projections only (leave attention alone)")
    ap.add_argument("--only", default="",
                    help="comma-separated subset of: train,export,upload")
    args = ap.parse_args()
    phases = set(args.only.split(",")) if args.only else {"train", "export", "upload"}

    OUT.mkdir(parents=True, exist_ok=True)
    adapter = OUT / "adapter"
    if "train" in phases:
        adapter = do_train(args)
    scales = [float(s) for s in args.alphas.split(",") if s.strip()]
    ggufs: list[Path] = []
    if "export" in phases:
        for scale in scales:
            tag = f"a{scale:g}".replace(".", "")
            ggufs.append(do_export(adapter, scale, tag))
    if "upload" in phases:
        do_upload(ggufs, args.repo, adapter if "train" in phases else None)
    log("DONE")


if __name__ == "__main__":
    main()
