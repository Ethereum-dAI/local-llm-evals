"""Average several LoRA checkpoints into one adapter ("model soup"), on Modal.

Weight averaging sometimes recovers generalization that any single checkpoint has
lost, for the cost of an elementwise mean — no training, no GPU. Worth trying here
because Phase 0c measured a monotonic slide across epochs on out-of-distribution
cases (67.6% -> 57.2% -> 53.1%) while in-distribution loss kept improving, so the
later checkpoints hold something the first one does not, even though on net they
score worse.

This is explicitly a lottery ticket, not a plan. Averaging LoRA adapters is only
approximately meaningful: the product B@A is what gets added to the base weights,
and mean(B)@mean(A) != mean(B@A). Two runs of the SAME shape (same r, same target
modules, same base) stay in a comparable enough basis for it to be worth a look,
which is why the shape check below is a hard failure rather than a warning. Do not
read a soup win as evidence about the recipe — score it on the dev set like any
other candidate.

Run:
    uv run --with modal modal run finetune/modal_soup_gemma4.py --tag e3-lr2e4
    uv run --with modal modal run finetune/modal_soup_gemma4.py \
        --tag e3-lr2e4 --out-tag soup-e12   --only "checkpoint-100,checkpoint-200"

Then score it exactly like a trained adapter:
    uv run --with modal modal run --detach finetune/modal_eval_gemma4.py \
        --tag soup-e3-lr2e4 --dataset pf/tests.dev.yaml --no-all-checkpoints
"""
from pathlib import Path

import modal

OUTPUTS_DIR = "/outputs"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "safetensors", "peft")
)

app = modal.App("gemma4-soup")
outputs = modal.Volume.from_name("gemma4-ft-outputs", create_if_missing=True)


@app.function(image=image, timeout=1800, volumes={OUTPUTS_DIR: outputs})
def soup(tag: str = "", out_tag: str = "", only: str = "") -> dict:
    """Elementwise-average every checkpoint of `tag` into /outputs/adapter-<out_tag>.

    CPU only — this is a mean over a few hundred small tensors, and paying for a GPU
    to do it would be theatre.
    """
    import json
    import os
    import re
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    run_dir = f"{OUTPUTS_DIR}/run-{tag}" if tag else OUTPUTS_DIR
    if not os.path.isdir(run_dir):
        raise SystemExit(f"no run directory {run_dir} — is --tag right?")

    wanted = [x.strip() for x in only.split(",") if x.strip()]
    names = sorted((d for d in os.listdir(run_dir)
                    if re.fullmatch(r"checkpoint-\d+", d)),
                   key=lambda d: int(d.split("-")[1]))
    if wanted:
        missing = [w for w in wanted if w not in names]
        if missing:
            raise SystemExit(f"requested {missing} but {run_dir} has {names}")
        names = [n for n in names if n in wanted]
    if len(names) < 2:
        raise SystemExit(f"need >=2 checkpoints to average, found {names} in {run_dir}")

    paths = [f"{run_dir}/{n}" for n in names]
    print(f"[soup] averaging {len(paths)} checkpoints: {names}", flush=True)

    # Averaging adapters trained with different shapes is meaningless, not merely
    # inaccurate — the tensors would not even be the same size. Fail loudly.
    configs = []
    for path in paths:
        with open(f"{path}/adapter_config.json") as fh:
            configs.append(json.load(fh))
    shape_keys = ("r", "lora_alpha", "target_modules", "base_model_name_or_path")
    def shape(cfg):
        return tuple(
            tuple(sorted(cfg[k])) if isinstance(cfg.get(k), list) else cfg.get(k)
            for k in shape_keys)
    shapes = {shape(c) for c in configs}
    if len(shapes) != 1:
        raise SystemExit(
            f"checkpoints do not share a LoRA shape ({shape_keys}): {shapes}")

    tensor_files = [f"{p}/adapter_model.safetensors" for p in paths]
    for f in tensor_files:
        if not os.path.isfile(f):
            raise SystemExit(f"missing {f} (only .safetensors adapters supported)")

    states = [load_file(f) for f in tensor_files]
    keys = set(states[0])
    for i, st in enumerate(states[1:], start=1):
        if set(st) != keys:
            raise SystemExit(
                f"{names[i]} has different tensor keys than {names[0]} "
                f"(+{sorted(set(st) - keys)[:5]} -{sorted(keys - set(st))[:5]})")

    averaged = {k: torch.stack([st[k].float() for st in states]).mean(0)
                   .to(states[0][k].dtype)
                for k in sorted(keys)}

    out_dir = f"{OUTPUTS_DIR}/adapter-{out_tag or f'soup-{tag}'}"
    os.makedirs(out_dir, exist_ok=True)
    # Copy the first checkpoint's metadata verbatim — the shape check above proved
    # they agree on everything that matters, so there is nothing to merge.
    for extra in ("adapter_config.json", "tokenizer_config.json", "tokenizer.json",
                  "special_tokens_map.json", "chat_template.jinja"):
        src = f"{paths[0]}/{extra}"
        if os.path.isfile(src):
            shutil.copy(src, f"{out_dir}/{extra}")
    save_file(averaged, f"{out_dir}/adapter_model.safetensors")
    outputs.commit()

    summary = {"out": out_dir, "averaged": names, "tensors": len(averaged)}
    print(f"[soup] SUMMARY: {json.dumps(summary)}", flush=True)
    print(f"[soup] score it:  modal run --detach finetune/modal_eval_gemma4.py "
          f"--tag {out_tag or f'soup-{tag}'} --dataset pf/tests.dev.yaml "
          f"--no-all-checkpoints", flush=True)
    return summary


@app.local_entrypoint()
def main(tag: str = "", out_tag: str = "", only: str = "") -> None:
    print(soup.remote(tag=tag, out_tag=out_tag, only=only))
