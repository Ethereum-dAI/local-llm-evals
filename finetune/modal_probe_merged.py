"""Ask the merged v4 model what it actually generates, and get the text BACK.

The export's smoke gate rejected v4 twice. Its diagnostic print is truncated out
of `modal app logs` once the app completes, so the only evidence — what the model
emitted — was unrecoverable both times. This probe loads the same merged bf16
directory from the Volume and RETURNS the generations, so they arrive as the
function's value rather than as logs that can be dropped.

Deliberately probes both renderings, because the difference between them is the
whole hypothesis:
  * enable_thinking=True  -> the bytes the wallet app sends, and what v4 trained on
  * enable_thinking=False -> what the old smoke gate used

    uv run modal run finetune/modal_probe_merged.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

HF_CACHE_DIR = "/root/.cache/huggingface"
OUTPUTS_DIR = "/outputs"
MERGED_DIR = f"{OUTPUTS_DIR}/merged_bf16_appcontract_v4"
DATA_REMOTE = "/data/train.jsonl"

_REPO = Path(__file__).resolve().parent.parent

hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("gemma4-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("transformers", "torch", "accelerate", "huggingface_hub",
                 "sentencepiece", "protobuf", "jinja2", "timm")
    .env({"HF_HOME": HF_CACHE_DIR})
)

if modal.is_local():
    image = image.add_local_file(
        str(_REPO / "data_for_finetune" / "gemma4_train.jsonl"), DATA_REMOTE)

app = modal.App("gemma4-probe-merged")


@app.function(image=image, gpu="A100",
              volumes={HF_CACHE_DIR: hf_cache, OUTPUTS_DIR: outputs},
              timeout=3600)
def probe() -> dict:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rows = [json.loads(l) for l in Path(DATA_REMOTE).read_text().splitlines() if l.strip()]
    # One wallet-path row (app contract) and one protocol row, so a failure can be
    # attributed to a contract rather than to the model as a whole.
    wallet = next(r for r in rows
                  if not r["category"].startswith(("aave-", "safe-"))
                  and "<|tool_call>" in r["messages"][-1]["content"])
    proto = next(r for r in rows if r["category"].startswith(("aave-", "safe-")))

    tok = AutoTokenizer.from_pretrained(MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    out: dict = {}
    for label, row in (("wallet", wallet), ("protocol", proto)):
        for thinking in (True, False):
            enc = tok.apply_chat_template(
                row["messages"][:-1], tools=row["tools"],
                add_generation_prompt=True, enable_thinking=thinking,
                return_tensors="pt", return_dict=True)
            enc = {k: v.to(model.device) for k, v in enc.items() if hasattr(v, "to")}
            with torch.no_grad():
                gen_ids = model.generate(**enc, max_new_tokens=512, do_sample=False)
            gen = tok.decode(gen_ids[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=False)
            key = f"{label}-thinking={thinking}"
            out[key] = {
                "id": row["id"],
                "target": row["messages"][-1]["content"][:300],
                "generated": gen[:1500],
                "has_tool_call": "<|tool_call>" in gen,
                "len": len(gen),
            }
            print(f"[probe] {key}: has_tool_call={'<|tool_call>' in gen} len={len(gen)}",
                  flush=True)
    return out


@app.local_entrypoint()
def main() -> None:
    result = probe.remote()
    Path("/tmp/v4_probe.json").write_text(json.dumps(result, indent=2))
    for key, r in result.items():
        print("=" * 70)
        print(f"{key}  id={r['id']}  has_tool_call={r['has_tool_call']}  len={r['len']}")
        print(f"TARGET   : {r['target'][:200]!r}")
        print(f"GENERATED: {r['generated'][:700]!r}")
