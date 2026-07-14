"""Produce a CORRECT GGUF from the fine-tuned adapter and upload it to HuggingFace.

The bug we found: unsloth's LoRA->16bit merge corrupts gemma3 (fp16 overflow), so
its GGUF is garbage while the adapter is fine. Fix: merge in BF16 with plain peft
(same exponent range as fp32 -> no overflow), convert to GGUF q8_0 via llama.cpp,
sanity-check it emits real tool calls, then upload from Modal (no local download).

Uploads to a public HF model repo:
  - functiongemma-270m-wallet-ft.Q8_0.gguf  (deployable, drops into the provider)
  - adapter/  (the proven-good LoRA adapter)

The HF token is read from the local ~/.cache/huggingface/token and passed to the
container as a Modal Secret (never printed).

Run:  uv run --with modal modal run finetune/modal_export.py
"""
from __future__ import annotations

from pathlib import Path

import modal

BASE_MODEL = "unsloth/functiongemma-270m-it"
ADAPTER = "/outputs/checkpoint-309"
HF_REPO = "gabrielfior/functiongemma-270m-wallet-ft"
GGUF_NAME = "functiongemma-270m-wallet-ft.Q8_0.gguf"
_REPO = Path(__file__).resolve().parent.parent

hf_cache = modal.Volume.from_name("functiongemma-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("functiongemma-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "transformers", "peft", "sentencepiece", "gguf",
                 "huggingface_hub", "numpy", "protobuf")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp",
        "pip install -r /llama.cpp/requirements/requirements-convert_hf_to_gguf.txt",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_file(str(_REPO / "finetune" / "diag_sample.jsonl"), "/data/diag.jsonl")
)

app = modal.App("functiongemma-export")


@app.function(image=image, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs})
def export_and_upload(hf_token: str) -> str:
    import json
    import subprocess

    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1. bf16 merge (the fix) — plain peft, no unsloth.
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    merged_dir = "/outputs/merged_bf16"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    print("[export] merged bf16 saved", flush=True)

    # 2. sanity-check the merged model emits a real call before we ship it — using
    # a REAL training example (messages + tools), exactly how it was evaluated.
    diag = [json.loads(l) for l in Path("/data/diag.jsonl").read_text().splitlines() if l.strip()]
    ex = next(r for r in diag if "<start_function_call>" in r["messages"][-1]["content"])
    enc = tok.apply_chat_template(ex["messages"][:-1], tools=ex["tools"],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)
    out = merged.generate(**enc, max_new_tokens=120, do_sample=False)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    print(f"[export] sanity ({ex['id']}) gen: {gen[:180]!r}", flush=True)
    if "<start_function_call>" not in gen:
        raise SystemExit("merged model did not emit a tool call — refusing to upload")

    # 3. convert to GGUF q8_0 via llama.cpp (no C++ build needed for this outtype).
    gguf_path = f"/outputs/{GGUF_NAME}"
    subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", merged_dir,
                    "--outfile", gguf_path, "--outtype", "q8_0"], check=True)
    size_mb = Path(gguf_path).stat().st_size / 1e6
    print(f"[export] GGUF {gguf_path} ({size_mb:.1f} MB)", flush=True)
    outputs.commit()

    # 4. upload to HF (GGUF + adapter). Needs a WRITE token.
    api = HfApi(token=hf_token)
    api.create_repo(HF_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=gguf_path, path_in_repo=GGUF_NAME, repo_id=HF_REPO)
    api.upload_folder(folder_path=ADAPTER, path_in_repo="adapter", repo_id=HF_REPO)
    url = f"https://huggingface.co/{HF_REPO}"
    print(f"[export] uploaded -> {url}", flush=True)
    return json.dumps({"repo": url, "gguf_mb": round(size_mb, 1), "sanity": gen[:120]})


@app.local_entrypoint()
def main() -> None:
    # Read the HF token LOCALLY only (the container has no token file) and pass it
    # to the remote function as an argument.
    token = (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()
    print("RESULT:", export_and_upload.remote(hf_token=token))
