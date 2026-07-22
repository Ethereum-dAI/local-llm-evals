"""Produce a CORRECT Q4_K_M GGUF from the fine-tuned Gemma-4 adapter and upload it
to HuggingFace — the quant local-wallet-mac actually ships (and issue #8 wants
benchmarked).

The gemma3-family "merge trap": unsloth's LoRA->16bit merge corrupts the weights
(fp16 overflow), so its GGUF is garbage while the adapter is fine. Fix: merge in
BF16 with plain peft (same exponent range as fp32 -> no overflow), convert to an
f16 GGUF via llama.cpp's convert_hf_to_gguf, then `llama-quantize` to Q4_K_M
(a k-quant convert_hf_to_gguf can't emit directly). Sanity-check it emits a real
tool call, then upload from Modal (no local download).

Uploads to a public HF model repo:
  - gemma-4-E4B-wallet-ft.Q4_K_M.gguf  (deployable, drops into the provider)
  - adapter/  (the proven-good LoRA adapter)

Run:  uv run --with modal modal run finetune/modal_export_gemma4.py
"""
from __future__ import annotations

from pathlib import Path

import modal

BASE_MODEL = "unsloth/gemma-4-E4B-it"
OUTPUTS_DIR = "/outputs"
HF_REPO = "gabrielfior/gemma-4-E4B-wallet-ft"
GGUF_NAME = "gemma-4-E4B-wallet-ft.Q4_K_M.gguf"
_REPO = Path(__file__).resolve().parent.parent

hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("gemma4-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "libcurl4-openssl-dev")
    # llama.cpp: clone + build llama-quantize for the Q4_K_M k-quant step. We do
    # NOT `pip install` its convert requirements — that pulls a CPU-only torch that
    # shadows unsloth's CUDA torch (→ unsloth "cannot find any torch accelerator").
    # convert_hf_to_gguf.py only needs numpy/torch/sentencepiece/gguf, all provided
    # below by unsloth (torch, transformers, numpy) + the explicit extras.
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp",
        "cmake -S /llama.cpp -B /llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF",
        "cmake --build /llama.cpp/build --target llama-quantize -j",
    )
    # unsloth registers the Gemma-4 arch (not native to transformers) and brings a
    # coherent CUDA torch + matching torchvision, so the base load goes through
    # FastModel — which requires torchvision for gemma4's vision processor. (Keep
    # torchvision: the earlier `torchvision::nms` ABI crash was only because the
    # convert-requirements CPU torch shadowed unsloth's CUDA torch; now fixed.)
    .pip_install("unsloth", "sentencepiece", "gguf", "huggingface_hub",
                 "protobuf", "numpy")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_file(str(_REPO / "data_for_finetune" / "gemma4_train.jsonl"),
                    "/data/train.jsonl")
)

app = modal.App("gemma4-export")


def _resolve_adapter() -> str:
    """Prefer the stable /outputs/adapter path; fall back to the newest
    checkpoint-N the trainer wrote (the probe-loop bug can skip the final save)."""
    import os
    import re
    stable = f"{OUTPUTS_DIR}/adapter"
    if os.path.isfile(f"{stable}/adapter_config.json"):
        return stable
    cks = [d for d in os.listdir(OUTPUTS_DIR) if re.fullmatch(r"checkpoint-\d+", d)]
    if not cks:
        raise SystemExit(f"no adapter or checkpoint-* under {OUTPUTS_DIR}")
    newest = max(cks, key=lambda d: int(d.split("-")[1]))
    return f"{OUTPUTS_DIR}/{newest}"


@app.function(image=image, gpu="A10G", timeout=5400,
              volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs})
def export_and_upload(hf_token: str) -> str:
    import json
    import subprocess

    # `gemma4` is NOT a native transformers arch here — unsloth registers it on
    # import. So the base MUST be loaded via unsloth's FastModel (plain
    # transformers AutoModel raises KeyError('gemma4')). We load in bf16 and merge
    # with peft in bf16 — same exponent range as fp32, dodging the fp16 merge trap.
    from unsloth import FastModel
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoTokenizer

    # 1. bf16 merge.
    adapter = _resolve_adapter()
    print(f"[export] adapter = {adapter}", flush=True)
    base, _ = FastModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048,
        load_in_4bit=False, full_finetuning=False,
    )
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(adapter)
    merged_dir = "/outputs/merged_bf16"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    print("[export] merged bf16 saved", flush=True)

    # 2. sanity-check the merged model emits a real call before we ship it — using
    # a REAL training example (messages + tools), exactly how it was evaluated.
    diag = [json.loads(l) for l in Path("/data/train.jsonl").read_text().splitlines() if l.strip()]
    ex = next(r for r in diag if "<|tool_call>" in r["messages"][-1]["content"])
    enc = tok.apply_chat_template(ex["messages"][:-1], tools=ex["tools"],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)
    enc = {k: v.to(merged.device) for k, v in enc.items() if hasattr(v, "to")}
    out = merged.generate(**enc, max_new_tokens=220, do_sample=False)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    print(f"[export] sanity ({ex['id']}) gen: {gen[:240]!r}", flush=True)
    if "<|tool_call>" not in gen:
        raise SystemExit("merged model did not emit a tool call — refusing to upload")

    # 3a. convert to an f16 GGUF (convert_hf_to_gguf can't emit k-quants directly).
    f16_path = "/outputs/gemma-4-E4B-wallet-ft.f16.gguf"
    subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", merged_dir,
                    "--outfile", f16_path, "--outtype", "f16"], check=True)
    # 3b. quantize to Q4_K_M (what the wallet ships).
    gguf_path = f"/outputs/{GGUF_NAME}"
    subprocess.run(["/llama.cpp/build/bin/llama-quantize", f16_path, gguf_path,
                    "Q4_K_M"], check=True)
    size_mb = Path(gguf_path).stat().st_size / 1e6
    print(f"[export] GGUF {gguf_path} ({size_mb:.1f} MB)", flush=True)
    outputs.commit()

    # 4. upload to HF (GGUF + adapter). Needs a WRITE token.
    api = HfApi(token=hf_token)
    api.create_repo(HF_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=gguf_path, path_in_repo=GGUF_NAME, repo_id=HF_REPO)
    api.upload_folder(folder_path=adapter, path_in_repo="adapter", repo_id=HF_REPO)
    url = f"https://huggingface.co/{HF_REPO}"
    print(f"[export] uploaded -> {url}", flush=True)
    return json.dumps({"repo": url, "gguf_mb": round(size_mb, 1), "sanity": gen[:160]})


@app.local_entrypoint()
def main() -> None:
    # Read the HF token LOCALLY only (the container has no token file) and pass it
    # to the remote function as an argument. spawn + --detach so a dropped client
    # connection can't cancel the ~20-min job; progress + the final repo URL are
    # printed to the logs ("[export] uploaded -> …"). Read with `modal app logs`.
    token = (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()
    call = export_and_upload.spawn(hf_token=token)
    print(f"SPAWNED export call_id={call.object_id} — poll logs for '[export] uploaded'.")
