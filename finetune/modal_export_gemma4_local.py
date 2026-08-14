"""Produce a CORRECT Q4_K_M GGUF from the app-contract fine-tuned Gemma-4 adapter
and leave it in the Modal outputs Volume for a LOCAL download — no HuggingFace
upload, ever.

This is a sibling of `modal_export_gemma4.py`, not a flag on it. That script
hardcodes `HF_REPO = "ef-dai-team/gemma-4-E4B-wallet-ft"` — the exact repo
local-wallet-mac downloads and content-pins by SHA256 in production. This
app-contract fine-tune is an experiment (human-unit amounts, verbatim
recipients, rewritten <think> traces — see finetune/README_gemma4.md and
finetune/README.md) and must never touch that repo, or any HF repo. So this
script strips the entire upload path rather than gating it behind a flag that
could be flipped by accident.

The gemma3-family "merge trap" and the merge/convert/quantize recipe are
UNCHANGED from `modal_export_gemma4.py` — reuse exactly, do not "improve":
unsloth's LoRA->16bit merge corrupts gemma3-family weights (fp16 overflow), so
merge in BF16 with plain peft (same exponent range as fp32 -> no overflow),
convert to an f16 GGUF via llama.cpp's convert_hf_to_gguf, then `llama-quantize`
to Q4_K_M (a k-quant convert_hf_to_gguf can't emit directly).

The smoke test is also kept, repurposed: the original refuses to UPLOAD a model
that doesn't emit a tool call. There is nothing to upload here, so this script
instead refuses to PUBLISH THE ARTIFACT AS GOOD — it still runs the merge,
convert and quantize (so you get a GGUF to inspect either way), but exits
non-zero and prints a loud failure banner instead of the success/sha256
summary. A silently broken GGUF that then gets evaluated would produce a
meaningless final number, so treat a non-zero exit here as "do not eval this
file".

Output lands at gemma4-ft-outputs:/outputs/gguf/gemma4-e4b-wallet-ft-appcontract.Q4_K_M.gguf
— a name that cannot collide with the production
`gemma-4-E4B-wallet-ft.Q4_K_M.gguf` (different subdir, different stem).

Run (spawns detached — a dropped client connection can't cancel the ~20-40 min
GPU job; poll `modal app logs` for "[export-local] DONE" or
"[export-local] SMOKE TEST FAILED"):

    uv run --with modal modal run finetune/modal_export_gemma4_local.py

Then pull the finished GGUF down into this repo's (gitignored) models/ dir —
the exact command is also printed by the run itself once it succeeds:

    uv run --with modal modal volume get gemma4-ft-outputs \\
        gguf/gemma4-e4b-wallet-ft-appcontract.Q4_K_M.gguf \\
        models/gemma4-e4b-wallet-ft-appcontract.Q4_K_M.gguf

The run prints the GGUF's size and SHA256 (computed in-container, over the
Volume copy — the referee for provenance disputes, same reasoning as
modal_hash_gguf.py) so the downloaded file can be checked against what Modal
actually produced:

    shasum -a 256 models/gemma4-e4b-wallet-ft-appcontract.Q4_K_M.gguf
"""
from __future__ import annotations

from pathlib import Path

import modal

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bundled import bundled  # noqa: E402  (needs the line above)

BASE_MODEL = "unsloth/gemma-4-E4B-it"
OUTPUTS_DIR = "/outputs"
GGUF_SUBDIR = "gguf"
GGUF_NAME = "gemma4-e4b-wallet-ft-appcontract.Q4_K_M.gguf"
_REPO = Path(__file__).resolve().parent.parent

# Same Volumes as modal_export_gemma4.py / modal_finetune_gemma4.py — this reads
# the adapter the app-contract training run wrote to gemma4-ft-outputs:/outputs/adapter
# and reuses the same HF weights cache. No new Volume needed.
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
    # FastModel — which requires torchvision for gemma4's vision processor.
    # NOTE: no huggingface_hub needed here — there is nothing to upload.
    .pip_install("unsloth", "sentencepiece", "gguf", "protobuf", "numpy")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_file(str(bundled(_REPO, "data_for_finetune/gemma4_train.jsonl",
                                       "data/gemma4_train.jsonl")), "/data/train.jsonl")
)

app = modal.App("gemma4-export-local")


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
def export_local() -> str:
    import hashlib
    import json
    import subprocess

    # `gemma4` is NOT a native transformers arch here — unsloth registers it on
    # import. So the base MUST be loaded via unsloth's FastModel (plain
    # transformers AutoModel raises KeyError('gemma4')). We load in bf16 and merge
    # with peft in bf16 — same exponent range as fp32, dodging the fp16 merge trap.
    from unsloth import FastModel
    from peft import PeftModel
    from transformers import AutoTokenizer

    # 1. bf16 merge — identical recipe to modal_export_gemma4.py.
    adapter = _resolve_adapter()
    print(f"[export-local] adapter = {adapter}", flush=True)
    base, _ = FastModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048,
        load_in_4bit=False, full_finetuning=False,
    )
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(adapter)
    merged_dir = "/outputs/merged_bf16_appcontract"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    print("[export-local] merged bf16 saved", flush=True)

    # 2. sanity-check the merged model emits a real call — using a REAL training
    # example (messages + tools), exactly how it was evaluated. This is the same
    # gate as modal_export_gemma4.py, repurposed: there is nothing to upload, so
    # a failure here means "do not trust/evaluate this GGUF", not "do not upload".
    diag = [json.loads(l) for l in Path("/data/train.jsonl").read_text().splitlines() if l.strip()]
    ex = next(r for r in diag if "<|tool_call>" in r["messages"][-1]["content"])
    enc = tok.apply_chat_template(ex["messages"][:-1], tools=ex["tools"],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)
    enc = {k: v.to(merged.device) for k, v in enc.items() if hasattr(v, "to")}
    out = merged.generate(**enc, max_new_tokens=220, do_sample=False)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    print(f"[export-local] sanity ({ex['id']}) gen: {gen[:240]!r}", flush=True)
    smoke_ok = "<|tool_call>" in gen

    # 3a. convert to an f16 GGUF (convert_hf_to_gguf can't emit k-quants directly).
    f16_path = "/outputs/gemma4-e4b-wallet-ft-appcontract.f16.gguf"
    subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", merged_dir,
                    "--outfile", f16_path, "--outtype", "f16"], check=True)
    # 3b. quantize to Q4_K_M (same quant the wallet ships, for an apples-to-apples eval).
    gguf_dir = f"{OUTPUTS_DIR}/{GGUF_SUBDIR}"
    subprocess.run(["mkdir", "-p", gguf_dir], check=True)
    gguf_path = f"{gguf_dir}/{GGUF_NAME}"
    subprocess.run(["/llama.cpp/build/bin/llama-quantize", f16_path, gguf_path,
                    "Q4_K_M"], check=True)
    size_bytes = Path(gguf_path).stat().st_size
    size_mb = size_bytes / 1e6
    print(f"[export-local] GGUF {gguf_path} ({size_mb:.1f} MB)", flush=True)

    # 4. hash the Volume copy, not a re-download — same referee logic as
    # modal_hash_gguf.py: the Volume is the source of truth for what Modal
    # actually produced.
    h = hashlib.sha256()
    with Path(gguf_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    sha256 = h.hexdigest()

    outputs.commit()

    volume_path = f"{GGUF_SUBDIR}/{GGUF_NAME}"
    download_cmd = (
        "uv run --with modal modal volume get gemma4-ft-outputs "
        f"{volume_path} models/{GGUF_NAME}"
    )

    if not smoke_ok:
        print("[export-local] SMOKE TEST FAILED — merged model did not emit a "
              "tool call. The GGUF was still produced (for inspection) at "
              f"{gguf_path}, sha256={sha256}, {size_bytes} bytes, but it must "
              "NOT be treated as a good artifact or scored in the benchmark.",
              flush=True)
        raise SystemExit(
            "merged model did not emit a tool call — refusing to publish as good"
        )

    print(f"[export-local] DONE — sha256={sha256}", flush=True)
    print(f"[export-local] size = {size_bytes} bytes ({size_mb:.1f} MB)", flush=True)
    print(f"[export-local] download with:\n  {download_cmd}", flush=True)
    return json.dumps({
        "volume": "gemma4-ft-outputs",
        "volume_path": volume_path,
        "size_bytes": size_bytes,
        "sha256": sha256,
        "download_cmd": download_cmd,
        "sanity": gen[:160],
    })


@app.local_entrypoint()
def main() -> None:
    # spawn + --detach (implicit via .spawn()) so a dropped client connection
    # can't cancel the ~20-40 min job; progress + the final sha256/size are
    # printed to the logs ("[export-local] DONE"). Read with `modal app logs`.
    # No token to read locally — there is nothing to authenticate to upload.
    call = export_local.spawn()
    print(f"SPAWNED export-local call_id={call.object_id} — poll logs for "
          f"'[export-local] DONE' (success) or "
          f"'[export-local] SMOKE TEST FAILED' (do not evaluate the GGUF).")
