"""Produce a Q4_K_M GGUF from the fine-tuned Qwen3-8B adapter.

Same shape as finetune/modal_export_gemma4.py and for the same reason: merge in
BF16 with plain peft (fp16 merging is the gemma3-family "merge trap" — same
exponent range as fp32, so bf16 dodges it), convert to f16 GGUF via llama.cpp's
convert_hf_to_gguf, then `llama-quantize` to Q4_K_M — the quant every other local
model in this comparison was measured at.

Two differences from the Gemma-4 job, both deliberate:

  * plain transformers, no unsloth. Qwen3 IS a native transformers architecture,
    unlike gemma4 (which only exists once unsloth registers it), so the extra
    dependency would add risk without buying anything.
  * no HuggingFace upload path used here. `upload()` still exists (opt-in,
    untouched) but the intended flow is: the GGUF stays on the outputs Volume
    and is pulled straight down with `modal volume get`.

**Stale-adapter guard.** The `qwen-ft-outputs` Volume is not cleared between
training runs — it can hold checkpoints/adapters/merges from a PREVIOUS run on
a DIFFERENT tool contract. `_resolve_adapter` used to trust `/outputs/adapter`
(or the newest `checkpoint-N`) unconditionally, which means a stale Volume
would silently merge the WRONG adapter under THIS run's (correctly-named)
filename — nothing downstream would notice. So `min_mtime` is a REQUIRED
argument to the export: the unix timestamp of when you launched the training
run whose adapter you want. Any candidate (the stable `adapter/` path
included) whose `adapter_config.json` predates that cutoff is rejected with a
loud `SystemExit` naming the candidate, its mtime, and the cutoff.

The GGUF name (`qwen3-8b-wallet-ft-appcontract.Q4_K_M.gguf`) is deliberately
different from the base run's `qwen3-8b-wallet-ft.Q4_K_M.gguf` so this export
can never overwrite or be confused with that previous artifact on the same
Volume; likewise the merged bf16 dir (`merged_bf16_appcontract`).

Run (required `--min-mtime`, the unix timestamp you launched THIS training
run, comfortably after any previous run on the Volume):
    uv run --with modal modal run --detach finetune/modal_export_qwen.py \
        --min-mtime <epoch>
    uv run --with modal modal volume get qwen-ft-outputs \
        qwen3-8b-wallet-ft-appcontract.Q4_K_M.gguf \
        models/qwen3-8b-wallet-ft-appcontract.Q4_K_M.gguf

On smoke-test failure (merged model doesn't emit a tool call) the GGUF is
still produced for inspection but renamed under a `REJECTED-` prefix so a
`modal volume get` by the normal name can't fetch it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bundled import bundled  # noqa: E402  (needs the line above)

BASE_MODEL = "unsloth/Qwen3-8B"
OUTPUTS_DIR = "/outputs"
GGUF_NAME = "qwen3-8b-wallet-ft-appcontract.Q4_K_M.gguf"
_REPO = Path(__file__).resolve().parent.parent

hf_cache = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("qwen-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "libcurl4-openssl-dev")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp",
        "cmake -S /llama.cpp -B /llama.cpp/build -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF",
        "cmake --build /llama.cpp/build --target llama-quantize -j",
    )
    .pip_install("torch", "transformers>=4.54", "peft", "accelerate",
                 "sentencepiece", "gguf", "huggingface_hub", "protobuf", "numpy")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_python_source("_bundled")
)

# `bundled()` is a LOCAL-only path helper — it resolves candidates relative to
# _REPO, which is only the repo root when this module is imported locally by
# the `modal run` CLI. Modal re-imports this same module INSIDE the container
# (to find the app/function objects after the image is already built), and
# there `__file__` is `/root/modal_export_qwen.py`, so `_REPO` becomes `/` and
# `bundled()` would raise FileNotFoundError before the container ever does
# anything useful. Gate on modal.is_local() (False inside a Function/container,
# True everywhere else) so `bundled()` and the `.add_local_file()` that
# consumes it are never evaluated remotely.
if modal.is_local():
    image = image.add_local_file(
        str(bundled(_REPO, "data_for_finetune/qwen_train.jsonl",
                            "data/qwen_train.jsonl")),
        "/data/train.jsonl",
    )

app = modal.App("qwen-export")


def _resolve_adapter(min_mtime: float = 0.0) -> tuple[str, float]:
    """Prefer the stable /outputs/adapter path; fall back to the newest
    checkpoint-N the trainer wrote (the probe-loop bug can skip the final save).

    Rejects ANY candidate — the stable path included — whose adapter_config.json
    is older than `min_mtime`. The outputs Volume persists across runs and is
    NOT cleared between them, so without this gate a failed/in-progress/never-
    finished training run — or simply a PREVIOUS run on a different tool
    contract — would silently fall through to a stale adapter, and this script
    would merge and export the WRONG model under the current run's (correctly-
    named) filename. `min_mtime` defaults to 0.0 (no-op gate) only so the
    unrelated `upload()` path below — which never calls export and is never
    invoked from this hardening — keeps working unchanged; `export()` always
    passes an explicit, required cutoff. Returns (path, mtime) so the caller
    can print/return which weights were actually merged.
    """
    import os
    import re
    from datetime import datetime, timezone

    def _fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def _check(path: str) -> float:
        cfg = f"{path}/adapter_config.json"
        if not os.path.isfile(cfg):
            raise SystemExit(f"{path} has no adapter_config.json")
        mtime = os.path.getmtime(cfg)
        if mtime < min_mtime:
            raise SystemExit(
                f"[export] REJECTED candidate adapter {path!r}: "
                f"adapter_config.json mtime {mtime:.0f} ({_fmt(mtime)}) is older "
                f"than --min-mtime cutoff {min_mtime:.0f} ({_fmt(min_mtime)}) — "
                "this looks like a stale adapter from a PREVIOUS run on the same "
                "Volume, refusing to merge it. Pass the unix timestamp of when "
                "you launched THIS training run."
            )
        return mtime

    stable = f"{OUTPUTS_DIR}/adapter"
    if os.path.isfile(f"{stable}/adapter_config.json"):
        mtime = _check(stable)
        print(f"[export] resolved adapter = {stable} "
              f"(mtime {mtime:.0f} = {_fmt(mtime)})", flush=True)
        return stable, mtime

    cks = [d for d in os.listdir(OUTPUTS_DIR) if re.fullmatch(r"checkpoint-\d+", d)]
    if not cks:
        raise SystemExit(f"no adapter or checkpoint-* under {OUTPUTS_DIR}")
    newest = max(cks, key=lambda d: int(d.split("-")[1]))
    candidate = f"{OUTPUTS_DIR}/{newest}"
    mtime = _check(candidate)
    print(f"[export] no stable /outputs/adapter yet — falling back to "
          f"checkpoint {candidate} (mtime {mtime:.0f} = {_fmt(mtime)})", flush=True)
    return candidate, mtime


@app.function(image=image, gpu="A100", timeout=7200,
              volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs})
def export(min_mtime: float) -> str:
    import hashlib
    import json
    import subprocess

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Adapter resolution is mtime-gated against min_mtime (see _resolve_adapter)
    # so a stale adapter left over from a previous run on this Volume — on a
    # DIFFERENT tool contract, in this case — can never be silently merged
    # under this run's filename.
    adapter, adapter_mtime = _resolve_adapter(min_mtime)
    print(f"[export] adapter = {adapter} (mtime {adapter_mtime:.0f})", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda")
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(adapter)
    merged_dir = f"{OUTPUTS_DIR}/merged_bf16_appcontract"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    print("[export] merged bf16 saved", flush=True)

    # Sanity: a REAL training row must still produce a real tool call after the
    # merge. This is the check that catches a corrupted merge before we spend an
    # hour benchmarking noise. Unlike before, a failure here no longer skips
    # straight to SystemExit — the GGUF is still produced (for inspection), then
    # quarantined under a REJECTED- name below, matching modal_export_gemma4_local.py.
    rows = [json.loads(l) for l in Path("/data/train.jsonl").read_text().splitlines()
            if l.strip()]
    ex = next(r for r in rows if "<tool_call>" in r["messages"][-1]["content"])
    enc = tok.apply_chat_template(ex["messages"][:-1], tools=ex["tools"],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)
    enc = {k: v.to(merged.device) for k, v in enc.items() if hasattr(v, "to")}
    out = merged.generate(**enc, max_new_tokens=320, do_sample=False)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    print(f"[export] sanity ({ex['id']}) gen: {gen[:300]!r}", flush=True)
    smoke_ok = "<tool_call>" in gen

    f16_path = f"{OUTPUTS_DIR}/qwen3-8b-wallet-ft-appcontract.f16.gguf"
    subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", merged_dir,
                    "--outfile", f16_path, "--outtype", "f16"], check=True)
    gguf_path = f"{OUTPUTS_DIR}/{GGUF_NAME}"
    subprocess.run(["/llama.cpp/build/bin/llama-quantize", f16_path, gguf_path,
                    "Q4_K_M"], check=True)
    size_bytes = Path(gguf_path).stat().st_size
    size_mb = size_bytes / 1e6
    # The f16 intermediate is ~16 GB and nothing downstream reads it.
    Path(f16_path).unlink(missing_ok=True)
    print(f"[export] GGUF {gguf_path} ({size_mb:.1f} MB)", flush=True)

    # Hash the Volume copy, not a re-download — same referee logic as
    # modal_hash_gguf.py: the Volume is the source of truth for what Modal
    # actually produced.
    h = hashlib.sha256()
    with Path(gguf_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    sha256 = h.hexdigest()

    if not smoke_ok:
        # Quarantine, don't just label-and-leave: a later `modal volume get`
        # that only pattern-matches GGUF_NAME must not be able to fetch this
        # file. Rename under a REJECTED- prefix (keeps it around for manual
        # inspection) rather than deleting outright.
        rejected_path = f"{OUTPUTS_DIR}/REJECTED-{GGUF_NAME}"
        Path(gguf_path).rename(rejected_path)
        outputs.commit()
        print("[export] SMOKE TEST FAILED — merged model did not emit a tool "
              f"call. The GGUF was still produced (for inspection) but renamed "
              f"to {rejected_path} so it cannot be fetched by the normal "
              f"download command. sha256={sha256}, {size_bytes} bytes. It must "
              "NOT be treated as a good artifact or scored in the benchmark.",
              flush=True)
        raise SystemExit(
            "merged model did not emit a tool call — refusing to publish as good"
        )

    outputs.commit()
    print(f"[export] DONE — adapter={adapter} (mtime {adapter_mtime:.0f}), "
          f"sha256={sha256}", flush=True)
    return json.dumps({"gguf": GGUF_NAME, "mb": round(size_mb, 1),
                       "bytes": size_bytes, "sha256": sha256,
                       "adapter": adapter, "adapter_mtime": adapter_mtime,
                       "sanity": gen[:200]})


MODEL_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags:
  - function-calling
  - tool-use
  - ethereum
  - wallet
  - gguf
---

# Qwen3-8B wallet fine-tune (Q4_K_M GGUF)

A LoRA fine-tune of **Qwen/Qwen3-8B** that turns a natural-language wallet request
into a structured tool call (`executeTx` / `swap` / `readTx` / `shield` /
`unshield`) for the on-device model in a macOS Ethereum wallet.

## Results — 307-case dev set

Scored by a deterministic binary scorer: a case passes only if the tool name and
every argument match a gold call computed from a structured intent.

| model | overall | cases needing a call (272) |
| --- | --- | --- |
| gpt-5 (hosted anchor) | 95.8% | 95.6% |
| **this model** | **86.0%** | **85.3%** |
| gemma-4-E4B wallet ft | 83.7% | 82.4% |
| Qwen3-8B (untuned base) | 41.0% | 36.4% |
| gemma-4-E4B (untuned) | 12.7% | 9.2% |

**+45.0 pp over its own base.** 35 of the 307 cases expect NO tool call
(missing-field and safety-refusal cases), so a model that never acts still scores
~11% — read the second column.

Trained on 1739 synthetic examples that are disjoint from the eval set by
construction (asserted in CI), with a `<think>` arithmetic trace before each call.

## Known weaknesses

- **Base-unit arithmetic on large amounts.** Its remaining failures are almost
  entirely one decimal place out (e.g. `12345.6 ETH` -> one zero too few).
- **Safety refusals: 5/7.** Training holds ~1 example per safety category, so do
  not rely on it to refuse burn-address sends or unverified-token swaps.

Not audited, and not a substitute for a transaction-confirmation UI: every call
it emits should be reviewed by the user before signing.

## Use

```python
from llama_cpp import Llama
llm = Llama.from_pretrained(repo_id="{repo}", filename="{gguf}", n_ctx=8192)
```

Sampling follows the Qwen3 card: `temperature=0.6, top_p=0.95, top_k=20`,
thinking enabled. `adapter/` holds the LoRA adapter for re-merging.
"""


@app.function(image=image, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs})
def upload(hf_token: str, repo: str, private: bool) -> str:
    """Push the GGUF + adapter from the Volume straight to the Hub.

    Uploading from here rather than from a laptop keeps 5 GB off a home
    connection — the file is already sitting on this Volume. No GPU needed, so
    this is a separate function from `export`: re-uploading must never re-run the
    merge.
    """
    import json

    from huggingface_hub import HfApi

    gguf = f"{OUTPUTS_DIR}/{GGUF_NAME}"
    if not Path(gguf).is_file():
        raise SystemExit(f"{gguf} missing — run the export first")

    api = HfApi(token=hf_token)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=gguf, path_in_repo=GGUF_NAME, repo_id=repo)
    adapter = _resolve_adapter()
    api.upload_folder(folder_path=adapter, path_in_repo="adapter", repo_id=repo)
    card = MODEL_CARD.format(repo=repo, gguf=GGUF_NAME)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=repo)
    url = f"https://huggingface.co/{repo}"
    print(f"[upload] {url} (private={private})", flush=True)
    return json.dumps({"repo": url, "private": private,
                       "mb": round(Path(gguf).stat().st_size / 1e6, 1)})


@app.local_entrypoint()
def main(min_mtime: float, do_export: bool = True, do_upload: bool = False,
         repo: str = "ef-dai-team/qwen3-8b-wallet-ft", private: bool = False) -> None:
    # min_mtime is REQUIRED: the unix timestamp of when you launched the
    # training run whose adapter you want exported. Without it, a Volume that
    # still holds a previous run's adapter/checkpoints (it is never cleared
    # between runs, and can be from a DIFFERENT tool contract) could get
    # silently merged instead — see the stale-adapter guard note in the module
    # docstring and _resolve_adapter.
    if do_export:
        print(export.remote(min_mtime=min_mtime))
    if do_upload:
        # The token is read LOCALLY (the container has none) and passed in, the
        # same way finetune/modal_export_gemma4.py does it.
        token = (Path.home() / ".cache" / "huggingface" / "token").read_text().strip()
        print(upload.remote(token, repo, private))
