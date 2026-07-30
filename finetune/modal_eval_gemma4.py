"""Evaluate the fine-tuned Gemma-4 LoRA ADAPTER on the eval set, scored by the
repo's own deterministic scorer — bypassing the GGUF merge/quantize path.

This is the fast checkpoint (and the fallback if the Q4_K_M GGUF ever comes out
corrupt): load base+adapter on a GPU, render every case in pf/tests.generated.yaml
with the real prompt (keeping the `system` role — no developer remap for Gemma-4),
greedy-generate, translate the raw output the same way pf/provider_functiongemma.py
does (GEMMA4 dialect, <think> stripped), and score with wallet_evals.scorer.

Returns per-category pass rates (captured locally via the entrypoint).

Run:  uv run --with modal modal run finetune/modal_eval_gemma4.py
"""
from __future__ import annotations

from pathlib import Path

import modal

OUTPUTS_DIR = "/outputs"
MAX_SEQ_LEN = 2048


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
_REPO = Path(__file__).resolve().parent.parent

hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("gemma4-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("unsloth", "huggingface_hub", "pyyaml", "pydantic")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_dir(str(_REPO / "src"), "/repo/src")
    .add_local_dir(str(_REPO / "pf"), "/repo/pf")
    .add_local_dir(str(_REPO / "datasets"), "/repo/datasets")
)

app = modal.App("gemma4-eval")


@app.function(image=image, gpu="A100", timeout=7200,
              volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs})
def evaluate() -> dict:
    import json
    import sys
    from collections import defaultdict

    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo")

    import yaml
    from unsloth import FastModel

    from wallet_evals.functiongemma import raw_output_to_scoreable
    from wallet_evals.gemma_dsl import GEMMA4
    from wallet_evals.parsing import parse_turn
    from wallet_evals.promptfoo import case_from_metadata
    from wallet_evals.scorer import score_case
    import importlib
    render = importlib.import_module("pf.prompt").render

    tools = json.loads(Path("/repo/pf/tools.json").read_text())
    tests = yaml.safe_load(Path("/repo/pf/tests.generated.yaml").read_text())

    adapter = _resolve_adapter()
    print(f"[eval] adapter = {adapter}", flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, full_finetuning=False,
    )
    FastModel.for_inference(model)

    # Gemma-4 E4B is multimodal: FastModel returns a PROCESSOR, whose __call__
    # expects text=/images= (a positional batch misroutes to images -> text=None).
    # Use the underlying text tokenizer for batch encode/decode. Keep the `system`
    # role (native <|turn>system) — no remap.
    tk = getattr(tokenizer, "tokenizer", tokenizer)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "left"
    texts = [tokenizer.apply_chat_template(render({"vars": t["vars"]}),
                                           tools=tools, add_generation_prompt=True,
                                           tokenize=False) for t in tests]

    BATCH = 16
    outs: list[str] = []
    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start + BATCH]
        enc = tk(chunk, return_tensors="pt", padding=True, truncation=True,
                 max_length=MAX_SEQ_LEN, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=256, do_sample=False,
                             pad_token_id=tk.pad_token_id)
        inlen = enc["input_ids"].shape[1]
        for j in range(len(chunk)):
            outs.append(tk.decode(gen[j][inlen:], skip_special_tokens=False))
        print(f"[eval] {start + len(chunk)}/{len(texts)}", flush=True)

    agg = defaultdict(lambda: [0, 0])
    total = [0, 0]
    for t, text in zip(tests, outs):
        md = t["metadata"]
        scoreable = raw_output_to_scoreable(text, GEMMA4)
        if scoreable.strip().startswith("["):
            turn = parse_turn(content=None, native_tool_calls=json.loads(scoreable),
                              raw_text="")
        else:
            turn = parse_turn(content=scoreable, native_tool_calls=None,
                              raw_text=scoreable)
        s = score_case(case_from_metadata(md), turn)
        cat = md.get("category", "?")
        agg[cat][0] += s
        agg[cat][1] += 1
        total[0] += s
        total[1] += 1

    summary = {"overall_pass": total[0], "overall_total": total[1],
               "overall_pct": round(100 * total[0] / total[1], 1),
               "by_category": {c: [v[0], v[1]] for c, v in sorted(agg.items())}}
    print("[eval] SUMMARY:", json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main() -> None:
    # spawn + --detach so a dropped client connection can't cancel the job. The
    # summary is printed to the logs ("[eval] SUMMARY: {...}"); read it with
    # `modal app logs <app-id>` if the client disconnects before it returns.
    call = evaluate.spawn()
    print(f"SPAWNED evaluate call_id={call.object_id} — poll logs for '[eval] SUMMARY'.")
