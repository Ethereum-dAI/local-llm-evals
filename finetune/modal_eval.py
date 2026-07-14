"""Evaluate the fine-tuned LoRA ADAPTER on the eval set, scored by the repo's own
deterministic scorer — bypassing the broken LoRA->16bit GGUF merge.

Diagnosis established: the merged/GGUF weights are corrupted (gemma3 + float16),
but the unmerged adapter (functiongemma-ft-outputs:/outputs/checkpoint-309)
generates correct DSL. So we load base+adapter on a T4, render every case in
pf/tests.generated.yaml with the real prompt, greedy-generate, translate the raw
output the same way the provider does, and score with wallet_evals.scorer.

Returns per-category pass rates (captured locally via the entrypoint).

Run:  uv run --with modal modal run finetune/modal_eval.py
"""
from __future__ import annotations

from pathlib import Path

import modal

ADAPTER = "/outputs/checkpoint-309"
MAX_SEQ_LEN = 2048
_REPO = Path(__file__).resolve().parent.parent

hf_cache = modal.Volume.from_name("functiongemma-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("functiongemma-ft-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("unsloth", "huggingface_hub", "pyyaml", "pydantic")
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_dir(str(_REPO / "src"), "/repo/src")
    .add_local_dir(str(_REPO / "pf"), "/repo/pf")
    .add_local_dir(str(_REPO / "datasets"), "/repo/datasets")
)

app = modal.App("functiongemma-eval")


@app.function(image=image, gpu="T4", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache, "/outputs": outputs})
def evaluate() -> dict:
    import json
    import sys
    from collections import defaultdict

    sys.path.insert(0, "/repo/src")
    sys.path.insert(0, "/repo")

    import yaml
    from unsloth import FastLanguageModel

    from wallet_evals.functiongemma import raw_output_to_scoreable
    from wallet_evals.parsing import parse_turn
    from wallet_evals.promptfoo import case_from_metadata
    from wallet_evals.scorer import score_case
    import importlib
    render = importlib.import_module("pf.prompt").render

    tools = json.loads(Path("/repo/pf/tools.json").read_text())
    tests = yaml.safe_load(Path("/repo/pf/tests.generated.yaml").read_text())

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, full_finetuning=False,
    )
    FastLanguageModel.for_inference(model)

    def remap(msgs):  # system -> developer, as the provider does at inference
        out = []
        for m in msgs:
            m = dict(m)
            if m.get("role") == "system":
                m["role"] = "developer"
            out.append(m)
        return out

    agg = defaultdict(lambda: [0, 0])
    total = [0, 0]
    for i, t in enumerate(tests):
        md = t["metadata"]
        messages = remap(render({"vars": t["vars"]}))
        enc = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        gen = model.generate(**enc, max_new_tokens=256, do_sample=False)
        text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=False)
        scoreable = raw_output_to_scoreable(text)
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
        if i % 40 == 0:
            print(f"[eval] {i}/{len(tests)} ...", flush=True)

    summary = {"overall_pass": total[0], "overall_total": total[1],
               "overall_pct": round(100 * total[0] / total[1], 1),
               "by_category": {c: [v[0], v[1]] for c, v in sorted(agg.items())}}
    print("[eval] SUMMARY:", json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main() -> None:
    import json
    print("RESULT:", json.dumps(evaluate.remote(), indent=2))
