"""Assert the training-time prompt equals the bytes the wallet app actually sends.

Why this exists: the fine-tune dataset and the promptfoo harness each rendered
their own system prompt and tool payload, and both had drifted from the app's.
Models trained under that scaffold scored 97-99% in the harness and 82-91% in
the app's own funnel — the ranking between fine-tunes even inverted. The drift
was silent because nothing compared the two renderings.

`pf/app_contract_reference.json` is produced by the app's own renderer:

    cd local-wallet-mac/wallet-macos
    swift build --product wallet-eval
    ./.build/debug/wallet-eval prompt-dump --model <gguf> --json <repo>/pf/app_contract_reference.json

That file carries the verbatim system prompt, the OpenAI-shape tools array for
`ToolDefinitions.phase1`, and the fully rendered prompt for two message shapes.
This job re-renders those same shapes through the HF tokenizer the trainer uses
and diffs. If they differ, training data built with `apply_chat_template` would
not match what the app sends, and the whole alignment exercise is void.

CPU only — the tokenizer is all that is needed, so this costs cents.

    uv run modal run finetune/modal_verify_prompt_parity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

BASE_MODEL = "unsloth/gemma-4-E4B-it"
HF_CACHE_DIR = "/root/.cache/huggingface"
REFERENCE_REMOTE = "/data/app_contract_reference.json"

_REPO = Path(__file__).resolve().parent.parent

hf_cache = modal.Volume.from_name("gemma4-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers", "huggingface_hub", "sentencepiece", "protobuf", "jinja2")
    .env({"HF_HOME": HF_CACHE_DIR})
)

if modal.is_local():
    image = image.add_local_file(
        str(_REPO / "pf" / "app_contract_reference.json"), REFERENCE_REMOTE
    )

app = modal.App("gemma4-verify-prompt-parity")


@app.function(image=image, volumes={HF_CACHE_DIR: hf_cache}, timeout=1800)
def verify() -> dict:
    from transformers import AutoTokenizer

    reference = json.loads(Path(REFERENCE_REMOTE).read_text())
    tools = json.loads(reference["toolsJSON"])
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # llama.cpp renders with thinking on (SamplerOptions.enableThinking defaults
    # true) and adds BOS as a *token* at tokenize time, so its text carries no
    # literal "<bos>". HF emits BOS into the text and needs thinking requested
    # explicitly. Neither is a prompt-content difference, but both must be
    # reconciled or the byte comparison is meaningless. Try the known kwarg
    # spellings and report which one reproduces the app.
    def render(messages, **kwargs) -> str:
        text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs
        )
        return text[len("<bos>"):] if text.startswith("<bos>") else text

    candidates = [{}, {"enable_thinking": True}, {"thinking": True},
                  {"add_thinking": True}, {"enable_thinking": True, "thinking": True}]
    probe = reference["cases"][0]
    chosen = None
    for kwargs in candidates:
        try:
            if render(probe["messages"], **kwargs) == probe["rendered"]:
                chosen = kwargs
                break
        except Exception as exc:  # unknown kwarg for this template
            print(f"  kwargs {kwargs} rejected: {type(exc).__name__}", flush=True)
    if chosen is None:
        print("no kwarg combination reproduced the app render; "
              "falling back to bare render for the diff", flush=True)
        chosen = {}
    print(f"[parity] template kwargs = {chosen}", flush=True)

    results = []
    for case in reference["cases"]:
        messages = case["messages"]
        rendered = render(messages, **chosen)
        expected = case["rendered"]
        match = rendered == expected
        results.append({"label": case["label"], "match": match,
                        "hf_len": len(rendered), "app_len": len(expected)})
        print(f"[{case['label']}] match={match} hf={len(rendered)} app={len(expected)}",
              flush=True)
        if not match:
            # Locate the first divergence so the fix is obvious rather than a
            # 3 KB eyeball diff.
            limit = min(len(rendered), len(expected))
            i = next((i for i in range(limit) if rendered[i] != expected[i]), limit)
            print(f"  first difference at char {i}", flush=True)
            print(f"  app: {expected[max(0, i - 120):i + 200]!r}", flush=True)
            print(f"  hf : {rendered[max(0, i - 120):i + 200]!r}", flush=True)

    ok = all(r["match"] for r in results)
    print(f"[parity] chosen kwargs {chosen}", flush=True)
    print(f"\nPARITY {'OK' if ok else 'FAILED'}", flush=True)
    return {"ok": ok, "results": results, "kwargs": chosen}


@app.local_entrypoint()
def main() -> None:
    outcome = verify.remote()
    print(json.dumps(outcome, indent=2))
    if not outcome["ok"]:
        raise SystemExit(
            "prompt parity FAILED — training data built with apply_chat_template "
            "would not match what the app sends"
        )
