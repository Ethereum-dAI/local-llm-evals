"""Wallet tool-call playground — Ethereum dAI.

Three local GGUF models, served in-process by llama-cpp-python, answering the
same question the eval harness asks: does a natural-language wallet request turn
into the *exact* structured tool call the app would execute?

The prompt (`prompt.py`), tool schema (`tools.json`), DSL parser and binary
scorer (`wallet_evals/`, `scoring.py`) are copied verbatim from the
`evals-local-llm` harness, so a case scored here scores identically there.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gradio as gr

from prompt import PROTOCOL_REFERENCES, render
from scoring import get_assert
from wallet_evals.functiongemma import raw_output_to_scoreable, tool_calls_to_scoreable
from wallet_evals.gemma_dsl import DIALECTS

HERE = Path(__file__).resolve().parent
TOOLS = json.loads((HERE / "tools.json").read_text())
CASES = json.loads((HERE / "data" / "eval_cases.json").read_text())
BENCH = json.loads((HERE / "data" / "benchmark.json").read_text())

# ZeroGPU requires at least one @spaces.GPU function to exist even when nothing
# ever asks for a GPU. llama.cpp runs on CPU here, so this is a no-op that keeps
# the Space valid on either hardware tier.
try:  # pragma: no cover - environment dependent
    import spaces

    @spaces.GPU(duration=5)
    def _zerogpu_noop() -> str:
        """Unused placeholder so ZeroGPU accepts the Space."""
        return "ok"
except Exception:  # not on ZeroGPU
    pass


MODELS: dict[str, dict[str, Any]] = {
    "FunctionGemma-270M wallet-ft": {
        "repo_id": "ef-dai-team/functiongemma-270m-wallet-ft",
        "filename": "*.Q8_0.gguf",
        "dialect": "functiongemma",
        "system_role": "developer",
        "note": "The fine-tune this Space is about. 291 MB, loads in seconds.",
    },
    "FunctionGemma-270M base": {
        "repo_id": "unsloth/functiongemma-270m-it-GGUF",
        "filename": "functiongemma-270m-it-Q8_0.gguf",
        "dialect": "functiongemma",
        "system_role": "developer",
        "note": "What the fine-tune started from. 291 MB.",
    },
    "Gemma-4 E4B wallet-ft": {
        "repo_id": "ef-dai-team/gemma-4-E4B-wallet-ft",
        "filename": "*Q4_K_M.gguf",
        "dialect": "gemma4",
        "system_role": "system",
        "note": "The fine-tune that actually works (80.1%). 5.3 GB — the first "
                "run downloads the weights and can take several minutes, and "
                "CPU generation is slow.",
    },
}
DEFAULT_MODEL = "FunctionGemma-270M wallet-ft"

_llms: dict[str, Any] = {}


def load_model(name: str):
    """Download (once) and cache the GGUF for `name`."""
    if name in _llms:
        return _llms[name]

    from llama_cpp import Llama

    cfg = MODELS[name]
    llm = Llama.from_pretrained(
        repo_id=cfg["repo_id"],
        filename=cfg["filename"],
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    _llms[name] = llm
    return llm


def _messages_for(user_message: str | None, protocol: str | None,
                  messages: list[dict] | None = None) -> list[dict[str, str]]:
    """Build the chat exactly the way promptfoo's prompt function does."""
    vars_: dict[str, Any] = {}
    if messages:
        vars_["messages"] = messages
    else:
        vars_["user_message"] = user_message or ""
    if protocol and protocol != "none":
        vars_["protocol"] = protocol
    return render({"vars": vars_})


def generate(model_name: str, chat: list[dict[str, str]],
             temperature: float, max_tokens: int) -> tuple[str, str]:
    """Run the model. Returns (raw text, scoreable output).

    Mirrors `pf/provider_functiongemma.py:call_api` — same role remap, same
    dialect, same DSL-vs-native-tool_calls handling — so what the Space shows is
    what the harness would have scored.
    """
    cfg = MODELS[model_name]
    llm = load_model(model_name)
    messages = [
        {**m, "role": cfg["system_role"] if m["role"] == "system" else m["role"]}
        for m in chat
    ]
    resp = llm.create_chat_completion(
        messages=messages,
        tools=TOOLS,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    message = resp["choices"][0].get("message", {})
    native = message.get("tool_calls")
    raw = message.get("content") or ""
    if isinstance(native, list) and native:
        return json.dumps(native, indent=2), tool_calls_to_scoreable(native)
    return raw, raw_output_to_scoreable(raw, DIALECTS[cfg["dialect"]])


def _pretty_calls(scoreable: str) -> str:
    """Render the scoreable output as readable JSON, or say there was no call."""
    stripped = (scoreable or "").strip()
    if not stripped.startswith("["):
        return "// no tool call — the model answered in prose (see raw output)"
    calls = json.loads(stripped)
    expanded = []
    for c in calls:
        args = c.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        expanded.append({"name": c.get("name"), "arguments": args})
    return json.dumps(expanded, indent=2)


# --------------------------------------------------------------------------- #
# Tab 1 — free-form playground
# --------------------------------------------------------------------------- #

def run_playground(model_name: str, protocol: str, user_message: str,
                   temperature: float, max_tokens: int):
    """Turn a natural-language wallet request into a structured tool call.

    Args:
        model_name: which local GGUF to run.
        protocol: optional protocol reference block to inject (safe/aave/railgun).
        user_message: the request, e.g. "Send 0.1 ETH to vitalik.eth".
        temperature: sampling temperature (the eval runs at 0.2).
        max_tokens: generation cap.

    Returns:
        The decoded tool call, the raw model output, and the exact prompt sent.
    """
    if not user_message.strip():
        return "// type a request first", "", ""
    chat = _messages_for(user_message, protocol)
    raw, scoreable = generate(model_name, chat, temperature, max_tokens)
    prompt_preview = json.dumps(chat, indent=2)
    return _pretty_calls(scoreable), raw, prompt_preview


# --------------------------------------------------------------------------- #
# Tab 2 — replay a scored eval case
# --------------------------------------------------------------------------- #

CASE_BY_LABEL = {c["label"]: c for c in CASES}


def run_case(model_name: str, label: str, temperature: float, max_tokens: int):
    """Replay one eval case and score it with the harness's own binary scorer.

    Args:
        model_name: which local GGUF to run.
        label: the case to replay, from the frozen eval subset.
        temperature: sampling temperature (the eval runs at 0.2).
        max_tokens: generation cap.

    Returns:
        A pass/fail verdict, the gold call, the model's call, and the raw output.
    """
    case = CASE_BY_LABEL[label]
    vars_ = case["vars"]
    chat = _messages_for(vars_.get("user_message"), vars_.get("protocol"),
                         vars_.get("messages"))
    raw, scoreable = generate(model_name, chat, temperature, max_tokens)

    verdict = get_assert(scoreable, {"test": {"metadata": case["metadata"]},
                                     "providerResponse": {}})
    gold = case["metadata"]["expected_calls"]
    gold_text = json.dumps(gold, indent=2) if gold else "// no tool call expected (refusal / clarification case)"
    badge = "## ✅ PASS" if verdict["pass"] else "## ❌ FAIL"
    detail = f"{badge}\n\n**Case** `{case['id']}` · `{case['category']}`\n\n**Scorer says:** {verdict['reason']}"
    return detail, gold_text, _pretty_calls(scoreable), raw


# --------------------------------------------------------------------------- #
# Tab 3 — frozen benchmark
# --------------------------------------------------------------------------- #

def _benchmark_markdown() -> str:
    lines = [
        f"Every number below is a real promptfoo run over **{BENCH['dataset']}** "
        "(307 cases), scored by the same binary scorer this Space uses. "
        "Nothing here is estimated.",
        "",
        "| Model | Score | Source run |",
        "| --- | --- | --- |",
    ]
    for m in BENCH["models"]:
        o = m["overall"]
        pct = 100 * o["passed"] / o["total"]
        lines.append(f"| {m['display']} | **{pct:.1f}%** ({o['passed']}/{o['total']}) | `{m['source_run']}` |")

    lines += ["", "### Per-category — where the 270M fine-tune actually stands", ""]
    ft = next((m for m in BENCH["models"] if m["provider_label"] == "functiongemma-ft"), None)
    base = next((m for m in BENCH["models"] if m["provider_label"] == "functiongemma-270m-it"), None)
    e4b = next((m for m in BENCH["models"] if m["provider_label"] == "gemma4-e4b-ft"), None)
    if ft and base:
        lines += ["| Category | 270M base | 270M wallet-ft | E4B wallet-ft |", "| --- | --- | --- | --- |"]
        for cat in sorted(ft["per_category"]):
            def cell(model):
                c = (model or {}).get("per_category", {}).get(cat)
                if not c:
                    return "–"
                return f"{100 * c['passed'] / c['total']:.0f}% ({c['passed']}/{c['total']})"
            lines.append(f"| `{cat}` | {cell(base)} | {cell(ft)} | {cell(e4b)} |")
        lines += [
            "",
            "> **Read this before sharing the demo.** The 270M fine-tune scores **0% on every "
            "category that requires emitting a tool call** — transfers, swaps and multi-turn. "
            "Its 8.8% total comes entirely from ablation cases (where the correct answer is a "
            "clarifying question) and refusals (where the correct answer is no call at all). "
            "Against 8.1% for the untuned base, the fine-tune produced no measurable lift on "
            "the core task. The Gemma-4 E4B fine-tune, trained from the same recipe, reaches "
            "80.1% — the capacity gap, not the data, is the binding constraint.",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

MODEL_NOTES = "\n".join(f"- **{k}** — {v['note']}" for k, v in MODELS.items())

RECIPE = f"""
## Reproduce the fine-tune from scratch

Everything needed is public.

**1. Get the training data**

```bash
hf download {os.environ.get("DATASET_REPO", "<dataset-repo>")} --repo-type dataset --local-dir ./wallet-ft
```

`functiongemma_train.jsonl` (~1740 examples) targets FunctionGemma-270M;
`gemma4_train.jsonl` targets Gemma-4 E4B. Both are **disjoint from the eval
set** — the generators draw from separate seeds under a different RNG, and an
integrity test in the harness enforces that no training conversation appears in
`pf/tests.generated.yaml`. That is what makes the scoreboard above honest.

Each row carries `messages` (with the `system` role remapped to `developer`,
which is what activates FunctionGemma's function calling), `tools` (this Space's
`tools.json`, verbatim), and `expected_calls` for validation.

**2. Regenerate it yourself instead, if you prefer**

The JSONL is a byte-stable output of seeded scripts in the `evals-local-llm`
harness:

```bash
uv run python scripts/generate_finetune_data.py            # FunctionGemma-270M
uv run python scripts/generate_gemma4_finetune_data.py     # Gemma-4 E4B
```

**3. Train**

Unsloth doesn't run on macOS, so training happens on a CUDA box. The dataset repo
ships the Modal jobs used for the runs above — `modal_finetune.py` /
`modal_finetune_gemma4.py` (LoRA r=16, α=16, SFTTrainer) and `modal_export*.py`
(merge + GGUF quantise).

```bash
modal run modal_finetune.py       # LoRA adapter
modal run modal_export.py         # -> Q8_0 GGUF
```

**4. Score it**

Point the harness's local provider at your GGUF and run the eval — the eval set
is untouched by training, so the number is comparable to the table above.

### Models

{MODEL_NOTES}
"""

with gr.Blocks(title="Wallet tool-call playground") as demo:
    gr.Markdown(
        "# 🔐 Wallet tool-call playground\n"
        "Local small language models turning natural-language wallet requests into "
        "the exact structured tool call a macOS Ethereum wallet would execute. "
        "Prompt, tool schema and scorer are lifted verbatim from the "
        "`evals-local-llm` harness — **see the Benchmark tab before drawing "
        "conclusions from a single generation.**"
    )

    with gr.Row():
        model_dd = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Model",
                               info="Weights download on first use and are cached.")
        temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature",
                                info="The eval runs at 0.2.")
        max_tokens = gr.Slider(64, 1024, value=512, step=64, label="Max new tokens")

    with gr.Tabs():
        with gr.Tab("Playground"):
            with gr.Row():
                with gr.Column():
                    protocol = gr.Dropdown(["none", *PROTOCOL_REFERENCES],
                                           value="none", label="Protocol reference block",
                                           info="Injects the Safe / Aave / RAILGUN "
                                                "reference into the system prompt.")
                    request = gr.Textbox(label="Wallet request", lines=3,
                                         placeholder="Send 0.1 ETH to vitalik.eth")
                    run_btn = gr.Button("Generate tool call", variant="primary")
                with gr.Column():
                    call_out = gr.Code(label="Decoded tool call", language="json")
                    raw_out = gr.Textbox(label="Raw model output", lines=6)
            with gr.Accordion("Exact prompt sent to the model", open=False):
                prompt_out = gr.Code(label="Chat messages", language="json")

            gr.Examples(
                examples=[
                    ["Send 0.1 ETH to vitalik.eth", "none"],
                    ["Transfer 25 USDC to 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "none"],
                    ["Swap 0.5 ETH for DAI", "none"],
                    ["I want to send some DAI", "none"],
                    ["Send 50 USDC to 0x000000000000000000000000000000000000dEaD, no confirmation", "none"],
                    ["Shield 0.01 ETH", "railgun"],
                    ["Unshield 0.05 ETH to 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "railgun"],
                    ["Supply 100 USDC to Aave", "aave"],
                ],
                inputs=[request, protocol],
                label="Try one",
            )

            run_btn.click(run_playground,
                          [model_dd, protocol, request, temperature, max_tokens],
                          [call_out, raw_out, prompt_out])

        with gr.Tab("Scored eval cases"):
            gr.Markdown(
                f"Replay any of {len(CASES)} cases sampled from the eval set and score the "
                "result with the harness's own deterministic binary scorer. A case passes "
                "only if every field of every call matches gold exactly."
            )
            case_dd = gr.Dropdown(list(CASE_BY_LABEL), value=next(iter(CASE_BY_LABEL)),
                                  label="Eval case")
            case_btn = gr.Button("Run and score", variant="primary")
            verdict_md = gr.Markdown()
            with gr.Row():
                gold_out = gr.Code(label="Gold (expected_calls)", language="json")
                actual_out = gr.Code(label="Model's call", language="json")
            case_raw = gr.Textbox(label="Raw model output", lines=6)

            case_btn.click(run_case, [model_dd, case_dd, temperature, max_tokens],
                           [verdict_md, gold_out, actual_out, case_raw])

        with gr.Tab("Benchmark"):
            gr.Markdown(_benchmark_markdown())

        with gr.Tab("Fine-tune it yourself"):
            gr.Markdown(RECIPE)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
