"""promptfoo custom Python provider for FunctionGemma-270m (local GGUF).

Referenced from promptfooconfig.yaml as:
    providers:
      - id: file://pf/provider_functiongemma.py:call_api
        config:
          repo_id: unsloth/functiongemma-270m-it-GGUF
          filename: "*Q8_0.gguf"      # or set model_path to a local .gguf
          n_ctx: 4096
          temperature: 0.2
          max_tokens: 1024

The model is a local GGUF served in-process by llama-cpp-python. promptfoo runs
this file under $PROMPTFOO_PYTHON (the uv venv, set by scripts/eval.sh), so
`wallet_evals` and `llama_cpp` import from the same interpreter as the scorer.
The persistent worker loads the model ONCE and reuses it across all cases.

FunctionGemma emits tool calls as plain-text DSL, not OpenAI `tool_calls`, so we
parse them here (wallet_evals.functiongemma) into a shape pf/assert.py already
scores. The scorer, tools.json, and dataset stay unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wallet_evals.functiongemma import decode_prompt, raw_output_to_scoreable

_TOOLS_PATH = Path(__file__).with_name("tools.json")
_llm = None  # loaded once per persistent worker


def _load_model(config: dict[str, Any]):
    """Lazily construct (and cache) the Llama model from config."""
    global _llm
    if _llm is not None:
        return _llm

    from llama_cpp import Llama  # heavy, optional dep — import only when serving

    n_ctx = int(config.get("n_ctx", 4096))
    model_path = config.get("model_path")
    if model_path:
        _llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
    else:
        _llm = Llama.from_pretrained(
            repo_id=config["repo_id"],
            filename=config.get("filename", "*.gguf"),
            n_ctx=n_ctx,
            verbose=False,
        )
    return _llm


def _load_tools(config: dict[str, Any]) -> list[dict]:
    path = config.get("tools_path", _TOOLS_PATH)
    return json.loads(Path(path).read_text())


def call_api(prompt: str, options: dict, context: dict) -> dict:
    config = (options or {}).get("config", {}) or {}
    try:
        llm = _load_model(config)
        messages = decode_prompt(prompt)
        resp = llm.create_chat_completion(
            messages=messages,
            tools=_load_tools(config),
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 1024)),
        )
    except Exception as e:  # surface as a case error, not a crashed run
        return {"output": "", "error": f"{type(e).__name__}: {e}"}

    message = resp["choices"][0].get("message", {})
    text = message.get("content") or ""
    result: dict[str, Any] = {"output": raw_output_to_scoreable(text)}
    if isinstance(resp.get("usage"), dict):
        u = resp["usage"]
        result["tokenUsage"] = {
            "prompt": u.get("prompt_tokens"),
            "completion": u.get("completion_tokens"),
            "total": u.get("total_tokens"),
        }
    return result
