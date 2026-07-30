"""promptfoo custom Python provider for local Gemma-family GGUF models.

Serves both FunctionGemma-270m and the shipped Gemma-4 E4B from one code path;
`config.dialect` + `config.system_role` select the model's conventions.

Referenced from promptfooconfig.yaml as:
    providers:
      # FunctionGemma-270m (defaults: dialect=functiongemma, system_role=developer)
      - id: file://pf/provider_functiongemma.py:call_api
        config:
          repo_id: unsloth/functiongemma-270m-it-GGUF
          filename: "*Q8_0.gguf"      # or set model_path to a local .gguf
          n_ctx: 4096
          temperature: 0.2
          max_tokens: 1024
      # Shipped Gemma-4 E4B (the exact SHA-pinned Q4_K_M the wallet installs; the
      # file was deleted from `main`, so pin the revision that still carries it).
      - id: file://pf/provider_functiongemma.py:call_api
        config:
          repo_id: ggml-org/gemma-4-E4B-it-GGUF
          filename: gemma-4-E4B-it-Q4_K_M.gguf
          revision: 1762c8e8713f
          dialect: gemma4
          system_role: system

The model is a local GGUF served in-process by llama-cpp-python. promptfoo runs
this file under $PROMPTFOO_PYTHON (the uv venv, set by scripts/eval.sh), so
`wallet_evals` and `llama_cpp` import from the same interpreter as the scorer.
The persistent worker loads the model ONCE and reuses it across all cases.

Gemma models emit tool calls as plain-text DSL, not OpenAI `tool_calls`, so we
parse them here (wallet_evals.functiongemma) into a shape pf/assert.py already
scores. If a GGUF chat template instead surfaces structured `tool_calls`, we
normalize those to the same shape. The scorer, tools.json, and dataset stay
unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wallet_evals.functiongemma import (
    decode_prompt,
    raw_output_to_scoreable,
    tool_calls_to_scoreable,
)
from wallet_evals.gemma_dsl import DIALECTS

_TOOLS_PATH = Path(__file__).with_name("tools.json")
# Cache keyed by model identity, NOT a single global: a base-vs-fine-tuned config
# has two providers from this same file, and if promptfoo serves them from one
# worker a single global would make the second silently reuse the first's weights.
_llms: dict[tuple, Any] = {}


def _load_model(config: dict[str, Any]):
    """Lazily construct (and cache) the Llama model, keyed by its identity."""
    n_ctx = int(config.get("n_ctx", 4096))
    model_path = config.get("model_path")
    revision = config.get("revision")
    key = (model_path, config.get("repo_id"), config.get("filename"), revision, n_ctx)
    if key in _llms:
        return _llms[key]

    from llama_cpp import Llama  # heavy, optional dep — import only when serving

    if model_path:
        llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
    elif revision:
        # Pinned revision: llama-cpp's from_pretrained globs the repo's `main`
        # branch, but the wallet's Q4_K_M was deleted from main (it lives only at
        # this pinned commit). Resolve the exact file ourselves so the benchmark
        # loads the byte-identical GGUF the wallet ships.
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=config["repo_id"],
                               filename=config["filename"], revision=revision)
        llm = Llama(model_path=path, n_ctx=n_ctx, verbose=False)
    else:
        llm = Llama.from_pretrained(
            repo_id=config["repo_id"],
            filename=config.get("filename", "*.gguf"),
            n_ctx=n_ctx,
            verbose=False,
        )
    _llms[key] = llm
    return llm


def _load_tools(config: dict[str, Any]) -> list[dict]:
    path = config.get("tools_path", _TOOLS_PATH)
    return json.loads(Path(path).read_text())


def call_api(prompt: str, options: dict, context: dict) -> dict:
    config = (options or {}).get("config", {}) or {}
    dialect = DIALECTS[config.get("dialect", "functiongemma")]
    system_role = config.get("system_role", "developer")
    try:
        llm = _load_model(config)
        messages = decode_prompt(prompt, system_role=system_role)
        resp = llm.create_chat_completion(
            messages=messages,
            tools=_load_tools(config),
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 1024)),
        )
    except Exception as e:  # surface as a case error, not a crashed run
        return {"output": "", "error": f"{type(e).__name__}: {e}"}

    message = resp["choices"][0].get("message", {})
    # Prefer structured tool_calls if the chat template produced them; otherwise
    # parse the DSL out of the text content.
    native = message.get("tool_calls")
    if isinstance(native, list) and native:
        output = tool_calls_to_scoreable(native)
    else:
        output = raw_output_to_scoreable(message.get("content") or "", dialect)
    result: dict[str, Any] = {"output": output}
    if isinstance(resp.get("usage"), dict):
        u = resp["usage"]
        result["tokenUsage"] = {
            "prompt": u.get("prompt_tokens"),
            "completion": u.get("completion_tokens"),
            "total": u.get("total_tokens"),
        }
    return result
