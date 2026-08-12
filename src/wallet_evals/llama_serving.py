"""llama-cpp serving helpers.

Kept out of `functiongemma.py` because everything there is pure translation of
prompts and outputs; this concerns how the model is SERVED. Sampling has to be
resolved in one place so the provider and the diagnostic scripts drive a model
identically — otherwise a trace collected for debugging is not the trace the
benchmark produced.
"""
from __future__ import annotations

from typing import Any


def sampling_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """The sampling arguments named in `config`, and only those.

    Passing a parameter the config did not set would silently replace
    llama-cpp's default, so a model whose card prescribes nothing (the three
    Gemma-family providers) must keep receiving nothing.
    """
    out: dict[str, Any] = {}
    for key in ("top_p", "min_p"):
        if config.get(key) is not None:
            out[key] = float(config[key])
    if config.get("top_k") is not None:
        out["top_k"] = int(config["top_k"])
    return out
