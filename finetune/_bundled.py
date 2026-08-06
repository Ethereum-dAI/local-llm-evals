"""Locate a bundled data file, whichever layout the script is running from.

These Modal jobs live in two places with different shapes around them:

  harness repo            published dataset repo
  finetune/modal_*.py     scripts/modal_*.py
  data_for_finetune/*.jsonl   data/*.jsonl
  finetune/diag_sample.jsonl  data/diag_sample.jsonl

Both resolve `_REPO` as the script's grandparent, so a single hardcoded relative
path can only ever work in one of them. Every one of these jobs was silently
broken in the dataset repo for exactly that reason — `add_local_file` raises
before Modal is even contacted.
"""
from __future__ import annotations

from pathlib import Path


def bundled(repo: Path, *candidates: str) -> Path:
    """Return the first of `candidates` (relative to `repo`) that exists.

    Raises with every path tried, so a genuine packaging mistake is obvious
    rather than surfacing as a confusing Modal error later.
    """
    for rel in candidates:
        path = repo / rel
        if path.is_file():
            return path
    tried = "\n  ".join(str(repo / rel) for rel in candidates)
    raise FileNotFoundError(
        f"none of these exist:\n  {tried}\n"
        "In the harness the fine-tuning JSONLs are gitignored — regenerate them with "
        "scripts/generate_finetune_data.py / scripts/generate_gemma4_finetune_data.py."
    )
