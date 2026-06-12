"""Shared protocol-eval layer.

Each protocol module exposes:
  NAME: str
  FIXTURES: Path                       # its frozen fixtures JSON
  build_cases(fixtures, rng, start_idx=1) -> list[dict]   # promptfoo test dicts

Register modules in PROTOCOL_MODULES so scripts/generate_protocol_cases.py can
iterate them. Aave will add itself here later.
"""
from __future__ import annotations

from wallet_evals.protocols import safe

PROTOCOL_MODULES = [safe]
