#!/usr/bin/env python
"""Regenerate `pf/tools.app.json` from the app's own contract dump.

The benchmark offers two tool sets, and which one a case gets is part of the
case:

  * `pf/tools.json` (executeTx, readTx, transfer, swap) — the transaction-builder
    contract. The Aave and Safe datasets are written against it and stay on it:
    the wallet has no lending or multisig tool today, but the capability is still
    worth measuring, so those cases keep their `executeTx` gold.
  * `pf/tools.app.json` (transfer, swap) — exactly what `ToolDefinitions.phase1`
    serialises, generated here so it cannot drift. Every wallet-path case uses
    it, because these are the only tools the app ever offers.

Hand-maintaining the app half is what let it drift the first time, so it is now
derived from `pf/app_contract_reference.json`, written by the app's own renderer
via `wallet-eval prompt-dump`.

Feeding a model different tool sets per case is deliberate, not a compromise: a
tool-calling model reads the tools out of its prompt, so training on both teaches
it to use what it is offered rather than memorising one fixed menu.

    uv run python scripts/sync_app_contract.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "pf" / "app_contract_reference.json"
OUT = ROOT / "pf" / "tools.app.json"


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    tools = json.loads(reference["toolsJSON"])
    names = [t["function"]["name"] for t in tools]

    OUT.write_text(json.dumps(tools, indent=2) + "\n")

    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  tools: {', '.join(names)}")
    print(f"  payload: {len(OUT.read_text())} chars")
    print(f"  system prompt: {len(reference['systemPrompt'])} chars")


if __name__ == "__main__":
    main()
