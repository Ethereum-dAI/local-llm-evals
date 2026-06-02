"""Preview a dataset: show the system prompt + tools the model receives, and
each case's user message and gold expected_calls.

Runnable now (no API key, no model) so you can eyeball the dataset and the
exact prompt/tooling before any model is wired up.

Usage:
    uv run python scripts/preview.py                      # uses datasets/tiny.json
    uv run python scripts/preview.py datasets/cases.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from wallet_evals.schema import Dataset
from wallet_evals.tools import SYSTEM_PROMPT, TOOLS, tool_names


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/tiny.json")
    dataset = Dataset.model_validate(json.loads(path.read_text()))

    print("=" * 72)
    print("SYSTEM PROMPT (sent with every case)")
    print("=" * 72)
    print(SYSTEM_PROMPT)
    print()
    print(f"TOOLS offered to the model: {tool_names()}")
    print("(full JSON schemas:)")
    print(json.dumps(TOOLS, indent=2))
    print()

    print("=" * 72)
    print(f"DATASET: {path}  ({len(dataset.cases)} cases)")
    print("=" * 72)
    for case in dataset.cases:
        print(f"\n[{case.id}]  level={case.level}  protocol={case.protocol}")
        print(f"  user: {case.user_message}")
        if not case.expected_calls:
            print("  expected: (no tool call)")
            continue
        for i, call in enumerate(case.expected_calls):
            print(f"  expected call #{i + 1}: {call.tool} -> {call.to}")
            print(f"      chainId={call.chainId} value={call.value} function={call.function}")
            print(f"      args={call.args}")


if __name__ == "__main__":
    main()
