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

from wallet_evals.schema import Dataset, PreviewContext
from wallet_evals.tools import format_preview_header


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/tiny.json")
    dataset = Dataset.model_validate(json.loads(path.read_text()))

    print(format_preview_header())
    print(dataset.format_preview(PreviewContext(source=path)))


if __name__ == "__main__":
    main()
