"""CLI entrypoint: run the eval through an adapter and print/save the report."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from wallet_evals.adapters.openrouter import OpenRouterAdapter
from wallet_evals.report import aggregate, format_summary
from wallet_evals.runner import run_eval
from wallet_evals.schema import Dataset


def _load_cases(path: Path):
    return Dataset.model_validate(json.loads(path.read_text())).cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the wallet eval harness.")
    parser.add_argument("--dataset", default="datasets/cases.json", type=Path)
    parser.add_argument("--backend", choices=["openrouter", "llama-server"], default="openrouter")
    parser.add_argument("--model", required=True, help="Model id (e.g. openai/gpt-4o-mini or local gguf name)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    cases = _load_cases(args.dataset)

    if args.backend == "openrouter":
        adapter = OpenRouterAdapter(model=args.model)
    else:
        from wallet_evals.adapters.llama_server import LlamaServerAdapter

        adapter = LlamaServerAdapter(model=args.model)

    results = run_eval(cases, adapter, repeats=args.repeats)
    report = aggregate(results)
    print(format_summary(report))

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"report": report, "results": [asdict(r) for r in results]}, indent=2) + "\n")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
