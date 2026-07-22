"""Generate a Gemma-4 E4B fine-tuning set — DISJOINT from the eval set.

Same disjoint sources, seed, distribution and anti-leakage guarantee as the
FunctionGemma set — this script REUSES generate_finetune_data's collection and
selection logic verbatim so the two training sets never drift. Only the ENCODING
differs: targets use the GEMMA4 dialect

    <|tool_call>call:NAME{key:<|"|>value<|"|>,...}<tool_call|>

(the exact shape local-wallet-mac's Gemma4FallbackParser reads), and the `system`
role is kept (Gemma-4's chat template has a native `<|turn>system`; FunctionGemma
remapped it to `developer`). Reasoning <think> traces are ON by default — base-unit
arithmetic is the eval's discriminator and Gemma-4 E4B is a thinking-capable model.

Run:
    uv run python scripts/generate_gemma4_finetune_data.py               # with reasoning
    uv run python scripts/generate_gemma4_finetune_data.py --no-reasoning
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # make the top-level `pf` package importable
sys.path.insert(0, str(ROOT / "scripts"))  # reuse the FunctionGemma generator

import generate_finetune_data as fg  # noqa: E402  (shared collection/selection)
from wallet_evals.finetune import case_to_example  # noqa: E402
from wallet_evals.gemma_dsl import GEMMA4  # noqa: E402
from pf.prompt import render  # noqa: E402

OUT = ROOT / "data_for_finetune" / "gemma4_train.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=True,
                    help="emit a <think> arithmetic trace before transfer/swap calls")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rng = random.Random(fg.SEED)
    selected = fg._select(fg._collect(rng), rng)

    examples: list[dict] = []
    for test, intent in selected:
        md = dict(test["metadata"])
        md["id"] = f"ft-{md['id']}"  # keep the id-space disjoint from the eval set
        reasoning = fg._reasoning_text(intent) if (args.reasoning and intent) else None
        messages = render({"vars": test["vars"]})
        examples.append(case_to_example(md, messages, fg.TOOLS, reasoning_text=reasoning,
                                        dialect=GEMMA4, to_developer=False))

    examples.sort(key=lambda e: e["id"])  # byte-stable output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(examples)} examples -> {args.out}"
          f"{'  (with reasoning)' if args.reasoning else ''}")


if __name__ == "__main__":
    main()
