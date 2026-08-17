"""Generate a Qwen3 fine-tuning set — DISJOINT from the eval set.

Third encoding of the SAME training rows. Like the Gemma-4 generator, this reuses
`generate_finetune_data`'s collection and selection verbatim — same seed, same
sources, same distribution, same anti-leakage guarantee — so the three training
sets can never drift apart in content. Only the ENCODING differs:

    Qwen3 emits Hermes-style JSON tool calls:
        <tool_call>
        {"name": "executeTx", "arguments": {...}}
        </tool_call>

The `system` role is kept (Qwen3's chat template has a native system turn), and
<think> reasoning traces are ON — Qwen3 is a thinking model, its card forbids
suppressing that for tool use, and base-unit arithmetic is precisely what the
eval discriminates on. The traces are the same ground-truth arithmetic strings
the Gemma-4 set uses.

Run:
    uv run python scripts/generate_qwen_finetune_data.py               # with reasoning
    uv run python scripts/generate_qwen_finetune_data.py --no-reasoning
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
from pf.prompt import render, tools_for  # noqa: E402
from wallet_evals.finetune import HERMES, case_to_example  # noqa: E402

OUT = ROOT / "data_for_finetune" / "qwen_train.jsonl"


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
        examples.append(case_to_example(md, messages, tools_for(test["vars"]),
                                        reasoning_text=reasoning, dialect=HERMES,
                                        to_developer=False))

    examples.sort(key=lambda e: e["id"])  # byte-stable output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, sort_keys=True) + "\n")
    print(f"wrote {len(examples)} examples -> {args.out}")


if __name__ == "__main__":
    main()
