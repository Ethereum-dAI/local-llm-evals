"""Offline smoke test: run the real parse->score pipeline on datasets/tiny.json
with canned (fake) model responses. No API key, no model needed.

It feeds each case a hand-written "model output" (some correct, some wrong, one
using the Gemma DSL text format instead of native tool calls) so you can watch
the scorer discriminate. Edit CANNED_RESPONSES to experiment.

Usage:
    uv run python scripts/smoke_score.py
"""
from __future__ import annotations

import json
from pathlib import Path

from wallet_evals.parsing import parse_turn
from wallet_evals.schema import Dataset
from wallet_evals.scorer import score_case

VITALIK = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
ROUTER = "0xE592427A0AEce92De3Edee1F18E0157C05861564"


def _native(name: str, **fields) -> dict:
    """Build a native OpenAI-style tool call (arguments is a JSON string)."""
    return {"name": name, "arguments": json.dumps(fields)}


# Canned "model responses" per case id: (native_tool_calls | None, raw_text).
# A mix of correct and intentionally-wrong answers to show real scoring.
CANNED_RESPONSES: dict[str, tuple[list[dict] | None, str]] = {
    # CORRECT native transfer
    "tiny-native-transfer-001": (
        [_native("executeTx", chainId="1", to=VITALIK, value="100000000000000000", function=None, args=[])],
        "",
    ),
    # CORRECT erc-20 transfer
    "tiny-erc20-transfer-001": (
        [_native("executeTx", chainId="1", to=USDC, value="0",
                 function="transfer(address,uint256)", args=[VITALIK, "100000000"])],
        "",
    ),
    # WRONG: approve amount off by a decimal (5 DAI instead of 50)
    "tiny-erc20-approve-001": (
        [_native("executeTx", chainId="1", to=DAI, value="0",
                 function="approve(address,uint256)", args=[ROUTER, "5000000000000000000"])],
        "",
    ),
    # CORRECT read, expressed via the Gemma DSL text format (no native tool call)
    "tiny-read-balance-001": (
        None,
        '<|tool_call>call:readTx{chainId:<|"|>1<|"|>,to:<|"|>' + USDC
        + '<|"|>,value:<|"|>0<|"|>,function:<|"|>balanceOf(address)<|"|>,args:<|"|>["' + VITALIK + '"]<|"|>}<tool_call|>',
    ),
    # CORRECT no-call: model answers in prose, makes no tool call
    "tiny-ambiguous-001": (
        None,
        "An Ethereum transaction is a signed message that changes chain state.",
    ),
}


def main() -> None:
    dataset = Dataset.model_validate(json.loads(Path("datasets/tiny.json").read_text()))

    total = 0
    passed = 0
    print(f"Scoring {len(dataset.cases)} canned responses against datasets/tiny.json\n")
    for case in dataset.cases:
        native, raw_text = CANNED_RESPONSES.get(case.id, (None, ""))
        turn = parse_turn(content=raw_text or None, native_tool_calls=native, raw_text=raw_text)
        score = score_case(case, turn)
        total += 1
        passed += score
        mark = "PASS" if score else "FAIL"
        n_calls = len(turn.tool_calls)
        print(f"  [{mark}] {case.id}  (model made {n_calls} call(s), expected {len(case.expected_calls)})")

    pct = (passed / total * 100) if total else 0.0
    print(f"\nOverall: {passed}/{total} = {pct:.0f}%")


if __name__ == "__main__":
    main()
