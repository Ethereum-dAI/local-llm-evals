"""Generic on-chain tool definitions + production system prompt.

The system prompt is ported verbatim from the Swift app
(RecognitionRunner.swift line 175 + ToolDefinitions.systemNudge) so eval numbers
transfer to production. The tool surface is the generic executeTx/readTx primitives
(spec §2); legacy transfer/swap are expressed AS executeTx sequences in the dataset.
"""
from __future__ import annotations

import json

_SYSTEM_NUDGE = (
    "When the user clearly expresses intent to perform an on-chain action (transfer, swap, "
    "etc.), you MUST call the corresponding tool with structured arguments instead of "
    "describing the action in prose. If essential information is missing, ask one short "
    "clarifying question in natural language and wait for the answer before calling the "
    "tool. Never invent recipient addresses, ENS names, contact names, token symbols, or "
    "amounts that the user has not provided."
)

SYSTEM_PROMPT = "You are the local AI inside a macOS Ethereum wallet app. " + _SYSTEM_NUDGE

_ENVELOPE_PROPERTIES = {
    "chainId": {"type": "string", "description": "EIP-155 chain id as a decimal string, e.g. \"1\" for Ethereum mainnet."},
    "to": {"type": "string", "description": "Target contract address (0x) the call is sent to. For an ERC-20 transfer/approve this is the token contract; for a native transfer this is the recipient."},
    "value": {"type": "string", "description": "Native value in wei as a decimal string. \"0\" for contract calls; the amount in wei for a native transfer."},
    "function": {"type": ["string", "null"], "description": "Function signature being called, e.g. \"transfer(address,uint256)\". Null for a plain native transfer with no calldata."},
    "args": {"type": "array", "items": {}, "description": "Positional arguments matching the function signature, in order. Tuple parameters are nested arrays. Empty for a native transfer."},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "executeTx",
            "description": "Execute a single on-chain transaction (state-changing). Use one call per transaction; multi-step actions like a swap emit several calls in order (e.g. approve then the swap).",
            "parameters": {
                "type": "object",
                "properties": _ENVELOPE_PROPERTIES,
                "required": ["chainId", "to", "args"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "readTx",
            "description": "Read state from the chain (read-only call, no value transfer).",
            "parameters": {
                "type": "object",
                "properties": {k: v for k, v in _ENVELOPE_PROPERTIES.items() if k != "value"},
                "required": ["chainId", "to", "args"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "swap",
            "description": "Express a token swap as a high-level intent (routing-free). The wallet resolves pool/router and encoding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chainId": {"type": "string", "description": "EIP-155 chain id as a decimal string."},
                    "currencyIn": {"type": "string", "description": "Address of the token being spent. Native ETH is the zero address 0x0000000000000000000000000000000000000000."},
                    "currencyOut": {"type": "string", "description": "Address of the token being received. Native ETH is the zero address."},
                    "amountIn": {"type": "string", "description": "Exact input amount in base units (decimal string)."},
                    "amountOutMinimum": {"type": "string", "description": "Minimum acceptable output in base units; \"0\" if unspecified."},
                    "recipient": {"type": "string", "description": "Recipient address; defaults to the user's own wallet."},
                },
                "required": ["chainId", "currencyIn", "currencyOut", "amountIn"],
            },
        },
    },
]


def tool_names() -> list[str]:
    return [t["function"]["name"] for t in TOOLS]


def format_preview_header() -> str:
    sep = "=" * 72
    lines = [
        sep,
        "SYSTEM PROMPT (sent with every case)",
        sep,
        SYSTEM_PROMPT,
        "",
        f"TOOLS offered to the model: {tool_names()}",
        "(full JSON schemas:)",
        json.dumps(TOOLS, indent=2),
        "",
    ]
    return "\n".join(lines)
