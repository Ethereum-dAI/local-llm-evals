"""promptfoo prompt function — builds the chat for both single- and multi-turn.

Referenced from promptfooconfig.yaml as:
    prompts:
      - file://pf/prompt.py:render

`render(context)` returns a chat message list. Legacy cases provide
`vars.user_message`; generated multi-turn cases provide `vars.messages`
(a full [user, assistant, user, ...] script). The system prompt is constant.
"""
from __future__ import annotations

from typing import Any

SYSTEM = (
    "You are the local AI inside a macOS Ethereum wallet app. When the user "
    "clearly expresses intent to perform an on-chain action (transfer, swap, "
    "etc.), you MUST call the corresponding tool with structured arguments "
    "instead of describing the action in prose. If essential information is "
    "missing (e.g. no amount, no recipient, no token), ask one short clarifying "
    "question in natural language and wait for the answer before calling the "
    "tool. Never invent a recipient, token, or amount the user has not provided.\n"
    "\n"
    "REFERENCE DATA (use it; do not ask the user for it):\n"
    "Tokens (symbol -> contract address, decimals):\n"
    "  ETH  -> native, 18 decimals. For swaps, native ETH is the zero address "
    "0x0000000000000000000000000000000000000000.\n"
    "  WETH -> 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2, 18 decimals\n"
    "  USDC -> 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48, 6 decimals\n"
    "  DAI  -> 0x6B175474E89094C44Da98b954EedeAC495271d0F, 18 decimals\n"
    "Names (ENS -> address):\n"
    "  vitalik.eth -> 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045\n"
    "\n"
    "CONVENTIONS (assume these defaults; do NOT ask about them):\n"
    "- chainId is always \"1\" (Ethereum mainnet).\n"
    "- Resolve any ENS name or token symbol to its address using the reference "
    "above. A raw 0x address is used as-is.\n"
    "- Convert every human amount to base units using the token's decimals "
    "(e.g. 0.1 ETH -> \"100000000000000000\"; 3 USDC -> \"3000000\"). All "
    "numeric fields are decimal strings.\n"
    "- Native ETH transfer: executeTx with to=recipient address, value=amount in "
    "wei, function=null, args=[].\n"
    "- ERC-20 transfer: executeTx with to=token contract address, value=\"0\", "
    "function=\"transfer(address,uint256)\", args=[recipient address, amount in "
    "base units].\n"
    "- Swap: call the swap tool IMMEDIATELY with currencyIn/currencyOut addresses "
    "and amountIn in base units. amountOutMinimum is ALWAYS \"0\" and recipient is "
    "ALWAYS \"<wallet>\" (the user's own wallet), unless the user explicitly names "
    "a different minimum or recipient. A swap request that has an input amount and "
    "both tokens is COMPLETE — never ask about slippage, minimum output, network, "
    "or recipient; those have fixed defaults. Emit exactly ONE swap call — never "
    "add an approval or any other transaction alongside it.\n"
    "\n"
    "Once you have the action plus its amount, token(s), and (for a transfer) a "
    "recipient, you have everything you need: emit the tool call. Do not ask for "
    "confirmation or for any value the conventions above already supply."
)


SAFE_REFERENCE = (
    "Safe owner management. To add or remove a Safe signer, emit a single "
    "executeTx whose `to` is your Safe's own address (a self-call), value \"0\":\n"
    "- Add: function \"addOwnerWithThreshold(address,uint256)\", args [newOwner, newThreshold].\n"
    "- Remove: function \"removeOwner(address,address,uint256)\", args [prevOwner, "
    "owner, threshold]. `prevOwner` is the owner listed IMMEDIATELY BEFORE the one "
    "being removed in the owners list below; if it is the first owner, use the "
    "sentinel 0x0000000000000000000000000000000000000001. Keep the current "
    "threshold unless the user asks to change it."
)


AAVE_REFERENCE = (
    "Aave v3 lending (Ethereum mainnet). For supply/withdraw/borrow/repay, emit a "
    "single executeTx to the Aave v3 Pool "
    "0x87870Bca3F3fD6335C3F4ce8392D69350B4fa4E2 (value \"0\"). Amounts are in base "
    "units using the asset's decimals.\n"
    "Assets (symbol -> address, decimals): "
    "USDC 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48 (6), "
    "USDT 0xdAC17F958D2ee523a2206206994597C13D831ec7 (6), "
    "DAI 0x6B175474E89094C44Da98b954EedeAC495271d0F (18), "
    "WETH 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2 (18), "
    "WBTC 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599 (8).\n"
    "Functions (exact arg order):\n"
    "- supply(address,uint256,address,uint16) — asset, amount, onBehalfOf, referralCode\n"
    "- withdraw(address,uint256,address) — asset, amount, to\n"
    "- borrow(address,uint256,uint256,uint16,address) — asset, amount, interestRateMode, referralCode, onBehalfOf\n"
    "- repay(address,uint256,uint256,address) — asset, amount, interestRateMode, onBehalfOf\n"
    "Defaults: onBehalfOf and to are your own wallet — emit \"<wallet>\"; "
    "referralCode is \"0\"; interestRateMode is \"2\" (variable)."
)

PROTOCOL_REFERENCES = {"safe": SAFE_REFERENCE, "aave": AAVE_REFERENCE}


def _format_account_context(ac: dict) -> str:
    owners = ", ".join(ac.get("owners", []))
    return (f"ACCOUNT CONTEXT\nSafe: {ac['safe']}\n"
            f"Owners (in order): {owners}\nCurrent threshold: {ac['threshold']}")


def render(context: dict[str, Any]) -> list[dict[str, str]]:
    vars_ = context.get("vars", {}) if isinstance(context, dict) else {}
    chat: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    parts: list[str] = []
    reference = PROTOCOL_REFERENCES.get(vars_.get("protocol"))
    if reference:
        parts.append(reference)
    account_context = vars_.get("account_context")
    if account_context:
        parts.append(_format_account_context(account_context))
    if parts:
        chat.append({"role": "system", "content": "\n\n".join(parts)})
    messages = vars_.get("messages")
    if messages:
        chat.extend(messages)
    else:
        chat.append({"role": "user", "content": vars_.get("user_message", "")})
    return chat
