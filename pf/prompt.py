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
    "missing, ask one short clarifying question in natural language and wait for "
    "the answer before calling the tool. Never invent recipient addresses, ENS "
    "names, contact names, token symbols, or amounts that the user has not provided."
)


def render(context: dict[str, Any]) -> list[dict[str, str]]:
    vars_ = context.get("vars", {}) if isinstance(context, dict) else {}
    chat: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    messages = vars_.get("messages")
    if messages:
        chat.extend(messages)
    else:
        chat.append({"role": "user", "content": vars_.get("user_message", "")})
    return chat
