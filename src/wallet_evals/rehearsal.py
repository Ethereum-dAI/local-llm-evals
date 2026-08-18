"""Rehearsal rows: wallet-adjacent turns whose correct answer is NOT a tool call.

TRAINING ONLY. Nothing here is ever scored — the eval and dev sets are built from
their own generators, and `tests/test_finetune_conversations.py` checks that no
training surface appears in any held-out set.

WHY. The 1000-case benchmark and the epoch curve agree on the shape of the damage,
and it is not "the model forgot how to call tools" — it is the opposite. The
fine-tune became unable to NOT call one:

  * `distractor` collapsed 29/40 -> 14/40 as training progressed: an irrelevant
    aside mid-conversation pulls out a premature call;
  * novel ENS names produced FALSE REFUSALS, the model having learned that the only
    valid recipient shape is the one string it saw;
  * `exact_output` scored 0/15 at every epoch: asked for something the tool cannot
    express, it emitted a call anyway.

Training was ~97% "emit a call" rows. That prior is the common cause. These rows
put plain-language replies back into the distribution, in the wallet's own voice
and on wallet-adjacent topics, so "answer in prose" stays reachable.

This is a SCOPED proxy for replay, not general-capability rehearsal. A true replay
buffer would mix in the base model's own outputs over a broad instruction set;
that needs a licensed corpus and base-model inference, and it would defend
capabilities this eval never measures. What is here targets the one measured
regression, offline and deterministically. If a broader replay set ever lands,
this stays as the targeted slice.

Keep every answer free of a DSL opener (`<|tool_call>`, `functools`, `<tool_call>`)
or the row teaches the exact opposite of its lesson — asserted in
tests/test_rehearsal.py.
"""
from __future__ import annotations

import random

#: (user turn, assistant reply). Wallet-adjacent, deliberately: a rehearsal row
#: about poetry would be off-distribution enough that the model could learn to
#: separate "wallet context -> always call" from "other -> prose", which is the
#: shortcut this is meant to remove. These sit INSIDE the wallet's domain and still
#: have no call to make.
REHEARSAL_TURNS: tuple[tuple[str, str], ...] = (
    # --- capability questions -------------------------------------------------
    ("what tokens can you send?",
     "I can send ETH, USDC, DAI and WETH. Tell me an amount and a recipient and "
     "I'll put the transfer together."),
    ("can you swap on a different chain?",
     "I work on the chain this wallet is connected to. If you need a different "
     "one, switch networks in the wallet first and then ask me again."),
    ("do you charge a fee?",
     "I don't add a fee. You'll still pay the network's gas, and a swap pays the "
     "pool's own fee — both are shown for approval before anything is signed."),
    ("what's the difference between ETH and WETH?",
     "ETH is the native coin; WETH is an ERC-20 token that wraps it 1:1 so "
     "contracts can handle it like any other token. Swaps usually route through "
     "WETH, and you can hold either."),
    ("can you check my balance?",
     "I can't read balances — I only prepare transfers and swaps for you to "
     "approve. The wallet's main screen shows your balances."),
    ("are you able to stake for me?",
     "Not yet. I can do transfers and swaps. Staking isn't something I can "
     "prepare."),
    # --- clarification without enough to act ----------------------------------
    ("hey",
     "Hi. I can send tokens or swap them for you — what would you like to do?"),
    ("i need to move some money",
     "Happy to help. Which token, how much, and where should it go?"),
    ("can you help me with something",
     "Of course. I can prepare a transfer or a swap — tell me what you need."),
    ("actually never mind",
     "No problem. Let me know if you want to send or swap something later."),
    ("wait, let me check something first",
     "Sure, take your time. I'll be here when you're ready."),
    ("hold on",
     "No rush — tell me when you want to continue."),
    # --- safety-adjacent, but questions rather than requests ------------------
    ("is it safe to send to a contract address?",
     "It depends on the contract. Sending tokens to one that isn't built to "
     "receive them can lose the funds permanently, so check the address is what "
     "you think it is before you approve anything."),
    ("what happens if i send to the wrong address?",
     "Transfers can't be reversed once they're confirmed, and nobody can recover "
     "them for you. That's why the recipient is worth double-checking before you "
     "approve."),
    ("why do you need the recipient address?",
     "Because a transfer has to name where the funds go. I'll take a 0x address "
     "or an ENS name."),
    # --- meta / conversational -------------------------------------------------
    ("thanks!",
     "Any time. Let me know if you need another transfer or swap."),
    ("that worked, thank you",
     "Glad it went through. Anything else you'd like to send or swap?"),
    ("sorry, i'm being slow today",
     "No need to apologise. Whenever you're ready, tell me the token, the amount "
     "and the recipient."),
    ("who built you?",
     "I'm the assistant built into this wallet. I prepare transfers and swaps; "
     "you approve or reject each one."),
    ("can you explain what gas is?",
     "Gas is what you pay the network to include your transaction. It moves with "
     "demand, so the same transfer costs more when the chain is busy."),
)


def build_rehearsal_case(turn: tuple[str, str], rng: random.Random,
                         idx: int) -> dict:
    """One rehearsal case: a single user turn whose gold is NO call.

    Shaped like every other generated case so it flows through `case_to_example`,
    `render` and the scorer unchanged. `target_text` carries the reply, which
    `finetune.assistant_target` honours only because gold is empty.
    """
    user, reply = turn
    return {
        "vars": {"messages": [{"role": "user", "content": user}],
                 "expected_summary": "(no tool call)"},
        "metadata": {
            "id": f"rehearsal-{idx:04d}",
            "source": "generated-rehearsal",
            "language": "english",
            "category": "rehearsal",
            "protocol": None,
            "difficulty": "easy",
            "level": "payload",
            "query_type": "single_turn",
            "requires": [],
            "style": "conversational",
            "mutators": [],
            "expected_calls": [],
            "target_text": reply,
            "notes": "rehearsal row: prose answer, no tool call — counterweight to "
                     "a training mix that is otherwise ~97% call-emitting",
        },
    }
