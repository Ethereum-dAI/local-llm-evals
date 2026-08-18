"""Deterministic builders for multi-round conversation eval cases.

`generation.build_multiturn_case` produces exactly one conversation shape — an
ablated turn 1, a canned clarification, a completing turn 3 — so every
multi-turn case in the app-contract dataset is 2 rounds long and tests one
thing: can the model carry a single missing field across one exchange. This
module generalises that to 2-6 rounds across six distinct failure modes.

A ROUND is one user turn plus the assistant's reply. An R-round case therefore
carries R user messages and R-1 canned assistant messages, and only the model's
reply to user message R is scored. Gold is computed from the FINAL EFFECTIVE
intent via `generation.gold_calls`, never parsed from any surface.

Every canned assistant turn is ON-POLICY for the app's own system prompt, which
is load-bearing: that prompt says to emit the tool call as soon as the action
plus its values are known and never to ask for confirmation, so a scripted
assistant that stalled on a complete request would be showing the model an
example of the behaviour we score it for not doing. Only three assistant moves
are used, and each is something the prompt actually sanctions:

  * ask for a genuinely missing field (the prompt's own instruction),
  * answer a non-actionable question in prose (no tool applies),
  * report that a completed request was prepared (what the app does after a call).

The four conversational-memory mechanisms:

`progressive` (2-4 rounds) — the action's three fields are revealed one per
  round in a seeded permutation. Tests state accumulation. Round count is capped
  at 4 because there are only three fields to withhold.

`correction` (2-6 rounds) — one field is withheld so the assistant has a
  standing, legitimate reason to keep asking, and the user revises an
  already-stated field in every later round, naming the old value as they go
  ("actually make it 7.77, not 6.02"). Gold takes the LATEST value of each
  field, so a model that latches onto the first one it saw fails. Fields may be
  revised more than once in a single conversation.

`distractor` (3-6 rounds) — one field is withheld, and the rounds between the
  ask and the answer are non-actionable questions the assistant answers in
  prose before re-asking. Some answers deliberately contain a NUMBER ("around
  12 gwei", "6 decimals") that does not belong in the call. Tests whether the
  pending intent survives interruption uncontaminated.

`switch` (2-6 rounds) — the user completes one request, the assistant reports it
  prepared, and the final round abandons it for a different one. Gold is the
  FINAL intent ALONE, so both re-emitting the stale call and emitting the pair
  score 0.

The last two mechanisms test the CONTRACT BOUNDARY rather than conversational
memory — what the wallet can and cannot execute — and both were added because a
field the scorer checked had become degenerate or untested:

`exact_output` (2-4 rounds) — the user answers "how much?" with an amount of the
  DESTINATION token. The wallet only does exact-input swaps, so gold is NO CALL.
  Fixes `amount_side` being `"input"` in all 436 swap golds, which made it a free
  field any model could hardcode.

`token_address` (2-3 rounds) — the withheld token field arrives as a 0x contract
  address, which `pf/tools.app.json` documents and the wallet's registry really
  does resolve. Gold carries the address verbatim. Tests 42-char hex pass-through,
  a failure mode nothing else in the benchmark exercises.

Everything here is a pure function of its inputs plus an explicit
`random.Random`, so a fixed seed yields byte-identical output.
"""
from __future__ import annotations

import random

from wallet_evals.generation import (
    ENS_NAMES, MUTATORS, SWAP_TEMPLATES, TRANSFER_TEMPLATES, gold_calls,
    random_address, render_surface,
)
from wallet_evals.intents import LOOKUP, format_expected_summary

#: The fields that fully specify an intent, per action. Sorted-tuple subsets of
#: these key PREFIX_TEMPLATES below.
ACTION_FIELDS: dict[str, tuple[str, ...]] = {
    "transfer": ("amount", "token", "recipient"),
    "swap": ("amount", "from_token", "to_token"),
}

#: Short id fragment per mechanism. Ids read `conv-corr-5r-0012`.
MECHANISM_ABBREV: dict[str, str] = {
    "progressive": "pd", "correction": "corr", "distractor": "dist", "switch": "switch",
    "exact_output": "xout", "token_address": "addr",
}

#: Rounds each mechanism can produce. `progressive` stops at 4 because it
#: discloses one of three fields per round; `distractor` starts at 3 because a
#: 2-round distractor case has no room for a distractor and would be a duplicate
#: of a 2-round `progressive` one.
MECHANISM_ROUNDS: dict[str, tuple[int, ...]] = {
    "progressive": (2, 3, 4),
    "correction": (2, 3, 4, 5, 6),
    "distractor": (3, 4, 5, 6),
    "switch": (2, 3, 4, 5, 6),
    # Both contract-boundary mechanisms stay short. Their point is a single
    # property of the FINAL turn (an output-side amount; an address in place of a
    # symbol), and stretching that over six rounds would re-test `distractor`'s
    # interruption-survival rather than the boundary itself.
    "exact_output": (2, 3, 4),
    "token_address": (2, 3),
}

# --------------------------------------------------------------------------
# Surface banks. Round-1 openers are keyed by (action, sorted tuple of the
# fields they STATE) so any subset can open a conversation — the three pair
# entries per action are the same partial surfaces generation.ABLATION_TEMPLATES
# uses, restated here so this module owns its own bank and cannot perturb the
# frozen app-contract selection by growing a shared one.
# --------------------------------------------------------------------------
PREFIX_TEMPLATES: dict[tuple[str, tuple[str, ...]], list[str]] = {
    ("transfer", ()): [
        "I need to move some funds out of my wallet.",
        "Let's do a transfer.",
        "I want to send someone some crypto.",
    ],
    ("transfer", ("amount",)): [
        "I want to send {amount} out of my wallet.",
        "Let's move {amount} today.",
    ],
    ("transfer", ("token",)): [
        "I want to send some {token}.",
        "Let's send a bit of my {token}.",
    ],
    ("transfer", ("recipient",)): [
        "I need to send something to {recipient}.",
        "Let's get some funds over to {recipient}.",
    ],
    ("transfer", ("amount", "token")): [
        "Send {amount} {token}",
        "Let's move {amount} {token}.",
    ],
    ("transfer", ("amount", "recipient")): [
        "Send {amount} to {recipient}",
        "Move {amount} over to {recipient}.",
    ],
    ("transfer", ("recipient", "token")): [
        "Send some {token} to {recipient}",
        "Move a bit of my {token} to {recipient}.",
    ],
    ("swap", ()): [
        "I'd like to do a swap.",
        "Let's trade some tokens.",
    ],
    ("swap", ("amount",)): [
        "I'd like to swap {amount}.",
        "Let's trade {amount} of one of my tokens.",
    ],
    ("swap", ("from_token",)): [
        "I want to swap some {from_token}.",
        "Let's move some of my {from_token} into something else.",
    ],
    ("swap", ("to_token",)): [
        "I'd like to pick up some {to_token}.",
        "I want to end up holding {to_token}.",
    ],
    ("swap", ("amount", "from_token")): [
        "Swap {amount} {from_token}",
        "Let's trade {amount} {from_token}.",
    ],
    ("swap", ("amount", "to_token")): [
        "Swap {amount} for {to_token}",
        "Get me {to_token} with {amount} of what I hold.",
    ],
    ("swap", ("from_token", "to_token")): [
        "Swap {from_token} for {to_token}",
        "Trade my {from_token} into {to_token}.",
    ],
}

#: The assistant's clarifying question for a missing field, per action.
ASKS: dict[tuple[str, str], list[str]] = {
    ("transfer", "amount"): ["How much would you like to send?",
                             "How much should I send?"],
    ("transfer", "token"): ["Which token?", "Which token would you like to send?"],
    ("transfer", "recipient"): ["Which address or ENS should I send it to?",
                               "Who should receive it?"],
    ("swap", "amount"): ["How much would you like to swap?",
                         "How much should I trade?"],
    ("swap", "from_token"): ["Which token do you want to swap from?",
                            "What are we swapping out of?"],
    ("swap", "to_token"): ["Which token do you want to receive?",
                          "What should I swap it into?"],
}

#: The user's completing fragment that supplies a missing field.
ANSWERS: dict[tuple[str, str], list[str]] = {
    ("transfer", "amount"): ["{amount}", "{amount} please", "make it {amount}"],
    ("transfer", "token"): ["{token}", "in {token}", "{token} please"],
    ("transfer", "recipient"): ["to {recipient}", "{recipient}", "send it to {recipient}"],
    ("swap", "amount"): ["{amount}", "{amount} please", "make it {amount}"],
    ("swap", "from_token"): ["{from_token}", "from {from_token}", "out of my {from_token}"],
    ("swap", "to_token"): ["{to_token}", "for {to_token}", "into {to_token}"],
}

#: The user revising an already-stated field. Both values are named, so the
#: surface contains the stale one — that is the whole point: gold carries only
#: the new value, and a model that keeps the first number it read fails.
REVISIONS: dict[tuple[str, str], list[str]] = {
    ("transfer", "amount"): [
        "actually make it {new}, not {old}",
        "wait — {new} instead of {old}",
        "correction: {new}, not {old}",
    ],
    ("transfer", "token"): [
        "actually let's use {new} instead of {old}",
        "sorry — {new}, not {old}",
    ],
    ("transfer", "recipient"): [
        "actually send it to {new} instead of {old}",
        "change the recipient to {new}, not {old}",
    ],
    ("swap", "amount"): [
        "actually make it {new}, not {old}",
        "wait — {new} instead of {old}",
        "correction: {new}, not {old}",
    ],
    ("swap", "from_token"): [
        "actually swap out of {new}, not {old}",
        "use my {new} instead of my {old}",
    ],
    ("swap", "to_token"): [
        "actually I want {new} instead of {old}",
        "make the output {new}, not {old}",
    ],
}

#: Revision targets. Disjoint from datasets/seeds.conversations.yaml,
#: datasets/seeds.yaml, datasets/seeds.arithmetic.yaml and
#: datasets/finetune_seeds.yaml by exact-string set intersection, so a
#: correction case's FINAL (scored) amount is never an amount any model was
#: trained on or already measured against.
ALT_AMOUNTS: tuple[str, ...] = (
    "0.0125", "7.77", "260000.5", "0.00000091", "1.4142", "93.06", "0.000505",
    "48200.75",
)

#: Every token the app contract knows. Revisions pick a different one from here,
#: so a revised symbol is always resolvable by `intents.build_*_call`.
ALT_TOKENS: tuple[str, ...] = tuple(LOOKUP["tokens"])

#: Assistant acknowledgements, prepended to a re-ask after an interruption.
ACKS: tuple[str, ...] = ("Got it.", "Noted.", "Understood.", "Updated.")

#: Non-actionable user turns and the assistant's prose answer. No tool applies
#: to any of them, so answering in prose is on-policy. Several answers carry a
#: number that must NOT reach the call — a contamination trap, not filler.
DISTRACTORS: tuple[tuple[str, str], ...] = (
    ("What's gas looking like right now?", "Base fee is around 12 gwei at the moment."),
    ("Remind me which chain we're on?", "Ethereum mainnet."),
    ("How long do transfers usually take?",
     "Usually a block or two — around 15 seconds."),
    ("Is USDC an ERC-20?", "Yes, USDC is a standard ERC-20 token with 6 decimals."),
    ("Do I need ETH for gas?", "Yes — gas is always paid in ETH on mainnet."),
    ("What's my wallet address again?",
     "It's whichever account is currently selected in the wallet."),
    ("Nice, this is much faster than the old app.", "Glad to hear it."),
    ("Can you show me my recent transactions later?", "Sure, just ask any time."),
    ("What's the current ETH price?", "I don't have live price data."),
    ("Is 25 gwei high?", "It's moderate — fine for a non-urgent transaction."),
)

#: The assistant reporting a completed request (what the app does after a call).
#: Names the stale values, which is exactly the pressure a `switch` case applies.
DONE_REPORTS: dict[str, list[str]] = {
    "transfer": [
        "I've prepared the transfer of {amount} {token} to {recipient}.",
        "Done — {amount} {token} to {recipient} is queued for your signature.",
    ],
    "swap": [
        "I've prepared the swap of {amount} {from_token} into {to_token}.",
        "Done — swapping {amount} {from_token} for {to_token} is queued.",
    ],
}

#: The user abandoning the pending request, and the assistant's acknowledgement.
CANCELS: tuple[str, ...] = (
    "Actually, scrap that.", "Forget that one.", "Cancel that, please.",
    "Never mind that one.",
)
CANCEL_ACKS: tuple[str, ...] = (
    "No problem — cancelled.", "Done, I've dropped it.", "Cancelled.",
)

#: Lead-in to the replacement request in a `switch` case's final round. The
#: 2- and 3-round variants have no dedicated cancel round, so their final turn
#: carries the cancellation itself (see `build_switch_case`).
SWITCH_LEADINS: tuple[str, ...] = (
    "Instead, ", "What I actually want is this: ", "Let's do this one instead — ",
)

#: The user answering "how much?" with an amount of the DESTINATION token — an
#: exact-OUTPUT swap. The wallet cannot execute one: `amount_side` is
#: `enum: ["input"]` in pf/tools.app.json, the `amount` description says "Do not
#: use this tool when the user specifies only the desired output amount", and two
#: independent guards enforce it (ChatDashboardView.swift:3335 and
#: SlashCommandParser.swift:66, the latter throwing
#: `malformedArgument("amount_side", "only input is supported")`). So gold is NO
#: CALL, and the wallet's own error text spells out the correct reply: "Only
#: exact-input swaps are supported. Say how much of the input token to spend."
#:
#: Every phrasing must be unambiguously output-side. "Buy {to_token} with
#: {amount} {from_token}" is already in SWAP_TEMPLATES as *input*-side
#: output-FIRST phrasing, and reading these as input-side would make the case
#: unanswerable rather than hard — so each one pins the amount to the received
#: token with "out"/"end up with"/"receive", and names the input token only as an
#: open quantity ("whatever that takes").
EXACT_OUTPUT_ASKS: tuple[str, ...] = (
    "I need exactly {amount} {to_token} out — spend whatever {from_token} that takes.",
    "I want to end up with exactly {amount} {to_token}, however much {from_token} it costs.",
    "Make the output exactly {amount} {to_token}, whatever {from_token} is needed.",
    "Buy precisely {amount} {to_token}; take however much {from_token} that requires.",
    "I need to receive exactly {amount} {to_token} — spend as much {from_token} as needed.",
)

#: Token symbol -> contract address, for the tokens that HAVE one. Native ETH is
#: excluded: `datasets/lookup.json` gives it `address: null` (the app models it as
#: `.native`), so there is no address form of ETH to name. These are the mainnet
#: addresses, matching both lookup.json and the mainnet rows of the wallet's
#: PortedAppEncoding table.
TOKEN_ADDRESSES: dict[str, str] = {
    symbol: meta["address"] for symbol, meta in LOOKUP["tokens"].items()
    if meta.get("address")
}

#: How the user names a token by its contract address. The bare-address variant
#: matters most: it is the surface with nothing but 42 characters of hex to copy.
ADDRESS_ANSWERS: tuple[str, ...] = (
    "{address}",
    "this one: {address}",
    "the token at {address}",
    "use {address}",
)

_FULL_TEMPLATES: dict[str, list[str]] = {
    "transfer": TRANSFER_TEMPLATES, "swap": SWAP_TEMPLATES,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _protocol(action: str) -> str:
    return "uniswap" if action == "swap" else "transfer"


def _prefix_key(action: str, stated: list[str]) -> tuple[str, tuple[str, ...]]:
    return action, tuple(sorted(stated))


def _pick(choices: list[str], blocked: set[str], rng: random.Random,
          field: str) -> str:
    remaining = [c for c in choices if c not in blocked]
    if not remaining:  # pragma: no cover - the banks are sized to prevent this
        raise ValueError(f"no revision left for {field!r} outside {sorted(blocked)}")
    return rng.choice(remaining)


def _revised_value(intent: dict, field: str, original: str, rng: random.Random,
                   ens_bank: tuple[str, ...] = ENS_NAMES) -> str:
    """A new value for `field`: different from the CURRENT value and from the
    ORIGINAL one.

    Excluding the original is what makes a correction case a real test. Without
    it a field could be revised away and then back again (USDC -> ETH -> USDC),
    leaving gold identical to the seed intent — so a model that ignored every
    correction and kept the first value it read would still score 1, and the case
    would measure nothing. Intermediate values may repeat; only the round trip to
    the starting value is forbidden, which keeps the small token bank usable
    across five revisions.

    A swap side additionally excludes the opposite side: a self-swap is not a
    real intent (`generate_conversation_cases._valid_intent` drops those).
    """
    current = intent[field]
    if field == "amount":
        return _pick(list(ALT_AMOUNTS), {current, original}, rng, field)
    if field == "recipient":
        # A fresh random address is new by construction; an ENS name has to be
        # checked, since the bank is finite. Coin-flip between the two forms so a
        # revised recipient is as likely to be an ENS name as an address —
        # otherwise corrections would quietly push the slice towards addresses.
        blocked = {current, original}
        if rng.random() < 0.5:
            return random_address(rng)
        # `ens_bank` is a parameter, not a constant, because a REVISED recipient
        # lands in gold. The dev set must draw from its own bank: sharing names with
        # the frozen test set would mean selecting checkpoints partly on values that
        # appear in the reported number. Caught by test_dev_set.py after exactly that
        # slipped through.
        return _pick(list(ens_bank), blocked, rng, field)
    if field == "token":
        return _pick(list(ALT_TOKENS), {current, original}, rng, field)
    other = "to_token" if field == "from_token" else "from_token"
    return _pick(list(ALT_TOKENS), {current, original, intent[other]}, rng, field)


def _mutated(text: str, rng: random.Random) -> tuple[str, list[str]]:
    """Surface noise for one USER turn — the assistant turns are the wallet's own
    scripted output, and mutating them would be noise the product never produces.

    The `punctuation` mutator is deliberately EXCLUDED here (the same
    skip-one-mutator shape `generation.build_separator_case` uses, for the
    opposite reason). It comma-groups amounts, so a case could fail because the
    model reread "260,000.5" as "260.0005" rather than because it lost track of
    a value across rounds — and thousands-separator handling already has its own
    labelled slice (the `arithmetic-*` cases from datasets/seeds.arithmetic.yaml).
    Confounding the two would make every conversation failure ambiguous. The
    other four mutators never touch digits, so a conversation surface stays
    digit-faithful while still reading as noisy human text.
    """
    labels: list[str] = []
    for name, fn in MUTATORS:
        if name == "punctuation":
            continue
        if rng.random() < 0.5:
            text = fn(text, rng)
            labels.append(name)
    return text, labels


def _build_case(*, final_intent: dict, mechanism: str, rounds: int,
                messages: list[dict], mutators: list[str], idx: int,
                notes: str, expected_calls: list[dict] | None = None) -> dict:
    """`expected_calls` overrides the computed gold, and is only ever passed as
    `[]` — by `build_exact_output_case`, whose request the wallet cannot execute at
    all. Gold is still never PARSED from a surface; an override just states that
    the correct action is to make no call."""
    action = final_intent["action"]
    calls = gold_calls(final_intent) if expected_calls is None else expected_calls
    md = {
        "id": f"conv-{MECHANISM_ABBREV[mechanism]}-{rounds}r-{idx:04d}",
        "source": "generated-conversation",
        "language": "english",
        "category": f"conversation-{mechanism}-{rounds}r",
        "protocol": _protocol(action),
        "difficulty": "hard" if rounds >= 4 else "medium",
        "level": "payload",
        "query_type": "multi_turn",
        "rounds": rounds,
        "mechanism": mechanism,
        "requires": [],
        "style": "conversational",
        "mutators": mutators,
        "expected_calls": calls,
        "notes": notes,
    }
    _assert_shape(messages, rounds, md["id"])
    return {"vars": {"messages": messages,
                     "expected_summary": format_expected_summary(calls)},
            "metadata": md}


def _assert_shape(messages: list[dict], rounds: int, case_id: str) -> None:
    """An R-round conversation is R user turns alternating with R-1 assistant
    turns, ending on the user turn the model is scored on. Asserted at build
    time rather than only in a test: a builder that miscounts would silently
    emit a case whose scored turn is not the last one, and the dataset would
    look fine."""
    roles = [m["role"] for m in messages]
    expected = ["user" if i % 2 == 0 else "assistant" for i in range(2 * rounds - 1)]
    assert roles == expected, f"{case_id}: turn roles {roles} != {expected}"


# --------------------------------------------------------------------------
# mechanism builders
# --------------------------------------------------------------------------
def build_progressive_case(intent: dict, order: tuple[str, ...], rounds: int,
                           rng: random.Random, idx: int) -> dict:
    """Reveal the action's fields one per round, in `order`.

    The LAST `rounds - 1` fields of `order` are withheld from the opener and
    supplied one per subsequent round, so a 4-round case opens with the bare
    action and a 2-round case opens with two of the three fields already given.
    """
    action = intent["action"]
    withheld = list(order[len(order) - (rounds - 1):])
    stated = [f for f in ACTION_FIELDS[action] if f not in withheld]

    opener, mutators = _mutated(
        render_surface(rng.choice(PREFIX_TEMPLATES[_prefix_key(action, stated)]), intent),
        rng)
    messages: list[dict] = [{"role": "user", "content": opener}]
    for field in withheld:
        messages.append({"role": "assistant",
                         "content": rng.choice(ASKS[(action, field)])})
        answer, labels = _mutated(
            render_surface(rng.choice(ANSWERS[(action, field)]), intent), rng)
        messages.append({"role": "user", "content": answer})
        mutators.extend(labels)

    notes = (f"progressive disclosure over {rounds} rounds; "
             f"opened with {stated or ['(action only)']}, "
             f"then revealed {withheld} one per round")
    return _build_case(final_intent=intent, mechanism="progressive", rounds=rounds,
                       messages=messages, mutators=sorted(set(mutators)), idx=idx,
                       notes=notes)


def build_correction_case(intent: dict, withheld: str, rounds: int,
                          rng: random.Random, idx: int,
                          ens_bank: tuple[str, ...] = ENS_NAMES) -> dict:
    """Withhold one field, then revise an already-stated field every round.

    `rounds - 1` revisions in total: one per intermediate round plus one riding
    along with the final answer. Gold is the intent AFTER all of them, so the
    stale values scattered through the transcript are all wrong answers.
    """
    action = intent["action"]
    stated = [f for f in ACTION_FIELDS[action] if f != withheld]
    current = dict(intent)

    opener, mutators = _mutated(
        render_surface(rng.choice(PREFIX_TEMPLATES[_prefix_key(action, stated)]), current),
        rng)
    messages: list[dict] = [{"role": "user", "content": opener}]
    messages.append({"role": "assistant", "content": rng.choice(ASKS[(action, withheld)])})

    trace: list[str] = []

    def revise() -> str:
        """Apply one revision to `current` and return the user's phrasing."""
        field = rng.choice(stated)
        old = current[field]
        new = _revised_value(current, field, intent[field], rng, ens_bank)
        current[field] = new
        trace.append(f"{field} {old}->{new}")
        return render_surface(rng.choice(REVISIONS[(action, field)]),
                              {"old": old, "new": new})

    for _ in range(rounds - 2):
        turn, labels = _mutated(revise(), rng)
        messages.append({"role": "user", "content": turn})
        mutators.extend(labels)
        messages.append({"role": "assistant",
                         "content": f"{rng.choice(ACKS)} "
                                    f"{rng.choice(ASKS[(action, withheld)])}"})

    # Final round: supply the withheld field AND slip in one last revision, so
    # even a 2-round case carries a correction.
    answer = render_surface(rng.choice(ANSWERS[(action, withheld)]), current)
    final, labels = _mutated(f"{answer} — {revise()}", rng)
    messages.append({"role": "user", "content": final})
    mutators.extend(labels)

    notes = (f"{rounds} rounds; withheld={withheld}; "
             f"{len(trace)} revisions: {', '.join(trace)}")
    return _build_case(final_intent=current, mechanism="correction", rounds=rounds,
                       messages=messages, mutators=sorted(set(mutators)), idx=idx,
                       notes=notes)


def build_distractor_case(intent: dict, withheld: str, rounds: int,
                          rng: random.Random, idx: int) -> dict:
    """Withhold one field, then interrupt with `rounds - 2` off-topic exchanges
    before the user finally answers. Gold is the unchanged intent."""
    action = intent["action"]
    stated = [f for f in ACTION_FIELDS[action] if f != withheld]

    opener, mutators = _mutated(
        render_surface(rng.choice(PREFIX_TEMPLATES[_prefix_key(action, stated)]), intent),
        rng)
    messages: list[dict] = [{"role": "user", "content": opener}]
    messages.append({"role": "assistant", "content": rng.choice(ASKS[(action, withheld)])})

    picked = rng.sample(DISTRACTORS, rounds - 2)
    for question, reply in picked:
        turn, labels = _mutated(question, rng)
        messages.append({"role": "user", "content": turn})
        mutators.extend(labels)
        messages.append({"role": "assistant",
                         "content": f"{reply} {rng.choice(ASKS[(action, withheld)])}"})

    answer, labels = _mutated(
        render_surface(rng.choice(ANSWERS[(action, withheld)]), intent), rng)
    messages.append({"role": "user", "content": answer})
    mutators.extend(labels)

    notes = (f"{rounds} rounds; withheld={withheld}; "
             f"{len(picked)} distractor exchange(s) in between")
    return _build_case(final_intent=intent, mechanism="distractor", rounds=rounds,
                       messages=messages, mutators=sorted(set(mutators)), idx=idx,
                       notes=notes)


def build_switch_case(first: dict, second: dict, rounds: int,
                      rng: random.Random, idx: int) -> dict:
    """Complete one request, then abandon it for `second` in the final round.

    Gold is `second` ALONE: re-emitting the first call, or emitting both, is a
    miss. For 4+ rounds one of the filler rounds is an explicit cancellation, so
    the abandonment is stated well before the replacement arrives.
    """
    opener, mutators = _mutated(
        render_surface(rng.choice(_FULL_TEMPLATES[first["action"]]), first), rng)
    messages: list[dict] = [{"role": "user", "content": opener}]
    messages.append({"role": "assistant",
                     "content": render_surface(
                         rng.choice(DONE_REPORTS[first["action"]]), first)})

    fillers = rounds - 2
    cancelled_early = fillers > 0 and rounds >= 4
    chatter = rng.sample(DISTRACTORS, fillers - 1 if cancelled_early else fillers)
    for question, reply in chatter:
        turn, labels = _mutated(question, rng)
        messages.append({"role": "user", "content": turn})
        mutators.extend(labels)
        messages.append({"role": "assistant", "content": reply})
    # The cancellation is NEVER mutated — same rule as generation._PROTECTED_WORDS
    # and for the same reason it shields RAILGUN's privacy verbs: the verb is the
    # only thing that tells the model to abandon the first request, so a typo'd
    # "Forget that one." ("Forgte that one.") makes the case unanswerable rather
    # than harder. Observed: the typo mutator produced exactly that.
    if cancelled_early:
        messages.append({"role": "user", "content": rng.choice(CANCELS)})
        messages.append({"role": "assistant", "content": rng.choice(CANCEL_ACKS)})

    replacement, labels = _mutated(
        render_surface(rng.choice(_FULL_TEMPLATES[second["action"]]), second), rng)
    lead = (rng.choice(SWITCH_LEADINS) if cancelled_early
            else f"{rng.choice(CANCELS)} ")
    messages.append({"role": "user", "content": f"{lead}{replacement}"})
    mutators.extend(labels)

    notes = (f"{rounds} rounds; abandoned a {first['action']} for a "
             f"{second['action']}; "
             f"{'explicit cancel round' if cancelled_early else 'cancel in the final turn'}"
             f"; gold is the second intent only")
    return _build_case(final_intent=second, mechanism="switch", rounds=rounds,
                       messages=messages, mutators=sorted(set(mutators)), idx=idx,
                       notes=notes)


def build_exact_output_case(intent: dict, rounds: int, rng: random.Random,
                            idx: int) -> dict:
    """A swap the wallet cannot execute: the user pins the OUTPUT amount.

    The conversation withholds `amount`, so the assistant asks how much to swap
    (its own on-policy clarifying question), and the user answers with an amount
    of the DESTINATION token instead of the source. Gold is NO CALL.

    This exists because `amount_side` was constant across all 436 swap golds —
    every one `"input"` — so a model that hardcoded it scored those cases for free
    and the field measured nothing. Adding gold with `amount_side: "output"` would
    have been worse than useless: the wallet rejects that value outright, so it
    would score models on emitting a call the app throws on. Withholding the call
    instead tests the same boundary against what the wallet actually does — a
    model that hardcodes `"input"` now fails, and one that emits `"output"` still
    fails.

    3- and 4-round variants put distractor exchanges between the ask and the
    answer, so length is tested here too rather than only at 2 rounds.
    """
    action = intent["action"]
    assert action == "swap", f"exact_output is swap-only, got {action!r}"
    stated = [f for f in ACTION_FIELDS[action] if f != "amount"]

    opener, mutators = _mutated(
        render_surface(rng.choice(PREFIX_TEMPLATES[_prefix_key(action, stated)]), intent),
        rng)
    messages: list[dict] = [{"role": "user", "content": opener}]
    messages.append({"role": "assistant", "content": rng.choice(ASKS[(action, "amount")])})

    picked = rng.sample(DISTRACTORS, rounds - 2)
    for question, reply in picked:
        turn, labels = _mutated(question, rng)
        messages.append({"role": "user", "content": turn})
        mutators.extend(labels)
        messages.append({"role": "assistant",
                         "content": f"{reply} {rng.choice(ASKS[(action, 'amount')])}"})

    # The output-side demand is NEVER mutated — same rule as the cancellation in
    # build_switch_case. "exactly ... out" is the entire signal that this is an
    # output-side request; a mutator that mangled it would leave a surface that
    # reads as an ordinary input-side swap, whose correct answer is a tool call,
    # while gold still said no call. That is an unanswerable case, not a harder one.
    messages.append({"role": "user",
                     "content": render_surface(rng.choice(EXACT_OUTPUT_ASKS), intent)})

    notes = (f"{rounds} rounds; exact-OUTPUT swap request "
             f"({intent['amount']} {intent['to_token']} out of {intent['from_token']}); "
             f"the wallet supports exact-input only (amount_side enum is [\"input\"]), "
             f"so gold is no call; {len(picked)} distractor exchange(s)")
    return _build_case(final_intent=intent, mechanism="exact_output", rounds=rounds,
                       messages=messages, mutators=sorted(set(mutators)), idx=idx,
                       notes=notes, expected_calls=[])


def build_token_address_case(intent: dict, field: str, rounds: int,
                             rng: random.Random, idx: int) -> dict:
    """Withhold a token field, then supply it as a 0x CONTRACT ADDRESS.

    `pf/tools.app.json` says every token slot accepts "a 0x-prefixed contract
    address", and unlike two of its other documented forms the executor honours
    this one: `WalletTokenRegistry.token(matching:)` matches a `0x` value against
    `contractAddress` case-insensitively. Gold therefore carries the address
    VERBATIM — translating it to a symbol would measure recall of a token table
    that APP_SYSTEM does not contain.

    What this actually tests is pass-through fidelity: 42 characters of hex,
    delivered a round after the request began, have to arrive in the call intact.
    No arithmetic is involved, which makes it a clean read on a failure mode the
    rest of the benchmark cannot see.

    Contrast the `safety-refusal-unverified-token-swap` cases, which name an
    address that is NOT in the registry and expect no call. Together the two form
    a matched pair: known address -> pass it through, unknown -> refuse.
    """
    action = intent["action"]
    symbol = intent[field]
    address = TOKEN_ADDRESSES[symbol]
    # Gold follows the surface: the effective intent names the token by address.
    effective = dict(intent, **{field: address})
    stated = [f for f in ACTION_FIELDS[action] if f != field]

    opener, mutators = _mutated(
        render_surface(rng.choice(PREFIX_TEMPLATES[_prefix_key(action, stated)]),
                       effective),
        rng)
    messages: list[dict] = [{"role": "user", "content": opener}]
    messages.append({"role": "assistant", "content": rng.choice(ASKS[(action, field)])})

    picked = rng.sample(DISTRACTORS, rounds - 2)
    for question, reply in picked:
        turn, labels = _mutated(question, rng)
        messages.append({"role": "user", "content": turn})
        mutators.extend(labels)
        messages.append({"role": "assistant",
                         "content": f"{reply} {rng.choice(ASKS[(action, field)])}"})

    # Not mutated: mutate_case would refold the hex, and while the scorer and the
    # wallet both compare addresses case-insensitively, a surface whose address
    # reads "0XA0B8..." tests the mutator rather than the copy.
    answer = render_surface(rng.choice(ADDRESS_ANSWERS), {"address": address})
    messages.append({"role": "user", "content": answer})

    notes = (f"{rounds} rounds; {field} supplied as the contract address for "
             f"{symbol} ({address}); gold carries the address verbatim, as "
             f"WalletTokenRegistry.token(matching:) resolves it; "
             f"{len(picked)} distractor exchange(s)")
    return _build_case(final_intent=effective, mechanism="token_address",
                       rounds=rounds, messages=messages,
                       mutators=sorted(set(mutators)), idx=idx, notes=notes)
