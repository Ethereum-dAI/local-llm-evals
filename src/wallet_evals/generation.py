"""Pure, deterministic helpers for generating eval cases from seed intents.

Everything here is a pure function of its inputs (including an explicit
`random.Random` for any sampling), so a fixed seed yields byte-identical output.
Gold is computed from the structured intent via wallet_evals.intents, never
parsed from the rendered surface — so surface mutation is always safe.
"""
from __future__ import annotations

import itertools
import random
import re
from typing import Callable

from wallet_evals.intents import (
    resolve_recipient, build_transfer_call, build_swap_call, format_expected_summary,
)


def random_address(rng: random.Random) -> str:
    """A realistic-looking lowercase 20-byte hex address from `rng`."""
    return "0x" + "".join(rng.choice("0123456789abcdef") for _ in range(40))


def mutate_case(text: str, rng: random.Random) -> str:
    """Random per-character upper/lower casing (preserves spelling)."""
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text)


def mutate_punctuation(text: str, rng: random.Random) -> str:
    """Add thousands-commas to amount-like digit runs and noisy trailing punctuation.

    Skips any whitespace-delimited token containing "0x" so hex addresses are
    never corrupted: gold is computed, but a comma-mangled address is unanswerable
    to a real model and would spuriously depress eval scores.
    """
    def comma(m: "re.Match[str]") -> str:
        # Group only the integer part; never touch fractional digits after a "."
        # (grouping a decimal like 12.3456 -> 12.3,456 corrupts the number).
        intpart, frac = m.group(1), m.group(2) or ""
        if len(intpart) <= 3:
            return intpart + frac
        rev = intpart[::-1]
        grouped = ",".join(rev[i:i + 3] for i in range(0, len(rev), 3))[::-1]
        return grouped + frac

    words = [
        w if "0x" in w.lower() else re.sub(r"(\d+)(\.\d+)?", comma, w)
        for w in text.split(" ")
    ]
    return " ".join(words) + rng.choice(["", "!", "!!", "...", " please"])


# Token symbols are never typo'd: a corrupted symbol (USDC->UDSC) makes the
# request genuinely ambiguous, which correctly causes a careful model to ask for
# clarification — that penalizes caution rather than testing parsing capability.
_PROTECTED_WORDS = {"ETH", "WETH", "USDC", "DAI", "USDT", "WBTC"}


def mutate_typos(text: str, rng: random.Random) -> str:
    """Swap two adjacent letters in one word — never digits/hex/token symbols."""
    words = text.split(" ")
    candidates = [
        i for i, w in enumerate(words)
        if w.isalpha() and len(w) > 3 and w.upper() not in _PROTECTED_WORDS
    ]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = list(words[i])
    j = rng.randrange(len(w) - 1)
    w[j], w[j + 1] = w[j + 1], w[j]
    words[i] = "".join(w)
    return " ".join(words)


def mutate_filler(text: str, rng: random.Random) -> str:
    """Wrap with a filler clause to lengthen the sentence."""
    prefix = rng.choice(["", "When you get a sec, ", "Hey, ", "Quick one — "])
    suffix = rng.choice(["", " thanks!", " if that's ok", " whenever"])
    return f"{prefix}{text}{suffix}"


def mutate_tone(text: str, rng: random.Random) -> str:
    """Prepend a tone wrapper (angry / funny / polite)."""
    return rng.choice(["", "Ugh, just ", "lol ok ", "Please kindly "]) + text


MUTATORS: list[tuple[str, Callable[[str, random.Random], str]]] = [
    ("case", mutate_case),
    ("punctuation", mutate_punctuation),
    ("typos", mutate_typos),
    ("filler", mutate_filler),
    ("tone", mutate_tone),
]


def apply_mutators(text: str, rng: random.Random) -> tuple[str, list[str]]:
    """Apply a seeded random subset of mutators; return (text, labels-applied)."""
    labels: list[str] = []
    for name, fn in MUTATORS:
        if rng.random() < 0.5:
            text = fn(text, rng)
            labels.append(name)
    return text, labels


TRANSFER_TEMPLATES: list[str] = [
    "Send {amount} {token} to {recipient}",
    "Move {amount} {token} to {recipient}",
    "Could you send {amount} {token} over to {recipient}?",
    "Transfer {amount} {token} to {recipient}",
    "/transfer {amount} {token} to {recipient}",
    "I want to send {amount} {token} to {recipient} now",
    "pls send {amount} {token} → {recipient}",
]

SWAP_TEMPLATES: list[str] = [
    "Swap {amount} {from_token} for {to_token}",
    "Convert {amount} {from_token} to {to_token}",
    "Trade {amount} {from_token} into {to_token}",
    "Exchange {amount} {from_token} into {to_token}",
    "/swap {amount} {from_token} to {to_token}",
    "send {amount} {from_token} to become {to_token}",  # wrong-verb hard phrasing
]

# Narrative (verbose, indirect) full-intent templates — style="narrative". All
# values are still present, but the intent is wrapped in conversational framing,
# distractor context, oblique verbs, and pronouns. Tests intent extraction from
# long prose rather than command parsing; gold is still computed from the intent.
TRANSFER_NARRATIVE_TEMPLATES: list[str] = [
    "It's been on my mind for a while — I really should get {amount} {token} over to {recipient}.",
    "Hey, quick favour: {recipient} covered me last week, so please move {amount} {token} their way.",
    "My {token} has just been sitting there doing nothing. Send {amount} of it to {recipient}, would you?",
    "So {recipient} needs paying back — go ahead and send them {amount} {token} from my wallet.",
    "I keep forgetting to do this: put {amount} {token} into {recipient} for me, thanks.",
]
SWAP_NARRATIVE_TEMPLATES: list[str] = [
    "I see that {to_token} is doing great, I think I should get some. Please use {amount} of my {from_token} for it.",
    "Honestly {from_token} has just been sitting in my wallet. Turn {amount} of it into {to_token} for me.",
    "{to_token} looks strong lately — grab some with {amount} of my {from_token}.",
    "I'm feeling bullish on {to_token}, so take {amount} {from_token} and get me into it.",
    "Time to rebalance the portfolio: move {amount} {from_token} over into {to_token}.",
]


def render_surface(template: str, intent: dict) -> str:
    """Fill a template from an intent dict (missing keys are an authoring error)."""
    return template.format(**intent)


# Param fields that participate in vary-expansion, per action.
_VARY_FIELDS = {
    "transfer": ("amount", "token", "recipient"),
    "swap": ("amount", "from_token", "to_token"),
}


def _resolve_value(raw_value, rng: random.Random):
    """Resolve one chosen param value, expanding the random_address sentinel."""
    if raw_value == "random_address":
        return random_address(rng)
    return raw_value


def expand_vary(seed: dict, rng: random.Random) -> list[dict]:
    """Expand a seed's `{vary: [...]}` fields into concrete intent dicts.

    Cross-product over all varied fields. `random_address` sentinels are drawn
    per produced intent. Non-param keys (action, ablate) are copied through.
    """
    action = seed["action"]
    fields = _VARY_FIELDS[action]
    choices: list[list] = []
    for field in fields:
        spec = seed[field]
        values = spec["vary"] if isinstance(spec, dict) and "vary" in spec else [spec]
        choices.append(values)

    intents: list[dict] = []
    for combo in itertools.product(*choices):
        intent = {"action": action}
        for field, raw_value in zip(fields, combo):
            intent[field] = _resolve_value(raw_value, rng)
        if "ablate" in seed:
            intent["ablate"] = list(seed["ablate"])
        intents.append(intent)
    return intents


def gold_calls(intent: dict) -> list[dict]:
    """Compute the gold expected_calls for a fully-specified concrete intent."""
    if intent["action"] == "transfer":
        recipient = resolve_recipient(intent["recipient"])
        if recipient is None:
            raise ValueError(f"unresolved recipient: {intent['recipient']!r}")
        return [build_transfer_call(intent["amount"], intent["token"], recipient)]
    if intent["action"] == "swap":
        return [build_swap_call(intent["amount"], intent["from_token"], intent["to_token"])]
    raise ValueError(f"unknown action: {intent['action']!r}")


def _protocol(action: str) -> str:
    return "uniswap" if action == "swap" else "transfer"


def _difficulty(action: str) -> str:
    # Mirrors the converter: swaps are medium, english transfers easy.
    return "medium" if action == "swap" else "easy"


def _base_metadata(intent: dict, kind: str, idx: int, **extra) -> dict:
    action = intent["action"]
    md = {
        "id": f"gen-{action}-{kind}-{idx:04d}",
        "source": "generated",
        "language": "english",
        "category": f"generated-{action}-{kind}",
        "protocol": _protocol(action),
        "difficulty": _difficulty(action),
        "query_type": None,
        "requires": [],
        "expected_calls": [],
        "notes": None,
    }
    md.update(extra)
    return md


def build_positive_case(intent: dict, template: str, rng: random.Random, idx: int,
                        style: str = "direct") -> dict:
    """A fully-specified intent rendered to a noisy surface; gold = computed call.

    `style` records whether the template is a direct command or narrative/indirect
    prose, so eval results can be sliced by phrasing style.
    """
    surface, labels = apply_mutators(render_surface(template, intent), rng)
    md = _base_metadata(intent, "pos", idx,
                        level="payload", query_type="one_shot", style=style,
                        mutators=labels, expected_calls=gold_calls(intent))
    return {"vars": {"user_message": surface,
                     "expected_summary": format_expected_summary(md["expected_calls"])},
            "metadata": md}


# Partial templates that OMIT one field (keyed by (action, missing_field)).
ABLATION_TEMPLATES: dict[tuple[str, str], str] = {
    ("transfer", "recipient"): "Send {amount} {token}",
    ("transfer", "amount"): "Send some {token} to {recipient}",
    ("transfer", "token"): "Send {amount} to {recipient}",
    ("swap", "to_token"): "Swap {amount} {from_token}",
    ("swap", "from_token"): "Swap {amount} for {to_token}",
    ("swap", "amount"): "Swap {from_token} for {to_token}",
}

# Canned assistant clarification for the missing field (multi-turn turn 2).
CLARIFICATIONS: dict[str, str] = {
    "recipient": "Which address or ENS should I send it to?",
    "amount": "How much would you like to send?",
    "token": "Which token?",
    "to_token": "Which token do you want to receive?",
    "from_token": "Which token do you want to swap from?",
}

# Completing user fragment that supplies the missing field (multi-turn turn 3).
COMPLETIONS: dict[tuple[str, str], str] = {
    ("transfer", "recipient"): "to {recipient}",
    ("transfer", "amount"): "{amount} {token}",
    ("transfer", "token"): "in {token}",
    ("swap", "to_token"): "for {to_token}",
    ("swap", "from_token"): "from {from_token}",
    ("swap", "amount"): "{amount}",
}

# Narrative (verbose, indirect) partial templates for multi-turn turn 1: each
# wraps the request in prose AND omits one field, which the user supplies in
# turn 3. Used only by the narrative-style multi-turn cases.
NARRATIVE_ABLATION_TEMPLATES: dict[tuple[str, str], str] = {
    ("transfer", "recipient"): "I've been meaning to send {amount} {token} to someone — let's get it done.",
    ("transfer", "amount"): "I want to move some {token} over to {recipient} today.",
    ("transfer", "token"): "Please send {amount} to {recipient} out of my wallet.",
    ("swap", "to_token"): "I've got {amount} {from_token} just doing nothing — let's put it to work and swap it.",
    ("swap", "from_token"): "I'd really like to pick up {amount} worth of {to_token}.",
    ("swap", "amount"): "I'm bullish on {to_token}, so let's convert some of my {from_token} into it.",
}

_ABLATION_BANKS = {"direct": ABLATION_TEMPLATES, "narrative": NARRATIVE_ABLATION_TEMPLATES}


def build_negative_case(intent: dict, field: str, rng: random.Random, idx: int) -> dict:
    """Drop `field` from the surface; expect no tool call (model should ask)."""
    template = ABLATION_TEMPLATES[(intent["action"], field)]
    surface, labels = apply_mutators(render_surface(template, intent), rng)
    md = _base_metadata(intent, "neg", idx, style="direct",
                        level="intent", category=f"ablation-{field}", mutators=labels)
    return {"vars": {"user_message": surface,
                     "expected_summary": format_expected_summary(md["expected_calls"])},
            "metadata": md}


def build_multiturn_case(intent: dict, field: str, rng: random.Random, idx: int,
                         style: str = "direct") -> dict:
    """Scripted convo: ablated turn 1, canned clarification, completing turn 3.

    Gold = the full computed call. Only the model's final response is scored.
    `style` picks the direct or narrative partial template for turn 1.
    """
    ablated_template = _ABLATION_BANKS[style][(intent["action"], field)]
    turn1, labels = apply_mutators(render_surface(ablated_template, intent), rng)
    completion = render_surface(COMPLETIONS[(intent["action"], field)], intent)
    messages = [
        {"role": "user", "content": turn1},
        {"role": "assistant", "content": CLARIFICATIONS[field]},
        {"role": "user", "content": completion},
    ]
    md = _base_metadata(intent, "mt", idx,
                        level="payload", query_type="multi_turn", style=style,
                        category=f"multiturn-{field}", mutators=labels,
                        expected_calls=gold_calls(intent))
    return {"vars": {"messages": messages,
                     "expected_summary": format_expected_summary(md["expected_calls"])},
            "metadata": md}
