"""Pure, deterministic helpers for generating eval cases from seed intents.

Everything here is a pure function of its inputs (including an explicit
`random.Random` for any sampling), so a fixed seed yields byte-identical output.
Gold is computed from the structured intent via wallet_evals.intents, never
parsed from the rendered surface — so surface mutation is always safe.
"""
from __future__ import annotations

import itertools
import random

from wallet_evals.intents import (
    resolve_recipient, build_transfer_call, build_swap_call,
)


def random_address(rng: random.Random) -> str:
    """A realistic-looking lowercase 20-byte hex address from `rng`."""
    return "0x" + "".join(rng.choice("0123456789abcdef") for _ in range(40))


def mutate_case(text: str, rng: random.Random) -> str:
    """Random per-character upper/lower casing (preserves spelling)."""
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text)


def mutate_punctuation(text: str, rng: random.Random) -> str:
    """Add commas into long digit runs and noisy trailing punctuation."""
    import re

    def comma(m: "re.Match[str]") -> str:
        digits = m.group(0)
        if len(digits) <= 3:
            return digits
        rev = digits[::-1]
        grouped = ",".join(rev[i:i + 3] for i in range(0, len(rev), 3))
        return grouped[::-1]

    out = re.sub(r"\d+", comma, text)
    return out + rng.choice(["", "!", "!!", "...", " please"])


def mutate_typos(text: str, rng: random.Random) -> str:
    """Swap two adjacent letters in one all-alphabetic word (never digits/hex)."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 3]
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


MUTATORS: list[tuple[str, "callable"]] = [
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


def build_positive_case(intent: dict, template: str, rng: random.Random, idx: int) -> dict:
    """A fully-specified intent rendered to a noisy surface; gold = computed call."""
    surface, labels = apply_mutators(render_surface(template, intent), rng)
    md = _base_metadata(intent, "pos", idx,
                        level="payload", query_type="one_shot",
                        mutators=labels, expected_calls=gold_calls(intent))
    return {"vars": {"user_message": surface}, "metadata": md}


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


def build_negative_case(intent: dict, field: str, rng: random.Random, idx: int) -> dict:
    """Drop `field` from the surface; expect no tool call (model should ask)."""
    template = ABLATION_TEMPLATES[(intent["action"], field)]
    surface, labels = apply_mutators(render_surface(template, intent), rng)
    md = _base_metadata(intent, "neg", idx,
                        level="intent", category=f"ablation-{field}", mutators=labels)
    return {"vars": {"user_message": surface}, "metadata": md}


def build_multiturn_case(intent: dict, field: str, rng: random.Random, idx: int) -> dict:
    """Scripted convo: ablated turn 1, canned clarification, completing turn 3.

    Gold = the full computed call. Only the model's final response is scored.
    """
    ablated_template = ABLATION_TEMPLATES[(intent["action"], field)]
    turn1, labels = apply_mutators(render_surface(ablated_template, intent), rng)
    completion = render_surface(COMPLETIONS[(intent["action"], field)], intent)
    messages = [
        {"role": "user", "content": turn1},
        {"role": "assistant", "content": CLARIFICATIONS[field]},
        {"role": "user", "content": completion},
    ]
    md = _base_metadata(intent, "mt", idx,
                        level="payload", query_type="multi_turn",
                        category=f"multiturn-{field}", mutators=labels,
                        expected_calls=gold_calls(intent))
    return {"vars": {"messages": messages}, "metadata": md}
