"""Pure, deterministic helpers for generating eval cases from seed intents.

Everything here is a pure function of its inputs (including an explicit
`random.Random` for any sampling), so a fixed seed yields byte-identical output.
Gold is computed from the structured intent via wallet_evals.intents, never
parsed from the rendered surface — so surface mutation is always safe.
"""
from __future__ import annotations

import random


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
