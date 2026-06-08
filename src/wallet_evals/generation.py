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
