"""ModelAdapter protocol: one interface, swappable backends."""
from __future__ import annotations

from typing import Protocol

from wallet_evals.schema import ParsedTurn


class ModelAdapter(Protocol):
    model: str

    def run(self, user_message: str) -> ParsedTurn:
        """Send one one-shot request and return the normalized turn."""
        ...
