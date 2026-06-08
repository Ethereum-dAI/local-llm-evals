"""Bridge between the promptfoo-native tests file (the single source of truth)
and our Python `Case` model.

`pf/tests.yaml` holds each case as a promptfoo test: `vars.user_message` is the
input and `metadata` carries the gold (`expected_calls`) plus the slice fields.
This module rebuilds a `Case` from that structure so the scorer can be reused
both inside the promptfoo python assertion and in the offline integrity test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from wallet_evals.schema import Case


def case_from_metadata(metadata: dict[str, Any], user_message: str = "") -> Case:
    """Rebuild a Case from a promptfoo test's metadata + user_message."""
    return Case(user_message=user_message, **metadata)


def _user_message_from_vars(vars_: dict[str, Any]) -> str:
    """The prompt input: explicit user_message, else the last user turn."""
    if "user_message" in vars_:
        return vars_["user_message"]
    messages = vars_.get("messages", [])
    user_turns = [m["content"] for m in messages if m.get("role") == "user"]
    return user_turns[-1] if user_turns else ""


def load_cases(path: str | Path) -> list[Case]:
    """Load all cases from a promptfoo-native tests YAML file."""
    import yaml

    tests = yaml.safe_load(Path(path).read_text()) or []
    return [
        case_from_metadata(test["metadata"], _user_message_from_vars(test["vars"]))
        for test in tests
    ]
