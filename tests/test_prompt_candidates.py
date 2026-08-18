"""Guards for pf/prompt_candidates.py — the A/B-only prompt variants.

These encode lessons that were paid for, so a future edit cannot quietly undo them.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    path = ROOT / "pf" / "prompt_candidates.py"
    spec = importlib.util.spec_from_file_location("pc", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ZERO_LITERAL = "0x0000000000000000000000000000000000000000"


def test_every_variant_is_registered_and_nonempty():
    m = _mod()
    assert m.PROMPT_CANDIDATES
    for name, parts in m.PROMPT_CANDIDATES.items():
        assert parts and all(p.strip() for p in parts), name


def test_augment_is_a_noop_for_the_control_arm():
    """The A arm of an A/B must go through the identical code path, or the comparison
    includes the code path as a variable."""
    m = _mod()
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    assert m.augment(msgs, "none") == msgs
    assert m.augment(msgs, "") == msgs


def test_augment_only_touches_the_system_turn():
    m = _mod()
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]
    out = m.augment(msgs, "safety")
    assert out[1] == msgs[1], "user turns must be untouched"
    assert out[0]["content"].startswith("S")
    assert len(out[0]["content"]) > len("S")


def test_unknown_variant_fails_loudly():
    """A typo'd variant name must not silently score the control prompt under a
    variant label."""
    m = _mod()
    with pytest.raises(SystemExit):
        m.augment([{"role": "system", "content": "S"}], "safetyy")


def test_augment_refuses_a_conversation_with_no_system_turn():
    m = _mod()
    with pytest.raises(SystemExit):
        m.augment([{"role": "user", "content": "u"}], "safety")


def test_safety_variants_never_colocate_swap_and_the_zero_address():
    """A PAID-FOR lesson. The sentence that once fixed base's zero-address refusals
    named `swap` and `0x0` together, and the swap-heavy fine-tune then emitted
    `swap` with `currencyIn=0x0` for plain requests. Keep them in separate sentences.
    """
    m = _mod()
    for name in ("safety", "safety-min"):
        text = " ".join(m.PROMPT_CANDIDATES[name])
        for sentence in re.split(r"(?<=[.;])\s+|\n", text):
            has_zero = ZERO_LITERAL in sentence or "zero address" in sentence.lower()
            has_swap = "swap" in sentence.lower()
            assert not (has_zero and has_swap), \
                f"{name}: sentence names both swap and the zero address: {sentence!r}"


def test_safety_variants_carry_no_mechanical_address_heuristic():
    """Also rejected before: "refuse if `to` starts with 4+ zeros" passes the cases
    and ships a false positive, because real addresses can begin with zeros. The rule
    must be stated by REASON, not by string shape."""
    m = _mod()
    banned = ("starts with", "begins with", "leading zero", "first four", "prefix")
    for name in ("safety", "safety-min"):
        text = " ".join(m.PROMPT_CANDIDATES[name]).lower()
        for phrase in banned:
            assert phrase not in text, f"{name}: mechanical heuristic {phrase!r}"


def test_safety_variants_still_permit_ordinary_transfers():
    """A safety clause that does not carve out the normal case buys refusals with
    false refusals — which the accuracy slice would catch, but stating it here makes
    the intent explicit."""
    m = _mod()
    for name in ("safety", "safety-min"):
        text = " ".join(m.PROMPT_CANDIDATES[name]).lower()
        assert "ordinary address or ens name is fine" in text, name


def test_safety_full_keeps_the_known_token_address_carve_out():
    """The app accepts a KNOWN token given as its 0x address (the token_address
    mechanism, whose gold carries it verbatim). A blanket "refuse raw addresses" rule
    would break a documented capability to win safety cases."""
    m = _mod()
    text = " ".join(m.PROMPT_CANDIDATES["safety"]).lower()
    assert "known token given as its address is fine" in text
