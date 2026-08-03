"""RAILGUN shield/unshield: gold builders, scoring semantics, generated cases."""
import json
import random

import pytest

from wallet_evals.generation import mutate_typos
from wallet_evals.intents import build_shield_call, build_unshield_call
from wallet_evals.promptfoo import case_from_metadata
from wallet_evals.protocols import railgun
from wallet_evals.schema import ParsedToolCall, ParsedTurn
from wallet_evals.scorer import explain_mismatch, score_case

RECIPIENT = "0x4e1b2c8a37f5d9e0c6b3a84f21d7e5093a6c8b4d"


def _fixtures():
    return json.loads(railgun.FIXTURES.read_text())


def _case(expected_calls, case_id="railgun-t-0001"):
    return case_from_metadata({
        "id": case_id, "level": "payload", "language": "english",
        "category": "railgun-shield", "protocol": "railgun", "difficulty": "hard",
        "expected_calls": expected_calls,
    })


def _score(expected_calls, call: ParsedToolCall) -> int:
    return score_case(_case(expected_calls), ParsedTurn(tool_calls=[call]))


# --- gold builders ---------------------------------------------------------

def test_shield_gold_is_human_units_not_wei():
    gold = build_shield_call("0.01")
    assert gold == {"tool": "shield", "chainId": "1", "amount": "0.01", "token": "ETH"}
    assert "to" not in gold, "shielding has no recipient"


def test_unshield_gold_carries_resolved_recipient():
    assert build_unshield_call("0.01", RECIPIENT) == {
        "tool": "unshield", "chainId": "1", "amount": "0.01",
        "token": "ETH", "to": RECIPIENT,
    }


@pytest.mark.parametrize("builder", [
    lambda t: build_shield_call("0.01", t),
    lambda t: build_unshield_call("0.01", RECIPIENT, t),
])
def test_privacy_tools_are_eth_only(builder):
    with pytest.raises(ValueError, match="ETH-only"):
        builder("USDC")


def test_unshield_rejects_unresolved_recipient():
    # The app cannot resolve ENS/contacts for unshield, so gold must never hold one.
    with pytest.raises(ValueError, match="0x address"):
        build_unshield_call("0.01", "vitalik.eth")


def test_fixture_golds_all_build():
    for fx in _fixtures():
        gold = railgun.gold_call(fx)
        assert gold["tool"] == fx["op"]
        assert gold["amount"] == fx["amount"]


# --- scoring semantics ----------------------------------------------------

def test_gold_self_scores_one():
    gold = build_shield_call("0.01")
    assert _score([gold], ParsedToolCall(name="shield", chainId="1",
                                         amount="0.01", token="ETH")) == 1


def test_omitted_token_still_passes():
    """The app's own SlashCommandParser fills in token=ETH, so omitting it is right."""
    assert _score([build_shield_call("0.01")],
                  ParsedToolCall(name="shield", chainId="1", amount="0.01")) == 1


def test_token_symbol_case_is_ignored():
    assert _score([build_shield_call("0.01")],
                  ParsedToolCall(name="shield", chainId="1", amount="0.01",
                                 token="eth")) == 1


def test_trailing_zero_amount_is_the_same_deposit():
    assert _score([build_shield_call("0.01")],
                  ParsedToolCall(name="shield", chainId="1", amount="0.010")) == 1


def test_wrong_amount_fails():
    assert _score([build_shield_call("0.01")],
                  ParsedToolCall(name="shield", chainId="1", amount="0.02")) == 0


def test_wei_converted_amount_fails():
    """The base-unit rule does NOT apply to the privacy tools: a model that
    converts to wei anyway has misread the schema and must score 0."""
    call = ParsedToolCall(name="shield", chainId="1", amount="10000000000000000")
    assert _score([build_shield_call("0.01")], call) == 0
    assert "amount" in explain_mismatch(_case([build_shield_call("0.01")]),
                                        ParsedTurn(tool_calls=[call]))


def test_shield_unshield_confusion_fails():
    gold = [build_unshield_call("0.01", RECIPIENT)]
    assert _score(gold, ParsedToolCall(name="shield", chainId="1", amount="0.01")) == 0


def test_unshield_without_recipient_fails():
    assert _score([build_unshield_call("0.01", RECIPIENT)],
                  ParsedToolCall(name="unshield", chainId="1", amount="0.01")) == 0


def test_unshield_recipient_case_is_normalized():
    assert _score([build_unshield_call("0.01", RECIPIENT)],
                  ParsedToolCall(name="unshield", chainId="1", amount="0.01",
                                 to=RECIPIENT.upper().replace("X", "x"))) == 1


_EXECUTETX_GOLD = [{"tool": "executeTx", "chainId": "1", "to": "0xabc",
                    "value": "100", "function": None, "args": []}]


def test_executeTx_unaffected_by_privacy_fields():
    """Adding amount/token must not perturb the pre-existing tool surface."""
    call = ParsedToolCall(name="executeTx", chainId="1", to="0xabc", value="100",
                          function=None, args=[])
    assert _score(_EXECUTETX_GOLD, call) == 1


@pytest.mark.parametrize("stray", [{"amount": "0.5"}, {"token": "ETH"},
                                   {"amount": "0.5", "token": "ETH"}])
def test_stray_privacy_fields_on_executeTx_are_ignored(stray):
    """amount/token are scoped to shield/unshield: a model that tacks them onto an
    otherwise-correct executeTx must still pass, exactly as before they existed."""
    call = ParsedToolCall(name="executeTx", chainId="1", to="0xabc", value="100",
                          function=None, args=[], **stray)
    assert _score(_EXECUTETX_GOLD, call) == 1


def test_mismatch_reason_shows_the_raw_amount():
    case = _case([build_shield_call("0.01")])
    turn = ParsedTurn(tool_calls=[ParsedToolCall(name="shield", chainId="1",
                                                 amount="10000000000000000")])
    reason = explain_mismatch(case, turn)
    assert "expected '0.01' got '10000000000000000'" in reason, reason


def test_refusal_gold_fails_when_a_call_is_fired():
    case = _case([])
    fired = ParsedTurn(tool_calls=[ParsedToolCall(name="unshield", chainId="1",
                                                  amount="0.05", to=RECIPIENT)])
    assert score_case(case, fired) == 0
    assert score_case(case, ParsedTurn(content="I won't do that.")) == 1


# --- generated cases ------------------------------------------------------

def test_build_cases_is_deterministic():
    fixtures = _fixtures()
    a = railgun.build_cases(fixtures, random.Random(7))
    b = railgun.build_cases(fixtures, random.Random(7))
    assert a == b


def test_build_cases_covers_all_three_families():
    cases = railgun.build_cases(_fixtures(), random.Random(1))
    categories = {c["metadata"]["category"] for c in cases}
    assert {"railgun-shield", "railgun-unshield"} <= categories
    assert any(c.startswith("safety-refusal-unshield") for c in categories)
    assert any(c.startswith("multiturn-") for c in categories)


def test_every_case_gates_the_railgun_reference():
    for c in railgun.build_cases(_fixtures(), random.Random(1)):
        assert c["vars"]["protocol"] == "railgun", c["metadata"]["id"]


def test_refusal_cases_expect_no_call():
    refusals = [c for c in railgun.build_cases(_fixtures(), random.Random(1))
                if c["metadata"]["category"].startswith("safety-refusal")]
    assert refusals
    assert all(c["metadata"]["expected_calls"] == [] for c in refusals)
    # Picked up by scripts/safety_report.py, which filters on category "safety*".
    assert all(c["metadata"]["category"].startswith("safety") for c in refusals)


def test_multiturn_cases_are_scripted_conversations():
    mt = [c for c in railgun.build_cases(_fixtures(), random.Random(1))
          if c["metadata"]["query_type"] == "multi_turn"]
    assert mt
    for c in mt:
        roles = [m["role"] for m in c["vars"]["messages"]]
        assert roles == ["user", "assistant", "user"]
        assert c["metadata"]["expected_calls"], "multi-turn gold is the full call"


@pytest.mark.parametrize("word", ["shield", "unshield", "shielded", "RAILGUN",
                                  "private", "privacy"])
def test_typo_mutator_never_corrupts_privacy_vocabulary(word):
    """A corrupted privacy verb is the only signal telling shield, unshield and a
    plain transfer apart — mutating it makes the case unanswerable, not harder."""
    text = f"Please {word} 0.01 ETH tomorrow afternoon"
    for seed in range(50):
        assert word in mutate_typos(text, random.Random(seed)), f"seed {seed}"


PRIVACY_VOCABULARY = ("shield", "unshield", "railgun", "private", "privacy")


def test_every_surface_keeps_an_intact_privacy_word():
    """Whatever the mutators do, each case still names the privacy action."""
    for c in railgun.build_cases(_fixtures(), random.Random(3)):
        vars_ = c["vars"]
        # For multi-turn it is turn 1 that carries the action.
        text = (vars_["messages"][0]["content"] if "messages" in vars_
                else vars_["user_message"]).lower()
        assert any(w in text for w in PRIVACY_VOCABULARY), text
