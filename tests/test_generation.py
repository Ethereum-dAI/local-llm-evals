import random

from wallet_evals.generation import random_address


def test_random_address_shape():
    addr = random_address(random.Random(0))
    assert addr.startswith("0x")
    assert len(addr) == 42
    int(addr, 16)  # all hex


def test_random_address_deterministic():
    assert random_address(random.Random(7)) == random_address(random.Random(7))


def test_random_address_varies_with_state():
    rng = random.Random(7)
    assert random_address(rng) != random_address(rng)  # advancing state changes output


from wallet_evals.generation import (
    mutate_case, mutate_punctuation, mutate_typos, mutate_filler,
    mutate_tone, apply_mutators, MUTATORS,
)


def test_each_mutator_is_pure_for_a_seed():
    for _, fn in MUTATORS:
        a = fn("Send 0.1 ETH to vitalik.eth", random.Random(3))
        b = fn("Send 0.1 ETH to vitalik.eth", random.Random(3))
        assert a == b
        assert isinstance(a, str) and a  # non-empty string


def test_mutate_case_changes_letter_case():
    out = mutate_case("Send ETH", random.Random(1))
    assert out.lower() == "send eth"  # only case differs


def test_apply_mutators_returns_text_and_labels():
    text, labels = apply_mutators("Send 0.1 ETH to vitalik.eth", random.Random(5))
    assert isinstance(text, str) and text
    assert isinstance(labels, list)
    assert all(label in {name for name, _ in MUTATORS} for label in labels)


def test_apply_mutators_deterministic():
    assert apply_mutators("hello world", random.Random(9)) == \
           apply_mutators("hello world", random.Random(9))


from wallet_evals.generation import (
    TRANSFER_TEMPLATES, SWAP_TEMPLATES, render_surface,
)


def test_transfer_render():
    intent = {"action": "transfer", "amount": "0.1", "token": "ETH",
              "recipient": "vitalik.eth"}
    out = render_surface("Send {amount} {token} to {recipient}", intent)
    assert out == "Send 0.1 ETH to vitalik.eth"


def test_swap_render():
    intent = {"action": "swap", "amount": "100", "from_token": "USDC",
              "to_token": "ETH"}
    out = render_surface("Swap {amount} {from_token} for {to_token}", intent)
    assert out == "Swap 100 USDC for ETH"


def test_swap_wrong_verb_template_present():
    assert any("become" in t for t in SWAP_TEMPLATES)  # Marcello's hard phrasing


def test_template_banks_nonempty():
    assert len(TRANSFER_TEMPLATES) >= 4 and len(SWAP_TEMPLATES) >= 4


from wallet_evals.generation import expand_vary


def test_expand_literal_seed_is_single_intent():
    seed = {"action": "transfer", "amount": "0.1", "token": "ETH",
            "recipient": "vitalik.eth", "ablate": ["recipient"]}
    out = expand_vary(seed, random.Random(0))
    assert len(out) == 1
    assert out[0]["amount"] == "0.1"
    assert out[0]["ablate"] == ["recipient"]


def test_expand_vary_cross_product():
    seed = {"action": "transfer",
            "amount": {"vary": ["0.1", "1000"]},
            "token": {"vary": ["ETH", "USDC"]},
            "recipient": "vitalik.eth"}
    out = expand_vary(seed, random.Random(0))
    assert len(out) == 4
    amounts = sorted({i["amount"] for i in out})
    assert amounts == ["0.1", "1000"]


def test_expand_random_address_resolves_to_hex():
    seed = {"action": "transfer", "amount": "0.1", "token": "ETH",
            "recipient": {"vary": ["random_address"]}}
    out = expand_vary(seed, random.Random(0))
    assert out[0]["recipient"].startswith("0x") and len(out[0]["recipient"]) == 42


def test_expand_vary_deterministic():
    seed = {"action": "transfer", "amount": "0.1", "token": "ETH",
            "recipient": {"vary": ["random_address", "vitalik.eth"]}}
    a = expand_vary(seed, random.Random(4))
    b = expand_vary(seed, random.Random(4))
    assert a == b


from wallet_evals.generation import gold_calls, build_positive_case


def test_gold_calls_transfer_resolves_ens():
    intent = {"action": "transfer", "amount": "0.1", "token": "ETH",
              "recipient": "vitalik.eth"}
    calls = gold_calls(intent)
    assert calls[0]["to"] == "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    assert calls[0]["value"] == "100000000000000000"


def test_gold_calls_swap():
    intent = {"action": "swap", "amount": "100", "from_token": "USDC",
              "to_token": "ETH"}
    calls = gold_calls(intent)
    assert calls[0]["tool"] == "swap"
    assert calls[0]["amountIn"] == "100000000"


def test_build_positive_case_structure():
    intent = {"action": "transfer", "amount": "0.1", "token": "ETH",
              "recipient": "vitalik.eth"}
    case = build_positive_case(intent, "Send {amount} {token} to {recipient}",
                               random.Random(0), idx=1)
    assert "vars" in case and "metadata" in case
    assert "user_message" in case["vars"]
    md = case["metadata"]
    assert md["id"] == "gen-transfer-pos-0001"
    assert md["level"] == "payload"
    assert md["protocol"] == "transfer"
    assert md["source"] == "generated"
    assert md["expected_calls"][0]["value"] == "100000000000000000"
