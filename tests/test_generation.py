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
