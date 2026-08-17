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


def test_gold_calls_transfer_passes_recipient_through_unresolved():
    intent = {"action": "transfer", "amount": "0.1", "token": "ETH",
              "recipient": "vitalik.eth"}
    calls = gold_calls(intent)
    assert calls[0]["to"] == "vitalik.eth"
    assert calls[0]["amount"] == "0.1"
    assert calls[0]["token"] == "ETH"


def test_gold_calls_swap():
    intent = {"action": "swap", "amount": "100", "from_token": "USDC",
              "to_token": "ETH"}
    calls = gold_calls(intent)
    assert calls[0]["tool"] == "swap"
    assert calls[0]["amount"] == "100"
    assert calls[0]["from_token"] == "USDC" and calls[0]["to_token"] == "ETH"


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
    assert md["expected_calls"][0]["amount"] == "0.1"


from wallet_evals.generation import (
    ABLATION_TEMPLATES, CLARIFICATIONS, COMPLETIONS, build_negative_case,
)


def test_ablation_maps_cover_seed_fields():
    for key in [("transfer", "recipient"), ("transfer", "amount"),
                ("transfer", "token"), ("swap", "to_token"),
                ("swap", "from_token"), ("swap", "amount")]:
        assert key in ABLATION_TEMPLATES
        assert key in COMPLETIONS
    for field in ["recipient", "amount", "token", "to_token", "from_token"]:
        assert field in CLARIFICATIONS


def test_build_negative_case_omits_field_and_expects_no_call():
    intent = {"action": "transfer", "amount": "0.1", "token": "ETH",
              "recipient": "vitalik.eth"}
    case = build_negative_case(intent, "recipient", random.Random(0), idx=2)
    md = case["metadata"]
    assert md["id"] == "gen-transfer-neg-0002"
    assert md["level"] == "intent"
    assert md["expected_calls"] == []
    assert "vitalik.eth" not in case["vars"]["user_message"]  # recipient dropped


from wallet_evals.generation import build_multiturn_case


def test_build_multiturn_case_three_turns_and_full_gold():
    intent = {"action": "transfer", "amount": "0.1", "token": "ETH",
              "recipient": "vitalik.eth"}
    case = build_multiturn_case(intent, "recipient", random.Random(0), idx=3)
    msgs = case["vars"]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == "Which address or ENS should I send it to?"
    assert "vitalik.eth" in msgs[2]["content"]
    md = case["metadata"]
    assert md["id"] == "gen-transfer-mt-0003"
    assert md["level"] == "payload"
    assert md["expected_calls"][0]["amount"] == "0.1"  # full gold


def test_multiturn_has_no_user_message_var():
    intent = {"action": "swap", "amount": "100", "from_token": "USDC",
              "to_token": "ETH"}
    case = build_multiturn_case(intent, "to_token", random.Random(0), idx=1)
    assert "messages" in case["vars"] and "user_message" not in case["vars"]


def test_build_all_is_deterministic():
    import yaml
    from scripts.generate_cases import build_all, SEEDS

    seeds = yaml.safe_load(SEEDS.read_text())
    a = build_all(seeds, random.Random(123))
    b = build_all(seeds, random.Random(123))
    assert a == b


def test_build_all_drops_self_swaps():
    import yaml
    from scripts.generate_cases import build_all, SEEDS

    seeds = yaml.safe_load(SEEDS.read_text())
    cases = build_all(seeds, random.Random(1)).get("swap", [])
    for case in cases:
        calls = case["metadata"]["expected_calls"]
        for call in calls:
            assert call.get("from_token") != call.get("to_token")


def test_load_cases_supports_messages(tmp_path):
    import yaml
    from wallet_evals.promptfoo import load_cases

    doc = [{
        "vars": {"messages": [
            {"role": "user", "content": "Send 0.1 ETH"},
            {"role": "assistant", "content": "Which address?"},
            {"role": "user", "content": "to vitalik.eth"},
        ]},
        "metadata": {
            "id": "gen-transfer-mt-0001", "level": "payload", "language": "english",
            "category": "multiturn-recipient", "protocol": "transfer",
            "difficulty": "easy", "expected_calls": [],
        },
    }]
    p = tmp_path / "mt.yaml"
    p.write_text(yaml.safe_dump(doc))
    cases = load_cases(p)
    assert len(cases) == 1
    assert cases[0].user_message == "to vitalik.eth"  # last user turn


def test_mutate_punctuation_preserves_hex_addresses():
    addr = "0x" + "1234567890" * 4  # 40 all-digit hex chars -> would be comma-mangled
    out = mutate_punctuation(f"Send 1000 ETH to {addr}", random.Random(0))
    assert addr in out          # address left intact
    assert "1,000" in out       # amount still gets thousands-commas


def test_mutate_punctuation_preserves_decimal_fraction():
    # Integer part is grouped; fractional digits are never split by a comma.
    out = mutate_punctuation("send 12.3456 ETH and 1234.56 USDC", random.Random(0))
    assert "12.3456" in out          # short int part, fraction intact
    assert "1,234.56" in out         # int part grouped, fraction intact
    assert "12.3,456" not in out     # the bug we fixed


def test_mutate_typos_never_corrupts_token_symbols():
    # Token symbols (WETH/USDC) must survive typo mutation intact; corrupting them
    # would make the request ambiguous and wrongly reward reckless models.
    for _ in range(50):
        out = mutate_typos("Swap 5 WETH for USDC please", random.Random(_))
        assert "WETH" in out and "USDC" in out


def test_narrative_banks_present_and_use_placeholders():
    from wallet_evals.generation import (
        TRANSFER_NARRATIVE_TEMPLATES, SWAP_NARRATIVE_TEMPLATES,
    )
    assert len(TRANSFER_NARRATIVE_TEMPLATES) >= 3
    assert len(SWAP_NARRATIVE_TEMPLATES) >= 3
    # The colleague's motivating example is reproducible from a narrative template.
    intent = {"action": "swap", "amount": "5", "from_token": "ETH", "to_token": "DAI"}
    rendered = [render_surface(t, intent) for t in SWAP_NARRATIVE_TEMPLATES]
    assert any("DAI is doing great" in r and "5 of my ETH" in r for r in rendered)


def test_positive_case_records_style():
    intent = {"action": "transfer", "amount": "0.1", "token": "ETH",
              "recipient": "vitalik.eth"}
    direct = build_positive_case(intent, "Send {amount} {token} to {recipient}",
                                 random.Random(0), idx=1)
    assert direct["metadata"]["style"] == "direct"
    from wallet_evals.generation import SWAP_NARRATIVE_TEMPLATES
    sw = {"action": "swap", "amount": "5", "from_token": "ETH", "to_token": "DAI"}
    narr = build_positive_case(sw, SWAP_NARRATIVE_TEMPLATES[0], random.Random(0),
                               idx=2, style="narrative")
    assert narr["metadata"]["style"] == "narrative"
    assert narr["metadata"]["expected_calls"][0]["tool"] == "swap"


def test_narrative_multiturn_turn1_omits_field():
    sw = {"action": "swap", "amount": "5", "from_token": "ETH", "to_token": "DAI"}
    case = build_multiturn_case(sw, "to_token", random.Random(0), idx=1, style="narrative")
    msgs = case["vars"]["messages"]
    assert case["metadata"]["style"] == "narrative"
    assert "DAI" not in msgs[0]["content"]      # to_token omitted in turn 1
    assert "DAI" in msgs[2]["content"]          # supplied in the completing turn


from wallet_evals.generation import group_amount, group_integer_part, build_separator_case


def test_group_integer_part_below_threshold_unchanged():
    assert group_integer_part("123") == "123"
    assert group_integer_part("1") == "1"


def test_group_integer_part_groups_by_threes():
    assert group_integer_part("1000") == "1,000"
    assert group_integer_part("1234567") == "1,234,567"


def test_group_amount_only_groups_integer_part():
    assert group_amount("20000") == "20,000"
    assert group_amount("1234.5") == "1,234.5"
    assert group_amount("12.3456") == "12.3456"  # short int part, fraction untouched
    assert group_amount("100") == "100"          # <=3 digits, untouched


def test_mutate_punctuation_still_matches_group_amount():
    """The refactor extracted mutate_punctuation's grouping into group_amount —
    assert the two agree on a plain amount (mutate_punctuation may still append
    trailing punctuation, so compare the amount-bearing prefix)."""
    out = mutate_punctuation("987654.32", random.Random(0))
    assert out.startswith(group_amount("987654.32"))


def test_build_separator_case_surface_carries_commas_gold_stays_plain():
    intent = {"action": "transfer", "amount": "20000", "token": "ETH",
              "recipient": "vitalik.eth"}
    case = build_separator_case(intent, "Send {amount} {token} to {recipient}",
                                random.Random(0), idx=1)
    assert "20,000" in case["vars"]["user_message"]
    assert case["metadata"]["expected_calls"][0]["amount"] == "20000"
    assert case["metadata"]["id"] == "gen-transfer-sep-0001"
    assert case["metadata"]["category"] == "generated-transfer-sep"


def test_build_separator_case_never_regroups_via_punctuation_mutator():
    """Punctuation is excluded from the mutator draw so it never fights the
    deterministic grouping already applied to the amount."""
    for seed in range(20):
        case = build_separator_case(
            {"action": "swap", "amount": "135790.24", "from_token": "USDC",
             "to_token": "ETH"},
            "Swap {amount} {from_token} for {to_token}", random.Random(seed), idx=1)
        assert "punctuation" not in case["metadata"]["mutators"]
        assert "135,790.24" in case["vars"]["user_message"]


def test_build_separator_case_deterministic():
    intent = {"action": "transfer", "amount": "654321.098765", "token": "USDC",
              "recipient": "vitalik.eth"}
    a = build_separator_case(intent, "Send {amount} {token} to {recipient}",
                             random.Random(3), idx=5)
    b = build_separator_case(intent, "Send {amount} {token} to {recipient}",
                             random.Random(3), idx=5)
    assert a == b
