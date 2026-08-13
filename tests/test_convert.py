from scripts.convert_recognition import convert_case, LOOKUP
from wallet_evals.intents import to_base_units


def test_to_base_units_eth():
    assert to_base_units("0.1", 18) == "100000000000000000"


def test_to_base_units_usdc():
    assert to_base_units("100", 6) == "100000000"


def test_to_base_units_integer():
    assert to_base_units("5", 18) == "5000000000000000000"


def test_convert_null_tool_to_empty_calls():
    raw = {"id": "ambiguous-001", "user_message": "Send ETH to bob.eth", "category": "ambiguous",
           "language": "english", "expected_tool": None, "expected_args": None, "notes": None}
    case, manual = convert_case(raw)
    assert manual is None
    assert case["expected_calls"] == []
    assert case["level"] == "intent"


def test_convert_native_transfer():
    raw = {"id": "transfer-en-001", "user_message": "Send 0.1 ETH to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "0.1"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, manual = convert_case(raw)
    assert manual is None
    call = case["expected_calls"][0]
    assert call["tool"] == "transfer"
    assert call["chainId"] == "1"
    assert call["to"] == "vitalik.eth"
    assert call["amount"] == "0.1"
    assert call["token"] == "ETH"


def test_convert_erc20_transfer():
    raw = {"id": "transfer-en-002", "user_message": "Send 100 USDC to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "100"},
                             "token": {"kind": "exact", "value": "USDC"}}, "notes": None}
    case, manual = convert_case(raw)
    call = case["expected_calls"][0]
    assert call["tool"] == "transfer"
    assert call["chainId"] == "1"
    assert call["to"] == "vitalik.eth"
    assert call["amount"] == "100"
    assert call["token"] == "USDC"


def test_incomplete_swap_routed_to_manual():
    raw = {"id": "swap-en-001", "user_message": "Swap 100 USDC for ETH", "category": "truePositiveSwap",
           "language": "english", "expected_tool": "swap",
           "expected_args": {"from_token": {"kind": "exact", "value": "USDC"}}, "notes": None}
    case, manual = convert_case(raw)
    assert case is None
    assert manual == "swap-en-001"


def test_convert_swap_exact_in():
    raw = {"id": "swap-en-001", "user_message": "Swap 100 USDC for DAI", "category": "truePositiveSwap",
           "language": "english", "expected_tool": "swap",
           "expected_args": {"from_token": {"kind": "exact", "value": "USDC"},
                             "to_token": {"kind": "exact", "value": "DAI"},
                             "amount": {"kind": "exact", "value": "100"},
                             "amount_side": {"kind": "exact", "value": "input"}}, "notes": None}
    case, manual = convert_case(raw)
    assert manual is None
    assert case["protocol"] == "uniswap"
    call = case["expected_calls"][0]
    assert call["tool"] == "swap"
    assert call["chainId"] == "1"
    assert call["from_token"] == "USDC"
    assert call["to_token"] == "DAI"
    assert call["amount"] == "100"
    assert call["amount_side"] == "input"


def test_convert_swap_native_eth_uses_zero_address():
    raw = {"id": "swap-en-002", "user_message": "Swap 1 ETH for USDC", "category": "truePositiveSwap",
           "language": "english", "expected_tool": "swap",
           "expected_args": {"from_token": {"kind": "exact", "value": "ETH"},
                             "to_token": {"kind": "exact", "value": "USDC"},
                             "amount": {"kind": "exact", "value": "1"},
                             "amount_side": {"kind": "exact", "value": "input"}}, "notes": None}
    case, manual = convert_case(raw)
    call = case["expected_calls"][0]
    assert call["from_token"] == "ETH"
    assert call["amount"] == "1"


def test_convert_swap_exact_output_to_manual():
    raw = {"id": "swap-en-003", "user_message": "Buy 1 ETH with USDC", "category": "truePositiveSwap",
           "language": "english", "expected_tool": "swap",
           "expected_args": {"from_token": {"kind": "exact", "value": "USDC"},
                             "to_token": {"kind": "exact", "value": "ETH"},
                             "amount": {"kind": "exact", "value": "1"},
                             "amount_side": {"kind": "exact", "value": "output"}}, "notes": None}
    case, manual = convert_case(raw)
    assert case is None
    assert manual == "swap-en-003"


def test_amount_all_routed_to_manual():
    raw = {"id": "transfer-en-003", "user_message": "Send all my ETH to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "all"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, manual = convert_case(raw)
    assert case is None
    assert manual == "transfer-en-003"


def test_ens_recipient_flagged_as_ens_resolution():
    raw = {"id": "transfer-en-001", "user_message": "Send 0.1 ETH to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "0.1"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, _ = convert_case(raw)
    assert case["requires"] == ["ens_resolution"]


def test_erc20_to_ens_carries_both_flags():
    raw = {"id": "transfer-en-002", "user_message": "Send 100 USDC to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "100"},
                             "token": {"kind": "exact", "value": "USDC"}}, "notes": None}
    case, _ = convert_case(raw)
    assert case["requires"] == ["ens_resolution", "token_address_lookup"]


def test_raw_address_recipient_has_no_ens_flag():
    raw = {"id": "transfer-en-004", "user_message": "Send 1 ETH to 0x1111111111111111111111111111111111111111",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "0x1111111111111111111111111111111111111111"},
                             "amount": {"kind": "exact", "value": "1"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, _ = convert_case(raw)
    assert case["requires"] == []


def test_unknown_ens_accepted_as_unresolved():
    raw = {"id": "transfer-en-005", "user_message": "Send 1 ETH to bob.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "bob.eth"},
                             "amount": {"kind": "exact", "value": "1"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, manual = convert_case(raw)
    # App contract: unresolved recipients (including unknown ENS) are accepted
    assert manual is None
    call = case["expected_calls"][0]
    assert call["to"] == "bob.eth"
    # bob.eth is not in LOOKUP, so no ens_resolution flag (only known ENS names get it)
    assert case["requires"] == []


def test_difficulty_derived_swap_and_multilingual_medium():
    swap_raw = {"id": "swap-en-001", "user_message": "Swap 100 USDC for DAI", "category": "truePositiveSwap",
                "language": "english", "expected_tool": "swap",
                "expected_args": {"from_token": {"kind": "exact", "value": "USDC"},
                                  "to_token": {"kind": "exact", "value": "DAI"},
                                  "amount": {"kind": "exact", "value": "100"},
                                  "amount_side": {"kind": "exact", "value": "input"}}, "notes": None}
    case, _ = convert_case(swap_raw)
    assert case["difficulty"] == "medium"

    it_raw = {"id": "transfer-it-001", "user_message": "Invia 1 ETH a vitalik.eth",
              "category": "multilingualTransfer", "language": "italian", "expected_tool": "transfer",
              "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                                "amount": {"kind": "exact", "value": "1"},
                                "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, _ = convert_case(it_raw)
    assert case["difficulty"] == "medium"

    en_raw = {"id": "transfer-en-001", "user_message": "Send 0.1 ETH to vitalik.eth",
              "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
              "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                                "amount": {"kind": "exact", "value": "0.1"},
                                "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, _ = convert_case(en_raw)
    assert case["difficulty"] == "easy"


def test_convert_transfer_emits_app_contract_gold():
    raw = {
        "id": "transfer-en-001", "user_message": "Send 0.1 ETH to vitalik.eth",
        "category": "truePositiveTransfer", "language": "english",
        "expected_tool": "transfer",
        "expected_args": {
            "to": {"kind": "exact", "value": "vitalik.eth"},
            "amount": {"kind": "exact", "value": "0.1"},
            "token": {"kind": "exact", "value": "ETH"},
        },
        "notes": None,
    }
    case, manual = convert_case(raw)
    assert manual is None
    assert case["expected_calls"] == [{
        "tool": "transfer", "chainId": "1", "to": "vitalik.eth",
        "amount": "0.1", "token": "ETH",
    }]
