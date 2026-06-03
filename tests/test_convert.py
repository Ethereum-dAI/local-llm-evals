from scripts.convert_recognition import to_base_units, convert_case, LOOKUP


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
    assert call["tool"] == "executeTx"
    assert call["to"] == LOOKUP["ens"]["vitalik.eth"]
    assert call["value"] == "100000000000000000"
    assert call["function"] is None
    assert call["args"] == []


def test_convert_erc20_transfer():
    raw = {"id": "transfer-en-002", "user_message": "Send 100 USDC to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "100"},
                             "token": {"kind": "exact", "value": "USDC"}}, "notes": None}
    case, manual = convert_case(raw)
    call = case["expected_calls"][0]
    assert call["to"] == LOOKUP["tokens"]["USDC"]["address"]
    assert call["function"] == "transfer(address,uint256)"
    assert call["args"] == [LOOKUP["ens"]["vitalik.eth"], "100000000"]


def test_swap_routed_to_manual():
    raw = {"id": "swap-en-001", "user_message": "Swap 100 USDC for ETH", "category": "truePositiveSwap",
           "language": "english", "expected_tool": "swap",
           "expected_args": {"from_token": {"kind": "exact", "value": "USDC"}}, "notes": None}
    case, manual = convert_case(raw)
    assert case is None
    assert manual == "swap-en-001"


def test_amount_all_routed_to_manual():
    raw = {"id": "transfer-en-003", "user_message": "Send all my ETH to vitalik.eth",
           "category": "truePositiveTransfer", "language": "english", "expected_tool": "transfer",
           "expected_args": {"to": {"kind": "exact", "value": "vitalik.eth"},
                             "amount": {"kind": "exact", "value": "all"},
                             "token": {"kind": "exact", "value": "ETH"}}, "notes": None}
    case, manual = convert_case(raw)
    assert case is None
    assert manual == "transfer-en-003"
