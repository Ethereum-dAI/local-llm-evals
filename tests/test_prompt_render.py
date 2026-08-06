from pf.prompt import render, SYSTEM


def test_render_single_turn():
    chat = render({"vars": {"user_message": "Send 0.1 ETH to vitalik.eth"}})
    assert chat[0] == {"role": "system", "content": SYSTEM}
    assert chat[1] == {"role": "user", "content": "Send 0.1 ETH to vitalik.eth"}
    assert len(chat) == 2


def test_render_multi_turn():
    msgs = [
        {"role": "user", "content": "Send 0.1 ETH"},
        {"role": "assistant", "content": "Which address?"},
        {"role": "user", "content": "to vitalik.eth"},
    ]
    chat = render({"vars": {"messages": msgs}})
    assert chat[0]["role"] == "system"
    assert chat[1:] == msgs


def test_render_without_account_context_unchanged():
    chat = render({"vars": {"user_message": "Send 0.1 ETH to vitalik.eth"}})
    assert len(chat) == 2 and chat[0]["role"] == "system" and chat[1]["role"] == "user"


def test_render_safe_protocol_adds_reference_and_context():
    ctx = {"safe": "0xSafe", "owners": ["0xA", "0xB"], "threshold": 2}
    chat = render({"vars": {"user_message": "Remove signer 0xB from my Safe.",
                            "protocol": "safe", "account_context": ctx}})
    assert [m["role"] for m in chat] == ["system", "system", "user"]
    addendum = chat[1]["content"]
    assert "addOwnerWithThreshold(address,uint256)" in addendum
    assert "removeOwner(address,address,uint256)" in addendum
    assert "0xSafe" in addendum and "0xA, 0xB" in addendum


def test_render_aave_protocol_adds_reference():
    chat = render({"vars": {"user_message": "Supply 3 USDC to Aave v3.", "protocol": "aave"}})
    assert [m["role"] for m in chat] == ["system", "system", "user"]
    ref = chat[1]["content"]
    assert "supply(address,uint256,address,uint16)" in ref
    assert "borrow(address,uint256,uint256,uint16,address)" in ref
    assert "0x87870Bca3F3fD6335C3F4ce8392D69350B4fa4E2" in ref
    assert "<wallet>" in ref


def test_render_railgun_protocol_adds_reference():
    chat = render({"vars": {"user_message": "Shield 0.01 ETH.", "protocol": "railgun"}})
    assert [m["role"] for m in chat] == ["system", "system", "user"]
    ref = chat[1]["content"]
    assert "shield" in ref and "unshield" in ref
    # The privacy tools are the one exception to the global base-unit rule.
    assert "HUMAN units" in ref and "do NOT convert to wei" in ref
    assert "0x000...dEaD" in ref, "unshield burn-address refusal must be stated"


def test_railgun_reference_is_not_rendered_for_other_cases():
    for vars_ in ({"user_message": "Send 0.1 ETH to vitalik.eth"},
                  {"user_message": "Supply 3 USDC to Aave v3.", "protocol": "aave"}):
        chat = render({"vars": vars_})
        assert all("do NOT convert to wei" not in m["content"] for m in chat)


def test_render_no_protocol_unchanged():
    chat = render({"vars": {"user_message": "Send 0.1 ETH to vitalik.eth"}})
    assert len(chat) == 2 and chat[0]["role"] == "system" and chat[1]["role"] == "user"


def test_expected_summary_var_is_not_leaked_to_model():
    # expected_summary is a viewer-only var; render must never put it in the chat.
    chat = render({"vars": {"user_message": "Send 0.1 ETH to vitalik.eth",
                            "expected_summary": "executeTx to 0xSECRETGOLD (native)"}})
    assert all("0xSECRETGOLD" not in m["content"] for m in chat)
