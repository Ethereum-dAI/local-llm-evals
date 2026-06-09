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
