from types import SimpleNamespace

from wallet_evals.adapters.openrouter import OpenRouterAdapter


class _FakeClient:
    def __init__(self, message):
        self._message = message
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.last_kwargs = None

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)])


def test_adapter_returns_parsed_turn_from_native_tool_calls():
    msg = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(function=SimpleNamespace(
            name="executeTx",
            arguments='{"chainId":"1","to":"0xabc","value":"0","function":null,"args":[]}'))],
    )
    adapter = OpenRouterAdapter(model="x/y", client=_FakeClient(msg))
    turn = adapter.run("Send 1 ETH to 0xabc")
    assert turn.tool_calls[0].name == "executeTx"
    assert turn.tool_calls[0].to == "0xabc"


def test_adapter_sends_system_prompt_and_tools():
    msg = SimpleNamespace(content="hi", tool_calls=None)
    fake = _FakeClient(msg)
    adapter = OpenRouterAdapter(model="x/y", client=fake)
    adapter.run("hello")
    sent = fake.last_kwargs
    assert sent["model"] == "x/y"
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][1] == {"role": "user", "content": "hello"}
    assert sent["temperature"] == 0.2
    assert len(sent["tools"]) == 3


def test_adapter_handles_text_only_response():
    msg = SimpleNamespace(content="I need more info.", tool_calls=None)
    adapter = OpenRouterAdapter(model="x/y", client=_FakeClient(msg))
    turn = adapter.run("ambiguous")
    assert turn.tool_calls == []
    assert turn.content == "I need more info."
