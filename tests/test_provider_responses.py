import json

import pytest

from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.provider.entities import LLMResponse

try:
    # When running tests from the plugin repository root
    from provider_responses import ProviderOpenAIResponsesPlugin
except ImportError:  # pragma: no cover
    # When running tests from an AstrBot repository checkout
    from data.plugins.astrbot_plugin_openai_responses.provider_responses import (
        ProviderOpenAIResponsesPlugin,
    )


def _make_provider(overrides: dict | None = None) -> ProviderOpenAIResponsesPlugin:
    provider_config = {
        "id": "test-openai-responses-plugin",
        "type": "openai_responses_plugin",
        "model": "gpt-4o-mini",
        "key": ["test-key"],
        "api_base": "https://api.openai.com/v1",
    }
    if overrides:
        provider_config.update(overrides)
    return ProviderOpenAIResponsesPlugin(
        provider_config=provider_config,
        provider_settings={},
    )


def _make_tool_set() -> ToolSet:
    tool = FunctionTool(
        name="test_tool",
        description="A test tool",
        parameters={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        handler=None,
    )
    return ToolSet([tool])


def test_convert_chat_messages_to_responses_input_maps_roles_and_multimodal_and_tools():
    provider = _make_provider()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
            ],
        },
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
    ]

    input_items = provider._convert_chat_messages_to_responses_input(messages)

    assert input_items[0]["type"] == "message"
    assert input_items[0]["role"] == "developer"

    assert input_items[1]["type"] == "message"
    assert input_items[1]["role"] == "user"
    assert input_items[1]["content"][0] == {"type": "input_text", "text": "hello"}
    assert input_items[1]["content"][1]["type"] == "input_image"

    assert input_items[2]["type"] == "message"
    assert input_items[2]["role"] == "assistant"
    assert input_items[2]["content"][0] == {
        "type": "output_text",
        "text": "calling tool",
    }

    assert input_items[3]["type"] == "function_call"
    assert input_items[3]["call_id"] == "call_1"
    assert input_items[3]["name"] == "test_tool"

    assert input_items[4]["type"] == "function_call_output"
    assert input_items[4]["call_id"] == "call_1"
    assert input_items[4]["output"] == '{"ok":true}'


def test_convert_chat_messages_to_responses_input_decodes_tool_json_unicode_escape():
    provider = _make_provider()
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"stdout":"\\u4f60\\u597d","stderr":"","exit_code":0}',
        },
    ]

    input_items = provider._convert_chat_messages_to_responses_input(messages)
    assert input_items[1]["type"] == "function_call_output"
    assert "\\u4f60\\u597d" not in input_items[1]["output"]
    assert json.loads(input_items[1]["output"]) == {
        "stdout": "你好",
        "stderr": "",
        "exit_code": 0,
    }


def test_convert_openai_tools_to_responses_tools_flattens_schema():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    converted = ProviderOpenAIResponsesPlugin._convert_openai_tools_to_responses_tools(
        tools
    )
    assert converted == [
        {
            "type": "function",
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


@pytest.mark.asyncio
async def test_query_stream_parses_text_reasoning_tool_calls_and_usage(monkeypatch):
    provider = _make_provider()
    tools = _make_tool_set()

    async def _fake_iter_responses_sse(*, request_body, api_key):
        assert request_body["model"] == "gpt-4o-mini"
        assert request_body["stream"] is True
        assert api_key == "test-key"

        yield (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        )
        yield (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "Hello"},
        )
        yield (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": " world"},
        )
        yield (
            "response.reasoning_summary_text.delta",
            {"type": "response.reasoning_summary_text.delta", "delta": "think"},
        )
        yield (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "test_tool",
                },
            },
        )
        yield (
            "response.function_call_arguments.delta",
            {"type": "response.function_call_arguments.delta", "delta": '{"q":"x"}'},
        )
        yield (
            "response.function_call_arguments.done",
            {"type": "response.function_call_arguments.done"},
        )
        yield (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 2},
                        "output_tokens": 3,
                    }
                },
            },
        )

    monkeypatch.setattr(provider, "_iter_responses_sse", _fake_iter_responses_sse)

    payloads = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}

    outputs = []
    async for resp in provider._query_stream(payloads, tools, api_key="test-key"):
        outputs.append(resp)

    assert outputs
    assert any(o.is_chunk and o.completion_text == "Hello" for o in outputs)
    assert any(o.is_chunk and o.completion_text == " world" for o in outputs)
    assert any(o.is_chunk and o.reasoning_content == "think" for o in outputs)

    final = outputs[-1]
    assert final.is_chunk is False
    assert final.id == "resp_1"
    assert final.completion_text == "Hello world"
    assert final.reasoning_content == "think"
    assert final.role == "tool"
    assert final.tools_call_ids == ["call_1"]
    assert final.tools_call_name == ["test_tool"]
    assert final.tools_call_args == [{"q": "x"}]
    assert final.usage is not None
    assert final.usage.input_other == 8
    assert final.usage.input_cached == 2
    assert final.usage.output == 3


@pytest.mark.asyncio
async def test_query_stream_keeps_mixed_output_text_and_content_part_delta_in_order(
    monkeypatch,
):
    provider = _make_provider()

    async def _fake_iter_responses_sse(*, request_body, api_key):
        assert request_body["stream"] is True
        assert api_key == "test-key"
        yield (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        )
        yield (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "Hello"},
        )
        yield (
            "response.content_part.delta",
            {
                "type": "response.content_part.delta",
                "delta": {"text": " world"},
            },
        )
        yield ("response.completed", {"type": "response.completed", "response": {}})

    monkeypatch.setattr(provider, "_iter_responses_sse", _fake_iter_responses_sse)

    payloads = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}

    outputs = []
    async for resp in provider._query_stream(payloads, None, api_key="test-key"):
        outputs.append(resp)

    chunks = [o.completion_text for o in outputs if o.is_chunk]
    assert chunks == ["Hello", " world"]
    assert outputs[-1].completion_text == "Hello world"


@pytest.mark.asyncio
async def test_text_chat_returns_refusal_text_when_refusal_events_exist(monkeypatch):
    provider = _make_provider()

    async def _fake_iter_responses_sse(*, request_body, api_key):
        assert request_body["stream"] is True
        assert api_key == "test-key"
        yield (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        )
        yield (
            "response.refusal.delta",
            {"type": "response.refusal.delta", "delta": "I cannot help with that."},
        )
        yield (
            "response.completed",
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 1, "output_tokens": 1}},
            },
        )

    monkeypatch.setattr(provider, "_iter_responses_sse", _fake_iter_responses_sse)
    resp = await provider.text_chat(prompt="hi")

    assert resp.role == "assistant"
    assert resp.completion_text == "I cannot help with that."
    assert resp.tools_call_args == []


@pytest.mark.asyncio
async def test_text_chat_raises_when_completion_is_empty_and_no_tool_calls(monkeypatch):
    provider = _make_provider()

    async def _fake_iter_responses_sse(*, request_body, api_key):
        assert request_body["stream"] is True
        assert api_key == "test-key"
        yield (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        )
        yield ("response.completed", {"type": "response.completed", "response": {}})

    monkeypatch.setattr(provider, "_iter_responses_sse", _fake_iter_responses_sse)

    with pytest.raises(Exception, match="为空且无工具调用"):
        await provider.text_chat(prompt="hi")


@pytest.mark.asyncio
async def test_query_stream_keeps_interleaved_tool_calls_without_loss(monkeypatch):
    provider = _make_provider()
    tools = ToolSet(
        [
            FunctionTool(
                name="tool_a",
                description="tool a",
                parameters={"type": "object", "properties": {"a": {"type": "integer"}}},
                handler=None,
            ),
            FunctionTool(
                name="tool_b",
                description="tool b",
                parameters={"type": "object", "properties": {"b": {"type": "integer"}}},
                handler=None,
            ),
        ]
    )

    async def _fake_iter_responses_sse(*, request_body, api_key):
        assert request_body["stream"] is True
        assert api_key == "test-key"
        yield (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        )
        yield (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "tool_a",
                },
            },
        )
        yield (
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '{"a":',
            },
        )
        yield (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "fc_2",
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "tool_b",
                },
            },
        )
        yield (
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_2",
                "delta": '{"b":2}',
            },
        )
        yield (
            "response.function_call_arguments.done",
            {"type": "response.function_call_arguments.done", "item_id": "fc_2"},
        )
        yield (
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": "1}",
            },
        )
        yield (
            "response.function_call_arguments.done",
            {"type": "response.function_call_arguments.done", "item_id": "fc_1"},
        )
        yield ("response.completed", {"type": "response.completed", "response": {}})

    monkeypatch.setattr(provider, "_iter_responses_sse", _fake_iter_responses_sse)
    payloads = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}

    outputs = []
    async for resp in provider._query_stream(payloads, tools, api_key="test-key"):
        outputs.append(resp)

    final = outputs[-1]
    assert final.role == "tool"
    assert final.tools_call_ids == ["call_1", "call_2"]
    assert final.tools_call_name == ["tool_a", "tool_b"]
    assert final.tools_call_args == [{"a": 1}, {"b": 2}]


@pytest.mark.asyncio
async def test_query_stream_ingests_tool_calls_from_response_output_without_deltas(
    monkeypatch,
):
    provider = _make_provider()
    tools = _make_tool_set()

    async def _fake_iter_responses_sse(*, request_body, api_key):
        assert request_body["stream"] is True
        assert api_key == "test-key"
        yield (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        )
        yield (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "test_tool",
                            "arguments": '{"q":"x"}',
                        }
                    ],
                },
            },
        )

    monkeypatch.setattr(provider, "_iter_responses_sse", _fake_iter_responses_sse)

    payloads = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    outputs = []
    async for resp in provider._query_stream(payloads, tools, api_key="test-key"):
        outputs.append(resp)

    final = outputs[-1]
    assert final.role == "tool"
    assert final.tools_call_ids == ["call_1"]
    assert final.tools_call_name == ["test_tool"]
    assert final.tools_call_args == [{"q": "x"}]


@pytest.mark.asyncio
async def test_query_stream_keeps_interleaved_tool_calls_with_output_item_done(
    monkeypatch,
):
    provider = _make_provider()
    tools = ToolSet(
        [
            FunctionTool(
                name="tool_a",
                description="tool a",
                parameters={"type": "object", "properties": {"a": {"type": "integer"}}},
                handler=None,
            ),
            FunctionTool(
                name="tool_b",
                description="tool b",
                parameters={"type": "object", "properties": {"b": {"type": "integer"}}},
                handler=None,
            ),
        ]
    )

    async def _fake_iter_responses_sse(*, request_body, api_key):
        assert request_body["stream"] is True
        assert api_key == "test-key"
        yield (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        )
        yield (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "tool_a",
                },
            },
        )
        yield (
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '{"a":',
            },
        )
        yield (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "fc_2",
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "tool_b",
                },
            },
        )
        yield (
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_2",
                "delta": '{"b":2}',
            },
        )
        yield (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "id": "fc_2",
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "tool_b",
                    "arguments": '{"b":2}',
                },
            },
        )
        yield (
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": "1}",
            },
        )
        yield (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "tool_a",
                    "arguments": '{"a":1}',
                },
            },
        )
        yield ("response.completed", {"type": "response.completed", "response": {}})

    monkeypatch.setattr(provider, "_iter_responses_sse", _fake_iter_responses_sse)
    payloads = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}

    outputs = []
    async for resp in provider._query_stream(payloads, tools, api_key="test-key"):
        outputs.append(resp)

    final = outputs[-1]
    assert final.role == "tool"
    assert final.tools_call_ids == ["call_1", "call_2"]
    assert final.tools_call_name == ["tool_a", "tool_b"]
    assert final.tools_call_args == [{"a": 1}, {"b": 2}]


@pytest.mark.asyncio
async def test_iter_responses_sse_supports_multiline_data_event(monkeypatch):
    provider = _make_provider()

    class _FakeStreamResponse:
        def __init__(self, lines: list[str]) -> None:
            self.status_code = 200
            self._lines = lines

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aread(self) -> bytes:
            return b""

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    class _FakeClient:
        def __init__(self, lines: list[str]) -> None:
            self._lines = lines

        def stream(self, *_args, **_kwargs):
            return _FakeStreamResponse(self._lines)

    lines = [
        "event: response.output_text.delta",
        "data: {",
        'data: "type": "response.output_text.delta",',
        'data: "delta": "Hello"',
        "data: }",
        "",
        "data: [DONE]",
    ]
    monkeypatch.setattr(provider, "_get_http_client", lambda: _FakeClient(lines))

    events = []
    async for event_type, event_data in provider._iter_responses_sse(
        request_body={"model": "gpt-4o-mini", "input": [], "stream": True},
        api_key="test-key",
    ):
        events.append((event_type, event_data))

    assert events == [
        (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "Hello"},
        )
    ]


@pytest.mark.asyncio
async def test_query_stream_falls_back_to_text_from_response_completed(monkeypatch):
    provider = _make_provider()

    async def _fake_iter_responses_sse(*, request_body, api_key):
        assert request_body["stream"] is True
        assert api_key == "test-key"
        yield (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        )
        yield (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 4,
                        "output_tokens": 2,
                    },
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "final text"}],
                        }
                    ],
                },
            },
        )

    monkeypatch.setattr(provider, "_iter_responses_sse", _fake_iter_responses_sse)

    payloads = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    outputs = []
    async for resp in provider._query_stream(payloads, None, api_key="test-key"):
        outputs.append(resp)

    final = outputs[-1]
    assert final.role == "assistant"
    assert final.completion_text == "final text"
    assert final.tools_call_args == []


@pytest.mark.asyncio
async def test_query_stream_falls_back_to_text_from_response_completed_without_usage(
    monkeypatch,
):
    provider = _make_provider()

    async def _fake_iter_responses_sse(*, request_body, api_key):
        assert request_body["stream"] is True
        assert api_key == "test-key"
        yield (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        )
        yield (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "final text"}],
                        }
                    ],
                },
            },
        )

    monkeypatch.setattr(provider, "_iter_responses_sse", _fake_iter_responses_sse)

    payloads = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    outputs = []
    async for resp in provider._query_stream(payloads, None, api_key="test-key"):
        outputs.append(resp)

    assert outputs[-1].role == "assistant"
    assert outputs[-1].completion_text == "final text"


@pytest.mark.asyncio
async def test_query_stream_aligns_completion_when_output_text_done_is_emitted(
    monkeypatch,
):
    provider = _make_provider()

    async def _fake_iter_responses_sse(*, request_body, api_key):
        assert request_body["stream"] is True
        assert api_key == "test-key"
        yield (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        )
        yield (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "Hello"},
        )
        yield (
            "response.output_text.done",
            {"type": "response.output_text.done", "text": "Hello world"},
        )
        yield ("response.completed", {"type": "response.completed", "response": {}})

    monkeypatch.setattr(provider, "_iter_responses_sse", _fake_iter_responses_sse)

    payloads = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    outputs = []
    async for resp in provider._query_stream(payloads, None, api_key="test-key"):
        outputs.append(resp)

    assert outputs[-1].completion_text == "Hello world"


def test_build_responses_request_sets_tool_choice_override_when_tools_present():
    provider = _make_provider()
    tools = _make_tool_set()
    payloads = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}

    request = provider._build_responses_request(
        payloads,
        tools,
        tool_choice_override="required",
    )

    assert request["stream"] is True
    assert request["tool_choice"] == "required"


def test_build_responses_request_is_stateless_and_does_not_set_previous_response_id():
    provider = _make_provider()
    payloads = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}

    request = provider._build_responses_request(payloads, None)

    assert "previous_response_id" not in request


@pytest.mark.asyncio
async def test_query_final_response_rotates_key_on_401_and_retries(monkeypatch):
    provider = _make_provider(overrides={"key": ["k1", "k2"]})

    import asyncio
    import random
    import sys

    try:
        from responses_errors import UpstreamResponsesError
    except ImportError:  # pragma: no cover
        from data.plugins.astrbot_plugin_openai_responses.responses_errors import (
            UpstreamResponsesError,
        )

    monkeypatch.setattr(random, "choice", lambda seq: seq[0])

    calls: list[str] = []

    async def _fake_query_stream(
        payloads,
        tool_set,
        *,
        api_key,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=None,
    ):
        calls.append(api_key)
        if len(calls) == 1:
            raise UpstreamResponsesError("unauthorized", status_code=401, body={})
        yield LLMResponse(role="assistant", completion_text="ok")

    monkeypatch.setattr(provider, "_query_stream", _fake_query_stream)

    resp, _ = await provider._query_final_response_with_retries(
        payloads={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        func_tool=None,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=None,
        max_retries=3,
    )

    assert resp.completion_text == "ok"
    assert calls == ["k1", "k2"]


@pytest.mark.asyncio
async def test_query_final_response_applies_backoff_on_429(monkeypatch):
    provider = _make_provider(overrides={"key": ["k1", "k2"]})

    import asyncio
    import random

    try:
        from responses_errors import UpstreamResponsesError
    except ImportError:  # pragma: no cover
        from data.plugins.astrbot_plugin_openai_responses.responses_errors import (
            UpstreamResponsesError,
        )

    monkeypatch.setattr(random, "choice", lambda seq: seq[0])

    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float):
        sleep_calls.append(float(seconds))

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    calls: list[str] = []

    async def _fake_query_stream(
        payloads,
        tool_set,
        *,
        api_key,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=None,
    ):
        calls.append(api_key)
        if len(calls) == 1:
            raise UpstreamResponsesError("rate limited", status_code=429, body={})
        yield LLMResponse(role="assistant", completion_text="ok")

    monkeypatch.setattr(provider, "_query_stream", _fake_query_stream)

    resp, _ = await provider._query_final_response_with_retries(
        payloads={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        func_tool=None,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=None,
        max_retries=3,
    )

    assert resp.completion_text == "ok"
    assert calls == ["k1", "k2"]
    assert sleep_calls and sleep_calls[0] > 0


@pytest.mark.asyncio
async def test_query_final_response_retries_on_connection_error_with_backoff(
    monkeypatch,
):
    provider = _make_provider(overrides={"key": ["k1"]})

    import asyncio
    import random
    import sys

    monkeypatch.setattr(random, "choice", lambda seq: seq[0])

    module = sys.modules[provider.__class__.__module__]
    monkeypatch.setattr(module, "is_connection_error", lambda _exc: True)
    monkeypatch.setattr(module, "log_connection_failure", lambda *_a, **_k: None)

    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float):
        sleep_calls.append(float(seconds))

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    calls = 0

    async def _fake_query_stream(
        payloads,
        tool_set,
        *,
        api_key,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=None,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Exception("connection dropped")
        yield LLMResponse(role="assistant", completion_text="ok")

    monkeypatch.setattr(provider, "_query_stream", _fake_query_stream)

    resp, _ = await provider._query_final_response_with_retries(
        payloads={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        func_tool=None,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=None,
        max_retries=3,
    )

    assert resp.completion_text == "ok"
    assert calls == 2
    assert sleep_calls and sleep_calls[0] > 0


@pytest.mark.asyncio
async def test_text_chat_converts_pseudo_tool_call_without_retry(monkeypatch):
    provider = _make_provider()
    tools = _make_tool_set()
    call_count = 0

    async def _fake_query_stream(
        payloads,
        tool_set,
        *,
        api_key,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=None,
    ):
        nonlocal call_count
        call_count += 1
        assert tool_set is tools
        assert api_key == "test-key"
        assert tool_choice_override is None
        yield LLMResponse(
            role="assistant",
            completion_text='assistant to=functions.test_tool\n{"q":"x"}',
        )

    monkeypatch.setattr(provider, "_query_stream", _fake_query_stream)

    resp = await provider.text_chat(prompt="hello", func_tool=tools)

    assert call_count == 1
    assert resp.role == "tool"
    assert resp.tools_call_name == ["test_tool"]
    assert resp.tools_call_args == [{"q": "x"}]
    assert resp.completion_text == ""


@pytest.mark.asyncio
async def test_text_chat_converts_pseudo_tool_call_even_if_not_in_allowlist(
    monkeypatch,
):
    provider = _make_provider()
    tools = _make_tool_set()

    async def _fake_query_stream(
        payloads,
        tool_set,
        *,
        api_key,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=None,
    ):
        assert tool_set is tools
        assert api_key == "test-key"
        assert tool_choice_override is None
        yield LLMResponse(
            role="assistant",
            completion_text='assistant to=functions.other_tool\n{"q":"x"}',
        )

    monkeypatch.setattr(provider, "_query_stream", _fake_query_stream)

    resp = await provider.text_chat(prompt="hello", func_tool=tools)

    assert resp.role == "tool"
    assert resp.tools_call_name == ["other_tool"]
    assert resp.tools_call_args == [{"q": "x"}]
    assert resp.completion_text == ""


@pytest.mark.asyncio
async def test_text_chat_retries_with_required_tool_choice_after_pseudo_parse_fail(
    monkeypatch,
):
    provider = _make_provider()
    tools = _make_tool_set()
    seen_tool_choice: list[str | dict | None] = []

    async def _fake_query_stream(
        payloads,
        tool_set,
        *,
        api_key,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=None,
    ):
        assert tool_set is tools
        assert api_key == "test-key"
        seen_tool_choice.append(tool_choice_override)
        if len(seen_tool_choice) == 1:
            yield LLMResponse(
                role="assistant",
                completion_text="assistant to=functions.test_tool\n{not-json}",
            )
            return
        resp = LLMResponse(role="tool")
        resp.tools_call_ids = ["call_1"]
        resp.tools_call_name = ["test_tool"]
        resp.tools_call_args = [{"q": "x"}]
        yield resp

    monkeypatch.setattr(provider, "_query_stream", _fake_query_stream)

    resp = await provider.text_chat(prompt="hello", func_tool=tools)

    assert seen_tool_choice == [None, "required"]
    assert resp.role == "tool"
    assert resp.tools_call_name == ["test_tool"]
    assert resp.tools_call_args == [{"q": "x"}]


@pytest.mark.asyncio
async def test_text_chat_stream_buffers_first_attempt_and_retries_on_pseudo_tool_call(
    monkeypatch,
):
    provider = _make_provider()
    tools = _make_tool_set()
    seen_tool_choice: list[str | dict | None] = []

    async def _fake_query_stream(
        payloads,
        tool_set,
        *,
        api_key,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=None,
    ):
        assert tool_set is tools
        assert api_key == "test-key"
        seen_tool_choice.append(tool_choice_override)
        if len(seen_tool_choice) == 1:
            yield LLMResponse(
                role="assistant",
                completion_text="assistant to=functions.test_tool",
                is_chunk=True,
            )
            yield LLMResponse(
                role="assistant",
                completion_text="assistant to=functions.test_tool\n{not-json}",
            )
            return
        resp = LLMResponse(role="tool")
        resp.tools_call_ids = ["call_1"]
        resp.tools_call_name = ["test_tool"]
        resp.tools_call_args = [{"q": "x"}]
        yield resp

    monkeypatch.setattr(provider, "_query_stream", _fake_query_stream)

    outputs = []
    async for response in provider.text_chat_stream(prompt="hello", func_tool=tools):
        outputs.append(response)

    assert seen_tool_choice == [None, "required"]
    assert len(outputs) == 1
    assert outputs[0].role == "tool"
    assert outputs[0].tools_call_name == ["test_tool"]
    assert outputs[0].tools_call_args == [{"q": "x"}]
