import pytest

from astrbot.core.agent.tool import FunctionTool, ToolSet

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
                    "function": {"name": "test_tool", "arguments": "{\"q\":\"x\"}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"},
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
    assert input_items[2]["content"][0] == {"type": "output_text", "text": "calling tool"}

    assert input_items[3]["type"] == "function_call"
    assert input_items[3]["call_id"] == "call_1"
    assert input_items[3]["name"] == "test_tool"

    assert input_items[4]["type"] == "function_call_output"
    assert input_items[4]["call_id"] == "call_1"
    assert input_items[4]["output"] == "{\"ok\":true}"


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

        yield ("response.created", {"type": "response.created", "response": {"id": "resp_1"}})
        yield ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "Hello"})
        yield ("response.output_text.delta", {"type": "response.output_text.delta", "delta": " world"})
        yield (
            "response.reasoning_summary_text.delta",
            {"type": "response.reasoning_summary_text.delta", "delta": "think"},
        )
        yield (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "call_1", "name": "test_tool"},
            },
        )
        yield (
            "response.function_call_arguments.delta",
            {"type": "response.function_call_arguments.delta", "delta": "{\"q\":\"x\"}"},
        )
        yield ("response.function_call_arguments.done", {"type": "response.function_call_arguments.done"})
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
