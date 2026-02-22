import pytest
import sys
import types

if "astrbot" not in sys.modules:
    astrbot_stub = types.ModuleType("astrbot")

    class _LoggerStub:
        def warning(self, *_args, **_kwargs):
            return None

    astrbot_stub.logger = _LoggerStub()
    sys.modules["astrbot"] = astrbot_stub

try:
    from responses_errors import UpstreamResponsesError
    from responses_sse import iter_responses_sse_events
except ImportError:  # pragma: no cover
    from data.plugins.astrbot_plugin_openai_responses.responses_errors import (
        UpstreamResponsesError,
    )
    from data.plugins.astrbot_plugin_openai_responses.responses_sse import (
        iter_responses_sse_events,
    )


async def _line_stream(lines: list[str]):
    for line in lines:
        yield line


@pytest.mark.asyncio
async def test_iter_responses_sse_raises_on_response_failed():
    lines = [
        "event: response.failed",
        'data: {"type":"response.failed","response":{"error":{"message":"boom","code":"bad_request"}}}',
        "",
    ]

    with pytest.raises(UpstreamResponsesError, match="response.failed"):
        async for _ in iter_responses_sse_events(_line_stream(lines)):
            pass


@pytest.mark.asyncio
async def test_iter_responses_sse_raises_on_response_incomplete():
    lines = [
        "event: response.incomplete",
        'data: {"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}',
        "",
    ]

    with pytest.raises(UpstreamResponsesError, match="response.incomplete"):
        async for _ in iter_responses_sse_events(_line_stream(lines)):
            pass
