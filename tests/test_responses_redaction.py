try:
    # When running tests from the plugin repository root
    from responses_redaction import (
        redact_headers,
        redact_event_data,
        redact_json,
        redact_proxy_url,
        redact_text,
    )
except ImportError:  # pragma: no cover
    # When running tests from an AstrBot repository checkout
    from data.plugins.astrbot_plugin_openai_responses.responses_redaction import (
        redact_headers,
        redact_event_data,
        redact_json,
        redact_proxy_url,
        redact_text,
    )


def test_redact_text_masks_sk_and_bearer():
    raw = "Authorization: Bearer sk-1234567890abcdef1234567890"
    redacted = redact_text(raw)
    assert "sk-1234567890abcdef" not in redacted
    assert "Bearer sk-" not in redacted
    assert "Bearer ***" in redacted


def test_redact_headers_masks_sensitive_values():
    redacted = redact_headers(
        {
            "Authorization": "Bearer sk-1234567890abcdef1234567890",
            "X-Test": "ok",
        }
    )
    assert redacted is not None
    assert redacted["Authorization"] == "***"
    assert redacted["X-Test"] == "ok"


def test_redact_proxy_url_masks_credentials():
    proxy = "http://user:pass@127.0.0.1:7890"
    redacted = redact_proxy_url(proxy)
    assert redacted is not None
    assert "user" not in redacted
    assert "pass" not in redacted
    assert "***:***@" in redacted


def test_redact_json_masks_sensitive_tokens():
    raw = {"headers": {"Authorization": "Bearer sk-1234567890abcdef1234567890"}}
    redacted = redact_json(raw)
    assert "sk-1234567890abcdef" not in redacted
    assert "Bearer sk-" not in redacted


def test_redact_event_data_redacts_tool_arguments_payload():
    raw = {"type": "response.function_call_arguments.delta", "delta": '{"q":"x"}'}
    redacted = redact_event_data("response.function_call_arguments.delta", raw)
    assert '{"q":"x"}' not in redacted
    assert "<redacted" in redacted
