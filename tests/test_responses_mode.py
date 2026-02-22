try:
    from responses_mode import is_codex_model_name, resolve_runtime_mode
except ImportError:  # pragma: no cover
    from data.plugins.astrbot_plugin_openai_responses.responses_mode import (
        is_codex_model_name,
        resolve_runtime_mode,
    )


def test_resolve_runtime_mode_prefers_explicit_config():
    resolved = resolve_runtime_mode(
        model="gpt-4o-mini",
        api_base="https://api.openai.com/v1",
        codex_mode="chatgpt",
    )
    assert resolved.mode == "chatgpt"
    assert resolved.reason == "explicit:chatgpt"


def test_resolve_runtime_mode_detects_chatgpt_codex_base_url():
    resolved = resolve_runtime_mode(
        model="gpt-4o-mini",
        api_base="https://chatgpt.com/backend-api/codex",
        codex_mode="auto",
    )
    assert resolved.mode == "chatgpt"


def test_resolve_runtime_mode_detects_codex_model_name():
    resolved = resolve_runtime_mode(
        model="gpt-5.2-codex",
        api_base="https://api.openai.com/v1",
        codex_mode="auto",
    )
    assert resolved.mode == "chatgpt"


def test_is_codex_model_name_recognizes_codex_suffix():
    assert is_codex_model_name("gpt-5-codex")
    assert is_codex_model_name("foo.codex.beta")
    assert not is_codex_model_name("gpt-4o-mini")
