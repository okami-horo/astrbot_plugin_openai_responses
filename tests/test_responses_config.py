try:
    # When running tests from the plugin repository root
    from responses_config import normalize_provider_config
except ImportError:  # pragma: no cover
    # When running tests from an AstrBot repository checkout
    from data.plugins.astrbot_plugin_openai_responses.responses_config import (
        normalize_provider_config,
    )


def test_normalize_provider_config_coerces_timeout_str_to_int():
    result = normalize_provider_config({"timeout": "120"})
    assert result.config["timeout"] == 120
    assert result.warnings == []


def test_normalize_provider_config_migrates_base_url_to_api_base_with_warning():
    result = normalize_provider_config({"base_url": "https://example.com/v1"})
    assert result.config["api_base"] == "https://example.com/v1"
    assert any("base_url" in w and "api_base" in w for w in result.warnings)


def test_normalize_provider_config_sets_codex_defaults():
    result = normalize_provider_config({})
    assert result.config["codex_mode"] == "auto"
    assert result.config["codex_transport"] == "auto"
    assert result.config["codex_strict_tool_call"] is True
    assert result.config["codex_disable_pseudo_tool_call"] is True
    assert result.config["codex_turn_state_enabled"] is True
    assert result.config["codex_parallel_tool_calls"] is True
    assert result.config["codex_context_prune_strategy"] == "pair_aware"


def test_normalize_provider_config_warns_on_invalid_codex_fields():
    result = normalize_provider_config(
        {
            "codex_mode": "invalid-mode",
            "codex_transport": "streaming",
            "codex_context_prune_strategy": "unknown",
        }
    )
    assert result.config["codex_mode"] == "auto"
    assert result.config["codex_transport"] == "auto"
    assert result.config["codex_context_prune_strategy"] == "pair_aware"
    assert any("codex_mode" in w for w in result.warnings)
    assert any("codex_transport" in w for w in result.warnings)
    assert any("codex_context_prune_strategy" in w for w in result.warnings)
