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

