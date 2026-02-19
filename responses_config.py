from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_TOOL_FALLBACK_MODES = {"parse_then_retry", "retry_only", "parse_only"}


@dataclass(frozen=True)
class ConfigNormalizationResult:
    config: dict[str, Any]
    warnings: list[str]


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def normalize_provider_config(raw_config: dict[str, Any]) -> ConfigNormalizationResult:
    config = dict(raw_config or {})
    warnings: list[str] = []

    def _migrate_key(old: str, new: str) -> None:
        if old in config and new not in config:
            config[new] = config[old]
            warnings.append(
                f"Config field `{old}` is deprecated; use `{new}` instead."
            )

    _migrate_key("base_url", "api_base")
    _migrate_key("api_url", "api_base")
    _migrate_key("headers", "custom_headers")
    _migrate_key("extra_body", "custom_extra_body")
    _migrate_key("tool_fallback_attempts", "tool_fallback_retry_attempts")
    _migrate_key("usage_log", "log_usage")

    timeout_raw = config.get("timeout", 120)
    try:
        timeout = int(timeout_raw)
    except (TypeError, ValueError):
        timeout = 120
        warnings.append(
            f"Invalid `timeout` value {timeout_raw!r}; fallback to {timeout} seconds."
        )
    if timeout <= 0:
        warnings.append(
            f"Invalid `timeout` value {timeout!r}; fallback to 120 seconds."
        )
        timeout = 120
    config["timeout"] = timeout

    api_base = config.get("api_base", "https://api.openai.com/v1")
    if not isinstance(api_base, str):
        api_base = str(api_base)
    api_base = api_base.strip() or "https://api.openai.com/v1"
    config["api_base"] = api_base

    proxy = config.get("proxy", "") or ""
    if not isinstance(proxy, str):
        proxy = str(proxy)
    config["proxy"] = proxy

    custom_headers = config.get("custom_headers", {})
    if not isinstance(custom_headers, dict) or not custom_headers:
        config["custom_headers"] = {}
    else:
        normalized_headers: dict[str, str] = {}
        for key, value in custom_headers.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            normalized_headers[normalized_key] = str(value)
        config["custom_headers"] = normalized_headers

    custom_extra_body = config.get("custom_extra_body", {})
    if isinstance(custom_extra_body, dict):
        config["custom_extra_body"] = custom_extra_body
    else:
        config["custom_extra_body"] = {}
        if custom_extra_body not in (None, "", []):
            warnings.append(
                "Invalid `custom_extra_body`; it must be an object/dict, ignored."
            )

    config["tool_fallback_enabled"] = _coerce_bool(
        config.get("tool_fallback_enabled", True),
        True,
    )

    tool_fallback_mode = config.get("tool_fallback_mode", "parse_then_retry")
    if not isinstance(tool_fallback_mode, str):
        tool_fallback_mode = "parse_then_retry"
    tool_fallback_mode = tool_fallback_mode.strip().lower()
    if tool_fallback_mode not in _TOOL_FALLBACK_MODES:
        warnings.append(
            f"Unknown `tool_fallback_mode` value {tool_fallback_mode!r}; fallback to parse_then_retry."
        )
        tool_fallback_mode = "parse_then_retry"
    config["tool_fallback_mode"] = tool_fallback_mode

    retry_attempts_raw = config.get("tool_fallback_retry_attempts", 1)
    try:
        retry_attempts = int(retry_attempts_raw)
    except (TypeError, ValueError):
        retry_attempts = 1
        warnings.append(
            f"Invalid `tool_fallback_retry_attempts` value {retry_attempts_raw!r}; fallback to {retry_attempts}."
        )
    config["tool_fallback_retry_attempts"] = max(0, min(3, retry_attempts))

    tool_choice = config.get("tool_fallback_force_tool_choice", "required")
    if isinstance(tool_choice, dict):
        config["tool_fallback_force_tool_choice"] = tool_choice
    elif isinstance(tool_choice, str) and tool_choice.strip():
        config["tool_fallback_force_tool_choice"] = tool_choice.strip().lower()
    else:
        config["tool_fallback_force_tool_choice"] = "required"

    config["tool_fallback_stream_buffer"] = _coerce_bool(
        config.get("tool_fallback_stream_buffer", True),
        True,
    )

    config["log_usage"] = _coerce_bool(config.get("log_usage", False), False)

    def _coerce_bounded_int(
        value: Any,
        *,
        default: int,
        minimum: int,
        maximum: int,
        field_name: str,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            warnings.append(
                f"Invalid `{field_name}` value {value!r}; fallback to {default}."
            )
            parsed = default
        return max(minimum, min(maximum, parsed))

    config["max_output_chars"] = _coerce_bounded_int(
        config.get("max_output_chars", 200_000),
        default=200_000,
        minimum=10_000,
        maximum=2_000_000,
        field_name="max_output_chars",
    )
    config["stream_buffer_max_chars"] = _coerce_bounded_int(
        config.get("stream_buffer_max_chars", 20_000),
        default=20_000,
        minimum=1_000,
        maximum=500_000,
        field_name="stream_buffer_max_chars",
    )
    config["stream_buffer_max_responses"] = _coerce_bounded_int(
        config.get("stream_buffer_max_responses", 512),
        default=512,
        minimum=64,
        maximum=10_000,
        field_name="stream_buffer_max_responses",
    )

    patterns = config.get("image_moderation_error_patterns", [])
    normalized_patterns: list[str] = []
    if isinstance(patterns, str):
        patterns = [patterns]
    if isinstance(patterns, list):
        for pattern in patterns:
            if not isinstance(pattern, str):
                continue
            stripped = pattern.strip()
            if stripped:
                normalized_patterns.append(stripped)
    config["image_moderation_error_patterns"] = normalized_patterns

    return ConfigNormalizationResult(config=config, warnings=warnings)
