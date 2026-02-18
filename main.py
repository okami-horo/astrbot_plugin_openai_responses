from __future__ import annotations

import copy

from astrbot.api import star
from astrbot.core import logger
from astrbot.core.config.default import CONFIG_METADATA_2
from astrbot.core.provider.entities import ProviderType

from .provider_responses import ProviderOpenAIResponsesPlugin
from .register_provider import register_or_replace_provider_adapter, unregister_provider_adapter


PROVIDER_TYPE_NAME = "openai_responses_plugin"
PROVIDER_TEMPLATE_KEY = "OpenAI (Responses, Plugin)"


def _inject_provider_source_template(template_key: str, template: dict) -> None:
    try:
        config_template = CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
            "config_template"
        ]
        config_template[template_key] = copy.deepcopy(template)
    except Exception as e:
        logger.warning(
            "Failed to inject provider source template into WebUI schema: %s",
            e,
        )


def _remove_provider_source_template(template_key: str) -> None:
    try:
        config_template = CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
            "config_template"
        ]
        config_template.pop(template_key, None)
    except Exception:
        return


def _register_provider() -> None:
    default_config_tmpl = {
        "id": PROVIDER_TYPE_NAME,
        "type": PROVIDER_TYPE_NAME,
        "provider": "openai",
        "provider_type": "chat_completion",
        "enable": True,
        "key": [],
        "api_base": "https://api.openai.com/v1",
        "timeout": 120,
        "proxy": "",
        "custom_headers": {},
        "reasoning_effort": "",
        "custom_extra_body": {},
        "tool_fallback_enabled": True,
        "tool_fallback_mode": "parse_then_retry",
        "tool_fallback_retry_attempts": 1,
        "tool_fallback_force_tool_choice": "required",
        "tool_fallback_stream_buffer": True,
    }

    register_or_replace_provider_adapter(
        provider_type_name=PROVIDER_TYPE_NAME,
        desc="OpenAI API Responses (/v1/responses) provider adapter (plugin)",
        cls_type=ProviderOpenAIResponsesPlugin,
        provider_type=ProviderType.CHAT_COMPLETION,
        default_config_tmpl=default_config_tmpl,
        provider_display_name=PROVIDER_TEMPLATE_KEY,
    )
    _inject_provider_source_template(PROVIDER_TEMPLATE_KEY, default_config_tmpl)


_register_provider()


class Main(star.Star):
    async def terminate(self) -> None:
        unregister_provider_adapter(PROVIDER_TYPE_NAME)
        _remove_provider_source_template(PROVIDER_TEMPLATE_KEY)
