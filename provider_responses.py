from __future__ import annotations

import asyncio
import base64
import json
import random
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from astrbot import logger
from astrbot.api.provider import Provider as AstrProvider
from astrbot.core.agent.message import ContentPart, ImageURLPart, TextPart
from astrbot.core.agent.tool import ToolSet
from astrbot.core.provider.entities import LLMResponse, ToolCallsResult
from astrbot.core.utils.io import download_image_by_url
from astrbot.core.utils.network_utils import (
    create_proxy_client,
    is_connection_error,
    log_connection_failure,
)

try:
    from responses_config import normalize_provider_config
except ImportError:  # pragma: no cover
    from .responses_config import normalize_provider_config

try:
    from responses_redaction import redact_proxy_url
except ImportError:  # pragma: no cover
    from .responses_redaction import redact_proxy_url

try:
    from responses_errors import UpstreamResponsesError, compute_backoff_seconds, error_type
except ImportError:  # pragma: no cover
    from .responses_errors import UpstreamResponsesError, compute_backoff_seconds, error_type

try:
    from responses_sse import iter_responses_sse_events
except ImportError:  # pragma: no cover
    from .responses_sse import iter_responses_sse_events

try:
    from responses_accumulator import ResponsesStreamAccumulator
except ImportError:  # pragma: no cover
    from .responses_accumulator import ResponsesStreamAccumulator

try:
    from responses_tools import (
        extract_allowed_tool_names,
        looks_like_pseudo_tool_call_text,
        maybe_convert_pseudo_tool_calls,
    )
except ImportError:  # pragma: no cover
    from .responses_tools import (
        extract_allowed_tool_names,
        looks_like_pseudo_tool_call_text,
        maybe_convert_pseudo_tool_calls,
    )

try:
    from responses_mode import RuntimeModeResolution, resolve_runtime_mode
except ImportError:  # pragma: no cover
    from .responses_mode import RuntimeModeResolution, resolve_runtime_mode

try:
    from responses_turn_state import ResponsesTurnState
except ImportError:  # pragma: no cover
    from .responses_turn_state import ResponsesTurnState


class ProviderOpenAIResponsesPlugin(AstrProvider):
    """OpenAI Responses API provider adapter for plugin distribution."""

    _ERROR_TEXT_CANDIDATE_MAX_CHARS = 4096
    _TOOL_FALLBACK_MODES = {"parse_then_retry", "retry_only", "parse_only"}

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        normalized = normalize_provider_config(provider_config)
        AstrProvider.__init__(self, normalized.config, provider_settings)
        for warning in normalized.warnings:
            logger.warning("%s", warning)

        self.api_keys: list[str] = AstrProvider.get_keys(self)
        self.chosen_api_key = self.api_keys[0] if self.api_keys else ""

        self.timeout = int(self.provider_config.get("timeout", 120))

        custom_headers = self.provider_config.get("custom_headers") or {}
        self.custom_headers = (
            custom_headers if isinstance(custom_headers, dict) and custom_headers else None
        )

        self.api_base = str(
            self.provider_config.get("api_base", "https://api.openai.com/v1")
        )
        self.proxy = str(self.provider_config.get("proxy", "") or "")

        self._http_client: httpx.AsyncClient | None = None
        self._turn_state = ResponsesTurnState()

        model = provider_config.get("model")
        if not model:
            model = "gpt-4o-mini"
        self.set_model(str(model))
        if str(self.provider_config.get("codex_transport", "auto")).lower() == "websocket":
            logger.warning(
                "codex_transport=websocket is reserved but not implemented yet; fallback to SSE."
            )

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        client = create_proxy_client("OpenAI", self.proxy)
        if client is not None:
            self._http_client = client
            return client
        self._http_client = httpx.AsyncClient()
        return self._http_client

    def _build_headers(self, api_key: str | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if self.custom_headers:
            headers.update({str(k): str(v) for k, v in self.custom_headers.items()})
        return headers

    def _responses_url(self) -> str:
        return self.api_base.rstrip("/") + "/responses"

    @staticmethod
    def _normalize_tool_output_text(content: str) -> str:
        stripped = content.strip()
        if not stripped:
            return content

        # Try to normalize JSON output to avoid inconsistent escaping (e.g. \\uXXXX)
        # and keep the payload stable across retries.
        if stripped[0] not in {'{', '[', '"'}:
            return content

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return content

        if isinstance(parsed, (dict, list)):
            return json.dumps(
                parsed,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
        if isinstance(parsed, str):
            return parsed
        return str(parsed)

    @staticmethod
    def _normalize_reasoning_effort(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        allowed = {"low", "medium", "high", "xhigh"}
        if normalized not in allowed:
            logger.warning(
                "Unknown reasoning_effort value: %s. It will be passed through as-is, "
                "and may be rejected by the upstream API.",
                normalized,
            )
        return normalized

    @staticmethod
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

    def _tool_fallback_enabled(self, tools: ToolSet | None) -> bool:
        if tools is None:
            return False
        if not extract_allowed_tool_names(tools):
            return False
        raw = self.provider_config.get("tool_fallback_enabled", True)
        return self._coerce_bool(raw, True)

    def _tool_fallback_mode(self) -> str:
        raw = self.provider_config.get("tool_fallback_mode", "parse_then_retry")
        if not isinstance(raw, str):
            return "parse_then_retry"
        normalized = raw.strip().lower()
        if normalized in self._TOOL_FALLBACK_MODES:
            return normalized
        logger.warning(
            "Unknown tool_fallback_mode=%s, fallback to parse_then_retry.",
            raw,
        )
        return "parse_then_retry"

    def _tool_fallback_retry_attempts(self) -> int:
        raw = self.provider_config.get("tool_fallback_retry_attempts", 1)
        try:
            attempts = int(raw)
        except (TypeError, ValueError):
            attempts = 1
        return max(0, min(3, attempts))

    def _tool_fallback_force_tool_choice(self) -> str | dict[str, Any]:
        raw = self.provider_config.get("tool_fallback_force_tool_choice", "required")
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized:
                return normalized
        return "required"

    def _tool_fallback_stream_buffer_enabled(self) -> bool:
        raw = self.provider_config.get("tool_fallback_stream_buffer", True)
        return self._coerce_bool(raw, True)

    def _resolve_runtime_mode(self, *, model: str) -> RuntimeModeResolution:
        return resolve_runtime_mode(
            model=model,
            api_base=self.api_base,
            codex_mode=str(self.provider_config.get("codex_mode", "auto")),
        )

    def _codex_disable_pseudo_tool_call(self) -> bool:
        raw = self.provider_config.get("codex_disable_pseudo_tool_call", True)
        return self._coerce_bool(raw, True)

    def _codex_strict_tool_call(self) -> bool:
        raw = self.provider_config.get("codex_strict_tool_call", True)
        return self._coerce_bool(raw, True)

    def _codex_parallel_tool_calls_enabled(self) -> bool:
        raw = self.provider_config.get("codex_parallel_tool_calls", True)
        return self._coerce_bool(raw, True)

    def _codex_turn_state_enabled(self) -> bool:
        raw = self.provider_config.get("codex_turn_state_enabled", True)
        return self._coerce_bool(raw, True)

    def _codex_context_prune_strategy(self) -> str:
        raw = self.provider_config.get("codex_context_prune_strategy", "pair_aware")
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"pair_aware", "legacy"}:
                return normalized
        return "pair_aware"

    @staticmethod
    def _extract_message_text_for_instructions(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        parts.append(text)
                    continue
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "")
                if item_type in {"text", "input_text", "output_text"}:
                    text = str(item.get("text") or "").strip()
                    if text:
                        parts.append(text)
                    continue
                if item_type == "refusal":
                    refusal = str(item.get("refusal") or item.get("text") or "").strip()
                    if refusal:
                        parts.append(refusal)
                    continue
                serialized = json.dumps(item, ensure_ascii=False, default=str).strip()
                if serialized:
                    parts.append(serialized)
            return "\n".join(parts).strip()

        if content is None:
            return ""
        return str(content).strip()

    def _split_codex_instructions_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        instructions: list[str] = []
        filtered_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role", "")).lower()
            if role in {"system", "developer"}:
                text = self._extract_message_text_for_instructions(msg.get("content"))
                if text:
                    instructions.append(text)
                continue
            filtered_messages.append(msg)

        if not instructions:
            return None, messages
        return "\n\n".join(instructions), filtered_messages

    def _record_turn_success(self, payloads: dict[str, Any], response: LLMResponse) -> None:
        if not self._codex_turn_state_enabled():
            return
        if response is None:
            return
        raw_completion = (
            response.raw_completion if isinstance(response.raw_completion, dict) else {}
        )
        tool_calls = raw_completion.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []
        self._turn_state.note_success(
            response_id=getattr(response, "id", None),
            request_messages=list(payloads.get("messages") or []),
            tool_calls=tool_calls,
        )

    async def _prune_context_on_context_length(
        self,
        context_query: list[dict[str, Any]],
    ) -> None:
        strategy = self._codex_context_prune_strategy()
        if (
            strategy == "pair_aware"
            and self._turn_state.prune_messages_for_context_limit(
                context_query, strategy=strategy
            )
        ):
            return
        await self.pop_record(context_query)

    @staticmethod
    def _convert_message_content(
        content: Any,
        *,
        role: str,
    ) -> list[dict[str, Any]]:
        def _text_item(text: str) -> dict[str, Any]:
            text_type = "output_text" if role == "assistant" else "input_text"
            return {"type": text_type, "text": text}

        if content is None:
            return []

        if isinstance(content, str):
            stripped = content.strip()
            if not stripped:
                return []
            return [_text_item(content)]

        if not isinstance(content, list):
            return [_text_item(str(content))]

        converted: list[dict[str, Any]] = []

        for part in content:
            if isinstance(part, dict):
                part_type = str(part.get("type", ""))

                if part_type in {"text", "input_text", "output_text"}:
                    converted.append(_text_item(str(part.get("text", ""))))
                    continue

                if part_type == "refusal":
                    if role == "assistant":
                        refusal = part.get("refusal")
                        if refusal is None:
                            refusal = part.get("text", "")
                        converted.append({"type": "refusal", "refusal": str(refusal)})
                    else:
                        converted.append(
                            {
                                "type": "input_text",
                                "text": str(
                                    part.get("refusal") or part.get("text") or ""
                                ),
                            }
                        )
                    continue

                if part_type in {"image_url", "input_image"}:
                    image_url_obj = part.get("image_url", {})
                    if isinstance(image_url_obj, dict):
                        image_url = image_url_obj.get("url", "")
                    else:
                        image_url = str(image_url_obj)

                    if role == "assistant":
                        converted.append(_text_item("[image]"))
                    else:
                        converted.append(
                            {"type": "input_image", "image_url": str(image_url)}
                        )
                    continue

                # Pass-through for already-valid response parts when possible
                if role == "assistant":
                    if part_type.startswith("output_") or part_type == "refusal":
                        converted.append(part)
                        continue
                else:
                    if part_type.startswith("input_"):
                        converted.append(part)
                        continue

                converted.append(
                    _text_item(json.dumps(part, ensure_ascii=False, default=str))
                )
                continue

            if isinstance(part, str):
                if part.strip():
                    converted.append(_text_item(part))
                continue

            converted.append(_text_item(str(part)))

        return converted

    @staticmethod
    def _convert_openai_tools_to_responses_tools(
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        converted_tools: list[dict[str, Any]] = []
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get("type") == "function"
                and isinstance(tool.get("function"), dict)
            ):
                func = tool["function"]
                converted_tool: dict[str, Any] = {
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                }
                if "parameters" in func:
                    converted_tool["parameters"] = func["parameters"]
                converted_tools.append(converted_tool)
            else:
                converted_tools.append(tool)
        return converted_tools

    def _convert_chat_messages_to_responses_input(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        pending_call_ids: list[str] = []

        for msg in messages:
            role = str(msg.get("role", ""))

            if role == "assistant" and msg.get("tool_calls"):
                content = msg.get("content")
                if content not in (None, "", []):
                    converted_content = self._convert_message_content(
                        content,
                        role="assistant",
                    )
                    if converted_content:
                        input_items.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": converted_content,
                            }
                        )

                for tool_call in msg.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                    pending_call_ids.append(call_id)

                    func = tool_call.get("function", {})
                    if not isinstance(func, dict):
                        func = {}

                    func_name = func.get("name") or ""
                    if not func_name:
                        func_name = f"unknown_function_{uuid.uuid4().hex[:8]}"
                        logger.warning(
                            "tool_call name is empty, inferred as: %s", func_name
                        )

                    arguments = func.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(
                            arguments, ensure_ascii=False, default=str
                        )

                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": func_name,
                            "arguments": arguments,
                        }
                    )
                continue

            if role == "tool":
                call_id = msg.get("tool_call_id")
                if not call_id:
                    call_id = (
                        pending_call_ids.pop(0)
                        if pending_call_ids
                        else f"call_{uuid.uuid4().hex[:24]}"
                    )
                content = msg.get("content")
                if isinstance(content, str):
                    output = self._normalize_tool_output_text(content)
                elif content is None:
                    output = ""
                else:
                    output = json.dumps(content, ensure_ascii=False, default=str)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    }
                )
                continue

            mapped_role = "developer" if role == "system" else role
            converted_content = self._convert_message_content(
                msg.get("content"),
                role=mapped_role,
            )
            if converted_content:
                input_items.append(
                    {
                        "type": "message",
                        "role": mapped_role,
                        "content": converted_content,
                    }
                )

        return input_items

    def _build_responses_request(
        self,
        payloads: dict[str, Any],
        tools: ToolSet | None,
        *,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice_override: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = str(payloads.get("model") or self.get_model())
        messages = payloads.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError("payloads.messages must be a list")

        runtime_mode = self._resolve_runtime_mode(model=model)
        instructions: str | None = None
        message_items = messages
        if runtime_mode.is_codex_chatgpt:
            instructions, message_items = self._split_codex_instructions_from_messages(
                messages
            )

        request_body: dict[str, Any] = {
            "model": model,
            "input": self._convert_chat_messages_to_responses_input(message_items),
            "stream": True,
        }
        if runtime_mode.is_codex_chatgpt and instructions:
            request_body["instructions"] = instructions

        if runtime_mode.is_codex_chatgpt and self._codex_turn_state_enabled():
            prompt_cache_key = self._turn_state.build_prompt_cache_key(
                session_id=(
                    str(payloads.get("session_id"))
                    if payloads.get("session_id") is not None
                    else None
                ),
                messages=message_items,
                model=model,
                api_base=self.api_base,
            )
            request_body["prompt_cache_key"] = prompt_cache_key

        if tools:
            omit_empty_param_field = "gemini" in model.lower()
            tool_list = tools.openai_schema(
                omit_empty_parameter_field=omit_empty_param_field,
            )
            converted = self._convert_openai_tools_to_responses_tools(tool_list)
            if converted:
                request_body["tools"] = converted
                if tool_choice_override is not None:
                    request_body["tool_choice"] = tool_choice_override
                elif runtime_mode.is_codex_chatgpt:
                    request_body["tool_choice"] = "auto"
                if runtime_mode.is_codex_chatgpt:
                    request_body["parallel_tool_calls"] = (
                        self._codex_parallel_tool_calls_enabled()
                    )

        effort_source = reasoning_effort
        if effort_source is None:
            effort_source = self.provider_config.get("reasoning_effort")

        normalized_effort = self._normalize_reasoning_effort(effort_source)
        if normalized_effort is not None:
            request_body["reasoning"] = {"effort": normalized_effort}

        if response_format is not None:
            fmt_type = (
                response_format.get("type")
                if isinstance(response_format, dict)
                else None
            )
            if fmt_type == "json_object":
                request_body["text"] = {"format": {"type": "json_object"}}
            elif fmt_type == "json_schema":
                json_schema_obj = response_format.get("json_schema", {})
                if isinstance(json_schema_obj, dict):
                    response_fmt: dict[str, Any] = {
                        "type": "json_schema",
                        "name": json_schema_obj.get("name", "response_schema"),
                        "schema": json_schema_obj.get("schema", {}),
                    }
                    if "strict" in json_schema_obj:
                        response_fmt["strict"] = json_schema_obj.get("strict")
                    request_body["text"] = {"format": response_fmt}

        custom_extra_body = self.provider_config.get("custom_extra_body", {})
        if isinstance(custom_extra_body, dict):
            request_body.update(custom_extra_body)

        return request_body

    async def _iter_responses_sse(
        self,
        *,
        request_body: dict[str, Any],
        api_key: str,
    ) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        url = self._responses_url()
        headers = self._build_headers(api_key)
        client = self._get_http_client()
        timeout = httpx.Timeout(float(self.timeout))

        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=request_body,
            timeout=timeout,
        ) as response:
            if response.status_code >= 400:
                raw_bytes = await response.aread()
                raw_text = raw_bytes.decode("utf-8", errors="replace")
                try:
                    body: dict[str, Any] | str = json.loads(raw_text)
                except json.JSONDecodeError:
                    body = raw_text
                raise UpstreamResponsesError(
                    f"Responses API HTTP {response.status_code}: {raw_text[:512]}",
                    status_code=response.status_code,
                    body=body,
                    response_text=raw_text,
                )
            async for event_type, event_data in iter_responses_sse_events(
                response.aiter_lines()
            ):
                yield event_type, event_data

    async def _query_stream(
        self,
        payloads: dict[str, Any],
        tools: ToolSet | None,
        *,
        api_key: str,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
        tool_choice_override: str | dict[str, Any] | None = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        request_body = self._build_responses_request(
            payloads,
            tools,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            tool_choice_override=tool_choice_override,
        )
        recovered_without_prompt_cache_key = False

        while True:
            processor = ResponsesStreamAccumulator(
                model=str(payloads.get("model") or self.get_model()),
                debug_events=self._coerce_bool(
                    self.provider_config.get("debug_sse_events", False), False
                ),
                max_text_chars=int(self.provider_config.get("max_output_chars", 200_000)),
            )
            try:
                async for event_type, event_data in self._iter_responses_sse(
                    request_body=request_body,
                    api_key=api_key,
                ):
                    for chunk in processor.handle_event(event_type, event_data):
                        yield chunk
            except Exception as exc:
                prompt_cache_key = request_body.get("prompt_cache_key")
                if (
                    not recovered_without_prompt_cache_key
                    and isinstance(prompt_cache_key, str)
                    and self._is_prompt_cache_key_invalid(exc)
                ):
                    recovered_without_prompt_cache_key = True
                    request_body = dict(request_body)
                    request_body.pop("prompt_cache_key", None)
                    logger.warning(
                        "Responses request rejected prompt_cache_key, retrying once without it. "
                        "model=%s stream=%s prompt_cache_key_len=%s",
                        payloads.get("model"),
                        True,
                        len(prompt_cache_key),
                    )
                    continue
                raise

            final_response = processor.build_final_response(tools_provided=tools is not None)
            if recovered_without_prompt_cache_key:
                logger.info(
                    "Responses request recovered after dropping prompt_cache_key. "
                    "model=%s stream=%s",
                    payloads.get("model"),
                    True,
                )
            yield final_response
            return

    def _finally_convert_payload(self, payloads: dict) -> None:
        for message in payloads.get("messages", []):
            if message.get("role") == "assistant" and isinstance(
                message.get("content"), list
            ):
                reasoning_content = ""
                new_content = []
                for part in message["content"]:
                    if isinstance(part, dict) and part.get("type") == "think":
                        reasoning_content += str(part.get("think"))
                    else:
                        new_content.append(part)
                message["content"] = new_content
                if reasoning_content:
                    message["reasoning_content"] = reasoning_content

    async def _prepare_chat_payload(
        self,
        prompt: str | None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        contexts: list[dict] | list[Any] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: ToolCallsResult | list[ToolCallsResult] | None = None,
        model: str | None = None,
        extra_user_content_parts: list[ContentPart] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if contexts is None:
            contexts = []
        new_record = None
        if prompt is not None:
            new_record = await self.assemble_context(
                prompt, image_urls, extra_user_content_parts
            )
        context_query = self._ensure_message_to_dicts(contexts)
        if new_record:
            context_query.append(new_record)
        if system_prompt:
            context_query.insert(0, {"role": "system", "content": system_prompt})

        for part in context_query:
            if "_no_save" in part:
                del part["_no_save"]

        if tool_calls_result:
            if isinstance(tool_calls_result, ToolCallsResult):
                context_query.extend(tool_calls_result.to_openai_messages())
            else:
                for tcr in tool_calls_result:
                    context_query.extend(tcr.to_openai_messages())

        model = model or self.get_model()
        payloads: dict[str, Any] = {
            "messages": context_query,
            "model": model,
            "session_id": session_id,
        }
        self._finally_convert_payload(payloads)
        return payloads, context_query

    def _remove_images_from_context(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        new_contexts: list[dict[str, Any]] = []
        for context in contexts:
            if "content" in context and isinstance(context["content"], list):
                new_content: list[Any] = []
                for item in context["content"]:
                    if isinstance(item, dict) and "image_url" in item:
                        continue
                    new_content.append(item)
                if not new_content:
                    new_content = [{"type": "text", "text": "[图片]"}]
                context["content"] = new_content
            new_contexts.append(context)
        return new_contexts

    def _context_contains_image(self, contexts: list[dict[str, Any]]) -> bool:
        for context in contexts:
            content = context.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        return True
        return False

    def _get_image_moderation_error_patterns(self) -> list[str]:
        configured = self.provider_config.get("image_moderation_error_patterns", [])
        patterns: list[str] = []
        if isinstance(configured, str):
            configured = [configured]
        if isinstance(configured, list):
            for pattern in configured:
                if not isinstance(pattern, str):
                    continue
                pattern = pattern.strip()
                if pattern:
                    patterns.append(pattern)
        return patterns

    def _extract_error_text_candidates(self, error: Exception) -> list[str]:
        candidates: list[str] = []

        def _append_candidate(candidate: Any) -> None:
            if candidate is None:
                return
            text = str(candidate).strip()
            if not text:
                return
            if len(text) > self._ERROR_TEXT_CANDIDATE_MAX_CHARS:
                text = text[: self._ERROR_TEXT_CANDIDATE_MAX_CHARS]
            candidates.append(text)

        _append_candidate(str(error))

        body = getattr(error, "body", None)
        if isinstance(body, dict):
            _append_candidate(json.dumps(body, ensure_ascii=False, default=str))
        elif isinstance(body, str):
            _append_candidate(body)

        response = getattr(error, "response", None)
        if response is not None:
            response_text = getattr(response, "text", None)
            if isinstance(response_text, str):
                _append_candidate(response_text)

        # de-dupe while preserving order
        seen = set()
        deduped: list[str] = []
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _is_content_moderated_upload_error(self, error: Exception) -> bool:
        patterns = [p.lower() for p in self._get_image_moderation_error_patterns()]
        if not patterns:
            return False
        candidates = [c.lower() for c in self._extract_error_text_candidates(error)]
        for pattern in patterns:
            for candidate in candidates:
                if pattern in candidate:
                    return True
        return False

    def _is_prompt_cache_key_invalid(self, error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code is not None and status_code != 400:
            return False

        body = getattr(error, "body", None)
        if isinstance(body, dict):
            err_payload = body.get("error")
            if isinstance(err_payload, dict):
                param = err_payload.get("param")
                if isinstance(param, str) and param.strip() == "prompt_cache_key":
                    return True
                message = err_payload.get("message")
                if isinstance(message, str) and "prompt_cache_key" in message.lower():
                    return True
            param = body.get("param")
            if isinstance(param, str) and param.strip() == "prompt_cache_key":
                return True

        candidates = self._extract_error_text_candidates(error)
        return any("prompt_cache_key" in candidate.lower() for candidate in candidates)

    async def _handle_api_error(
        self,
        e: Exception,
        payloads: dict[str, Any],
        context_query: list[dict[str, Any]],
        func_tool: ToolSet | None,
        chosen_key: str,
        available_api_keys: list[str],
        retry_cnt: int,
        max_retries: int,
        image_fallback_used: bool = False,
    ) -> tuple[
        bool, str, list[str], dict[str, Any], list[dict[str, Any]], ToolSet | None, bool
    ]:
        """Return (stop, chosen_key, available_keys, payloads, context, tools, image_fallback_used)."""
        candidates = self._extract_error_text_candidates(e)
        joined = "\n".join(candidates)
        status_code = getattr(e, "status_code", None)
        if status_code is None and isinstance(e, UpstreamResponsesError):
            status_code = e.status_code

        if status_code in (401, 403):
            if chosen_key in available_api_keys:
                available_api_keys.remove(chosen_key)
            if available_api_keys:
                chosen_key = random.choice(available_api_keys)
                return (
                    False,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    context_query,
                    func_tool,
                    image_fallback_used,
                )
            raise e

        if status_code == 429 or "429" in joined:
            logger.warning("Responses API rate limited, rotating key and retrying.")
            if retry_cnt < max_retries - 1:
                backoff_s = compute_backoff_seconds(
                    attempt=retry_cnt,
                    base_seconds=1.0,
                    max_seconds=10.0,
                    jitter_ratio=0.2,
                )
                await asyncio.sleep(backoff_s)
            if chosen_key in available_api_keys:
                available_api_keys.remove(chosen_key)
            if available_api_keys:
                chosen_key = random.choice(available_api_keys)
                return (
                    False,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    context_query,
                    func_tool,
                    image_fallback_used,
                )
            raise e

        if "maximum context length" in joined:
            logger.warning(
                "Context length exceeded, popping oldest records and retrying. records=%s",
                len(context_query),
            )
            await self._prune_context_on_context_length(context_query)
            payloads["messages"] = context_query
            return (
                False,
                chosen_key,
                available_api_keys,
                payloads,
                context_query,
                func_tool,
                image_fallback_used,
            )

        if "Function calling is not enabled" in joined or (
            "tool" in joined.lower() and "support" in joined.lower()
        ):
            logger.info("Model does not support tool calling, retry without tools.")
            return (
                False,
                chosen_key,
                available_api_keys,
                payloads,
                context_query,
                None,
                image_fallback_used,
            )

        if (
            "The model is not a VLM" in joined
            or self._is_content_moderated_upload_error(e)
        ):
            if image_fallback_used or not self._context_contains_image(context_query):
                raise e
            logger.warning(
                "Image input rejected by upstream, retrying with text-only context."
            )
            context_query = self._remove_images_from_context(context_query)
            payloads["messages"] = context_query
            return (
                False,
                chosen_key,
                available_api_keys,
                payloads,
                context_query,
                func_tool,
                True,
            )

        if is_connection_error(e):
            log_connection_failure("OpenAI", e, redact_proxy_url(self.proxy) or "")
            if retry_cnt < max_retries - 1:
                backoff_s = compute_backoff_seconds(
                    attempt=retry_cnt,
                    base_seconds=1.0,
                    max_seconds=10.0,
                    jitter_ratio=0.2,
                )
                await asyncio.sleep(backoff_s)
                return (
                    False,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    context_query,
                    func_tool,
                    image_fallback_used,
                )

        raise e

    async def _query_final_response_with_retries(
        self,
        *,
        payloads: dict[str, Any],
        func_tool: ToolSet | None,
        reasoning_effort: str | None,
        response_format: dict[str, Any] | None,
        tool_choice_override: str | dict[str, Any] | None,
        max_retries: int = 10,
    ) -> tuple[LLMResponse, ToolSet | None]:
        available_api_keys = self.api_keys.copy()
        chosen_key = random.choice(available_api_keys) if available_api_keys else ""
        image_fallback_used = False
        last_exception: Exception | None = None
        retry_cnt = 0

        for retry_cnt in range(max_retries):
            attempt_no = retry_cnt + 1
            attempt_start = time.perf_counter()
            try:
                self.chosen_api_key = chosen_key
                last_chunk: LLMResponse | None = None
                async for resp in self._query_stream(
                    payloads,
                    func_tool,
                    api_key=chosen_key,
                    reasoning_effort=reasoning_effort,
                    response_format=response_format,
                    tool_choice_override=tool_choice_override,
                ):
                    last_chunk = resp
                if last_chunk is None:
                    raise Exception("Empty responses stream")
                duration_ms = int((time.perf_counter() - attempt_start) * 1000)
                log_usage_enabled = self._coerce_bool(
                    self.provider_config.get("log_usage", False), False
                )
                usage_summary = None
                if log_usage_enabled and last_chunk.usage is not None:
                    usage_summary = {
                        "input_other": last_chunk.usage.input_other,
                        "input_cached": last_chunk.usage.input_cached,
                        "output": last_chunk.usage.output,
                    }
                logger.info(
                    "Responses request succeeded. attempt=%s duration_ms=%s error_type=%s response_id=%s model=%s stream=%s tool_enabled=%s usage=%s",
                    attempt_no,
                    duration_ms,
                    None,
                    getattr(last_chunk, "id", None),
                    payloads.get("model"),
                    False,
                    bool(func_tool),
                    usage_summary,
                )
                self._record_turn_success(payloads, last_chunk)
                return last_chunk, func_tool
            except Exception as e:
                last_exception = e
                duration_ms = int((time.perf_counter() - attempt_start) * 1000)
                logger.warning(
                    "Responses request failed. attempt=%s duration_ms=%s error_type=%s status_code=%s model=%s stream=%s tool_enabled=%s",
                    attempt_no,
                    duration_ms,
                    error_type(e),
                    getattr(e, "status_code", None),
                    payloads.get("model"),
                    False,
                    bool(func_tool),
                )
                (
                    _,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    _,
                    func_tool,
                    image_fallback_used,
                ) = await self._handle_api_error(
                    e,
                    payloads,
                    payloads.get("messages", []),
                    func_tool,
                    chosen_key,
                    available_api_keys,
                    retry_cnt,
                    max_retries,
                    image_fallback_used=image_fallback_used,
                )

        logger.error("Responses API call failed after %s retries.", max_retries)
        if last_exception is None:
            raise Exception("Unknown error")
        raise last_exception

    async def text_chat(
        self,
        prompt=None,
        session_id=None,
        image_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        extra_user_content_parts=None,
        **kwargs,
    ) -> LLMResponse:
        payloads, _ = await self._prepare_chat_payload(
            prompt,
            session_id,
            image_urls,
            contexts,
            system_prompt,
            tool_calls_result,
            model=model,
            extra_user_content_parts=extra_user_content_parts,
            **kwargs,
        )

        reasoning_effort = kwargs.get("reasoning_effort")
        response_format = kwargs.get("response_format")

        llm_response, effective_func_tool = await self._query_final_response_with_retries(
            payloads=payloads,
            func_tool=func_tool,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            tool_choice_override=None,
        )

        runtime_mode = self._resolve_runtime_mode(
            model=str(payloads.get("model") or self.get_model())
        )
        codex_mode = runtime_mode.is_codex_chatgpt
        tool_names = extract_allowed_tool_names(effective_func_tool)
        tools_available = bool(tool_names)
        strict_tool_call = codex_mode and tools_available and self._codex_strict_tool_call()
        codex_disable_pseudo = (
            codex_mode and tools_available and self._codex_disable_pseudo_tool_call()
        )
        tool_fallback_enabled = self._tool_fallback_enabled(effective_func_tool)
        tool_fallback_mode = self._tool_fallback_mode() if tool_fallback_enabled else ""
        allow_parse = tools_available and not codex_disable_pseudo and not strict_tool_call
        allow_retry = (
            not codex_mode
            and tool_fallback_enabled
            and tool_fallback_mode in {"parse_then_retry", "retry_only"}
        )

        if allow_parse and looks_like_pseudo_tool_call_text(llm_response):
            converted = maybe_convert_pseudo_tool_calls(llm_response, tools=effective_func_tool)
            if converted is not None:
                llm_response = converted

        tool_fallback_remaining = (
            self._tool_fallback_retry_attempts()
            if allow_retry
            else 0
        )
        while (
            allow_retry
            and tool_fallback_remaining > 0
            and looks_like_pseudo_tool_call_text(llm_response)
            and not llm_response.tools_call_name
        ):
            tool_fallback_remaining -= 1
            force_tool_choice = self._tool_fallback_force_tool_choice()
            logger.warning(
                "Pseudo tool-call text detected, retrying Responses request with tool_choice=%s. remaining=%s",
                force_tool_choice,
                tool_fallback_remaining,
            )
            llm_response, effective_func_tool = await self._query_final_response_with_retries(
                payloads=payloads,
                func_tool=effective_func_tool,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
                tool_choice_override=force_tool_choice,
            )
            if allow_parse and looks_like_pseudo_tool_call_text(llm_response):
                converted = maybe_convert_pseudo_tool_calls(llm_response, tools=effective_func_tool)
                if converted is not None:
                    llm_response = converted
            if llm_response.tools_call_name:
                break

        if (
            codex_mode
            and tools_available
            and looks_like_pseudo_tool_call_text(llm_response)
            and not llm_response.tools_call_name
        ):
            logger.error(
                "Codex mode detected pseudo tool-call text without structured tool call."
            )
            raise Exception("Codex 模式检测到伪工具调用文本，已中止以避免 JSON 泄露。")

        completion_text = llm_response.completion_text
        has_text = isinstance(completion_text, str) and bool(completion_text.strip())
        if not has_text and not llm_response.tools_call_args:
            logger.error("Responses API parsed empty completion without tool calls.")
            raise Exception("API 返回的 completion 为空且无工具调用。")
        return llm_response

    async def text_chat_stream(
        self,
        prompt=None,
        session_id=None,
        image_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        extra_user_content_parts=None,
        **kwargs,
    ) -> AsyncGenerator[LLMResponse, None]:
        payloads, context_query = await self._prepare_chat_payload(
            prompt,
            session_id,
            image_urls,
            contexts,
            system_prompt,
            tool_calls_result,
            model=model,
            extra_user_content_parts=extra_user_content_parts,
            **kwargs,
        )

        reasoning_effort = kwargs.get("reasoning_effort")
        response_format = kwargs.get("response_format")
        runtime_mode = self._resolve_runtime_mode(
            model=str(payloads.get("model") or self.get_model())
        )
        codex_mode = runtime_mode.is_codex_chatgpt
        tools_available = bool(extract_allowed_tool_names(func_tool))
        strict_tool_call = codex_mode and tools_available and self._codex_strict_tool_call()
        codex_disable_pseudo = (
            codex_mode and tools_available and self._codex_disable_pseudo_tool_call()
        )
        tool_fallback_enabled = self._tool_fallback_enabled(func_tool)
        tool_fallback_mode = self._tool_fallback_mode() if tool_fallback_enabled else ""
        allow_parse = tools_available and not codex_disable_pseudo and not strict_tool_call
        allow_retry = (
            not codex_mode
            and tool_fallback_enabled
            and tool_fallback_mode in {"parse_then_retry", "retry_only"}
        )
        stream_buffer_enabled = (
            (tools_available and self._tool_fallback_stream_buffer_enabled())
            or strict_tool_call
            or codex_disable_pseudo
        )

        # Keep old immediate-yield behavior when stream buffering is disabled.
        if not stream_buffer_enabled:
            max_retries = 10
            available_api_keys = self.api_keys.copy()
            chosen_key = (
                random.choice(available_api_keys) if available_api_keys else ""
            )
            image_fallback_used = False
            last_exception: Exception | None = None
            retry_cnt = 0

            for retry_cnt in range(max_retries):
                attempt_no = retry_cnt + 1
                attempt_start = time.perf_counter()
                yielded_any = False
                last_response: LLMResponse | None = None
                try:
                    self.chosen_api_key = chosen_key
                    async for response in self._query_stream(
                        payloads,
                        func_tool,
                        api_key=chosen_key,
                        reasoning_effort=reasoning_effort,
                        response_format=response_format,
                        tool_choice_override=None,
                    ):
                        if (
                            not response.is_chunk
                            and allow_parse
                            and looks_like_pseudo_tool_call_text(response)
                        ):
                            converted = maybe_convert_pseudo_tool_calls(response, tools=func_tool)
                            if converted is not None:
                                response = converted
                        yielded_any = True
                        last_response = response
                        yield response
                    duration_ms = int((time.perf_counter() - attempt_start) * 1000)
                    log_usage_enabled = self._coerce_bool(
                        self.provider_config.get("log_usage", False), False
                    )
                    usage_summary = None
                    if log_usage_enabled and last_response and last_response.usage is not None:
                        usage_summary = {
                            "input_other": last_response.usage.input_other,
                            "input_cached": last_response.usage.input_cached,
                            "output": last_response.usage.output,
                        }
                    logger.info(
                        "Responses stream succeeded. attempt=%s duration_ms=%s error_type=%s response_id=%s model=%s stream=%s tool_enabled=%s usage=%s",
                        attempt_no,
                        duration_ms,
                        None,
                        getattr(last_response, "id", None) if last_response else None,
                        payloads.get("model"),
                        True,
                        bool(func_tool),
                        usage_summary,
                    )
                    if last_response is not None:
                        self._record_turn_success(payloads, last_response)
                    break
                except Exception as e:
                    last_exception = e
                    duration_ms = int((time.perf_counter() - attempt_start) * 1000)
                    logger.warning(
                        "Responses stream failed. attempt=%s duration_ms=%s error_type=%s status_code=%s model=%s stream=%s tool_enabled=%s",
                        attempt_no,
                        duration_ms,
                        error_type(e),
                        getattr(e, "status_code", None),
                        payloads.get("model"),
                        True,
                        bool(func_tool),
                    )
                    if yielded_any:
                        raise
                    (
                        _,
                        chosen_key,
                        available_api_keys,
                        payloads,
                        context_query,
                        func_tool,
                        image_fallback_used,
                    ) = await self._handle_api_error(
                        e,
                        payloads,
                        context_query,
                        func_tool,
                        chosen_key,
                        available_api_keys,
                        retry_cnt,
                        max_retries,
                        image_fallback_used=image_fallback_used,
                    )

            if retry_cnt == max_retries - 1:
                logger.error(
                    "Responses API streaming call failed after %s retries.",
                    max_retries,
                )
                if last_exception is None:
                    raise Exception("Unknown error")
                raise last_exception
            return

        max_retries = 10
        available_api_keys = self.api_keys.copy()
        chosen_key = random.choice(available_api_keys) if available_api_keys else ""
        image_fallback_used = False

        tool_fallback_remaining = (
            self._tool_fallback_retry_attempts() if allow_retry else 0
        )
        tool_choice_override: str | dict[str, Any] | None = None
        last_exception: Exception | None = None
        retry_cnt = 0
        for retry_cnt in range(max_retries):
            attempt_no = retry_cnt + 1
            attempt_start = time.perf_counter()
            try:
                released = False
                yielded_any = False
                last_response: LLMResponse | None = None
                buffered_text = ""
                buffered_responses: list[LLMResponse] = []
                max_buffer_chars = int(
                    self.provider_config.get("stream_buffer_max_chars", 20_000)
                )
                max_buffer_responses = int(
                    self.provider_config.get("stream_buffer_max_responses", 512)
                )
                if strict_tool_call or codex_disable_pseudo:
                    # Buffer until stream end to avoid leaking pseudo-call text.
                    release_after_chars = max_buffer_chars + 1
                else:
                    release_after_chars = 64

                self.chosen_api_key = chosen_key
                async for response in self._query_stream(
                    payloads,
                    func_tool,
                    api_key=chosen_key,
                    reasoning_effort=reasoning_effort,
                    response_format=response_format,
                    tool_choice_override=tool_choice_override,
                ):
                    if released:
                        if response.is_chunk:
                            if looks_like_pseudo_tool_call_text(response):
                                raise Exception(
                                    "Pseudo tool-call text detected after streaming started; abort to avoid leakage."
                                )
                            yielded_any = True
                            yield response
                            continue

                        if looks_like_pseudo_tool_call_text(response):
                            if not allow_parse:
                                raise Exception(
                                    "Pseudo tool-call text detected but tool parsing is disabled; abort to avoid leakage."
                                )
                            converted = maybe_convert_pseudo_tool_calls(
                                response, tools=func_tool
                            )
                            if converted is None:
                                raise Exception(
                                    "Pseudo tool-call text detected but failed to convert; abort to avoid leakage."
                                )
                            response = converted

                        yielded_any = True
                        last_response = response
                        yield response
                        continue

                    buffered_responses.append(response)
                    if response.is_chunk and isinstance(response.completion_text, str):
                        buffered_text += response.completion_text

                    pseudo_suspected = False
                    if buffered_text:
                        preview = buffered_text[-256:]
                        pseudo_suspected = looks_like_pseudo_tool_call_text(
                            LLMResponse(role="assistant", completion_text=preview)
                        )

                    if (
                        len(buffered_text) > max_buffer_chars
                        or len(buffered_responses) > max_buffer_responses
                    ):
                        if pseudo_suspected:
                            raise Exception(
                                "Stream buffer exceeded while pseudo tool-call is suspected; abort to avoid leakage."
                            )
                        for item in buffered_responses:
                            yielded_any = True
                            last_response = item
                            yield item
                        buffered_responses = []
                        released = True
                        continue

                    if pseudo_suspected:
                        # Keep buffering to ensure we don't leak any pseudo-call prefix.
                        continue

                    if len(buffered_text) >= release_after_chars:
                        for item in buffered_responses:
                            yielded_any = True
                            last_response = item
                            yield item
                        buffered_responses = []
                        released = True

                if not buffered_responses:
                    if not yielded_any:
                        raise Exception("Empty responses stream")
                    duration_ms = int((time.perf_counter() - attempt_start) * 1000)
                    log_usage_enabled = self._coerce_bool(
                        self.provider_config.get("log_usage", False), False
                    )
                    usage_summary = None
                    if log_usage_enabled and last_response and last_response.usage is not None:
                        usage_summary = {
                            "input_other": last_response.usage.input_other,
                            "input_cached": last_response.usage.input_cached,
                            "output": last_response.usage.output,
                        }
                    logger.info(
                        "Responses stream succeeded. attempt=%s duration_ms=%s error_type=%s response_id=%s model=%s stream=%s tool_enabled=%s usage=%s",
                        attempt_no,
                        duration_ms,
                        None,
                        getattr(last_response, "id", None) if last_response else None,
                        payloads.get("model"),
                        True,
                        bool(func_tool),
                        usage_summary,
                    )
                    if last_response is not None:
                        self._record_turn_success(payloads, last_response)
                    return

                final_response = buffered_responses[-1]
                if (
                    codex_mode
                    and tools_available
                    and looks_like_pseudo_tool_call_text(final_response)
                    and not final_response.tools_call_name
                ):
                    raise Exception(
                        "Codex 模式检测到伪工具调用文本，已中止以避免 JSON 泄露。"
                    )

                if allow_parse and looks_like_pseudo_tool_call_text(final_response):
                    converted = maybe_convert_pseudo_tool_calls(
                        final_response, tools=func_tool
                    )
                    if converted is not None:
                        final_response = converted
                        # Drop buffered text chunks to avoid leaking pseudo-call text.
                        buffered_responses = [final_response]

                if (
                    allow_retry
                    and tool_fallback_remaining > 0
                    and looks_like_pseudo_tool_call_text(final_response)
                    and not final_response.tools_call_name
                ):
                    tool_fallback_remaining -= 1
                    tool_choice_override = self._tool_fallback_force_tool_choice()
                    logger.warning(
                        "Pseudo tool-call text detected in stream, retrying with tool_choice=%s. remaining=%s",
                        tool_choice_override,
                        tool_fallback_remaining,
                    )
                    continue

                if buffered_responses and buffered_responses[-1] is not final_response:
                    buffered_responses[-1] = final_response

                for item in buffered_responses:
                    last_response = item
                    yield item
                duration_ms = int((time.perf_counter() - attempt_start) * 1000)
                log_usage_enabled = self._coerce_bool(
                    self.provider_config.get("log_usage", False), False
                )
                usage_summary = None
                if log_usage_enabled and last_response and last_response.usage is not None:
                    usage_summary = {
                        "input_other": last_response.usage.input_other,
                        "input_cached": last_response.usage.input_cached,
                        "output": last_response.usage.output,
                    }
                logger.info(
                    "Responses stream succeeded. attempt=%s duration_ms=%s error_type=%s response_id=%s model=%s stream=%s tool_enabled=%s usage=%s",
                    attempt_no,
                    duration_ms,
                    None,
                    getattr(last_response, "id", None) if last_response else None,
                    payloads.get("model"),
                    True,
                    bool(func_tool),
                    usage_summary,
                )
                if last_response is not None:
                    self._record_turn_success(payloads, last_response)
                return
            except Exception as e:
                last_exception = e
                duration_ms = int((time.perf_counter() - attempt_start) * 1000)
                logger.warning(
                    "Responses stream failed. attempt=%s duration_ms=%s error_type=%s status_code=%s model=%s stream=%s tool_enabled=%s",
                    attempt_no,
                    duration_ms,
                    error_type(e),
                    getattr(e, "status_code", None),
                    payloads.get("model"),
                    True,
                    bool(func_tool),
                )
                if yielded_any:
                    raise
                (
                    _,
                    chosen_key,
                    available_api_keys,
                    payloads,
                    context_query,
                    func_tool,
                    image_fallback_used,
                ) = await self._handle_api_error(
                    e,
                    payloads,
                    context_query,
                    func_tool,
                    chosen_key,
                    available_api_keys,
                    retry_cnt,
                    max_retries,
                    image_fallback_used=image_fallback_used,
                )

        if retry_cnt == max_retries - 1:
            logger.error(
                "Responses API streaming call failed after %s retries.", max_retries
            )
            if last_exception is None:
                raise Exception("Unknown error")
            raise last_exception

    def get_current_key(self) -> str:
        return self.chosen_api_key

    def get_keys(self) -> list[str]:
        return self.api_keys

    def set_key(self, key: str) -> None:
        self.chosen_api_key = key

    async def get_models(self) -> list[str]:
        url = self.api_base.rstrip("/") + "/models"
        client = self._get_http_client()
        headers = self._build_headers(self.chosen_api_key)

        try:
            resp = await client.get(
                url, headers=headers, timeout=httpx.Timeout(float(self.timeout))
            )
        except Exception as exc:
            raise Exception(f"获取模型列表失败：{exc}") from exc

        if resp.status_code >= 400:
            raise Exception(f"获取模型列表失败：HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise Exception(f"获取模型列表失败：unexpected response: {data}")

        model_ids: list[str] = []
        for item in data["data"]:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                model_ids.append(item["id"])
        return sorted(model_ids)

    async def assemble_context(
        self,
        text: str,
        image_urls: list[str] | None = None,
        extra_user_content_parts: list[ContentPart] | None = None,
    ) -> dict:
        """Assemble an OpenAI-style user message, with multimodal parts if needed."""

        async def resolve_image_part(image_url: str) -> dict | None:
            if image_url.startswith("http"):
                image_path = await download_image_by_url(image_url)
                image_data = await self.encode_image_bs64(image_path)
            elif image_url.startswith("file:///"):
                image_path = image_url.replace("file:///", "")
                image_data = await self.encode_image_bs64(image_path)
            else:
                image_data = await self.encode_image_bs64(image_url)
            if not image_data:
                logger.warning("Empty image data for %s, skipped.", image_url)
                return None
            return {"type": "image_url", "image_url": {"url": image_data}}

        content_blocks: list[dict[str, Any]] = []

        if text:
            content_blocks.append({"type": "text", "text": text})
        elif image_urls:
            content_blocks.append({"type": "text", "text": "[图片]"})
        elif extra_user_content_parts:
            content_blocks.append({"type": "text", "text": " "})

        if extra_user_content_parts:
            for part in extra_user_content_parts:
                if isinstance(part, TextPart):
                    content_blocks.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageURLPart):
                    image_part = await resolve_image_part(part.image_url.url)
                    if image_part:
                        content_blocks.append(image_part)
                else:
                    raise ValueError(f"Unsupported extra content part: {type(part)}")

        if image_urls:
            for image_url in image_urls:
                image_part = await resolve_image_part(image_url)
                if image_part:
                    content_blocks.append(image_part)

        if (
            text
            and not extra_user_content_parts
            and not image_urls
            and len(content_blocks) == 1
            and content_blocks[0]["type"] == "text"
        ):
            return {"role": "user", "content": content_blocks[0]["text"]}

        return {"role": "user", "content": content_blocks}

    async def encode_image_bs64(self, image_url: str) -> str:
        if image_url.startswith("base64://"):
            return image_url.replace("base64://", "data:image/jpeg;base64,")
        with open(image_url, "rb") as f:
            image_bs64 = base64.b64encode(f.read()).decode("utf-8")
            return "data:image/jpeg;base64," + image_bs64

    async def terminate(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
