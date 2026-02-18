from __future__ import annotations

import asyncio
import base64
import json
import random
import re
import uuid
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import httpx

import astrbot.core.message.components as Comp
from astrbot import logger
from astrbot.api.provider import Provider as AstrProvider
from astrbot.core.agent.message import ContentPart, ImageURLPart, TextPart
from astrbot.core.agent.tool import ToolSet
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, TokenUsage, ToolCallsResult
from astrbot.core.utils.io import download_image_by_url
from astrbot.core.utils.network_utils import (
    create_proxy_client,
    is_connection_error,
    log_connection_failure,
)


class _UpstreamResponsesError(Exception):
    """A lightweight error wrapper with fields used by existing handlers."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: dict[str, Any] | str | None = None,
        response_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        if response_text is None:
            if isinstance(body, dict):
                response_text = json.dumps(body, ensure_ascii=False, default=str)
            else:
                response_text = "" if body is None else str(body)
        self.response = SimpleNamespace(text=response_text)


class _ResponsesStreamProcessor:
    """Accumulates OpenAI Responses SSE events into AstrBot LLMResponse objects."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.response_id: str | None = None
        self.accumulated_text = ""
        self.accumulated_refusal = ""
        self.accumulated_reasoning = ""
        self.usage: dict[str, Any] | None = None

        self.tool_calls_by_id: dict[str, dict[str, Any]] = {}
        self.tool_call_order: list[str] = []
        self.active_tool_call_ids: list[str] = []
        self.tool_call_item_map: dict[str, str] = {}
        self.tool_call_output_index_map: dict[int, str] = {}

    @staticmethod
    def _convert_usage_to_token_usage(
        usage: dict[str, Any] | None,
    ) -> TokenUsage | None:
        if not usage:
            return None
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cached_tokens = 0
        details = usage.get("input_tokens_details")
        if isinstance(details, dict):
            cached_tokens = int(details.get("cached_tokens") or 0)
        return TokenUsage(
            input_other=max(input_tokens - cached_tokens, 0),
            input_cached=max(cached_tokens, 0),
            output=max(output_tokens, 0),
        )

    @staticmethod
    def _normalize_output_index(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    def _append_text_chunk(
        self,
        chunks: list[LLMResponse],
        text: str,
        *,
        stream_chunk: bool = True,
    ) -> None:
        if not text:
            return
        self.accumulated_text += text
        if stream_chunk:
            chunks.append(
                LLMResponse(
                    role="assistant",
                    result_chain=MessageChain(chain=[Comp.Plain(text)]),
                    is_chunk=True,
                    id=self.response_id,
                )
            )

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if not isinstance(content, list):
            return ""
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "")
            if part_type == "refusal":
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        return "".join(text_parts)

    @classmethod
    def _extract_text_from_item(cls, item: dict[str, Any]) -> str:
        text_parts: list[str] = []
        item_type = str(item.get("type") or "")
        if item_type in {"output_text", "text"}:
            text = item.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        content = item.get("content")
        content_text = cls._extract_text_from_content(content)
        if content_text:
            text_parts.append(content_text)
        return "".join(text_parts)

    @classmethod
    def _extract_text_from_response(cls, response_obj: dict[str, Any]) -> str:
        text_parts: list[str] = []
        output = response_obj.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                text = cls._extract_text_from_item(item)
                if text:
                    text_parts.append(text)
        output_text = response_obj.get("output_text")
        if isinstance(output_text, str) and output_text:
            text_parts.append(output_text)
        return "".join(text_parts)

    def _append_refusal_text(
        self,
        chunks: list[LLMResponse],
        text: str,
        *,
        stream_chunk: bool = True,
    ) -> None:
        if not text:
            return
        self.accumulated_refusal += text
        if stream_chunk:
            chunks.append(
                LLMResponse(
                    role="assistant",
                    result_chain=MessageChain(chain=[Comp.Plain(text)]),
                    is_chunk=True,
                    id=self.response_id,
                )
            )

    def _ensure_tool_call(self, call_id: str) -> dict[str, Any]:
        tool_call = self.tool_calls_by_id.get(call_id)
        if tool_call is None:
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            }
            self.tool_calls_by_id[call_id] = tool_call
            self.tool_call_order.append(call_id)
        return tool_call

    def _mark_tool_call_active(self, call_id: str) -> None:
        if call_id in self.active_tool_call_ids:
            self.active_tool_call_ids.remove(call_id)
        self.active_tool_call_ids.append(call_id)

    def _mark_tool_call_done(self, call_id: str) -> None:
        if call_id in self.active_tool_call_ids:
            self.active_tool_call_ids.remove(call_id)

    def _register_tool_call(
        self,
        *,
        call_id: str,
        name: str,
        item_id: str | None = None,
        output_index: int | None = None,
        arguments: str | None = None,
    ) -> None:
        tool_call = self._ensure_tool_call(call_id)
        if name:
            tool_call["function"]["name"] = name
        if arguments:
            tool_call["function"]["arguments"] = arguments
        if item_id:
            self.tool_call_item_map[item_id] = call_id
        if output_index is not None:
            self.tool_call_output_index_map[output_index] = call_id
        self._mark_tool_call_active(call_id)

    def _append_tool_call_arguments(self, call_id: str, delta_args: str) -> None:
        if not delta_args:
            return
        tool_call = self._ensure_tool_call(call_id)
        curr_args = tool_call["function"].get("arguments", "")
        tool_call["function"]["arguments"] = f"{curr_args}{delta_args}"
        self._mark_tool_call_active(call_id)

    def _resolve_tool_call_id(self, event_data: dict[str, Any]) -> str | None:
        call_id = event_data.get("call_id")
        if isinstance(call_id, str) and call_id:
            return call_id

        item_id = event_data.get("item_id")
        if isinstance(item_id, str) and item_id:
            mapped_call_id = self.tool_call_item_map.get(item_id)
            if mapped_call_id:
                return mapped_call_id

        output_index = self._normalize_output_index(event_data.get("output_index"))
        if output_index is not None:
            mapped_call_id = self.tool_call_output_index_map.get(output_index)
            if mapped_call_id:
                return mapped_call_id

        candidate_id = event_data.get("id")
        if isinstance(candidate_id, str) and candidate_id in self.tool_calls_by_id:
            return candidate_id

        if self.active_tool_call_ids:
            return self.active_tool_call_ids[-1]
        if self.tool_call_order:
            return self.tool_call_order[-1]
        return None

    def _finalize_tool_call(
        self,
        *,
        call_id: str,
        arguments: str | None = None,
    ) -> None:
        tool_call = self._ensure_tool_call(call_id)
        if isinstance(arguments, str) and arguments:
            tool_call["function"]["arguments"] = arguments
        self._mark_tool_call_done(call_id)

    @staticmethod
    def _extract_refusal_text_from_content(content: Any) -> str:
        if not isinstance(content, list):
            return ""
        refusal_text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "")
            if part_type != "refusal" and "refusal" not in part:
                continue
            refusal = part.get("refusal")
            if isinstance(refusal, str) and refusal:
                refusal_text_parts.append(refusal)
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                refusal_text_parts.append(text)
        return "".join(refusal_text_parts)

    @classmethod
    def _extract_refusal_text_from_item(cls, item: dict[str, Any]) -> str:
        refusal_text_parts: list[str] = []
        item_type = str(item.get("type") or "")
        if item_type == "refusal" or "refusal" in item:
            refusal = item.get("refusal")
            if isinstance(refusal, str) and refusal:
                refusal_text_parts.append(refusal)
            else:
                text = item.get("text")
                if isinstance(text, str) and text:
                    refusal_text_parts.append(text)

        content = item.get("content")
        refusal_from_content = cls._extract_refusal_text_from_content(content)
        if refusal_from_content:
            refusal_text_parts.append(refusal_from_content)

        return "".join(refusal_text_parts)

    @classmethod
    def _extract_refusal_text_from_response(cls, response_obj: dict[str, Any]) -> str:
        refusal_text_parts: list[str] = []
        output = response_obj.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                refusal_text = cls._extract_refusal_text_from_item(item)
                if refusal_text:
                    refusal_text_parts.append(refusal_text)
        return "".join(refusal_text_parts)

    def _ingest_tool_calls_from_response(self, response_obj: dict[str, Any]) -> None:
        output = response_obj.get("output")
        if not isinstance(output, list):
            return
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call":
                continue

            item_id = item.get("id")
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                if isinstance(item_id, str) and item_id:
                    call_id = item_id
                else:
                    call_id = f"call_{uuid.uuid4().hex[:24]}"

            name = item.get("name")
            if not isinstance(name, str):
                name = ""

            output_index = self._normalize_output_index(item.get("output_index"))
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                arguments = None

            self._register_tool_call(
                call_id=call_id,
                name=name,
                item_id=item_id if isinstance(item_id, str) else None,
                output_index=output_index,
                arguments=arguments,
            )
            self._mark_tool_call_done(call_id)

    def handle_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
    ) -> list[LLMResponse]:
        chunks: list[LLMResponse] = []

        if event_type == "response.created":
            response_obj = event_data.get("response")
            if isinstance(response_obj, dict):
                rid = response_obj.get("id")
                if isinstance(rid, str) and rid:
                    self.response_id = rid
            return chunks

        if event_type == "response.output_text.delta":
            delta_text = event_data.get("delta")
            if isinstance(delta_text, str) and delta_text:
                self._append_text_chunk(chunks, delta_text)
            return chunks

        if event_type == "response.content_part.delta":
            delta = event_data.get("delta")
            if isinstance(delta, dict):
                text = delta.get("text")
                if isinstance(text, str) and text:
                    self._append_text_chunk(chunks, text)
                refusal = delta.get("refusal")
                if isinstance(refusal, str) and refusal:
                    self._append_refusal_text(chunks, refusal)
            return chunks

        if event_type in {"response.refusal.delta", "response.output_refusal.delta"}:
            delta_text = event_data.get("delta")
            if isinstance(delta_text, str) and delta_text:
                self._append_refusal_text(chunks, delta_text)
            return chunks

        if event_type in {"response.refusal.done", "response.output_refusal.done"}:
            if not self.accumulated_refusal:
                refusal = event_data.get("refusal")
                if isinstance(refusal, str) and refusal:
                    self._append_refusal_text(chunks, refusal, stream_chunk=False)
            return chunks

        if event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning.delta",
        }:
            delta_text = event_data.get("delta")
            if isinstance(delta_text, str) and delta_text:
                self.accumulated_reasoning += delta_text
                chunks.append(
                    LLMResponse(
                        role="assistant",
                        reasoning_content=delta_text,
                        is_chunk=True,
                        id=self.response_id,
                    )
                )
            return chunks

        if event_type == "response.output_item.added":
            item = event_data.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                item_id = item.get("id")
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    if isinstance(item_id, str) and item_id:
                        call_id = item_id
                    else:
                        call_id = f"call_{uuid.uuid4().hex[:24]}"
                name = item.get("name")
                if not isinstance(name, str):
                    name = ""
                output_index = self._normalize_output_index(item.get("output_index"))
                if output_index is None:
                    output_index = self._normalize_output_index(
                        event_data.get("output_index")
                    )
                arguments = item.get("arguments")
                if not isinstance(arguments, str):
                    arguments = None
                self._register_tool_call(
                    call_id=call_id,
                    name=name,
                    item_id=item_id if isinstance(item_id, str) else None,
                    output_index=output_index,
                    arguments=arguments,
                )
            return chunks

        if event_type == "response.output_item.done":
            item = event_data.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    item_id = item.get("id")
                    if isinstance(item_id, str) and item_id:
                        call_id = self.tool_call_item_map.get(item_id, item_id)
                    else:
                        call_id = self._resolve_tool_call_id(event_data)
                if isinstance(call_id, str) and call_id:
                    arguments = item.get("arguments")
                    if not isinstance(arguments, str):
                        arguments = None
                    self._finalize_tool_call(call_id=call_id, arguments=arguments)
            return chunks

        if event_type == "response.function_call_arguments.delta":
            call_id = self._resolve_tool_call_id(event_data)
            if call_id is None:
                logger.warning(
                    "Received function_call_arguments.delta but no tool call can be resolved. event=%s",
                    event_data,
                )
                return chunks
            delta_args = event_data.get("delta")
            if isinstance(delta_args, str) and delta_args:
                self._append_tool_call_arguments(call_id, delta_args)
            return chunks

        if event_type == "response.function_call_arguments.done":
            call_id = self._resolve_tool_call_id(event_data)
            if call_id is None:
                logger.warning(
                    "Received function_call_arguments.done but no tool call can be resolved. event=%s",
                    event_data,
                )
                return chunks
            arguments = event_data.get("arguments")
            if not isinstance(arguments, str):
                arguments = None
            self._finalize_tool_call(call_id=call_id, arguments=arguments)
            return chunks

        if event_type in {"response.completed", "response.done"}:
            response_obj = event_data.get("response")
            if isinstance(response_obj, dict) and isinstance(
                response_obj.get("usage"), dict
            ):
                self.usage = response_obj["usage"]
                self._ingest_tool_calls_from_response(response_obj)
                final_text = self._extract_text_from_response(response_obj)
                if final_text:
                    if not self.accumulated_text:
                        self._append_text_chunk(
                            chunks,
                            final_text,
                            stream_chunk=False,
                        )
                    elif final_text != self.accumulated_text:
                        if final_text.startswith(self.accumulated_text):
                            missing = final_text[len(self.accumulated_text) :]
                            if missing:
                                self._append_text_chunk(
                                    chunks,
                                    missing,
                                    stream_chunk=False,
                                )
                        elif len(final_text) > len(self.accumulated_text):
                            self.accumulated_text = final_text
                if not self.accumulated_refusal:
                    refusal_text = self._extract_refusal_text_from_response(
                        response_obj
                    )
                    if refusal_text:
                        self._append_refusal_text(
                            chunks,
                            refusal_text,
                            stream_chunk=False,
                        )
            elif isinstance(event_data.get("usage"), dict):
                self.usage = event_data["usage"]
            return chunks

        return chunks

    def build_final_response(self, *, tools_provided: bool) -> LLMResponse:
        llm_response = LLMResponse(role="assistant")
        llm_response.id = self.response_id

        if self.accumulated_text:
            llm_response.result_chain = MessageChain().message(self.accumulated_text)
        elif self.accumulated_refusal:
            llm_response.result_chain = MessageChain().message(self.accumulated_refusal)

        if self.accumulated_reasoning:
            llm_response.reasoning_content = self.accumulated_reasoning

        ordered_tool_calls = [
            self.tool_calls_by_id[call_id]
            for call_id in self.tool_call_order
            if call_id in self.tool_calls_by_id
        ]

        if ordered_tool_calls:
            if not tools_provided:
                raise Exception("工具集未提供")
            tool_args: list[dict[str, Any]] = []
            tool_names: list[str] = []
            tool_ids: list[str] = []
            for tool_call in ordered_tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}")
                func = tool_call.get("function")
                if not isinstance(func, dict):
                    continue
                name = str(func.get("name") or "")
                raw_arguments = func.get("arguments", "{}")
                if isinstance(raw_arguments, str):
                    try:
                        args_obj = json.loads(raw_arguments)
                        if not isinstance(args_obj, dict):
                            args_obj = {"_raw": raw_arguments}
                    except json.JSONDecodeError:
                        args_obj = {"_raw": raw_arguments}
                elif isinstance(raw_arguments, dict):
                    args_obj = raw_arguments
                else:
                    args_obj = {"_raw": str(raw_arguments)}

                tool_ids.append(call_id)
                tool_names.append(name)
                tool_args.append(args_obj)

            llm_response.role = "tool"
            llm_response.tools_call_ids = tool_ids
            llm_response.tools_call_name = tool_names
            llm_response.tools_call_args = tool_args

        llm_response.usage = self._convert_usage_to_token_usage(self.usage)
        llm_response.raw_completion = {
            "id": self.response_id,
            "usage": self.usage,
            "tool_calls": ordered_tool_calls,
            "refusal": self.accumulated_refusal,
        }
        return llm_response


class ProviderOpenAIResponsesPlugin(AstrProvider):
    """OpenAI Responses API provider adapter for plugin distribution."""

    _ERROR_TEXT_CANDIDATE_MAX_CHARS = 4096
    _TOOL_FALLBACK_MODES = {"parse_then_retry", "retry_only", "parse_only"}
    _PSEUDO_TOOL_CALL_RE = re.compile(
        r"assistant\s+to\s*=\s*functions\.([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    )

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        AstrProvider.__init__(self, provider_config, provider_settings)

        self.api_keys: list[str] = AstrProvider.get_keys(self)
        self.chosen_api_key = self.api_keys[0] if self.api_keys else ""

        self.timeout = provider_config.get("timeout", 120)
        if isinstance(self.timeout, str):
            self.timeout = int(self.timeout)

        self.custom_headers = provider_config.get("custom_headers", {})
        if not isinstance(self.custom_headers, dict) or not self.custom_headers:
            self.custom_headers = None
        else:
            for key in list(self.custom_headers.keys()):
                self.custom_headers[key] = str(self.custom_headers[key])

        self.api_base = str(
            provider_config.get("api_base", "https://api.openai.com/v1")
        )
        self.proxy = str(provider_config.get("proxy", "") or "")

        self._http_client: httpx.AsyncClient | None = None

        model = provider_config.get("model")
        if not model:
            model = "gpt-4o-mini"
        self.set_model(str(model))

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
        if not stripped or "\\" not in content:
            return content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return content
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False, default=str)
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
        if not self._extract_allowed_tool_names(tools):
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

    @staticmethod
    def _extract_allowed_tool_names(tools: ToolSet | None) -> set[str]:
        if tools is None:
            return set()
        names: set[str] = set()
        func_list = getattr(tools, "func_list", None)
        if not isinstance(func_list, list):
            return names
        for tool in func_list:
            name = getattr(tool, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
        return names

    @staticmethod
    def _extract_json_object_after_index(
        text: str, start_idx: int
    ) -> tuple[str | None, int]:
        obj_start: int | None = None
        depth = 0
        in_string = False
        escaping = False

        for idx in range(start_idx, len(text)):
            ch = text[idx]
            if obj_start is None:
                if ch == "{":
                    obj_start = idx
                    depth = 1
                continue

            if in_string:
                if escaping:
                    escaping = False
                    continue
                if ch == "\\":
                    escaping = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0 and obj_start is not None:
                    return text[obj_start : idx + 1], idx + 1

        return None, start_idx

    def _parse_pseudo_tool_calls(
        self,
        text: str,
        *,
        allowed_tool_names: set[str],
    ) -> list[dict[str, Any]]:
        parsed_calls: list[dict[str, Any]] = []
        if not text:
            return parsed_calls

        use_allowlist = bool(allowed_tool_names)
        for match in self._PSEUDO_TOOL_CALL_RE.finditer(text):
            tool_name = match.group(1)
            if use_allowlist and tool_name not in allowed_tool_names:
                continue
            raw_json, _ = self._extract_json_object_after_index(text, match.end())
            if not raw_json:
                continue
            try:
                arguments_obj = json.loads(raw_json)
            except json.JSONDecodeError:
                continue

            if not isinstance(arguments_obj, dict):
                arguments_obj = {"_raw": raw_json}

            parsed_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "name": tool_name,
                    "arguments": arguments_obj,
                    "raw_arguments": raw_json,
                }
            )
        return parsed_calls

    def _convert_pseudo_calls_to_llm_response(
        self,
        llm_response: LLMResponse,
        parsed_calls: list[dict[str, Any]],
    ) -> LLMResponse:
        converted = LLMResponse(role="tool", completion_text="")
        converted.id = llm_response.id
        converted.usage = llm_response.usage
        converted.reasoning_content = llm_response.reasoning_content
        converted.reasoning_signature = llm_response.reasoning_signature
        converted.tools_call_ids = [str(item["id"]) for item in parsed_calls]
        converted.tools_call_name = [str(item["name"]) for item in parsed_calls]
        converted.tools_call_args = [item["arguments"] for item in parsed_calls]

        raw_completion: dict[str, Any]
        if isinstance(llm_response.raw_completion, dict):
            raw_completion = dict(llm_response.raw_completion)
        else:
            raw_completion = {"upstream_raw_completion": llm_response.raw_completion}
        raw_completion["pseudo_tool_call_detected"] = True
        raw_completion["pseudo_tool_calls"] = [
            {
                "id": str(item["id"]),
                "name": str(item["name"]),
                "raw_arguments": str(item["raw_arguments"]),
            }
            for item in parsed_calls
        ]
        converted.raw_completion = raw_completion
        return converted

    def _maybe_parse_pseudo_tool_calls(
        self,
        llm_response: LLMResponse,
        tools: ToolSet | None,
    ) -> LLMResponse | None:
        if llm_response.tools_call_name:
            return None
        text = llm_response.completion_text
        if not isinstance(text, str) or not text.strip():
            return None
        allowed_tool_names = self._extract_allowed_tool_names(tools)
        parsed_calls = self._parse_pseudo_tool_calls(
            text,
            allowed_tool_names=allowed_tool_names,
        )
        if not parsed_calls and self._PSEUDO_TOOL_CALL_RE.search(text):
            if allowed_tool_names:
                parsed_calls = self._parse_pseudo_tool_calls(
                    text,
                    allowed_tool_names=set(),
                )
                if parsed_calls:
                    logger.warning(
                        "Detected pseudo tool-call text with undeclared tool names, converted without allow-list guard. tools=%s",
                        [item["name"] for item in parsed_calls],
                    )
            if not parsed_calls:
                logger.warning(
                    "Detected pseudo tool-call marker but failed to parse tool-call JSON arguments."
                )
        if not parsed_calls:
            return None
        logger.warning(
            "Detected pseudo tool-call text, converted to structured tool calls. tools=%s",
            [item["name"] for item in parsed_calls],
        )
        return self._convert_pseudo_calls_to_llm_response(llm_response, parsed_calls)

    def _looks_like_pseudo_tool_call_text(self, llm_response: LLMResponse) -> bool:
        if llm_response.tools_call_name:
            return False
        text = llm_response.completion_text
        if not isinstance(text, str) or not text.strip():
            return False
        return bool(self._PSEUDO_TOOL_CALL_RE.search(text))

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

        request_body: dict[str, Any] = {
            "model": model,
            "input": self._convert_chat_messages_to_responses_input(messages),
            "stream": True,
        }

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
                raise _UpstreamResponsesError(
                    f"Responses API HTTP {response.status_code}: {raw_text[:512]}",
                    status_code=response.status_code,
                    body=body,
                    response_text=raw_text,
                )

            current_event_type: str | None = None
            current_data_lines: list[str] = []

            def _parse_event_data(
                *,
                event_type_hint: str | None,
                data_lines: list[str],
            ) -> tuple[str, dict[str, Any]] | None:
                if not data_lines:
                    return None
                data_str = "\n".join(data_lines)
                if data_str == "[DONE]":
                    return ("[DONE]", {})
                try:
                    event_data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning(
                        "Responses SSE JSON decode failed: %s",
                        data_str[:200],
                    )
                    return None

                if isinstance(event_data, dict) and "error" in event_data:
                    err = event_data.get("error", {})
                    err_code = ""
                    err_msg = ""
                    if isinstance(err, dict):
                        err_code = str(err.get("code") or "")
                        err_msg = str(err.get("message") or "")
                    raise _UpstreamResponsesError(
                        f"Responses SSE error {err_code}: {err_msg}",
                        body=event_data,
                        response_text=data_str,
                    )

                if not isinstance(event_data, dict):
                    return None
                event_type = event_type_hint or str(event_data.get("type") or "")
                if not event_type:
                    return None
                return event_type, event_data

            def _flush_current_event() -> tuple[str, dict[str, Any]] | None:
                nonlocal current_event_type, current_data_lines
                parsed = _parse_event_data(
                    event_type_hint=current_event_type,
                    data_lines=current_data_lines,
                )
                current_event_type = None
                current_data_lines = []
                return parsed

            async for raw_line in response.aiter_lines():
                line = raw_line.rstrip("\r")
                if not line:
                    parsed = _flush_current_event()
                    if parsed is None:
                        continue
                    if parsed[0] == "[DONE]":
                        break
                    yield parsed
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    data_segment = line[5:]
                    if data_segment.startswith(" "):
                        data_segment = data_segment[1:]
                    current_data_lines.append(data_segment)
                    continue
                # Ignore non-data fields like id/retry.
                if line.startswith("id:") or line.startswith("retry:"):
                    continue

            parsed = _flush_current_event()
            if parsed and parsed[0] != "[DONE]":
                yield parsed

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
        processor = _ResponsesStreamProcessor(
            model=str(payloads.get("model") or self.get_model())
        )

        async for event_type, event_data in self._iter_responses_sse(
            request_body=request_body,
            api_key=api_key,
        ):
            for chunk in processor.handle_event(event_type, event_data):
                yield chunk

        yield processor.build_final_response(tools_provided=tools is not None)

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
        payloads: dict[str, Any] = {"messages": context_query, "model": model}
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
        if status_code is None and isinstance(e, _UpstreamResponsesError):
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
                await asyncio.sleep(1)
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
            await self.pop_record(context_query)
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
            log_connection_failure("OpenAI", e, self.proxy)
            if retry_cnt < max_retries - 1:
                await asyncio.sleep(1)
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
                return last_chunk, func_tool
            except Exception as e:
                last_exception = e
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

        has_tool_context = effective_func_tool is not None
        tool_fallback_enabled = self._tool_fallback_enabled(effective_func_tool)
        tool_fallback_mode = self._tool_fallback_mode() if tool_fallback_enabled else ""
        allow_parse = has_tool_context
        allow_retry = tool_fallback_enabled and tool_fallback_mode in {
            "parse_then_retry",
            "retry_only",
        }

        if allow_parse:
            converted = self._maybe_parse_pseudo_tool_calls(
                llm_response,
                effective_func_tool,
            )
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
            and self._looks_like_pseudo_tool_call_text(llm_response)
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
            if allow_parse:
                converted = self._maybe_parse_pseudo_tool_calls(
                    llm_response,
                    effective_func_tool,
                )
                if converted is not None:
                    llm_response = converted
            if llm_response.tools_call_name:
                break

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
        has_tool_context = func_tool is not None
        tool_fallback_enabled = self._tool_fallback_enabled(func_tool)
        tool_fallback_mode = self._tool_fallback_mode() if tool_fallback_enabled else ""
        allow_parse = has_tool_context
        allow_retry = tool_fallback_enabled and tool_fallback_mode in {
            "parse_then_retry",
            "retry_only",
        }
        stream_buffer_enabled = (
            has_tool_context and self._tool_fallback_stream_buffer_enabled()
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
                yielded_any = False
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
                        ):
                            converted = self._maybe_parse_pseudo_tool_calls(
                                response,
                                func_tool,
                            )
                            if converted is not None:
                                response = converted
                        yielded_any = True
                        yield response
                    break
                except Exception as e:
                    last_exception = e
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
            buffered_responses: list[LLMResponse] = []
            try:
                self.chosen_api_key = chosen_key
                async for response in self._query_stream(
                    payloads,
                    func_tool,
                    api_key=chosen_key,
                    reasoning_effort=reasoning_effort,
                    response_format=response_format,
                    tool_choice_override=tool_choice_override,
                ):
                    buffered_responses.append(response)

                if not buffered_responses:
                    raise Exception("Empty responses stream")

                final_response = buffered_responses[-1]
                if allow_parse:
                    converted = self._maybe_parse_pseudo_tool_calls(
                        final_response,
                        func_tool,
                    )
                    if converted is not None:
                        final_response = converted
                        # Drop buffered text chunks to avoid leaking pseudo-call text.
                        buffered_responses = [final_response]

                if (
                    allow_retry
                    and tool_fallback_remaining > 0
                    and self._looks_like_pseudo_tool_call_text(final_response)
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

                for response in buffered_responses:
                    yield response
                return
            except Exception as e:
                last_exception = e
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
