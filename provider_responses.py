from __future__ import annotations

import asyncio
import base64
import json
import random
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
        self.accumulated_reasoning = ""
        self.usage: dict[str, Any] | None = None

        self.tool_calls: list[dict[str, Any]] = []
        self.current_tool_call: dict[str, Any] | None = None

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
                self.accumulated_text += delta_text
                chunks.append(
                    LLMResponse(
                        role="assistant",
                        result_chain=MessageChain(chain=[Comp.Plain(delta_text)]),
                        is_chunk=True,
                        id=self.response_id,
                    )
                )
            return chunks

        if event_type == "response.content_part.delta":
            delta = event_data.get("delta")
            if isinstance(delta, dict):
                text = delta.get("text")
                if isinstance(text, str) and text:
                    self.accumulated_text += text
                    chunks.append(
                        LLMResponse(
                            role="assistant",
                            result_chain=MessageChain(chain=[Comp.Plain(text)]),
                            is_chunk=True,
                            id=self.response_id,
                        )
                    )
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
                self.current_tool_call = {
                    "id": item.get("call_id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": "",
                    },
                }
            return chunks

        if event_type == "response.function_call_arguments.delta":
            if self.current_tool_call is None:
                return chunks
            delta_args = event_data.get("delta")
            if isinstance(delta_args, str) and delta_args:
                self.current_tool_call["function"]["arguments"] += delta_args
            return chunks

        if event_type == "response.function_call_arguments.done":
            if self.current_tool_call is not None:
                self.tool_calls.append(self.current_tool_call)
                self.current_tool_call = None
            return chunks

        if event_type in {"response.completed", "response.done"}:
            response_obj = event_data.get("response")
            if isinstance(response_obj, dict) and isinstance(
                response_obj.get("usage"), dict
            ):
                self.usage = response_obj["usage"]
            elif isinstance(event_data.get("usage"), dict):
                self.usage = event_data["usage"]
            return chunks

        return chunks

    def build_final_response(self, *, tools_provided: bool) -> LLMResponse:
        llm_response = LLMResponse(role="assistant")
        llm_response.id = self.response_id

        if self.accumulated_text:
            llm_response.result_chain = MessageChain().message(self.accumulated_text)

        if self.accumulated_reasoning:
            llm_response.reasoning_content = self.accumulated_reasoning

        if self.tool_calls:
            if not tools_provided:
                raise Exception("工具集未提供")
            tool_args: list[dict[str, Any]] = []
            tool_names: list[str] = []
            tool_ids: list[str] = []
            for tool_call in self.tool_calls:
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
            "tool_calls": self.tool_calls,
        }
        return llm_response


class ProviderOpenAIResponsesPlugin(AstrProvider):
    """OpenAI Responses API provider adapter for plugin distribution."""

    _ERROR_TEXT_CANDIDATE_MAX_CHARS = 4096

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

        self.api_base = str(provider_config.get("api_base", "https://api.openai.com/v1"))
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
                        converted.append({"type": "input_text", "text": str(part.get("refusal") or part.get("text") or "")})
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
                        converted.append({"type": "input_image", "image_url": str(image_url)})
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
                        logger.warning("tool_call name is empty, inferred as: %s", func_name)

                    arguments = func.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False, default=str)

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
                    output = content
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

        effort_source = reasoning_effort
        if effort_source is None:
            effort_source = self.provider_config.get("reasoning_effort")

        normalized_effort = self._normalize_reasoning_effort(effort_source)
        if normalized_effort is not None:
            request_body["reasoning"] = {"effort": normalized_effort}

        if response_format is not None:
            fmt_type = response_format.get("type") if isinstance(response_format, dict) else None
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
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    event_data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Responses SSE JSON decode failed: %s", data_str[:200])
                    continue

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
                    continue
                event_type = current_event_type or str(event_data.get("type") or "")
                if not event_type:
                    continue
                yield event_type, event_data

    async def _query_stream(
        self,
        payloads: dict[str, Any],
        tools: ToolSet | None,
        *,
        api_key: str,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        request_body = self._build_responses_request(
            payloads,
            tools,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
        )
        processor = _ResponsesStreamProcessor(model=str(payloads.get("model") or self.get_model()))

        async for event_type, event_data in self._iter_responses_sse(
            request_body=request_body,
            api_key=api_key,
        ):
            for chunk in processor.handle_event(event_type, event_data):
                yield chunk

        yield processor.build_final_response(tools_provided=tools is not None)

    def _finally_convert_payload(self, payloads: dict) -> None:
        for message in payloads.get("messages", []):
            if message.get("role") == "assistant" and isinstance(message.get("content"), list):
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

    def _remove_images_from_context(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    ) -> tuple[bool, str, list[str], dict[str, Any], list[dict[str, Any]], ToolSet | None, bool]:
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

        if "The model is not a VLM" in joined or self._is_content_moderated_upload_error(e):
            if image_fallback_used or not self._context_contains_image(context_query):
                raise e
            logger.warning("Image input rejected by upstream, retrying with text-only context.")
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

        llm_response: LLMResponse | None = None
        max_retries = 10
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
                    reasoning_effort=kwargs.get("reasoning_effort"),
                    response_format=kwargs.get("response_format"),
                ):
                    last_chunk = resp
                if last_chunk is None:
                    raise Exception("Empty responses stream")
                llm_response = last_chunk
                break
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

        if retry_cnt == max_retries - 1 or llm_response is None:
            logger.error("Responses API call failed after %s retries.", max_retries)
            if last_exception is None:
                raise Exception("Unknown error")
            raise last_exception
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

        max_retries = 10
        available_api_keys = self.api_keys.copy()
        chosen_key = random.choice(available_api_keys) if available_api_keys else ""
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
                    reasoning_effort=kwargs.get("reasoning_effort"),
                    response_format=kwargs.get("response_format"),
                ):
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
            logger.error("Responses API streaming call failed after %s retries.", max_retries)
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
