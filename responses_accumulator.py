from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import astrbot.core.message.components as Comp
from astrbot import logger
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, TokenUsage

try:
    from responses_redaction import redact_event_data
except ImportError:  # pragma: no cover
    from .responses_redaction import redact_event_data


class ResponsesStreamAccumulator:
    """Accumulates OpenAI Responses SSE events into AstrBot LLMResponse objects."""

    def __init__(
        self,
        model: str,
        *,
        debug_events: bool = False,
        max_text_chars: int = 200_000,
    ) -> None:
        self.model = model
        self.debug_events = debug_events
        self.max_text_chars = max(0, int(max_text_chars))
        self.truncated = False
        self.response_id: str | None = None
        self.accumulated_text = ""
        self.accumulated_refusal = ""
        self.accumulated_reasoning = ""
        self.usage: dict[str, Any] | None = None

        self.tool_calls_by_id: dict[str, ToolCallDraft] = {}
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

        remaining = self.max_text_chars - len(self.accumulated_text)
        if remaining <= 0:
            self.truncated = True
            return

        if len(text) > remaining:
            self.truncated = True
            text = text[:remaining]

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

    def _align_authoritative_text(self, final_text: str) -> None:
        if not final_text:
            return
        if self.max_text_chars > 0 and len(final_text) > self.max_text_chars:
            self.truncated = True
            final_text = final_text[: self.max_text_chars]
        if not self.accumulated_text:
            self.accumulated_text = final_text
            return
        if final_text == self.accumulated_text:
            return
        if final_text.startswith(self.accumulated_text):
            missing = final_text[len(self.accumulated_text) :]
            if missing:
                self.accumulated_text += missing
            return
        if self.accumulated_text.startswith(final_text):
            # Drop duplicated suffix produced by out-of-order events.
            self.accumulated_text = final_text
            return
        if len(final_text) >= len(self.accumulated_text):
            self.accumulated_text = final_text

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if isinstance(content, str):
            return content
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
        output_text = response_obj.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        text_parts: list[str] = []
        output = response_obj.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                text = cls._extract_text_from_item(item)
                if text:
                    text_parts.append(text)
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
        remaining = self.max_text_chars - len(self.accumulated_refusal)
        if remaining <= 0:
            self.truncated = True
            return
        if len(text) > remaining:
            self.truncated = True
            text = text[:remaining]
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

    def _ensure_tool_call(self, call_id: str) -> "ToolCallDraft":
        tool_call = self.tool_calls_by_id.get(call_id)
        if tool_call is None:
            tool_call = ToolCallDraft(call_id=call_id)
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
            tool_call.name = name
        if arguments:
            tool_call.arguments_text = arguments
        if item_id:
            self.tool_call_item_map[item_id] = call_id
        if output_index is not None:
            self.tool_call_output_index_map[output_index] = call_id
        self._mark_tool_call_active(call_id)

    def _append_tool_call_arguments(self, call_id: str, delta_args: str) -> None:
        if not delta_args:
            return
        tool_call = self._ensure_tool_call(call_id)
        tool_call.arguments_text = f"{tool_call.arguments_text}{delta_args}"
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
            tool_call.arguments_text = arguments
        tool_call.status = "completed"
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

        if event_type == "response.output_text.done":
            text = event_data.get("text")
            if isinstance(text, str) and text:
                self._align_authoritative_text(text)
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

        if event_type == "response.content_part.done":
            part = event_data.get("part")
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    self._align_authoritative_text(text)
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
                remaining = self.max_text_chars - len(self.accumulated_reasoning)
                if remaining <= 0:
                    self.truncated = True
                    return chunks
                if len(delta_text) > remaining:
                    self.truncated = True
                    delta_text = delta_text[:remaining]
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
                    redact_event_data(event_type, event_data, max_len=512),
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
                    redact_event_data(event_type, event_data, max_len=512),
                )
                return chunks
            arguments = event_data.get("arguments")
            if not isinstance(arguments, str):
                arguments = None
            self._finalize_tool_call(call_id=call_id, arguments=arguments)
            return chunks

        if event_type in {"response.completed", "response.done"}:
            response_obj = event_data.get("response")
            if isinstance(response_obj, dict):
                usage = response_obj.get("usage")
                if isinstance(usage, dict):
                    self.usage = usage
                self._ingest_tool_calls_from_response(response_obj)

                final_text = self._extract_text_from_response(response_obj)
                if isinstance(final_text, str) and final_text:
                    self._align_authoritative_text(final_text)

                if not self.accumulated_refusal:
                    refusal_text = self._extract_refusal_text_from_response(response_obj)
                    if refusal_text:
                        self.accumulated_refusal = refusal_text

            usage = event_data.get("usage")
            if isinstance(usage, dict):
                self.usage = usage
            return chunks

        if event_type == "response.in_progress":
            response_obj = event_data.get("response")
            if isinstance(response_obj, dict) and not self.response_id:
                rid = response_obj.get("id")
                if isinstance(rid, str) and rid:
                    self.response_id = rid
            return chunks

        if event_type == "response.failed":
            response_obj = event_data.get("response")
            if isinstance(response_obj, dict):
                self._ingest_tool_calls_from_response(response_obj)
                final_text = self._extract_text_from_response(response_obj)
                if isinstance(final_text, str) and final_text:
                    self._align_authoritative_text(final_text)
                if not self.accumulated_refusal:
                    refusal_text = self._extract_refusal_text_from_response(response_obj)
                    if refusal_text:
                        self.accumulated_refusal = refusal_text
            return chunks

        if self.debug_events:
            logger.debug(
                "Ignored Responses SSE event. type=%s data=%s",
                event_type,
                redact_event_data(event_type, event_data, max_len=512),
            )
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
                call_id = str(tool_call.call_id or f"call_{uuid.uuid4().hex[:24]}")
                name = str(tool_call.name or "")
                raw_arguments = tool_call.arguments_text or "{}"
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
        if self.truncated:
            logger.warning(
                "Responses output truncated by max_output_chars=%s. response_id=%s",
                self.max_text_chars,
                self.response_id,
            )
        llm_response.raw_completion = {
            "id": self.response_id,
            "usage": self.usage,
            "tool_calls": [asdict(call) for call in ordered_tool_calls],
            "refusal": self.accumulated_refusal,
            "truncated": self.truncated,
            "max_output_chars": self.max_text_chars,
        }
        return llm_response


@dataclass
class ToolCallDraft:
    call_id: str
    name: str = ""
    arguments_text: str = ""
    status: str = "in_progress"
