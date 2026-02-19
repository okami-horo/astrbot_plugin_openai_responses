from __future__ import annotations

import json
import re
import uuid
from typing import Any

from astrbot import logger
from astrbot.core.agent.tool import ToolSet
from astrbot.core.provider.entities import LLMResponse


_PSEUDO_TOOL_CALL_RE = re.compile(
    r"assistant\s+to\s*=\s*functions\.([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)


def extract_allowed_tool_names(tools: ToolSet | None) -> set[str]:
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


def _extract_json_object_after_index(text: str, start_idx: int) -> tuple[str | None, int]:
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
    text: str,
    *,
    allowed_tool_names: set[str],
) -> list[dict[str, Any]]:
    parsed_calls: list[dict[str, Any]] = []
    if not text:
        return parsed_calls

    use_allowlist = bool(allowed_tool_names)
    for match in _PSEUDO_TOOL_CALL_RE.finditer(text):
        tool_name = match.group(1)
        if use_allowlist and tool_name not in allowed_tool_names:
            continue
        raw_json, _ = _extract_json_object_after_index(text, match.end())
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


def maybe_convert_pseudo_tool_calls(
    llm_response: LLMResponse,
    *,
    tools: ToolSet | None,
) -> LLMResponse | None:
    if llm_response.tools_call_name:
        return None
    text = llm_response.completion_text
    if not isinstance(text, str) or not text.strip():
        return None

    allowed_tool_names = extract_allowed_tool_names(tools)
    parsed_calls = _parse_pseudo_tool_calls(text, allowed_tool_names=allowed_tool_names)

    if not parsed_calls and _PSEUDO_TOOL_CALL_RE.search(text):
        if allowed_tool_names:
            parsed_calls = _parse_pseudo_tool_calls(text, allowed_tool_names=set())
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
    return _convert_pseudo_calls_to_llm_response(llm_response, parsed_calls)


def looks_like_pseudo_tool_call_text(llm_response: LLMResponse) -> bool:
    if llm_response.tools_call_name:
        return False
    text = llm_response.completion_text
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(_PSEUDO_TOOL_CALL_RE.search(text))

