from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from astrbot.core.agent.tool import FunctionTool, ToolSet

try:
    from provider_responses import ProviderOpenAIResponsesPlugin
except ImportError:  # pragma: no cover
    from data.plugins.astrbot_plugin_openai_responses.provider_responses import (
        ProviderOpenAIResponsesPlugin,
    )

try:
    from responses_redaction import redact_proxy_url, redact_text
except ImportError:  # pragma: no cover
    from data.plugins.astrbot_plugin_openai_responses.responses_redaction import (
        redact_proxy_url,
        redact_text,
    )


@dataclass
class RunStats:
    turns: int = 0
    ok: int = 0
    tool_turns: int = 0
    errors: int = 0
    duration_ms_total: int = 0


def _env_keys() -> list[str]:
    keys = os.getenv("OPENAI_API_KEYS") or os.getenv("OPENAI_API_KEY") or ""
    items = [k.strip() for k in keys.split(",") if k.strip()]
    return items


def _make_provider() -> ProviderOpenAIResponsesPlugin:
    keys = _env_keys()
    if not keys:
        raise RuntimeError(
            "Missing OPENAI_API_KEY (or OPENAI_API_KEYS as comma-separated list)."
        )
    provider_config: dict[str, Any] = {
        "id": "reliability-smoke",
        "type": "openai_responses_plugin",
        "model": os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        "key": keys,
        "api_base": os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1",
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "120")),
        "proxy": os.getenv("OPENAI_PROXY", "") or "",
        "tool_fallback_enabled": True,
        "tool_fallback_mode": "parse_then_retry",
        "tool_fallback_retry_attempts": 1,
        "tool_fallback_force_tool_choice": "required",
        "tool_fallback_stream_buffer": True,
        "log_usage": True,
    }
    return ProviderOpenAIResponsesPlugin(provider_config=provider_config, provider_settings={})


def _make_tools() -> ToolSet:
    echo = FunctionTool(
        name="smoke_echo",
        description="Echo back the input payload.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=None,
    )
    add = FunctionTool(
        name="smoke_add",
        description="Add two integers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        handler=None,
    )
    return ToolSet([echo, add])


def _execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "smoke_echo":
        return {"ok": True, "echo": args.get("text", "")}
    if name == "smoke_add":
        a = int(args.get("a", 0))
        b = int(args.get("b", 0))
        return {"ok": True, "sum": a + b}
    return {"ok": False, "error": f"unknown tool: {name}"}


def _append_assistant_tool_calls(
    messages: list[dict[str, Any]],
    *,
    call_ids: list[str],
    names: list[str],
    args_list: list[dict[str, Any]],
) -> None:
    tool_calls = []
    for call_id, name, args in zip(call_ids, names, args_list, strict=False):
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )
    messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})


def _append_tool_outputs(
    messages: list[dict[str, Any]],
    *,
    call_ids: list[str],
    names: list[str],
    args_list: list[dict[str, Any]],
) -> None:
    for call_id, name, args in zip(call_ids, names, args_list, strict=False):
        output = _execute_tool(name, args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
            }
        )


async def _run_once(
    *,
    provider: ProviderOpenAIResponsesPlugin,
    tools: ToolSet,
    messages: list[dict[str, Any]],
    tool_choice_override: str | dict[str, Any] | None,
) -> tuple[bool, str, dict[str, Any] | None]:
    payloads = {"model": provider.get_model(), "messages": messages}
    start = time.perf_counter()
    resp, _ = await provider._query_final_response_with_retries(
        payloads=payloads,
        func_tool=tools,
        reasoning_effort=None,
        response_format=None,
        tool_choice_override=tool_choice_override,
        max_retries=int(os.getenv("SMOKE_MAX_RETRIES", "5")),
    )
    duration_ms = int((time.perf_counter() - start) * 1000)
    usage = None
    if resp.usage is not None:
        usage = {
            "input_other": resp.usage.input_other,
            "input_cached": resp.usage.input_cached,
            "output": resp.usage.output,
        }
    if resp.role == "tool" and resp.tools_call_name:
        _append_assistant_tool_calls(
            messages,
            call_ids=resp.tools_call_ids,
            names=resp.tools_call_name,
            args_list=resp.tools_call_args,
        )
        _append_tool_outputs(
            messages,
            call_ids=resp.tools_call_ids,
            names=resp.tools_call_name,
            args_list=resp.tools_call_args,
        )
        follow, _ = await provider._query_final_response_with_retries(
            payloads={"model": provider.get_model(), "messages": messages},
            func_tool=tools,
            reasoning_effort=None,
            response_format=None,
            tool_choice_override=None,
            max_retries=int(os.getenv("SMOKE_MAX_RETRIES", "5")),
        )
        messages.append({"role": "assistant", "content": follow.completion_text or ""})
        return True, f"tool->assistant ({duration_ms}ms)", usage

    messages.append({"role": "assistant", "content": resp.completion_text or ""})
    return True, f"assistant ({duration_ms}ms)", usage


async def main() -> None:
    turns = int(os.getenv("SMOKE_TURNS", "100"))
    force_tool_every = int(os.getenv("SMOKE_FORCE_TOOL_EVERY", "10"))
    max_history = int(os.getenv("SMOKE_MAX_HISTORY", "24"))

    provider = _make_provider()
    tools = _make_tools()
    stats = RunStats()

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a concise assistant. Use tools when required and keep answers short.",
        }
    ]

    try:
        for idx in range(turns):
            stats.turns += 1
            prompt = (
                f"[turn {idx+1}/{turns}] Please reply with one short paragraph."
                if (idx + 1) % force_tool_every != 0
                else f"[turn {idx+1}/{turns}] Use a tool call. Add a={idx} and b=1 with smoke_add, then explain the result."
            )
            messages.append({"role": "user", "content": prompt})

            tool_choice_override = (
                "required" if (idx + 1) % force_tool_every == 0 else None
            )
            if tool_choice_override is not None:
                stats.tool_turns += 1

            ok, mode, usage = await _run_once(
                provider=provider,
                tools=tools,
                messages=messages,
                tool_choice_override=tool_choice_override,
            )
            if ok:
                stats.ok += 1
            else:
                stats.errors += 1

            # Keep context bounded to reduce accidental context blow-up.
            if len(messages) > max_history:
                messages = messages[-max_history:]

    except Exception as exc:
        stats.errors += 1
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": redact_text(str(exc)),
                    "proxy": redact_proxy_url(os.getenv("OPENAI_PROXY", "") or ""),
                },
                ensure_ascii=False,
            )
        )
        raise
    finally:
        await provider.terminate()

    print(
        json.dumps(
            {
                "ok": True,
                "turns": stats.turns,
                "ok_count": stats.ok,
                "error_count": stats.errors,
                "tool_turns": stats.tool_turns,
                "model": provider.get_model(),
                "api_base": os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1",
                "proxy": redact_proxy_url(os.getenv("OPENAI_PROXY", "") or ""),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

