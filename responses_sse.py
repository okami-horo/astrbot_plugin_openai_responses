from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from astrbot import logger

try:
    from responses_errors import UpstreamResponsesError
except ImportError:  # pragma: no cover
    from .responses_errors import UpstreamResponsesError


async def iter_responses_sse_events(
    lines: AsyncIterable[str],
) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
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
                "Responses SSE JSON decode failed. event=%s bytes=%s",
                event_type_hint,
                len(data_str),
            )
            return None

        if not isinstance(event_data, dict):
            return None

        event_type = event_type_hint or str(event_data.get("type") or "")
        if not event_type:
            return None

        if event_type == "error" or "error" in event_data:
            err = event_data.get("error", {})
            err_code = ""
            err_msg = ""
            if isinstance(err, dict):
                err_code = str(err.get("code") or "")
                err_msg = str(err.get("message") or "")
            raise UpstreamResponsesError(
                f"Responses SSE error {err_code}: {err_msg}",
                body=event_data,
                response_text=data_str,
            )

        if event_type in {"response.failed", "response.incomplete"}:
            response_obj = event_data.get("response", {})
            err_msg = ""
            err_code = ""
            if isinstance(response_obj, dict):
                if event_type == "response.failed":
                    err_obj = response_obj.get("error", {})
                    if isinstance(err_obj, dict):
                        err_msg = str(err_obj.get("message") or "")
                        err_code = str(err_obj.get("code") or "")
                elif event_type == "response.incomplete":
                    details = response_obj.get("incomplete_details", {})
                    if isinstance(details, dict):
                        reason = str(details.get("reason") or "unknown")
                        err_msg = f"incomplete response: {reason}"
                        err_code = reason

            raise UpstreamResponsesError(
                f"Responses SSE {event_type} {err_code}: {err_msg}",
                body=event_data,
                response_text=data_str,
            )

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

    async for raw_line in lines:
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
