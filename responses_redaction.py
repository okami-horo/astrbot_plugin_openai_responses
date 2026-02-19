from __future__ import annotations

import json
import re
from typing import Any, Mapping
from urllib.parse import SplitResult, urlsplit, urlunsplit


_SK_RE = re.compile(r"sk-[A-Za-z0-9]{10,}")
_BEARER_RE = re.compile(r"(Bearer)\s+([A-Za-z0-9._-]+)", re.IGNORECASE)

_SENSITIVE_HEADER_KEYS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
}


def redact_text(text: Any, *, max_len: int = 512) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    redacted = _BEARER_RE.sub(r"\1 ***", text)
    redacted = _SK_RE.sub("sk-***", redacted)

    if max_len > 0 and len(redacted) > max_len:
        redacted = redacted[:max_len] + "...(truncated)"
    return redacted


def redact_headers(headers: Mapping[str, Any] | None) -> dict[str, str] | None:
    if not headers:
        return None
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        normalized_key = str(key)
        if normalized_key.lower() in _SENSITIVE_HEADER_KEYS:
            redacted[normalized_key] = "***"
            continue
        redacted[normalized_key] = redact_text(value, max_len=256)
    return redacted


def redact_proxy_url(proxy_url: str | None) -> str | None:
    if not proxy_url:
        return proxy_url
    if not isinstance(proxy_url, str):
        proxy_url = str(proxy_url)
    try:
        parts: SplitResult = urlsplit(proxy_url)
    except Exception:
        return redact_text(proxy_url, max_len=256)

    netloc = parts.netloc
    if "@" not in netloc:
        return proxy_url

    _userinfo, hostport = netloc.rsplit("@", 1)
    redacted_netloc = f"***:***@{hostport}"
    return urlunsplit((parts.scheme, redacted_netloc, parts.path, parts.query, parts.fragment))


def redact_json(value: Any, *, max_len: int = 512) -> str:
    try:
        dumped = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        dumped = str(value)
    return redact_text(dumped, max_len=max_len)


def redact_event_data(event_type: str, event_data: Any, *, max_len: int = 512) -> str:
    """Redact structured SSE event payload for logging.

    This is intentionally conservative: tool arguments/output are replaced by
    placeholders to avoid leaking structured payloads into logs.
    """

    keys_to_redact = {
        "authorization",
        "proxy",
        "headers",
        "custom_headers",
        "cookie",
        "set-cookie",
        "proxy-authorization",
        "x-api-key",
    }

    if event_type.startswith("response.function_call_arguments."):
        keys_to_redact |= {"delta", "arguments"}
    if "function_call" in event_type:
        keys_to_redact |= {"arguments"}
    if "function_call_output" in event_type:
        keys_to_redact |= {"output"}

    def _redact_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            redacted: dict[str, Any] = {}
            for k, v in obj.items():
                key = str(k)
                if key.lower() in keys_to_redact:
                    if isinstance(v, str):
                        redacted[key] = f"<redacted len={len(v)}>"
                    else:
                        redacted[key] = "<redacted>"
                    continue
                redacted[key] = _redact_obj(v)
            return redacted
        if isinstance(obj, list):
            return [_redact_obj(item) for item in obj[:50]]
        if isinstance(obj, str):
            return redact_text(obj, max_len=128)
        return obj

    return redact_json(_redact_obj(event_data), max_len=max_len)
