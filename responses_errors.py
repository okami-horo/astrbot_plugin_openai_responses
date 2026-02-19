from __future__ import annotations

import json
import random
from types import SimpleNamespace
from typing import Any, Callable


class UpstreamResponsesError(Exception):
    """A lightweight error wrapper with fields used by provider handlers."""

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


def error_type(exc: Exception) -> str:
    """Classify an exception into a coarse-grained error type string.

    This is intentionally lightweight. Provider-specific handlers may further refine
    behavior based on upstream payloads and AstrBot utilities.
    """

    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403):
        return "auth"
    if status_code == 429:
        return "rate_limit"
    if isinstance(status_code, int) and status_code >= 500:
        return "server"

    message = str(exc).lower()
    if "maximum context length" in message:
        return "context_length"
    if "function calling is not enabled" in message or (
        "tool" in message and "support" in message
    ):
        return "tool_unsupported"
    if "the model is not a vlm" in message:
        return "image_rejected"
    if isinstance(exc, (OSError, TimeoutError, ConnectionError)):
        return "connection"
    return "unknown"


def should_retry(err_type: str, *, attempt: int, max_attempts: int) -> bool:
    if max_attempts <= 0:
        return False
    if attempt >= max_attempts:
        return False
    return err_type in {"rate_limit", "server", "connection"}


def compute_backoff_seconds(
    *,
    attempt: int,
    base_seconds: float = 1.0,
    max_seconds: float = 60.0,
    jitter_ratio: float = 0.1,
    random_fn: Callable[[], float] | None = None,
) -> float:
    """Exponential backoff with optional jitter.

    - attempt is 0-based (0 => base_seconds)
    - jitter_ratio controls +/- percentage around the base backoff
    """

    if attempt < 0:
        attempt = 0
    backoff = float(base_seconds) * (2.0**attempt)
    if max_seconds > 0:
        backoff = min(backoff, float(max_seconds))

    jitter_ratio = max(0.0, float(jitter_ratio))
    if jitter_ratio > 0:
        if random_fn is None:
            random_fn = random.random
        r = float(random_fn())
        factor = 1.0 + jitter_ratio * (2.0 * r - 1.0)
        backoff *= factor

    return max(0.0, backoff)
