from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_RUNTIME_MODES = {"auto", "openai", "chatgpt"}
_CODEX_MODEL_RE = re.compile(r"(^|[-_.])codex($|[-_.])", re.IGNORECASE)


@dataclass(frozen=True)
class RuntimeModeResolution:
    mode: str
    reason: str

    @property
    def is_codex_chatgpt(self) -> bool:
        return self.mode == "chatgpt"


def normalize_codex_mode(value: Any, default: str = "auto") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _RUNTIME_MODES:
            return normalized
    return default


def is_codex_model_name(model: str) -> bool:
    return bool(_CODEX_MODEL_RE.search(model or ""))


def is_chatgpt_codex_base_url(api_base: str) -> bool:
    normalized = (api_base or "").strip().lower().rstrip("/")
    return normalized.startswith("https://chatgpt.com/backend-api/codex")


def resolve_runtime_mode(
    *,
    model: str,
    api_base: str,
    codex_mode: str | None,
) -> RuntimeModeResolution:
    explicit = normalize_codex_mode(codex_mode, default="auto")
    if explicit != "auto":
        return RuntimeModeResolution(mode=explicit, reason=f"explicit:{explicit}")

    if is_chatgpt_codex_base_url(api_base):
        return RuntimeModeResolution(mode="chatgpt", reason="api_base:chatgpt_codex")

    if is_codex_model_name(model):
        return RuntimeModeResolution(mode="chatgpt", reason="model:codex")

    return RuntimeModeResolution(mode="openai", reason="default:openai")
