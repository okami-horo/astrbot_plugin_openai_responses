from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


_PRUNE_STRATEGIES = {"pair_aware", "legacy"}
_PROMPT_CACHE_KEY_PREFIX = "astrbot:pc:v1:"
_PROMPT_CACHE_KEY_MAX_LEN = 64
_PROMPT_CACHE_KEY_DIGEST_HEX_LEN = min(
    32,
    max(8, _PROMPT_CACHE_KEY_MAX_LEN - len(_PROMPT_CACHE_KEY_PREFIX)),
)
_PROMPT_CACHE_SEED_FALLBACK_MODEL = "unknown-model"
_PROMPT_CACHE_SEED_FALLBACK_API_BASE = "unknown-api-base"


def normalize_prune_strategy(value: Any, default: str = "pair_aware") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _PRUNE_STRATEGIES:
            return normalized
    return default


def _safe_json_dumps(value: Any, *, sort_keys: bool = False) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
            sort_keys=sort_keys,
        )
    except Exception:
        return str(value)


@dataclass
class TurnStateSnapshot:
    last_response_id: str | None = None
    last_successful_turn_digest: str | None = None
    last_tool_chain_snapshot: list[dict[str, Any]] | None = None


class ResponsesTurnState:
    def __init__(self) -> None:
        self.snapshot = TurnStateSnapshot(last_tool_chain_snapshot=[])

    def build_prompt_cache_key(
        self,
        *,
        session_id: str | None,
        messages: list[dict[str, Any]],
        model: str | None = None,
        api_base: str | None = None,
    ) -> str:
        normalized_session_id = (
            session_id.strip()
            if isinstance(session_id, str) and session_id.strip()
            else ""
        )
        normalized_model = (
            model.strip()
            if isinstance(model, str) and model.strip()
            else _PROMPT_CACHE_SEED_FALLBACK_MODEL
        )
        normalized_api_base = (
            api_base.strip().rstrip("/")
            if isinstance(api_base, str) and api_base.strip()
            else _PROMPT_CACHE_SEED_FALLBACK_API_BASE
        )

        seed_payload: dict[str, Any] = {
            "session_id": normalized_session_id,
            "model": normalized_model,
            "api_base": normalized_api_base,
        }
        if not normalized_session_id:
            # When session_id is unavailable, include message content to reduce collisions.
            messages_digest = hashlib.sha256(
                _safe_json_dumps(messages, sort_keys=True).encode("utf-8")
            ).hexdigest()
            seed_payload["messages_digest"] = messages_digest

        digest_source = _safe_json_dumps(seed_payload, sort_keys=True)
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[
            :_PROMPT_CACHE_KEY_DIGEST_HEX_LEN
        ]
        key = f"{_PROMPT_CACHE_KEY_PREFIX}{digest}"
        return key[:_PROMPT_CACHE_KEY_MAX_LEN]

    def note_success(
        self,
        *,
        response_id: str | None,
        request_messages: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]] | None,
    ) -> None:
        if isinstance(response_id, str) and response_id:
            self.snapshot.last_response_id = response_id

        digest_source = _safe_json_dumps(request_messages)
        self.snapshot.last_successful_turn_digest = hashlib.sha1(
            digest_source.encode("utf-8")
        ).hexdigest()

        if isinstance(tool_calls, list):
            self.snapshot.last_tool_chain_snapshot = tool_calls
        else:
            self.snapshot.last_tool_chain_snapshot = []

    def prune_messages_for_context_limit(
        self,
        messages: list[dict[str, Any]],
        *,
        strategy: str = "pair_aware",
    ) -> bool:
        if not messages:
            return False

        strategy = normalize_prune_strategy(strategy)
        if strategy == "legacy":
            messages.pop(0)
            return True

        return self._pair_aware_prune(messages)

    @staticmethod
    def _pair_aware_prune(messages: list[dict[str, Any]]) -> bool:
        if not messages:
            return False

        removable_idx = None
        tool_pair_idx = None
        for idx, msg in enumerate(messages):
            role = str(msg.get("role", "")).lower()
            if role not in {"system", "developer"}:
                if removable_idx is None:
                    removable_idx = idx
                if role == "assistant" and msg.get("tool_calls"):
                    tool_pair_idx = idx
                    break

        if tool_pair_idx is not None:
            removable_idx = tool_pair_idx

        if removable_idx is None:
            messages.pop(0)
            return True

        msg = messages[removable_idx]
        role = str(msg.get("role", "")).lower()
        remove_indices: set[int] = {removable_idx}

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            call_ids: set[str] = set()
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if isinstance(call, dict):
                        call_id = call.get("id")
                        if isinstance(call_id, str) and call_id:
                            call_ids.add(call_id)
            if call_ids:
                prev_idx = removable_idx - 1
                if prev_idx >= 0 and str(messages[prev_idx].get("role", "")).lower() == "user":
                    remove_indices.add(prev_idx)
                for idx in range(removable_idx + 1, len(messages)):
                    next_msg = messages[idx]
                    if str(next_msg.get("role", "")).lower() != "tool":
                        continue
                    tool_call_id = next_msg.get("tool_call_id")
                    if isinstance(tool_call_id, str) and tool_call_id in call_ids:
                        remove_indices.add(idx)
        elif role == "tool":
            for idx in range(removable_idx + 1, len(messages)):
                if str(messages[idx].get("role", "")).lower() == "tool":
                    remove_indices.add(idx)
                else:
                    break
        elif role == "user":
            candidate_idx = removable_idx + 1
            if candidate_idx < len(messages):
                next_msg = messages[candidate_idx]
                if (
                    str(next_msg.get("role", "")).lower() == "assistant"
                    and not next_msg.get("tool_calls")
                ):
                    remove_indices.add(candidate_idx)

        new_messages = [
            message
            for idx, message in enumerate(messages)
            if idx not in remove_indices
        ]
        if len(new_messages) == len(messages):
            messages.pop(removable_idx)
            return True

        messages[:] = new_messages
        return True
