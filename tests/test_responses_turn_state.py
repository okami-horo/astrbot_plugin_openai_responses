try:
    from responses_turn_state import ResponsesTurnState
except ImportError:  # pragma: no cover
    from data.plugins.astrbot_plugin_openai_responses.responses_turn_state import (
        ResponsesTurnState,
    )


def test_turn_state_prompt_cache_key_uses_fixed_ascii_format():
    state = ResponsesTurnState()
    key = state.build_prompt_cache_key(
        session_id="session-1",
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5.3-codex",
        api_base="https://chatgpt.com/backend-api/codex",
    )
    assert key.startswith("astrbot:pc:v1:")
    assert len(key.split(":")[-1]) == 32
    assert len(key) <= 64
    assert key.isascii()


def test_turn_state_prompt_cache_key_truncates_long_session_id():
    state = ResponsesTurnState()
    key = state.build_prompt_cache_key(
        session_id="\u8d85\u957f\u4f1a\u8bdd-" + "x" * 512,
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5.3-codex",
        api_base="https://chatgpt.com/backend-api/codex/",
    )
    assert key.startswith("astrbot:pc:v1:")
    assert len(key) <= 64
    assert key.isascii()


def test_turn_state_prompt_cache_key_is_stable_for_same_session_context():
    state = ResponsesTurnState()
    params = {
        "session_id": "session-stable",
        "messages": [{"role": "user", "content": "hello"}],
        "model": "gpt-5.3-codex",
        "api_base": "https://chatgpt.com/backend-api/codex",
    }
    assert state.build_prompt_cache_key(**params) == state.build_prompt_cache_key(**params)


def test_turn_state_prompt_cache_key_differs_for_different_models():
    state = ResponsesTurnState()
    common = {
        "session_id": "session-1",
        "messages": [{"role": "user", "content": "hello"}],
        "api_base": "https://chatgpt.com/backend-api/codex",
    }
    key_a = state.build_prompt_cache_key(model="gpt-5.2-codex", **common)
    key_b = state.build_prompt_cache_key(model="gpt-5.3-codex", **common)
    assert key_a != key_b


def test_turn_state_prompt_cache_key_without_session_depends_on_messages():
    state = ResponsesTurnState()
    common = {
        "session_id": None,
        "model": "gpt-5.3-codex",
        "api_base": "https://chatgpt.com/backend-api/codex",
    }
    key_a = state.build_prompt_cache_key(
        messages=[{"role": "user", "content": "hello"}],
        **common,
    )
    key_b = state.build_prompt_cache_key(
        messages=[{"role": "user", "content": "hello 2"}],
        **common,
    )
    assert key_a != key_b


def test_pair_aware_prune_removes_assistant_tool_and_outputs():
    state = ResponsesTurnState()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "assistant", "content": "final"},
    ]

    removed = state.prune_messages_for_context_limit(messages, strategy="pair_aware")

    assert removed is True
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "final"},
    ]


def test_legacy_prune_removes_first_message():
    state = ResponsesTurnState()
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    removed = state.prune_messages_for_context_limit(messages, strategy="legacy")
    assert removed is True
    assert messages == [{"role": "assistant", "content": "a1"}]
