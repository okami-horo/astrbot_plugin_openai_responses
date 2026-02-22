try:
    from responses_turn_state import ResponsesTurnState
except ImportError:  # pragma: no cover
    from data.plugins.astrbot_plugin_openai_responses.responses_turn_state import (
        ResponsesTurnState,
    )


def test_turn_state_prompt_cache_key_prefers_session_id():
    state = ResponsesTurnState()
    key = state.build_prompt_cache_key(
        session_id="session-1",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert key == "astrbot:session-1"


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
