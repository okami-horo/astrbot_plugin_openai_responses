import pytest

try:
    # When running tests from the plugin repository root
    from responses_errors import (
        UpstreamResponsesError,
        compute_backoff_seconds,
        error_type,
        should_retry,
    )
except ImportError:  # pragma: no cover
    # When running tests from an AstrBot repository checkout
    from data.plugins.astrbot_plugin_openai_responses.responses_errors import (
        UpstreamResponsesError,
        compute_backoff_seconds,
        error_type,
        should_retry,
    )


def test_error_type_classifies_by_status_code():
    assert error_type(UpstreamResponsesError("x", status_code=401)) == "auth"
    assert error_type(UpstreamResponsesError("x", status_code=403)) == "auth"
    assert error_type(UpstreamResponsesError("x", status_code=429)) == "rate_limit"
    assert error_type(UpstreamResponsesError("x", status_code=500)) == "server"


def test_should_retry_respects_max_attempts():
    assert should_retry("rate_limit", attempt=0, max_attempts=1) is True
    assert should_retry("rate_limit", attempt=1, max_attempts=1) is False
    assert should_retry("auth", attempt=0, max_attempts=10) is False


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        (0, 1.0),
        (1, 2.0),
        (2, 4.0),
        (10, 60.0),
    ],
)
def test_compute_backoff_seconds_is_exponential_and_capped(attempt, expected):
    assert (
        compute_backoff_seconds(
            attempt=attempt,
            base_seconds=1.0,
            max_seconds=60.0,
            jitter_ratio=0.0,
        )
        == expected
    )


def test_compute_backoff_seconds_applies_jitter_deterministically():
    base = compute_backoff_seconds(
        attempt=1,
        base_seconds=2.0,
        max_seconds=60.0,
        jitter_ratio=0.1,
        random_fn=lambda: 0.0,
    )
    assert base == pytest.approx((2.0 * 2.0) * (1 - 0.1))
