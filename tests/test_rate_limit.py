"""
Tests for the in-memory rate limiter decorator and its bounded eviction.

The decorator needs a Flask request context, so each test builds a tiny app
with a decorated route and drives it through the test client. The shared
_counters store is cleared between tests to avoid cross-test bleed.
"""

import time

import flask
import pytest

from src.utils import rate_limit as rl


@pytest.fixture(autouse=True)
def _isolate_counters():
    rl._counters.clear()
    rl._last_cleanup = 0.0
    yield
    rl._counters.clear()


def _client(max_calls=3, window=3600):
    app = flask.Flask(__name__)

    @app.route("/ping", methods=["POST"])
    @rl.rate_limit(max_calls=max_calls, window_seconds=window)
    def ping():
        return {"ok": True}

    return app.test_client()


class TestRateLimitDecorator:
    def test_allows_up_to_the_limit(self):
        client = _client(max_calls=3)
        for _ in range(3):
            assert client.post("/ping").status_code == 200

    def test_blocks_over_the_limit_with_429(self):
        client = _client(max_calls=3)
        for _ in range(3):
            client.post("/ping")
        assert client.post("/ping").status_code == 429

    def test_new_window_allows_again(self):
        client = _client(max_calls=1)
        assert client.post("/ping").status_code == 200
        assert client.post("/ping").status_code == 429
        # Simulate the window elapsing for the (single) tracked client.
        key = next(iter(rl._counters))
        rl._counters[key]["reset_at"] = time.monotonic() - 1
        assert client.post("/ping").status_code == 200


class TestEviction:
    def test_evict_expired_drops_stale_keeps_active(self):
        now = time.monotonic()
        rl._counters["old:1.2.3.4"] = {"count": 9, "reset_at": now - 10}    # window passed
        rl._counters["live:5.6.7.8"] = {"count": 1, "reset_at": now + 1000}  # still active
        rl._last_cleanup = 0.0  # force a cleanup pass

        rl._evict_expired(now)

        assert "old:1.2.3.4" not in rl._counters
        assert "live:5.6.7.8" in rl._counters

    def test_eviction_is_throttled_by_interval(self):
        now = time.monotonic()
        rl._counters["old:1.2.3.4"] = {"count": 9, "reset_at": now - 10}
        rl._last_cleanup = now  # just cleaned → next call should be a no-op

        rl._evict_expired(now + 1)

        assert "old:1.2.3.4" in rl._counters  # not yet purged
