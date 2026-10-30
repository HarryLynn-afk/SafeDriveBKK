"""
Lightweight in-memory rate limiter.

Uses a fixed-window counter per (endpoint, client IP). Thread-safe via a
threading.Lock. Counters are per process; Docker starts two Gunicorn workers.
For multi-process or distributed deployments, replace the in-memory store
with Redis (e.g. flask-limiter with RedisStorage).

Usage
-----
    from .utils.rate_limit import rate_limit

    @bp.route("/expensive-endpoint", methods=["POST"])
    @rate_limit(max_calls=20, window_seconds=3600)   # 20 per hour per IP
    def my_endpoint():
        ...
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from functools import wraps

from flask import jsonify, request

_counters: dict = defaultdict(lambda: {"count": 0, "reset_at": 0.0})
_lock = threading.Lock()

# Periodically drop counters for IPs whose window has fully elapsed so the
# in-memory store cannot grow without bound under many distinct clients.
_CLEANUP_INTERVAL_SECONDS = 600.0
_last_cleanup = 0.0


def _evict_expired(now: float) -> None:
    """Purge counters whose window has elapsed. Caller must hold _lock."""
    global _last_cleanup
    if now - _last_cleanup < _CLEANUP_INTERVAL_SECONDS:
        return
    _last_cleanup = now
    stale = [key for key, entry in _counters.items() if now > entry["reset_at"]]
    for key in stale:
        del _counters[key]


def rate_limit(max_calls: int, window_seconds: int):
    """Decorator: allow at most *max_calls* requests per IP per *window_seconds*."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            ip = request.remote_addr or "unknown"
            key = f"{fn.__qualname__}:{ip}"
            now = time.monotonic()

            with _lock:
                _evict_expired(now)
                entry = _counters[key]
                if now > entry["reset_at"]:
                    entry["count"] = 0
                    entry["reset_at"] = now + window_seconds
                entry["count"] += 1
                over_limit = entry["count"] > max_calls

            if over_limit:
                return jsonify({"error": "Too many requests. Please try again later."}), 429

            return fn(*args, **kwargs)
        return wrapper
    return decorator
