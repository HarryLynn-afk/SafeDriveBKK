"""
Tests for src/chatbot_intents.py

Covers the registry boundary:
  - @intent decorator registers a handler with its patterns
  - detect_intent() returns the intent name matching the message
  - detect_intent() respects _INTENT_ORDER priority
  - detect_intent() returns None when nothing matches
  - get_scripted_response() dispatches to the registered handler

Handler body logic is NOT tested here — each handler has its own concerns
(load_chatbot_data, etc.) that belong in higher-level / behavioural tests.
"""

import re

import pytest

from src import chatbot_intents
from src.chatbot_intents import detect_intent, get_scripted_response, intent


# ---------------------------------------------------------------------------
# detect_intent — known patterns route to the expected intent
# ---------------------------------------------------------------------------

class TestDetectIntentKnownPatterns:
    def test_top_danger_corridors_query(self):
        assert detect_intent("Which areas are most dangerous?") == "top_danger_corridors"

    def test_peak_risk_time_query(self):
        assert detect_intent("When is it most dangerous?") == "peak_risk_time"

    def test_route_safety_query(self):
        assert detect_intent("What's the safest route to Asok?") == "route_safety"


# ---------------------------------------------------------------------------
# detect_intent — no-match cases
# ---------------------------------------------------------------------------

class TestDetectIntentNoMatch:
    def test_unrelated_text_returns_none(self):
        assert detect_intent("Tell me a joke") is None

    def test_weather_query_returns_none(self):
        assert detect_intent("What's the weather like?") is None

    def test_empty_string_returns_none(self):
        assert detect_intent("") is None

    def test_whitespace_only_returns_none(self):
        assert detect_intent("   \t\n  ") is None


# ---------------------------------------------------------------------------
# detect_intent — priority ordering via _INTENT_ORDER
# ---------------------------------------------------------------------------

class TestDetectIntentPriority:
    def test_earlier_intent_wins_when_patterns_overlap(self):
        # "high-risk speeding zones" matches both list_speeding_zones (priority
        # 5: "speed.*zones") and high_risk_zones (priority 16: "high.*risk.*zones").
        # The earlier one in _INTENT_ORDER must win.
        assert detect_intent("high-risk speeding zones") == "list_speeding_zones"


# ---------------------------------------------------------------------------
# @intent decorator + get_scripted_response dispatch
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_registry():
    """Snapshot the global registry before the test; restore after.

    Tests that register a temporary intent need this fixture so the new
    entry doesn't leak into later tests in the same session.
    """
    saved_patterns = dict(chatbot_intents._INTENT_PATTERNS)
    saved_handlers = dict(chatbot_intents._INTENT_BY_NAME)
    saved_order = list(chatbot_intents._INTENT_ORDER)
    yield
    chatbot_intents._INTENT_PATTERNS.clear()
    chatbot_intents._INTENT_PATTERNS.update(saved_patterns)
    chatbot_intents._INTENT_BY_NAME.clear()
    chatbot_intents._INTENT_BY_NAME.update(saved_handlers)
    chatbot_intents._INTENT_ORDER[:] = saved_order


class TestIntentDecoratorAndDispatch:
    def test_decorator_registers_handler_and_patterns(self, clean_registry):
        @intent(
            name="__test_widget__",
            patterns=[re.compile(r"widget-marker-xyz", re.I)],
        )
        def _widget_handler(user_message=""):
            return {"type": "info", "message": "widget"}

        assert "__test_widget__" in chatbot_intents._INTENT_PATTERNS
        assert "__test_widget__" in chatbot_intents._INTENT_BY_NAME
        assert chatbot_intents._INTENT_BY_NAME["__test_widget__"] is _widget_handler

    def test_get_scripted_response_dispatches_to_registered_handler(self, clean_registry):
        captured: dict = {}

        @intent(
            name="__test_capture__",
            patterns=[re.compile(r"capture-marker-abc", re.I)],
        )
        def _capture_handler(user_message=""):
            captured["msg"] = user_message
            return {"type": "info", "message": "captured"}

        # Put our intent at the top of priority order so detect_intent finds it.
        chatbot_intents._INTENT_ORDER.insert(0, "__test_capture__")

        assert detect_intent("please capture-marker-abc now") == "__test_capture__"
        result = get_scripted_response("__test_capture__", "please capture-marker-abc now")

        assert captured["msg"] == "please capture-marker-abc now"
        assert result == {"type": "info", "message": "captured"}
