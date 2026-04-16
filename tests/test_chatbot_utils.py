"""
Tests for src/chatbot_utils.py

Covers: emoji stripping, filler-opening removal, stat bolding, recursive
payload sanitization, trip-planner detection, action attachment, and
chat-history normalisation.
"""

import pytest

from src.chatbot_utils import (
    attach_trip_planner_action,
    needs_trip_planner_action,
    normalize_chat_history,
    sanitize_chatbot_payload,
)


# ---------------------------------------------------------------------------
# sanitize_chatbot_payload — emoji stripping
# ---------------------------------------------------------------------------

class TestSanitizeChatbotPayloadEmoji:
    def test_removes_common_emoji(self):
        result = sanitize_chatbot_payload("Hello 😊 world")
        assert "😊" not in result

    def test_plain_text_unchanged(self):
        text = "Bangkok has 2,669 accident records."
        result = sanitize_chatbot_payload(text)
        assert "2,669" in result

    def test_handles_empty_string(self):
        assert sanitize_chatbot_payload("") == ""

    def test_processes_list_recursively(self):
        payload = ["Hello 😊", "World 🌏"]
        result = sanitize_chatbot_payload(payload)
        assert isinstance(result, list)
        assert "😊" not in result[0]
        assert "🌏" not in result[1]

    def test_processes_dict_recursively(self):
        payload = {"message": "Safe 🛡️ route", "other": "data"}
        result = sanitize_chatbot_payload(payload)
        assert "🛡️" not in result["message"]

    def test_non_string_passthrough(self):
        assert sanitize_chatbot_payload(42) == 42
        assert sanitize_chatbot_payload(None) is None


# ---------------------------------------------------------------------------
# sanitize_chatbot_payload — rich_text (filler + stat bolding)
# ---------------------------------------------------------------------------

class TestSanitizeChatbotPayloadRichText:
    def test_filler_opening_stripped(self):
        text = "Sure! Bangkok has 50 accidents per zone."
        result = sanitize_chatbot_payload(text, rich_text=True)
        assert not result.lower().startswith("sure")

    def test_percentage_bolded(self):
        result = sanitize_chatbot_payload("The model has 83.8% accuracy.", rich_text=True)
        assert "**83.8%**" in result

    def test_existing_bold_not_double_bolded(self):
        text = "Already **83.8%** accurate."
        result = sanitize_chatbot_payload(text, rich_text=True)
        assert "****" not in result

    def test_dict_message_key_gets_rich_text(self):
        payload = {"message": "Sure! 83.8% accuracy.", "other": "plain 50% here"}
        result = sanitize_chatbot_payload(payload)
        # "message" key triggers rich_text
        assert "**83.8%**" in result["message"]


# ---------------------------------------------------------------------------
# needs_trip_planner_action
# ---------------------------------------------------------------------------

class TestNeedsTripPlannerAction:
    def test_route_safety_intent_triggers(self):
        assert needs_trip_planner_action("", intent_name="route_safety") is True

    def test_route_keyword_in_message_triggers(self):
        assert needs_trip_planner_action("Show me the safest route") is True

    def test_from_to_pattern_triggers(self):
        assert needs_trip_planner_action("How do I get from Siam to Don Mueang?") is True

    def test_unrelated_message_does_not_trigger(self):
        assert needs_trip_planner_action("What is the most dangerous hour?") is False

    def test_trip_planner_in_response_triggers(self):
        response = {"message": "You can use the trip planner for that."}
        assert needs_trip_planner_action("", response=response) is True


# ---------------------------------------------------------------------------
# attach_trip_planner_action
# ---------------------------------------------------------------------------

class TestAttachTripPlannerAction:
    def test_adds_action_when_absent(self):
        response = {"message": "Test"}
        result = attach_trip_planner_action(response)
        hrefs = [a.get("href") for a in result.get("actions", [])]
        assert "/trip-planner" in hrefs

    def test_does_not_duplicate_existing_action(self):
        response = {
            "message": "Test",
            "actions": [{"label": "Open Trip Planner", "href": "/trip-planner", "style": "primary"}],
        }
        result = attach_trip_planner_action(response)
        trip_actions = [a for a in result["actions"] if a.get("href") == "/trip-planner"]
        assert len(trip_actions) == 1

    def test_non_dict_response_returned_unchanged(self):
        assert attach_trip_planner_action("string") == "string"


# ---------------------------------------------------------------------------
# normalize_chat_history
# ---------------------------------------------------------------------------

class TestNormalizeChatHistory:
    def test_non_list_returns_empty(self):
        assert normalize_chat_history(None) == []
        assert normalize_chat_history("bad input") == []

    def test_valid_history_preserved(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = normalize_chat_history(history)
        assert len(result) == 2
        assert result[0]["role"] == "user"

    def test_invalid_role_filtered_out(self):
        history = [{"role": "system", "content": "Ignore this"}]
        assert normalize_chat_history(history) == []

    def test_content_truncated_to_1200_chars(self):
        long_content = "x" * 2000
        history = [{"role": "user", "content": long_content}]
        result = normalize_chat_history(history)
        assert len(result[0]["content"]) == 1200

    def test_max_messages_respected(self):
        history = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        result = normalize_chat_history(history, max_messages=5)
        assert len(result) <= 5
