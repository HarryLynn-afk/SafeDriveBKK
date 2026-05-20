"""
Tests for src/chatbot.py — the chatbot orchestration boundary.

Covers the public contract of respond_to_chat_message():
  - Tier dispatch (scripted → ML cause → Groq), first tier wins
  - Tier annotation (source/intent on scripted responses)
  - Trip-planner CTA attachment for route-shaped messages
  - Sanitization invariant (no emojis in returned message)
  - raw_history=None is handled safely

External call (Groq) is patched; all internal tiers run real.
"""

from unittest.mock import patch

from src.chatbot import respond_to_chat_message


# ---------------------------------------------------------------------------
# Tier 1 — Scripted intent dispatch
# ---------------------------------------------------------------------------

class TestScriptedTier:
    def test_scripted_intent_message_returns_scripted_source(self):
        result = respond_to_chat_message("Which areas are most dangerous?")
        assert result["source"] == "scripted"

    def test_scripted_intent_includes_intent_name(self):
        result = respond_to_chat_message("Which areas are most dangerous?")
        assert result["intent"] == "top_danger_corridors"


# ---------------------------------------------------------------------------
# Tier 2 — ML cause detection fallback
# ---------------------------------------------------------------------------

class TestMlCauseTier:
    def test_drowsy_keyword_with_no_scripted_match_returns_ml_detected(self):
        # "I am drowsy when driving" matches no scripted intent but the cause
        # keyword "drowsy" fires the ML-cause tier with confidence >= 0.6.
        result = respond_to_chat_message("I am drowsy when driving")
        assert result["source"] == "ml_detected"


# ---------------------------------------------------------------------------
# Tier 3 — Groq LLM fallback
# ---------------------------------------------------------------------------

class TestGroqTier:
    def test_groq_called_when_no_scripted_or_ml_match(self):
        # "Tell me a joke" matches no scripted intent and no cause keyword.
        fake_groq = {"type": "ai", "message": "stubbed groq reply", "source": "groq"}

        with patch("src.chatbot.get_groq_response", return_value=fake_groq) as mock_groq:
            result = respond_to_chat_message("Tell me a joke")

        assert mock_groq.called
        assert result["message"] == "stubbed groq reply"
        assert result["source"] == "groq"


# ---------------------------------------------------------------------------
# Trip-planner CTA attachment
# ---------------------------------------------------------------------------

class TestTripPlannerCta:
    def test_route_safety_intent_attaches_trip_planner_action(self):
        result = respond_to_chat_message("What's the safest route to Asok?")
        actions = result.get("actions") or []
        assert any(
            isinstance(a, dict) and a.get("href") == "/trip-planner" for a in actions
        )


# ---------------------------------------------------------------------------
# Sanitization invariant
# ---------------------------------------------------------------------------

class TestSanitization:
    def test_emojis_stripped_from_returned_message(self):
        # A scripted handler that emits emojis still returns a clean message.
        result = respond_to_chat_message("Show high risk zones")
        # Common emoji ranges should never survive sanitisation.
        forbidden = "\U0001F600\U0001F680\U0001F6A8\U000026A0\U00002705"
        assert not any(ch in result["message"] for ch in forbidden)


# ---------------------------------------------------------------------------
# History argument safety
# ---------------------------------------------------------------------------

class TestHistoryArgument:
    def test_none_raw_history_does_not_raise(self):
        result = respond_to_chat_message("Which areas are most dangerous?", raw_history=None)
        assert result["source"] == "scripted"

    def test_omitted_raw_history_does_not_raise(self):
        result = respond_to_chat_message("Which areas are most dangerous?")
        assert result["source"] == "scripted"
