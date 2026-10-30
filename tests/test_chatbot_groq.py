"""
Boundary tests for the chatbot_groq integration and deterministic guardrails.

These pure functions run *before* any Groq API call and encode the bot's policy:
what gets redirected as off-topic, which known data gaps get a canned answer,
what counts as an ambiguous follow-up, and which questions are served from the
fast cache. The integration test mocks HTTP; guardrail tests are pure functions.
"""

import json
from functools import partial

import httpx
from groq import Groq

from src import chatbot_groq
from src.chatbot_groq import (
    _check_fast_response_cache,
    _check_known_data_gap,
    _check_unrelated_topic,
    _dataset_limitations_context,
    _format_recent_history,
    _needs_clarification,
)


def test_open_ended_question_uses_available_model(monkeypatch):
    requests = []

    def groq_response(request):
        body = json.loads(request.content)
        requests.append(body)
        if body["model"] != "openai/gpt-oss-120b":
            return httpx.Response(404, json={"error": {"code": "model_not_found"}})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "Check your tyres before driving."}}],
        })

    with httpx.Client(transport=httpx.MockTransport(groq_response)) as http_client:
        monkeypatch.setattr(chatbot_groq, "GROQ_API_KEY", "test-key")
        monkeypatch.setattr(chatbot_groq, "Groq", partial(Groq, http_client=http_client))
        response = chatbot_groq.get_groq_response(
            "How can I prepare for a long drive?", {"stats": {"total_accidents": 2669}},
        )

    assert response["source"] == "groq"
    assert response["message"] == "Check your tyres before driving."
    assert requests[0]["reasoning_format"] == "hidden"


class TestCheckUnrelatedTopic:
    def test_off_topic_message_redirects(self):
        result = _check_unrelated_topic("tell me a joke about cats")
        assert result is not None
        assert result["type"] == "ai"
        assert result["source"] == "scripted"

    def test_road_safety_message_defers_to_llm(self):
        # A genuine road-safety question must NOT be hijacked (returns None).
        assert _check_unrelated_topic("is Sukhumvit road safe at night") is None

    def test_road_safety_term_overrides_unrelated_term(self):
        # "football" is off-topic, but "drive"/"safe" are road terms and win.
        assert _check_unrelated_topic("is it safe to drive to the football match") is None

    def test_empty_message_returns_none(self):
        assert _check_unrelated_topic("") is None
        assert _check_unrelated_topic(None) is None

    def test_neutral_message_without_either_term_returns_none(self):
        # Neither a road term nor a known off-topic term → defer, don't redirect.
        assert _check_unrelated_topic("hello there how are you") is None


class TestCheckKnownDataGap:
    def test_weather_plus_safety_returns_rain_guidance(self):
        result = _check_known_data_gap("is it dangerous to drive in the rain")
        assert result is not None
        assert "rain" in result["message"].lower()
        assert result["source"] == "scripted"

    def test_weather_without_safety_context_returns_none(self):
        # "rain" alone, no risk/safety word → not a known gap, defer to LLM.
        assert _check_known_data_gap("do you like rain") is None

    def test_taxi_comparison_returns_guidance(self):
        result = _check_known_data_gap("are taxis safer than my own car")
        assert result is not None
        assert result["type"] == "ai"

    def test_expressway_comparison_returns_guidance(self):
        result = _check_known_data_gap("compare expressway vs regular road risk")
        assert result is not None

    def test_plain_question_returns_none(self):
        assert _check_known_data_gap("what causes most accidents") is None


class TestNeedsClarification:
    def test_are_you_sure_is_ambiguous(self):
        assert _needs_clarification("are you sure?") is True

    def test_tell_me_more_is_ambiguous(self):
        assert _needs_clarification("tell me more") is True

    def test_which_one_is_ambiguous(self):
        assert _needs_clarification("which one is safer") is True

    def test_self_contained_question_is_not_ambiguous(self):
        assert _needs_clarification("what time of day is safest") is False


class TestCheckFastResponseCache:
    def test_keyword_hit_returns_cached_response(self):
        result = _check_fast_response_cache("what is the main cause of accidents")
        assert result is not None
        assert result["source"] == "cached"

    def test_no_keyword_returns_none(self):
        assert _check_fast_response_cache("some unrelated rambling text") is None

    def test_time_sensitive_hint_suppresses_cache(self):
        # "safest time to drive" IS a cache key, but "safest time" is a
        # time-sensitive hint, so the cache is intentionally bypassed.
        assert _check_fast_response_cache("what is the safest time to drive") is None

    def test_weekend_keyword_is_suppressed_as_time_sensitive(self):
        # "weekend" is both a cache key and a time-sensitive hint → suppressed.
        # (Documents that this cache entry can never fire.)
        assert _check_fast_response_cache("is the weekend dangerous") is None


class TestFormatRecentHistory:
    def test_empty_history_returns_empty_string(self):
        assert _format_recent_history([]) == ""
        assert _format_recent_history(None) == ""

    def test_formats_user_and_assistant_lines(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = _format_recent_history(history)
        assert "User: hi" in out
        assert "Assistant: hello" in out

    def test_skips_invalid_roles_and_empty_content(self):
        history = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": ""},
            {"role": "user", "content": "real question"},
        ]
        out = _format_recent_history(history)
        assert "ignored" not in out
        assert out.count("User:") == 1

    def test_truncates_to_max_messages(self):
        history = [{"role": "user", "content": f"msg {i}"} for i in range(12)]
        out = _format_recent_history(history, max_messages=8)
        assert out.count("User:") == 8
        assert "msg 11" in out  # most recent kept
        assert "msg 0" not in out  # oldest dropped


class TestDatasetLimitationsContext:
    def test_weather_question_notes_rain_limitation(self):
        out = _dataset_limitations_context("is it riskier when raining")
        assert "rain" in out.lower()

    def test_route_question_points_to_trip_planner(self):
        out = _dataset_limitations_context("give me directions to the airport")
        assert "Trip Planner" in out

    def test_unrelated_question_has_no_limitations(self):
        assert _dataset_limitations_context("what day is most dangerous") == ""
