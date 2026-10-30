"""Opt-in live smoke checks, not a comprehensive factuality evaluation.

Run: python -m pytest tests/test_chatbot_groq_live.py --live-groq -q -s
Requires GROQ_API_KEY in the environment or .env; consumes API quota.
"""

import re

import pytest

from src.chatbot_groq import get_groq_response
from src.chatbot_intents import load_chatbot_data
from src.config import GROQ_API_KEY


@pytest.fixture
def ask_groq(request):
    if not request.config.getoption("--live-groq"):
        pytest.skip("Live Groq checks require --live-groq")
    if not GROQ_API_KEY or GROQ_API_KEY == "gsk_your_key_here":
        pytest.fail("Configure GROQ_API_KEY before requesting live tests")

    def ask(question, history=None):
        response = get_groq_response(question, load_chatbot_data(), history)
        assert response.get("source") == "groq", response
        message = response["message"]
        assert isinstance(message, str) and message.strip()
        print(f"\nQuestion: {question}\nAnswer: {message}\n")
        return message.lower().replace("**", "").replace("’", "'")

    return ask


def test_live_dataset_total(ask_groq):
    answer = ask_groq("How many accident records are in the Bangkok dataset? Give the total.")
    assert re.search(r"\b2,?669\b", answer), answer


def test_live_missing_exposure_statistic(ask_groq):
    answer = ask_groq("Give the exact Bangkok electric-car crash rate per million kilometres driven from your dataset.")
    assert any(term in answer for term in ("not available", "does not", "doesn't", "cannot", "can't", "don't have", "do not have", "unavailable", "not include", "lacks")), answer
    # The requested exposure-based rate has no supporting data.
    assert not re.search(r"\d+(?:\.\d+)?\s*(?:crashes|accidents)\s+per", answer), answer


def test_live_route_redirect(ask_groq):
    answer = ask_groq("Give me turn-by-turn directions from Siam Paragon to Asok.")
    assert "trip planner" in answer, answer
    assert not re.search(r"turn (?:left|right)|\d+\s*(?:km|metres|meters)\b", answer), answer


def test_live_followup_preserves_context(ask_groq):
    answer = ask_groq("Repeat that total as a number, please.", [
        {"role": "user", "content": "How many accident records are in the Bangkok dataset?"},
        {"role": "assistant", "content": "The Bangkok dataset contains 2,669 accident records from 2024."},
    ])
    assert re.search(r"\b2,?669\b", answer), answer
