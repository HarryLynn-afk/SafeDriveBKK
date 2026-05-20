"""
Top-level chatbot entry point.

Owns the full "user message + history → sanitized response dict" pipeline.
Mirrors the role that `risk.score_route()` and `routing.resolve_route()`
play for their domains: a single deep function that the blueprint calls,
hiding the tier-by-tier fallback decisions inside.

Pipeline (first tier that returns wins):
    1. Scripted intent matcher  (chatbot_intents)
    2. ML cause detection + cause-specific stats  (chatbot_intents)
    3. Groq LLM with deterministic guardrails     (chatbot_groq)

Trip-planner CTA attachment and emoji/filler sanitization always run
on the final response.
"""

from __future__ import annotations

from . import extensions
from .chatbot_groq import get_groq_response
from .chatbot_intents import (
    detect_cause_from_question,
    detect_intent,
    get_scripted_response,
    handle_ml_cause_response,
)
from .chatbot_utils import (
    attach_trip_planner_action,
    needs_trip_planner_action,
    normalize_chat_history,
    sanitize_chatbot_payload,
)

_ML_CAUSE_CONFIDENCE_THRESHOLD = 0.6


def respond_to_chat_message(
    user_message: str,
    raw_history: list | None = None,
) -> dict:
    """Compute a sanitized chatbot response for the given user message.

    Returns the response dict ready to drop into the API envelope:
        {"success": True, "response": <returned dict>}.
    """
    history = normalize_chat_history(raw_history or [])
    intent = detect_intent(user_message)

    if intent:
        response = get_scripted_response(intent, user_message)
        response["source"] = "scripted"
        response["intent"] = intent
    else:
        response = _try_ml_cause_response(user_message) or get_groq_response(
            user_message,
            extensions.CHATBOT_DATA,
            conversation_history=history,
        )

    if needs_trip_planner_action(user_message, intent, response):
        response = attach_trip_planner_action(response)

    return sanitize_chatbot_payload(response)


def _try_ml_cause_response(user_message: str) -> dict | None:
    cause_key, confidence = detect_cause_from_question(user_message)
    if cause_key and confidence >= _ML_CAUSE_CONFIDENCE_THRESHOLD:
        return handle_ml_cause_response(cause_key, confidence, user_message)
    return None
