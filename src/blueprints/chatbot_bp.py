"""
Chatbot API endpoints.

Routes:
    POST /api/chatbot
    GET  /api/chatbot/suggestions
"""

from flask import Blueprint, jsonify, request

from .. import extensions
from ..chatbot_intents import (
    detect_cause_from_question,
    detect_intent,
    get_scripted_response,
    handle_ml_cause_response,
)
from ..chatbot_groq import get_groq_response
from ..chatbot_utils import (
    attach_trip_planner_action,
    needs_trip_planner_action,
    normalize_chat_history,
    sanitize_chatbot_payload,
)

chatbot_bp = Blueprint("chatbot", __name__, url_prefix="/api/chatbot")


@chatbot_bp.route("", methods=["POST"])
def chatbot():
    """Hybrid chatbot: scripted patterns → ML cause detection → Groq LLM."""
    try:
        data = request.get_json(silent=True) or {}
        user_message = str(data.get("message", "")).strip()
        history = normalize_chat_history(data.get("history", []))

        if not user_message:
            return jsonify({"success": False, "error": "Empty message"}), 400

        # Step 1: scripted intent
        intent = detect_intent(user_message)
        if intent:
            response = get_scripted_response(intent, user_message)
            response["source"] = "scripted"
            response["intent"] = intent
        else:
            # Step 2: ML cause detection
            cause_key, confidence = detect_cause_from_question(user_message)
            ml_response = None
            if cause_key and confidence >= 0.6:
                ml_response = handle_ml_cause_response(cause_key, confidence, user_message)

            if ml_response:
                response = ml_response
            else:
                # Step 3: Groq LLM
                response = get_groq_response(
                    user_message,
                    extensions.CHATBOT_DATA,
                    conversation_history=history,
                )

        if needs_trip_planner_action(user_message, intent, response):
            response = attach_trip_planner_action(response)

        return jsonify({"success": True, "response": sanitize_chatbot_payload(response)})

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc),
            "fallback": "I encountered an error. Please try asking your question differently or use one of the suggested questions.",
        }), 500


@chatbot_bp.route("/suggestions")
def suggestions():
    return jsonify({
        "success": True,
        "suggestions": [
            "Which areas are most dangerous?",
            "What time are accidents most common?",
            "What day is most dangerous for drivers?",
            "What causes most accidents?",
            "Which areas have the most speeding accidents?",
            "How reliable is the model?",
            "What should Bangkok do to improve road safety?",
            "Is Sukhumvit safe at night?",
        ],
    })
