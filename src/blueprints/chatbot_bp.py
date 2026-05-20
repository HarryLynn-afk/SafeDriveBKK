"""
Chatbot API endpoints.

Routes:
    POST /api/chatbot
    GET  /api/chatbot/suggestions
"""

import logging

from flask import Blueprint, jsonify, request

from ..chatbot import respond_to_chat_message
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

chatbot_bp = Blueprint("chatbot", __name__, url_prefix="/api/chatbot")


@chatbot_bp.route("", methods=["POST"])
@rate_limit(max_calls=20, window_seconds=3600)
def chatbot():
    """Hybrid chatbot: scripted patterns → ML cause detection → Groq LLM."""
    try:
        data = request.get_json(silent=True) or {}
        user_message = str(data.get("message", "")).strip()

        if not user_message:
            return jsonify({"success": False, "error": "Empty message"}), 400

        response = respond_to_chat_message(user_message, raw_history=data.get("history"))
        return jsonify({"success": True, "response": response})

    except Exception:
        logger.exception("Unhandled error in /api/chatbot")
        return jsonify({
            "success": False,
            "error": "An error occurred. Please try asking your question differently.",
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
