"""
Chatbot response sanitization and action helpers.

Pure functions — no Flask, no I/O, no side-effects.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Pre-compiled patterns
# ---------------------------------------------------------------------------

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U000024C2-\U0001F251"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)

_BOLD_SEGMENT_PATTERN = re.compile(r"(\*\*.*?\*\*)", flags=re.DOTALL)

_FILLER_OPENING_PATTERN = re.compile(
    r"^\s*(?:(?:great|good)\s+question|sure|absolutely|of course|certainly|happy to help|"
    r"here(?:'s| is)(?:\s+(?:the answer|what I found))?|let me explain|let(?:'|')s break (?:this )?down)"
    r"[\s!,:.\-]+",
    flags=re.IGNORECASE,
)

_STAT_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b\d{1,2}:\d{2}(?:\s*[-–]\s*\d{1,2}:\d{2})\b|"
        r"\b\d{1,2}(?::\d{2})?(?:\s?(?:AM|PM|am|pm))?(?:\s*[-–]\s*\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm))\b|"
        r"\b\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)\b"
    ),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?(?:\s*[-–]\s*\d[\d,]*(?:\.\d+)?)?%"),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?(?:×|x)\b"),
    re.compile(r"\b\d[\d,]*(?:\.\d+)?/\d[\d,]*(?:\.\d+)?\b"),
    re.compile(r"\b\d+\s+out\s+of\s+\d+\b", flags=re.IGNORECASE),
    re.compile(r"#\d+\b"),
    re.compile(
        r"\b\d[\d,]*(?:\.\d+)?(?:\s*[-–]\s*\d[\d,]*(?:\.\d+)?)?\s*"
        r"(?:accidents?|deaths?|injuries?|incidents?|zones?|records?|hours?|days?|weekdays?|"
        r"weekends?|months?|years?|km²?|km|risk(?: score)?|accuracy|recall|lift)\b",
        flags=re.IGNORECASE,
    ),
]

_ROUTE_REQUEST_PATTERN = re.compile(
    r"\b(?:route|routes|directions?|navigate|navigation|safest way|best way)\b|"
    r"\bfrom\b.+\bto\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def _strip_emojis(text: str) -> str:
    cleaned = _EMOJI_PATTERN.sub("", text)
    cleaned = cleaned.replace("\u200d", "").replace("\ufe0f", "")
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _bold_stats(text: str, pattern: re.Pattern) -> str:
    """Wrap matched stat expressions in markdown bold, skipping existing bold spans."""
    segments = _BOLD_SEGMENT_PATTERN.split(text)
    for i, seg in enumerate(segments):
        if not seg or seg.startswith("**"):
            continue
        segments[i] = pattern.sub(lambda m: f"**{m.group(0).strip()}**", seg)
    return "".join(segments)


def _normalize_rich_text(text: str) -> str:
    """Strip emojis, remove filler openings, and bold numeric statistics."""
    cleaned = _strip_emojis(text)
    cleaned = _FILLER_OPENING_PATTERN.sub("", cleaned).strip()
    for pattern in _STAT_PATTERNS:
        cleaned = _bold_stats(cleaned, pattern)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# ---------------------------------------------------------------------------
# Public sanitization
# ---------------------------------------------------------------------------

def sanitize_chatbot_payload(payload, rich_text: bool = False):
    """Recursively sanitize chatbot responses before returning them to the UI."""
    if isinstance(payload, str):
        return _normalize_rich_text(payload) if rich_text else _strip_emojis(payload)
    if isinstance(payload, list):
        return [sanitize_chatbot_payload(item, rich_text=rich_text) for item in payload]
    if isinstance(payload, dict):
        return {
            key: sanitize_chatbot_payload(value, rich_text=key in {"message", "followup"})
            for key, value in payload.items()
        }
    return payload


# ---------------------------------------------------------------------------
# Trip-planner action helpers
# ---------------------------------------------------------------------------

def needs_trip_planner_action(
    user_message: str,
    intent_name: str | None = None,
    response=None,
) -> bool:
    """Return True when the reply should include a Trip Planner CTA."""
    if intent_name == "route_safety":
        return True
    if _ROUTE_REQUEST_PATTERN.search(user_message or ""):
        return True
    if isinstance(response, dict):
        combined = " ".join(
            str(response.get(k, "") or "") for k in ("message", "followup")
        ).lower()
        if "trip planner" in combined:
            return True
    return False


def attach_trip_planner_action(response: dict) -> dict:
    """Add a stable Trip Planner CTA without duplicating existing actions."""
    if not isinstance(response, dict):
        return response
    actions = list(response.get("actions") or [])
    if not any(
        isinstance(a, dict) and a.get("href") == "/trip-planner" for a in actions
    ):
        actions.append({"label": "Open Trip Planner", "href": "/trip-planner", "style": "primary"})
    response["actions"] = actions
    return response


# ---------------------------------------------------------------------------
# History normalisation
# ---------------------------------------------------------------------------

def normalize_chat_history(raw_history, max_messages: int = 8) -> list[dict]:
    """Normalise and truncate conversation history before sending to the LLM."""
    if not isinstance(raw_history, list):
        return []
    normalized = []
    for item in raw_history[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content[:1200]})
    return normalized
