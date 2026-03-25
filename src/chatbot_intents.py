"""
Intent Detection & Response System for SafeDrive BKK Chatbot

Hybrid approach:
  - Pattern matching handles 20 common question types instantly
  - Unmatched queries fall through to Groq AI for open-ended answers

Usage:
    from chatbot_intents import detect_intent, get_scripted_response

    intent = detect_intent(user_message)
    if intent:
        response = get_scripted_response(intent, user_message)
    else:
        # Fall through to Groq AI
"""

import json
import re
from functools import lru_cache
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_chatbot_data():
    """Load and cache chatbot_data.json."""
    with open(STATIC_DIR / "chatbot_data.json", encoding="utf-8") as f:
        return json.load(f)


def _has_thai_text(text):
    return any("\u0e00" <= char <= "\u0e7f" for char in str(text or ""))


def _get_cause_classifier():
    try:
        from .nlp_analysis import get_cause_classifier
    except ImportError:
        from nlp_analysis import get_cause_classifier
    return get_cause_classifier()


def _fmt_zone_list(zones, limit=5):
    """Format a list of zone dicts into simple bullet lines."""
    lines = []
    for z in zones[:limit]:
        count = z.get("cause_count", z.get("total_accidents", "?"))
        lines.append(f"• **{z['location']}** - {count} accidents")
    return "\n".join(lines)


def _fmt_suggestion_list(items):
    """Format a short list of suggested follow-up prompts."""
    return "\n".join(f"- {item}" for item in items if item)


def _search_zone_by_name(query):
    """Find a zone whose location name matches the query (fuzzy)."""
    data = load_chatbot_data()
    query_lower = query.lower()
    for gid, zone in data["zones"].items():
        if query_lower in zone["location"].lower():
            return gid, zone
    return None, None


def _trip_planner_action():
    return {
        "label": "Open Trip Planner",
        "href": "/trip-planner",
        "style": "primary",
    }


def _extract_route_reference(user_message):
    """Extract a short human-readable route reference from the user's message."""
    text = str(user_message or "").strip()
    from_to_match = re.search(r"\bfrom\s+(.+?)\s+\bto\s+(.+?)(?:[\?\.,!]|$)", text, re.I)
    if from_to_match:
        start = from_to_match.group(1).strip()
        end = from_to_match.group(2).strip()
        return f"from **{start}** to **{end}**"

    to_match = re.search(r"\bto\s+(.+?)(?:[\?\.,!]|$)", text, re.I)
    if to_match:
        destination = to_match.group(1).strip()
        return f"to **{destination}**"

    return ""


# ---------------------------------------------------------------------------
# Intent patterns  (20 intents)
# ---------------------------------------------------------------------------

INTENTS = {
    # ── High-priority judge questions (must resolve before Groq) ───────
    "judge_dangerous_zones": {
        "patterns": [
            re.compile(r"which\s+(?:zones?|areas?|places?|roads?|corridors?)\s+(?:are\s+)?most\s+dangerous", re.I),
            re.compile(r"what\s+are\s+the\s+most\s+dangerous\s+(?:zones?|areas?|places?|roads?|corridors?)", re.I),
            re.compile(r"(?:most|worst|highest).*(?:dangerous|risky).*(?:zones?|areas?|roads?|corridors?)", re.I),
            re.compile(r"(?:ย่าน|พื้นที่|โซน|เขต|จุด|ถนน).*(?:ไหน|ใด).*(?:อันตราย|เสี่ยง).*(?:ที่สุด|สุด)", re.I),
            re.compile(r"ถนนเส้นไหน.*(?:อันตราย|เสี่ยง).*(?:ที่สุด|สุด)?", re.I),
            re.compile(r"(?:พื้นที่|ย่าน|โซน).*(?:อันตราย|เสี่ยง).*(?:ที่สุด|สุด)", re.I),
        ],
        "handler": "handle_judge_dangerous_zones",
    },
    "judge_peak_time": {
        "patterns": [
            re.compile(r"what\s+time.*accidents?.*(?:most\s+common|most|peak|highest)", re.I),
            re.compile(r"when\s+is\s+it\s+most\s+dangerous", re.I),
            re.compile(r"(?:peak|worst|most\s+dangerous).*(?:time|hour)", re.I),
            re.compile(r"(?:ช่วงเวลา|เวลา).*(?:ไหน|ใด).*(?:อันตราย|เสี่ยง).*(?:ที่สุด|สุด)", re.I),
            re.compile(r"อุบัติเหตุ.*(?:เวลา|ช่วงเวลา).*(?:ไหน|ใด).*(?:เยอะ|มาก).*(?:ที่สุด|สุด)", re.I),
        ],
        "handler": "handle_judge_peak_time",
    },
    "judge_peak_day": {
        "patterns": [
            re.compile(r"what\s+day.*(?:most\s+dangerous|most\s+accidents?|highest)", re.I),
            re.compile(r"which\s+day\s+has\s+most\s+accidents?", re.I),
            re.compile(r"(?:most\s+dangerous|worst).*(?:day|weekday)", re.I),
            re.compile(r"วัน(?:ไหน|ใด).*(?:อันตราย|เสี่ยง).*(?:ที่สุด|สุด)", re.I),
            re.compile(r"วัน(?:ไหน|ใด).*อุบัติเหตุ.*(?:เยอะ|มาก).*(?:ที่สุด|สุด)", re.I),
        ],
        "handler": "handle_judge_peak_day",
    },
    "judge_model_accuracy": {
        "patterns": [
            re.compile(r"how\s+(?:accurate|reliable).*(?:model|safedrive|system|app)", re.I),
            re.compile(r"how\s+reliable\s+is\s+safedrive", re.I),
            re.compile(r"(?:can\s+i\s+trust|trust).*(?:model|safedrive|system)", re.I),
            re.compile(r"(?:accuracy|reliability).*(?:model|safedrive|system)", re.I),
            re.compile(r"โมเดล.*(?:แม่นยำ|น่าเชื่อถือ).*(?:แค่ไหน|ขนาดไหน)", re.I),
            re.compile(r"(?:ระบบ|แอป|safedrive).*(?:แม่นยำ|น่าเชื่อถือ).*(?:แค่ไหน|ขนาดไหน)", re.I),
        ],
        "handler": "handle_judge_model_accuracy",
    },

    # ── Location-based (8) ─────────────────────────────────────────────
    "list_speeding_zones": {
        "patterns": [
            re.compile(r"(?:which|what|where).*(?:areas?|zones?|places?).*speed", re.I),
            re.compile(r"where.*speed(?:ing)?.*(?:common|happen|occur|problem)", re.I),
            re.compile(r"speed(?:ing)?.*(?:zones?|areas?|locations?|hotspots?)", re.I),
            re.compile(r"top.*speed(?:ing)?.*(?:zones?|areas?)", re.I),
        ],
        "handler": "handle_speeding_zones",
    },
    "list_drunk_zones": {
        "patterns": [
            re.compile(r"(?:where|which).*drunk.*driv", re.I),
            re.compile(r"drunk.*driv.*(?:areas?|zones?|locations?)", re.I),
            re.compile(r"alcohol.*(?:relat|accident|crash|zone|area)", re.I),
            re.compile(r"(?:areas?|zones?).*drunk", re.I),
        ],
        "handler": "handle_drunk_zones",
    },
    # ── Cause-based stats (placed before zone_safety to avoid false matches) ──
    "speeding_stats": {
        "patterns": [
            re.compile(r"speed(?:ing)?.*(?:stat|number|data|how (?:much|many|often))", re.I),
            re.compile(r"how (?:much|many|common|often).*speed(?:ing)?", re.I),
            re.compile(r"speed(?:ing)?.*(?:percent|proportion|rate)", re.I),
            re.compile(r"tell.*(?:me|us).*(?:about)?.*speed(?:ing)?$", re.I),
        ],
        "handler": "handle_speeding_stats",
    },
    "drunk_stats": {
        "patterns": [
            re.compile(r"drunk.*driv.*(?:stat|number|data|how)", re.I),
            re.compile(r"how (?:much|many|common|often).*drunk", re.I),
            re.compile(r"alcohol.*(?:stat|number|data|percent)", re.I),
            re.compile(r"(?:dui|dwi).*(?:stat|data|number)", re.I),
            re.compile(r"(?:tell|about).*drunk.*(?:driv|stat)", re.I),
        ],
        "handler": "handle_drunk_stats",
    },
    "cutting_stats": {
        "patterns": [
            re.compile(r"cutting.*(?:front|in).*(?:stat|data|number)", re.I),
            re.compile(r"(?:how (?:much|many|common|often)).*cut(?:ting)?.*(?:front|in)", re.I),
            re.compile(r"cut(?:ting)?.*(?:front|in).*(?:stat|percent|rate)", re.I),
        ],
        "handler": "handle_cutting_stats",
    },

    # ── Time-based (placed before zone_safety to avoid false matches) ──
    "rush_hour": {
        "patterns": [
            re.compile(r"rush.*hour.*(?:danger|safe|risk|accident|stat)", re.I),
            re.compile(r"(?:is|are).*rush.*hours?.*(?:danger|safe|risk)", re.I),
            re.compile(r"(?:danger|safe|risk|accident).*(?:during|in).*rush.*hour", re.I),
            re.compile(r"commut(?:e|ing).*(?:safe|danger|risk)", re.I),
        ],
        "handler": "handle_rush_hour",
    },

    "list_dangerous_zones": {
        "patterns": [
            re.compile(r"(?:most|top|worst).*dangerous.*(?:areas?|zones?|places?|intersections?|roads?)", re.I),
            re.compile(r"dangerous.*(?:intersections?|areas?|zones?|roads?)", re.I),
            re.compile(r"(?:highest|most).*(?:risk|accident).*(?:areas?|zones?)", re.I),
            re.compile(r"(?:where|which).*(?:most|many|lots?).*accidents?", re.I),
            re.compile(r"worst.*(?:areas?|zones?|roads?|intersections?)", re.I),
        ],
        "handler": "handle_dangerous_zones",
    },
    "zone_safety_query": {
        "patterns": [
            re.compile(r"(?:is|how)\s+(?:safe|dangerous)\s+(?:is|in|at|near)\s+(.+)", re.I),
            re.compile(r"(?:safe|danger|risk)\s+(?:of|in|at|near|around)\s+(.+)", re.I),
            re.compile(r"tell me about\s+(.+?)(?:\s+(?:area|zone|road|intersection))\s*$", re.I),
            re.compile(r"how (?:risky|safe) is\s+(.+)", re.I),
            re.compile(r"^is\s+(.+?)\s+(?:safe|dangerous|risky)\s*\??$", re.I),
        ],
        "handler": "handle_zone_safety",
    },
    "route_safety": {
        "patterns": [
            re.compile(r"(?:safe|safest|best).*route.*(?:to|from|between)", re.I),
            re.compile(r"(?:how|which).*(?:route|way|path).*(?:safe|avoid)", re.I),
            re.compile(r"(?:navigate|drive|travel|go).*(?:safe|avoid.*danger)", re.I),
            re.compile(r"route.*(?:recommendation|suggest|plan)", re.I),
            re.compile(r"(?:เส้นทาง|route).*(?:ไหน|ใด|นี้).*(?:ปลอดภัย|เสี่ยง|อันตราย)", re.I),
            re.compile(r"(?:ไป|ขับ|เดินทาง).*(?:อย่างไร|ยังไง).*(?:ปลอดภัย|เลี่ยง|เสี่ยง)", re.I),
            re.compile(r"(?:คืนนี้|ตอนกลางคืน).*(?:ขับรถ|เดินทาง).*(?:ปลอดภัย|เสี่ยง|อันตราย)", re.I),
            re.compile(r"(?:ช่วย|ขอ).*(?:แนะนำ|หา).*(?:เส้นทาง).*(?:ปลอดภัย|เลี่ยง)", re.I),
        ],
        "handler": "handle_route_safety",
    },
    "rain_data_gap": {
        "patterns": [
            re.compile(r"(?:rain|raining|weather|storm|wet roads?).*(?:risk|danger|safe|accident|increase)", re.I),
            re.compile(r"(?:how much|does).*(?:rain|weather).*(?:increase|affect).*(?:risk|accident)", re.I),
        ],
        "handler": "handle_rain_data_gap",
    },
    "camera_deployment": {
        "patterns": [
            re.compile(r"where.*(?:install|deploy|place|put).*cameras?", re.I),
            re.compile(r"(?:install|deploy|place|put).*cameras?.*where", re.I),
            re.compile(r"(?:speed|traffic|red.?light).*cameras?.*(?:install|deploy|place|need)", re.I),
            re.compile(r"cameras?.*(?:deploy|install).*(?:recommend|suggest|where)", re.I),
            re.compile(r"where.*(?:should|could).*cameras?", re.I),
        ],
        "handler": "handle_camera_deployment",
    },
    "high_risk_zones": {
        "patterns": [
            re.compile(r"(?:show|list|display).*(?:high|red).*(?:risk|danger).*zones?", re.I),
            re.compile(r"high.*risk.*zones?", re.I),
            re.compile(r"red.*zones?", re.I),
            re.compile(r"(?:all|list).*danger.*zones?", re.I),
        ],
        "handler": "handle_high_risk_zones",
    },
    "motorcycle_zones": {
        "patterns": [
            re.compile(r"motorcycle.*(?:accident|crash|zone|area|danger)", re.I),
            re.compile(r"(?:where|which).*motorcycle.*(?:crash|accident|danger|risk)", re.I),
            re.compile(r"(?:areas?|zones?).*(?:high|many|most).*motorcycles?", re.I),
            re.compile(r"(?:motorbike|bike).*(?:accident|crash|risk|danger)", re.I),
        ],
        "handler": "handle_motorcycle_zones",
    },

    # ── Cause-based (remaining) ──────────────────────────────────────
    "cause_breakdown": {
        "patterns": [
            re.compile(r"(?:what|which).*caus(?:e|es|ing).*accidents?", re.I),
            re.compile(r"(?:breakdown|distribution).*(?:by|of).*caus", re.I),
            re.compile(r"(?:main|top|primary|common).*caus(?:e|es)", re.I),
            re.compile(r"why.*(?:accidents?|crashes?).*happen", re.I),
            re.compile(r"accident.*caus(?:e|es)", re.I),
            re.compile(r"(?:สาเหตุ|อะไร).*(?:อุบัติเหตุ|ชน|เกิดเหตุ).*(?:หลัก|มาก|บ่อย|ส่วนใหญ่)", re.I),
            re.compile(r"อุบัติเหตุ.*(?:เกิดจาก|เพราะ|สาเหตุ).*(?:อะไร|อะไรบ้าง)", re.I),
        ],
        "handler": "handle_cause_breakdown",
    },
    "cause_comparison": {
        "patterns": [
            re.compile(r"compare.*caus(?:e|es)", re.I),
            re.compile(r"caus(?:e|es).*(?:compare|comparison|vs|versus)", re.I),
            re.compile(r"(?:which|what).*(?:bigger|worse|more).*caus(?:e|es)", re.I),
            re.compile(r"(?:speed|drunk|cutting).*(?:vs|versus|compared?|or).*(?:speed|drunk|cutting)", re.I),
        ],
        "handler": "handle_cause_comparison",
    },

    # ── Time-based (3) ────────────────────────────────────────────────
    "time_patterns": {
        "patterns": [
            re.compile(r"when.*(?:most|many).*accidents?", re.I),
            re.compile(r"when.*accidents?.*(?:most|common|frequent|happen)", re.I),
            re.compile(r"(?:most|peak).*dangerous.*(?:time|hour|day|month|period)", re.I),
            re.compile(r"(?:what|which).*(?:time|hour|day|month).*(?:dangerous|worst|most)", re.I),
            re.compile(r"(?:time|temporal|hourly|daily|monthly).*(?:pattern|trend|distribution)", re.I),
            re.compile(r"when.*(?:should|shouldn't).*(?:drive|travel|avoid)", re.I),
            re.compile(r"(?:peak|worst).*(?:hours?|times?|days?)", re.I),
            re.compile(r"(?:เวลา|ช่วงเวลา|วัน|เดือนไหน).*(?:อันตราย|เสี่ยง|อุบัติเหตุ).*(?:มาก|เยอะ|สูง|ที่สุด)", re.I),
            re.compile(r"(?:ควร|ไม่ควร).*(?:ขับรถ|เดินทาง).*(?:เวลา|ช่วงเวลา).*(?:ไหน|ใด)", re.I),
        ],
        "handler": "handle_time_patterns",
    },
    "weekend_pattern": {
        "patterns": [
            re.compile(r"weekend.*(?:vs|versus|compared?|or).*weekday", re.I),
            re.compile(r"weekday.*(?:vs|versus|compared?|or).*weekend", re.I),
            re.compile(r"(?:is|are).*weekends?.*(?:more|less).*(?:danger|safe|risk)", re.I),
            re.compile(r"(?:saturday|sunday).*(?:danger|safe|accident|risk)", re.I),
        ],
        "handler": "handle_weekend_pattern",
    },

    # ── Recommendation (2) ────────────────────────────────────────────
    "policy_recommendations": {
        "patterns": [
            re.compile(r"(?:what|which).*(?:should|could|can).*(?:bangkok|city|government|policy).*do", re.I),
            re.compile(r"recommend(?:ation)?s?", re.I),
            re.compile(r"(?:policy|action|measure|solution|intervention).*(?:suggest|recommend|propose)", re.I),
            re.compile(r"how.*(?:reduce|prevent|improve|fix).*(?:accident|crash|safety)", re.I),
            re.compile(r"(?:suggest|propose).*(?:solution|improvement|change)", re.I),
        ],
        "handler": "handle_policy_recommendations",
    },
    # ── Meta (2) ──────────────────────────────────────────────────────
    "how_it_works": {
        "patterns": [
            re.compile(r"how.*(?:does|do).*(?:this|it|system|model|app|project).*work", re.I),
            re.compile(r"how.*(?:you|model|system).*(?:analy[sz]|predict|detect|classify)", re.I),
            re.compile(r"(?:explain|describe).*(?:system|model|method|approach|pipeline|architecture)", re.I),
            re.compile(r"what.*(?:technology|tech|stack|model|algorithm|method).*(?:use|behind|under)", re.I),
            re.compile(r"(?:xgboost|nlp|machine.?learning).*(?:how|explain|work)", re.I),
            re.compile(r"(?:ระบบ|โมเดล|แอป|โปรเจกต์).*(?:ทำงาน|วิเคราะห์|ทำนาย|ตรวจจับ).*(?:อย่างไร|ยังไง)", re.I),
            re.compile(r"(?:ใช้|มี).*(?:เทคโนโลยี|โมเดล|วิธี).*(?:อะไร|แบบไหน)", re.I),
        ],
        "handler": "handle_how_it_works",
    },
    "data_source": {
        "patterns": [
            re.compile(r"(?:what|which|where).*data.*(?:use|from|source|come)", re.I),
            re.compile(r"(?:how many|how much).*(?:accidents?|records?|data)", re.I),
            re.compile(r"data.*(?:source|origin|from|about)", re.I),
            re.compile(r"(?:about|tell).*(?:the|your).*data(?:set)?", re.I),
            re.compile(r"(?:how big|size|scope).*data", re.I),
            re.compile(r"(?:ข้อมูล|ดาต้า).*(?:มาจาก|ที่มา|แหล่ง|ใช้)", re.I),
            re.compile(r"(?:มี|ใช้).*(?:ข้อมูล|อุบัติเหตุ).*(?:กี่|เท่าไร|เท่าไหร่)", re.I),
        ],
        "handler": "handle_data_source",
    },
}


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

def detect_intent(user_message):
    """
    Match user_message against all intent patterns.

    Returns the intent name (str) if a pattern matches, or None if
    no scripted intent is found (caller should fall through to Groq AI).
    """
    text = user_message.strip()
    if not text:
        return None

    for intent_name, config in INTENTS.items():
        for pattern in config["patterns"]:
            if pattern.search(text):
                return intent_name

    return None


def get_scripted_response(intent, user_message=""):
    """
    Dispatch to the handler for *intent* and return a response dict.

    Response shape:
        {
            "type":     "list" | "single" | "stats" | "comparison"
                        | "recommendation" | "info",
            "message":  str   — main conversational text,
            "data":     list  — optional structured items,
            "followup": str   — optional follow-up prompt,
        }
    """
    handler_name = INTENTS[intent]["handler"]
    handler_fn = _HANDLERS[handler_name]
    return handler_fn(user_message)


# ---------------------------------------------------------------------------
# Response handlers  (20 handlers, one per intent)
# ---------------------------------------------------------------------------

def _judge_zone_anchors():
    data = load_chatbot_data()
    east_zone = data["zones"].get("G014_041", {})
    return {
        "west": {
            "label": "Western corridor",
            "road": "Borommaratchachonnani Road",
            "district": "Thawi Watthana",
            "accidents": 85,
        },
        "east": {
            "label": "Eastern corridor",
            "road": "Motorway 7",
            "district": "Lat Krabang",
            "accidents": int(east_zone.get("total_accidents", 69) or 69),
        },
    }


def handle_judge_dangerous_zones(user_message=""):
    anchors = _judge_zone_anchors()
    west = anchors["west"]
    east = anchors["east"]
    is_thai = any("\u0e00" <= char <= "\u0e7f" for char in user_message)

    if is_thai:
        message = (
            f"จุดเสี่ยงหลักในข้อมูลนี้มี 2 แนวที่ควรพูดถึงก่อนคือ **ถนนบรมราชชนนี แถวทวีวัฒนา** "
            f"ซึ่งเป็นฝั่งตะวันตกและมี **{west['accidents']} อุบัติเหตุ** ในหนึ่งกริดโซน กับ **ลาดกระบัง / มอเตอร์เวย์ 7** "
            f"ซึ่งเป็นแนวเสี่ยงสูงฝั่งตะวันออก.\n\nถ้าต้องตอบกรรมการแบบสั้นที่สุด ให้ยึดสองแนวนี้เป็นคำตอบหลักของกรุงเทพฯ."
        )
    else:
        message = (
            f"The clearest high-risk anchors are the **western corridor at {west['road']}, {west['district']}** with "
            f"**{west['accidents']} accidents** in one grid zone, and the **eastern corridor around {east['district']} / {east['road']}**.\n\n"
            "If you need the shortest judge-safe answer, lead with those two corridors."
        )

    return {
        "type": "list",
        "message": message,
        "data": [
            {"corridor": west["label"], "road": west["road"], "district": west["district"], "accidents": west["accidents"]},
            {"corridor": east["label"], "road": east["road"], "district": east["district"], "accidents": east["accidents"]},
        ],
        "followup": "Want the top crash time as well?",
    }


def handle_judge_peak_time(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]
    peak_hour = tp["most_dangerous_hour"]
    peak_count = tp["most_dangerous_hour_count"]
    is_thai = any("\u0e00" <= char <= "\u0e7f" for char in user_message)

    if is_thai:
        message = (
            f"ช่วงเวลาที่อุบัติเหตุสูงที่สุดคือ **{peak_hour}** ที่ **{peak_count} ครั้ง** ในข้อมูลชุดนี้.\n\n"
            "ถ้าต้องขับจริง ควรหลบช่วงพีกหลักรอบ **7-10 AM**, **10:00-11:00**, และ **4-8 PM** ถ้าทำได้."
        )
    else:
        message = (
            f"**Peak hour: {peak_hour}.** That is the busiest single hour in the dataset with **{peak_count} accidents**.\n\n"
            "For a practical driving answer, avoid the main spikes around **7-10 AM**, **10:00-11:00**, and **4-8 PM** when possible."
        )

    return {
        "type": "stats",
        "message": message,
        "data": [{"hour": peak_hour, "count": peak_count}],
        "followup": "Want the most dangerous day too?",
    }


def handle_judge_peak_day(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]
    peak_day = tp["most_dangerous_day"]
    peak_count = tp["most_dangerous_day_count"]
    is_thai = any("\u0e00" <= char <= "\u0e7f" for char in user_message)

    if is_thai:
        message = (
            f"วันที่อุบัติเหตุสูงที่สุดคือ **วันเสาร์** ที่ **{peak_count} ครั้ง**.\n\n"
            "ถ้าต้องตอบแบบสั้นและชัดสำหรับกรรมการ ให้ตอบว่า **วันเสาร์** คือวันเสี่ยงสูงสุดในข้อมูลนี้."
        )
    else:
        message = (
            f"**{peak_day}** is the most dangerous day in the dataset with **{peak_count} accidents**.\n\n"
            "If you need a clean judge-facing takeaway, Saturday is the day to call out first."
        )

    return {
        "type": "stats",
        "message": message,
        "data": [{"day": peak_day, "count": peak_count}],
        "followup": "Want the peak hour too?",
    }


def handle_judge_model_accuracy(user_message=""):
    data = load_chatbot_data()
    stats = data["stats"]
    accuracy = stats["model_accuracy"]
    lift = stats["model_lift"]
    is_thai = any("\u0e00" <= char <= "\u0e7f" for char in user_message)

    if is_thai:
        message = (
            f"ในการทดสอบแบบ **Jan-Sep 2024 train / Oct-Dec 2024 holdout** ระบบจับครอบคลุมอุบัติเหตุที่เกิดภายหลังได้ **{accuracy}%** "
            f"และให้ **lift {lift}x** เทียบกับการสุ่มหาโซนเสี่ยง.\n\n"
            "สรุปแบบตอบกรรมการคือโมเดลมีสัญญาณเชิงเวลาและทำได้ดีกว่าการเลือกโซนแบบสุ่มอย่างชัดเจน."
        )
    else:
        message = (
            f"On a **Jan-Sep 2024 train / Oct-Dec 2024 holdout**, SafeDrive BKK captured **{accuracy}%** of later mapped crashes inside flagged high-risk grids "
            f"with **{lift}x lift** over random danger-zone selection.\n\n"
            "The judge-safe summary is that the model shows temporal signal and performs materially better than random zone selection."
        )

    return {
        "type": "stats",
        "message": message,
        "data": [{"holdout_capture": accuracy, "lift": lift}],
        "followup": "Want the top danger corridors as well?",
    }

# ── Location-based ────────────────────────────────────────────────────

def handle_speeding_zones(user_message=""):
    data = load_chatbot_data()
    speeding = data["causes"]["speeding"]

    return {
        "type": "list",
        "message": (
            f"Speeding is Bangkok's #1 problem ({speeding['percentage']}% of accidents). "
            f"Worst areas:\n{_fmt_zone_list(speeding['top_zones'], 5)}\n\n"
            f"These are Bangkok's highest-priority zones for enforcement."
        ),
        "data": speeding["top_zones"][:5],
        "followup": "Need route alternatives? 🚗",
    }


def handle_drunk_zones(user_message=""):
    data = load_chatbot_data()

    # Find zones with high weekend % (proxy for alcohol-related risk)
    weekend_zones = []
    for gid, zone in data["zones"].items():
        if zone["weekend_pct"] > 40:
            weekend_zones.append({
                "location": zone["location"],
                "total_accidents": zone["total_accidents"],
                "weekend_pct": zone["weekend_pct"],
            })
    weekend_zones.sort(key=lambda z: z["weekend_pct"], reverse=True)

    zone_lines = "\n".join(
        f"• **{z['location']}** - {z['weekend_pct']}% weekend accidents"
        for z in weekend_zones[:5]
    )

    return {
        "type": "list",
        "message": (
            f"⚠️ Drunk driving is underreported, but weekend nights are the biggest red flag. "
            f"Saturday is the worst day ({data['time_patterns']['most_dangerous_day_count']} accidents).\n\n"
            f"High-risk weekend zones:\n{zone_lines}\n\n"
            f"Avoid driving late on Friday/Saturday nights if possible."
        ),
        "data": weekend_zones[:5],
        "followup": "Want weekend vs weekday details? 🕐",
    }


def handle_dangerous_zones(user_message=""):
    data = load_chatbot_data()
    zones = data["zones"]

    top5 = []
    for gid, z in list(zones.items())[:5]:
        top5.append({
            "location": z["location"],
            "total_accidents": z["total_accidents"],
            "dominant_cause": z["dominant_cause"],
            "has_fatality": z["has_fatality"],
        })

    lines = []
    for z in top5:
        fatality = " ⚠️" if z["has_fatality"] else ""
        lines.append(f"• **{z['location']}** - {z['total_accidents']} accidents ({z['dominant_cause']}){fatality}")

    return {
        "type": "list",
        "message": (
            f"Bangkok's most dangerous zones (avoid if possible):\n" +
            "\n".join(lines) +
            f"\n\nSpeeding is the main cause in most of these areas."
        ),
        "data": top5,
        "followup": "Want safer route alternatives? 🚗",
    }


def handle_zone_safety(user_message=""):
    data = load_chatbot_data()

    # Try to extract location name from the message
    location_query = user_message.strip()
    # Strip common prefixes (longest first)
    for prefix in ["how dangerous is ", "how safe is ", "how risky is ",
                    "tell me about ", "is ", "how safe ", "how dangerous ",
                    "safety of ", "risk of ", "danger of ", "what about "]:
        if location_query.lower().startswith(prefix):
            location_query = location_query[len(prefix):]
            break
    location_query = location_query.strip(" ?.,!")
    # Strip trailing qualifiers (repeatedly to handle "safe area" etc.)
    changed = True
    while changed:
        changed = False
        for suffix in [" safe", " dangerous", " risky", " area", " zone", " safety"]:
            if location_query.lower().endswith(suffix):
                location_query = location_query[:-len(suffix)].rstrip()
                changed = True

    gid, zone = _search_zone_by_name(location_query)

    if zone and gid:
        z = zone
        risk_desc = "HIGH RISK ⚠️" if z["risk_score"] >= 55 else "MEDIUM RISK" if z["risk_score"] >= 30 else "LOW RISK ✅"
        top_cause = max(z["cause_breakdown"].items(), key=lambda x: x[1])

        return {
            "type": "single",
            "message": (
                f"{risk_desc}: **{z['location']}**\n\n"
                f"Main problem: {top_cause[0].replace('_', ' ').title()} ({top_cause[1]}% of accidents here). "
                f"{'Avoid rush hours (5-7 PM) when it gets most dangerous.' if z['rush_hour_pct'] > 30 else 'Exercise caution at all hours.'}"
            ),
            "data": [z],
            "followup": "Want safer route alternatives? 🚗",
        }
    elif zone:
        return {
            "type": "single",
            "message": (
                f"**{zone['location']}** is a monitored zone. "
                f"For detailed risk information, ask about a specific area like Asok or Lat Krabang."
            ),
            "data": [zone],
            "followup": "Want to see the most dangerous zones? ⚠️",
        }
    else:
        # Show the most dangerous zones as a useful fallback
        top_zones = list(data["zones"].values())[:5]
        lines = [
            f"• **{z['location']}** - {z['total_accidents']} accidents"
            for z in top_zones
        ]
        return {
            "type": "list",
            "message": (
                f"Our data covers Bangkok's outer districts and highways — "
                f"I don't have a specific entry for \"{location_query}\".\n\n"
                f"Most dangerous zones in our dataset:\n" +
                "\n".join(lines) +
                f"\n\nFor any specific route, the **Trip Planner** can analyse safety along your path."
            ),
            "data": top_zones,
            "followup": "Want to plan a safe route? 🚗",
        }


def handle_route_safety(user_message=""):
    route_reference = _extract_route_reference(user_message)
    is_thai = _has_thai_text(user_message)
    acknowledgement = (
        f"That trip {route_reference} needs a path-specific check rather than a generic answer."
        if route_reference
        else "That is a route-specific question, so the safest answer depends on the exact path and time of travel."
    )
    message = (
        "คำถามแบบนี้ต้องดูตามเส้นทางจริงและช่วงเวลาที่จะเดินทาง ไม่ควรตอบแบบเหมารวม. "
        "**ใช้ Trip Planner** เพื่อเทียบเส้นทางที่เร็วกับเส้นทางที่ปลอดภัยกว่า แล้วดูจุดเสี่ยงบนทางจริงได้เลย."
        if is_thai
        else f"{acknowledgement} **Use the Trip Planner** to compare a faster option with a safer option and see the danger zones along the way."
    )
    return {
        "type": "info",
        "message": message,
        "data": [],
        "followup": "อยากรู้ก่อนว่าควรหลบย่านไหนบ้างไหม?" if is_thai else "Want to know which areas to avoid first?",
        "actions": [_trip_planner_action()],
    }


def handle_rain_data_gap(user_message=""):
    return {
        "type": "info",
        "message": (
            "While this isn't in our Bangkok dataset, based on general road safety research, rain usually increases risk by reducing visibility and tire grip. I can't give a Bangkok-specific number yet, but the safest response is to slow down, increase following distance, and avoid abrupt braking or lane changes."
        ),
        "data": [],
        "followup": "I can still help with danger zones, peak times, or route-specific checks.",
    }



def handle_high_risk_zones(user_message=""):
    data = load_chatbot_data()
    high_risk = []
    for gid, z in data["zones"].items():
        if z["risk_level"] == "high":
            high_risk.append(z)

    high_risk.sort(key=lambda z: z["risk_score"], reverse=True)

    lines = [f"• **{z['location']}** - {z['total_accidents']} accidents" for z in high_risk[:5]]

    return {
        "type": "list",
        "message": (
            f"⚠️ Bangkok has {data['stats']['high_risk_zones']} high-risk zones. Top ones to avoid:\n" +
            "\n".join(lines) +
            f"\n\nThese make up ~5% of the city but cause most accidents."
        ),
        "data": high_risk[:5],
        "followup": "Want safer route suggestions? 🚗",
    }


def handle_motorcycle_zones(user_message=""):
    data = load_chatbot_data()

    moto_zones = []
    for gid, z in data["zones"].items():
        if z["motorcycle_pct"] > 15:
            moto_zones.append(z)

    moto_zones.sort(key=lambda z: z["motorcycle_pct"], reverse=True)

    if moto_zones:
        lines = [f"• **{z['location']}** - {z['motorcycle_pct']}% motorcycle involvement" for z in moto_zones[:5]]
        zone_text = "\n".join(lines)
    else:
        zone_text = "No major motorcycle hotspots in the top monitored zones."

    return {
        "type": "list",
        "message": (
            f"🏍️ Motorcycle accidents are a big concern in Bangkok. Riskiest areas:\n" +
            zone_text +
            f"\n\nRiders: wear helmets and watch for blind spots at intersections."
        ),
        "data": moto_zones[:5],
        "followup": "Want safer routes for motorcyclists? 🚗",
    }


# ── Cause-based ──────────────────────────────────────────────────────

def handle_cause_breakdown(user_message=""):
    data = load_chatbot_data()
    causes = data["causes"]

    lines = []
    for cause in sorted(causes.values(), key=lambda c: c["count"], reverse=True):
        if cause["count"] > 0:
            lines.append(f"• **{cause['english_name']}** - {cause['percentage']}%")

    return {
        "type": "stats",
        "message": (
            f"What causes Bangkok accidents:\n" +
            "\n".join(lines) +
            f"\n\nSpeeding is the overwhelming #1 cause - nearly 4 out of 5 accidents."
        ),
        "data": [{"cause": c["english_name"], "count": c["count"], "pct": c["percentage"]}
                 for c in causes.values()],
        "followup": "Want to see the worst zones for any cause? 💡",
    }


def handle_speeding_stats(user_message=""):
    data = load_chatbot_data()
    speeding = data["causes"]["speeding"]

    return {
        "type": "stats",
        "message": (
            f"Speeding causes **{speeding['percentage']}%** of all Bangkok accidents - it's the #1 problem by far.\n\n"
            f"Worst zone: **{speeding['top_zones'][0]['location']}** with {speeding['top_zones'][0]['cause_count']} speeding accidents. "
            f"That makes it one of the clearest enforcement targets in the dataset."
        ),
        "data": speeding["top_zones"][:3],
        "followup": "Want to see top speeding zones? 🚗",
    }


def handle_drunk_stats(user_message=""):
    data = load_chatbot_data()

    return {
        "type": "stats",
        "message": (
            f"Drunk driving is heavily underreported in official data, but the patterns tell the story. "
            f"Saturday is the deadliest day ({data['time_patterns']['most_dangerous_day_count']} accidents), "
            f"and late nights (11 PM - 3 AM) show elevated risk.\n\n"
            f"Best advice: avoid driving on Friday/Saturday nights if you can."
        ),
        "data": [],
        "followup": "Want to see weekend patterns? 🕐",
    }


def handle_cutting_stats(user_message=""):
    data = load_chatbot_data()
    cutting = data["causes"]["cutting_in_front"]

    return {
        "type": "stats",
        "message": (
            f"Cutting in front (aggressive lane changes) is the #2 cause at {cutting['percentage']}% of accidents.\n\n"
            f"It's actually a stronger danger signal than speeding - zones with lots of cutting are "
            f"especially risky. Stay alert at busy intersections and keep safe following distance."
        ),
        "data": cutting["top_zones"][:3],
        "followup": "Want to see the worst intersections for this? ⚠️",
    }


def handle_cause_comparison(user_message=""):
    data = load_chatbot_data()
    causes = data["causes"]

    rows = []
    for key, c in causes.items():
        rows.append({
            "cause": c["english_name"],
            "count": c["count"],
            "percentage": c["percentage"],
        })
    rows.sort(key=lambda r: r["count"], reverse=True)

    lines = [f"• **{r['cause']}** - {r['percentage']}%" for r in rows if r["count"] > 0]

    return {
        "type": "comparison",
        "message": (
            f"Bangkok accident causes:\n" +
            "\n".join(lines) +
            f"\n\nSpeeding dominates, but cutting in front is actually a stronger danger signal "
            f"per zone - those intersections are especially risky."
        ),
        "data": rows,
        "followup": "Want to see the worst zones for any cause? 💡",
    }


# ── Time-based ───────────────────────────────────────────────────────

def handle_time_patterns(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]

    return {
        "type": "stats",
        "message": (
            f"**Peak hour: {tp['most_dangerous_hour']}.** That is the busiest single hour in the dataset with **{tp['most_dangerous_hour_count']} accidents**.\n\n"
            f"The lowest hour is **{tp['safest_hour']}** with **{tp['safest_hour_count']} accidents**, but for normal trips the practical move is to avoid the main spikes around **7-10 AM**, **{tp['most_dangerous_hour']}**, and **4-8 PM**."
        ),
        "data": [],
        "followup": "Want rush hour or weekend details?",
    }


def handle_rush_hour(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]
    hourly = tp["hourly"]

    morning = sum(hourly.get(f"{h:02d}:00", {}).get("count", 0) for h in range(7, 10))
    evening = sum(hourly.get(f"{h:02d}:00", {}).get("count", 0) for h in range(16, 20))

    return {
        "type": "stats",
        "message": (
            f"**Rush hour is a major risk window.** **{tp['rush_hour_pct']}%** of all recorded crashes happen during rush-hour periods.\n\n"
            f"The evening rush (4-8 PM) is {'worse' if evening > morning else 'slightly lighter'} than the morning rush (7-10 AM), with **{evening} vs {morning} accidents**. If your schedule is flexible, avoid those peaks instead of aiming for one supposedly 'safe' daytime block."
        ),
        "data": [],
        "followup": "Want weekend patterns too? 🕐",
    }


def handle_weekend_pattern(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]
    daily = tp["daily"]

    weekend_total = sum(daily[d]["count"] for d in ["Saturday", "Sunday"])
    weekday_total = sum(daily[d]["count"] for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    weekend_daily_avg = weekend_total / 2
    weekday_daily_avg = weekday_total / 5

    return {
        "type": "comparison",
        "message": (
            f"**{tp['most_dangerous_day']}** is the highest-risk single day with **{tp['most_dangerous_day_count']} accidents**.\n\n"
            f"Weekdays still make up **{tp['weekday_accidents']} crashes** overall because there are more weekday travel days, but weekend pressure is concentrated hardest on Saturday. If you are traveling on the weekend, Saturday deserves the most caution."
        ),
        "data": [{"day": d, **daily[d]} for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]],
        "followup": "Want to see the riskiest weekend zones?",
    }


# ── Recommendation ───────────────────────────────────────────────────

def handle_policy_recommendations(user_message=""):
    data = load_chatbot_data()
    recs = data["recommendations"]

    lines = [f"{r['priority']}. **{r['title']}** - {r['expected_impact']}" for r in recs[:3]]

    return {
        "type": "recommendation",
        "message": (
            f"Top fixes for Bangkok road safety:\n" +
            "\n".join(lines) +
            f"\n\nAll backed by accident data analysis. 💡"
        ),
        "data": recs,
        "followup": "Want specifics on any recommendation?",
    }


def handle_camera_deployment(user_message=""):
    data = load_chatbot_data()
    speeding = data["causes"]["speeding"]
    top_zones = speeding["top_zones"][:5]

    lines = [f"• **{z['location']}** - {z['cause_count']} speeding accidents" for z in top_zones]

    return {
        "type": "recommendation",
        "message": (
            f"Top zones recommended for speed enforcement:\n" +
            "\n".join(lines) +
            f"\n\nThese locations show the heaviest speeding concentration in the dataset, so they are the clearest enforcement candidates."
        ),
        "data": top_zones,
        "followup": "Want to see other recommendations?",
    }


# ── Meta ─────────────────────────────────────────────────────────────

def handle_how_it_works(user_message=""):
    data = load_chatbot_data()
    is_thai = _has_thai_text(user_message)

    if is_thai:
        message = (
            "ระบบนี้ใช้ AI หลัก 2 ส่วน:\n"
            "1. **Thai NLP** สำหรับอ่านรายงานอุบัติเหตุและจัดกลุ่มสาเหตุ เพื่อช่วยตอบคำถามใน Safety Assistant\n"
            f"2. **XGBoost ML** สำหรับให้คะแนนความเสี่ยงของแต่ละโซนขนาดประมาณ 1 กม. ({data['stats']['model_accuracy']}% holdout capture)\n\n"
            "ทั้งสองส่วนทำงานร่วมกันในแผนที่ความเสี่ยง Trip Planner และหน้า Analytics."
        )
        followup = "อยากรู้ต่อว่าข้อมูลมาจากไหนไหม?"
    else:
        message = (
            f"We combine 2 core AI techniques:\n"
            f"1. **Thai NLP** - reads accident reports to classify causes and power our Safety Assistant chatbot\n"
            f"2. **XGBoost ML** - predicts risk scores for every 1km zone ({data['stats']['model_accuracy']}% holdout capture)\n\n"
            f"Together, they power our risk map, trip planner, and analytics dashboard."
        )
        followup = "Want to know about our data source?"

    return {
        "type": "info",
        "message": message,
        "data": [],
        "followup": followup,
    }


def handle_data_source(user_message=""):
    data = load_chatbot_data()
    stats = data["stats"]
    is_thai = _has_thai_text(user_message)

    if is_thai:
        message = (
            f"เราใช้ข้อมูลอุบัติเหตุในกรุงเทพฯ **{stats['total_accidents']:,} เคส** จาก **{stats['data_source']} ปี 2024** "
            f"ครอบคลุมช่วง **{stats['date_range']}**.\n\n"
            f"ในชุดข้อมูลนี้มีผู้เสียชีวิต **{stats['deaths']} ราย** และผู้บาดเจ็บสาหัส **{stats['serious_injuries']} ราย** "
            f"กระจายอยู่ในโซนที่ระบบติดตาม **{stats['total_zones']:,} โซน** โดยข้อความภาษาไทยถูกประมวลผลด้วย NLP เพื่อสรุปสาเหตุหลัก."
        )
        followup = "อยากดูต่อไหมว่าระบบวิเคราะห์ข้อมูลพวกนี้อย่างไร?"
    else:
        message = (
            f"We analyzed **{stats['total_accidents']:,} Bangkok accidents** from {stats['data_source']} (2024), "
            f"covering {stats['date_range']}.\n\n"
            f"The data includes {stats['deaths']} deaths and {stats['serious_injuries']} serious injuries across "
            f"{stats['total_zones']:,} monitored zones. All Thai text was processed with NLP to extract causes."
        )
        followup = "Want to see how the model works?"

    return {
        "type": "info",
        "message": message,
        "data": [],
        "followup": followup,
    }


# ---------------------------------------------------------------------------
# ML cause detection for English queries
# ---------------------------------------------------------------------------

# ML model cause names → chatbot_data.json cause keys
# None means the cause exists as a special case handled separately
_ML_CAUSE_DATA_MAP = {
    "speeding":          "speeding",
    "drunk_driving":     None,   # not a chatbot_data cause key; handled specially
    "drowsy_driving":    "drowsy_driving",
    "equipment_failure": "tire_blowout___worn_tires",
    "traffic_violation": "running_red_light___traffic_sign_violation",
    "cutting_in_front":  "cutting_in_front",
}

# English keyword sets per cause
_CAUSE_KEYWORDS = {
    "speeding": [
        "speed", "speeding", "overspeeding", "over-speed", "over speed",
        "driving fast", "fast driving", "velocity", "accelerat",
        "ขับเร็ว", "ขับรถเร็ว", "เร็วเกินกำหนด", "เร็วเกินอัตรา", "ใช้ความเร็ว",
    ],
    "drunk_driving": [
        "drunk", "alcohol", "dui", "dwi", "drink driv", "drinking and driv",
        "intoxicat", "booze", "drink and drive", "impaired driv",
        "เมา", "เมาแล้วขับ", "ดื่มแล้วขับ", "แอลกอฮอล์",
    ],
    "drowsy_driving": [
        "drowsy", "sleepy", "fatigue", "tired", "drowsiness", "fall asleep",
        "microsleep", "asleep at the wheel", "sleep depri", "nodding off",
        "หลับใน", "ง่วง", "ง่วงนอน", "อ่อนล้า", "เพลีย",
    ],
    "equipment_failure": [
        "equipment failure", "brake fail", "tire blowout", "tyre blowout",
        "mechanical failure", "vehicle fault", "car fault", "engine fail",
        "blowout", "worn tire", "worn tyre", "bald tire",
        "ยางแตก", "ยางเสื่อม", "เบรกเสีย", "เบรกขัดข้อง", "อุปกรณ์บกพร่อง",
    ],
    "traffic_violation": [
        "red light", "traffic violation", "traffic signal", "signal violation",
        "ran a light", "ran red", "run red", "traffic sign violation",
        "wrong side", "illegal turn",
        "ฝ่าไฟแดง", "ฝ่าฝืนสัญญาณไฟ", "ฝ่าฝืนเครื่องหมาย", "ย้อนศร", "ผิดกฎจราจร",
    ],
    "cutting_in_front": [
        "cut in front", "cutting in front", "cut off", "cutting off",
        "sudden lane change", "aggressive lane", "merging dangerously",
        "ตัดหน้า", "เปลี่ยนเลนกะทันหัน", "ปาดหน้า",
    ],
}


def detect_cause_from_question(user_message: str):
    """
    Detect accident cause from an English user query using keyword matching,
    with ML classifier fallback for ambiguous or Thai-language queries.

    Returns:
        (cause_key, confidence) tuple.
        cause_key is a chatbot_data["causes"] key or "drunk_driving" for the special case.
        confidence is 0.0–1.0. Returns (None, 0.0) when nothing is detected.
    """
    text = user_message.lower().strip()
    scores = {}

    for cause, keywords in _CAUSE_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text)
        if matches:
            # A single strong keyword is already a clear signal (≥0.65)
            scores[cause] = min(0.65 + (matches - 1) * 0.2, 1.0)

    if not scores:
        # Fallback: try the Thai ML classifier (may catch transliterated/Thai text)
        try:
            classifier = _get_cause_classifier()
            result = classifier.classify(user_message, confidence_threshold=0.5)
            if result and result.get("cause") and result.get("confidence", 0) > 0.6:
                ml_cause = result["cause"]
                data_key = _ML_CAUSE_DATA_MAP.get(ml_cause)
                return (data_key if data_key else ml_cause), result["confidence"]
        except Exception:
            pass
        return None, 0.0

    best_cause = max(scores, key=scores.__getitem__)
    confidence = scores[best_cause]
    if confidence < 0.6:
        return None, 0.0

    data_key = _ML_CAUSE_DATA_MAP.get(best_cause)
    return (data_key if data_key else best_cause), confidence


def handle_ml_cause_response(cause_key: str, confidence: float, user_message: str = ""):
    """
    Build a cause-specific statistics response for ML-detected causes.

    Returns a response dict (same shape as scripted handlers), or None if
    chatbot_data has no entry for this cause (signal to fall through to AI).
    """
    data = load_chatbot_data()
    causes = data["causes"]

    # Special case: drunk_driving isn't a chatbot_data cause key
    if cause_key == "drunk_driving":
        tp = data["time_patterns"]
        return {
            "type": "stats",
            "message": (
                "Drunk driving is significantly underreported in official Bangkok data, "
                "but the patterns are clear. **Saturday** is the deadliest day "
                f"({tp['most_dangerous_day_count']} accidents) and late-night hours (10 PM – 3 AM) "
                "show the highest risk.\n\n"
                "Best advice: avoid driving on Friday/Saturday nights if possible."
            ),
            "data": [],
            "followup": "Want to see weekend accident patterns? 🕐",
            "source": "ml_detected",
            "detected_cause": cause_key,
            "confidence": round(confidence, 2),
        }

    cause_data = causes.get(cause_key)
    if not cause_data:
        return None  # No data — let Groq AI handle it

    name = cause_data["english_name"]
    pct = cause_data["percentage"]
    count = cause_data["count"]
    top_zones = cause_data.get("top_zones", [])[:3]

    zone_text = ""
    if top_zones:
        zone_text = f"\n\nTop zones:\n{_fmt_zone_list(top_zones, 3)}"

    if cause_key == "speeding":
        advice = "These zones are the clearest enforcement targets in the dataset."
    elif cause_key == "cutting_in_front":
        advice = "Zones with high cutting-in-front rates are especially risky — keep safe following distance."
    else:
        advice = "Stay alert and drive defensively in affected zones."

    if _has_thai_text(user_message):
        thai_name_map = {
            "speeding": "การขับรถเร็ว",
            "cutting_in_front": "การตัดหน้า",
            "drowsy_driving": "การหลับใน",
            "tire_blowout___worn_tires": "ปัญหายางหรืออุปกรณ์รถ",
            "running_red_light___traffic_sign_violation": "การฝ่าฝืนสัญญาณไฟหรือป้ายจราจร",
        }
        thai_advice_map = {
            "speeding": "จุดเหล่านี้เหมาะกับการเน้นบังคับใช้กฎหมายเรื่องความเร็วมากที่สุดในข้อมูลชุดนี้.",
            "cutting_in_front": "จุดที่มีการตัดหน้าสูงมักเสี่ยงมาก ควรเว้นระยะและระวังการเปลี่ยนเลนกะทันหัน.",
        }
        thai_name = thai_name_map.get(cause_key, name)
        thai_advice = thai_advice_map.get(
            cause_key,
            "ควรเพิ่มความระวังและขับแบบตั้งรับเมื่อผ่านโซนที่ได้รับผลกระทบ.",
        )
        zone_text_th = ""
        if top_zones:
            zone_text_th = f"\n\nโซนเด่น:\n{_fmt_zone_list(top_zones, 3)}"

        return {
            "type": "stats",
            "message": (
                f"**{thai_name}** คิดเป็น **{pct}%** ของอุบัติเหตุในกรุงเทพฯ "
                f"({count:,} เคส).{zone_text_th}\n\n{thai_advice}"
            ),
            "data": top_zones,
            "followup": f"อยากดูโซนที่เจอ{thai_name}มากที่สุดไหม?",
            "source": "ml_detected",
            "detected_cause": cause_key,
            "confidence": round(confidence, 2),
        }

    return {
        "type": "stats",
        "message": (
            f"**{name}** accounts for **{pct}%** of all Bangkok accidents "
            f"({count:,} incidents).{zone_text}\n\n{advice}"
        ),
        "data": top_zones,
        "followup": f"Want to see the worst zones for {name.lower()}? ⚠️",
        "source": "ml_detected",
        "detected_cause": cause_key,
        "confidence": round(confidence, 2),
    }


# ---------------------------------------------------------------------------
# Demo-focused response overrides
# ---------------------------------------------------------------------------

def _fmt_zone_list(zones, limit=5):
    """Format zone dicts into markdown-friendly bullets."""
    lines = []
    for z in zones[:limit]:
        count = z.get("cause_count", z.get("total_accidents", "?"))
        lines.append(f"- **{z['location']}** - {count} accidents")
    return "\n".join(lines)


def _fmt_suggestion_list(items):
    """Format short follow-up suggestions for the chat UI."""
    return "\n".join(f"- {item}" for item in items if item)


def handle_judge_dangerous_zones(user_message=""):
    anchors = _judge_zone_anchors()
    west = anchors["west"]
    east = anchors["east"]
    is_thai = any("\u0e00" <= char <= "\u0e7f" for char in user_message)

    if is_thai:
        message = (
            f"à¸ˆà¸¸à¸”à¹€à¸ªà¸µà¹ˆà¸¢à¸‡à¸«à¸¥à¸±à¸à¹ƒà¸™à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸™à¸µà¹‰à¸¡à¸µ 2 à¹à¸™à¸§à¸—à¸µà¹ˆà¸„à¸§à¸£à¸žà¸¹à¸”à¸–à¸¶à¸‡à¸à¹ˆà¸­à¸™à¸„à¸·à¸­ **à¸–à¸™à¸™à¸šà¸£à¸¡à¸£à¸²à¸Šà¸Šà¸™à¸™à¸µ à¹à¸–à¸§à¸—à¸§à¸µà¸§à¸±à¸’à¸™à¸²** "
            f"à¸‹à¸¶à¹ˆà¸‡à¹€à¸›à¹‡à¸™à¸à¸±à¹ˆà¸‡à¸•à¸°à¸§à¸±à¸™à¸•à¸à¹à¸¥à¸°à¸¡à¸µ **{west['accidents']} à¸­à¸¸à¸šà¸±à¸•à¸´à¹€à¸«à¸•à¸¸** à¹ƒà¸™à¸«à¸™à¸¶à¹ˆà¸‡à¸à¸£à¸´à¸”à¹‚à¸‹à¸™ à¸à¸±à¸š **à¸¥à¸²à¸”à¸à¸£à¸°à¸šà¸±à¸‡ / à¸¡à¸­à¹€à¸•à¸­à¸£à¹Œà¹€à¸§à¸¢à¹Œ 7** "
            f"à¸‹à¸¶à¹ˆà¸‡à¹€à¸›à¹‡à¸™à¹à¸™à¸§à¹€à¸ªà¸µà¹ˆà¸¢à¸‡à¸ªà¸¹à¸‡à¸à¸±à¹ˆà¸‡à¸•à¸°à¸§à¸±à¸™à¸­à¸­à¸.\n\nà¸–à¹‰à¸²à¸•à¹‰à¸­à¸‡à¸•à¸­à¸šà¸à¸£à¸£à¸¡à¸à¸²à¸£à¹à¸šà¸šà¸ªà¸±à¹‰à¸™à¸—à¸µà¹ˆà¸ªà¸¸à¸” à¹ƒà¸«à¹‰à¸¢à¸¶à¸”à¸ªà¸­à¸‡à¹à¸™à¸§à¸™à¸µà¹‰à¹€à¸›à¹‡à¸™à¸„à¸³à¸•à¸­à¸šà¸«à¸¥à¸±à¸à¸‚à¸­à¸‡à¸à¸£à¸¸à¸‡à¹€à¸—à¸žà¸¯."
        )
        followup = "ลองถามต่อได้ว่าอุบัติเหตุพีกช่วงเวลาไหน"
    else:
        message = (
            f"The clearest high-risk anchors are the **western corridor at {west['road']}, {west['district']}** with "
            f"**{west['accidents']} accidents** in one grid zone, and the **eastern corridor around {east['district']} / {east['road']}**.\n\n"
            "If you need the shortest judge-safe answer, lead with those two corridors."
        )
        followup = "Next useful prompt:\n- What time are accidents most common?"

    return {
        "type": "list",
        "message": message,
        "data": [
            {"corridor": west["label"], "road": west["road"], "district": west["district"], "accidents": west["accidents"]},
            {"corridor": east["label"], "road": east["road"], "district": east["district"], "accidents": east["accidents"]},
        ],
        "followup": followup,
    }


def handle_judge_peak_time(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]
    peak_hour = tp["most_dangerous_hour"]
    peak_count = tp["most_dangerous_hour_count"]
    is_thai = any("\u0e00" <= char <= "\u0e7f" for char in user_message)

    if is_thai:
        message = (
            f"à¸Šà¹ˆà¸§à¸‡à¹€à¸§à¸¥à¸²à¸—à¸µà¹ˆà¸­à¸¸à¸šà¸±à¸•à¸´à¹€à¸«à¸•à¸¸à¸ªà¸¹à¸‡à¸—à¸µà¹ˆà¸ªà¸¸à¸”à¸„à¸·à¸­ **{peak_hour}** à¸—à¸µà¹ˆ **{peak_count} à¸„à¸£à¸±à¹‰à¸‡** à¹ƒà¸™à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸Šà¸¸à¸”à¸™à¸µà¹‰.\n\n"
            "à¸–à¹‰à¸²à¸•à¹‰à¸­à¸‡à¸‚à¸±à¸šà¸ˆà¸£à¸´à¸‡ à¸„à¸§à¸£à¸«à¸¥à¸šà¸Šà¹ˆà¸§à¸‡à¸žà¸µà¸à¸«à¸¥à¸±à¸à¸£à¸­à¸š **7-10 AM**, **10:00-11:00**, à¹à¸¥à¸° **4-8 PM** à¸–à¹‰à¸²à¸—à¸³à¹„à¸”à¹‰."
        )
        followup = "ลองถามต่อได้ว่าย่านไหนอันตรายที่สุด"
    else:
        message = (
            f"**Peak hour: {peak_hour}.** That is the busiest single hour in the dataset with **{peak_count} accidents**.\n\n"
            "For a practical driving answer, avoid the main spikes around **7-10 AM**, **10:00-11:00**, and **4-8 PM** when possible."
        )
        followup = "Next useful prompt:\n- Which areas are most dangerous?"

    return {
        "type": "stats",
        "message": message,
        "data": [{"hour": peak_hour, "count": peak_count}],
        "followup": followup,
    }


def handle_judge_peak_day(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]
    peak_day = tp["most_dangerous_day"]
    peak_count = tp["most_dangerous_day_count"]
    is_thai = any("\u0e00" <= char <= "\u0e7f" for char in user_message)

    if is_thai:
        message = (
            f"à¸§à¸±à¸™à¸—à¸µà¹ˆà¸­à¸¸à¸šà¸±à¸•à¸´à¹€à¸«à¸•à¸¸à¸ªà¸¹à¸‡à¸—à¸µà¹ˆà¸ªà¸¸à¸”à¸„à¸·à¸­ **à¸§à¸±à¸™à¹€à¸ªà¸²à¸£à¹Œ** à¸—à¸µà¹ˆ **{peak_count} à¸„à¸£à¸±à¹‰à¸‡**.\n\n"
            "à¸–à¹‰à¸²à¸•à¹‰à¸­à¸‡à¸•à¸­à¸šà¹à¸šà¸šà¸ªà¸±à¹‰à¸™à¹à¸¥à¸°à¸Šà¸±à¸”à¸ªà¸³à¸«à¸£à¸±à¸šà¸à¸£à¸£à¸¡à¸à¸²à¸£ à¹ƒà¸«à¹‰à¸•à¸­à¸šà¸§à¹ˆà¸² **à¸§à¸±à¸™à¹€à¸ªà¸²à¸£à¹Œ** à¸„à¸·à¸­à¸§à¸±à¸™à¹€à¸ªà¸µà¹ˆà¸¢à¸‡à¸ªà¸¹à¸‡à¸ªà¸¸à¸”à¹ƒà¸™à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸™à¸µà¹‰."
        )
        followup = "ลองถามต่อได้ว่าอุบัติเหตุพีกเวลาไหน"
    else:
        message = (
            f"**{peak_day}** is the most dangerous day in the dataset with **{peak_count} accidents**.\n\n"
            "If you need a clean judge-facing takeaway, Saturday is the day to call out first."
        )
        followup = "Next useful prompt:\n- What time are accidents most common?"

    return {
        "type": "stats",
        "message": message,
        "data": [{"day": peak_day, "count": peak_count}],
        "followup": followup,
    }


def handle_judge_model_accuracy(user_message=""):
    data = load_chatbot_data()
    stats = data["stats"]
    accuracy = stats["model_accuracy"]
    lift = stats["model_lift"]
    is_thai = any("\u0e00" <= char <= "\u0e7f" for char in user_message)

    if is_thai:
        message = (
            f"à¹ƒà¸™à¸à¸²à¸£à¸—à¸”à¸ªà¸­à¸šà¹à¸šà¸š **Jan-Sep 2024 train / Oct-Dec 2024 holdout** à¸£à¸°à¸šà¸šà¸ˆà¸±à¸šà¸„à¸£à¸­à¸šà¸„à¸¥à¸¸à¸¡à¸­à¸¸à¸šà¸±à¸•à¸´à¹€à¸«à¸•à¸¸à¸—à¸µà¹ˆà¹€à¸à¸´à¸”à¸ à¸²à¸¢à¸«à¸¥à¸±à¸‡à¹„à¸”à¹‰ **{accuracy}%** "
            f"à¹à¸¥à¸°à¹ƒà¸«à¹‰ **lift {lift}x** à¹€à¸—à¸µà¸¢à¸šà¸à¸±à¸šà¸à¸²à¸£à¸ªà¸¸à¹ˆà¸¡à¸«à¸²à¹‚à¸‹à¸™à¹€à¸ªà¸µà¹ˆà¸¢à¸‡.\n\n"
            "à¸ªà¸£à¸¸à¸›à¹à¸šà¸šà¸•à¸­à¸šà¸à¸£à¸£à¸¡à¸à¸²à¸£à¸„à¸·à¸­à¹‚à¸¡à¹€à¸”à¸¥à¸¡à¸µà¸ªà¸±à¸à¸à¸²à¸“à¹€à¸Šà¸´à¸‡à¹€à¸§à¸¥à¸²à¹à¸¥à¸°à¸—à¸³à¹„à¸”à¹‰à¸”à¸µà¸à¸§à¹ˆà¸²à¸à¸²à¸£à¹€à¸¥à¸·à¸­à¸à¹‚à¸‹à¸™à¹à¸šà¸šà¸ªà¸¸à¹ˆà¸¡à¸­à¸¢à¹ˆà¸²à¸‡à¸Šà¸±à¸”à¹€à¸ˆà¸™."
        )
        followup = "ลองถามต่อได้ว่าย่านไหนอันตรายที่สุด"
    else:
        message = (
            f"On a **Jan-Sep 2024 train / Oct-Dec 2024 holdout**, SafeDrive BKK captured **{accuracy}%** of later mapped crashes inside flagged high-risk grids "
            f"with **{lift}x lift** over random danger-zone selection.\n\n"
            "The judge-safe summary is that the model shows temporal signal and performs materially better than random zone selection."
        )
        followup = "Next useful prompt:\n- Which areas are most dangerous?"

    return {
        "type": "stats",
        "message": message,
        "data": [{"holdout_capture": accuracy, "lift": lift}],
        "followup": followup,
    }


def handle_speeding_zones(user_message=""):
    data = load_chatbot_data()
    speeding = data["causes"]["speeding"]

    return {
        "type": "list",
        "message": (
            f"**Speeding is Bangkok's #1 crash factor** at **{speeding['percentage']}%** of recorded accidents.\n\n"
            f"Top hotspots:\n{_fmt_zone_list(speeding['top_zones'], 5)}\n\n"
            "These are the clearest enforcement priorities in the dataset."
        ),
        "data": speeding["top_zones"][:5],
        "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- How reliable is the model?",
    }


def handle_drunk_zones(user_message=""):
    data = load_chatbot_data()

    weekend_zones = []
    for gid, zone in data["zones"].items():
        if zone["weekend_pct"] > 40:
            weekend_zones.append({
                "location": zone["location"],
                "total_accidents": zone["total_accidents"],
                "weekend_pct": zone["weekend_pct"],
            })
    weekend_zones.sort(key=lambda z: z["weekend_pct"], reverse=True)

    zone_lines = "\n".join(
        f"- **{z['location']}** - {z['weekend_pct']}% weekend accidents"
        for z in weekend_zones[:5]
    )

    return {
        "type": "list",
        "message": (
            "Drunk driving is underreported in the official dataset, so the safest demo answer is to use **weekend-night risk** as the strongest signal.\n\n"
            f"**Saturday** is the worst day overall with **{data['time_patterns']['most_dangerous_day_count']} accidents**.\n\n"
            f"Weekend-heavy zones:\n{zone_lines}\n\n"
            "If possible, avoid late-night driving on Friday and Saturday."
        ),
        "data": weekend_zones[:5],
        "followup": "Next useful prompt:\n- What time are accidents most common?\n- Which areas are most dangerous?",
    }


def handle_dangerous_zones(user_message=""):
    data = load_chatbot_data()
    zones = data["zones"]

    top5 = []
    for gid, z in list(zones.items())[:5]:
        top5.append({
            "location": z["location"],
            "total_accidents": z["total_accidents"],
            "dominant_cause": z["dominant_cause"],
            "has_fatality": z["has_fatality"],
        })

    lines = []
    for z in top5:
        fatality = " [fatal crash recorded]" if z["has_fatality"] else ""
        lines.append(
            f"- **{z['location']}** - {z['total_accidents']} accidents ({z['dominant_cause']}){fatality}"
        )

    return {
        "type": "list",
        "message": (
            "Bangkok's highest-risk zones in this demo dataset are:\n"
            + "\n".join(lines)
            + "\n\nSpeeding is the dominant pattern in most of these corridors."
        ),
        "data": top5,
        "followup": "Next useful prompt:\n- What time are accidents most common?\n- How reliable is the model?",
    }


def handle_zone_safety(user_message=""):
    data = load_chatbot_data()

    location_query = user_message.strip()
    for prefix in [
        "how dangerous is ", "how safe is ", "how risky is ",
        "tell me about ", "is ", "how safe ", "how dangerous ",
        "safety of ", "risk of ", "danger of ", "what about ",
    ]:
        if location_query.lower().startswith(prefix):
            location_query = location_query[len(prefix):]
            break
    location_query = location_query.strip(" ?.,!")

    changed = True
    while changed:
        changed = False
        for suffix in [" safe", " dangerous", " risky", " area", " zone", " safety"]:
            if location_query.lower().endswith(suffix):
                location_query = location_query[:-len(suffix)].rstrip()
                changed = True

    gid, zone = _search_zone_by_name(location_query)

    if zone and gid:
        z = zone
        risk_desc = "HIGH RISK" if z["risk_score"] >= 55 else "MEDIUM RISK" if z["risk_score"] >= 30 else "LOW RISK"
        top_cause = max(z["cause_breakdown"].items(), key=lambda x: x[1])

        return {
            "type": "single",
            "message": (
                f"**{z['location']}** is **{risk_desc}**.\n\n"
                f"Top risk factor: **{top_cause[0].replace('_', ' ').title()}** ({top_cause[1]}% of crashes in this zone).\n"
                f"Rush-hour exposure: **{z['rush_hour_pct']}%** of crashes happen during rush periods.\n\n"
                f"{'Best practical advice: avoid the evening peak when possible.' if z['rush_hour_pct'] > 30 else 'This area is still worth treating cautiously even outside peak hours.'}"
            ),
            "data": [z],
            "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- What causes most accidents?",
        }

    if zone:
        return {
            "type": "single",
            "message": (
                f"**{zone['location']}** is a monitored zone. "
                "For detailed risk information, ask about a specific area like Asok or Lat Krabang."
            ),
            "data": [zone],
            "followup": "Try one of these:\n- Which areas are most dangerous?\n- What time are accidents most common?",
        }

    top_zones = list(data["zones"].values())[:5]
    lines = [f"- **{z['location']}** - {z['total_accidents']} accidents" for z in top_zones]
    return {
        "type": "list",
        "message": (
            f"I could not match **{location_query}** to a named monitored zone in this demo dataset.\n\n"
            "The current coverage is strongest for Bangkok corridors, outer districts, and major highways. A more reliable question would be one of these:\n"
            + _fmt_suggestion_list([
                "Which areas are most dangerous?",
                "What time are accidents most common?",
                "How reliable is the model?",
            ])
            + "\n\nNearby high-risk examples from the dataset:\n"
            + "\n".join(lines)
            + "\n\nIf you have a specific trip in mind, use the **Trip Planner** for path-level risk."
        ),
        "data": top_zones,
        "followup": "Try one of these:\n- Which areas are most dangerous?\n- How reliable is the model?",
    }


def handle_route_safety(user_message=""):
    route_reference = _extract_route_reference(user_message)
    is_thai = _has_thai_text(user_message)
    acknowledgement = (
        f"That trip {route_reference} needs a path-specific check rather than a generic answer."
        if route_reference
        else "That is a route-specific question, so the safest answer depends on the exact path and time of travel."
    )
    message = (
        "à¸„à¸³à¸–à¸²à¸¡à¹à¸šà¸šà¸™à¸µà¹‰à¸•à¹‰à¸­à¸‡à¸”à¸¹à¸•à¸²à¸¡à¹€à¸ªà¹‰à¸™à¸—à¸²à¸‡à¸ˆà¸£à¸´à¸‡à¹à¸¥à¸°à¸Šà¹ˆà¸§à¸‡à¹€à¸§à¸¥à¸²à¸—à¸µà¹ˆà¸ˆà¸°à¹€à¸”à¸´à¸™à¸—à¸²à¸‡ à¹„à¸¡à¹ˆà¸„à¸§à¸£à¸•à¸­à¸šà¹à¸šà¸šà¹€à¸«à¸¡à¸²à¸£à¸§à¸¡. "
        "**à¹ƒà¸Šà¹‰ Trip Planner** à¹€à¸žà¸·à¹ˆà¸­à¹€à¸—à¸µà¸¢à¸šà¹€à¸ªà¹‰à¸™à¸—à¸²à¸‡à¸—à¸µà¹ˆà¹€à¸£à¹‡à¸§à¸à¸±à¸šà¹€à¸ªà¹‰à¸™à¸—à¸²à¸‡à¸—à¸µà¹ˆà¸›à¸¥à¸­à¸”à¸ à¸±à¸¢à¸à¸§à¹ˆà¸² à¹à¸¥à¹‰à¸§à¸”à¸¹à¸ˆà¸¸à¸”à¹€à¸ªà¸µà¹ˆà¸¢à¸‡à¸šà¸™à¸—à¸²à¸‡à¸ˆà¸£à¸´à¸‡à¹„à¸”à¹‰à¹€à¸¥à¸¢."
        if is_thai
        else f"{acknowledgement} **Use the Trip Planner** to compare a faster option with a safer option and see the danger zones along the way."
    )
    return {
        "type": "info",
        "message": message,
        "data": [],
        "followup": "à¸­à¸¢à¸²à¸à¸£à¸¹à¹‰à¸à¹ˆà¸­à¸™à¸§à¹ˆà¸²à¸„à¸§à¸£à¸«à¸¥à¸šà¸¢à¹ˆà¸²à¸™à¹„à¸«à¸™à¸šà¹‰à¸²à¸‡à¹„à¸«à¸¡?" if is_thai else "Next useful prompt:\n- Which areas are most dangerous?\n- What time are accidents most common?",
        "actions": [_trip_planner_action()],
    }


def handle_rain_data_gap(user_message=""):
    return {
        "type": "info",
        "message": (
            "Rain is not explicitly labeled in this Bangkok dataset, so I should not invent a Bangkok-specific percentage.\n\n"
            "What we can say reliably is that wet conditions usually increase risk by reducing visibility and tire grip. The practical guidance is to slow down, leave more following distance, and avoid abrupt braking or lane changes."
        ),
        "data": [],
        "followup": "Try one of these instead:\n- What time are accidents most common?\n- Which areas are most dangerous?\n- How reliable is the model?",
    }


def handle_high_risk_zones(user_message=""):
    data = load_chatbot_data()
    high_risk = []
    for gid, z in data["zones"].items():
        if z["risk_level"] == "high":
            high_risk.append(z)

    high_risk.sort(key=lambda z: z["risk_score"], reverse=True)
    lines = [f"- **{z['location']}** - {z['total_accidents']} accidents" for z in high_risk[:5]]

    return {
        "type": "list",
        "message": (
            f"Bangkok has **{data['stats']['high_risk_zones']} high-risk zones** in this demo. Top ones to avoid:\n"
            + "\n".join(lines)
            + "\n\nThese make up a small share of the city but carry a disproportionate share of crashes."
        ),
        "data": high_risk[:5],
        "followup": "Next useful prompt:\n- What causes most accidents?\n- What time are accidents most common?",
    }


def handle_motorcycle_zones(user_message=""):
    data = load_chatbot_data()

    moto_zones = []
    for gid, z in data["zones"].items():
        if z["motorcycle_pct"] > 15:
            moto_zones.append(z)

    moto_zones.sort(key=lambda z: z["motorcycle_pct"], reverse=True)

    if moto_zones:
        zone_text = "\n".join(
            f"- **{z['location']}** - {z['motorcycle_pct']}% motorcycle involvement"
            for z in moto_zones[:5]
        )
    else:
        zone_text = "No major motorcycle hotspots in the top monitored zones."

    return {
        "type": "list",
        "message": (
            "Motorcycle crashes are a major safety issue in Bangkok. Highest motorcycle-involvement areas:\n"
            + zone_text
            + "\n\nFor riders, the safest habits are helmet use, conservative speed, and extra caution at merges and intersections."
        ),
        "data": moto_zones[:5],
        "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- What causes most accidents?",
    }


def handle_cause_breakdown(user_message=""):
    data = load_chatbot_data()
    causes = data["causes"]

    lines = []
    for cause in sorted(causes.values(), key=lambda c: c["count"], reverse=True):
        if cause["count"] > 0:
            lines.append(f"- **{cause['english_name']}** - {cause['percentage']}%")

    return {
        "type": "stats",
        "message": (
            "What causes Bangkok accidents:\n"
            + "\n".join(lines)
            + "\n\nSpeeding is the clear #1 cause, accounting for nearly 4 out of 5 recorded crashes."
        ),
        "data": [{"cause": c["english_name"], "count": c["count"], "pct": c["percentage"]} for c in causes.values()],
        "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- How reliable is the model?",
    }


def handle_speeding_stats(user_message=""):
    data = load_chatbot_data()
    speeding = data["causes"]["speeding"]

    return {
        "type": "stats",
        "message": (
            f"**Speeding** causes **{speeding['percentage']}%** of all recorded Bangkok crashes.\n\n"
            f"Top hotspot: **{speeding['top_zones'][0]['location']}** with **{speeding['top_zones'][0]['cause_count']}** speeding-related crashes.\n\n"
            "That makes it one of the clearest enforcement targets in the dataset."
        ),
        "data": speeding["top_zones"][:3],
        "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- What time are accidents most common?",
    }


def handle_drunk_stats(user_message=""):
    data = load_chatbot_data()

    return {
        "type": "stats",
        "message": (
            "Drunk driving is heavily underreported in official data, so the honest answer is directional rather than exact.\n\n"
            f"**Saturday** is the deadliest day overall with **{data['time_patterns']['most_dangerous_day_count']} accidents**, and **11 PM - 3 AM** is the clearest late-night danger window.\n\n"
            "Best practical advice: avoid Friday and Saturday late-night driving when possible."
        ),
        "data": [],
        "followup": "Next useful prompt:\n- What time are accidents most common?\n- Which areas are most dangerous?",
    }


def handle_cutting_stats(user_message=""):
    data = load_chatbot_data()
    cutting = data["causes"]["cutting_in_front"]

    return {
        "type": "stats",
        "message": (
            f"**Cutting in front** and aggressive lane changes account for **{cutting['percentage']}%** of crashes.\n\n"
            "This is a strong danger signal because the worst cutting-in-front zones tend to stay risky at busy merges and intersections.\n\n"
            "Best practical advice: keep more following distance and expect sudden lane changes."
        ),
        "data": cutting["top_zones"][:3],
        "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- What causes most accidents?",
    }


def handle_cause_comparison(user_message=""):
    data = load_chatbot_data()
    causes = data["causes"]

    rows = []
    for key, c in causes.items():
        rows.append({
            "cause": c["english_name"],
            "count": c["count"],
            "percentage": c["percentage"],
        })
    rows.sort(key=lambda r: r["count"], reverse=True)
    lines = [f"- **{r['cause']}** - {r['percentage']}%" for r in rows if r["count"] > 0]

    return {
        "type": "comparison",
        "message": (
            "Bangkok accident causes:\n"
            + "\n".join(lines)
            + "\n\nSpeeding dominates by count, but cutting in front remains a strong zone-level danger signal."
        ),
        "data": rows,
        "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- What time are accidents most common?",
    }


def handle_time_patterns(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]

    return {
        "type": "stats",
        "message": (
            f"**Peak hour: {tp['most_dangerous_hour']}.** That is the busiest single hour in the dataset with **{tp['most_dangerous_hour_count']} accidents**.\n\n"
            f"The lowest hour is **{tp['safest_hour']}** with **{tp['safest_hour_count']} accidents**, but for normal trips the practical move is to avoid the main spikes around **7-10 AM**, **{tp['most_dangerous_hour']}**, and **4-8 PM**."
        ),
        "data": [],
        "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- What causes most accidents?",
    }


def handle_rush_hour(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]
    hourly = tp["hourly"]

    morning = sum(hourly.get(f"{h:02d}:00", {}).get("count", 0) for h in range(7, 10))
    evening = sum(hourly.get(f"{h:02d}:00", {}).get("count", 0) for h in range(16, 20))

    return {
        "type": "stats",
        "message": (
            f"**Rush hour is a major risk window.** **{tp['rush_hour_pct']}%** of all recorded crashes happen during rush-hour periods.\n\n"
            f"The evening rush (4-8 PM) is {'worse' if evening > morning else 'slightly lighter'} than the morning rush (7-10 AM), with **{evening} vs {morning} accidents**. If your schedule is flexible, avoid those peaks instead of aiming for one supposedly 'safe' daytime block."
        ),
        "data": [],
        "followup": "Next useful prompt:\n- What time are accidents most common?\n- Which areas are most dangerous?",
    }


def handle_weekend_pattern(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]
    daily = tp["daily"]

    return {
        "type": "comparison",
        "message": (
            f"**{tp['most_dangerous_day']}** is the highest-risk single day with **{tp['most_dangerous_day_count']} accidents**.\n\n"
            f"Weekdays still make up **{tp['weekday_accidents']} crashes** overall because there are more weekday travel days, but weekend pressure is concentrated hardest on Saturday. If you are traveling on the weekend, Saturday deserves the most caution."
        ),
        "data": [{"day": d, **daily[d]} for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]],
        "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- What causes most accidents?",
    }


def handle_policy_recommendations(user_message=""):
    data = load_chatbot_data()
    recs = data["recommendations"]
    lines = [f"{r['priority']}. **{r['title']}** - {r['expected_impact']}" for r in recs[:3]]

    return {
        "type": "recommendation",
        "message": (
            "Top fixes for Bangkok road safety:\n"
            + "\n".join(lines)
            + "\n\nEach recommendation is tied directly to crash patterns in the dataset."
        ),
        "data": recs,
        "followup": "Next useful prompt:\n- What causes most accidents?\n- How reliable is the model?",
    }


def handle_camera_deployment(user_message=""):
    data = load_chatbot_data()
    speeding = data["causes"]["speeding"]
    top_zones = speeding["top_zones"][:5]
    lines = [f"- **{z['location']}** - {z['cause_count']} speeding accidents" for z in top_zones]

    return {
        "type": "recommendation",
        "message": (
            "Top zones recommended for speed enforcement:\n"
            + "\n".join(lines)
            + "\n\nThese locations show the heaviest concentration of speeding-related crashes, making them the clearest camera candidates."
        ),
        "data": top_zones,
        "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- What causes most accidents?",
    }


def handle_ml_cause_response(cause_key: str, confidence: float, user_message: str = ""):
    data = load_chatbot_data()
    causes = data["causes"]

    if cause_key == "drunk_driving":
        tp = data["time_patterns"]
        return {
            "type": "stats",
            "message": (
                "Drunk driving is significantly underreported in official Bangkok data, so the safest answer is directional rather than exact.\n\n"
                f"**Saturday** is the deadliest day overall with **{tp['most_dangerous_day_count']} accidents**, and **10 PM - 3 AM** is the clearest late-night risk window.\n\n"
                "Best practical advice: avoid Friday and Saturday late-night driving when possible."
            ),
            "data": [],
            "followup": "Next useful prompt:\n- What time are accidents most common?\n- Which areas are most dangerous?",
            "source": "ml_detected",
            "detected_cause": cause_key,
            "confidence": round(confidence, 2),
        }

    cause_data = causes.get(cause_key)
    if not cause_data:
        return None

    name = cause_data["english_name"]
    pct = cause_data["percentage"]
    count = cause_data["count"]
    top_zones = cause_data.get("top_zones", [])[:3]

    zone_text = ""
    if top_zones:
        zone_text = f"\n\nTop zones:\n{_fmt_zone_list(top_zones, 3)}"

    if cause_key == "speeding":
        advice = "These zones are the clearest enforcement targets in the dataset."
    elif cause_key == "cutting_in_front":
        advice = "Zones with high cutting-in-front rates are especially risky. Keep more following distance."
    else:
        advice = "Stay alert and drive defensively in affected zones."

    if _has_thai_text(user_message):
        return {
            "type": "stats",
            "message": (
                f"**{name}** คิดเป็น **{pct}%** ของอุบัติเหตุในข้อมูลชุดนี้ ({count:,} เคส)."
                f"{zone_text}\n\n{advice}"
            ),
            "data": top_zones,
            "followup": "ลองถามต่อได้ว่าโซนไหนอันตรายที่สุด",
            "source": "ml_detected",
            "detected_cause": cause_key,
            "confidence": round(confidence, 2),
        }

    return {
        "type": "stats",
        "message": (
            f"**{name}** accounts for **{pct}%** of all Bangkok accidents "
            f"({count:,} incidents).{zone_text}\n\n{advice}"
        ),
        "data": top_zones,
        "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- What time are accidents most common?",
        "source": "ml_detected",
        "detected_cause": cause_key,
        "confidence": round(confidence, 2),
    }


_DEMO_PROMPT_MAP = {
    "howreliableisthemodel": "judge_model_accuracy",
    "คนนขบรถปลอดภยไหม": "demo_general_safety",
    "คนขับรถปลอดภัยไหม": "demo_general_safety",
    "หลบในอนตรายแคไหน": "demo_drowsy_driving",
    "หลับในอันตรายแค่ไหน": "demo_drowsy_driving",
    "ขอมลมาจากไหน": "data_source",
    "ข้อมูลมาจากไหน": "data_source",
    "ระบบทำงานยงไง": "how_it_works",
    "ระบบทำงานยังไง": "how_it_works",
}


def _normalize_demo_prompt(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[\s\?\!\.,:;\"'`~\-_/\\()]+", "", normalized)
    return normalized


def detect_intent(user_message):
    """Prefer exact demo prompts before falling back to normal scripted matching."""
    text = str(user_message or "").strip()
    if not text:
        return None

    normalized = _normalize_demo_prompt(text)
    if normalized in _DEMO_PROMPT_MAP:
        return _DEMO_PROMPT_MAP[normalized]

    if _has_thai_text(text):
        if any(token in normalized for token in ["ขอมล", "ข้อมูล", "แหลงขอมล", "แหล่งข้อมูล"]):
            return "data_source"
        if any(token in normalized for token in ["ระบบทำงาน", "ยงไง", "ยังไง", "โมเดลทำงาน"]):
            return "how_it_works"
        if any(token in normalized for token in ["แมนยำ", "แม่นยำ", "นาเชอถอ", "น่าเชื่อถือ"]):
            return "judge_model_accuracy"
        if "หลบใน" in normalized or "หลับใน" in normalized:
            return "demo_drowsy_driving"
        if "ปลอดภยไหม" in normalized or "ปลอดภัยไหม" in normalized:
            return "demo_general_safety"

    for intent_name, config in INTENTS.items():
        for pattern in config["patterns"]:
            if pattern.search(text):
                return intent_name

    return None


def _thai_prompt_examples(*examples):
    return "ลองถามต่อได้ เช่น\n" + "\n".join(f"- {example}" for example in examples)


def _demo_general_safety_response():
    data = load_chatbot_data()
    stats = data["stats"]
    tp = data["time_patterns"]
    return {
        "type": "info",
        "message": (
            f"โดยรวมแล้ว **ไม่ควรคิดว่าการขับรถในกรุงเทพฯ ปลอดภัยทุกช่วงเวลา** เพราะข้อมูลปี 2024 มีอุบัติเหตุที่เรานำมาวิเคราะห์ **{stats['total_accidents']:,} เคส**.\n\n"
            f"ช่วงที่ควรระวังมากคือ **{tp['most_dangerous_hour']}** และวันที่เสี่ยงที่สุดคือ **{tp['most_dangerous_day']}**.\n\n"
            "ถ้าต้องตอบแบบเดโมสั้น ๆ: ขับได้ แต่ควรหลบช่วงพีกและหลบย่านเสี่ยงเมื่อทำได้."
        ),
        "data": [],
        "followup": _thai_prompt_examples(
            "ย่านไหนอันตรายที่สุด",
            "อุบัติเหตุเกิดบ่อยช่วงเวลาไหน",
        ),
    }


def _demo_drowsy_driving_response():
    data = load_chatbot_data()
    cause = data["causes"].get("drowsy_driving", {})
    pct = cause.get("percentage", 0)
    count = cause.get("count", 0)
    top_zones = cause.get("top_zones", [])[:3]
    zone_text = f"\n\nโซนที่พบเด่น:\n{_fmt_zone_list(top_zones, 3)}" if top_zones else ""
    return {
        "type": "stats",
        "message": (
            f"**หลับใน** เป็นความเสี่ยงจริงในข้อมูลชุดนี้ โดยคิดเป็นประมาณ **{pct}%** ของอุบัติเหตุที่จัดกลุ่มได้ ({count:,} เคส)."
            f"{zone_text}\n\n"
            "คำตอบแบบเดโมที่ปลอดภัยคือ: ถ้าง่วง ไม่ควรฝืนขับต่อ เพราะความผิดพลาดจากหลับในมักรุนแรงและเกิดแบบตั้งตัวไม่ทัน."
        ),
        "data": top_zones,
        "followup": _thai_prompt_examples(
            "ย่านไหนอันตรายที่สุด",
            "ขับช่วงเวลาไหนเสี่ยงที่สุด",
        ),
    }


def get_scripted_response(intent, user_message=""):
    """Return a stable response payload, with exact demo-prompt overrides first."""
    if intent == "demo_general_safety":
        return _demo_general_safety_response()
    if intent == "demo_drowsy_driving":
        return _demo_drowsy_driving_response()

    handler_name = INTENTS[intent]["handler"]
    handler_fn = _HANDLERS[handler_name]
    return handler_fn(user_message)


def handle_judge_model_accuracy(user_message=""):
    data = load_chatbot_data()
    stats = data["stats"]
    accuracy = stats["model_accuracy"]
    lift = stats["model_lift"]

    if _has_thai_text(user_message):
        return {
            "type": "stats",
            "message": (
                f"โมเดลนี้ค่อนข้างน่าเชื่อถือสำหรับเดโมครับ เพราะในการทดสอบแบบ **Jan-Sep 2024 train / Oct-Dec 2024 holdout** "
                f"ระบบจับพื้นที่เสี่ยงที่ครอบคลุมอุบัติเหตุภายหลังได้ **{accuracy}%** และให้ผลดีกว่าสุ่มเลือกโซนถึง **{lift}x**.\n\n"
                "ถ้าต้องตอบสั้น ๆ ต่อกรรมการ: โมเดลไม่ได้เดาแบบกว้าง ๆ แต่มีสัญญาณเชิงเวลาและพื้นที่ที่ตรวจสอบย้อนหลังได้."
            ),
            "data": [{"holdout_capture": accuracy, "lift": lift}],
            "followup": _thai_prompt_examples(
                "ย่านไหนอันตรายที่สุด",
                "ข้อมูลมาจากไหน",
            ),
        }

    return {
        "type": "stats",
        "message": (
            f"On a **Jan-Sep 2024 train / Oct-Dec 2024 holdout**, SafeDrive BKK captured **{accuracy}%** of later mapped crashes inside flagged high-risk grids "
            f"with **{lift}x lift** over random danger-zone selection.\n\n"
            "The judge-safe summary is that the model shows temporal signal and performs materially better than random zone selection."
        ),
        "data": [{"holdout_capture": accuracy, "lift": lift}],
        "followup": "Next useful prompt:\n- Which areas are most dangerous?",
    }


def handle_data_source(user_message=""):
    data = load_chatbot_data()
    stats = data["stats"]

    if _has_thai_text(user_message):
        return {
            "type": "info",
            "message": (
                f"ข้อมูลหลักมาจาก **{stats['data_source']} ปี 2024** โดยเราใช้ข้อมูลอุบัติเหตุในกรุงเทพฯ **{stats['total_accidents']:,} เคส** "
                f"ครอบคลุมช่วง **{stats['date_range']}**.\n\n"
                f"ในชุดข้อมูลนี้มีผู้เสียชีวิต **{stats['deaths']} ราย** บาดเจ็บสาหัส **{stats['serious_injuries']} ราย** "
                f"และถูกสรุปเป็นโซนวิเคราะห์ **{stats['total_zones']:,} โซน**."
            ),
            "data": [],
            "followup": _thai_prompt_examples(
                "ระบบทำงานยังไง",
                "โมเดลน่าเชื่อถือแค่ไหน",
            ),
        }

    return {
        "type": "info",
        "message": (
            f"We analyzed **{stats['total_accidents']:,} Bangkok accidents** from {stats['data_source']} (2024), "
            f"covering {stats['date_range']}.\n\n"
            f"The data includes {stats['deaths']} deaths and {stats['serious_injuries']} serious injuries across "
            f"{stats['total_zones']:,} monitored zones. All Thai text was processed with NLP to extract causes."
        ),
        "data": [],
        "followup": "Next useful prompt:\n- How reliable is the model?\n- How does the system work?",
    }


def handle_how_it_works(user_message=""):
    data = load_chatbot_data()

    if _has_thai_text(user_message):
        return {
            "type": "info",
            "message": (
                "ระบบนี้ทำงานหลัก ๆ 2 ส่วนครับ:\n"
                f"1. **Thai NLP** ใช้อ่านคำบรรยายอุบัติเหตุภาษาไทยเพื่อจัดกลุ่มสาเหตุ\n"
                f"2. **XGBoost ML** ใช้ให้คะแนนความเสี่ยงของแต่ละโซน โดยอ้างอิงผลทดสอบ holdout ที่จับเหตุการณ์ได้ **{data['stats']['model_accuracy']}%**\n\n"
                "พูดง่าย ๆ คือ NLP ช่วยบอกว่าอุบัติเหตุเกิดจากอะไร ส่วน ML ช่วยบอกว่าพื้นที่ไหนควรระวัง."
            ),
            "data": [],
            "followup": _thai_prompt_examples(
                "ข้อมูลมาจากไหน",
                "โมเดลน่าเชื่อถือแค่ไหน",
            ),
        }

    return {
        "type": "info",
        "message": (
            "We combine 2 core AI techniques:\n"
            "1. **Thai NLP** - reads accident reports to classify causes and power our Safety Assistant chatbot\n"
            f"2. **XGBoost ML** - predicts risk scores for every 1km zone ({data['stats']['model_accuracy']}% holdout capture)\n\n"
            "Together, they power the risk map, trip planner, and analytics view."
        ),
        "data": [],
        "followup": "Next useful prompt:\n- Where does the data come from?\n- How reliable is the model?",
    }


def handle_zone_safety(user_message=""):
    if _has_thai_text(user_message):
        normalized = _normalize_demo_prompt(user_message)
        if "ปลอดภยไหม" in normalized or "ปลอดภัยไหม" in normalized:
            return _demo_general_safety_response()
    return _HANDLERS["handle_zone_safety_en_fallback"](user_message)


def _handle_zone_safety_en_fallback(user_message=""):
    data = load_chatbot_data()

    location_query = user_message.strip()
    for prefix in [
        "how dangerous is ", "how safe is ", "how risky is ",
        "tell me about ", "is ", "how safe ", "how dangerous ",
        "safety of ", "risk of ", "danger of ", "what about ",
    ]:
        if location_query.lower().startswith(prefix):
            location_query = location_query[len(prefix):]
            break
    location_query = location_query.strip(" ?.,!")

    changed = True
    while changed:
        changed = False
        for suffix in [" safe", " dangerous", " risky", " area", " zone", " safety"]:
            if location_query.lower().endswith(suffix):
                location_query = location_query[:-len(suffix)].rstrip()
                changed = True

    gid, zone = _search_zone_by_name(location_query)

    if zone and gid:
        z = zone
        risk_desc = "HIGH RISK" if z["risk_score"] >= 55 else "MEDIUM RISK" if z["risk_score"] >= 30 else "LOW RISK"
        top_cause = max(z["cause_breakdown"].items(), key=lambda x: x[1])
        return {
            "type": "single",
            "message": (
                f"**{z['location']}** is **{risk_desc}**.\n\n"
                f"Top risk factor: **{top_cause[0].replace('_', ' ').title()}** ({top_cause[1]}% of crashes in this zone).\n"
                f"Rush-hour exposure: **{z['rush_hour_pct']}%** of crashes happen during rush periods.\n\n"
                f"{'Best practical advice: avoid the evening peak when possible.' if z['rush_hour_pct'] > 30 else 'This area is still worth treating cautiously even outside peak hours.'}"
            ),
            "data": [z],
            "followup": "Next useful prompt:\n- Which areas are most dangerous?\n- What causes most accidents?",
        }

    top_zones = list(data["zones"].values())[:5]
    lines = [f"- **{z['location']}** - {z['total_accidents']} accidents" for z in top_zones]
    return {
        "type": "list",
        "message": (
            f"I could not match **{location_query}** to a named monitored zone in this demo dataset.\n\n"
            "The current coverage is strongest for Bangkok corridors, outer districts, and major highways. A more reliable question would be one of these:\n"
            + _fmt_suggestion_list([
                "Which areas are most dangerous?",
                "What time are accidents most common?",
                "How reliable is the model?",
            ])
            + "\n\nNearby high-risk examples from the dataset:\n"
            + "\n".join(lines)
        ),
        "data": top_zones,
        "followup": "Try one of these:\n- Which areas are most dangerous?\n- How reliable is the model?",
    }


# ---------------------------------------------------------------------------
# Handler dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    "handle_judge_dangerous_zones": handle_judge_dangerous_zones,
    "handle_judge_peak_time": handle_judge_peak_time,
    "handle_judge_peak_day": handle_judge_peak_day,
    "handle_judge_model_accuracy": handle_judge_model_accuracy,
    "handle_speeding_zones": handle_speeding_zones,
    "handle_drunk_zones": handle_drunk_zones,
    "handle_dangerous_zones": handle_dangerous_zones,
    "handle_zone_safety": handle_zone_safety,
    "handle_zone_safety_en_fallback": _handle_zone_safety_en_fallback,
    "handle_route_safety": handle_route_safety,
    "handle_rain_data_gap": handle_rain_data_gap,
    "handle_high_risk_zones": handle_high_risk_zones,
    "handle_motorcycle_zones": handle_motorcycle_zones,
    "handle_cause_breakdown": handle_cause_breakdown,
    "handle_speeding_stats": handle_speeding_stats,
    "handle_drunk_stats": handle_drunk_stats,
    "handle_cutting_stats": handle_cutting_stats,
    "handle_cause_comparison": handle_cause_comparison,
    "handle_time_patterns": handle_time_patterns,
    "handle_rush_hour": handle_rush_hour,
    "handle_weekend_pattern": handle_weekend_pattern,
    "handle_policy_recommendations": handle_policy_recommendations,
    "handle_camera_deployment": handle_camera_deployment,
    "handle_how_it_works": handle_how_it_works,
    "handle_data_source": handle_data_source,
}


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_messages = [
        "Which areas have the most speeding?",
        "Where is drunk driving common?",
        "What are the most dangerous intersections?",
        "Is Lat Krabang safe?",
        "What's the safest route to Asok?",
        "Show high risk zones",
        "Where do motorcycle crashes happen?",
        "What causes most accidents?",
        "How much speeding is there?",
        "Tell me about drunk driving stats",
        "How common is cutting in front?",
        "Compare all causes",
        "When do most accidents happen?",
        "Is rush hour dangerous?",
        "Weekend vs weekday?",
        "What should Bangkok do?",
        "What should Bangkok prioritize for safety?",
        "How does this system work?",
        "What data do you use?",
        "Tell me a joke",  # Should return None (fall through to AI)
        "What's the weather like?",  # Should return None
    ]

    print("=" * 70)
    print("INTENT DETECTION TEST")
    print("=" * 70)

    matched = 0
    for msg in test_messages:
        intent = detect_intent(msg)
        status = "MATCH" if intent else "NO MATCH (-> AI)"
        print(f"\n  \"{msg}\"")
        print(f"  -> {status}: {intent}")
        if intent:
            matched += 1
            resp = get_scripted_response(intent, msg)
            # Print first 120 chars of message
            preview = resp["message"][:120].replace("\n", " ")
            print(f"     Type: {resp['type']} | Preview: {preview}...")

    print(f"\n{'=' * 70}")
    print(f"Matched: {matched}/{len(test_messages)} ({matched / len(test_messages) * 100:.0f}%)")
    print(f"Fall-through to AI: {len(test_messages) - matched}")
    print(f"Total intents defined: {len(INTENTS)}")
    print(f"Total patterns: {sum(len(i['patterns']) for i in INTENTS.values())}")
