"""
Intent Detection & Response System for RoadRisk Bangkok Chatbot

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


def _fmt_zone_list(zones, limit=5):
    """Format a list of zone dicts into numbered lines."""
    lines = []
    for i, z in enumerate(zones[:limit], 1):
        risk_badge = "HIGH" if z.get("risk_score", 0) >= 55 else "MEDIUM" if z.get("risk_score", 0) >= 30 else "LOW"
        lines.append(
            f"{i}. {z['location']} — {z.get('cause_count', z.get('total_accidents', '?'))} accidents, "
            f"risk score {z.get('risk_score', '?')} ({risk_badge})"
        )
    return "\n".join(lines)


def _search_zone_by_name(query):
    """Find a zone whose location name matches the query (fuzzy)."""
    data = load_chatbot_data()
    query_lower = query.lower()
    for gid, zone in data["zones"].items():
        if query_lower in zone["location"].lower():
            return gid, zone
    # Also check camera locations
    for cam in data["cameras"]:
        if query_lower in cam["location"].lower():
            # Find nearest zone — return first zone as fallback
            return None, cam
    return None, None


# ---------------------------------------------------------------------------
# Intent patterns  (20 intents)
# ---------------------------------------------------------------------------

INTENTS = {
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
        ],
        "handler": "handle_route_safety",
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
    "camera_locations": {
        "patterns": [
            re.compile(r"(?:where|which|show|list).*cameras?", re.I),
            re.compile(r"cameras?.*(?:locations?|placed?)", re.I),
            re.compile(r"(?:cctv|surveillance|traffic).*cameras?", re.I),
            re.compile(r"(?:yolo|detection).*cameras?", re.I),
        ],
        "handler": "handle_camera_locations",
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
            re.compile(r"(?:most|peak).*dangerous.*(?:time|hour|day|month|period)", re.I),
            re.compile(r"(?:what|which).*(?:time|hour|day|month).*(?:dangerous|worst|most)", re.I),
            re.compile(r"(?:time|temporal|hourly|daily|monthly).*(?:pattern|trend|distribution)", re.I),
            re.compile(r"when.*(?:should|shouldn't).*(?:drive|travel|avoid)", re.I),
            re.compile(r"(?:peak|worst).*(?:hours?|times?|days?)", re.I),
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
            re.compile(r"(?:xgboost|yolo|nlp|machine.?learning).*(?:how|explain|work)", re.I),
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

# ── Location-based ────────────────────────────────────────────────────

def handle_speeding_zones(user_message=""):
    data = load_chatbot_data()
    speeding = data["causes"]["speeding"]
    total = data["stats"]["total_accidents"]

    return {
        "type": "list",
        "message": (
            f"Speeding is the #1 cause of accidents in Bangkok, responsible for "
            f"{speeding['percentage']}% of all {total:,} accidents ({speeding['count']:,} cases).\n\n"
            f"Top speeding hotspots:\n{_fmt_zone_list(speeding['top_zones'], 5)}"
        ),
        "data": speeding["top_zones"][:5],
        "followup": "Would you like details on any specific zone, or want to see the cause breakdown?",
    }


def handle_drunk_zones(user_message=""):
    data = load_chatbot_data()
    stats = data["stats"]

    # Drunk driving data from thai_keywords
    drunk_kw = data["thai_keywords"].get("drunk_driving", {})
    drunk_words = drunk_kw.get("related_words", [])
    drunk_freq = drunk_words[0]["frequency"] if drunk_words else 0

    # Find zones with high weekend % (proxy for alcohol-related risk)
    weekend_zones = []
    for gid, zone in data["zones"].items():
        if zone["weekend_pct"] > 40:
            weekend_zones.append({
                "grid_id": gid,
                "location": zone["location"],
                "total_accidents": zone["total_accidents"],
                "weekend_pct": zone["weekend_pct"],
                "risk_score": zone["risk_score"],
            })
    weekend_zones.sort(key=lambda z: z["weekend_pct"], reverse=True)

    zone_text = ""
    if weekend_zones:
        lines = []
        for i, z in enumerate(weekend_zones[:5], 1):
            lines.append(
                f"{i}. {z['location']} — {z['weekend_pct']}% weekend accidents, "
                f"risk score {z['risk_score']}"
            )
        zone_text = "\n\nZones with highest weekend accident concentration (often alcohol-related):\n" + "\n".join(lines)

    return {
        "type": "list",
        "message": (
            f"Our dataset recorded {drunk_freq} explicitly drunk-driving incidents in Bangkok (2024) "
            f"from the Thai term 'เมาสุรา'. However, alcohol involvement is likely underreported in "
            f"official records.\n\n"
            f"Weekend accidents (28.6% of total) often correlate with alcohol — "
            f"Saturday is the most dangerous day with {data['time_patterns']['most_dangerous_day_count']} accidents."
            f"{zone_text}"
        ),
        "data": weekend_zones[:5],
        "followup": "Want to see the weekend vs weekday breakdown, or time-of-day patterns?",
    }


def handle_dangerous_zones(user_message=""):
    data = load_chatbot_data()
    zones = data["zones"]
    stats = data["stats"]

    top5 = []
    for gid, z in list(zones.items())[:5]:
        top5.append({
            "grid_id": gid,
            "location": z["location"],
            "total_accidents": z["total_accidents"],
            "risk_score": z["risk_score"],
            "dominant_cause": z["dominant_cause"],
            "dominant_cause_pct": z["dominant_cause_pct"],
            "has_fatality": z["has_fatality"],
        })

    lines = []
    for i, z in enumerate(top5, 1):
        fatality_flag = " ⚠ fatalities recorded" if z["has_fatality"] else ""
        lines.append(
            f"{i}. {z['location']} — {z['total_accidents']} accidents, "
            f"risk {z['risk_score']}, main cause: {z['dominant_cause']} ({z['dominant_cause_pct']}%)"
            f"{fatality_flag}"
        )

    return {
        "type": "list",
        "message": (
            f"Bangkok has {stats['high_risk_zones']} high-risk zones out of {stats['total_zones']:,} "
            f"monitored grid cells. The top 5% of zones contain the majority of accidents.\n\n"
            f"Top 5 most dangerous zones:\n" + "\n".join(lines)
        ),
        "data": top5,
        "followup": "Want to know what causes accidents in a specific zone, or see the full heatmap?",
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
        cause_lines = []
        for cause, pct in sorted(z["cause_breakdown"].items(), key=lambda x: x[1], reverse=True):
            if pct > 0:
                cause_lines.append(f"  - {cause.replace('_', ' ').title()}: {pct}%")

        risk_desc = "HIGH RISK" if z["risk_score"] >= 55 else "MEDIUM RISK" if z["risk_score"] >= 30 else "LOW RISK"

        return {
            "type": "single",
            "message": (
                f"Zone {gid} — {z['location']}\n"
                f"Risk level: {risk_desc} (score: {z['risk_score']})\n"
                f"Total accidents: {z['total_accidents']}\n"
                f"Fatalities recorded: {'Yes' if z['has_fatality'] else 'No'}\n"
                f"Motorcycle involvement: {z['motorcycle_pct']}%\n"
                f"Rush hour accidents: {z['rush_hour_pct']}%\n\n"
                f"Cause breakdown:\n" + "\n".join(cause_lines)
            ),
            "data": [z],
            "followup": "Would you like route recommendations to avoid this area?",
        }
    elif zone:
        # Matched a camera, not a zone
        return {
            "type": "single",
            "message": (
                f"We have a YOLOv8 camera at {zone['location']} "
                f"(coordinates: {zone['lat']}, {zone['lon']}). "
                f"This camera monitors real-time vehicle composition for our risk model.\n\n"
                f"For detailed zone risk data, try asking about one of our top 30 monitored zones."
            ),
            "data": [zone],
            "followup": "Want to see all camera locations or the most dangerous zones?",
        }
    else:
        # No match — list what we know
        available = [z["location"] for z in list(data["zones"].values())[:10]]
        return {
            "type": "single",
            "message": (
                f"I don't have specific data for \"{location_query}\". "
                f"Our model covers {data['stats']['total_zones']:,} grid zones in Bangkok.\n\n"
                f"Try asking about one of these high-risk areas:\n" +
                "\n".join(f"  - {loc}" for loc in available)
            ),
            "data": [],
            "followup": "Or try the Trip Planner for route-specific risk analysis.",
        }


def handle_route_safety(user_message=""):
    return {
        "type": "info",
        "message": (
            "For personalized route safety analysis, use our Trip Planner!\n\n"
            "It compares multiple routes using the XGBoost risk model and shows:\n"
            "- Risk score per 1km segment\n"
            "- Number of danger zones on each route\n"
            "- Safer alternative routes that avoid high-risk areas\n"
            "- Real-time navigation with voice warnings\n\n"
            "Enter your start and destination in the Trip Planner to get "
            "AI-powered route recommendations."
        ),
        "data": [],
        "followup": "Would you like to know which areas to avoid, or see the most dangerous zones?",
    }


def handle_camera_locations(user_message=""):
    data = load_chatbot_data()
    cameras = data["cameras"]

    lines = []
    for cam in cameras:
        lines.append(f"  - {cam['camera_id']}: {cam['location']} ({cam['lat']}, {cam['lon']})")

    return {
        "type": "list",
        "message": (
            f"We have {len(cameras)} YOLOv8 traffic cameras deployed across Bangkok as a "
            f"proof-of-concept for our smart city monitoring network:\n\n" +
            "\n".join(lines) +
            f"\n\nThese cameras detect vehicles in real-time and feed vehicle composition data "
            f"(car/motorcycle/truck ratios) into our XGBoost risk prediction model."
        ),
        "data": cameras,
        "followup": "Want to know how vehicle detection feeds into risk prediction?",
    }


def handle_high_risk_zones(user_message=""):
    data = load_chatbot_data()
    high_risk = []
    for gid, z in data["zones"].items():
        if z["risk_level"] == "high":
            high_risk.append({"grid_id": gid, **z})

    high_risk.sort(key=lambda z: z["risk_score"], reverse=True)

    lines = []
    for i, z in enumerate(high_risk[:10], 1):
        lines.append(
            f"{i}. {z['location']} — {z['total_accidents']} accidents, "
            f"risk {z['risk_score']}"
        )

    return {
        "type": "list",
        "message": (
            f"Bangkok has {data['stats']['high_risk_zones']} high-risk zones (risk score >= 55). "
            f"These zones represent about 5% of the city but contain the majority of accidents.\n\n"
            f"Top 10 high-risk zones:\n" + "\n".join(lines)
        ),
        "data": high_risk[:10],
        "followup": "Want to dive into a specific zone, or see recommendations for reducing risk?",
    }


def handle_motorcycle_zones(user_message=""):
    data = load_chatbot_data()

    moto_zones = []
    for gid, z in data["zones"].items():
        if z["motorcycle_pct"] > 15:
            moto_zones.append({"grid_id": gid, **z})

    moto_zones.sort(key=lambda z: z["motorcycle_pct"], reverse=True)

    lines = []
    for i, z in enumerate(moto_zones[:5], 1):
        lines.append(
            f"{i}. {z['location']} — {z['motorcycle_pct']}% motorcycle involvement, "
            f"{z['total_accidents']} accidents, risk {z['risk_score']}"
        )

    if not lines:
        lines = ["No zones with >15% motorcycle involvement in the top 30."]

    return {
        "type": "list",
        "message": (
            f"Motorcycle accidents are a major concern in Bangkok. "
            f"Our model tracks motorcycle involvement as a percentage of vehicles per zone.\n\n"
            f"Zones with highest motorcycle involvement:\n" + "\n".join(lines)
        ),
        "data": moto_zones[:5],
        "followup": "Want to see how motorcycle density correlates with accident severity?",
    }


# ── Cause-based ──────────────────────────────────────────────────────

def handle_cause_breakdown(user_message=""):
    data = load_chatbot_data()
    causes = data["causes"]
    total = data["stats"]["total_accidents"]

    lines = []
    for key, cause in causes.items():
        if cause["count"] > 0:
            lines.append(f"  - {cause['english_name']}: {cause['count']:,} ({cause['percentage']}%)")

    return {
        "type": "stats",
        "message": (
            f"Analysis of {total:,} Bangkok accident reports (2024) using Thai NLP "
            f"text classification identified these causes:\n\n" +
            "\n".join(lines) +
            f"\n\nKey insight: Speeding alone accounts for nearly 4 out of 5 accidents. "
            f"The Thai term 'ขับรถเร็วเกินอัตรากำหนด' (driving faster than the speed limit) "
            f"appears in {causes['speeding']['count']:,} reports."
        ),
        "data": [{"cause": c["english_name"], "count": c["count"], "pct": c["percentage"]}
                 for c in causes.values()],
        "followup": "Want to see which zones are worst for a specific cause?",
    }


def handle_speeding_stats(user_message=""):
    data = load_chatbot_data()
    speeding = data["causes"]["speeding"]
    total = data["stats"]["total_accidents"]
    importance = data["model_importance"]

    return {
        "type": "stats",
        "message": (
            f"Speeding in Bangkok (2024):\n\n"
            f"- {speeding['count']:,} out of {total:,} accidents ({speeding['percentage']}%)\n"
            f"- Thai NLP term: '{speeding['thai_name']}'\n"
            f"- XGBoost feature importance: {speeding['feature_importance']}%\n"
            f"- Most common collision type in speeding zones: rear-end collision (ชนท้าย)\n"
            f"- {len(speeding['top_zones'])} high-concentration zones identified\n\n"
            f"Top speeding zone: {speeding['top_zones'][0]['location']} with "
            f"{speeding['top_zones'][0]['cause_count']} speeding-related accidents."
        ),
        "data": speeding["top_zones"][:3],
        "followup": "Want to see the top speeding zones on a map, or compare with other causes?",
    }


def handle_drunk_stats(user_message=""):
    data = load_chatbot_data()
    drunk_kw = data["thai_keywords"].get("drunk_driving", {})
    drunk_words = drunk_kw.get("related_words", [])
    drunk_freq = drunk_words[0]["frequency"] if drunk_words else 0
    total = data["stats"]["total_accidents"]

    return {
        "type": "stats",
        "message": (
            f"Drunk driving in Bangkok (2024):\n\n"
            f"- {drunk_freq} explicitly recorded cases out of {total:,} accidents\n"
            f"- Thai NLP term: 'เมาสุรา'\n"
            f"- This represents only {round(drunk_freq / total * 100, 2)}% of records\n\n"
            f"Important caveat: Drunk driving is significantly underreported in Thai "
            f"government accident data. The official cause classification relies on "
            f"the investigating officer's assessment, and alcohol testing is not "
            f"performed at every accident scene.\n\n"
            f"Proxy indicator: Saturday has the most accidents "
            f"({data['time_patterns']['most_dangerous_day_count']}), and late-night hours "
            f"(11PM-3AM) show elevated risk — patterns consistent with alcohol involvement."
        ),
        "data": [],
        "followup": "Want to see weekend vs weekday patterns, or time-of-day analysis?",
    }


def handle_cutting_stats(user_message=""):
    data = load_chatbot_data()
    cutting = data["causes"]["cutting_in_front"]
    total = data["stats"]["total_accidents"]

    return {
        "type": "stats",
        "message": (
            f"Cutting in front (aggressive lane changes) in Bangkok (2024):\n\n"
            f"- {cutting['count']:,} cases out of {total:,} accidents ({cutting['percentage']}%)\n"
            f"- Thai NLP term: '{cutting['thai_name']}'\n"
            f"- XGBoost feature importance: {cutting['feature_importance']}%\n"
            f"- This is the 2nd most common cause after speeding\n\n"
            f"Despite being only {cutting['percentage']}% of total accidents, cutting-in-front "
            f"has a {cutting['feature_importance']}% feature importance in our XGBoost model — "
            f"meaning it's a stronger predictor of zone danger than its raw frequency suggests."
        ),
        "data": cutting["top_zones"][:3],
        "followup": "Want to see which intersections have the most cutting-in-front incidents?",
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
            "importance": c["feature_importance"],
        })
    rows.sort(key=lambda r: r["count"], reverse=True)

    lines = []
    for r in rows:
        bar_len = int(r["percentage"] / 2)
        bar = "█" * bar_len
        lines.append(f"  {r['cause']:<45} {r['count']:>5}  ({r['percentage']:>5.1f}%)  {bar}")

    return {
        "type": "comparison",
        "message": (
            f"Cause comparison across {data['stats']['total_accidents']:,} Bangkok accidents (2024):\n\n" +
            "\n".join(lines) +
            f"\n\nSpeeding is overwhelmingly dominant. However, XGBoost feature importance "
            f"reveals that 'cutting in front' ({causes['cutting_in_front']['feature_importance']}%) "
            f"is a stronger risk predictor than speeding ({causes['speeding']['feature_importance']}%) "
            f"despite far fewer incidents — suggesting cutting-in-front concentrates in "
            f"genuinely dangerous zones."
        ),
        "data": rows,
        "followup": "Want to see the spatial distribution of any specific cause?",
    }


# ── Time-based ───────────────────────────────────────────────────────

def handle_time_patterns(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]

    # Find top 3 hours
    hourly = tp["hourly"]
    top_hours = sorted(hourly.items(), key=lambda x: x[1]["count"], reverse=True)[:3]

    return {
        "type": "stats",
        "message": (
            f"When do Bangkok accidents happen? (2024 data)\n\n"
            f"Peak hour:  {tp['most_dangerous_hour']} ({tp['most_dangerous_hour_count']} accidents)\n"
            f"Safest hour: {tp['safest_hour']} ({tp['safest_hour_count']} accidents)\n"
            f"Peak day:   {tp['most_dangerous_day']} ({tp['most_dangerous_day_count']} accidents)\n"
            f"Peak month: {tp['most_dangerous_month']} ({tp['most_dangerous_month_count']} accidents)\n\n"
            f"Top 3 most dangerous hours:\n" +
            "\n".join(f"  - {h}: {d['count']} accidents ({d['percentage']}%)" for h, d in top_hours) +
            f"\n\nRush hour accounts for {tp['rush_hour_pct']}% of all accidents. "
            f"Weekends account for {tp['weekend_pct']}% despite being only 2/7 of the week."
        ),
        "data": [{"hour": h, **d} for h, d in top_hours],
        "followup": "Want to see the rush hour breakdown or weekend patterns?",
    }


def handle_rush_hour(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]
    total = data["stats"]["total_accidents"]
    hourly = tp["hourly"]

    # Morning rush (7-9), evening rush (16-19)
    morning = sum(hourly.get(f"{h:02d}:00", {}).get("count", 0) for h in range(7, 10))
    evening = sum(hourly.get(f"{h:02d}:00", {}).get("count", 0) for h in range(16, 20))

    return {
        "type": "stats",
        "message": (
            f"Rush hour safety analysis:\n\n"
            f"Rush hour accidents: {tp['rush_hour_accidents']:,} ({tp['rush_hour_pct']}% of total)\n\n"
            f"Morning rush (7-10 AM): {morning} accidents\n"
            f"Evening rush (4-8 PM): {evening} accidents\n\n"
            f"The evening rush is {'more' if evening > morning else 'less'} dangerous than morning, "
            f"with {abs(evening - morning)} {'more' if evening > morning else 'fewer'} accidents.\n\n"
            f"However, the single most dangerous hour is {tp['most_dangerous_hour']} — "
            f"which falls {'within' if '07' <= tp['most_dangerous_hour'][:2] <= '09' or '16' <= tp['most_dangerous_hour'][:2] <= '19' else 'outside'} "
            f"traditional rush hours."
        ),
        "data": [],
        "followup": "Want to see the full 24-hour breakdown or weekend patterns?",
    }


def handle_weekend_pattern(user_message=""):
    data = load_chatbot_data()
    tp = data["time_patterns"]
    daily = tp["daily"]

    weekend_total = sum(daily[d]["count"] for d in ["Saturday", "Sunday"])
    weekday_total = sum(daily[d]["count"] for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    weekend_daily_avg = weekend_total / 2
    weekday_daily_avg = weekday_total / 5

    lines = []
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        d = daily[day]
        marker = " <<<" if day in ["Saturday", "Sunday"] else ""
        lines.append(f"  {day:<12} {d['count']:>4} accidents ({d['percentage']}%){marker}")

    return {
        "type": "comparison",
        "message": (
            f"Weekend vs Weekday comparison:\n\n"
            f"Weekend total: {weekend_total} accidents ({tp['weekend_pct']}%)\n"
            f"Weekday total: {weekday_total} accidents ({round(100 - tp['weekend_pct'], 1)}%)\n\n"
            f"Daily average:\n"
            f"  Weekend: {weekend_daily_avg:.0f} accidents/day\n"
            f"  Weekday: {weekday_daily_avg:.0f} accidents/day\n\n"
            f"Per-day breakdown:\n" + "\n".join(lines) +
            f"\n\n{tp['most_dangerous_day']} is the most dangerous day overall. "
            f"The weekend daily average is {'higher' if weekend_daily_avg > weekday_daily_avg else 'lower'} "
            f"than weekday — {round(weekend_daily_avg / weekday_daily_avg, 2)}x the weekday rate."
        ),
        "data": [{"day": d, **daily[d]} for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]],
        "followup": "Want to see which zones are most dangerous on weekends specifically?",
    }


# ── Recommendation ───────────────────────────────────────────────────

def handle_policy_recommendations(user_message=""):
    data = load_chatbot_data()
    recs = data["recommendations"]
    stats = data["stats"]

    lines = []
    for r in recs:
        lines.append(
            f"#{r['priority']}. {r['title']}\n"
            f"   Target: {r['target_cause']} | Zones: {r['zones_affected']} | "
            f"Impact: {r['expected_impact']}\n"
            f"   Cost: {r['cost_estimate']}\n"
            f"   {r['description']}"
        )

    return {
        "type": "recommendation",
        "message": (
            f"Based on analysis of {stats['total_accidents']:,} accidents across "
            f"{stats['high_risk_zones']} high-risk zones, here are our top policy "
            f"recommendations:\n\n" + "\n\n".join(lines) +
            f"\n\nThese recommendations are derived from NLP classification of Thai accident "
            f"reports combined with spatial risk modeling."
        ),
        "data": recs,
        "followup": "Want details on camera deployment strategy, or the data behind a specific recommendation?",
    }


def handle_camera_deployment(user_message=""):
    data = load_chatbot_data()
    speeding = data["causes"]["speeding"]
    top_zones = speeding["top_zones"][:5]

    lines = []
    for i, z in enumerate(top_zones, 1):
        lines.append(
            f"{i}. {z['location']} (Zone {z['grid_id']}) — "
            f"{z['cause_pct']}% speeding rate, {z['accidents']} total accidents"
        )

    return {
        "type": "recommendation",
        "message": (
            f"Speed camera deployment strategy based on our analysis:\n\n"
            f"Priority locations (highest speeding concentration):\n" +
            "\n".join(lines) +
            f"\n\nDeployment rationale:\n"
            f"- Speeding causes {speeding['percentage']}% of all accidents\n"
            f"- The top 5 speeding zones account for {sum(z['cause_count'] for z in top_zones)} "
            f"speeding-related accidents\n"
            f"- Estimated impact: 30-40% reduction in speed-related accidents per zone\n"
            f"- Cost: approximately ฿2.5M per camera installation\n\n"
            f"We currently have 7 YOLOv8 cameras as proof-of-concept. Scaling to these "
            f"priority locations would provide real-time vehicle speed monitoring."
        ),
        "data": top_zones,
        "followup": "Want to see the current camera network, or other policy recommendations?",
    }


# ── Meta ─────────────────────────────────────────────────────────────

def handle_how_it_works(user_message=""):
    data = load_chatbot_data()
    importance = data["model_importance"]

    return {
        "type": "info",
        "message": (
            f"RoadRisk Bangkok uses three AI techniques in one integrated system:\n\n"
            f"1. Thai NLP (Track 1)\n"
            f"   - Processes {data['stats']['total_accidents']:,} Thai accident reports\n"
            f"   - pythainlp tokenization classifies causes (speeding, cutting in front, etc.)\n"
            f"   - NLP features contribute {importance['nlp_text']}% to the prediction model\n\n"
            f"2. XGBoost Risk Prediction (Core Engine)\n"
            f"   - 44 features per 1km grid cell\n"
            f"   - Trained on Jan-Sep 2024, validated on Oct-Dec 2024\n"
            f"   - {data['stats']['model_accuracy']}% accuracy, {data['stats']['model_lift']}x better than random\n"
            f"   - Top predictor: {importance['top_feature']}\n\n"
            f"3. YOLOv8 Vehicle Detection (Track 3)\n"
            f"   - 7 Bangkok cameras detect cars, motorcycles, trucks in real-time\n"
            f"   - Vehicle composition feeds into XGBoost as live features\n"
            f"   - Vehicle features contribute {importance['vehicle']}% to model importance\n\n"
            f"Key insight: {importance['key_insight']}"
        ),
        "data": [],
        "followup": "Want to know about the data source, or how the Trip Planner uses this model?",
    }


def handle_data_source(user_message=""):
    data = load_chatbot_data()
    stats = data["stats"]

    return {
        "type": "info",
        "message": (
            f"Our data comes from the {stats['data_source']} (2024):\n\n"
            f"- Total records: {stats['total_accidents']:,} Bangkok accidents\n"
            f"- Coverage: {stats['date_range']}\n"
            f"- Deaths: {stats['deaths']}\n"
            f"- Serious injuries: {stats['serious_injuries']}\n"
            f"- Minor injuries: {stats['minor_injuries']}\n"
            f"- Grid zones monitored: {stats['total_zones']:,} (1km x 1km cells)\n"
            f"- High-risk zones identified: {stats['high_risk_zones']}\n"
            f"- Danger zones detected by model: {stats['danger_zones_detected']}\n\n"
            f"The original dataset contains 23,952 records nationwide. We filtered to "
            f"Bangkok and cleaned GPS coordinates, timestamps, and Thai text fields.\n\n"
            f"NLP processing: Thai accident cause text was tokenized with pythainlp and "
            f"classified into {len(data['causes'])} cause categories."
        ),
        "data": [],
        "followup": "Want to learn how the model works, or see the cause breakdown?",
    }


# ---------------------------------------------------------------------------
# Handler dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    "handle_speeding_zones": handle_speeding_zones,
    "handle_drunk_zones": handle_drunk_zones,
    "handle_dangerous_zones": handle_dangerous_zones,
    "handle_zone_safety": handle_zone_safety,
    "handle_route_safety": handle_route_safety,
    "handle_camera_locations": handle_camera_locations,
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
        "Where are the cameras?",
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
        "Where should cameras be installed?",
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
