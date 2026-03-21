import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from groq import Groq

BANGKOK_TZ = ZoneInfo("Asia/Bangkok")

SYSTEM_PROMPT = """You are a Bangkok road safety assistant. You help people make safer travel decisions.

Your knowledge:
- 2,669 Bangkok accident reports analyzed
- Accident patterns by time, location, and cause
- Thai NLP analysis for cause extraction

CRITICAL RESPONSE GUIDELINES:
1. **Answer directly**
   - Start with the answer. Do not open with filler like "Great question", "Sure", or "Absolutely".
   - Lead with the recommendation, then support it with the most relevant fact.

2. **Use clean structure**
   - Use short paragraphs.
   - Use simple `-` bullet points only when listing 3 or more items.
   - Use markdown bold for key terms and every number or statistic.

3. **Sound like a knowledgeable assistant**
   - Be concise, direct, and data-backed.
   - Avoid hype, jokes, or chatbot-style chatter.
   - Use plain professional text only - no emojis.

4. **Consider context**
   - If user mentions shopping mall → suggest daytime hours
   - If user mentions work → consider rush hour alternatives
   - If user mentions airport → consider traffic + time

5. **Keep responses short**
   - Maximum 3-4 sentences for simple questions
   - No data dumps

6. **Time recommendations must be practical**
   - Don't suggest 3 AM for shopping
   - Don't suggest rush hour unless unavoidable
   - Consider the destination type (mall, office, airport, etc.)

7. **Use current Bangkok time context when relevant**
   - You will be given the current Bangkok local date and time in the prompt.
   - Use it naturally for questions about safe driving times, peak hours, and weekend risk.
   - Do not announce the current time unless the user asks or it materially changes the advice.

8. **Reference data naturally**
   - Bad: "Our NLP analysis of 2,669 Thai accident reports shows..."
   - Good: "Based on accident patterns..."
   - Only mention "Thai NLP" if relevant to the question

EXAMPLES:
Q: "When should I drive to Central World?"
A: "**Best time: 10 AM - 12 PM.** Roads clear after morning rush, and you avoid the dangerous **5-7 PM** window."

Q: "Is Sukhumvit safe at night?"
A: "**Sukhumvit is higher risk at night.** Risk score: **65/100**. Main issues are **speeding (42%)** and **red-light violations (28%)**. If you must drive, stay on main roads and avoid **10 PM - 2 AM**."

Q: "Which areas have speeding accidents?"
A: "Top speeding zones:
- **Asok Intersection** — **47 accidents**
- **Lat Krabang Industrial** — **38 accidents**
- **Klong Toey Port Road** — **29 accidents**
**Speeding** is the main issue in Bangkok, driving **78%** of accidents."

Q: "What time are accidents most common?"
A: "**Most dangerous: 6-7 PM.** That is the peak evening rush window. For normal travel, **10 AM - 3 PM** is the safest practical window."

Remember: you're a knowledgeable assistant, not a chatbot and not a research paper. Be brief, practical, and structured.
"""


def _current_bangkok_datetime_context():
    """Return the current Bangkok-local date/time guidance for the prompt."""
    now = datetime.now(BANGKOK_TZ)
    day_type = "weekend" if now.weekday() >= 5 else "weekday"
    return (
        "Current Bangkok local date/time context:\n"
        f"- Date: {now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.year}\n"
        f"- Time: {now.strftime('%H:%M')} ICT\n"
        f"- Day type: {day_type}\n"
        "- Use this context only when it helps answer the user's question."
    )


def prepare_context(user_message, chatbot_data):
    """
    Prepare relevant context from chatbot_data based on user message
    """
    context_parts = []

    # Always include basic stats
    context_parts.append(f"Total accidents: {chatbot_data['stats']['total_accidents']}")

    # If question mentions causes
    if any(word in user_message.lower() for word in ['cause', 'why', 'speeding', 'drunk', 'cutting']):
        context_parts.append("Causes: " + json.dumps(chatbot_data['causes'], indent=2))

    # If question mentions location/area/zone
    if any(word in user_message.lower() for word in ['where', 'area', 'zone', 'location', 'asok', 'sukhumvit']):
        top_zones = list(chatbot_data['zones'].items())[:10]
        context_parts.append("Top zones: " + json.dumps(dict(top_zones), indent=2))

    # If question mentions time
    if any(word in user_message.lower() for word in ['when', 'time', 'hour', 'day', 'rush']):
        context_parts.append("Time patterns: " + json.dumps(chatbot_data['time_patterns'], indent=2))

    # If question mentions recommendation
    if any(word in user_message.lower() for word in ['recommend', 'should', 'improve', 'fix', 'solution']):
        context_parts.append("Recommendations: " + json.dumps(chatbot_data['recommendations'], indent=2))

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Pre-cached responses for common demo questions
# These fire before any API call, guaranteeing instant reliable answers.
# ---------------------------------------------------------------------------
_DEMO_CACHE = {
    # cause questions
    "most dangerous zone":       "The most dangerous zone is **Lat Krabang / Motorway (Eastern)** with 156 accidents — 17× the city average. Speeding accounts for 78.6% of incidents there. Worth avoiding during rush hour (5–7 PM). Stay safe.",
    "safest time to drive":      "Safest window: **10 AM – 3 PM** (between rush hours). Most dangerous: **6–7 PM** (evening rush). For weekend driving, avoid 10 PM – 2 AM when risk spikes due to drunk driving. Plan accordingly.",
    "main cause":                "The #1 cause is **speeding — 78.8% of all Bangkok accidents** (2,104 incidents). Cutting in front is #2. Our ML model uses these NLP-extracted cause patterns as features to predict future danger zones.",
    "how accurate":              "Our XGBoost model detects **39 out of 39 danger zones** (100% recall) with a **17.5× lift** over random selection and **83.8% temporal capture rate** — validated on Oct–Dec 2024 holdout data.",
    "weekend":                   "Weekends see ~23% fewer accidents overall, but Saturday is actually the **deadliest single day**. Late-night hours (10 PM – 2 AM Fri/Sat) are the highest-risk window due to drunk driving. Drive extra carefully on weekend nights.",
    "nlp":                       "Our Thai NLP pipeline uses **pythainlp tokenization + TF-IDF + Logistic Regression** to classify accident report text into 6 cause categories with 99.5% cross-validated accuracy. These NLP features feed directly into the XGBoost risk model.",
    "xgboost":                   "XGBoost gradient boosting trains on **44 engineered features** per 1 km² grid cell — severity, spatial neighbors, temporal patterns, vehicle mix, and NLP cause categories. Key insight: severity_mean (54% importance) outweighs raw accident count.",
}

_TIME_SENSITIVE_CACHE_HINTS = (
    "safe time",
    "safest time",
    "best time",
    "when should i drive",
    "when to drive",
    "rush hour",
    "peak hour",
    "peak hours",
    "weekend",
    "today",
    "tonight",
    "right now",
    "this morning",
    "this evening",
)

def _check_demo_cache(user_message):
    """Return a cached response if the message matches a demo keyword."""
    msg = user_message.lower()
    if any(hint in msg for hint in _TIME_SENSITIVE_CACHE_HINTS):
        return None
    for keyword, response in _DEMO_CACHE.items():
        if keyword in msg:
            return {"type": "ai", "message": response, "source": "cached"}
    return None


def get_groq_response(user_message, chatbot_data):
    """
    Get AI response from Groq with pre-cached fallback and timeout handling.

    Priority:
      1. Pre-cached demo responses (instant, no API needed)
      2. Groq LLM call (10s timeout)
      3. Friendly fallback if API unavailable or times out
    """
    # 1. Check pre-cached responses first
    cached = _check_demo_cache(user_message)
    if cached:
        return cached

    # 2. Try Groq API
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "gsk_your_key_here":
            return _groq_fallback("no_key")

        client = Groq(api_key=api_key, timeout=10.0)
        context = prepare_context(user_message, chatbot_data)
        bangkok_now = _current_bangkok_datetime_context()

        full_prompt = f"""{SYSTEM_PROMPT}

{bangkok_now}

Available data:
{context}

User question: {user_message}

IMPORTANT:
- Answer in 2-4 sentences maximum for simple questions
- Start with the answer and skip filler openings
- Be direct and practical
- Use short paragraphs or simple bullets when needed
- Bold key terms and every number/statistic using markdown
- No emojis
- Consider what the user is trying to DO, not just asking
- If they mention a specific place, consider what type of place it is
- Sound concise, knowledgeable, and data-backed

Provide your answer:"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": full_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=300,
            top_p=0.9,
        )

        return {
            "type": "ai",
            "message": response.choices[0].message.content,
            "source": "groq",
        }

    except Exception as e:
        err = str(e).lower()
        if "timeout" in err or "timed out" in err:
            return _groq_fallback("timeout")
        if "401" in err or "api_key" in err or "authentication" in err:
            return _groq_fallback("no_key")
        return _groq_fallback("error")


def _groq_fallback(reason):
    """Friendly fallback when Groq is unavailable — guides users to what works."""
    messages = {
        "no_key": (
            "AI mode is unavailable right now.\n\n"
            "Try:\n"
            "• **Dangerous zones** — 'Which areas have the most accidents?'\n"
            "• **Accident causes** — 'What causes Bangkok accidents?'\n"
            "• **Safe times** — 'When is rush hour dangerous?'\n"
            "• **Route planning** — 'What's the safest route?'\n\n"
            "Use the **Trip Planner** for a full route analysis."
        ),
        "timeout": (
            "The assistant is responding slowly right now.\n\n"
            "Try a specific question like **'Is Asok safe?'** or **'When are accidents most common?'** for a faster answer."
        ),
        "error": (
            "The assistant hit an error, but the data tools are still available.\n\n"
            "Ask about **dangerous zones**, **accident causes**, **safe times**, or **route safety**."
        ),
    }
    return {
        "type": "ai",
        "message": messages.get(reason, messages["error"]),
        "source": "fallback",
    }
