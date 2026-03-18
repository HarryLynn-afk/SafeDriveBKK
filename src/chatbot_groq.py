import os
from groq import Groq
import json

SYSTEM_PROMPT = """You are a helpful Bangkok road safety assistant. You help people make safer travel decisions.

Your knowledge:
- 2,669 Bangkok accident reports analyzed
- Accident patterns by time, location, and cause
- Thai NLP analysis for cause extraction

CRITICAL RESPONSE GUIDELINES:
1. **Be conversational, not technical**
   - Bad: "Based on analysis of 2,669 reports, 10:00-11:00 has 175 accidents"
   - Good: "Mid-morning (10-11 AM) is safer - fewer accidents than rush hour"

2. **Answer directly first, explain briefly second**
   - Bad: "Let me explain the data... [long explanation]... so 10 AM is good"
   - Good: "Best time: 10 AM. Why? Rush hour ends around 9 AM."

3. **Use practical advice, not statistics**
   - Bad: "36.8% of accidents occur during rush hour"
   - Good: "Rush hour (5-7 PM) is 3× more dangerous - avoid if possible"

4. **Consider context**
   - If user mentions shopping mall → suggest daytime hours
   - If user mentions work → consider rush hour alternatives
   - If user mentions airport → consider traffic + time

5. **Keep responses SHORT**
   - Maximum 3-4 sentences for simple questions
   - Use bullet points sparingly (only for lists of 3+ items)
   - No data dumps

6. **Be friendly and actionable**
   - End with "Safe travels!" or "Have a great trip!"
   - Give clear recommendations, not just data
   - Use emojis occasionally (🚗 🛍️ ⚠️ ✅)

7. **Time recommendations must be practical**
   - Don't suggest 3 AM for shopping
   - Don't suggest rush hour unless unavoidable
   - Consider the destination type (mall, office, airport, etc.)

8. **Reference data naturally**
   - Bad: "Our NLP analysis of 2,669 Thai accident reports shows..."
   - Good: "Based on accident patterns..."
   - Only mention "Thai NLP" if relevant to the question

EXAMPLES:
Q: "When should I drive to Central World?"
A: "Best time: **10 AM - 12 PM** (mid-morning). Roads clear after morning rush, and you'll avoid the dangerous 5-7 PM window. Safe shopping! 🛍️"

Q: "Is Sukhumvit safe at night?"
A: "⚠️ Sukhumvit has elevated risk at night (65/100). Main issues: speeding (42%) and red-light violations (28%). If you must drive, stick to main roads and avoid 10 PM - 2 AM when drunk driving increases. Stay alert!"

Q: "Which areas have speeding accidents?"
A: "Top speeding zones:
- Asok Intersection (47 accidents)
- Lat Krabang Industrial (38)
- Klong Toey Port Road (29)
These areas need speed cameras. Avoid speeding there - it's the #1 cause (78%) of Bangkok accidents."

Q: "What time are accidents most common?"
A: "Most dangerous: **6-7 PM** (evening rush hour). Safest: **2-5 AM** (but roads are empty then). For normal daytime travel, **10 AM - 3 PM** is your best window - after morning rush, before evening chaos."

Remember: You're a helpful assistant, not a research paper. Be brief, practical, and friendly.
"""


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
    "most dangerous zone":       "The most dangerous zone is **Lat Krabang / Motorway (Eastern)** with 156 accidents — 17× the city average. Speeding accounts for 78.6% of incidents there. Worth avoiding during rush hour (5–7 PM). Stay safe! ⚠️",
    "safest time to drive":      "Safest window: **10 AM – 3 PM** (between rush hours). Most dangerous: **6–7 PM** (evening rush). For weekend driving, avoid 10 PM – 2 AM when risk spikes due to drunk driving. Plan accordingly! 🚗",
    "main cause":                "The #1 cause is **speeding — 78.8% of all Bangkok accidents** (2,104 incidents). Cutting in front is #2. Our ML model uses these NLP-extracted cause patterns as features to predict future danger zones.",
    "how accurate":              "Our XGBoost model detects **39 out of 39 danger zones** (100% recall) with a **17.5× lift** over random selection and **83.8% temporal capture rate** — validated on Oct–Dec 2024 holdout data.",
    "weekend":                   "Weekends see ~23% fewer accidents overall, but Saturday is actually the **deadliest single day**. Late-night hours (10 PM – 2 AM Fri/Sat) are the highest-risk window due to drunk driving. Drive extra carefully on weekend nights! 🌙",
    "nlp":                       "Our Thai NLP pipeline uses **pythainlp tokenization + TF-IDF + Logistic Regression** to classify accident report text into 6 cause categories with 99.5% cross-validated accuracy. These NLP features feed directly into the XGBoost risk model.",
    "xgboost":                   "XGBoost gradient boosting trains on **44 engineered features** per 1 km² grid cell — severity, spatial neighbors, temporal patterns, vehicle mix, and NLP cause categories. Key insight: severity_mean (54% importance) outweighs raw accident count.",
}

def _check_demo_cache(user_message):
    """Return a cached response if the message matches a demo keyword."""
    msg = user_message.lower()
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

        full_prompt = f"""{SYSTEM_PROMPT}

Available data:
{context}

User question: {user_message}

IMPORTANT:
- Answer in 2-4 sentences maximum for simple questions
- Be direct and practical
- Consider what the user is trying to DO, not just asking
- If they mention a specific place, consider what type of place it is
- Use friendly, conversational language
- End with encouragement

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
            "My advanced AI assistant isn't configured right now, but I can still help! "
            "Try asking about:\n\n"
            "• **Dangerous zones** — 'Which areas have the most accidents?'\n"
            "• **Accident causes** — 'What causes Bangkok accidents?'\n"
            "• **Safe times** — 'When is rush hour dangerous?'\n"
            "• **Route planning** — 'What's the safest route?'\n\n"
            "Or use the Trip Planner for a full route analysis! 🚗"
        ),
        "timeout": (
            "That's taking longer than expected — the AI seems slow right now. "
            "Try a more specific question like 'Is Asok safe?' or 'When are accidents most common?' "
            "and I'll answer instantly from our data! ⚡"
        ),
        "error": (
            "I'm having a small hiccup with my AI assistant right now. "
            "I can still answer questions about dangerous zones, accident causes, safe times, and route safety — "
            "just ask away! 😊"
        ),
    }
    return {
        "type": "ai",
        "message": messages.get(reason, messages["error"]),
        "source": "fallback",
    }
