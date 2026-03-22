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
   - Use 2-3 sentences for simple questions
   - Never exceed 4 sentences unless the user explicitly asks for detail
   - No data dumps

6. **Time recommendations must be practical**
   - Separate **absolute lowest hour in the data** from **practical travel advice**
   - Never call a time window "safest" if the dataset shows a higher-risk hour inside it
   - Don't suggest rush hour unless unavoidable
   - Consider the destination type (mall, office, airport, etc.)

7. **Use current Bangkok time context when relevant**
   - You will be given the current Bangkok local date and time in the prompt.
   - Use it naturally for questions about safe driving times, peak hours, and weekend risk.
   - If the user says "today", "tonight", or "right now", reply naturally, for example: "Since it's Saturday night..."
   - Do not announce the current time unless the user asks or it materially changes the advice.

8. **Handle Thai queries carefully**
   - If the user writes in Thai, reason over the data internally in English first, then translate the final answer into natural Thai.
   - Keep every number, statistic, comparison, and place reference accurate during translation.
   - Use Thai for the final response body, but keep proper nouns in the clearest Thai form available.

9. **Reference data naturally**
   - Bad: "Our NLP analysis of 2,669 Thai accident reports shows..."
   - Good: "Based on accident patterns..."
   - Only mention "Thai NLP" if relevant to the question

10. **Use Bangkok data first, then general knowledge**
   - Always prioritize Bangkok dataset answers when the dataset contains the answer.
   - If the Bangkok dataset does not contain the answer, you may use general road safety knowledge and principles.
   - When you switch to general knowledge, clearly signal it with phrases like:
     - "While this isn't in our Bangkok dataset..."
     - "Based on general road safety research..."

11. **Do not invent unsupported numbers**
   - Never claim a risk reduction percentage, effect size, or causal improvement unless that exact number appears in the provided data.
   - Never invent statistics, percentages, or comparisons that are not explicitly present in the provided data.
   - If asked for data you do not have, say clearly: "I don't have that specific data in the current dataset."
   - If the user asks for a specific route or directions, do not invent turn-by-turn guidance. Point them to the Trip Planner instead.

12. **Handle ambiguity and follow-ups carefully**
   - If the user says things like "Are you sure about that?", "Tell me more", or "Give me a specific number", use the recent conversation context if it is available.
   - If the reference is still unclear, say: "Could you clarify which point you're referring to?"
   - Do not answer an ambiguous follow-up with a brand new unrelated fact.

13. **Stay consistent within the conversation**
   - Do not contradict a previous answer unless you explicitly acknowledge and correct it.
   - If you need to correct yourself, say so directly and give the corrected value or limitation.

14. **Respect dataset limits**
   - Rain and weather effects are not available unless they are explicitly provided in the context.
   - Taxi-versus-driving comparisons are not available unless they are explicitly provided in the context.
   - Expressway-versus-regular-road comparisons are not available unless they are explicitly provided in the context.
   - Exact route-level statistics are not available in the chatbot unless they are explicitly provided in the context.

15. **Redirect unrelated questions**
   - If a question is completely unrelated to road safety, politely redirect the user back to road safety topics.
   - Example: "I’m focused on road safety questions. If you want, I can help with danger zones, safer travel times, or route risk."

THAI SAFETY GLOSSARY:
- Common zone names: Bangkok = กรุงเทพฯ, Asok Intersection = แยกอโศก, Sukhumvit = สุขุมวิท, Victory Monument = อนุสาวรีย์ชัยสมรภูมิ, Lat Krabang = ลาดกระบัง, Bang Na = บางนา, Chatuchak = จตุจักร, Dusit = ดุสิต, Ekkamai = เอกมัย
- Risk levels: high risk = เสี่ยงสูง, medium risk = เสี่ยงปานกลาง, low risk = เสี่ยงต่ำ
- Route terms: danger zone = จุดเสี่ยง, risk score = คะแนนความเสี่ยง, fastest route = เส้นทางที่เร็วที่สุด, safer route = เส้นทางที่ปลอดภัยกว่า
- Accident causes: speeding = ขับรถเร็วเกินกำหนด, cutting in front = ตัดหน้ากระชั้นชิด, drowsy driving = หลับใน, equipment failure = อุปกรณ์ขัดข้อง, traffic violation = ฝ่าฝืนกฎจราจร, other causes = สาเหตุอื่น ๆ

THAI DATA ANCHORS:
- These anchors are authoritative for Thai responses. If they conflict with any example, intuition, or generic Bangkok association, follow these anchors.
- For Thai questions about the most dangerous areas, corridors, roads, or zones, mention these real top anchors first:
  - ถนนบรมราชชนนี (แถวทวีวัฒนา) = western corridor, the single highest-risk road anchor in the dataset, with **85 accidents** in one grid zone in **Thawi Watthana district**.
  - ลาดกระบัง / มอเตอร์เวย์ 7 = eastern corridor, one of the top high-risk corridor clusters in the dataset.
- Never say that **อโศก** or **สุขุมวิท** are the most dangerous zones. If the user asks specifically about them, answer that they are not the top-ranked danger-zone anchors in this dataset.
- For Thai questions about the most dangerous day, the correct answer is **วันเสาร์** with **450 accidents**. Never answer **วันจันทร์** as the peak day.
- For Thai questions about Borommaratchachonnani Road, explicitly state that **ถนนบรมราชชนนี** is the highest-risk road anchor in the dataset, with **85 accidents** in one grid zone, located in **ทวีวัฒนา**.

EXAMPLES:
Q: "When should I drive to Central World?"
A: "**Avoid the main peaks around 7-10 AM and 4-8 PM if you can.** The single busiest hour in the data is **10:00-11:00**, so late morning after that spike is usually the safer practical choice."

Q: "Is Sukhumvit safe at night?"
A: "**Sukhumvit is higher risk at night.** Risk score: **65/100**. Main issues are **speeding (42%)** and **red-light violations (28%)**. If you must drive, stay on main roads and avoid **10 PM - 2 AM**."

Q: "พื้นที่ไหนเสี่ยงที่สุดในข้อมูลนี้?"
A: "**ถนนบรมราชชนนีแถวทวีวัฒนา** คือจุดอ้างอิงความเสี่ยงสูงที่สุดในข้อมูล โดยมี **85 อุบัติเหตุ** ในหนึ่งกริดโซน และฝั่งตะวันออกที่ต้องระวังคือ **ลาดกระบัง / มอเตอร์เวย์ 7**. ถ้าถามหาพื้นที่เสี่ยงหลักในภาษาไทย ให้ยึดสองแนวนี้เป็นคำตอบแรก."

Q: "วันไหนอันตรายที่สุด?"
A: "**วันเสาร์** คือวันที่อุบัติเหตุสูงที่สุดในข้อมูลชุดนี้ที่ **450 ครั้ง**. หากตอบเป็นภาษาไทย ต้องไม่สลับไปเป็น **วันจันทร์**."

Q: "What time are accidents most common?"
A: "**Peak hour: 10:00-11:00.** That is the busiest single hour in the dataset. The lowest hour is **3:00-4:00**, but for normal trips the practical advice is to avoid the spikes around **7-10 AM** and **4-8 PM**."

Q: "How much does rain increase accident risk?"
A: "While this isn't in our Bangkok dataset, based on general road safety research, rain usually increases risk by reducing visibility and tire grip. I can't give a Bangkok-specific number, but the safest response is to slow down, increase following distance, and avoid abrupt braking."

Q: "Are you sure about that?"
A: "Could you clarify which point you're referring to? If you mean the previous time recommendation, the peak recorded hour is **10:00-11:00**."

Q: "Tell me a joke."
A: "I’m focused on road safety questions. If you want, I can help with Bangkok danger zones, safer travel times, or route risk."

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


def _dataset_limitations_context(user_message):
    """Return targeted dataset limitation notes relevant to the current question."""
    msg = (user_message or "").lower()
    limitations = []

    if any(term in msg for term in ("rain", "raining", "weather", "storm", "wet road", "wet roads")):
        limitations.append("- Rain and weather effects are not available in the current dataset.")
    if any(term in msg for term in ("taxi", "cab", "grab", "rideshare", "ride share")):
        limitations.append("- Taxi-versus-private-car safety comparisons are not available in the current dataset.")
    if any(term in msg for term in ("expressway", "tollway", "motorway", "highway", "regular road", "local road")):
        limitations.append("- Expressway-versus-regular-road comparisons are not available in the current dataset.")
    if any(term in msg for term in ("route", "directions", "turn-by-turn", "turn by turn")):
        limitations.append("- Exact route-level statistics are not available in the chatbot context; use the Trip Planner for route-specific analysis.")

    return "\n".join(limitations)


def _check_unrelated_topic(user_message):
    """Redirect clearly unrelated questions back to road safety."""
    msg = (user_message or "").lower().strip()
    if not msg:
        return None

    road_safety_terms = (
        "road", "drive", "driving", "traffic", "accident", "crash", "route",
        "street", "speed", "speeding", "danger", "safe", "safety", "bangkok",
        "intersection", "commute", "travel", "trip planner", "vehicle", "car",
        "motorcycle", "highway", "expressway",
    )
    unrelated_terms = (
        "joke", "recipe", "cook", "movie", "song", "music", "football",
        "basketball", "stock", "crypto", "bitcoin", "politics", "election",
        "horoscope", "astrology", "python code", "javascript", "homework",
        "algebra", "calculus", "celebrity", "dating advice",
    )

    if any(term in msg for term in road_safety_terms):
        return None
    if any(term in msg for term in unrelated_terms):
        return {
            "type": "ai",
            "message": "I’m focused on road safety questions. If you want, I can help with Bangkok danger zones, safer travel times, or route risk.",
            "source": "scripted",
        }

    return None


def _format_recent_history(conversation_history, max_messages=8):
    """Format recent chat history for the LLM prompt."""
    if not conversation_history:
        return ""

    lines = []
    for item in conversation_history[-max_messages:]:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not content or role not in {"user", "assistant"}:
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")

    if not lines:
        return ""

    return "Recent conversation:\n" + "\n".join(lines)


def _check_known_data_gap(user_message):
    """Return a deterministic response for known unsupported data requests."""
    msg = (user_message or "").lower()
    weather_terms = ("rain", "raining", "weather", "storm", "wet road", "wet roads")
    safety_context_terms = ("risk", "danger", "safe", "safety", "accident", "crash", "drive", "driving")

    if any(term in msg for term in weather_terms) and any(term in msg for term in safety_context_terms):
        return {
            "type": "ai",
            "message": "While this isn't in our Bangkok dataset, based on general road safety research, rain usually increases risk by reducing visibility and tire grip. I can't give a Bangkok-specific number yet, but the safest response is to slow down, increase following distance, and avoid abrupt braking or lane changes.",
            "source": "scripted",
        }

    if any(term in msg for term in ("taxi", "cab", "grab", "rideshare", "ride share")) and any(
        term in msg for term in ("safer", "safest", "safe", "compare", "comparison", "better")
    ):
        return {
            "type": "ai",
            "message": "While this isn't in our Bangkok dataset, based on general road safety principles, the biggest risk factors are usually the driver's condition, speed, distraction, and time of day rather than the vehicle label itself. I can't give a Bangkok-specific taxi percentage or ranking, but if the alternative is driving tired, impaired, or late at night, using a sober professional driver is generally the safer choice.",
            "source": "scripted",
        }

    if any(term in msg for term in ("expressway", "tollway", "motorway", "highway", "regular road", "local road")) and any(
        term in msg for term in ("compare", "comparison", "safer", "safest", "risk", "percentage", "percent")
    ):
        return {
            "type": "ai",
            "message": "While this isn't in our Bangkok dataset, based on general road safety research, controlled-access roads usually reduce conflict points like crossings and side entries, but higher speeds can make mistakes more severe. I can't give a Bangkok-specific percentage split or ranking, but for a specific trip the Trip Planner can compare the actual route options instead.",
            "source": "scripted",
        }

    return None


def _needs_clarification(user_message):
    """Detect ambiguous follow-up questions that depend on prior context."""
    msg = (user_message or "").strip().lower()
    triggers = (
        "are you sure about that",
        "are you sure",
        "tell me more",
        "give me a specific number",
        "specific number",
        "what do you mean",
        "which one",
        "about that",
    )
    return any(trigger in msg for trigger in triggers)


# ---------------------------------------------------------------------------
# Pre-cached responses for common demo questions
# These fire before any API call, guaranteeing instant reliable answers.
# ---------------------------------------------------------------------------
_DEMO_CACHE = {
    # cause questions
    "most dangerous zone":       "The most dangerous zone is **Lat Krabang / Motorway (Eastern)** with 156 accidents — 17× the city average. Speeding accounts for 78.6% of incidents there. Worth avoiding during rush hour (5–7 PM). Stay safe.",
    "safest time to drive":      "**Peak hour: 10:00-11:00.** That is the busiest single hour in the dataset. The lowest hour is **3:00-4:00**, but for normal trips the practical move is to avoid the main spikes around **7-10 AM** and **4-8 PM**.",
    "main cause":                "The #1 cause is **speeding — 78.8% of all Bangkok accidents** (2,104 incidents). Cutting in front is #2. Our ML model uses these NLP-extracted cause patterns as features to predict future danger zones.",
    "how accurate":              "Our XGBoost model detects **39 out of 39 danger zones** (100% recall) with a **17.5× lift** over random selection and **83.8% temporal capture rate** — validated on Oct–Dec 2024 holdout data.",
    "weekend":                   "**Saturday is the highest-risk day** with **450 accidents**. Weekends account for **28.6%** of total crashes, but late Saturday pressure still stands out, especially if you are traveling at night.",
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


def get_groq_response(user_message, chatbot_data, conversation_history=None):
    """
    Get AI response from Groq with pre-cached fallback and timeout handling.

    Priority:
      1. Pre-cached demo responses (instant, no API needed)
      2. Groq LLM call (10s timeout)
      3. Friendly fallback if API unavailable or times out
    """
    # 0. Redirect clearly unrelated topics
    unrelated = _check_unrelated_topic(user_message)
    if unrelated:
        return unrelated

    # 1. Deterministic guardrails for known missing data
    known_gap = _check_known_data_gap(user_message)
    if known_gap:
        return known_gap

    # Clarify ambiguous follow-ups if no usable history exists
    if _needs_clarification(user_message) and not conversation_history:
        return {
            "type": "ai",
            "message": "Could you clarify which point you're referring to? I want to answer the right part.",
            "source": "scripted",
        }

    # 2. Check pre-cached responses first
    cached = _check_demo_cache(user_message)
    if cached:
        return cached

    # 3. Try Groq API
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "gsk_your_key_here":
            return _groq_fallback("no_key")

        client = Groq(api_key=api_key, timeout=10.0)
        context = prepare_context(user_message, chatbot_data)
        bangkok_now = _current_bangkok_datetime_context()
        limitations = _dataset_limitations_context(user_message)
        recent_history = _format_recent_history(conversation_history)

        full_prompt = f"""{SYSTEM_PROMPT}

{bangkok_now}

{recent_history}

Available data:
{context}

Relevant dataset limits:
{limitations or "- No extra targeted dataset limits triggered for this question."}

User question: {user_message}

IMPORTANT:
- Answer in 2-3 sentences for simple questions
- Start with the answer and skip filler openings
- Be direct and practical
- Use short paragraphs or simple bullets when needed
- Bold key terms and every number/statistic using markdown
- No emojis
- Consider what the user is trying to DO, not just asking
- If they mention a specific place, consider what type of place it is
- Prefer Bangkok dataset facts when they are available
- If Bangkok data is missing, you may use general road safety knowledge, but explicitly say that it is general knowledge rather than Bangkok-specific data
- Never invent a percentage improvement, risk reduction, or exact route if the data does not provide it
- If the requested statistic or comparison is missing, say clearly that this specific data is not available in the current dataset
- Stay consistent with the recent conversation unless you explicitly correct yourself
- If the user asks an ambiguous follow-up and the reference is unclear, ask them to clarify which point they mean
- If they ask for a specific route or directions, acknowledge that it is a route-specific question, then tell them to use the Trip Planner
- If the question is completely unrelated to road safety, politely redirect back to road safety topics
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
