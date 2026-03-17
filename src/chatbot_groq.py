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


def get_groq_response(user_message, chatbot_data):
    """
    Get AI response from Groq
    """
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "gsk_your_key_here":
            return {
                "type": "error",
                "message": "Groq API key not configured. Please set GROQ_API_KEY in .env file.",
                "error": "missing_api_key"
            }

        client = Groq(api_key=api_key)
        context = prepare_context(user_message, chatbot_data)

        # Enhanced prompt with response formatting instructions
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
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,  # Increased from 0.3 for more natural language
            max_tokens=300,   # Reduced from 500 to enforce brevity
            top_p=0.9
        )

        return {
            "type": "ai",
            "message": response.choices[0].message.content,
            "source": "groq"
        }

    except Exception as e:
        return {
            "type": "error",
            "message": "I'm having trouble connecting right now. Could you rephrase your question?",
            "error": str(e)
        }
