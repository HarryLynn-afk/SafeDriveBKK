import os
from groq import Groq
import json

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are a Bangkok road safety assistant powered by NLP analysis of Thai accident reports.

Your knowledge base:
- 2,669 Thai accident reports analyzed using pythainlp
- 276 grid zones across Bangkok
- Accident causes extracted from Thai text
- Risk scores from XGBoost ML model

Guidelines:
1. ONLY use data provided in context - never make up facts
2. Be conversational and helpful
3. Keep responses under 200 words
4. Mention "Thai NLP analysis" when relevant
5. Include specific numbers/locations from data
6. End with a helpful follow-up question
7. Format lists with bullet points or numbers
8. Never say you don't have real-time data - you have analyzed historical data

If asked about:
- Locations: Provide specific zone names and risk scores
- Causes: Reference Thai keywords and percentages
- Times: Give hour/day patterns
- Recommendations: Cite evidence-based interventions
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
        context = prepare_context(user_message, chatbot_data)

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Data context:\n{context}\n\n"
            f"User question: {user_message}\n\n"
            f"Provide a helpful, conversational answer using ONLY the data provided above."
        )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            max_tokens=500,
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
            "message": "I'm having trouble connecting right now. Could you rephrase your question or try one of the suggested questions below?",
            "error": str(e)
        }
