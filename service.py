"""
claude_service.py — BankIQ
Using Groq FREE API (llama model)
Install: pip install groq
"""

import json
from groq import Groq

# ── Paste your Groq API key here ──
client = Groq(api_key="enter your api key")  # Replace with your key

PROFILES = {
    "Student":      "Explain using school analogies. Simple words only. No jargon.",
    "Professional": "Use proper financial terms. Be concise and data-driven.",
    "Elder":        "Very simple language. Use daily life examples. Be warm and friendly."
}

LANGS = {
    "Tamil":   "Respond entirely in Tamil language.",
    "Hindi":   "Respond entirely in Hindi language.",
    "Telugu":  "Respond entirely in Telugu language.",
    "English": "Respond in English."
}

# ══════════════════════════════════════
# MAIN CHAT — Ask Banking Question
# ══════════════════════════════════════
async def ask_claude(question, profile, language, topic, history=[]):
    system = f"""{PROFILES.get(profile, PROFILES['Student'])}
Topic: {topic}. {LANGS.get(language, 'Respond in English.')}

Reply ONLY as valid JSON — no extra text:
{{
  "answer": "2-3 sentence direct answer",
  "confidence": "high/medium/low",
  "confidence_reason": "one sentence why",
  "explanation_points": ["point1", "point2", "point3"],
  "analogy": "simple analogy matching the profile",
  "follow_up_questions": ["question1?", "question2?"]
}}"""

    messages = [{"role": "system", "content": system}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )

    text = response.choices[0].message.content
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ══════════════════════════════════════
# SHAP → Plain Language via Groq
# ══════════════════════════════════════
async def shap_to_plain(shap_dict, prediction, profile, language, model_type):
    top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    summary = ", ".join([
        f"{f} ({'helped' if v > 0 else 'hurt'}, {abs(v):.2f})"
        for f, v in top
    ])

    verdict = {
        "loan":  "approved" if prediction else "rejected",
        "churn": "likely to leave bank" if prediction else "likely to stay"
    }.get(model_type, "decided")

    lang = LANGS.get(language, "Respond in English.")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "system",
            "content": f"{PROFILES.get(profile, PROFILES['Student'])} {lang}"
        }, {
            "role": "user",
            "content": f"""
AI decision: {verdict}
Top SHAP factors: {summary}
Explain this to a {profile} in plain simple language.
Reply ONLY as valid JSON:
{{
  "explanation_points": ["plain point 1", "plain point 2", "plain point 3"],
  "analogy": "simple everyday analogy",
  "transparency_score": 85
}}"""
        }],
        temperature=0.5,
        max_tokens=500
    )

    text = response.choices[0].message.content
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ══════════════════════════════════════
# SIMPLIFY — Make Explanation Simpler
# ══════════════════════════════════════
async def simplify(prev_answer, profile, language):
    lang = LANGS.get(language, "Respond in English.")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "system",
            "content": f"{PROFILES.get(profile, PROFILES['Student'])} {lang}"
        }, {
            "role": "user",
            "content": f"""
Make this even simpler for a {profile}: "{prev_answer}"
Use one very simple sentence and a daily life analogy.
Reply ONLY as valid JSON:
{{
  "explanation_points": ["simple one sentence explanation"],
  "analogy": "very simple daily life analogy"
}}"""
        }],
        temperature=0.5,
        max_tokens=300
    )

    text = response.choices[0].message.content
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)
