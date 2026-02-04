import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"

def qwen_rewrite_only(user_question: str, grounded_answer: str, lang: str) -> str:
    """
    Rewrite the grounded_answer conversationally WITHOUT adding new facts.
    lang: "si" or "ta"
    """
    lang_name = "Sinhala" if lang == "si" else "Tamil"

    prompt = f"""
You are Coco-Guide, an agricultural advisory assistant for coconut farmers in Sri Lanka.

STRICT RULES:
1) Use ONLY the information in "Grounded Answer".
2) Do NOT add new facts, steps, pesticide names, fertilizer amounts, or any advice not present in Grounded Answer.
3) If Grounded Answer is not sufficient, ask ONE clarification question.
4) Respond ONLY in {lang_name}. No English.
5) Keep it short and friendly (3–6 sentences).

User Question:
{user_question}

Grounded Answer:
{grounded_answer}

Now rewrite the Grounded Answer in a conversational way (same meaning, same facts):
""".strip()

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.8
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json().get("response", "").strip()
