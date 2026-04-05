# llm/gemini.py
import requests
from config import GEMINI_API_KEY, GEMINI_MODEL

def chat(messages: list[dict], system: str = "") -> str:
    if not GEMINI_API_KEY:
        return "[ERROR] GEMINI_API_KEY not set"

    # convert to Gemini format
    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": f"[SYSTEM]\n{system}"}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})

    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048}
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[ERROR] {str(e)}"