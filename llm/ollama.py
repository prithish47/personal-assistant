# llm/ollama.py
import requests
import json
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

def chat(messages: list[dict], system: str = "") -> str:
    """Non-streaming — kept for internal use."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0.7,
            "num_ctx": 8192,
        }
    }
    if system:
        payload["messages"] = [{"role": "system", "content": system}] + messages

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"[ERROR] {str(e)}"


def chat_stream(messages: list[dict], system: str = ""):
    """Streaming version — yields text chunks as they generate."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": -1,
        "options": {
            "temperature": 0.7,
            "num_ctx": 8192,
        }
    }
    if system:
        payload["messages"] = [{"role": "system", "content": system}] + messages

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=120
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
    except requests.exceptions.ConnectionError:
        yield "[ERROR] Ollama is not running."
    except Exception as e:
        yield f"[ERROR] {str(e)}"


def unload_model():
    """Force Ollama to unload the model from VRAM."""
    try:
        requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "keep_alive": 0},
            timeout=10
        )
        print("[ARIA] Model unloaded from memory.")
    except Exception as e:
        print(f"[ARIA] Could not unload model: {e}")