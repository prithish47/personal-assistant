# memory/embeddings.py
import requests
from config import OLLAMA_BASE_URL

EMBED_MODEL = "nomic-embed-text"

def embed(text: str) -> list[float]:
    """Convert any text into a vector using Ollama locally."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"[EMBED ERROR] {e}")
        return []