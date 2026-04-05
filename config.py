# config.py
import os
from pathlib import Path

# --- LLM Settings ---
LLM_PROVIDER = "ollama"  # "ollama" or "gemini"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# --- ARIA Settings ---
ARIA_NAME = "ARIA"
USER_NAME = "Prithish"

MAX_HISTORY_TURNS = 20  # how many conversation turns to keep in memory

# --- Paths ---
BASE_DIR = Path(__file__).parent
MEMORY_DIR = BASE_DIR / "memory" / "store"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)