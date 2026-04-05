# llm/__init__.py
from config import LLM_PROVIDER

if LLM_PROVIDER == "ollama":
    from llm.ollama import chat
elif LLM_PROVIDER == "gemini":
    from llm.gemini import chat
else:
    raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")