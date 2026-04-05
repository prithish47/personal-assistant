# agents/base.py
from llm.ollama import chat_stream
from memory.context import build_system_prompt

class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def run_stream(self, user_message: str, conversation_history: list[dict]):
        """Yields tokens one by one for live streaming output."""
        system_prompt = build_system_prompt(conversation_history, user_message)
        messages = [{"role": "user", "content": user_message}]
        yield from chat_stream(messages, system=system_prompt)