# memory/context.py
from datetime import datetime
from memory.rag import search_all
from config import USER_NAME, ARIA_NAME, MAX_HISTORY_TURNS


def build_system_prompt(
    conversation_history: list[dict],
    current_user_message: str
) -> str:
    """
    Full Jarvis brain assembly.
    Searches ChromaDB for relevant memories and injects into every LLM call.
    """

    now = datetime.now()
    datetime_str = now.strftime("%A, %d %B %Y — %I:%M %p")

    # RAG — search all collections with current message as query
    relevant_memories = search_all(current_user_message, top_k=4)

    # format memories block
    memory_block = ""
    for collection, docs in relevant_memories.items():
        if docs:
            memory_block += f"\n[{collection.upper()}]\n"
            for doc in docs:
                memory_block += f"  • {doc}\n"

    # trim history to last N turns
    recent_history = conversation_history[-MAX_HISTORY_TURNS:]
    history_text = ""
    for msg in recent_history:
        role = USER_NAME if msg["role"] == "user" else ARIA_NAME
        history_text += f"{role}: {msg['content']}\n"

    system_prompt = f"""You are {ARIA_NAME} — a highly intelligent personal assistant for {USER_NAME}.
You are sharp, direct, and always aware. You respond like Jarvis responds to Tony Stark.
Never say "As an AI". Never say "I don't have feelings". Just respond — clean, confident, useful.
You have a persistent memory database. Everything relevant has already been pulled and given to you below.
Use it naturally without saying "according to my memory" — just know it.

You were built by Prithish S

=== CURRENT TIME ===
{datetime_str}

=== RELEVANT MEMORY (pulled from your database) ===
{memory_block if memory_block else "No specific memories retrieved. Respond based on conversation."}

=== CONVERSATION ===
{history_text if history_text else "Conversation just started."}

Respond directly to {USER_NAME}'s latest message now.
If {USER_NAME} shares something important, it will be stored automatically after this response.
"""
    return system_prompt