# memory/rag.py
import chromadb
import uuid
from datetime import datetime
from memory.embeddings import embed
from config import MEMORY_DIR

# initialize persistent ChromaDB
client = chromadb.PersistentClient(path=str(MEMORY_DIR))

# collections — each is a different type of memory
collections = {
    "profile":       client.get_or_create_collection("user_profile"),
    "conversations": client.get_or_create_collection("conversations"),
    "notes":         client.get_or_create_collection("notes"),
    "research":      client.get_or_create_collection("research"),
    "tasks":         client.get_or_create_collection("tasks"),
}


def store(text: str, collection: str = "notes", metadata: dict = {}):
    """Store any text into ChromaDB with its embedding."""
    if collection not in collections:
        print(f"[RAG] Unknown collection: {collection}")
        return

    vector = embed(text)
    if not vector:
        return

    collections[collection].add(
        ids=[str(uuid.uuid4())],
        embeddings=[vector],
        documents=[text],
        metadatas=[{"timestamp": datetime.now().isoformat(), **metadata}]
    )


def search(query: str, collection: str = "notes", top_k: int = 5) -> list[str]:
    """Search ChromaDB for most relevant memories given a query."""
    if collection not in collections:
        return []

    vector = embed(query)
    if not vector:
        return []

    try:
        results = collections[collection].query(
            query_embeddings=[vector],
            n_results=top_k
        )
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        print(f"[RAG SEARCH ERROR] {e}")
        return []


def search_all(query: str, top_k: int = 3) -> dict[str, list[str]]:
    """Search across ALL collections at once — full brain search."""
    results = {}
    for name in collections:
        found = search(query, collection=name, top_k=top_k)
        if found:
            results[name] = found
    return results


def store_conversation_summary(user_msg: str, aria_response: str):
    """Summarize and store a conversation turn permanently."""
    summary = f"User said: {user_msg} | ARIA responded: {aria_response[:200]}"
    store(summary, collection="conversations")


def extract_and_store_facts(text: str):
    """
    Detect if the text contains a personal fact worth storing.
    Simple keyword detection — we'll make this smarter later with LLM extraction.
    """
    triggers = [
        "remember", "don't forget", "i am", "i'm", "i have",
        "i hate", "i love", "i prefer", "i always", "i never",
        "my ", "i work", "i study", "i live"
    ]
    text_lower = text.lower()
    if any(t in text_lower for t in triggers):
        store(text, collection="profile", metadata={"source": "user_statement"})
        print("[ARIA] Got it, stored to memory.")