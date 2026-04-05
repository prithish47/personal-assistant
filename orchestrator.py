# orchestrator.py
from agents.general import GeneralAgent
from memory.rag import store_conversation_summary, extract_and_store_facts

agents = {
    "general": GeneralAgent(),
}

def route_stream(user_message: str, conversation_history: list[dict]):
    """Streams response tokens and returns full response at end."""
    agent = agents["general"]
    full_response = ""

    for token in agent.run_stream(user_message, conversation_history):
        full_response += token
        yield token

    # store to memory after full response assembled
    store_conversation_summary(user_message, full_response)
    extract_and_store_facts(user_message)