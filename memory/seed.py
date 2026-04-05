# memory/seed.py
# Run this ONCE on first boot to load your life into ChromaDB
from memory.rag import store

def seed_memory():
    print("[ARIA] Seeding memory database...")

    # === IDENTITY ===
    identity = [
        "User's name is Prithish.",
        "Prithish is approximately 19-20 years old.",
        "Prithish lives in Chennai, in the Medavakkam and Perumbakkam area.",
        "Prithish is currently finishing his 2nd year of B.Tech in Artificial Intelligence and Data Science at Chennai Institute of Technology.",
        "Prithish is graduating in 2028.",
        "Prithish's current CGPA is 8.09.",
    ]

    # === SCHEDULE ===
    schedule = [
        "Prithish's college schedule runs from 6 AM to 5 PM daily.",
        "Prithish does home workouts using 5kg and 7.5kg dumbbells.",
        "Prithish plays Valorant at Gold 3 rank.",
        "Prithish plays football management simulation games, currently managing Notts County (2034/35 season) and FC Barcelona (2030 season).",
    ]

    # === GOALS ===
    goals = [
        "Prithish's immediate goal is to land an AI/ML or Agentic AI internship at a startup or research lab.",
        "Prithish's long term career goal is to become a Software Engineer at a high frequency trading or quantitative finance firm.",
        "Prithish is actively building a technical portfolio focused on GenAI, agentic systems, and backend engineering.",
        "Prithish is targeting internships on Internshala and startup-focused platforms.",
        "Prithish applied for a Glacis Agentic AI Engineering internship with ARIA listed as a resume project.",
        "Prithish is building skills in FastAPI, Go, system design, and HFT-relevant technologies year by year.",
    ]

    # === PROJECTS ===
    projects = [
        "Prithish is building ARIA — an Autonomous Research and Inbox Assistant. It is a multi-agent personal assistant with an Orchestrator, Research, Email, and Calendar agent. Built with raw API calls, FastAPI, React, and ChromaDB.",
        "Prithish built FlowML — a no-code ML pipeline builder with DAG execution engine. Built with React, Node.js, and FastAPI.",
        "Prithish built a Transformer-based Web Application Firewall using DistilBERT fine-tuned for adversaive HTTP payload detection, deployed as a FastAPI reverse proxy. Achieved 96% F1 score.",
        "Prithish built an AI Resume Screener using the Google Gemini API.",
    ]

    # === SKILLS ===
    skills = [
        "Prithish's primary programming language is Python.",
        "Prithish is proficient in FastAPI, React, ChromaDB, and REST API design.",
        "Prithish has a LeetCode rating of approximately 1703 with over 450 problems solved.",
        "Prithish is a hackathon finalist at IIT Madras and IIT Jammu.",
        "Prithish is learning Go for systems and HFT-relevant backend development.",
        "Prithish is reading Designing Data-Intensive Applications for system design fundamentals.",
        "Prithish's hardware is a Lenovo LOQ with i5-12450HX, RTX 3050 6GB, and 16GB RAM.",
        "Prithish's GitHub is github.com/prithish47.",
    ]

    # === PREFERENCES ===
    preferences = [
        "Prithish prefers direct, no-sugarcoating communication.",
        "Prithish prefers plain text answers without unnecessary formatting.",
        "Prithish communicates informally, often using the word bro.",
        "Prithish likes being addressed as sir in a warm and slightly humorous tone.",
        "Prithish prefers building real deployed systems over theoretical projects.",
    ]

    # === ACADEMIC INTERESTS ===
    academic = [
        "Prithish is deeply interested in GenAI, agentic systems, and LLM architecture.",
        "Prithish is interested in IoT, Edge Computing, TinyML, TFLite, and ONNX Runtime.",
        "Prithish is interested in RAG pipelines and vector database systems.",
        "Prithish is interested in high frequency trading, quantitative finance, and low latency systems.",
        "Prithish studies competitive programming on LeetCode and CodeChef alongside his college work.",
    ]

    all_facts = {
        "profile": identity + schedule + preferences + academic,
        "notes": goals,
        "research": projects + skills,
    }

    total = 0
    for collection, facts in all_facts.items():
        for fact in facts:
            store(fact, collection=collection)
            total += 1

    print(f"[ARIA] Memory seeded. {total} facts loaded into ChromaDB.")
    print("[ARIA] ARIA now knows you. Don't run this again.")


if __name__ == "__main__":
    seed_memory()