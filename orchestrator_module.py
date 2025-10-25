# orchestrator_module.py

# For now, this is a simple placeholder
# Later you can plug in your retrieval + Groq model logic

def orchestrate(query: str):
    """
    Orchestrates between different agents based on query.
    Returns a dict with agent used, answer, and sources.
    """
    # Simple keyword-based routing
    if "clause" in query.lower():
        agent = "Clause Agent"
        answer = "This is a placeholder clause answer."
    elif "summary" in query.lower() or "summarize" in query.lower():
        agent = "Summary Agent"
        answer = "This is a placeholder summary."
    else:
        agent = "Q&A Agent"
        answer = "This is a placeholder direct answer."

    # Return a structured response
    return {
        "agent_used": agent,
        "answer": answer,
        "sources": ["Sample Policy.pdf", "Travel Policy.docx"]
    }
