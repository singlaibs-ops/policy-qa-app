import streamlit as st
from orchestrator_module import orchestrate  # we'll fill this later

# --- Page Setup ---
st.set_page_config(page_title="Policy Q&A Assistant", page_icon="📜")
st.title("📜 Policy Q&A Assistant")
st.write("Ask questions directly from ingested policy documents (powered by RAG).")

# --- Optional Password ---
PASSWORD = "demo123"
password = st.text_input("Enter access password", type="password")
if password != PASSWORD:
    st.warning("Please enter the correct password to continue.")
    st.stop()

# --- Query Box ---
query = st.text_input("Enter your question", placeholder="e.g., What is the policy on booking flights?")

# --- Orchestrate the Response ---
if st.button("Submit") and query:
    with st.spinner("Processing your question..."):
        result = orchestrate(query)

    st.subheader("🧭 Agent Used")
    st.write(result.get("agent_used", "N/A"))

    st.subheader("📄 Answer")
    st.write(result.get("answer", "No answer returned."))

    st.subheader("📚 Sources")
    for s in result.get("sources", []):
        st.write(f"- {s}")
