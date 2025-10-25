import streamlit as st
from orchestrator_module import orchestrate  # we'll fill this later

import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Embedding model and Chroma DB setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection("policy_docs")

# --- Debug/Status: show vector DB count in the UI ---
try:
    doc_count = collection.count()
except Exception:
    doc_count = "unknown"

with st.sidebar:
    st.markdown("### 🧰 Vector DB status")
    st.write(f"Records stored: {doc_count}")
    st.caption("Upload PDFs/TXT on this sidebar to add records.")

def ingest_document(file):
    text = ""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        text = file.read().decode("utf-8", errors="ignore")

    # Simple chunking into 1000-char segments
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    embeddings = embedder.encode(chunks).tolist()

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            metadatas=[{"source": file.name}],
            ids=[f"{file.name}-{i}"]
        )

# 📥 Sidebar for uploading
st.sidebar.subheader("📥 Upload Policy Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF or TXT file")
if uploaded_file is not None:
    ingest_document(uploaded_file)
    st.sidebar.success(f"✅ Document '{uploaded_file.name}' ingested successfully!")

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



