import os
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

# ✅ Load API key from Streamlit Secrets (already added)
api_key = os.environ["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# ✅ Load embedding model and Chroma vector DB
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection("policy_docs")

def orchestrate(query: str):
    try:
        # Step 1: Encode the query to a vector
        query_vector = embedder.encode([query])[0].tolist()

        # Step 2: Retrieve top matching chunks from Chroma DB
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3,
            include=["documents", "metadatas"]
        )
        retrieved_chunks = [doc for doc in results["documents"][0]]

        if not retrieved_chunks:
            return {
                "agent_used": "Q&A Agent",
                "answer": "No relevant information found in the ingested documents.",
                "sources": []
            }

        # Step 3: Construct prompt
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""Answer the question strictly based on the policy context below.
If no relevant information is found, say so clearly.

Context:
{context}

Question:
{query}

Answer:
"""

        # Step 4: Call Groq LLM
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        answer = completion.choices[0].message.content

        # Step 5: Return the final answer with source documents
        sources = [meta.get("source", "Unknown") for meta in results["metadatas"][0]]
        return {
            "agent_used": "Q&A Agent",
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        return {
            "agent_used": "Q&A Agent",
            "answer": f"⚠️ Error: {e}",
            "sources": []
        }



