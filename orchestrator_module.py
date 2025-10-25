import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# ================================
# 1. Load Embedding Model
# ================================
print("🚀 Loading embedding model...")
embedder = SentenceTransformer("all-mpnet-base-v2")

# ================================
# 2. Initialize Chroma Vector DB
# ================================
print("🧠 Initializing Chroma DB...")
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection("policy_docs")

# ================================
# 3. Helper Function to Extract Text
# ================================
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
    return text

# ================================
# 4. Ingest Documents from ./docs
# ================================
DOCS_DIR = "./docs"

def ingest_documents():
    """Loop through docs folder and add new documents to Chroma."""
    existing_ids = set(collection.get()["ids"])
    print(f"📂 Scanning folder: {DOCS_DIR}")

    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        file_id = filename  # can use hash for uniqueness if needed

        if file_id in existing_ids:
            print(f"⏭️ Skipping already ingested: {filename}")
            continue

        if filename.lower().endswith(".pdf"):
            print(f"📄 Ingesting: {filename}")
            text = extract_text_from_pdf(file_path)

            if text.strip():
                embedding = embedder.encode([text])[0].tolist()
                collection.add(
                    documents=[text],
                    embeddings=[embedding],
                    ids=[file_id]
                )
                print(f"✅ Ingested: {filename}")
            else:
                print(f"⚠️ No text found in: {filename}")

    print("✅ Ingestion complete.")

# Call ingestion once at startup
ingest_documents()

# ================================
# 5. Orchestrate Query
# ================================
def orchestrate(query: str) -> str:
    """Perform semantic search on ingested policy docs and return top answer."""
    if not query or not query.strip():
        return "⚠️ Please enter a valid question."

    # Create embedding for query
    query_vector = embedder.encode([query])[0].tolist()

    # Search top matching chunks
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    if not results["documents"] or not results["documents"][0]:
        return "❌ No relevant information found in the ingested documents."

    # Combine top matches into a single answer
    top_docs = results["documents"][0]
    answer = "\n\n".join(top_docs)

    return answer
