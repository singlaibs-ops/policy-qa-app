# orchestrator_module.py
import os
from typing import List, Dict

# ----------------------------
# 📦 Core Libraries
# ----------------------------
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings

# ----------------------------
# ⚙️ Configuration
# ----------------------------
EMBED_MODEL_NAME = "all-mpnet-base-v2"   # Same model as Colab
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

DOCS_DIR = "docs"                       # Folder in GitHub repo
PERSIST_DIR = ".vector_db"              # Local folder to store index
COLLECTION_NAME = "policy_docs"

# Global variables
_embedder = None
_chroma_client = None
_collection = None

# ----------------------------
# 🔹 Helper Functions
# ----------------------------
def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(allow_reset=False)
        )
    return _chroma_client

def _get_collection():
    global _collection
    if _collection is None:
        client = _get_chroma_client()
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            _collection = client.create_collection(COLLECTION_NAME)
    return _collection

def _split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return splitter.split_text(text)

def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_embedder()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

def _already_ingested(filename: str, collection) -> bool:
    try:
        res = collection.get(where={"source": filename})
        return len(res.get("ids", [])) > 0
    except Exception:
        return False

def _add_to_collection(chunks: List[str], source_name: str, collection):
    if not chunks:
        return
    ids = [f"{source_name}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name} for _ in chunks]
    embeddings = _embed_texts(chunks)
    collection.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)

def _ingest_file(path: str, collection):
    filename = os.path.basename(path)
    if _already_ingested(filename, collection):
        return  # Skip duplicates

    text = ""
    if filename.lower().endswith(".pdf"):
        text = _read_pdf(path)
    elif filename.lower().endswith(".txt"):
        text = _read_txt(path)

    if text.strip():
        chunks = _split_text(text)
        _add_to_collection(chunks, filename, collection)

# ----------------------------
# 🧠 Vector DB Load or Build
# ----------------------------
def load_or_build_vector_store():
    collection = _get_collection()
    # Check if already has data
    try:
        probe = collection.peek()
        has_data = probe and len(probe.get("ids", [])) > 0
    except Exception:
        has_data = False

    if not has_data and os.path.isdir(DOCS_DIR):
        print("🚀 Building vector store from docs/ ...")
        for name in os.listdir(DOCS_DIR):
            if name.lower().endswith((".pdf", ".txt")):
                _ingest_file(os.path.join(DOCS_DIR, name), collection)
        print("✅ Ingestion complete!")
    else:
        print("📂 Vector store already loaded.")

# ----------------------------
# 🔍 Query
# ----------------------------
def query_vector_store(question: str, k: int = 5) -> Dict:
    collection = _get_collection()
    qvec = _get_embedder().encode([question], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    return {
        "chunks": res.get("documents", [[]])[0],
        "metadatas": res.get("metadatas", [[]])[0],
        "distances": res.get("distances", [[]])[0]
    }

# ----------------------------
# 🧠 Orchestrator (used in Streamlit)
# ----------------------------
def orchestrate(user_query: str) -> Dict:
    retrieved = query_vector_store(user_query, k=5)
    chunks = retrieved["chunks"]

    if not chunks:
        answer = "No relevant information found in the ingested documents."
    else:
        # For now, simply return the top chunk(s) — later you can plug in GPT or other LLM here.
        answer = chunks[0]

    return {
        "agent_used": "Q&A Agent",
        "chunks_used": chunks,
        "answer": answer
    }
