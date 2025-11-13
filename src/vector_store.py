from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

def load_vector_store(path: str = "faiss_store"):
    """Load the FAISS store using local SentenceTransformer embeddings."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, model, allow_dangerous_deserialization=True)

def save_vector_store(docs, path: str = "faiss_store"):
    """Save document chunks into FAISS with SentenceTransformer embeddings."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [d.page_content for d in docs]
    embeddings = [model.encode(t) for t in texts]
    vectorstore = FAISS.from_embeddings(embeddings, texts)
    vectorstore.save_local(path)
    print(f"✅ FAISS store saved at: {path}")
