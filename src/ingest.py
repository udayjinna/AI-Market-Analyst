import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def build_vector_store():
    # Load text file
    loader = TextLoader("data.txt")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]

    print(f"📄 Total chunks: {len(texts)}")

    # Use the correct LangChain embedding class
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Build FAISS vectorstore
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local("faiss_store")

    print("✅ FAISS vector store created successfully using SentenceTransformer embeddings!")

if __name__ == "__main__":
    build_vector_store()
