import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

# =====================================
# INIT
# =====================================
load_dotenv()

st.set_page_config(page_title="Market Analyst RAG", layout="wide")
st.title("📊 Market Analyst RAG Agent")

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = None

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)

# Try loading saved FAISS
if os.path.exists("faiss_index"):
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings)
        st.sidebar.success("Loaded existing FAISS index.")
    except:
        st.sidebar.error("Failed to load FAISS index.")


# =====================================
# HELPERS
# =====================================
def classify_query(q: str) -> str:
    classify_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
Classify this query into ONE type: general_qa, market_summary, or structured_data

Question: {question}

Return ONLY the type word.
"""
    )
    classify_chain = classify_prompt | llm | StrOutputParser()
    resp = classify_chain.invoke({"question": q}).strip().lower()

    if "structured" in resp or any(x in q.lower() for x in ["cap", "cagr", "competitors", "share"]):
        return "structured_data"
    elif "summary" in resp or "overview" in resp:
        return "market_summary"
    return "general_qa"


def handle_qa(q: str, ctx: list) -> dict:
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Based on this context, answer the question clearly.
If information is unavailable, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    )
    qa_chain = qa_prompt | llm | StrOutputParser()
    ans = qa_chain.invoke({"context": "\n".join(ctx), "question": q})
    return {"answer": ans, "sources": ctx[:2]}


def handle_summary(q: str, ctx: list) -> dict:
    summary_prompt = PromptTemplate(
        input_variables=["context"],
        template="""
From this context, create a market summary.

Return ONLY JSON:
{"summary": "", "key_insights": []}

CONTEXT:
{context}

JSON:
"""
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    resp = summary_chain.invoke({"context": "\n".join(ctx)})

    try:
        return json.loads(resp)
    except:
        return {"summary": resp[:200], "key_insights": []}


def handle_structured(q: str, ctx: list) -> dict:
    structured_prompt = PromptTemplate(
        input_variables=["context"],
        template="""
Extract structured market data.

Return STRICT valid JSON:
{
  "company": "",
  "market_size": "",
  "cagr": "",
  "market_share": "",
  "competitors": []
}

CONTEXT:
{context}

JSON:
"""
    )
    structured_chain = structured_prompt | llm | StrOutputParser()
    resp = structured_chain.invoke({"context": "\n".join(ctx)})
    resp = resp.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(resp)
    except:
        return {"company": "", "market_size": "", "cagr": "", "market_share": "", "competitors": []}


# =====================================
# STREAMLIT UI
# =====================================

st.sidebar.header("📁 Upload Documents")

uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    file_path = uploads_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader(str(file_path)) if uploaded_file.name.endswith(".pdf") else TextLoader(str(file_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)

    vector_store.save_local("faiss_index")
    st.sidebar.success(f"Indexed {len(chunks)} chunks.")


st.subheader("🧠 Ask a Question")

question = st.text_input("Your question")

if st.button("Ask"):
    if vector_store is None:
        st.error("Upload a document first.")
    else:
        docs = vector_store.similarity_search(question, k=5)
        ctx = [d.page_content for d in docs]

        qtype = classify_query(question)

        if qtype == "general_qa":
            resp = handle_qa(question, ctx)
        elif qtype == "market_summary":
            resp = handle_summary(question, ctx)
        else:
            resp = handle_structured(question, ctx)

        st.write(f"### Query Type: `{qtype}`")
        st.json(resp)
        st.write("### Top Sources:")
        for i, c in enumerate(ctx[:2]):
            st.write(f"**Source {i+1}:** {c[:300]}...")
