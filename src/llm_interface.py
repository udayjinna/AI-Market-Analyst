import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

def get_local_retriever():
    # Load local SentenceTransformer model

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_store", model, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")

def classify_intent(user_input: str):
    llm = get_llm()
    system_prompt = """
    You are an intelligent intent classifier.
    Classify the user's query as one of:
    - "qa" (for factual or analytical questions)
    - "findings" (for summaries or insights)
    - "extract" (for structured or JSON output)
    Respond with only one word.
    """
    response = llm.invoke(system_prompt + "\nUser query: " + user_input)
    intent = response.content.strip().lower()
    return intent if intent in ["qa", "findings", "extract"] else "qa"

def answer_question(query: str):
    retriever = get_local_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=get_llm(), retriever=retriever)
    return qa_chain.run(query)

def summarize_findings():
    retriever = get_local_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=get_llm(), retriever=retriever)
    prompt = """Summarize the Innovate Inc. report focusing on:
    - Market size and CAGR
    - Competitors and shares
    - SWOT highlights
    - Strategic recommendations"""
    return qa_chain.run(prompt)

def extract_structured_data():
    retriever = get_local_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=get_llm(), retriever=retriever)
    prompt = """Extract key data as JSON:
    {
      "company_name": "",
      "market_size": "",
      "CAGR": "",
      "competitors": [],
      "SWOT": {
        "strengths": [],
        "weaknesses": [],
        "opportunities": [],
        "threats": []
      }
    }"""
    return qa_chain.run(prompt)

def ai_agent(user_input: str):
    intent = classify_intent(user_input)
    if intent == "qa":
        return {"mode": "qa", "response": answer_question(user_input)}
    elif intent == "findings":
        return {"mode": "findings", "response": summarize_findings()}
    elif intent == "extract":
        return {"mode": "extract", "response": extract_structured_data()}
