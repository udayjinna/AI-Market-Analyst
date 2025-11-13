# 🧠 AI Market Analyst

Transform raw market research documents into actionable insights, structured data, and intelligent answers using FastAPI, FAISS vector search, Groq LLMs, and SentenceTransformer embeddings.

## ✨ Features

- Document ingestion with automatic text chunking  
- Semantic embeddings using SentenceTransformers (`all-MiniLM-L6-v2`)  
- FAISS vector store for fast similarity search  
- Three intelligent agent modes:
  - **General Q&A**
  - **Market Findings Extraction**
  - **Structured JSON Extraction**
- Automatic mode selection powered by Groq LLM  
- Web UI (HTML + Jinja2)  
- REST API endpoints for developer integration  

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11  
- Conda (recommended)  
- Groq API Key  
- Internet access (SentenceTransformer model downloads on first run)

---

## ⚙️ Setup

### 1️⃣ Create a clean Conda environment

```bash
conda create -n analyst python=3.11 -y
conda activate analyst
```
### 2 Install dependencies

```bash
pip install -r requirements.txt
```

##   3️⃣ Create a .env file
```bash
GROQ_API_KEY=your_groq_api_key_here
```

##   📥 Build the Vector Store

Before starting the API, generate embeddings and create the FAISS index:

```bash
python src/ingest.py
```