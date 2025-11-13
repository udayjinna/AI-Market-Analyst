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

# 🏃 Running the Application

Start the FastAPI server:
```bash 
uvicorn src.api:app --reload
```

Then open:
📍 [http://127.0.0.1:8000](http://127.0.0.1:8000)  
This loads the HTML UI for interacting with the AI Market Analyst.

# 🧠 How the System Works

The project follows a multi-stage intelligent processing pipeline.

## 1️⃣ Text Chunking
- Documents are split using: `RecursiveCharacterTextSplitter`
- **Chunk size:** 800 characters  
- **Chunk overlap:** 200  
- Chosen to preserve semantic continuity while optimizing embedding accuracy.

## 2️⃣ Embeddings
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Reasoning:**
  - High accuracy for semantic similarity
  - Lightweight and fast on CPU
  - Free & open-source
  - Stable embeddings ideal for FAISS indexing

## 3️⃣ Vector Database (FAISS)
- FAISS is used for:
  - Fast nearest-neighbor retrieval
  - Efficient cosine similarity search
  - Persistent storage under `vector_store/`
- **Why FAISS?**
  - Best-in-class performance
  - Works offline
  - Easy integration with LangChain

## 4️⃣ LLM Router (Mode Selection)
The Groq-powered LLM analyzes a user question and decides:

| User Query Type       | Selected Mode |
|----------------------|---------------|
| Direct questions     | `qa`          |
| Insight extraction   | `findings`    |
| JSON output needed   | `structure`   |

This gives the system flexibility of three agents inside one unified interface.

## 5️⃣ Agents / Modes

### 🔍 General Q&A
Retrieves context using FAISS and answers based purely on the document.

### 📊 Market Findings
Summarizes insights such as:
- Market size
- CAGR
- Competitive landscape
- SWOT analysis
- Opportunities & threats

### 🧱 Structured JSON Extraction
Produces structured output like:
```bash
{
"company": "Innovate Inc.",
"market_size": "$15 billion",
"cagr": "22%",
"competitors": [
{"name": "Synergy Systems", "share": "18%"},
{"name": "FutureFlow", "share": "15%"},
{"name": "QuantumLeap", "share": "3%"}
]
}
```


This is useful for automation pipelines.

# 🔌 API Endpoints

## POST /upload
Upload a PDF or TXT document to process:  
Example (cURL):
```bash
curl -X POST -F "file=@report.pdf" http://127.0.0.1:8000/upload
```


## POST /analyze
Send a query to the AI Market Analyst:
```bash
{
"query": "Summarize the competitive landscape."
}
```

**Response:**
```bash
{
"answer": "Innovate Inc. holds 12% market share. Competitors include Synergy Systems (18%) and FutureFlow (15%). A new entrant, QuantumLeap, holds 3% but has strong funding.",
"mode_used": "findings"
}
```

# 📌 Notes & Limitations

- SentenceTransformer may take time to download on first run
- FAISS index must be rebuilt if the document changes
- The model is **not trained on financial or legal advice**
- `.env` file must remain private
- Best tested on Python 3.11 with Conda

# 📄 License (MIT)

MIT License  
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction...  
