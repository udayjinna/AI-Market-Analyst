import os
import tempfile
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.llm_interface import ai_agent

app = FastAPI(title="AI Market Analyst (Groq + SentenceTransformer)")

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, file: UploadFile = File(None), query: str = Form(...)):
    if file:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(file.file.read())

            # Load document
            if file.filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            docs = loader.load()

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            texts = [chunk.page_content for chunk in chunks]

            # Create embeddings
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = [model.encode(t) for t in texts]

            # Build vectorstore
            vectorstore = FAISS.from_embeddings(embeddings, texts)
            vectorstore.save_local("faiss_store")

    result = ai_agent(query)
    mode = result.get("mode", "unknown")
    response = result.get("response", "No response generated.")

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "query": query, "mode": mode, "response": response}
    )
