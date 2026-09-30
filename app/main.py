import os
import shutil
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.ingestion import ingestion_manager
from app.graph import rag_graph

# FastAPI App
app = FastAPI(
    title="Technical Documentation Assistant API",
    description="Self-corrective RAG system built with LangGraph, ChromaDB, and Groq",
    version="1.0.0"
)

# In-memory storage for feedback
feedback_store = []

# Pydantic Schemas
class QueryRequest(BaseModel):
    question: str = Field(..., example="How do I use Depends in FastAPI?")

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    query_type: Optional[str] = None
    retry_count: int
    is_fallback: bool
    trace: List[str]

class IngestUrlRequest(BaseModel):
    url: str = Field(..., example="https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md")

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str = Field(..., description="'thumbs_up' or 'thumbs_down'", example="thumbs_up")
    comment: Optional[str] = Field(None, example="Clear answer with correct dependency snippet")

# Startup Event: Auto-index sample documentation if DB is empty
@app.on_event("startup")
async def startup_event():
    existing_docs = ingestion_manager.list_indexed_documents()
    if not existing_docs:
        sample_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
        if os.path.exists(sample_dir):
            print("Auto-indexing sample technical documents...")
            ingestion_manager.ingest_sample_directory(sample_dir)
            print("Sample documents indexed successfully.")

# 1. POST /query
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    initial_state = {
        "raw_question": req.question,
        "processed_question": "",
        "query_type": None,
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "citations": [],
        "retry_count": 0,
        "is_fallback": False,
        "status": "processing",
        "trace": []
    }
    
    try:
        result = rag_graph.invoke(initial_state)
        return QueryResponse(
            question=req.question,
            answer=result["generation"],
            citations=result["citations"],
            query_type=result.get("query_type"),
            retry_count=result.get("retry_count", 0),
            is_fallback=result.get("is_fallback", False),
            trace=result.get("trace", [])
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing LangGraph pipeline: {str(e)}")

# 2. POST /ingest
@app.post("/ingest")
async def ingest_endpoint(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    raw_text: Optional[str] = Form(None),
    doc_name: Optional[str] = Form(None)
):
    if not file and not url and not raw_text:
        raise HTTPException(status_code=400, detail="Must provide either a file upload, a URL, or raw_text.")
    
    chunks_created = 0
    source_name = "unknown"
    
    try:
        if file:
            source_name = file.filename
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
            
            try:
                chunks_created = ingestion_manager.ingest_file(tmp_path, source_name=source_name)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        elif url:
            source_name = url
            chunks_created = ingestion_manager.ingest_url(url)
            
        elif raw_text:
            source_name = doc_name or "raw_snippet.txt"
            chunks_created = ingestion_manager.ingest_text(raw_text, source_name=source_name)
            
        return {
            "status": "success",
            "message": f"Successfully ingested '{source_name}'",
            "chunks_created": chunks_created
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

# 3. GET /documents
@app.get("/documents")
async def get_documents_endpoint():
    try:
        docs = ingestion_manager.list_indexed_documents()
        return {
            "total_documents": len(docs),
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

# 4. POST /feedback
@app.post("/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    if req.feedback not in ["thumbs_up", "thumbs_down"]:
        raise HTTPException(status_code=400, detail="Feedback must be 'thumbs_up' or 'thumbs_down'.")
    
    feedback_entry = {
        "question": req.question,
        "answer": req.answer,
        "feedback": req.feedback,
        "comment": req.comment
    }
    feedback_store.append(feedback_entry)
    return {
        "status": "success",
        "message": "Feedback recorded successfully",
        "total_feedback_count": len(feedback_store)
    }

# Interactive Web UI Endpoint
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>FastAPI Documentation Assistant Running</h1><p>Visit /docs for OpenAPI specs.</p>"
