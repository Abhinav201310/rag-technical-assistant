# DocuMind: Self-Corrective RAG Technical Documentation Assistant

A high-performance, self-corrective Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **FastAPI**, **ChromaDB**, and **Groq (Llama 3.3 70B)**.

---

## 🏛️ System Architecture

```
User Query ──► [Node 1: Query Analysis & Expansion]
                     │
                     ▼
             [Node 2: Vector Retrieval (ChromaDB)]
                     │
                     ▼
             [Node 3: LLM Document Grading]
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
(Relevant Docs Found)     (All Docs Irrelevant)
        │                         │
        │               Retry Count < Max?
        │                ├── Yes ──► [Node: Query Rewriter] ──► (Back to Node 2)
        │                └── No  ──► [Node: Graceful Fallback] ──► END
        ▼
[Node 4: Context Generation & Source Citations]
        │
        ▼
       END
```

---

## 🌟 Key Features

1. **Self-Corrective RAG (CRAG Pattern):**
   - Evaluates retrieved chunks with an LLM evaluator before feeding them into generation.
   - Automatically rewrites and re-retrieves if documents fail relevance grading (with configurable retry limits).
2. **Query Analysis & Classification:**
   - Classifies queries into categories (`conceptual`, `how-to`, `troubleshooting`, `api_reference`).
   - Expands ambiguous user queries with domain-specific terminology.
3. **Flexible Ingestion Pipeline:**
   - Ingests Markdown, text files, and remote URLs.
   - Sentence-transformer embeddings (`all-MiniLM-L6-v2`) with local ChromaDB vector store.
4. **Complete REST API + Interactive UI:**
   - `POST /query`: Run question through LangGraph pipeline with execution trace.
   - `POST /ingest`: Ingest new files or web URLs.
   - `GET /documents`: List indexed corpus and chunk distributions.
   - `POST /feedback`: Capture user satisfaction metrics (thumbs up/down).
   - Built-in modern web UI at `/`.

---

## 🚀 Quick Setup & Installation

### 1. Clone & Navigate
```bash
git clone <your-repo-url>
cd rag-technical-assistant
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_DB_DIR=./data/chroma_db
PORT=8000
HOST=0.0.0.0
```

### 3. Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Run the Application
```bash
uvicorn app.main:app --reload --port 8000
```
- Open UI: `http://localhost:8000`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`

---

## 📡 API Reference & Example Requests

### 1. Query Assistant
`POST /query`
```json
{
  "question": "How does dependency injection work in FastAPI?"
}
```
**Response:**
```json
{
  "question": "How does dependency injection work in FastAPI?",
  "answer": "FastAPI uses the `Depends()` function for dependency injection...",
  "citations": ["fastapi_guide.md"],
  "query_type": "conceptual",
  "retry_count": 0,
  "is_fallback": false,
  "trace": [
    "Executing Node 1: Query Analysis & Expansion",
    "Query classified as 'conceptual', expanded: 'FastAPI dependency injection Depends usage'",
    "Executing Node 2: Retrieval for query 'FastAPI dependency injection Depends usage'",
    "Retrieved 4 candidate chunks from vector database",
    "Executing Node 3: Document Grading (Self-Corrective evaluation)",
    "Grading complete: 4 / 4 chunks deemed relevant",
    "Executing Node 4: Answer Generation with Context & Citations"
  ]
}
```

### 2. Ingest New Documentation
`POST /ingest`
- Form fields: `file` (Multipart file) OR `url` (e.g. `https://raw.githubusercontent.com/...`)

### 3. View Indexed Corpus
`GET /documents`
```json
{
  "total_documents": 4,
  "documents": [
    { "source": "fastapi_guide.md", "type": "file", "chunk_count": 3 },
    { "source": "langgraph_concepts.md", "type": "file", "chunk_count": 4 },
    { "source": "pydantic_guide.md", "type": "file", "chunk_count": 3 },
    { "source": "chromadb_guide.md", "type": "file", "chunk_count": 2 }
  ]
}
```

### 4. Submit Feedback
`POST /feedback`
```json
{
  "question": "How to create StateGraph?",
  "answer": "Use StateGraph(GraphState)...",
  "feedback": "thumbs_up",
  "comment": "Accurate syntax provided"
}
```

---

## 🧠 Design Decisions & Tradeoffs

1. **Why LangGraph for Self-Corrective Flow?**
   - LangGraph allows explicit state modeling (`GraphState`), looping (retry cycle on retrieval failure), and conditional transitions that are much easier to debug and observe than black-box autonomous agents.
2. **Why Local Embeddings (`all-MiniLM-L6-v2`)?**
   - High speed, 0 API cost, deterministic vector dimensionality, and zero rate-limiting during document batch ingestion.
3. **Chunking Strategy:**
   - `RecursiveCharacterTextSplitter` with technical Markdown separators (`##`, `###`, ```` ``` ````). Preserves complete code snippets and header context inside the same chunk.
4. **Future Improvements with More Time:**
   - Multi-turn conversation memory with session threads.
   - Hybrid retrieval (BM25 lexical search + dense vector cosine similarity with Cross-Encoder reranking).
   - Real-time Tavily web search fallback when the document corpus has zero match.
