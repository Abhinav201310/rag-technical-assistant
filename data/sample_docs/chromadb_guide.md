# ChromaDB Vector Database Guide

## What is ChromaDB?
Chroma is an open-source embedding database. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.

## Key Features
- **In-Memory and Persistent Storage**: Run locally inside your Python process or persist data to disk.
- **Metadata Filtering**: Filter retrieved chunks using SQL-like where clauses on document metadata.
- **Distance Metrics**: Supports Cosine similarity, Squared L2 distance, and Inner Product (IP).

## Python Usage Example
```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection = client.get_or_create_collection(
    name="technical_docs",
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=["FastAPI is a modern web framework", "LangGraph builds stateful agents"],
    metadatas=[{"source": "fastapi_guide.md"}, {"source": "langgraph_concepts.md"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["How to build agents in Python?"],
    n_results=2
)
print(results)
```
