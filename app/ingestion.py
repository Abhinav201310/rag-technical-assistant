import os
import uuid
import glob
from typing import List, Dict, Any, Optional
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config import settings

class IngestionManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n## ", "\n### ", "\n```", "\n\n", "\n", " ", ""]
        )
        os.makedirs(settings.chroma_db_dir, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_db_dir
        )

    def split_and_add_documents(self, documents: List[Document]) -> int:
        chunks = self.text_splitter.split_documents(documents)
        if chunks:
            # Assign unique IDs if not present
            ids = [str(uuid.uuid4()) for _ in chunks]
            self.vector_store.add_documents(documents=chunks, ids=ids)
            self.vector_store.persist()
        return len(chunks)

    def ingest_file(self, file_path: str, source_name: Optional[str] = None) -> int:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        source = source_name or os.path.basename(file_path)
        doc = Document(page_content=content, metadata={"source": source, "type": "file"})
        return self.split_and_add_documents([doc])

    def ingest_text(self, text_content: str, source_name: str) -> int:
        doc = Document(page_content=text_content, metadata={"source": source_name, "type": "text"})
        return self.split_and_add_documents([doc])

    def ingest_url(self, url: str) -> int:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        doc = Document(page_content=response.text, metadata={"source": url, "type": "url"})
        return self.split_and_add_documents([doc])

    def ingest_sample_directory(self, dir_path: str) -> Dict[str, int]:
        results = {}
        files = glob.glob(os.path.join(dir_path, "*.md")) + glob.glob(os.path.join(dir_path, "*.txt"))
        for file in files:
            source = os.path.basename(file)
            count = self.ingest_file(file, source_name=source)
            results[source] = count
        return results

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        collection = self.vector_store._collection
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", []) or []
        
        # Aggregate unique sources with chunk counts
        summary = {}
        for meta in metadatas:
            src = meta.get("source", "unknown")
            doc_type = meta.get("type", "file")
            if src not in summary:
                summary[src] = {"source": src, "type": doc_type, "chunk_count": 0}
            summary[src]["chunk_count"] += 1
            
        return list(summary.values())

    def get_retriever(self, k: int = 4):
        return self.vector_store.as_retriever(search_kwargs={"k": k})

ingestion_manager = IngestionManager()
