import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings(BaseModel):
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_db_dir: str = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
    collection_name: str = "technical_documentation"
    top_k_retrieval: int = 4
    max_retries: int = 2
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

settings = Settings()
