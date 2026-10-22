import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure stdout and stderr support UTF-8 on Windows without crashing on charmap
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables from .env file if available
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseModel):
    """Application settings and paths."""
    
    # Provider & Model configurations
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = Field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    vision_model: str = Field(default_factory=lambda: os.getenv("VISION_MODEL", "gpt-4o-mini"))
    
    # Embeddings & Reranker
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    reranker_model: str = Field(default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"))
    
    # Storage directories
    data_dir: Path = Field(default_factory=lambda: BASE_DIR / os.getenv("DATA_DIR", "data"))
    images_dir: Path = Field(default_factory=lambda: BASE_DIR / os.getenv("IMAGES_DIR", "data/images"))
    parent_store_dir: Path = Field(default_factory=lambda: BASE_DIR / os.getenv("PARENT_STORE_DIR", "data/parent_store"))
    chroma_persist_dir: Path = Field(default_factory=lambda: BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    bm25_persist_dir: Path = Field(default_factory=lambda: BASE_DIR / os.getenv("BM25_PERSIST_DIR", "data/bm25"))
    observability_dir: Path = Field(default_factory=lambda: BASE_DIR / os.getenv("OBSERVABILITY_DIR", "data/observability"))
    
    # Chunking
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))

    # Supabase-backed observability
    supabase_url: str = Field(default_factory=lambda: os.getenv("SUPABASE_URL", "").strip())
    supabase_key: str = Field(default_factory=lambda: os.getenv("SUPABASE_KEY", "").strip())
    supabase_table: str = Field(default_factory=lambda: os.getenv("SUPABASE_TABLE", "observability_records").strip())
    storage_backend: str = Field(default_factory=lambda: os.getenv("STORAGE_BACKEND", "auto").strip().lower())

    @property
    def supabase_configured(self) -> bool:
        if not self.supabase_url or not self.supabase_key:
            return False
        placeholders = ("your-project", "your_service_role_key")
        return not any(token in self.supabase_url or token in self.supabase_key for token in placeholders)

    def ensure_directories(self) -> None:
        """Create necessary data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.parent_store_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_persist_dir.mkdir(parents=True, exist_ok=True)
        self.observability_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
