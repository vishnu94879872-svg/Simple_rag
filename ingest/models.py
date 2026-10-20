"""Data models for BookRAG Ingestion."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Supported content types for parsed blocks and chunks."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    CODE = "code"


class ChunkMetadata(BaseModel):
    """Metadata attached to every indexed chunk."""
    book_title: str
    chapter: str
    section: str
    page: int
    type: str
    chunk_id: str
    parent_id: str
    keywords: List[str] = Field(default_factory=list)
    heading_path: str = ""
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to flat dictionary for storage (e.g. Chroma/BM25)."""
        data = self.model_dump()
        # Chroma requires list to be string or handled properly if not primitive
        if isinstance(data.get("keywords"), list):
            data["keywords"] = ", ".join(data["keywords"])
        return data


class Chunk(BaseModel):
    """A single retrievable document chunk."""
    chunk_id: str
    content: str
    metadata: ChunkMetadata


class ParentSection(BaseModel):
    """A parent section containing multiple chunks for context expansion."""
    parent_id: str
    book_title: str
    chapter: str
    section: str
    page_start: int
    page_end: int
    heading_path: str
    full_content: str
    chunk_ids: List[str] = Field(default_factory=list)


class ParsedBlock(BaseModel):
    """A raw block parsed from a PDF page before chunking."""
    content_type: ContentType = ContentType.TEXT
    content: str = ""
    page: int = 1
    chapter: Optional[str] = "Unknown"
    section: Optional[str] = "Unknown"
    heading_path: Optional[str] = ""
    keywords: Optional[List[str]] = Field(default_factory=list)
    image_path: Optional[str] = None
    table_markdown: Optional[str] = None
