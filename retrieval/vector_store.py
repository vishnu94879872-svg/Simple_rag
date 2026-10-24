"""Persistent ChromaDB vector store wrapper with idempotent upsert."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from ingest.models import Chunk

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "bookrag_chunks"


class ChromaVectorStore:
    """Manages persistent ChromaDB vector storage and query execution."""

    def __init__(self, persist_dir: Optional[Path] = None, collection_name: str = DEFAULT_COLLECTION):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        # Get or create collection; metadata specifies cosine distance
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            "Initialized ChromaVectorStore at '%s' with collection '%s' (current doc count: %d)",
            self.persist_dir,
            self.collection_name,
            self.collection.count()
        )

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """
        Idempotently upsert chunks with their vector embeddings into ChromaDB.
        Re-running with the same chunk_id will update in-place without duplicating.
        """
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata.to_dict() for chunk in chunks]

        # Chroma upsert updates existing documents by ID or inserts new ones
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info("Successfully upserted %d chunks into Chroma collection '%s'", len(chunks), self.collection_name)
        return len(chunks)

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        return self.collection.count()

    def get_distinct_books(self) -> List[str]:
        """Fetch all distinct book titles stored in ChromaDB collection."""
        try:
            metas = self.collection.get(include=["metadatas"])["metadatas"]
            titles = sorted(list({m.get("book_title") for m in metas if m and m.get("book_title")}))
            return titles
        except Exception as e:
            logger.warning("Could not fetch distinct books from Chroma: %s", e)
            return []

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        Returns list of dicts with id, document, metadata, and score.
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        hits: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for idx, doc_id in enumerate(ids):
            # For cosine distance, similarity can be computed as 1 - distance
            dist = distances[idx] if idx < len(distances) else 1.0
            similarity = max(0.0, 1.0 - dist)
            hits.append({
                "chunk_id": doc_id,
                "content": docs[idx] if idx < len(docs) else "",
                "metadata": metas[idx] if idx < len(metas) else {},
                "score": similarity,
                "distance": dist
            })

        return hits

    def delete_by_book_title(self, book_title: str) -> int:
        """Remove all chunks associated with book_title to prevent duplicates or orphaned chunks."""
        try:
            self.collection.delete(where={"book_title": book_title})
            logger.info("Removed existing chunks for '%s' from Chroma collection '%s'", book_title, self.collection_name)
            return 1
        except Exception as e:
            logger.warning("Could not delete prior chunks for '%s' from Chroma: %s", book_title, e)
            return 0

    def reset(self) -> None:
        """Reset / delete collection for testing or re-indexing from scratch."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Reset collection '%s'", self.collection_name)
        except Exception as e:
            logger.warning("Error resetting collection: %s", e)

