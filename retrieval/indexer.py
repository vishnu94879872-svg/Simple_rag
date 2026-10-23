"""Orchestrator for idempotent indexing across Chroma vector store and BM25 store."""

import logging
from typing import List, Dict, Any, Optional

from config import settings
from ingest.models import Chunk
from retrieval.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class BookIndexer:
    """Coordinates embedding, vector indexing, and BM25 index creation."""

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_store=None,
        bm25_store=None,
    ):
        from retrieval._store_factory import get_vector_store, get_bm25_store
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vector_store = vector_store or get_vector_store()
        self.bm25_store = bm25_store or get_bm25_store()

    def index(self, chunks: List[Chunk], clean_existing: bool = True) -> Dict[str, Any]:
        """
        Embed and index all chunks into ChromaDB and BM25 store.
        Guaranteed to be idempotent: re-indexing existing chunks replaces previous
        entries cleanly without duplicate or orphaned entries.
        """
        if not chunks:
            logger.warning("No chunks provided to index.")
            return {
                "chunks_received": 0,
                "chroma_count": self.vector_store.count(),
                "bm25_count": self.bm25_store.count()
            }

        # Clean prior index entries for the same book to ensure idempotency on retries
        if clean_existing:
            book_titles = {c.metadata.book_title for c in chunks if getattr(c, "metadata", None) and getattr(c.metadata, "book_title", None)}
            for bt in book_titles:
                self.vector_store.delete_by_book_title(bt)
                self.bm25_store.delete_by_book_title(bt)

        logger.info("Computing dense embeddings for %d chunks using %s...", len(chunks), self.embedding_manager.model_name)
        texts_to_embed = [chunk.content for chunk in chunks]
        embeddings = self.embedding_manager.embed_documents(texts_to_embed)

        logger.info("Upserting %d vectors into Chroma collection...", len(chunks))
        self.vector_store.upsert_chunks(chunks, embeddings)

        logger.info("Updating BM25 inverted index with %d documents...", len(chunks))
        self.bm25_store.upsert_chunks(chunks)

        chroma_total = self.vector_store.count()
        bm25_total = self.bm25_store.count()

        summary = {
            "chunks_processed": len(chunks),
            "chroma_total_count": chroma_total,
            "bm25_total_count": bm25_total,
            "embedding_dim": self.embedding_manager.dimension
        }
        logger.info("Indexing complete! Chroma documents: %d, BM25 documents: %d", chroma_total, bm25_total)
        return summary
