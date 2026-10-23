"""Supabase pgvector store — cloud-persistent replacement for ChromaDB.

Used automatically when SUPABASE_URL and SUPABASE_KEY are set and
STORAGE_BACKEND is 'supabase' or 'auto'. Stores chunk embeddings in the
public.rag_chunks table (see scripts/supabase_rag_schema.sql).
"""

import logging
from typing import List, Dict, Any, Optional

from config import settings
from ingest.models import Chunk

logger = logging.getLogger(__name__)


class SupabaseVectorStore:
    """Cloud-persistent vector store backed by Supabase pgvector."""

    def __init__(self):
        try:
            from supabase import create_client
        except ImportError:
            raise ImportError(
                "supabase package is required for cloud storage. "
                "Run: pip install supabase"
            )
        self._client = create_client(settings.supabase_url, settings.supabase_key)
        logger.info("SupabaseVectorStore connected to %s", settings.supabase_url)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """Idempotently upsert chunks with their vector embeddings into Supabase."""
        if not chunks:
            return 0

        rows = []
        for chunk, emb in zip(chunks, embeddings):
            rows.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "embedding": emb,          # list[float] – Supabase accepts this for vector cols
                "metadata": chunk.metadata.to_dict(),
            })

        # Upsert in batches of 200 to stay well within Supabase request limits
        batch_size = 200
        for i in range(0, len(rows), batch_size):
            batch = rows[i: i + batch_size]
            self._client.table("rag_chunks").upsert(batch, on_conflict="chunk_id").execute()

        logger.info("Upserted %d chunks into Supabase rag_chunks", len(chunks))
        return len(chunks)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine similarity search via Supabase RPC (rag_vector_search)."""
        filter_book = None
        if where_filter and "book_title" in where_filter:
            filter_book = where_filter["book_title"]

        try:
            resp = self._client.rpc(
                "rag_vector_search",
                {
                    "query_embedding": query_embedding,
                    "top_k": top_k,
                    "filter_book": filter_book,
                },
            ).execute()
        except Exception as e:
            logger.error("Supabase vector search RPC failed: %s", e)
            return []

        hits: List[Dict[str, Any]] = []
        for row in (resp.data or []):
            dist = float(row.get("distance", 1.0))
            similarity = max(0.0, 1.0 - dist)
            hits.append({
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "metadata": row["metadata"],
                "score": similarity,
                "distance": dist,
            })
        return hits

    def count(self) -> int:
        """Return total number of chunk rows in rag_chunks."""
        try:
            resp = self._client.table("rag_chunks").select("chunk_id", count="exact").execute()
            return resp.count or 0
        except Exception as e:
            logger.warning("Could not count rag_chunks: %s", e)
            return 0

    # ------------------------------------------------------------------
    # Delete / Reset
    # ------------------------------------------------------------------

    def delete_by_book_title(self, book_title: str) -> int:
        """Delete all chunks for a given book_title (idempotent re-index)."""
        try:
            self._client.table("rag_chunks").delete().eq("book_title", book_title).execute()
            logger.info("Deleted existing chunks for '%s' from Supabase", book_title)
            return 1
        except Exception as e:
            logger.warning("Could not delete chunks for '%s': %s", book_title, e)
            return 0

    def reset(self) -> None:
        """Truncate all chunks (use with caution)."""
        try:
            self._client.table("rag_chunks").delete().neq("chunk_id", "").execute()
            logger.info("Reset rag_chunks table")
        except Exception as e:
            logger.warning("Error resetting rag_chunks: %s", e)
