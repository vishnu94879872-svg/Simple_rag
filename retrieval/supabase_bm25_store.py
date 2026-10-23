"""Supabase-persisted BM25 store for cloud deployments.

Serializes the full BM25 index state to JSON and stores it in the
public.rag_bm25_index table so it survives across Streamlit Cloud restarts.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from config import settings
from ingest.models import Chunk

logger = logging.getLogger(__name__)

WORD_REGEX = re.compile(r"\b\w+\b")
SUPABASE_ROW_ID = "singleton"


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return WORD_REGEX.findall(text.lower())


class SupabaseBM25Store:
    """Cloud-persistent BM25 index backed by Supabase JSONB storage."""

    def __init__(self):
        try:
            from supabase import create_client
        except ImportError:
            raise ImportError("supabase package required. Run: pip install supabase")

        self._client = create_client(settings.supabase_url, settings.supabase_key)

        self._documents: Dict[str, Dict[str, Any]] = {}
        self._doc_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Serialize index to JSON and upsert into rag_bm25_index."""
        payload = {
            "documents": self._documents,
            "doc_ids": self._doc_ids,
            "tokenized_corpus": self._tokenized_corpus,
        }
        try:
            self._client.table("rag_bm25_index").upsert(
                {"id": SUPABASE_ROW_ID, "payload": payload},
                on_conflict="id",
            ).execute()
            logger.debug("Saved BM25 index (%d docs) to Supabase", len(self._documents))
        except Exception as e:
            logger.error("Failed to save BM25 index to Supabase: %s", e)

    def _load(self) -> bool:
        """Deserialize BM25 index from Supabase. Returns True if loaded."""
        try:
            resp = (
                self._client.table("rag_bm25_index")
                .select("payload")
                .eq("id", SUPABASE_ROW_ID)
                .maybe_single()
                .execute()
            )
        except Exception as e:
            logger.warning("Could not load BM25 index from Supabase: %s", e)
            return False

        if not resp or not resp.data:
            logger.info("No BM25 index found in Supabase — starting fresh")
            return False

        payload = resp.data.get("payload", {})
        self._documents = payload.get("documents", {})
        self._doc_ids = payload.get("doc_ids", [])
        self._tokenized_corpus = payload.get("tokenized_corpus", [])

        if self._tokenized_corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info("Loaded BM25 index with %d documents from Supabase", len(self._documents))
        return True

    # ------------------------------------------------------------------
    # Public API (mirrors BM25Store)
    # ------------------------------------------------------------------

    def count(self) -> int:
        return len(self._documents)

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0

        for chunk in chunks:
            self._documents[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata.to_dict(),
            }

        self._doc_ids = list(self._documents.keys())
        self._tokenized_corpus = [
            _tokenize(self._documents[cid]["content"]) for cid in self._doc_ids
        ]
        self.bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        self._save()
        logger.info("Upserted %d chunks into Supabase BM25 store", len(chunks))
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 20,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.bm25 or not self._doc_ids:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        doc_scores = self.bm25.get_scores(tokens)
        results: List[Dict[str, Any]] = []

        for idx, score in enumerate(doc_scores):
            if score <= 0.0:
                continue
            doc_id = self._doc_ids[idx]
            doc_info = self._documents[doc_id]
            metadata = doc_info["metadata"]

            if where_filter:
                if any(metadata.get(k) != v for k, v in where_filter.items()):
                    continue

            results.append({
                "chunk_id": doc_id,
                "content": doc_info["content"],
                "metadata": metadata,
                "score": float(score),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_by_book_title(self, book_title: str) -> int:
        to_remove = [
            cid for cid, doc in self._documents.items()
            if doc.get("metadata", {}).get("book_title") == book_title
        ]
        for cid in to_remove:
            del self._documents[cid]
        if to_remove:
            self._doc_ids = list(self._documents.keys())
            self._tokenized_corpus = [
                _tokenize(self._documents[cid]["content"]) for cid in self._doc_ids
            ]
            self.bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
            self._save()
            logger.info("Removed %d BM25 chunks for '%s' from Supabase", len(to_remove), book_title)
        return len(to_remove)

    def reset(self) -> None:
        self._documents.clear()
        self._doc_ids.clear()
        self._tokenized_corpus.clear()
        self.bm25 = None
        try:
            self._client.table("rag_bm25_index").delete().eq("id", SUPABASE_ROW_ID).execute()
            logger.info("Reset Supabase BM25 index")
        except Exception as e:
            logger.warning("Error resetting Supabase BM25 index: %s", e)
