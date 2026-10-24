"""Disk-persistent BM25 index with idempotent upsert and filtering."""

import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from config import settings
from ingest.models import Chunk

logger = logging.getLogger(__name__)

INDEX_FILENAME = "bm25_index.pkl"
WORD_REGEX = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase terms for BM25."""
    if not text:
        return []
    return WORD_REGEX.findall(text.lower())


class BM25Store:
    """Persistent BM25 inverted index supporting idempotent indexing and metadata filtering."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or settings.bm25_persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.persist_dir / INDEX_FILENAME

        self._documents: Dict[str, Dict[str, Any]] = {}  # chunk_id -> {content, metadata}
        self._doc_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        self.load()

    def reload(self) -> bool:
        """Reload BM25 index from disk."""
        return self.load()

    def count(self) -> int:
        """Return number of documents in BM25 index."""
        return len(self._documents)

    def get_distinct_books(self) -> List[str]:
        """Return list of distinct book titles in this BM25 index."""
        return sorted(list({
            doc.get("metadata", {}).get("book_title")
            for doc in self._documents.values()
            if doc.get("metadata", {}).get("book_title")
        }))

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        """
        Idempotently add or update chunks in the BM25 index and save to disk.
        Duplicate chunk_ids replace previous entries.
        """
        if not chunks:
            return 0

        # Update in-memory document store
        for chunk in chunks:
            self._documents[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata.to_dict()
            }

        # Rebuild index structures
        self._doc_ids = list(self._documents.keys())
        self._tokenized_corpus = [
            tokenize(self._documents[cid]["content"])
            for cid in self._doc_ids
        ]

        if self._tokenized_corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self.bm25 = None

        self.save()
        logger.info("Successfully indexed %d chunks into BM25 store at '%s'", len(self._documents), self.index_file)
        return len(chunks)

    def save(self) -> None:
        """Serialize index structures and documents to disk."""
        data = {
            "documents": self._documents,
            "doc_ids": self._doc_ids,
            "tokenized_corpus": self._tokenized_corpus,
            "bm25": self.bm25
        }
        with open(self.index_file, "wb") as f:
            pickle.dump(data, f)
        logger.debug("Saved BM25 index to %s", self.index_file)

    def load(self) -> bool:
        """Deserialize index structures from disk if available."""
        if not self.index_file.exists():
            return False

        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
                self._documents = data.get("documents", {})
                self._doc_ids = data.get("doc_ids", [])
                self._tokenized_corpus = data.get("tokenized_corpus", [])
                self.bm25 = data.get("bm25")
            logger.info("Loaded BM25 index with %d documents from %s", len(self._documents), self.index_file)
            return True
        except Exception as e:
            logger.warning("Failed to load existing BM25 index: %s", e)
            return False

    def search(
        self,
        query: str,
        top_k: int = 20,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching query using BM25 scoring.
        Applies optional where_filter on chunk metadata.
        """
        if not self.bm25 or not self._doc_ids:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        doc_scores = self.bm25.get_scores(tokens)

        # Pair scores with document IDs
        results: List[Dict[str, Any]] = []
        for idx, score in enumerate(doc_scores):
            if score <= 0.0:
                continue
            doc_id = self._doc_ids[idx]
            doc_info = self._documents[doc_id]
            metadata = doc_info["metadata"]

            # Apply metadata filter if provided
            if where_filter:
                match = True
                for k, v in where_filter.items():
                    if metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            results.append({
                "chunk_id": doc_id,
                "content": doc_info["content"],
                "metadata": metadata,
                "score": float(score)
            })

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_by_book_title(self, book_title: str) -> int:
        """Remove all chunks associated with book_title to ensure clean idempotent re-indexing."""
        to_remove = [
            cid for cid, doc in self._documents.items()
            if doc.get("metadata", {}).get("book_title") == book_title
        ]
        for cid in to_remove:
            del self._documents[cid]
        if to_remove:
            self._doc_ids = list(self._documents.keys())
            self._tokenized_corpus = [
                tokenize(self._documents[cid]["content"])
                for cid in self._doc_ids
            ]
            self.bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
            self.save()
            logger.info("Removed %d chunks for '%s' from BM25 store", len(to_remove), book_title)
        return len(to_remove)

    def reset(self) -> None:
        """Reset and wipe BM25 index from memory and disk."""
        self._documents.clear()
        self._doc_ids.clear()
        self._tokenized_corpus.clear()
        self.bm25 = None
        if self.index_file.exists():
            self.index_file.unlink()
        logger.info("Reset BM25 store")
