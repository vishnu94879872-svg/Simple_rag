"""Unit and integration tests for Phase 2 Indexing (ChromaDB + BM25 + Idempotency)."""

import pytest
from pathlib import Path

from ingest.models import Chunk, ChunkMetadata, ContentType
from retrieval.embeddings import EmbeddingManager
from retrieval.vector_store import ChromaVectorStore
from retrieval.bm25_store import BM25Store
from retrieval.indexer import BookIndexer


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    chunks = []
    texts = [
        ("c_1", "Quantum superposition allows qubits to represent both 0 and 1 simultaneously.", "Introduction"),
        ("c_2", "Quantum entanglement creates correlations between particles at any distance.", "Entanglement"),
        ("c_3", "Shor's algorithm can factor large integers exponentially faster than classical algorithms.", "Algorithms"),
    ]
    for cid, content, section in texts:
        meta = ChunkMetadata(
            book_title="Test Book",
            chapter="Chapter 1",
            section=section,
            page=1,
            type=ContentType.TEXT.value,
            chunk_id=cid,
            parent_id="p_1",
            keywords=["Quantum", "Qubits"],
            heading_path=f"Chapter 1 > {section}"
        )
        chunks.append(Chunk(chunk_id=cid, content=content, metadata=meta))
    return chunks


def test_embedding_manager():
    """Verify BAAI/bge-small-en-v1.5 computes normalized 384-dimensional embeddings."""
    mgr = EmbeddingManager()
    assert mgr.dimension == 384
    embs = mgr.embed_documents(["Hello world", "Quantum computing"])
    assert len(embs) == 2
    assert len(embs[0]) == 384


def test_chroma_indexing_and_idempotency(tmp_path, sample_chunks):
    """Verify Chroma stores vectors and re-running is idempotent."""
    chroma_dir = tmp_path / "chroma"
    store = ChromaVectorStore(persist_dir=chroma_dir, collection_name="test_coll")
    emb_mgr = EmbeddingManager()

    embeddings = emb_mgr.embed_documents([c.content for c in sample_chunks])

    # First indexing run
    store.upsert_chunks(sample_chunks, embeddings)
    assert store.count() == 3

    # Query Chroma
    q_emb = emb_mgr.embed_query("superposition of qubits")
    hits = store.query(q_emb, top_k=2)
    assert len(hits) == 2
    assert hits[0]["chunk_id"] == "c_1"

    # Second indexing run (Idempotency check: count MUST remain 3)
    store.upsert_chunks(sample_chunks, embeddings)
    assert store.count() == 3


def test_bm25_indexing_persistence_and_idempotency(tmp_path, sample_chunks):
    """Verify BM25 persists to disk, supports searching, and is idempotent."""
    bm25_dir = tmp_path / "bm25"
    store = BM25Store(persist_dir=bm25_dir)

    # First index
    store.upsert_chunks(sample_chunks)
    assert store.count() == 3
    assert (bm25_dir / "bm25_index.pkl").exists()

    # Search BM25
    hits = store.search("entanglement particles", top_k=2)
    assert len(hits) >= 1
    assert hits[0]["chunk_id"] == "c_2"

    # Second index (Idempotency check: count MUST remain 3)
    store.upsert_chunks(sample_chunks)
    assert store.count() == 3

    # Reload from disk into new store instance
    reloaded_store = BM25Store(persist_dir=bm25_dir)
    assert reloaded_store.count() == 3
    reloaded_hits = reloaded_store.search("Shor integer factor", top_k=1)
    assert len(reloaded_hits) == 1
    assert reloaded_hits[0]["chunk_id"] == "c_3"


def test_book_indexer_pipeline(tmp_path, sample_chunks):
    """Verify end-to-end BookIndexer coordinates Chroma and BM25 idempotently."""
    chroma_store = ChromaVectorStore(persist_dir=tmp_path / "chroma", collection_name="test_pipeline")
    bm25_store = BM25Store(persist_dir=tmp_path / "bm25")
    indexer = BookIndexer(vector_store=chroma_store, bm25_store=bm25_store)

    summary1 = indexer.index(sample_chunks)
    assert summary1["chroma_total_count"] == 3
    assert summary1["bm25_total_count"] == 3

    # Run again: idempotency check
    summary2 = indexer.index(sample_chunks)
    assert summary2["chroma_total_count"] == 3
    assert summary2["bm25_total_count"] == 3
