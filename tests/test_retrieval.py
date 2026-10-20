"""Unit and integration tests for Phase 3 Retrieval."""

import pytest
from pathlib import Path

from ingest.models import Chunk, ChunkMetadata, ParentSection, ContentType
from ingest.parent_store import ParentStore
from retrieval.embeddings import EmbeddingManager
from retrieval.vector_store import ChromaVectorStore
from retrieval.bm25_store import BM25Store
from retrieval.indexer import BookIndexer
from retrieval.query_rewriter import QueryRewriter
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import BGEReranker
from retrieval.context_expansion import ContextExpander
from retrieval.engine import RetrievalEngine


@pytest.fixture(scope="module")
def populated_index(tmp_path_factory):
    """Create a temporary Chroma and BM25 index with rich test chunks."""
    temp_dir = tmp_path_factory.mktemp("retrieval_data")
    chroma_dir = temp_dir / "chroma"
    bm25_dir = temp_dir / "bm25"
    parent_dir = temp_dir / "parents"

    parent_store = ParentStore(store_dir=parent_dir)
    p1 = ParentSection(
        parent_id="p_intro",
        book_title="Quantum Book",
        chapter="Chapter 1: Foundations",
        section="Section 1.1: Qubits",
        page_start=1,
        page_end=1,
        heading_path="Chapter 1: Foundations > Section 1.1: Qubits",
        full_content="Full parent section details: Quantum superposition allows a qubit to be both zero and one at the same time. This is fundamental to all quantum algorithms.",
        chunk_ids=["c_short", "c_qubit"]
    )
    p2 = ParentSection(
        parent_id="p_algo",
        book_title="Quantum Book",
        chapter="Chapter 2: Algorithms",
        section="Section 2.1: Performance",
        page_start=2,
        page_end=2,
        heading_path="Chapter 2: Algorithms > Section 2.1: Performance",
        full_content="Full parent section for algorithms: Shor's algorithm provides exponential speedup for prime factorization.",
        chunk_ids=["c_table", "c_shor"]
    )
    parent_store.save_all([p1, p2])

    chunks = [
        Chunk(
            chunk_id="c_short",
            content="[Chapter 1 > Section 1.1]\n\nQubit state.",  # Very short chunk
            metadata=ChunkMetadata(
                book_title="Quantum Book", chapter="Chapter 1: Foundations", section="Section 1.1: Qubits",
                page=1, type=ContentType.TEXT.value, chunk_id="c_short", parent_id="p_intro",
                keywords=["Qubit"], heading_path="Chapter 1 > Section 1.1"
            )
        ),
        Chunk(
            chunk_id="c_qubit",
            content="[Chapter 1 > Section 1.1]\n\nA qubit is defined as the fundamental unit of quantum information capable of superposition.",
            metadata=ChunkMetadata(
                book_title="Quantum Book", chapter="Chapter 1: Foundations", section="Section 1.1: Qubits",
                page=1, type=ContentType.TEXT.value, chunk_id="c_qubit", parent_id="p_intro",
                keywords=["Qubit", "superposition"], heading_path="Chapter 1 > Section 1.1"
            )
        ),
        Chunk(
            chunk_id="c_shor",
            content="[Chapter 2 > Section 2.1]\n\nShor's algorithm can factor integers in polynomial time O(n^3), breaking classical RSA cryptography.",
            metadata=ChunkMetadata(
                book_title="Quantum Book", chapter="Chapter 2: Algorithms", section="Section 2.1: Performance",
                page=2, type=ContentType.TEXT.value, chunk_id="c_shor", parent_id="p_algo",
                keywords=["Shor's algorithm", "RSA", "cryptography"], heading_path="Chapter 2 > Section 2.1"
            )
        ),
        Chunk(
            chunk_id="c_table",
            content="[Chapter 2 > Section 2.1]\n\n**Table Summary**: Comparison of quantum vs classical algorithm time complexities.\n\n| Algorithm | Classical | Quantum |\n| --- | --- | --- |\n| Shor | Exp | Poly |",
            metadata=ChunkMetadata(
                book_title="Quantum Book", chapter="Chapter 2: Algorithms", section="Section 2.1: Performance",
                page=2, type=ContentType.TABLE.value, chunk_id="c_table", parent_id="p_algo",
                keywords=["Algorithm", "Complexity"], heading_path="Chapter 2 > Section 2.1"
            )
        ),
    ]

    chroma_store = ChromaVectorStore(persist_dir=chroma_dir, collection_name="test_retrieval")
    bm25_store = BM25Store(persist_dir=bm25_dir)
    indexer = BookIndexer(vector_store=chroma_store, bm25_store=bm25_store)
    indexer.index(chunks)

    return {
        "chroma": chroma_store,
        "bm25": bm25_store,
        "parent_store": parent_store,
        "chunks": chunks
    }


def test_query_rewriter():
    """Test 1: Standalone query resolution and pronoun handling."""
    rewriter = QueryRewriter()
    # Simple query stays intact
    res1 = rewriter.rewrite("What is a quantum qubit?")
    assert "qubit" in res1.lower()

    # Query with history pronoun fallback
    history = [
        {"role": "user", "content": "Tell me about Shor's algorithm."},
        {"role": "assistant", "content": "Shor's algorithm is a quantum algorithm for integer factorization."}
    ]
    res2 = rewriter.rewrite("How fast is it?", chat_history=history)
    assert len(res2) > 0


def test_reciprocal_rank_fusion():
    """Test 2: RRF combines dense and sparse ranks; dual-match ranks highest."""
    vector_hits = [
        {"chunk_id": "doc_A", "content": "Doc A", "metadata": {}, "score": 0.95},
        {"chunk_id": "doc_B", "content": "Doc B", "metadata": {}, "score": 0.85},
    ]
    bm25_hits = [
        {"chunk_id": "doc_B", "content": "Doc B", "metadata": {}, "score": 15.0},
        {"chunk_id": "doc_C", "content": "Doc C", "metadata": {}, "score": 10.0},
    ]

    merged = reciprocal_rank_fusion(vector_hits, bm25_hits, rrf_k=60)
    assert len(merged) == 3
    # doc_B appeared in both lists, so its RRF score must be highest!
    assert merged[0]["chunk_id"] == "doc_B"
    assert merged[0]["vector_rank"] == 2
    assert merged[0]["bm25_rank"] == 1


def test_bge_reranker():
    """Test 3: CrossEncoder ranks most relevant text first."""
    reranker = BGEReranker()
    query = "How does Shor's algorithm impact RSA?"
    candidates = [
        {"chunk_id": "irrelevant", "content": "The weather in Seattle is rainy in autumn."},
        {"chunk_id": "relevant", "content": "Shor's algorithm can efficiently factor large integers, compromising RSA public-key encryption."},
    ]
    ranked = reranker.rerank(query, candidates, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["chunk_id"] == "relevant"
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]


def test_context_expansion(populated_index):
    """Test 4: Short chunk is expanded to parent section."""
    expander = ContextExpander(parent_store=populated_index["parent_store"], threshold_chars=100)
    short_hit = [
        {
            "chunk_id": "c_short",
            "content": "[Chapter 1 > Section 1.1]\n\nQubit state.",
            "metadata": {"parent_id": "p_intro", "chunk_id": "c_short"}
        }
    ]
    expanded = expander.expand_hits(short_hit)
    assert len(expanded) == 1
    assert expanded[0]["metadata"]["is_expanded"] is True
    assert "Full parent section details" in expanded[0]["content"]


def test_retrieval_engine_with_filters(populated_index):
    """Test 5: Full retrieval engine with metadata filters."""
    engine = RetrievalEngine(
        vector_store=populated_index["chroma"],
        bm25_store=populated_index["bm25"],
        expander=ContextExpander(parent_store=populated_index["parent_store"])
    )

    # 1. Unfiltered query
    res = engine.retrieve("How does Shor's algorithm factor integers?", top_k=3)
    assert len(res.hits) >= 1
    assert any("Shor" in h["content"] for h in res.hits)

    # 2. Filtered by type=table
    res_table = engine.retrieve("complexity comparison", filters={"type": "table"}, top_k=2)
    assert len(res_table.hits) >= 1
    assert all(h["metadata"]["type"] == "table" for h in res_table.hits)
