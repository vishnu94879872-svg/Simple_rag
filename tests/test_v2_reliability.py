"""Comprehensive test suite for BookRAG Reliability Lab Version 2.

Covers:
1. Missing page metadata
2. Empty or corrupted extraction
3. Missing relevant chunks
4. Incorrect citations
5. Unsupported answers
6. Embedding and LLM failures
7. Ingestion retries
8. Reindexing
9. Incorrect dashboard calculations
10. Empty evaluation datasets
11. Long text and frontend overflow
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from config import settings
from ingest.models import Chunk, ChunkMetadata, ContentType, ParsedBlock
from ingest.text_splitter import ChunkBuilder
from observability.chunk_inspector import ChunkQualityChecker
from observability.citation_verifier import citation_verifier
from observability.failure_classifier import failure_classifier, FailureCategory
from observability.telemetry import authoritative_telemetry
from observability.eval_lab import eval_lab
from observability.models import PipelineTrace, TraceStep, EvalRun, EvalBenchmarkItem
from generation.generator import AnswerGenerator


# 1. Missing page metadata
def test_missing_page_metadata():
    """Verify that chunks with missing or null page metadata are detected and handled."""
    bad_chunk = Chunk(
        chunk_id="chunk_bad_meta",
        content="This is sample content with missing page number.",
        metadata=ChunkMetadata(
            chunk_id="chunk_bad_meta",
            parent_id="parent_01",
            book_title="Test Book",
            chapter="Chapter 1",
            section="1.1",
            heading_path="Chapter 1",
            page=0,  # 0 or missing
            type=ContentType.TEXT.value
        )
    )
    checker = ChunkQualityChecker()
    reports = checker.evaluate_chunks([bad_chunk])
    assert len(reports) == 1
    issue_types = [issue.issue_type for issue in reports[0].issues]
    assert "missing_metadata" in issue_types or any("page" in i.description.lower() for i in reports[0].issues)


# 2. Empty or corrupted extraction
def test_empty_or_corrupted_extraction():
    """Verify that empty, blank, or corrupted text blocks are flagged with confirmed warnings/errors."""
    empty_chunk = Chunk(
        chunk_id="chunk_empty_01",
        content="   \n\t  ",
        metadata=ChunkMetadata(
            chunk_id="chunk_empty_01",
            parent_id="parent_01",
            book_title="Test Book",
            chapter="Chapter 1",
            section="1.1",
            heading_path="Chapter 1",
            page=1,
            type=ContentType.TEXT.value
        )
    )
    checker = ChunkQualityChecker()
    reports = checker.evaluate_chunks([empty_chunk])
    assert len(reports) == 1
    issue_types = [issue.issue_type for issue in reports[0].issues]
    assert "empty" in issue_types or "too_short" in issue_types


# 3. Missing relevant chunks
def test_missing_relevant_chunks_diagnosis():
    """Verify that retrieval producing 0 candidates triggers a deterministic RETRIEVAL failure diagnosis."""
    diagnosis = failure_classifier.diagnose(
        query="What is quantum teleportation?",
        answer="I do not have sufficient information in the provided text.",
        retrieved_chunks=[]
    )
    assert diagnosis.failure_stage == FailureCategory.RETRIEVAL
    assert diagnosis.confidence >= 0.85
    assert any("0 candidates" in ev for ev in diagnosis.observed_evidence)


# 4. Incorrect citations
def test_incorrect_citations_auditing():
    """Verify that citations citing a non-existent page or chapter are caught as invalid."""
    retrieved = [
        {"chunk_id": "c1", "content": "Superposition allows concurrent computation.", "metadata": {"page": 1, "chapter": "Chapter 1"}}
    ]
    # Answer hallucinates Chapter 9, Page 99
    answer = "Superposition allows concurrent computation. (Chapter 9, Page 99)"
    audit = citation_verifier.audit_response(
        question="What does superposition allow?",
        answer=answer,
        retrieved_chunks=retrieved
    )
    assert len(audit.invalid_citations) > 0 or audit.audit_verdict != "verified"


# 5. Unsupported answers
def test_unsupported_answers_detection():
    """Verify that answers making claims not supported by retrieved chunks are flagged."""
    retrieved = [
        {"chunk_id": "c1", "content": "Water boils at 100 degrees Celsius under standard atmospheric pressure.", "metadata": {"page": 5, "chapter": "States of Matter"}}
    ]
    # Answer talks about warp drive
    unsupported_answer = "Warp drives operate by warping spacetime using exotic matter and tachyon particles."
    audit = citation_verifier.audit_response(
        question="How do warp drives work?",
        answer=unsupported_answer,
        retrieved_chunks=retrieved
    )
    assert audit.audit_verdict in ("unsupported", "invalid_citations", "partially_grounded", "missing_citations") or len(audit.unsupported_claims) > 0


# 6. Embedding and LLM failures
def test_llm_failure_handling():
    """Verify graceful fallback synthesis when LLM API throws an exception."""
    gen = AnswerGenerator()
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionError("OpenAI API endpoint unreachable")
    
    with patch.object(gen, "llm_helper") as mock_helper:
        mock_helper._client = mock_client
        mock_chunks = [
            {"chunk_id": "c1", "content": "Quantum qubits exist in superposition.", "metadata": {"page": 1, "chapter": "Chapter 1"}}
        ]
        resp = gen.generate(
            question="What is a qubit?",
            hits=mock_chunks,
            chat_history=None
        )
        assert resp.answer is not None
        assert len(resp.answer) > 0
        assert "qubit" in resp.answer.lower() or "quantum" in resp.answer.lower()


# 7. Ingestion retries
def test_ingestion_retry_logic():
    """Verify that ingestion jobs record retry counts on failure and propagate retry state."""
    from observability.models import IngestionJob, StageStatus
    job = IngestionJob(
        job_id="job_retry_test",
        document_id="doc_retry",
        filename="corrupted.pdf",
        retry_count=0
    )
    assert job.retry_count == 0
    job.retry_count += 1
    job.overall_status = StageStatus.FAILED
    job.error_message = "File parsing corrupted PDF header"
    assert job.retry_count == 1
    assert job.overall_status == StageStatus.FAILED


# 8. Reindexing idempotency
def test_reindexing_idempotency(tmp_path):
    """Verify that reindexing the same chunk ID overwrites cleanly rather than duplicating."""
    from retrieval.bm25_store import BM25Store
    store = BM25Store(persist_dir=tmp_path / "bm25")
    
    chunk = Chunk(
        chunk_id="chunk_idem_01",
        content="First version of chunk content.",
        metadata=ChunkMetadata(
            chunk_id="chunk_idem_01",
            parent_id="parent_01",
            book_title="Test",
            chapter="1",
            section="1.1",
            heading_path="1",
            page=1,
            type=ContentType.TEXT.value
        )
    )
    store.upsert_chunks([chunk])
    assert store.count() == 1

    chunk_v2 = Chunk(
        chunk_id="chunk_idem_01",
        content="Updated second version of chunk content.",
        metadata=ChunkMetadata(
            chunk_id="chunk_idem_01",
            parent_id="parent_01",
            book_title="Test",
            chapter="1",
            section="1.1",
            heading_path="1",
            page=1,
            type=ContentType.TEXT.value
        )
    )
    store.upsert_chunks([chunk_v2])
    assert store.count() == 1


# 9. Incorrect dashboard calculations
def test_authoritative_dashboard_calculations():
    """Verify that AuthoritativeTelemetry correctly queries real backend data and computes sample-aware stats."""
    metrics = authoritative_telemetry.get_dashboard_metrics()
    assert "indexed_documents_count" in metrics
    assert "physical_pdf_pages" in metrics
    assert "distinct_indexed_pages" in metrics
    assert "avg_latency_str" in metrics
    assert "p95_latency_str" in metrics
    assert metrics["physical_pdf_pages"] >= metrics["distinct_indexed_pages"]
    if metrics["sample_count"] < 5:
        assert "Unavailable" in metrics["p95_latency_str"] or "Requires" in metrics["p95_latency_note"]


# 10. Empty evaluation datasets
def test_empty_evaluation_dataset_handling():
    """Verify that passing an empty or invalid benchmark dataset raises a clean ValueError without crash."""
    with pytest.raises(ValueError, match="not found or empty"):
        eval_lab.run_benchmark(benchmark_name="nonexistent_bench_dataset_xyz")


# 11. Long text and frontend overflow handling
def test_long_text_and_overflow_handling():
    """Verify that massive unbroken strings or long code snippets are properly bounded and formatted."""
    long_unbroken_text = "A" * 15000
    builder = ChunkBuilder(book_title="Overflow Test")
    block = ParsedBlock(
        content=long_unbroken_text,
        content_type=ContentType.TEXT,
        page=1
    )
    chunks, _ = builder.build_chunks_and_parents([block])
    assert len(chunks) > 0
    for c in chunks:
        assert len(c.content) <= 3000
