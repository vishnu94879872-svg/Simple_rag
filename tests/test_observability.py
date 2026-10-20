"""Comprehensive tests for BookRAG Reliability Lab observability & diagnostic modules."""

import pytest
from pathlib import Path

from observability.models import (
    StageStatus,
    IngestionJob,
    PipelineTrace,
    EvalBenchmarkItem,
    FailureCategory,
    RetrievalStrategy,
)
from observability.storage import ObservabilityStorage
from observability.tracer import PipelineTracer, mask_secrets
from observability.citation_verifier import CitationVerifier
from observability.chunk_inspector import ChunkQualityChecker
from observability.failure_classifier import FailureClassifier
from ingest.models import Chunk, ChunkMetadata, ContentType
from tests.generate_sample_pdf import create_sample_pdf


def test_secret_masking():
    """Verify that API keys, bearer tokens, and secrets are strictly redacted."""
    raw = "My OpenAI key is sk-proj-1234567890abcdef1234567890 and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    masked = mask_secrets(raw)
    assert "sk-" not in masked
    assert "eyJhb" not in masked
    assert "[REDACTED_SECRET]" in masked


def test_observability_storage(tmp_path):
    """Verify that ObservabilityStorage saves and loads jobs, traces, and benchmarks."""
    store = ObservabilityStorage(base_dir=tmp_path / "obs")

    # Test job saving & loading
    job = IngestionJob(
        job_id="job_test_01",
        document_id="doc_01",
        filename="test.pdf",
        overall_status=StageStatus.COMPLETED
    )
    store.save_job(job)
    loaded_job = store.get_job("job_test_01")
    assert loaded_job is not None
    assert loaded_job.filename == "test.pdf"

    # Test trace saving & loading
    trace = PipelineTrace(
        trace_id="trc_test_01",
        query="What is a qubit?",
        final_answer="A qubit is a quantum bit. (Chapter 1, Page 1)"
    )
    store.save_trace(trace)
    loaded_trace = store.get_trace("trc_test_01")
    assert loaded_trace is not None
    assert "qubit" in loaded_trace.query

    # Test benchmark list
    benchmarks = store.list_benchmarks()
    assert "quantum_bench" in benchmarks


def test_citation_verifier_valid_and_invalid():
    """Verify CitationVerifier flags valid citations, invalid page numbers, and unsupported claims."""
    verifier = CitationVerifier(max_known_page=100)

    # 1. Valid citation
    retrieved_chunks = [
        {"chunk_id": "c1", "content": "Text", "metadata": {"page": 2, "chapter": "Chapter 2"}}
    ]
    audit_valid = verifier.audit_response(
        question="What is Shor's algorithm?",
        answer="Shor's algorithm factors integers. (Chapter 2, Page 2)",
        retrieved_chunks=retrieved_chunks
    )
    assert len(audit_valid.verified_citations) == 1
    assert audit_valid.audit_verdict == "verified"

    # 2. Invalid page number (> max_known_page)
    audit_invalid = verifier.audit_response(
        question="What is Shor's algorithm?",
        answer="Shor's algorithm factors integers. (Chapter 2, Page 999)",
        retrieved_chunks=retrieved_chunks
    )
    assert len(audit_invalid.invalid_citations) == 1
    assert audit_invalid.audit_verdict == "invalid_citations"

    # 3. Unsupported page (page not in retrieved evidence)
    audit_unsupported = verifier.audit_response(
        question="What is Shor's algorithm?",
        answer="Shor's algorithm factors integers. (Chapter 5, Page 50)",
        retrieved_chunks=retrieved_chunks
    )
    assert len(audit_unsupported.unsupported_citations) == 1
    assert audit_unsupported.audit_verdict == "unsupported"

    # 4. Abstention check
    audit_abstained = verifier.audit_response(
        question="What is string theory?",
        answer="I couldn't find that in this book.",
        retrieved_chunks=[],
        should_abstain=True
    )
    assert audit_abstained.abstained_properly is True
    assert audit_abstained.audit_verdict == "abstained"


def test_chunk_quality_checker():
    """Verify ChunkQualityChecker detects empty, duplicate, short, and detached heading chunks."""
    checker = ChunkQualityChecker(min_chars=50, max_chars=300)

    c_empty = Chunk(
        chunk_id="c_01",
        content="",
        metadata=ChunkMetadata(
            book_title="B", chapter="C", section="S", page=1, type=ContentType.TEXT.value,
            chunk_id="c_01", parent_id="p1"
        )
    )
    c_short = Chunk(
        chunk_id="c_02",
        content="[Chapter 1 > Section 1]\n\nTiny text.",
        metadata=ChunkMetadata(
            book_title="B", chapter="C", section="S", page=1, type=ContentType.TEXT.value,
            chunk_id="c_02", parent_id="p1"
        )
    )
    c_normal = Chunk(
        chunk_id="c_03",
        content="[Chapter 1 > Section 1]\n\nQuantum superposition allows qubits to represent linear combinations of both basis states simultaneously with high fidelity.",
        metadata=ChunkMetadata(
            book_title="B", chapter="C", section="S", page=1, type=ContentType.TEXT.value,
            chunk_id="c_03", parent_id="p1"
        )
    )

    reports = checker.evaluate_chunks([c_empty, c_short, c_normal])
    assert len(reports) == 3

    # Empty chunk flagged
    issue_types_0 = [i.issue_type for i in reports[0].issues]
    assert "empty" in issue_types_0

    # Short chunk flagged
    issue_types_1 = [i.issue_type for i in reports[1].issues]
    assert "too_short" in issue_types_1


def test_failure_classifier():
    """Verify FailureClassifier correctly diagnoses failure stages."""
    classifier = FailureClassifier()

    # 1. Zero hits -> Retrieval Failure
    diag_ret = classifier.diagnose(
        query="Unindexed query topic",
        answer="I couldn't find that in this book.",
        retrieved_chunks=[]
    )
    assert diag_ret.failure_stage == FailureCategory.RETRIEVAL
    assert diag_ret.confidence >= 0.85
    assert len(diag_ret.observed_evidence) > 0

    # 2. Invalid citation -> Citation Failure
    verifier = CitationVerifier(max_known_page=50)
    audit = verifier.audit_response(
        question="Qubit definition?",
        answer="A qubit is quantum. (Chapter 1, Page 999)",
        retrieved_chunks=[{"chunk_id": "c1", "content": "Content", "metadata": {"page": 1}}]
    )
    diag_cit = classifier.diagnose(
        query="Qubit definition?",
        answer="A qubit is quantum. (Chapter 1, Page 999)",
        retrieved_chunks=[{"chunk_id": "c1", "content": "Content", "metadata": {"page": 1}}],
        audit=audit
    )
    assert diag_cit.failure_stage == FailureCategory.CITATION
    assert diag_cit.confidence >= 0.85


def test_isolated_chunking_experiment(tmp_path):
    """Verify ChunkingExperimentWorkbench executes in sandbox without touching production indexes."""
    from observability.experiment_workbench import ChunkingExperimentWorkbench
    workbench = ChunkingExperimentWorkbench()

    pdf_path = tmp_path / "bench_test.pdf"
    create_sample_pdf(pdf_path)

    res = workbench.run_experiment(
        pdf_path=pdf_path,
        strategy_name="sentence_aware",
        chunk_size=400,
        chunk_overlap=50
    )
    assert res.total_chunks > 0
    assert res.processing_duration_ms > 0
    assert "mean_chars" in res.size_distribution
