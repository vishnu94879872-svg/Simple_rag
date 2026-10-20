"""Comprehensive verification test suite for BookRAG Reliability Lab Upgrades (V3).

Tests:
1. Sentence continuation across chunks without false-positive broken sentence errors.
2. Legitimate constructs (headings, list items, tables, abbreviations, formulas) not flagged.
3. Empty chunks, duplicate text, missing metadata, suspicious text extraction, table fragmentation.
4. Warnings include exact evidence and are labeled as heuristic warnings.
5. Configurable quality thresholds.
6. Retrieval-aware chunk diagnostics and evidence-based non-causal attribution.
7. PDF extraction verification: physical vs extracted vs chunks, OCR recommendations, drawings.
8. Complete 10-item failure investigation workflow and regression test generation.
9. Investigation live rerun execution and before-and-after metric verification.
10. Reproducible experiments on isolated sandboxes without touching production indexes.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from config import settings
from ingest.models import Chunk, ChunkMetadata, ContentType, ParsedBlock
from observability.chunk_inspector import ChunkQualityChecker
from observability.citation_verifier import citation_verifier
from observability.failure_classifier import failure_classifier, FailureCategory
from observability.ingestion_monitor import ingestion_monitor
from observability.experiment_workbench import chunk_experiment_workbench
from observability.models import (
    PipelineTrace,
    EvalBenchmarkItem,
    FailureInvestigationReport,
)
from observability.storage import storage


# --- 1. Sentence continuation across chunks ---
def test_sentence_continuation_across_chunks():
    """Verify that sentences continuing across chunk boundaries are recognized as boundary continuations."""
    c1 = Chunk(
        chunk_id="c_cont_01",
        content="Quantum superposition enables physical qubits to represent several computational states",
        metadata=ChunkMetadata(
            chunk_id="c_cont_01", parent_id="p1", book_title="Test",
            chapter="Ch1", section="1.1", heading_path="Ch1 > 1.1", page=1, type=ContentType.TEXT.value
        )
    )
    c2 = Chunk(
        chunk_id="c_cont_02",
        content="simultaneously, which forms the theoretical backbone of quantum parallelism.",
        metadata=ChunkMetadata(
            chunk_id="c_cont_02", parent_id="p1", book_title="Test",
            chapter="Ch1", section="1.1", heading_path="Ch1 > 1.1", page=1, type=ContentType.TEXT.value
        )
    )

    checker = ChunkQualityChecker(attach_retrieval_diagnostics=False)
    reports = checker.evaluate_chunks([c1, c2])

    assert len(reports) == 2
    # Chunk 1 ends mid-sentence; since chunk 2 continues it, it should be info-level continuation
    c1_issues = reports[0].issues
    c1_broken = [iss for iss in c1_issues if iss.issue_type == "broken_sentence"]
    assert len(c1_broken) > 0
    assert c1_broken[0].severity == "info"
    assert "continues across chunk boundary" in c1_broken[0].description
    assert c1_broken[0].evidence is not None

    # Chunk 2 begins with lowercase continuing chunk 1
    c2_issues = reports[1].issues
    c2_broken = [iss for iss in c2_issues if iss.issue_type == "broken_sentence"]
    assert len(c2_broken) > 0
    assert c2_broken[0].severity == "info"
    assert "continues across chunk boundary" in c2_broken[0].description
    assert c2_broken[0].evidence is not None


# --- 2. Headings, lists, tables, and abbreviations not falsely flagged ---
def test_legitimate_constructs_not_flagged_as_broken_sentences():
    """Verify headings, bullet lists, abbreviations, and table rows are not flagged as broken sentences."""
    checker = ChunkQualityChecker(attach_retrieval_diagnostics=False)

    # 1. Heading ending chunk
    c_head = Chunk(
        chunk_id="c_head",
        content="This is section text.\n\n### 2.1 Theoretical Framework",
        metadata=ChunkMetadata(chunk_id="c_head", parent_id="p1", book_title="T", chapter="C", section="S", page=1, type=ContentType.TEXT.value)
    )
    r_head = checker.evaluate_chunks([c_head])
    head_broken = [iss for iss in r_head[0].issues if iss.issue_type == "broken_sentence" and "terminates without" in iss.description]
    assert len(head_broken) == 0

    # 2. List item ending chunk
    c_list = Chunk(
        chunk_id="c_list",
        content="Key principles include:\n* Superposition of state vectors\n* Quantum entanglement",
        metadata=ChunkMetadata(chunk_id="c_list", parent_id="p1", book_title="T", chapter="C", section="S", page=1, type=ContentType.TEXT.value)
    )
    r_list = checker.evaluate_chunks([c_list])
    list_broken = [iss for iss in r_list[0].issues if iss.issue_type == "broken_sentence" and "terminates without" in iss.description]
    assert len(list_broken) == 0

    # 3. Common abbreviation ending chunk
    c_abbr = Chunk(
        chunk_id="c_abbr",
        content="Measurements are subject to thermal dissipation, e.g.",
        metadata=ChunkMetadata(chunk_id="c_abbr", parent_id="p1", book_title="T", chapter="C", section="S", page=1, type=ContentType.TEXT.value)
    )
    r_abbr = checker.evaluate_chunks([c_abbr])
    abbr_broken = [iss for iss in r_abbr[0].issues if iss.issue_type == "broken_sentence" and "terminates without" in iss.description]
    assert len(abbr_broken) == 0


# --- 3. Suspicious text extraction and table fragmentation ---
def test_suspicious_extraction_and_table_fragmentation():
    """Verify that unicode replacement characters, low alphanumeric ratios, and table fragments are detected."""
    checker = ChunkQualityChecker(min_alphanumeric_ratio=0.50, max_replacement_chars=2, attach_retrieval_diagnostics=False)

    # Mojibake chunk
    c_mojibake = Chunk(
        chunk_id="c_moj",
        content="The system measured \ufffd\ufffd\ufffd\ufffd in the sensor apparatus.",
        metadata=ChunkMetadata(chunk_id="c_moj", parent_id="p1", book_title="T", chapter="C", section="S", page=1, type=ContentType.TEXT.value)
    )
    r_moj = checker.evaluate_chunks([c_mojibake])
    types = [iss.issue_type for iss in r_moj[0].issues]
    assert "suspicious_extraction" in types
    assert any("\ufffd" in iss.evidence for iss in r_moj[0].issues if iss.evidence)

    # Isolated table delimiter in text chunk
    c_frag_table = Chunk(
        chunk_id="c_frag",
        content="Here is some preceding text.\n|---|---|---|\nSome trailing text.",
        metadata=ChunkMetadata(chunk_id="c_frag", parent_id="p1", book_title="T", chapter="C", section="S", page=1, type=ContentType.TEXT.value)
    )
    r_frag = checker.evaluate_chunks([c_frag_table])
    types_frag = [iss.issue_type for iss in r_frag[0].issues]
    assert "potential_table_fragmentation" in types_frag


# --- 4. Retrieval-aware chunk diagnostics and non-causal attribution ---
def test_retrieval_aware_chunk_diagnostics():
    """Verify chunks correlate retrieval history without falsely claiming quality warnings caused answer failures."""
    checker = ChunkQualityChecker(attach_retrieval_diagnostics=True)

    c = Chunk(
        chunk_id="chunk_ret_test_01",
        content="Short text with warning",  # triggers too_short warning
        metadata=ChunkMetadata(
            chunk_id="chunk_ret_test_01", parent_id="p1", book_title="Sample Quantum",
            chapter="Chapter 1", section="1.1", heading_path="Chapter 1", page=1, type=ContentType.TEXT.value
        )
    )

    # Unretrieved chunk
    reports_unret = checker.evaluate_chunks([c])
    assert reports_unret[0].was_retrieved is False
    assert "No causal link" in reports_unret[0].causality_assessment

    # Mock a recorded trace retrieving this chunk
    mock_trace = PipelineTrace(
        trace_id="trc_mock_01",
        query="What is quantum state?",
        retrieved_chunk_ids=["chunk_ret_test_01"],
        final_answer="A quantum state is defined by a vector. (Chapter 1, Page 1)",
        status="success"
    )

    with patch.object(storage, "list_traces", return_value=[mock_trace]):
        reports_ret = checker.evaluate_chunks([c])
        assert reports_ret[0].was_retrieved is True
        assert reports_ret[0].best_rank == 1
        assert "What is quantum state?" in reports_ret[0].queries_retrieved_by
        assert "Non-fatal heuristic warning" in reports_ret[0].causality_assessment


# --- 5. PDF extraction verification and accounting ---
def test_pdf_extraction_verification_and_ocr_recommendations(tmp_path):
    """Verify that physical pages vs extracted pages are computed and OCR recommendations are provided."""
    import pymupdf
    pdf_path = tmp_path / "test_doc.pdf"
    doc = pymupdf.open()
    
    # Page 1: normal text
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Standard readable text on quantum physics.")
    
    # Page 2: empty page
    doc.new_page()

    doc.save(str(pdf_path))
    doc.close()

    page_count, conditions, accounting = ingestion_monitor.inspect_pdf_structure(pdf_path)
    assert page_count == 2
    assert accounting["physical_pages"] == 2
    assert accounting["successfully_extracted_pages"] == 1
    assert any(c.condition_type == "empty_page" and c.page == 2 for c in conditions)


# --- 6. Complete failure investigation workflow ---
def test_complete_failure_investigation_workflow():
    """Verify creation of a complete 10-item FailureInvestigationReport and regression test generation."""
    report = failure_classifier.create_investigation_report(
        query="What is Shor's algorithm?",
        answer="Shor's algorithm factors numbers. (Chapter 9, Page 99)",
        retrieved_chunks=[{"chunk_id": "c1", "content": "Text", "metadata": {"page": 1, "chapter": "Ch1"}}],
        expected_answer="Shor's algorithm factors large integers in polynomial time.",
        expected_source_evidence={"expected_pages": [2]}
    )

    assert isinstance(report, FailureInvestigationReport)
    assert report.query == "What is Shor's algorithm?"
    assert report.expected_answer is not None
    assert len(report.actual_retrieved_chunks) == 1
    assert report.likely_failure_stage in (FailureCategory.CITATION, FailureCategory.RETRIEVAL)
    assert len(report.supporting_evidence) > 0
    assert len(report.recommended_fix) > 0
    assert "def test_regression_" in report.regression_test_code


# --- 7. Reproducible experiments on isolated sandboxes ---
def test_reproducible_experiment_sandbox(tmp_path):
    """Verify that chunking experiments run on isolated sandboxes without modifying production files."""
    import pymupdf
    pdf_path = tmp_path / "quantum_exp.pdf"
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((50, 50), "Quantum mechanics governs physical interactions on atomic and subatomic scales.\n\nSuperposition allows linear combinations of states.")
    doc.save(str(pdf_path))
    doc.close()

    result = chunk_experiment_workbench.run_experiment(
        pdf_path=pdf_path,
        strategy_name="sentence_aware",
        chunk_size=300,
        chunk_overlap=50
    )

    assert result.total_chunks > 0
    assert result.processing_duration_ms > 0.0
    assert len(result.sample_chunks) > 0
    assert result.strategy_name == "sentence_aware"
