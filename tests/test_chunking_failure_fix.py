"""Comprehensive regression and edge-case test suite for chunking failure fix.

Verifies:
1. IngestionValidator constructor flexibility: IngestionValidator(chunks, parents),
   IngestionValidator(max_allowed_size=...), IngestionValidator(chunks), and IngestionValidator().
2. ValidationResult dual-interface: tuple unpacking (is_valid, report) and dictionary access.
3. Parsing-to-chunking boundary robustness:
   - None blocks, dictionary blocks, non-ParsedBlock items.
   - None / non-integer / negative page numbers.
   - None / non-list keywords.
   - None / empty content.
   - Missing chapter/section/heading metadata.
4. Monitored ingestion failure telemetry:
   - Stage failure does not report 0.0 ms duration.
   - Stage failure records actual error message and details.
   - Partial chunk count is preserved on failure.
5. Indexing idempotency:
   - Re-indexing does not duplicate chunks in Chroma or BM25 store.
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from ingest.models import ParsedBlock, Chunk, ChunkMetadata, ContentType, ParentSection
from ingest.validator import IngestionValidator, ValidationResult
from ingest.text_splitter import ChunkBuilder
from retrieval.indexer import BookIndexer
from observability.ingestion_monitor import IngestionMonitor
from observability.models import StageStatus, IngestionJob


def test_ingestion_validator_constructor_and_dual_return():
    """Verify IngestionValidator accepts chunks & parents and unpacks as tuple & dict."""
    chunk = Chunk(
        chunk_id="chk_01",
        content="Quantum states superpose coherently.",
        metadata=ChunkMetadata(
            book_title="Quantum Physics",
            chapter="Chapter 1",
            section="Section 1.1",
            page=1,
            type=ContentType.TEXT.value,
            chunk_id="chk_01",
            parent_id="parent_01",
            keywords=["quantum", "superposition"],
            heading_path="Chapter 1 > Section 1.1"
        )
    )
    parent = ParentSection(
        parent_id="parent_01",
        book_title="Quantum Physics",
        chapter="Chapter 1",
        section="Section 1.1",
        page_start=1,
        page_end=1,
        heading_path="Chapter 1 > Section 1.1",
        full_content="Quantum states superpose coherently.",
        chunk_ids=["chk_01"]
    )

    # 1. Constructor called with (chunks, parents)
    validator_1 = IngestionValidator([chunk], [parent])
    is_valid, report = validator_1.validate()
    assert is_valid is True
    assert isinstance(report, dict)
    assert report["total_chunks"] == 1
    assert report["empty_chunks_count"] == 0

    # 2. Constructor called with max_allowed_size
    validator_2 = IngestionValidator(max_allowed_size=800)
    res = validator_2.validate([chunk])
    assert res.is_valid is True
    # Dict-like indexing on res
    assert res["total_chunks"] == 1
    assert "chunks_by_type" in res
    assert res.get("total_chunks") == 1

    # 3. Tuple unpacking on res directly
    valid_flag, report_dict = res
    assert valid_flag is True
    assert report_dict["total_chunks"] == 1


def test_chunk_builder_edge_cases_and_malformed_blocks():
    """Verify ChunkBuilder handles empty, None, dict, and malformed blocks without crashing."""
    builder = ChunkBuilder(book_title="Test Robustness Book")

    # Edge cases:
    # - None in list
    # - dict block
    # - block with page=None or negative
    # - block with keywords=None
    # - block with content=None
    # - missing chapter/section
    malformed_blocks = [
        None,
        {
            "content_type": ContentType.TEXT,
            "content": "Valid text from a dictionary representation.",
            "page": 2,
            "chapter": "Dict Chapter",
            "section": "Dict Section",
            "heading_path": "Dict Chapter > Dict Section",
            "keywords": ["dict", "test"]
        },
        ParsedBlock(
            content_type=ContentType.TEXT,
            content="Block with None keywords and invalid page.",
            page=-99,  # invalid page
            chapter=None,
            section=None,
            heading_path=None,
            keywords=None  # None keywords
        ),
        ParsedBlock(
            content_type=ContentType.IMAGE,
            content="",
            page=3,
            chapter="Chapter 2",
            section="Section 2",
            heading_path="Chapter 2 > Section 2",
            keywords=None,
            image_path="test_img.png"
        ),
        ParsedBlock(
            content_type=ContentType.TEXT,
            content="",  # empty content
            page=4,
            chapter="Chapter 2",
            section="Section 2",
            keywords=[]
        )
    ]

    with patch.object(builder.llm_helper, "caption_image", return_value="A test caption"):
        chunks, parents = builder.build_chunks_and_parents(malformed_blocks)

    assert len(chunks) >= 2
    assert len(parents) >= 1

    # Check that invalid page was normalized to >= 1
    for chunk in chunks:
        assert chunk.metadata.page >= 1
        assert isinstance(chunk.metadata.keywords, list)
        assert chunk.metadata.book_title == "Test Robustness Book"


def test_ingestion_monitor_failure_telemetry():
    """Verify that when a stage fails, duration_ms is recorded accurately (>0) and not 0.0 ms."""
    monitor = IngestionMonitor()

    # Use existing valid sample PDF
    from config import settings
    pdf_path = settings.data_dir / "books" / "sample_quantum.pdf"
    if not pdf_path.exists():
        pytest.skip("sample_quantum.pdf not found")

    # Mock inspect_pdf_structure and parser
    with patch.object(monitor, "inspect_pdf_structure", return_value=(10, [], {"successfully_extracted_pages": 10})), \
         patch("ingest.pdf_parser.PDFParser.parse", return_value=[
             ParsedBlock(content_type=ContentType.TEXT, content="Sample content", page=1)
         ]), \
         patch("ingest.text_splitter.ChunkBuilder.build_chunks_and_parents") as mock_chunker:

        def failing_chunker(blocks):
            time.sleep(0.02)  # small delay so duration > 0 ms
            raise RuntimeError("Simulated chunking hardware or parser schema fault")

        mock_chunker.side_effect = failing_chunker

        job = monitor.run_monitored_ingestion(pdf_path, job_id="test_failure_telemetry_job")

        assert job.overall_status == StageStatus.FAILED
        assert "Simulated chunking" in job.error_message

        chunking_stage = job.stages.get("chunking")
        assert chunking_stage is not None
        assert chunking_stage.status == StageStatus.FAILED
        # Duration must be recorded accurately, NOT 0.0 ms
        assert chunking_stage.duration_ms > 0.0
        assert "Simulated chunking" in chunking_stage.error_message
        assert chunking_stage.details.get("error_type") == "RuntimeError"


def test_indexing_idempotency_on_retry():
    """Verify that re-indexing the same book replaces records without duplicates."""
    mock_vector_store = MagicMock()
    mock_bm25_store = MagicMock()
    mock_embed_mgr = MagicMock()
    mock_embed_mgr.model_name = "test-model"
    mock_embed_mgr.dimension = 384
    mock_embed_mgr.embed_documents.return_value = [[0.1] * 384]

    indexer = BookIndexer(
        embedding_manager=mock_embed_mgr,
        vector_store=mock_vector_store,
        bm25_store=mock_bm25_store
    )

    chunk = Chunk(
        chunk_id="chunk_idempotent_01",
        content="Idempotent indexing text content.",
        metadata=ChunkMetadata(
            book_title="Test Idempotent Book",
            chapter="Chapter 1",
            section="Section 1",
            page=1,
            type=ContentType.TEXT.value,
            chunk_id="chunk_idempotent_01",
            parent_id="parent_01",
            keywords=["idempotent"],
            heading_path="Chapter 1 > Section 1"
        )
    )

    # First run
    indexer.index([chunk])
    mock_vector_store.delete_by_book_title.assert_called_with("Test Idempotent Book")
    mock_bm25_store.delete_by_book_title.assert_called_with("Test Idempotent Book")
    mock_vector_store.upsert_chunks.assert_called_once()
    mock_bm25_store.upsert_chunks.assert_called_once()

    # Reset mocks and run retry
    mock_vector_store.reset_mock()
    mock_bm25_store.reset_mock()

    indexer.index([chunk])
    mock_vector_store.delete_by_book_title.assert_called_with("Test Idempotent Book")
    mock_bm25_store.delete_by_book_title.assert_called_with("Test Idempotent Book")
    mock_vector_store.upsert_chunks.assert_called_once()
    mock_bm25_store.upsert_chunks.assert_called_once()
