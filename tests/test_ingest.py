"""Unit and integration tests for Phase 1 Ingestion."""

import pytest
from pathlib import Path

from config import settings
from ingest.models import ContentType
from ingest.pdf_parser import PDFParser
from ingest.text_splitter import ChunkBuilder
from ingest.parent_store import ParentStore
from ingest.validator import IngestionValidator
from ingest.pipeline import IngestionPipeline
from tests.generate_sample_pdf import create_sample_pdf


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory) -> Path:
    temp_dir = tmp_path_factory.mktemp("books")
    pdf_path = temp_dir / "sample_quantum.pdf"
    create_sample_pdf(pdf_path)
    return pdf_path


def test_pdf_parsing_structure(sample_pdf_path):
    """Test 1: Parsing keeps heading hierarchy and extracts blocks, tables, and images."""
    parser = PDFParser(sample_pdf_path)
    blocks = parser.parse()

    assert len(blocks) > 0
    assert parser.book_title != ""

    # Verify heading hierarchy exists
    chapters = {b.chapter for b in blocks}
    assert any("Chapter 1" in c for c in chapters)
    assert any("Chapter 2" in c for c in chapters)

    # Verify content types detected
    types_found = {b.content_type for b in blocks}
    assert ContentType.TABLE in types_found
    assert ContentType.IMAGE in types_found
    assert ContentType.TEXT in types_found
    assert ContentType.CODE in types_found or ContentType.LIST in types_found


def test_table_and_image_handling(sample_pdf_path):
    """Test 2: Tables are kept as one chunk with summary; images saved and captioned."""
    parser = PDFParser(sample_pdf_path)
    blocks = parser.parse()

    builder = ChunkBuilder(book_title=parser.book_title)
    chunks, parents = builder.build_chunks_and_parents(blocks)

    table_chunks = [c for c in chunks if c.metadata.type == ContentType.TABLE.value]
    image_chunks = [c for c in chunks if c.metadata.type == ContentType.IMAGE.value]

    # Tables: 1-sentence summary at top, never split
    assert len(table_chunks) >= 1
    for tc in table_chunks:
        assert "Table Summary" in tc.content
        assert "|" in tc.content  # Markdown format
        assert tc.metadata.type == "table"

    # Images: saved to data/images, caption chunk created
    assert len(image_chunks) >= 1
    for ic in image_chunks:
        assert "Caption" in ic.content or "Figure" in ic.content
        assert ic.metadata.image_path is not None
        assert Path(ic.metadata.image_path).exists()


def test_text_splitting_and_heading_prefix(sample_pdf_path):
    """Test 3: Text chunks include heading prefix and proper metadata."""
    parser = PDFParser(sample_pdf_path)
    blocks = parser.parse()

    builder = ChunkBuilder(book_title=parser.book_title)
    chunks, parents = builder.build_chunks_and_parents(blocks)

    for chunk in chunks:
        # Every chunk must have heading path prefix
        assert chunk.content.startswith("[")
        # Metadata must have all required fields
        meta = chunk.metadata
        assert meta.book_title != ""
        assert meta.chapter != ""
        assert meta.section != ""
        assert meta.page >= 1
        assert meta.type in [t.value for t in ContentType]
        assert meta.chunk_id != ""
        assert meta.parent_id != ""
        assert isinstance(meta.keywords, list)


def test_parent_store_persistence(sample_pdf_path, tmp_path):
    """Test 4: Parent sections are stored separately and retrievable."""
    parser = PDFParser(sample_pdf_path)
    blocks = parser.parse()

    builder = ChunkBuilder(book_title=parser.book_title)
    chunks, parents = builder.build_chunks_and_parents(blocks)

    store = ParentStore(store_dir=tmp_path / "parents")
    store.save_all(parents)

    assert len(parents) >= 2
    for p in parents:
        loaded = store.get(p.parent_id)
        assert loaded is not None
        assert loaded.parent_id == p.parent_id
        assert loaded.book_title == p.book_title
        assert len(loaded.chunk_ids) > 0


def test_validation_report(sample_pdf_path):
    """Test 5: Validator prints report and calculates metrics correctly."""
    pipeline = IngestionPipeline()
    chunks, parents = pipeline.run(sample_pdf_path, validate=True)

    validator = IngestionValidator()
    report = validator.validate(chunks, sample_count=5)

    assert report["total_chunks"] == len(chunks)
    assert report["empty_chunks_count"] == 0
    assert report["oversized_chunks_count"] == 0
    assert len(report["chunks_by_type"]) > 0
