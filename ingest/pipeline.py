"""Ingestion pipeline orchestrator for BookRAG."""

import logging
from pathlib import Path
from typing import Tuple, List

from ingest.models import Chunk, ParentSection
from ingest.pdf_parser import PDFParser
from ingest.text_splitter import ChunkBuilder
from ingest.parent_store import ParentStore
from ingest.validator import IngestionValidator
from ingest.llm_helper import LLMHelper

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end ingestion pipeline for processing PDF books."""

    def __init__(self, llm_helper: LLMHelper = None, parent_store: ParentStore = None):
        self.llm_helper = llm_helper or LLMHelper()
        self.parent_store = parent_store or ParentStore()
        self.validator = IngestionValidator()

    def run(self, pdf_path: str | Path, validate: bool = True) -> Tuple[List[Chunk], List[ParentSection]]:
        """
        Run Phase 1 ingestion on a PDF file:
        1. Parse pages, tables, images, headings, and code.
        2. Chunk text and process tables/images.
        3. Persist parent sections.
        4. Validate chunk outputs and display statistics.
        """
        pdf_path = Path(pdf_path)
        logger.info("Starting ingestion pipeline for: %s", pdf_path)

        # 1. Parse PDF
        parser = PDFParser(pdf_path)
        blocks = parser.parse()

        # 2. Build Chunks and Parent Sections
        chunk_builder = ChunkBuilder(book_title=parser.book_title, llm_helper=self.llm_helper)
        chunks, parents = chunk_builder.build_chunks_and_parents(blocks)

        # 3. Store Parent Sections separately
        self.parent_store.save_all(parents)
        logger.info("Saved %d parent sections to store", len(parents))

        # 4. Run Validation
        if validate:
            self.validator.validate(chunks)

        return chunks, parents
