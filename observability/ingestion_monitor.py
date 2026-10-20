"""Document Ingestion Monitor with suspicious condition detection and PDF extraction verification."""

import time
import re
import uuid
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pymupdf

from config import settings
from ingest.models import ParsedBlock, Chunk, ParentSection, ContentType
from ingest.pipeline import IngestionPipeline
from retrieval.indexer import BookIndexer
from observability.models import (
    IngestionJob,
    IngestionStageInfo,
    SuspiciousCondition,
    StageStatus,
)
from observability.storage import storage

logger = logging.getLogger(__name__)


class IngestionMonitor:
    """Orchestrates document ingestion with stage-by-stage diagnostics, accounting, and heuristic checks."""

    def __init__(self):
        self.storage = storage

    def inspect_pdf_structure(self, pdf_path: Path) -> Tuple[int, List[SuspiciousCondition], Dict[str, Any]]:
        """
        Inspect raw PDF pages before or during ingestion to verify extraction and detect suspicious conditions:
        - Compares physical pages vs successfully extracted pages
        - Scanned pages (raster images without extractable text) with actionable OCR recommendation
        - Empty or sparse pages (blank margins, cover art)
        - Vector graphics detection so diagrams are not falsely claimed as missing
        - Strict evidence verification before flagging missing tables or figures
        """
        doc = pymupdf.open(str(pdf_path))
        page_count = len(doc)
        conditions: List[SuspiciousCondition] = []

        page_texts: List[str] = []
        page_lines_set: List[List[str]] = []
        extracted_pages_count = 0

        for idx in range(page_count):
            page_num = idx + 1
            page = doc[idx]
            text = page.get_text().strip()
            page_texts.append(text)
            page_lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 10]
            page_lines_set.append(page_lines)

            images = page.get_images()
            try:
                drawings = page.get_drawings()
            except Exception:
                drawings = []

            # Page extraction tracking
            if len(text) > 0 or len(images) > 0 or len(drawings) > 0:
                extracted_pages_count += 1

            # 1. Zero-text pages
            if len(text) == 0:
                if images:
                    conditions.append(
                        SuspiciousCondition(
                            condition_type="scanned_pdf_possible_ocr",
                            severity="warning",
                            page=page_num,
                            description=(
                                f"Page {page_num} contains 0 extractable text characters but contains {len(images)} raster image(s). "
                                "OCR Recommendation: Run Tesseract or a Vision LLM to extract text from this scanned page."
                            ),
                            heuristic_rule="Character count == 0 and page has raster images"
                        )
                    )
                elif drawings:
                    conditions.append(
                        SuspiciousCondition(
                            condition_type="vector_graphic_page",
                            severity="info",
                            page=page_num,
                            description=(
                                f"Page {page_num} has 0 extractable text characters but contains {len(drawings)} vector path element(s) "
                                "(e.g., full-page vector chart, diagram, or graphic divider)."
                            ),
                            heuristic_rule="Character count == 0 and page has vector drawings"
                        )
                    )
                else:
                    conditions.append(
                        SuspiciousCondition(
                            condition_type="empty_page",
                            severity="info",
                            page=page_num,
                            description=f"Page {page_num} contains no extracted text, images, or vector drawings (blank page/margin).",
                            heuristic_rule="Page character count == 0, image count == 0, drawing count == 0"
                        )
                    )

            # 2. Suspiciously short extraction (< 40 characters)
            elif len(text) < 40:
                if images:
                    conditions.append(
                        SuspiciousCondition(
                            condition_type="scanned_pdf_possible_ocr",
                            severity="warning",
                            page=page_num,
                            description=(
                                f"Page {page_num} has suspiciously short text ({len(text)} characters) with {len(images)} raster image(s). "
                                "OCR Recommendation: Run OCR to recover possible text embedded in images."
                            ),
                            heuristic_rule="Character count < 40 and page has raster images"
                        )
                    )
                else:
                    conditions.append(
                        SuspiciousCondition(
                            condition_type="low_text_density",
                            severity="info",
                            page=page_num,
                            description=f"Page {page_num} has sparse text ({len(text)} characters). Likely title, header, or brief section transition.",
                            heuristic_rule="Character count < 40 without raster images"
                        )
                    )

            # 3. High replacement/broken characters
            broken_char_count = text.count("\ufffd")
            if broken_char_count > 3:
                conditions.append(
                    SuspiciousCondition(
                        condition_type="suspicious_extraction",
                        severity="warning",
                        page=page_num,
                        description=f"Page {page_num} contains {broken_char_count} unicode replacement characters (\\ufffd), indicating garbled font encoding.",
                        heuristic_rule="Count of unicode replacement characters > 3"
                    )
                )

            # 4. Tables mentioned in text - requires SUFFICIENT EVIDENCE before claiming missing
            lower_text = text.lower()
            # Only check if explicit forward/local reference on this page
            if re.search(r"\b(as shown in the (?:following|below) table|table below)\b", lower_text):
                try:
                    tabs = page.find_tables()
                    has_table_borders = bool(tabs and len(tabs.tables) > 0)
                    has_pipe_lines = any("|" in l for l in text.splitlines())
                    has_vector_lines = len(drawings) >= 4  # tables often drawn with vector borders
                    
                    if not has_table_borders and not has_pipe_lines and not has_vector_lines and not images:
                        conditions.append(
                            SuspiciousCondition(
                                condition_type="missing_table_suspected",
                                severity="info",
                                page=page_num,
                                description=(
                                    f"Page {page_num} explicitly references a table ('table below'), but no table borders, "
                                    "pipe delimiters, or vector line borders were detected on this page."
                                ),
                                heuristic_rule="Explicit inline reference to table below with 0 table borders, pipes, or drawings"
                            )
                        )
                except Exception:
                    pass

            # 5. Figures/diagrams mentioned in text - requires SUFFICIENT EVIDENCE before claiming missing
            if re.search(r"\b(shown in the diagram below|illustrated in the chart below)\b", lower_text):
                has_raster = bool(images)
                has_vector_art = len(drawings) > 0
                if not has_raster and not has_vector_art:
                    conditions.append(
                        SuspiciousCondition(
                            condition_type="missing_image_suspected",
                            severity="info",
                            page=page_num,
                            description=(
                                f"Page {page_num} mentions 'diagram/chart below', but no embedded raster images "
                                "or vector drawing elements were found on this page."
                            ),
                            heuristic_rule="Explicit reference to diagram below with 0 raster images and 0 vector drawings"
                        )
                    )

        # 6. Excessive repeated text across multiple pages (boilerplate header/footer check)
        all_lines: Dict[str, int] = {}
        for lines in page_lines_set:
            for l in set(lines):
                all_lines[l] = all_lines.get(l, 0) + 1

        for line, count in all_lines.items():
            if count >= max(3, int(page_count * 0.7)) and page_count > 3:
                conditions.append(
                    SuspiciousCondition(
                        condition_type="excessive_repeated_text",
                        severity="info",
                        page=1,
                        description=f"Repetitive text line found across {count}/{page_count} pages: '{line[:60]}...'",
                        heuristic_rule="Line appears on >70% of document pages"
                    )
                )

        doc.close()

        accounting_summary = {
            "physical_pages": page_count,
            "successfully_extracted_pages": extracted_pages_count,
            "empty_or_sparse_pages": page_count - extracted_pages_count,
            "explanation": (
                "Physical Pages: Total page count in original PDF file. "
                "Successfully Extracted Pages: Pages providing readable text, images, or vector diagrams. "
                "Indexed Chunks: Total searchable retrieval units after recursive text splitting."
            )
        }

        return page_count, conditions, accounting_summary

    def run_monitored_ingestion(
        self,
        pdf_path: Path,
        job_id: Optional[str] = None
    ) -> IngestionJob:
        """Execute full ingestion pipeline with stage tracking, diagnostics, and persistence."""
        pdf_path = Path(pdf_path)
        job_id = job_id or f"job_{uuid.uuid4().hex[:8]}"
        doc_id = f"doc_{pdf_path.stem[:20]}_{int(pdf_path.stat().st_size)}"

        job = IngestionJob(
            job_id=job_id,
            document_id=doc_id,
            filename=pdf_path.name,
            file_size_bytes=pdf_path.stat().st_size,
            overall_status=StageStatus.RUNNING
        )
        self.storage.save_job(job)

        stage_start_times: Dict[str, float] = {}

        try:
            # Stage 0: Pre-flight structural inspection & extraction verification
            logger.info("[%s] Inspecting PDF structure...", job_id)
            t_struct = time.time()
            stage_start_times["structural_inspection"] = t_struct
            page_count, conditions, accounting = self.inspect_pdf_structure(pdf_path)
            struct_dur = round(max(0.1, (time.time() - t_struct) * 1000.0), 2)
            job.page_count = page_count
            job.suspicious_conditions = conditions
            job.stages["structural_inspection"] = IngestionStageInfo(
                stage_name="structural_inspection",
                status=StageStatus.COMPLETED,
                duration_ms=struct_dur,
                details=accounting
            )
            self.storage.save_job(job)

            # Stage 1: Parsing
            t0 = time.time()
            stage_start_times["parsing"] = t0
            job.stages["parsing"] = IngestionStageInfo(stage_name="parsing", status=StageStatus.RUNNING)
            self.storage.save_job(job)

            pipeline = IngestionPipeline()
            from ingest.pdf_parser import PDFParser
            parser = PDFParser(pdf_path)
            blocks = parser.parse()
            parse_dur = round((time.time() - t0) * 1000.0, 2)
            job.stages["parsing"] = IngestionStageInfo(
                stage_name="parsing",
                status=StageStatus.COMPLETED,
                duration_ms=parse_dur,
                details={
                    "parsed_blocks": len(blocks),
                    "book_title": parser.book_title,
                    "physical_pages": page_count,
                    "successfully_extracted_pages": accounting["successfully_extracted_pages"]
                }
            )
            self.storage.save_job(job)

            # Stage 2: Chunking & Parent Building
            t0 = time.time()
            stage_start_times["chunking"] = t0
            job.stages["chunking"] = IngestionStageInfo(stage_name="chunking", status=StageStatus.RUNNING)
            self.storage.save_job(job)

            from ingest.text_splitter import ChunkBuilder
            from ingest.parent_store import ParentStore
            from ingest.validator import IngestionValidator

            builder = ChunkBuilder(book_title=parser.book_title)
            chunks, parents = builder.build_chunks_and_parents(blocks)
            job.chunk_count = len(chunks)
            job.parent_count = len(parents)

            parent_store = ParentStore()
            parent_store.save_all(parents)

            validator = IngestionValidator(chunks, parents)
            is_valid, validation_report = validator.validate()

            chunk_dur = round((time.time() - t0) * 1000.0, 2)
            job.stages["chunking"] = IngestionStageInfo(
                stage_name="chunking",
                status=StageStatus.COMPLETED if is_valid else StageStatus.WARNING,
                duration_ms=chunk_dur,
                details={
                    "chunks": len(chunks),
                    "parents": len(parents),
                    "is_valid": is_valid,
                    "validation_report": validation_report,
                    "indexed_chunks_to_physical_pages_ratio": round(len(chunks) / max(1, page_count), 2)
                }
            )
            self.storage.save_job(job)

            # Stage 3: Embedding & Indexing
            t0 = time.time()
            stage_start_times["embedding_and_indexing"] = t0
            job.stages["embedding_and_indexing"] = IngestionStageInfo(stage_name="embedding_and_indexing", status=StageStatus.RUNNING)
            self.storage.save_job(job)

            indexer = BookIndexer()
            index_summary = indexer.index(chunks)
            index_dur = round((time.time() - t0) * 1000.0, 2)

            job.stages["embedding_and_indexing"] = IngestionStageInfo(
                stage_name="embedding_and_indexing",
                status=StageStatus.COMPLETED,
                duration_ms=index_dur,
                details=index_summary
            )

            # Determine overall status
            has_warnings = any(c.severity == "warning" for c in conditions) or not is_valid
            job.overall_status = StageStatus.WARNING if has_warnings else StageStatus.COMPLETED
            logger.info("[%s] Ingestion completed with status %s", job_id, job.overall_status)

        except Exception as e:
            logger.error("[%s] Ingestion failed: %s", job_id, e, exc_info=True)
            job.overall_status = StageStatus.FAILED
            job.error_message = str(e)
            now = time.time()
            if "chunks" in locals() and chunks:
                job.chunk_count = len(chunks)
            if "parents" in locals() and parents:
                job.parent_count = len(parents)

            for sname in ["structural_inspection", "parsing", "chunking", "embedding_and_indexing"]:
                if sname in job.stages and job.stages[sname].status == StageStatus.RUNNING:
                    t_start = stage_start_times.get(sname, now)
                    elapsed_ms = round(max(0.1, (now - t_start) * 1000.0), 2)
                    job.stages[sname].status = StageStatus.FAILED
                    job.stages[sname].duration_ms = elapsed_ms
                    job.stages[sname].error_message = str(e)
                    job.stages[sname].details = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "chunks_at_failure": len(chunks) if "chunks" in locals() and chunks else 0,
                        "parents_at_failure": len(parents) if "parents" in locals() and parents else 0
                    }

        self.storage.save_job(job)
        return job


ingestion_monitor = IngestionMonitor()
