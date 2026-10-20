"""Authoritative telemetry provider querying actual backend databases and files."""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pymupdf

from config import settings
from observability.storage import storage
from observability.models import IngestionJob, PipelineTrace, EvalRun

logger = logging.getLogger(__name__)


class AuthoritativeTelemetry:
    """Calculates ground-truth metrics directly from backend storage, Chroma, and PDF files."""

    @staticmethod
    def get_chroma_page_and_book_stats() -> Dict[str, Any]:
        """Query ChromaDB SQLite database directly for distinct indexed pages and books."""
        db_path = settings.chroma_persist_dir / "chroma.sqlite3"
        if not db_path.exists():
            return {
                "distinct_indexed_pages": 0,
                "books": {},
                "total_chunks": 0
            }

        try:
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            # Query distinct (book, page) combinations
            cur.execute("""
                SELECT m1.string_value, count(distinct m2.int_value), min(m2.int_value), max(m2.int_value), count(m1.id)
                FROM embedding_metadata m1
                JOIN embedding_metadata m2 ON m1.id = m2.id
                WHERE m1.key = 'book_title' AND m2.key = 'page'
                GROUP BY m1.string_value
            """)
            rows = cur.fetchall()

            # Distinct pages total
            cur.execute("""
                SELECT count(distinct m1.string_value || '_' || m2.int_value)
                FROM embedding_metadata m1
                JOIN embedding_metadata m2 ON m1.id = m2.id
                WHERE m1.key = 'book_title' AND m2.key = 'page'
            """)
            distinct_pages_total = cur.fetchone()[0] or 0

            # Total chunks
            cur.execute("SELECT count(*) FROM embeddings")
            total_chunks = cur.fetchone()[0] or 0

            conn.close()

            books_stats = {}
            for row in rows:
                title, page_count, min_p, max_p, chunk_count = row
                books_stats[title] = {
                    "distinct_pages": page_count,
                    "page_range": (min_p, max_p),
                    "chunks": chunk_count
                }

            return {
                "distinct_indexed_pages": distinct_pages_total,
                "books": books_stats,
                "total_chunks": total_chunks
            }
        except Exception as e:
            logger.error("Failed to query Chroma SQLite stats: %s", e)
            return {
                "distinct_indexed_pages": 0,
                "books": {},
                "total_chunks": 0,
                "error": str(e)
            }

    @staticmethod
    def get_physical_pdf_stats() -> Dict[str, Any]:
        """Inspect physical PDF files in data/books/."""
        books_dir = settings.data_dir / "books"
        if not books_dir.exists():
            return {"file_count": 0, "total_pages": 0, "files": []}

        files_info = []
        total_pages = 0
        for pdf_path in books_dir.glob("*.pdf"):
            try:
                doc = pymupdf.open(str(pdf_path))
                p_count = len(doc)
                doc.close()
                total_pages += p_count
                files_info.append({
                    "filename": pdf_path.name,
                    "pages": p_count,
                    "size_bytes": pdf_path.stat().st_size
                })
            except Exception as e:
                logger.warning("Could not read PDF %s: %s", pdf_path, e)
                files_info.append({
                    "filename": pdf_path.name,
                    "pages": 0,
                    "size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
                    "error": str(e)
                })

        return {
            "file_count": len(files_info),
            "total_pages": total_pages,
            "files": files_info
        }

    @classmethod
    def get_dashboard_metrics(cls) -> Dict[str, Any]:
        """Aggregate authoritative dashboard metrics with sample-size awareness."""
        chroma_stats = cls.get_chroma_page_and_book_stats()
        pdf_stats = cls.get_physical_pdf_stats()

        jobs = storage.list_jobs(limit=100)
        traces = storage.list_traces(limit=200)
        eval_runs = storage.list_eval_runs(limit=50)

        # Ingestion jobs breakdown
        total_jobs = len(jobs)
        active_jobs = sum(1 for j in jobs if j.overall_status == "running")
        completed_jobs = sum(1 for j in jobs if j.overall_status == "completed")
        warning_jobs = sum(1 for j in jobs if j.overall_status == "warning")
        failed_jobs = sum(1 for j in jobs if j.overall_status == "failed")

        # Job delta explanation
        if total_jobs == 0:
            jobs_delta = f"0 monitored runs ({pdf_stats['file_count']} pre-indexed)"
            jobs_delta_color = "off"
        elif failed_jobs > 0:
            jobs_delta = f"{failed_jobs} failed of {total_jobs}"
            jobs_delta_color = "inverse"
        else:
            jobs_delta = f"{completed_jobs + warning_jobs}/{total_jobs} succeeded"
            jobs_delta_color = "normal"

        # Authoritative page calculation:
        # If monitored jobs exist, count job pages; also cross-reference with Chroma indexed pages
        job_pages = sum(j.page_count for j in jobs)
        distinct_indexed_pages = chroma_stats["distinct_indexed_pages"]
        physical_pdf_pages = pdf_stats["total_pages"]

        # Latency statistics with sample-size awareness
        durations = [t.total_duration_ms for t in traces if t.total_duration_ms > 0]
        n_samples = len(durations)

        if n_samples == 0:
            avg_latency_str = "Unavailable"
            avg_latency_note = "0 recorded query measurements"
            p95_latency_str = "Unavailable"
            p95_latency_note = "Requires ≥ 5 samples (n = 0)"
        elif n_samples < 5:
            mean_val = float(np.mean(durations))
            avg_latency_str = f"{mean_val:.0f} ms"
            avg_latency_note = f"Preliminary (n = {n_samples})"
            p95_latency_str = "Unavailable"
            p95_latency_note = f"Requires ≥ 5 samples (n = {n_samples})"
        else:
            mean_val = float(np.mean(durations))
            p95_val = float(np.percentile(durations, 95))
            avg_latency_str = f"{mean_val:.0f} ms"
            avg_latency_note = f"n = {n_samples}" if n_samples >= 20 else f"Preliminary (n = {n_samples})"
            p95_latency_str = f"{p95_val:.0f} ms"
            p95_latency_note = f"n = {n_samples}" if n_samples >= 20 else f"Preliminary (n = {n_samples})"

        # Retrieval failure rate
        failed_retrieval_count = sum(1 for t in traces if len(t.retrieved_chunk_ids) == 0)
        retrieval_failure_rate = (failed_retrieval_count / len(traces) * 100.0) if traces else 0.0

        return {
            "indexed_documents_count": pdf_stats["file_count"],
            "physical_pdf_pages": physical_pdf_pages,
            "distinct_indexed_pages": distinct_indexed_pages,
            "job_pages_recorded": job_pages,
            "total_chunks": chroma_stats["total_chunks"],
            "books_breakdown": chroma_stats["books"],
            "pdf_files": pdf_stats["files"],
            "total_jobs": total_jobs,
            "active_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "warning_jobs": warning_jobs,
            "failed_jobs": failed_jobs,
            "jobs_delta": jobs_delta,
            "jobs_delta_color": jobs_delta_color,
            "questions_processed": len(traces),
            "retrieval_failure_rate": retrieval_failure_rate,
            "failed_retrieval_count": failed_retrieval_count,
            "avg_latency_str": avg_latency_str,
            "avg_latency_note": avg_latency_note,
            "p95_latency_str": p95_latency_str,
            "p95_latency_note": p95_latency_note,
            "sample_count": n_samples,
            "eval_runs_count": len(eval_runs),
            "recent_traces": traces[:15],
            "recent_eval_runs": eval_runs[:15]
        }


authoritative_telemetry = AuthoritativeTelemetry()
