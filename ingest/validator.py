import re
import sys
import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import settings
from ingest.models import Chunk

logger = logging.getLogger(__name__)

# Initialize console with safe fallback
console = Console(highlight=False, soft_wrap=True)


def sanitize_for_console(text: str) -> str:
    """Normalize problematic unicode characters that cause charmap encoding errors on Windows."""
    if not text:
        return ""
    # Normalize unicode spaces and quotes
    text = re.sub(r"[\u2000-\u200b\u00a0\u3000\ufeff]", " ", text)
    text = re.sub(r"[\u2018\u2019`´]", "'", text)
    text = re.sub(r"[\u201c\u201d«»]", '"', text)
    text = re.sub(r"[\u2013\u2014―]", "-", text)
    text = re.sub(r"[\u2022\u25cf\u25e6\u2219]", "*", text)
    
    encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding)
    except Exception:
        return text.encode("ascii", errors="replace").decode("ascii")


class ValidationResult(tuple):
    """
    A 2-tuple (is_valid, report_dict) that also provides dictionary-style
    access to the report fields for backward compatibility.
    """

    def __new__(cls, is_valid: bool, report: Dict[str, Any]):
        return super().__new__(cls, (is_valid, report))

    def __init__(self, is_valid: bool, report: Dict[str, Any]):
        self.is_valid = is_valid
        self.report = report

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)
        return self.report[key]

    def __contains__(self, key):
        return key in self.report

    def get(self, key, default=None):
        return self.report.get(key, default)

    def keys(self):
        return self.report.keys()

    def values(self):
        return self.report.values()

    def items(self):
        return self.report.items()


class IngestionValidator:
    """Validates processed chunks and outputs comprehensive diagnostic reports."""

    def __init__(
        self,
        chunks_or_max_size: Any = None,
        parents: Optional[List[Any]] = None,
        max_allowed_size: Optional[int] = None
    ):
        """
        Accepts either:
        - IngestionValidator(chunks, parents)
        - IngestionValidator(max_allowed_size=1200)
        - IngestionValidator(chunks)
        - IngestionValidator()
        """
        if isinstance(chunks_or_max_size, (int, float)):
            self.max_allowed_size = int(chunks_or_max_size)
            self.chunks: Optional[List[Chunk]] = None
            self.parents: Optional[List[Any]] = parents
        elif isinstance(chunks_or_max_size, list):
            self.chunks = chunks_or_max_size
            self.parents = parents
            self.max_allowed_size = max_allowed_size or (settings.chunk_size + 400)
        else:
            self.chunks = None
            self.parents = parents
            self.max_allowed_size = max_allowed_size or (settings.chunk_size + 400)

    def validate(self, chunks: Optional[List[Chunk]] = None, sample_count: int = 5) -> ValidationResult:
        """
        Run validation checks:
        1. Total chunks
        2. Chunks per type
        3. Empty chunks
        4. Chunks over max size (excluding full tables)
        5. Print random samples for review
        """
        target_chunks = chunks if chunks is not None else (self.chunks or [])
        total_chunks = len(target_chunks)
        chunks_by_type: Dict[str, int] = {}
        empty_chunks: List[str] = []
        oversized_chunks: List[Tuple[str, int]] = []

        for chunk in target_chunks:
            c_type = chunk.metadata.type
            chunks_by_type[c_type] = chunks_by_type.get(c_type, 0) + 1

            if not chunk.content or not chunk.content.strip():
                empty_chunks.append(chunk.chunk_id)

            content_len = len(chunk.content)
            if c_type != "table" and content_len > self.max_allowed_size:
                oversized_chunks.append((chunk.chunk_id, content_len))

        try:
            # Pretty print validation report
            console.print("\n" + "=" * 60)
            console.print("[bold cyan]BookRAG Ingestion Validation Report[/bold cyan]")
            console.print("=" * 60)

            # Overview Table
            table = Table(title="Summary Metrics", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="dim", width=30)
            table.add_column("Value", justify="right")

            table.add_row("Total Chunks", str(total_chunks))
            for c_type, count in sorted(chunks_by_type.items()):
                table.add_row(f"Chunks ({c_type})", str(count))
            table.add_row("Empty Chunks", f"[red]{len(empty_chunks)}[/red]" if empty_chunks else "[green]0[/green]")
            table.add_row(
                f"Oversized Non-Table Chunks (>{self.max_allowed_size} chars)",
                f"[yellow]{len(oversized_chunks)}[/yellow]" if oversized_chunks else "[green]0[/green]"
            )

            console.print(table)

            if empty_chunks:
                console.print(f"[bold red]WARNING:[/bold red] Found {len(empty_chunks)} empty chunks: {empty_chunks[:5]}")

            # Pick random samples for review
            sample_size = min(sample_count, total_chunks)
            if sample_size > 0:
                console.print(f"\n[bold yellow]{sample_size} Random Samples for Ingestion Review:[/bold yellow]")
                samples = random.sample(target_chunks, sample_size)
                for i, sample in enumerate(samples, 1):
                    preview = sanitize_for_console(sample.content.strip())
                    if len(preview) > 300:
                        preview = preview[:297] + "..."

                    raw_keywords = [sanitize_for_console(k) for k in sample.metadata.keywords[:8]]
                    keywords_str = ", ".join(raw_keywords) or "None"
                    heading = sanitize_for_console(sample.metadata.heading_path)

                    info = (
                        f"[bold]ID:[/bold] {sample.chunk_id}\n"
                        f"[bold]Type:[/bold] {sample.metadata.type} | [bold]Page:[/bold] {sample.metadata.page}\n"
                        f"[bold]Heading:[/bold] {heading}\n"
                        f"[bold]Keywords:[/bold] {keywords_str}\n"
                        f"[bold]Content Preview:[/bold]\n{preview}"
                    )
                    console.print(Panel(info, title=f"Sample {i}/{sample_size}", border_style="blue"))
        except Exception as e:
            logger.warning("Console output formatting error in validator: %s. Continuing gracefully.", e)

        is_valid = len(empty_chunks) == 0 and total_chunks > 0
        report = {
            "is_valid": is_valid,
            "total_chunks": total_chunks,
            "chunks_by_type": chunks_by_type,
            "empty_chunks_count": len(empty_chunks),
            "empty_chunk_ids": empty_chunks,
            "oversized_chunks_count": len(oversized_chunks),
            "oversized_chunk_ids": oversized_chunks
        }
        return ValidationResult(is_valid, report)
