"""CLI entrypoint for ingesting PDF books into BookRAG."""

import sys
import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler
from ingest.pipeline import IngestionPipeline
from retrieval.indexer import BookIndexer

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("bookrag.ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF book into BookRAG.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF book file.")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation report.")
    parser.add_argument("--no-index", action="store_true", help="Skip Chroma and BM25 indexing.")
    args = parser.parse_args()

    pdf_file = Path(args.pdf_path)
    if not pdf_file.exists():
        logger.error("File not found: %s", pdf_file)
        sys.exit(1)

    pipeline = IngestionPipeline()
    chunks, parents = pipeline.run(pdf_file, validate=not args.no_validate)
    logger.info("Ingestion completed: %d chunks generated across %d parent sections.", len(chunks), len(parents))

    if not args.no_index:
        logger.info("Starting indexing into ChromaDB and BM25...")
        indexer = BookIndexer()
        index_summary = indexer.index(chunks)
        logger.info(
            "Indexing finished: Chroma count = %d, BM25 count = %d",
            index_summary["chroma_total_count"],
            index_summary["bm25_total_count"]
        )


if __name__ == "__main__":
    main()
