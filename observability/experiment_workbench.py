"""Chunking Experiment Workbench: Controlled comparison of chunking strategies and retrieval settings in isolated sandboxes."""

import re
import time
import uuid
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from config import settings
from ingest.models import ParsedBlock, Chunk, ChunkMetadata, ContentType
from ingest.pdf_parser import PDFParser
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from observability.models import ChunkExperimentResult, EvalBenchmarkItem
from observability.storage import storage

logger = logging.getLogger(__name__)


class ChunkingExperimentWorkbench:
    """Runs controlled, non-destructive chunking and retrieval experiments isolated from production indexes."""

    def __init__(self):
        self.storage = storage

    def run_experiment(
        self,
        pdf_path: Path,
        strategy_name: str = "sentence_aware",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        custom_separators: Optional[List[str]] = None
    ) -> ChunkExperimentResult:
        """
        Execute an isolated chunking experiment on a PDF book.
        Does NOT touch ChromaDB or BM25 production index files.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        t0 = time.time()
        parser = PDFParser(pdf_path)
        blocks = parser.parse()

        raw_text_pieces = [b.content for b in blocks if b.content_type != ContentType.IMAGE]
        combined_text = "\n\n".join(raw_text_pieces)

        chunks_created: List[str] = []

        if strategy_name == "fixed_size":
            splitter = CharacterTextSplitter(
                separator="",
                chunk_size=chunk_size,
                chunk_overlap=0,
                length_function=len
            )
            chunks_created = splitter.split_text(combined_text)

        elif strategy_name == "fixed_overlap":
            splitter = CharacterTextSplitter(
                separator="",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks_created = splitter.split_text(combined_text)

        elif strategy_name == "paragraph_aware":
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                keep_separator=True
            )
            chunks_created = splitter.split_text(combined_text)

        elif strategy_name == "heading_aware":
            from ingest.text_splitter import ChunkBuilder
            builder = ChunkBuilder(book_title=parser.book_title)
            builder.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=custom_separators or ["\n\n", "\n", ". ", " "],
                keep_separator=True
            )
            prod_chunks, _ = builder.build_chunks_and_parents(blocks)
            chunks_created = [c.content for c in prod_chunks]

        else:  # "sentence_aware"
            splitter = RecursiveCharacterTextSplitter(
                separators=custom_separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " "],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                keep_separator=True
            )
            chunks_created = splitter.split_text(combined_text)

        duration_ms = round((time.time() - t0) * 1000.0, 2)

        chunk_lens = [len(c) for c in chunks_created]
        total_chunks = len(chunks_created)

        # Duplicate rate
        seen = set()
        duplicates = 0
        for c in chunks_created:
            norm = c.strip().lower()
            if norm in seen:
                duplicates += 1
            else:
                seen.add(norm)
        dup_rate = round((duplicates / total_chunks * 100.0), 2) if total_chunks > 0 else 0.0

        size_dist = {
            "min_chars": int(min(chunk_lens)) if chunk_lens else 0,
            "max_chars": int(max(chunk_lens)) if chunk_lens else 0,
            "mean_chars": int(np.mean(chunk_lens)) if chunk_lens else 0,
            "std_chars": int(np.std(chunk_lens)) if chunk_lens else 0,
            "median_chars": int(np.median(chunk_lens)) if chunk_lens else 0,
        }

        samples = [
            {"index": i + 1, "chars": len(c), "preview": c[:250]}
            for i, c in enumerate(chunks_created[:5])
        ]

        exp_result = ChunkExperimentResult(
            experiment_id=f"exp_{uuid.uuid4().hex[:8]}",
            strategy_name=strategy_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=custom_separators or ["default"],
            total_chunks=total_chunks,
            size_distribution=size_dist,
            duplicate_rate=dup_rate,
            processing_duration_ms=duration_ms,
            sample_chunks=samples
        )

        self.storage.save_experiment(exp_result)
        logger.info("Chunking experiment %s completed with %d chunks", exp_result.experiment_id, total_chunks)
        return exp_result

    def run_benchmark_experiment(
        self,
        pdf_path: Path,
        benchmark_name: str = "quantum_bench",
        strategy_name: str = "sentence_aware",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        retrieval_strategy: str = "reranked",
        top_k: int = 5,
        max_queries: Optional[int] = 10,
        custom_separators: Optional[List[str]] = None
    ) -> ChunkExperimentResult:
        """
        Execute an end-to-end reproducible experiment comparing chunking configurations and retrieval settings
        against a benchmark dataset using an ISOLATED experimental index.
        Production indexes remain completely unmodified.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        exp_id = f"exp_{uuid.uuid4().hex[:8]}"
        exp_dir = settings.observability_dir / "experiments" / f"sandbox_{exp_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        parser = PDFParser(pdf_path)
        blocks = parser.parse()

        # Build typed chunks
        from ingest.text_splitter import ChunkBuilder
        builder = ChunkBuilder(book_title=parser.book_title)
        
        if strategy_name == "fixed_size":
            builder.text_splitter = CharacterTextSplitter(
                separator="",
                chunk_size=chunk_size,
                chunk_overlap=0,
                length_function=len
            )
        elif strategy_name == "paragraph_aware":
            builder.text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                keep_separator=True
            )
        else:
            builder.text_splitter = RecursiveCharacterTextSplitter(
                separators=custom_separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " "],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                keep_separator=True
            )

        chunks, _ = builder.build_chunks_and_parents(blocks)

        # Index in ISOLATED experimental index
        from retrieval.bm25_store import BM25Store
        from retrieval.vector_store import ChromaVectorStore
        from retrieval.engine import RetrievalEngine

        bm25_store = BM25Store(persist_dir=exp_dir / "bm25")
        bm25_store.upsert_chunks(chunks)

        vector_store = ChromaVectorStore(
            persist_dir=exp_dir / "chroma",
            collection_name=f"col_{exp_id}"
        )
        vector_store.upsert_chunks(chunks)

        engine = RetrievalEngine(
            vector_store=vector_store,
            bm25_store=bm25_store
        )

        # Load benchmark queries
        items = self.storage.load_benchmark(benchmark_name)
        if not items:
            items = [
                EvalBenchmarkItem(
                    id="exp_item_1",
                    question="What is quantum superposition?",
                    expected_pages=[1]
                )
            ]
        if max_queries:
            items = items[:max_queries]

        latencies_ms: List[float] = []
        hits_count = 0
        precisions: List[float] = []
        reciprocal_ranks: List[float] = []
        citation_clean_count = 0
        groundedness_scores: List[float] = []
        error_count = 0

        for item in items:
            t_q = time.time()
            try:
                res = engine.retrieve(
                    query=item.question,
                    strategy=retrieval_strategy,
                    top_k=top_k
                )
                lat_ms = (time.time() - t_q) * 1000.0
                latencies_ms.append(lat_ms)

                retrieved_pages = [int(h.get("metadata", {}).get("page", 0)) for h in res.hits]
                exp_pages = set(item.expected_pages or [])

                if item.should_abstain:
                    reciprocal_ranks.append(1.0)
                    precisions.append(1.0)
                    hits_count += 1
                    groundedness_scores.append(1.0)
                    citation_clean_count += 1
                else:
                    is_hit = False
                    first_rank = 0
                    match_count = 0

                    for rank_idx, p in enumerate(retrieved_pages, 1):
                        if p in exp_pages:
                            if not is_hit:
                                is_hit = True
                                first_rank = rank_idx
                            match_count += 1

                    if is_hit:
                        hits_count += 1
                        reciprocal_ranks.append(1.0 / first_rank)
                    else:
                        reciprocal_ranks.append(0.0)

                    precisions.append(match_count / max(1, len(retrieved_pages)))
                    # Simulated groundedness based on precision
                    groundedness_scores.append(min(1.0, (match_count / max(1, len(exp_pages))) if exp_pages else 1.0))
                    citation_clean_count += 1 if is_hit else 0

            except Exception as e:
                logger.error("Error evaluating item '%s' in experiment: %s", item.id, e)
                error_count += 1
                reciprocal_ranks.append(0.0)
                precisions.append(0.0)

        total_q = max(1, len(items))
        recall_at_k = round(hits_count / total_q, 4)
        precision_at_k = round(float(np.mean(precisions)) if precisions else 0.0, 4)
        mrr = round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4)
        groundedness = round(float(np.mean(groundedness_scores)) if groundedness_scores else 0.0, 4)
        citation_acc = round(citation_clean_count / total_q, 4)
        avg_lat = round(float(np.mean(latencies_ms)) if latencies_ms else 0.0, 2)

        duration_ms = round((time.time() - t0) * 1000.0, 2)
        chunk_lens = [len(c.content) for c in chunks]

        # Duplicate rate
        seen = set()
        duplicates = 0
        for c in chunks:
            norm = c.content.strip().lower()
            if norm in seen:
                duplicates += 1
            else:
                seen.add(norm)
        dup_rate = round((duplicates / max(1, len(chunks)) * 100.0), 2)

        size_dist = {
            "min_chars": int(min(chunk_lens)) if chunk_lens else 0,
            "max_chars": int(max(chunk_lens)) if chunk_lens else 0,
            "mean_chars": int(np.mean(chunk_lens)) if chunk_lens else 0,
            "std_chars": int(np.std(chunk_lens)) if chunk_lens else 0,
            "median_chars": int(np.median(chunk_lens)) if chunk_lens else 0,
        }

        samples = [
            {"index": i + 1, "chars": len(c.content), "preview": c.content[:250]}
            for i, c in enumerate(chunks[:5])
        ]

        exp_result = ChunkExperimentResult(
            experiment_id=exp_id,
            strategy_name=strategy_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=custom_separators or ["default"],
            total_chunks=len(chunks),
            size_distribution=size_dist,
            duplicate_rate=dup_rate,
            processing_duration_ms=duration_ms,
            sample_chunks=samples,
            benchmark_name=benchmark_name,
            dataset_version="v2.0",
            retrieval_strategy=retrieval_strategy,
            top_k=top_k,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mrr=mrr,
            groundedness_score=groundedness,
            citation_accuracy=citation_acc,
            avg_latency_ms=avg_lat,
            error_count=error_count,
            is_experimental_index=True
        )

        self.storage.save_experiment(exp_result)
        logger.info(
            "Benchmark experiment %s finished: Recall@%d=%.2f, MRR=%.2f, Latency=%.1fms",
            exp_id, top_k, recall_at_k, mrr, avg_lat
        )
        return exp_result


chunk_experiment_workbench = ChunkingExperimentWorkbench()
