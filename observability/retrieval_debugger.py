"""Retrieval Playground and strategy debugger."""

import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from retrieval.engine import RetrievalEngine
from observability.models import RetrievedCandidate, RetrievalStrategy
from observability.tracer import PipelineTracer

logger = logging.getLogger(__name__)


class StrategyComparisonResult(BaseModel):
    """Encapsulates a side-by-side comparison between two retrieval runs."""
    query: str
    strategy_a: str
    strategy_b: str
    top_k: int
    hits_a: List[RetrievedCandidate]
    hits_b: List[RetrievedCandidate]
    duration_a_ms: float
    duration_b_ms: float
    common_chunk_ids: List[str]
    only_in_a_chunk_ids: List[str]
    only_in_b_chunk_ids: List[str]


class RetrievalDebugger:
    """Provides developer-level inspection and comparative benchmarking across retrieval strategies."""

    def __init__(self, engine: Optional[RetrievalEngine] = None):
        self.engine = engine or RetrievalEngine()

    def run_query(
        self,
        query: str,
        strategy: str = RetrievalStrategy.RERANKED.value,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        tracer: Optional[PipelineTracer] = None
    ) -> Dict[str, Any]:
        """Execute retrieval under a specified strategy and format diagnostic candidates."""
        t0 = time.time()
        res = self.engine.retrieve(
            query=query,
            chat_history=chat_history,
            filters=filters,
            top_k=top_k,
            strategy=strategy,
            tracer=tracer
        )
        duration_ms = round((time.time() - t0) * 1000.0, 2)

        candidates: List[RetrievedCandidate] = []
        for rank, hit in enumerate(res.hits, 1):
            score = 0.0
            score_type = "unknown"
            if "rerank_score" in hit:
                score = float(hit["rerank_score"])
                score_type = "cross_encoder_score"
            elif "rrf_score" in hit:
                score = float(hit["rrf_score"])
                score_type = "rrf_score"
            elif "bm25_score" in hit:
                score = float(hit["bm25_score"])
                score_type = "bm25_score"
            elif "score" in hit:
                score = float(hit["score"])
                score_type = "cosine_similarity"

            candidates.append(
                RetrievedCandidate(
                    chunk_id=hit.get("chunk_id", f"c_{rank}"),
                    rank=rank,
                    score=round(score, 5),
                    raw_score=score,
                    score_type=score_type,
                    content=hit.get("content", ""),
                    metadata=hit.get("metadata", {})
                )
            )

        return {
            "original_query": res.original_query,
            "rewritten_query": res.rewritten_query,
            "strategy": strategy,
            "top_k": top_k,
            "duration_ms": duration_ms,
            "candidates": candidates,
            "vector_candidates_count": res.vector_candidates_count,
            "bm25_candidates_count": res.bm25_candidates_count,
            "merged_candidates_count": res.merged_candidates_count
        }

    def compare_strategies(
        self,
        query: str,
        strategy_a: str = RetrievalStrategy.VECTOR_ONLY.value,
        strategy_b: str = RetrievalStrategy.RERANKED.value,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> StrategyComparisonResult:
        """Run two strategies and generate a diff of retrieved chunks and performance."""
        res_a = self.run_query(query=query, strategy=strategy_a, top_k=top_k, filters=filters)
        res_b = self.run_query(query=query, strategy=strategy_b, top_k=top_k, filters=filters)

        chunks_a = [c.chunk_id for c in res_a["candidates"]]
        chunks_b = [c.chunk_id for c in res_b["candidates"]]

        set_a = set(chunks_a)
        set_b = set(chunks_b)

        common = [cid for cid in chunks_a if cid in set_b]
        only_a = [cid for cid in chunks_a if cid not in set_b]
        only_b = [cid for cid in chunks_b if cid not in set_a]

        return StrategyComparisonResult(
            query=query,
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            top_k=top_k,
            hits_a=res_a["candidates"],
            hits_b=res_b["candidates"],
            duration_a_ms=res_a["duration_ms"],
            duration_b_ms=res_b["duration_ms"],
            common_chunk_ids=common,
            only_in_a_chunk_ids=only_a,
            only_in_b_chunk_ids=only_b
        )


retrieval_debugger = RetrievalDebugger()
