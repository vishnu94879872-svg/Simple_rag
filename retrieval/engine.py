"""Unified retrieval engine combining query rewriting, hybrid search, RRF, reranking, and expansion."""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from retrieval.query_rewriter import QueryRewriter
from retrieval.embeddings import EmbeddingManager
from retrieval.vector_store import ChromaVectorStore
from retrieval.bm25_store import BM25Store
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import BGEReranker
from retrieval.context_expansion import ContextExpander
from retrieval._store_factory import get_vector_store, get_bm25_store

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """Encapsulates the final retrieval results and search diagnostics."""
    original_query: str
    rewritten_query: str
    hits: List[Dict[str, Any]] = Field(default_factory=list)
    vector_candidates_count: int = 0
    bm25_candidates_count: int = 0
    merged_candidates_count: int = 0


class RetrievalEngine:
    """Production-grade hybrid RAG retrieval engine."""

    def __init__(
        self,
        query_rewriter: Optional[QueryRewriter] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_store=None,
        bm25_store=None,
        reranker: Optional[BGEReranker] = None,
        expander: Optional[ContextExpander] = None
    ):
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vector_store = vector_store or get_vector_store()
        self.bm25_store = bm25_store or get_bm25_store()
        self.reranker = reranker or BGEReranker()
        self.expander = expander or ContextExpander()

    def reload_bm25(self) -> None:
        """Reload the underlying BM25 store from disk or cloud."""
        if hasattr(self.bm25_store, "reload"):
            self.bm25_store.reload()

    def get_distinct_books(self) -> List[str]:
        """Return distinct book titles available in vector and sparse stores."""
        books = set()
        if hasattr(self.vector_store, "get_distinct_books"):
            books.update(self.vector_store.get_distinct_books())
        if hasattr(self.bm25_store, "get_distinct_books"):
            books.update(self.bm25_store.get_distinct_books())
        return sorted(list(books))

    def retrieve(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        strategy: str = "reranked",
        tracer: Optional[Any] = None
    ) -> RetrievalResult:
        """
        Execute Phase 3 retrieval pipeline under configured strategy:
        - "vector_only": Dense semantic search only.
        - "bm25_only": Sparse keyword search only.
        - "hybrid_rrf": Dense + Sparse with Reciprocal Rank Fusion without neural reranking.
        - "reranked": Dense + Sparse + RRF + Neural Cross-Encoder Reranker.
        """
        logger.info("Executing retrieval for query: '%s' (strategy: %s, top_k: %d, filters: %s)", query, strategy, top_k, filters)

        # 1. Query processing
        if tracer:
            tracer.start_step("query_processing", {"original_query": query, "history_turns": len(chat_history) if chat_history else 0})

        rewritten_query = self.query_rewriter.rewrite(query, chat_history=chat_history)
        effective_query = rewritten_query or query

        if tracer:
            tracer.finish_step("query_processing", output_summary={"effective_query": effective_query})

        vector_hits: List[Dict[str, Any]] = []
        bm25_hits: List[Dict[str, Any]] = []
        merged_candidates: List[Dict[str, Any]] = []

        # 2. Retrieval stage
        if tracer:
            tracer.start_step("retrieval", {"strategy": strategy, "effective_query": effective_query, "dense_top_k": dense_top_k, "sparse_top_k": sparse_top_k})

        if strategy in ("vector_only", "hybrid_rrf", "reranked"):
            query_emb = self.embedding_manager.embed_query(effective_query)
            vector_hits = self.vector_store.query(
                query_embedding=query_emb,
                top_k=dense_top_k,
                where_filter=filters
            )

        if strategy in ("bm25_only", "hybrid_rrf", "reranked"):
            bm25_hits = self.bm25_store.search(
                query=effective_query,
                top_k=sparse_top_k,
                where_filter=filters
            )

        if strategy in ("hybrid_rrf", "reranked"):
            merged_candidates = reciprocal_rank_fusion(vector_hits, bm25_hits)

        if tracer:
            tracer.finish_step("retrieval", output_summary={
                "dense_hits": len(vector_hits),
                "sparse_hits": len(bm25_hits),
                "merged_candidates": len(merged_candidates)
            })

        # 3. Reranking stage (if used)
        if tracer:
            tracer.start_step("reranking", {"strategy": strategy, "candidates_available": len(merged_candidates)})

        if strategy == "vector_only":
            candidates_to_rank = vector_hits[:top_k]
            if tracer:
                tracer.finish_step("reranking", status="skipped", output_summary={"reason": "Strategy 'vector_only' bypasses neural reranking"})
        elif strategy == "bm25_only":
            candidates_to_rank = bm25_hits[:top_k]
            if tracer:
                tracer.finish_step("reranking", status="skipped", output_summary={"reason": "Strategy 'bm25_only' bypasses neural reranking"})
        elif strategy == "hybrid_rrf":
            candidates_to_rank = merged_candidates[:top_k]
            if tracer:
                tracer.finish_step("reranking", status="skipped", output_summary={"reason": "Strategy 'hybrid_rrf' uses rank fusion without neural reranking"})
        else:
            # Default "reranked": Neural CrossEncoder on top 15 candidates
            rerank_pool = merged_candidates[:15]
            candidates_to_rank = self.reranker.rerank(
                query=effective_query,
                candidates=rerank_pool,
                top_k=top_k
            )
            if tracer:
                tracer.finish_step("reranking", status="success", output_summary={
                    "candidates_scored": len(rerank_pool),
                    "reranked_top_k": len(candidates_to_rank)
                })

        # Context Expansion for short chunks
        final_hits = self.expander.expand_hits(candidates_to_rank)

        result = RetrievalResult(
            original_query=query,
            rewritten_query=effective_query,
            hits=final_hits,
            vector_candidates_count=len(vector_hits),
            bm25_candidates_count=len(bm25_hits),
            merged_candidates_count=len(merged_candidates)
        )
        logger.info("Retrieval returned %d final hits for query: '%s' under strategy '%s'", len(final_hits), query, strategy)
        return result
