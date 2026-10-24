"""End-to-end QA Pipeline connecting hybrid retrieval with answer generation."""

import logging
from typing import List, Dict, Any, Optional

from retrieval.engine import RetrievalEngine, RetrievalResult
from generation.generator import AnswerGenerator, AnswerResponse

logger = logging.getLogger(__name__)


class BookRAGPipeline:
    """High-level interface for querying the ingested book."""

    def __init__(
        self,
        retrieval_engine: Optional[RetrievalEngine] = None,
        generator: Optional[AnswerGenerator] = None
    ):
        self.retrieval_engine = retrieval_engine or RetrievalEngine()
        self.generator = generator or AnswerGenerator()

    def reload_bm25(self) -> None:
        """Reload BM25 store to reflect newly ingested documents."""
        self.retrieval_engine.reload_bm25()

    def get_distinct_books(self) -> List[str]:
        """Fetch all indexed book titles across stores."""
        return self.retrieval_engine.get_distinct_books()

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        strategy: str = "reranked",
        trace_id: Optional[str] = None,
        enable_tracing: bool = True,
        log_prompts: bool = True
    ) -> AnswerResponse:
        """
        Execute full RAG pipeline:
        1. Query rewriting with chat history.
        2. Configurable retrieval (Vector, BM25, Hybrid RRF, Neural Reranker, Parent Expansion).
        3. Prompt construction with expert tutor persona.
        4. LLM generation with strictly grounded answer and citations.
        5. Citation auditing and execution trace persistence.
        """
        logger.info("BookRAG Pipeline received query: '%s' (strategy: %s, top_k: %d)", question, strategy, top_k)

        tracer = None
        if enable_tracing:
            from observability.tracer import PipelineTracer
            tracer = PipelineTracer(
                query=question,
                trace_id=trace_id,
                strategy=strategy,
                top_k=top_k,
                filters=filters,
                log_prompts=log_prompts
            )
            # Stage 1: Request validation
            tracer.start_step("request_validation", {
                "question_length": len(question or ""),
                "has_filters": bool(filters),
                "strategy": strategy,
                "top_k": top_k
            })
            if not question or not question.strip():
                tracer.finish_step("request_validation", status="failed", error="Empty question string provided")
                raise ValueError("Question cannot be empty.")
            tracer.finish_step("request_validation", status="success", output_summary={"validation": "passed", "sanitized_query_length": len(question.strip())})

        # Stages 2, 3, 4: Query processing, Retrieval, and Reranking
        retrieval_result: RetrievalResult = self.retrieval_engine.retrieve(
            query=question,
            chat_history=chat_history,
            filters=filters,
            top_k=top_k,
            strategy=strategy,
            tracer=tracer
        )

        # Stages 5, 6, 7: Context construction, LLM generation, Citation verification
        response: AnswerResponse = self.generator.generate(
            question=question,
            hits=retrieval_result.hits,
            chat_history=chat_history,
            rewritten_query=retrieval_result.rewritten_query,
            tracer=tracer
        )

        # Stage 8: Response delivery
        if tracer:
            tracer.start_step("response_delivery", {
                "answer_length": len(response.answer),
                "sources_count": len(response.sources)
            })
            tracer.finish_step("response_delivery", status="success", output_summary={
                "delivered": True,
                "answer_preview": response.answer[:80],
                "sources_count": len(response.sources)
            })
            tracer.finalize(
                final_answer=response.answer,
                citations=[s.model_dump() for s in response.sources],
                retrieved_chunk_ids=[h.get("chunk_id", "") for h in response.retrieved_chunks],
                rewritten_query=response.rewritten_query,
                status="success"
            )

        return response
