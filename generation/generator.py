"""Answer generator module executing LLM synthesis with strict context adherence."""

import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from config import settings
from ingest.llm_helper import LLMHelper
from generation.prompt_templates import SYSTEM_PROMPT, format_context_from_hits, build_user_prompt

logger = logging.getLogger(__name__)


class SourceCitation(BaseModel):
    """Citation referencing a chapter and page in the book."""
    chapter: str
    page: int
    type: str
    chunk_id: str


class AnswerResponse(BaseModel):
    """Structured response containing the tutor answer, citations, and chunks."""
    question: str
    rewritten_query: str = ""
    answer: str
    sources: List[SourceCitation] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    trace_id: Optional[str] = None
    audit: Optional[Any] = None


class AnswerGenerator:
    """Generates tutor-style answers using retrieved context and configurable LLM."""

    def __init__(self, llm_helper: Optional[LLMHelper] = None):
        self.llm_helper = llm_helper or LLMHelper()

    def generate(
        self,
        question: str,
        hits: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        rewritten_query: str = "",
        tracer: Optional[Any] = None
    ) -> AnswerResponse:
        """
        Generate answer based strictly on retrieved hits.
        Uses the required expert tutor system prompt.
        """
        from observability.citation_verifier import citation_verifier

        # If no hits retrieved, immediately return standard not found message
        if not hits:
            answer_text = "I couldn't find that in this book."
            audit = citation_verifier.audit_response(
                question=question,
                answer=answer_text,
                retrieved_chunks=[],
                should_abstain=True
            )
            if tracer:
                tracer.finalize(
                    final_answer=answer_text,
                    citations=[],
                    retrieved_chunk_ids=[],
                    rewritten_query=rewritten_query,
                    status="success"
                )
            return AnswerResponse(
                question=question,
                rewritten_query=rewritten_query,
                answer=answer_text,
                sources=[],
                retrieved_chunks=[],
                trace_id=tracer.trace_id if tracer else None,
                audit=audit
            )

        if tracer:
            tracer.start_step("context_construction", {"chunks_count": len(hits)})

        context_text = format_context_from_hits(hits)
        user_prompt = build_user_prompt(question, context_text)

        if tracer:
            tracer.set_prompt_and_context(user_prompt, context_text)
            tracer.finish_step("context_construction", output_summary={"context_length": len(context_text)})

        # Collect citations from hits
        sources: List[SourceCitation] = []
        seen_sources = set()
        for h in hits:
            meta = h.get("metadata", {})
            chap = meta.get("chapter", "Unknown Chapter")
            page = int(meta.get("page", 1))
            cid = h.get("chunk_id", "")
            ctype = meta.get("type", "text")
            src_key = (chap, page)
            if src_key not in seen_sources:
                seen_sources.add(src_key)
                sources.append(SourceCitation(chapter=chap, page=page, type=ctype, chunk_id=cid))

        answer_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        # Attempt generation via LLM API if available
        if self.llm_helper and self.llm_helper._client:
            if tracer:
                tracer.start_step("llm_generation", {"model": settings.llm_model, "provider": settings.llm_provider})
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                if chat_history:
                    for turn in chat_history[-4:]:
                        messages.append({
                            "role": turn.get("role", "user"),
                            "content": turn.get("content", "")
                        })
                messages.append({"role": "user", "content": user_prompt})

                response = self.llm_helper._client.chat.completions.create(
                    model=settings.llm_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=600
                )
                answer_text = response.choices[0].message.content.strip()
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = response.usage.prompt_tokens or 0
                    completion_tokens = response.usage.completion_tokens or 0
                if tracer:
                    tracer.set_token_usage(prompt_tokens, completion_tokens)
                    tracer.finish_step("llm_generation", output_summary={"answer_length": len(answer_text)})
            except Exception as e:
                logger.warning("LLM generation API error: %s. Using deterministic fallback synthesis.", e)
                if tracer:
                    tracer.finish_step("llm_generation", status="failed", error=str(e))

        # Fallback generator if LLM API is offline or key is unset
        if not answer_text:
            if tracer:
                tracer.start_step("llm_generation", {"mode": "offline_fallback_synthesis"})
            answer_text = self._fallback_synthesis(question, hits, sources)
            # Estimate token usage
            prompt_tokens = len(user_prompt.split()) * 2
            completion_tokens = len(answer_text.split()) * 2
            if tracer:
                tracer.set_token_usage(prompt_tokens, completion_tokens)
                tracer.finish_step("llm_generation", output_summary={"answer_length": len(answer_text), "mode": "deterministic_fallback"})

        # Citation Verification & Auditing
        if tracer:
            tracer.start_step("citation_verification", {"citations_detected": len(sources)})
        audit = citation_verifier.audit_response(
            question=question,
            answer=answer_text,
            retrieved_chunks=hits
        )
        if tracer:
            tracer.finish_step(
                "citation_verification",
                output_summary={
                    "audit_verdict": audit.audit_verdict,
                    "verified_citations": len(audit.verified_citations),
                    "unsupported_citations": len(audit.unsupported_citations)
                }
            )

        return AnswerResponse(
            question=question,
            rewritten_query=rewritten_query,
            answer=answer_text,
            sources=sources,
            retrieved_chunks=hits,
            trace_id=tracer.trace_id if tracer else None,
            audit=audit
        )

    def _fallback_synthesis(
        self,
        question: str,
        hits: List[Dict[str, Any]],
        sources: List[SourceCitation]
    ) -> str:
        """
        Rule-based tutor response generator conforming to the system prompt
        when running without an active LLM API key.
        """
        q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
        matched_pieces = []
        table_piece = ""
        image_piece = ""

        for hit in hits:
            meta = hit.get("metadata", {})
            c_type = meta.get("type")
            content = hit.get("content", "")

            # Check for table
            if c_type == "table" and not table_piece:
                table_piece = f"\n\nHere is the relevant data table from {meta.get('chapter', 'the book')}:\n{content}"

            # Check for image
            if c_type == "image" and not image_piece:
                image_piece = f"\n\nAs shown in the image on page {meta.get('page')}: {content}"

            # Check text matches
            h_words = set(re.findall(r"\b\w{3,}\b", content.lower()))
            overlap = q_words.intersection(h_words) or any(
                qw in hw or hw in qw for qw in q_words for hw in h_words
            )
            if overlap:
                clean_lines = [l for l in content.splitlines() if not l.startswith("[") and l.strip()]
                if clean_lines:
                    matched_pieces.append(" ".join(clean_lines[:2]))

        if not matched_pieces and not table_piece and not image_piece:
            return "I couldn't find that in this book."

        # Natural synthesis
        intro = f"Based on the book, "
        body = " ".join(matched_pieces[:2]) if matched_pieces else ""
        if table_piece:
            body += table_piece
        if image_piece:
            body += image_piece

        # Format sources
        source_str = ", ".join(f"({s.chapter}, Page {s.page})" for s in sources[:3])
        return f"{intro}{body}\n\nSources: {source_str}"
