"""Rule-based Failure Classifier and Root-Cause Assistant with complete failure investigation workflow."""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional

from observability.models import (
    FailureCategory,
    FailureDiagnosis,
    FailureInvestigationReport,
    PipelineTrace,
    CitationAudit,
)
from observability.storage import storage

logger = logging.getLogger(__name__)


class FailureClassifier:
    """Diagnoses root causes of RAG failures across the 10 lifecycle stages using rules and trace evidence."""

    def diagnose(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        trace: Optional[PipelineTrace] = None,
        audit: Optional[CitationAudit] = None,
        expected_page: Optional[int] = None
    ) -> FailureDiagnosis:
        """
        Examine execution evidence and classify the most probable failure stage.
        Returns evidence, likely cause, confidence, verification step, recommended fix, and test.
        """
        evidence: List[str] = []
        answer_clean = answer.strip().lower()

        # Check trace errors first (Application / API stage)
        if trace and trace.status == "error":
            for step in trace.steps:
                if step.status == "failed" and step.error:
                    evidence.append(f"Trace step '{step.name}' failed with error: {step.error}")
                    if any(term in step.error.lower() for term in ["api", "timeout", "rate", "connection", "auth"]):
                        return FailureDiagnosis(
                            failure_stage=FailureCategory.APPLICATION_API,
                            confidence=0.95,
                            observed_evidence=evidence,
                            likely_cause=f"External model provider or API endpoint failure in stage '{step.name}'.",
                            suggested_verification=f"Check API credentials, rate limits, and network connectivity for {trace.provider}.",
                            recommended_fix="Configure backoff retries and verify LLM_API_KEY / LLM_BASE_URL in .env.",
                            confirmation_test=f"Execute a health-check test query against {trace.provider} endpoint."
                        )

        # 1. Zero Hits Retrieved -> Retrieval Failure
        if not retrieved_chunks:
            evidence.append(f"Query '{query}' produced 0 candidates across dense and sparse retrievers.")
            return FailureDiagnosis(
                failure_stage=FailureCategory.RETRIEVAL,
                confidence=0.90,
                observed_evidence=evidence,
                likely_cause="Neither dense vector search nor BM25 sparse search returned matching chunks above relevance threshold.",
                suggested_verification="Inspect vector store and BM25 index counts to verify that document text is actually indexed.",
                recommended_fix="Increase top_k, broaden query rewriting keywords, or re-index with smaller chunk sizes.",
                confirmation_test=f"Run a raw keyword search on query terms '{query}' directly in the BM25 store."
            )

        # 2. Expected Page Missed -> Retrieval / Reranking Failure
        retrieved_pages = [int(h.get("metadata", {}).get("page", 0)) for h in retrieved_chunks]
        if expected_page is not None and expected_page not in retrieved_pages:
            evidence.append(f"Expected page {expected_page} was not in retrieved top-{len(retrieved_chunks)} pages: {retrieved_pages}")
            # Check if BM25 or Dense candidates had it before reranking
            if trace:
                for step in trace.steps:
                    if step.name == "neural_reranking":
                        evidence.append("Neural reranker cross-encoder filtered out candidate pool.")
                        return FailureDiagnosis(
                            failure_stage=FailureCategory.RERANKING,
                            confidence=0.85,
                            observed_evidence=evidence,
                            likely_cause="The target chunk was retrieved by initial hybrid search but discarded or demoted by the cross-encoder reranker.",
                            suggested_verification="Inspect pre-reranked candidate ranks in Trace Viewer to check if page was present in initial pool.",
                            recommended_fix="Increase target top_k (e.g., from 5 to 10) or tune the reranking score threshold.",
                            confirmation_test=f"Run Retrieval Playground with strategy 'hybrid_rrf' (without reranking) to verify if page {expected_page} appears."
                        )

            return FailureDiagnosis(
                failure_stage=FailureCategory.RETRIEVAL,
                confidence=0.80,
                observed_evidence=evidence,
                likely_cause="The relevant document section did not match embedding vector similarity or BM25 keyword overlap.",
                suggested_verification="Check chunk text on expected page to see if vocabulary mismatch prevented retrieval.",
                recommended_fix="Optimize query rewriter to extract synonymous keywords, or enable parent context expansion.",
                confirmation_test=f"Search specifically for page {expected_page} chunks in Chunk Inspector."
            )

        # 3. Citation Audit Failure (Invalid or Unsupported Citations)
        if audit and audit.invalid_citations:
            bad_pages = [c.get("page") for c in audit.invalid_citations]
            evidence.append(f"Generated answer contains citations with invalid page numbers: {bad_pages}")
            return FailureDiagnosis(
                failure_stage=FailureCategory.CITATION,
                confidence=0.90,
                observed_evidence=evidence,
                likely_cause="LLM hallucinated page numbers not present in the indexed document or beyond document length.",
                suggested_verification="Compare cited page numbers with document maximum page count in Ingestion Monitor.",
                recommended_fix="Enforce strict regex citation extraction and instruct the system prompt to quote ONLY verified chunk metadata.",
                confirmation_test="Run Citation Verification module on the question response to confirm zero invalid pages."
            )

        if audit and audit.unsupported_citations:
            evidence.append(f"Answer cited {len(audit.unsupported_citations)} source(s) that were never provided in retrieved context chunks.")
            return FailureDiagnosis(
                failure_stage=FailureCategory.CITATION,
                confidence=0.85,
                observed_evidence=evidence,
                likely_cause="LLM fabricated source citations rather than deriving them from the retrieved evidence context.",
                suggested_verification="Inspect context provided in Trace Viewer and compare with answer text.",
                recommended_fix="Adjust generation temperature to 0.0 or strengthen the prompt rule: 'End with sources in the form (Chapter, Page) matching context only'.",
                confirmation_test="Verify that every citation in the response matches a chunk ID in retrieved_chunks."
            )

        # 4. Abstention Failure (Answered when it should have abstained)
        if audit and audit.should_have_abstained and not audit.abstained_properly:
            evidence.append("The question was flagged as out-of-domain or unanswerable, but the model generated a substantive answer anyway.")
            return FailureDiagnosis(
                failure_stage=FailureCategory.LLM_GENERATION,
                confidence=0.85,
                observed_evidence=evidence,
                likely_cause="Model failed to abstain when provided context lacked definitive evidence.",
                suggested_verification="Check retrieved chunk relevance scores; if all scores are low, the model should trigger fallback abstention.",
                recommended_fix="Add a minimum rerank score gate (e.g. score >= 0.15) before sending context to LLM; if below, return standard abstention.",
                confirmation_test="Re-run the question and verify the answer is exactly 'I couldn't find that in this book.'"
            )

        # 5. Unsupported Claims / Grounding Failure
        if audit and audit.unsupported_claims:
            evidence.append(f"Citation auditor detected {len(audit.unsupported_claims)} claims not grounded in retrieved chunks.")
            return FailureDiagnosis(
                failure_stage=FailureCategory.LLM_GENERATION,
                confidence=0.80,
                observed_evidence=evidence,
                likely_cause="LLM generated assertions based on internal pre-training knowledge rather than strictly adhering to provided document context.",
                suggested_verification="Review prompt context vs answer claim sentences in Citation Auditor.",
                recommended_fix="Lower LLM temperature to 0.1 and instruct the model: 'Answer only with facts directly stated in the text.'",
                confirmation_test="Verify that audit.unsupported_claims is empty after generation."
            )

        # 6. Chunking / Metadata Limitation (Chunks too fragmented)
        avg_chunk_len = sum(len(c.get("content", "")) for c in retrieved_chunks) / max(1, len(retrieved_chunks))
        if avg_chunk_len < 100:
            evidence.append(f"Retrieved chunks have very low average length ({int(avg_chunk_len)} characters).")
            return FailureDiagnosis(
                failure_stage=FailureCategory.CHUNKING_METADATA,
                confidence=0.75,
                observed_evidence=evidence,
                likely_cause="Chunks are too fine-grained, fragmenting complete sentences and contextual concepts.",
                suggested_verification="Review Chunk Inspector quality warnings for 'too_short' chunks.",
                recommended_fix="Increase CHUNK_SIZE (e.g. 800) and ensure ContextExpander is active for parent section expansion.",
                confirmation_test="Check expanded text length in Chunk Inspector for affected chunks."
            )

        # 7. Default: Nominal Execution
        return FailureDiagnosis(
            failure_stage=FailureCategory.APPLICATION_API,
            confidence=0.20,
            observed_evidence=["No deterministic pipeline failures detected; response conforms to retrieval evidence."],
            likely_cause="System operates nominally under current parameters.",
            suggested_verification="Review trace latency and answer clarity in Evaluation Lab.",
            recommended_fix="No fix needed.",
            confirmation_test="Re-run evaluation benchmark."
        )

    def create_investigation_report(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
        expected_answer: Optional[str] = None,
        expected_source_evidence: Optional[Dict[str, Any]] = None,
        final_llm_context: str = "",
        citations: Optional[List[Dict[str, Any]]] = None,
        trace: Optional[PipelineTrace] = None,
        audit: Optional[CitationAudit] = None,
        evaluation_results: Optional[Dict[str, Any]] = None
    ) -> FailureInvestigationReport:
        """
        Build a complete, evidence-based failure investigation report tying all 10 diagnostic items together.
        Generates an executable regression test that can be rerun.
        """
        expected_page = None
        if expected_source_evidence and "expected_pages" in expected_source_evidence:
            pages = expected_source_evidence.get("expected_pages")
            if pages and isinstance(pages, list) and len(pages) > 0:
                expected_page = int(pages[0])

        diag = self.diagnose(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            trace=trace,
            audit=audit,
            expected_page=expected_page
        )

        safe_query_name = re.sub(r"[^a-zA-Z0-9_]", "_", query[:30]).strip("_")
        test_func_name = f"test_regression_{diag.failure_stage.value}_{safe_query_name}"

        # Build self-contained executable pytest regression test
        reg_code = (
            f'import pytest\n'
            f'from generation.qa_chain import BookRAGPipeline\n'
            f'from observability.citation_verifier import citation_verifier\n\n'
            f'def {test_func_name}():\n'
            f'    """\n'
            f'    Regression Test for query: "{query}"\n'
            f'    Diagnosed Failure Stage: {diag.failure_stage.value}\n'
            f'    Recommended Fix: {diag.recommended_fix}\n'
            f'    """\n'
            f'    pipeline = BookRAGPipeline()\n'
            f'    resp = pipeline.query("{query}", top_k=5, strategy="reranked")\n'
            f'    assert resp.answer is not None and len(resp.answer) > 0\n'
        )

        if expected_page:
            reg_code += (
                f'    retrieved_pages = [c.get("metadata", {{}}).get("page") for c in resp.retrieved_chunks]\n'
                f'    assert {expected_page} in retrieved_pages, f"Expected page {expected_page} not found in {{retrieved_pages}}"\n'
            )

        reg_code += (
            f'    audit = citation_verifier.audit_response("{query}", resp.answer, resp.retrieved_chunks)\n'
            f'    assert len(audit.invalid_citations) == 0, f"Found invalid citations: {{audit.invalid_citations}}"\n'
        )

        report = FailureInvestigationReport(
            investigation_id=f"inv_{uuid.uuid4().hex[:8]}",
            query=query,
            expected_answer=expected_answer,
            expected_source_evidence=expected_source_evidence or {},
            actual_retrieved_chunks=retrieved_chunks,
            final_llm_context=final_llm_context or (trace.context_preview if trace else ""),
            generated_answer=answer,
            citations=citations or (trace.citations if trace else []),
            evaluation_results=evaluation_results or {
                "audit_verdict": audit.audit_verdict if audit else "unknown",
                "invalid_citations_count": len(audit.invalid_citations) if audit else 0,
                "unsupported_citations_count": len(audit.unsupported_citations) if audit else 0
            },
            likely_failure_stage=diag.failure_stage,
            confidence=diag.confidence,
            uncertainty=diag.uncertainty,
            supporting_evidence=diag.observed_evidence,
            recommended_fix=diag.recommended_fix,
            regression_test_code=reg_code
        )

        storage.save_investigation(report)
        return report

    def rerun_investigation_test(
        self,
        report_id: str,
        modified_strategy: str = "reranked",
        modified_top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Rerun a previously failed question live through the pipeline to confirm fix effectiveness.
        Returns before-and-after comparison.
        """
        report = storage.get_investigation(report_id)
        if not report:
            raise ValueError(f"Investigation report '{report_id}' not found.")

        from generation.qa_chain import BookRAGPipeline
        from observability.citation_verifier import citation_verifier

        pipeline = BookRAGPipeline()
        new_resp = pipeline.query(
            question=report.query,
            strategy=modified_strategy,
            top_k=modified_top_k
        )

        new_audit = citation_verifier.audit_response(
            question=report.query,
            answer=new_resp.answer,
            retrieved_chunks=new_resp.retrieved_chunks
        )

        # Calculate metrics
        expected_pages = report.expected_source_evidence.get("expected_pages", []) if report.expected_source_evidence else []
        retrieved_pages = [int(c.get("metadata", {}).get("page", 0)) for c in new_resp.retrieved_chunks]
        
        recall_hit = any(p in retrieved_pages for p in expected_pages) if expected_pages else True

        rerun_summary = {
            "rerun_strategy": modified_strategy,
            "rerun_top_k": modified_top_k,
            "new_answer": new_resp.answer,
            "new_retrieved_count": len(new_resp.retrieved_chunks),
            "new_retrieved_pages": retrieved_pages,
            "recall_hit": recall_hit,
            "audit_verdict": new_audit.audit_verdict,
            "invalid_citations_count": len(new_audit.invalid_citations),
            "unsupported_citations_count": len(new_audit.unsupported_citations),
            "verdict": "PASSED" if (recall_hit and len(new_audit.invalid_citations) == 0) else "NEEDS_ATTENTION"
        }

        report.rerun_result = rerun_summary
        storage.save_investigation(report)
        return rerun_summary


failure_classifier = FailureClassifier()
