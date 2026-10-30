"""Typed data models for the BookRAG Reliability Lab."""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import datetime


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class IngestionStageInfo(BaseModel):
    stage_name: str
    status: StageStatus = StageStatus.PENDING
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class SuspiciousCondition(BaseModel):
    condition_type: str  # e.g. empty_page, low_text_density, excessive_repeated_text, scanned_pdf_possible_ocr, missing_table, missing_image
    severity: str = "warning"  # warning | error | info
    page: int = 1
    description: str
    heuristic_rule: str


class IngestionJob(BaseModel):
    job_id: str
    document_id: str
    filename: str
    file_size_bytes: int = 0
    page_count: int = 0
    upload_timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    overall_status: StageStatus = StageStatus.PENDING
    stages: Dict[str, IngestionStageInfo] = Field(default_factory=dict)
    suspicious_conditions: List[SuspiciousCondition] = Field(default_factory=list)
    chunk_count: int = 0
    parent_count: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0


class ChunkQualityIssue(BaseModel):
    issue_type: str  # empty, duplicate, too_short, too_long, broken_sentence, detached_heading, split_table, missing_metadata, excessive_overlap, boilerplate, suspicious_extraction
    severity: str = "warning"
    description: str
    recommendation: str
    evidence: Optional[str] = None
    is_heuristic: bool = True


class ChunkQualityReport(BaseModel):
    chunk_id: str
    char_count: int
    token_count_estimate: int
    overlap_prev_chars: int = 0
    overlap_prev_pct: float = 0.0
    overlap_next_chars: int = 0
    overlap_next_pct: float = 0.0
    issues: List[ChunkQualityIssue] = Field(default_factory=list)
    # Retrieval-aware diagnostics
    retrieval_history: List[Dict[str, Any]] = Field(default_factory=list)
    was_retrieved: bool = False
    best_rank: Optional[int] = None
    best_score: Optional[float] = None
    queries_retrieved_by: List[str] = Field(default_factory=list)
    used_in_final_context: bool = False
    associated_answer_passed_eval: Optional[bool] = None
    source_document: Optional[str] = None
    source_page: Optional[int] = None
    source_section: Optional[str] = None
    causality_assessment: Optional[str] = None


class RetrievalStrategy(str, Enum):
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID_RRF = "hybrid_rrf"
    RERANKED = "reranked"


class RetrievedCandidate(BaseModel):
    chunk_id: str
    rank: int
    score: float
    raw_score: float = 0.0
    score_type: str  # cosine_similarity, bm25_score, rrf_score, cross_encoder_score
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    step_id: str
    name: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    duration_ms: float = 0.0
    status: str = "success"
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class PipelineTrace(BaseModel):
    trace_id: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    query: str
    rewritten_query: str = ""
    strategy: str = RetrievalStrategy.RERANKED.value
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    steps: List[TraceStep] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    prompt_preview: Optional[str] = None
    context_preview: Optional[str] = None
    final_answer: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "success"
    token_usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    estimated_cost_usd: Optional[float] = None
    retrieved_chunk_ids: List[str] = Field(default_factory=list)


class CitationAudit(BaseModel):
    answer_text: str
    detected_citations: List[Dict[str, Any]] = Field(default_factory=list)
    verified_citations: List[Dict[str, Any]] = Field(default_factory=list)
    invalid_citations: List[Dict[str, Any]] = Field(default_factory=list)
    unsupported_citations: List[Dict[str, Any]] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    missing_citations_flag: bool = False
    should_have_abstained: bool = False
    abstained_properly: bool = False
    audit_verdict: str = "verified"  # verified, partially_grounded, unsupported, abstained, invalid_citations


class EvalBenchmarkItem(BaseModel):
    id: str
    question: str
    expected_answer: Optional[str] = None
    expected_document: Optional[str] = None
    expected_pages: List[int] = Field(default_factory=list)
    expected_chunk_ids: List[str] = Field(default_factory=list)
    should_abstain: bool = False
    topic: Optional[str] = None


class EvalRun(BaseModel):
    run_id: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    benchmark_name: str = "Default Benchmark"
    dataset_version: str = "v2.0"
    configuration: Dict[str, Any] = Field(default_factory=dict)
    total_queries: int = 0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    correctness_score: float = 0.0
    groundedness_score: float = 0.0
    citation_accuracy: float = 0.0
    abstention_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    stage_latencies_avg_ms: Dict[str, float] = Field(default_factory=dict)
    error_rate: float = 0.0
    strategy: str = "reranked"
    top_k: int = 5
    per_query_results: List[Dict[str, Any]] = Field(default_factory=list)


class FailureCategory(str, Enum):
    PDF_PARSING = "pdf_parsing"
    OCR_VISUAL = "ocr_visual"
    CHUNKING_METADATA = "chunking_metadata"
    EMBEDDING_INDEXING = "embedding_indexing"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    PROMPT_CONSTRUCTION = "prompt_construction"
    LLM_GENERATION = "llm_generation"
    CITATION = "citation"
    APPLICATION_API = "application_api"


class FailureDiagnosis(BaseModel):
    failure_stage: FailureCategory
    failure_category: Optional[str] = None
    confidence: float = 0.85  # 0.0 to 1.0
    uncertainty: str = "Low"  # Low, Moderate, High
    uncertainty_score: float = 0.15
    observed_evidence: List[str] = Field(default_factory=list)
    supporting_trace_evidence: List[str] = Field(default_factory=list)
    likely_cause: str
    suggested_verification: str = ""
    investigation_steps: List[str] = Field(default_factory=list)
    recommended_fix: str
    confirmation_test: str = ""
    regression_test: str = ""
    before_and_after_eval: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        if not self.failure_category:
            self.failure_category = self.failure_stage.value
        if not self.supporting_trace_evidence and self.observed_evidence:
            self.supporting_trace_evidence = list(self.observed_evidence)
        if not self.investigation_steps and self.suggested_verification:
            self.investigation_steps = [self.suggested_verification]
        if not self.regression_test and self.confirmation_test:
            self.regression_test = self.confirmation_test
        if self.confidence >= 0.8:
            self.uncertainty = "Low (High Confidence)"
            self.uncertainty_score = round(1.0 - self.confidence, 2)
        elif self.confidence >= 0.5:
            self.uncertainty = "Moderate"
            self.uncertainty_score = round(1.0 - self.confidence, 2)
        else:
            self.uncertainty = "High (Inconclusive Evidence)"
            self.uncertainty_score = round(1.0 - self.confidence, 2)



class ChunkExperimentResult(BaseModel):
    experiment_id: str
    strategy_name: str
    chunk_size: int
    chunk_overlap: int
    separators: List[str] = Field(default_factory=list)
    total_chunks: int = 0
    size_distribution: Dict[str, int] = Field(default_factory=dict)
    duplicate_rate: float = 0.0
    processing_duration_ms: float = 0.0
    sample_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    # Reproducible retrieval benchmark measurements
    benchmark_name: Optional[str] = None
    dataset_version: Optional[str] = None
    retrieval_strategy: Optional[str] = None
    top_k: Optional[int] = None
    recall_at_k: Optional[float] = None
    precision_at_k: Optional[float] = None
    mrr: Optional[float] = None
    groundedness_score: Optional[float] = None
    citation_accuracy: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    error_count: int = 0
    is_experimental_index: bool = True


class FailureInvestigationReport(BaseModel):
    """Complete failure investigation record tying query, evidence, root cause, and regression test."""
    investigation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    query: str
    expected_answer: Optional[str] = None
    expected_source_evidence: Optional[Dict[str, Any]] = None
    actual_retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    final_llm_context: str = ""
    generated_answer: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_results: Dict[str, Any] = Field(default_factory=dict)
    likely_failure_stage: FailureCategory
    confidence: float = 0.85
    uncertainty: str = "Low"
    supporting_evidence: List[str] = Field(default_factory=list)
    recommended_fix: str = ""
    regression_test_code: str = ""
    rerun_result: Optional[Dict[str, Any]] = None


class SupportTicket(BaseModel):
    """User-reported issue or feedback ticket."""
    ticket_id: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    category: str  # e.g. "Wrong Answer", "Missing Citation", "Upload/Chunking Error", "Slow Response", "Feedback"
    book_title: Optional[str] = None
    query: Optional[str] = None
    description: str
    user_contact: Optional[str] = None
    status: str = "open"  # open, investigating, resolved
    trace_id: Optional[str] = None

