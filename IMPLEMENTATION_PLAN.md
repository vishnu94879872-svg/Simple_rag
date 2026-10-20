# Implementation Plan: BookRAG Reliability Lab

## Objective
Upgrade the existing **BookRAG** application into the **BookRAG Reliability Lab** — a production-oriented RAG debugging, evaluation, and observability platform that diagnoses every stage of the RAG pipeline (PDF ingestion → parsing → chunking → embedding → indexing → retrieval → reranking → prompt construction → LLM generation → citation verification → evaluation).

---

## 1. Baseline Assessment & Architecture Preservation
- **Existing Frameworks & Services Preserved:**
  - PDF Parsing: PyMuPDF (`fitz`), `pymupdf4llm`, PIL
  - Vector Store: `ChromaDB` (384-dimensional embeddings via `BAAI/bge-small-en-v1.5`)
  - Sparse Index: `rank_bm25` (pickled to `data/bm25/`)
  - Reranking: `BAAI/bge-reranker-base` via `sentence_transformers.CrossEncoder`
  - Generation: OpenAI / LLM client with deterministic fallback synthesis for offline testing
  - User Interface: Streamlit web application
  - Test Suite: 19 passing unit & integration tests covering ingestion, indexing, retrieval, and generation.

---

## 2. Phased Implementation Roadmap

### Phase 1: Observability Core & Pipeline Instrumentation
- **Data Models & Storage (`observability/models.py`, `observability/storage.py`):**
  - Typed models for `IngestionJob`, `DocumentDiagnostics`, `ChunkQualityReport`, `PipelineTrace`, `TraceStep`, `CitationAudit`, `EvalRun`, `FailureDiagnosis`, `ChunkExperiment`.
  - Thread-safe persistent storage under `data/observability/` for real metrics, traces, jobs, and evaluations.
- **Correlation ID & Pipeline Tracer (`observability/tracer.py`):**
  - Request-level correlation ID (`trace_id`) attaching timestamped stages: request received, query processing, dense retrieval, sparse retrieval, RRF fusion, neural reranking, parent expansion, prompt construction, LLM generation, citation verification.
  - Secret masking (redacting API keys, bearer tokens) and configurable privacy policy for prompt logging.
- **Instrument Engine & QA Chain (`retrieval/engine.py`, `generation/qa_chain.py`):**
  - Add execution hooks to trace durations, inputs, outputs, and candidate counts without altering search results.

### Phase 2: Ingestion Diagnostics & Chunk Inspector (Modules B & C)
- **Ingestion Monitoring (`observability/ingestion_monitor.py`):**
  - Track document ID, filename, file size, page count, upload time, duration per stage (parsing, chunking, embedding, indexing), and job status.
  - Suspicious document heuristics: empty pages, low text density (<50 chars), scanned PDF / OCR indicators, excessive repeated text, missing tables/images.
- **Chunk Quality Checker (`observability/chunk_inspector.py`):**
  - Diagnostic heuristics: empty chunks, duplicate chunks, excessively short/long chunks, broken sentences, heading detached from body, split tables/lists, metadata missing, excessive overlap (>40%), boilerplate patterns.
- **Two-Panel PDF & Chunk Inspector UI:**
  - Left panel: PDF page viewer / text preview with page selector.
  - Right panel: extracted text, chunk list with badges, token/char counts, overlap preview, embedding text, and non-destructive quality warning badges.
  - Fix horizontal text overflow with responsive CSS card wrapping (`word-break: break-word`, `overflow-x: auto`).

### Phase 3: Retrieval Playground & Trace Viewer (Modules D & E)
- **Retrieval Playground (`observability/retrieval_debugger.py`):**
  - Multi-strategy retrieval comparator: Dense Vector Only, BM25 Keyword Only, Hybrid RRF, Reranked.
  - Top-k variation (k = 3, 5, 10) with comparative diff view showing chunks added or dropped between configurations.
  - Score distinction: Vector cosine distance, BM25 raw score, RRF reciprocal rank score, Cross-Encoder reranker score.
- **Trace Viewer UI:**
  - Interactive trace explorer by correlation ID.
  - Per-stage breakdown: timestamps, durations (ms), status, inputs/outputs, model parameters, token counts, and privacy toggles.

### Phase 4: Citation Verification & Evaluation Lab (Modules F & G)
- **Citation & Answer Verification (`observability/citation_verifier.py`):**
  - Parse citations `(Chapter, Page)` from answer text.
  - Audit against retrieved chunks: verify existence of cited chapters/pages, highlight unsupported claims, detect invalid pages, check for missing citations, and evaluate whether the system should have abstained.
- **RAG Evaluation Lab (`observability/eval_lab.py`):**
  - Benchmark test runner with test dataset import/export (JSON & CSV).
  - Metrics: Retrieval Recall@k, Precision@k, MRR; Generation correctness, groundedness, citation accuracy, abstention quality; System latency and cost estimates.
  - Historical evaluation run storage with side-by-side run comparison.

### Phase 5: Root-Cause Assistant, Experiment Workbench & Security (Modules H, I, 6)
- **Failure Classification & Root-Cause Assistant (`observability/failure_classifier.py`):**
  - Rule-based root-cause diagnosis across all 10 stages: PDF parsing, OCR limitation, chunking/metadata, embedding/indexing, retrieval, reranker, prompt construction, LLM generation, citation failure, system API error.
  - Actionable diagnostics: observed evidence, likely cause, confidence, verification step, recommended fix, test to confirm.
- **Chunking Experiment Workbench (`observability/experiment_workbench.py`):**
  - Isolated temporary indexes to test strategies (fixed size, fixed + overlap, sentence-aware, paragraph-aware, heading-aware) without corrupting production indexes.
- **Security & Settings:**
  - File validation, API key masking, prompt injection defense in context wrappers, and confirmation dialogs for destructive actions (reindexing/deleting).

### Phase 6: Automated Testing & Verification
- Unit and integration tests for tracing, citation verification, chunk quality checking, failure classification, and retrieval strategies.
- Verify existing tutor chat and evaluation functionality remain 100% operational.
