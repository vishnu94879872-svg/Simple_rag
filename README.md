# 🛡️ BookRAG Reliability Lab — Advanced RAG Engineering Platform

**BookRAG Reliability Lab** is a production-oriented RAG debugging, evaluation, and observability platform engineered to diagnose, evaluate, and trace the entire RAG pipeline from PDF ingestion to answer synthesis:

$$\text{PDF Ingestion} \rightarrow \text{Parsing} \rightarrow \text{Chunking} \rightarrow \text{Embedding} \rightarrow \text{Hybrid Retrieval} \rightarrow \text{Reranking} \rightarrow \text{Prompt Construction} \rightarrow \text{LLM Synthesis} \rightarrow \text{Citation Verification} \rightarrow \text{Evaluation}$$

---

## 🌟 Key Architecture & Features

```
                                [ PDF Book Upload ]
                                         │
                        (Module B: Ingestion Diagnostics)
                ┌────────────────────────┼────────────────────────┐
                ▼                        ▼                        ▼
      [Heading Hierarchy]       [Block Classification]   [Tables & Figures]
      (Chapter > Sec > Sub)     (Text, Code, Lists)      (Markdown + Captions)
                                         │
                         (Module C: Chunk Inspector)
                         [Sentence-Safe Text Splitter]
                         (Quality Heuristic Auditing)
                                         │
                                [Indexing Pipeline]
                ┌────────────────────────┴────────────────────────┐
                ▼                                                 ▼
      [Dense ChromaDB Vectors]                         [Sparse BM25 Index]
      (BAAI/bge-small-en-v1.5)                         (Tokenized Okapi)
                │                                                 │
                └────────────────────────┬────────────────────────┘
                                         ▼
                      (Module D: Retrieval Playground)
                  ┌──────────────────────┼──────────────────────┐
                  ▼                      ▼                      ▼
           [Vector Search]        [BM25 Search]           [Hybrid RRF]
                                         │
                                [Neural CrossEncoder]
                              (BAAI/bge-reranker-base)
                                         │
                          [Context Expansion (ParentStore)]
                                         │
                         (Module E: Trace Waterfall)
                        [Prompt Assembly & Redaction]
                                         │
                         (Module F: Citation Auditor)
                   [Tutor Generation & Claim Verification]
                                         │
                ┌────────────────────────┴────────────────────────┐
                ▼                                                 ▼
   (Module G: Evaluation Lab)                        (Module H: Root-Cause)
  [Recall@k, MRR, Grounding]                        [10-Stage Diagnostic]
```

---

## 🚀 Application Modules

| Module | Name | Purpose |
|---|---|---|
| **Module A** | **RAG Observability Dashboard** | Real-time production metrics: indexed chunks, pages, failure rates, P95 latencies, and error summaries. |
| **Module B** | **Document Ingestion Diagnostics** | Stage tracking (parse, chunk, embed, index), duration timers, and heuristic detection of unreadable/scanned pages. |
| **Module C** | **PDF & Chunk Inspector** | Two-panel inspector: original PDF raster preview alongside extracted text, chunk quality flags, and metadata. |
| **Module D** | **Retrieval Playground** | Pre-generation query debugging: test vector search, BM25, hybrid RRF, and neural rerankers with candidate diffs. |
| **Module E** | **Prompt & LLM Trace Viewer** | Waterfall execution tracer with Correlation IDs (`trace_id`), per-stage latency, secret masking, and cost estimation. |
| **Module F** | **Citation & Answer Verification** | Deterministic citation auditing verifying whether cited `(Chapter, Page)` references actually exist in retrieved context. |
| **Module G** | **RAG Evaluation Lab** | Repeatable testing benchmark computing Recall@k, Precision@k, MRR, groundedness, and CSV dataset import/export. |
| **Module H** | **Root-Cause Assistant** | 10-stage failure classification engine outputting observed evidence, likely causes, recommended fixes, and confirmation tests. |
| **Module I** | **Chunking Experiment Workbench** | Isolated, non-destructive sandboxes comparing fixed, sentence, paragraph, and heading chunking strategies. |
| **Settings** | **Security & Threat Model** | Masked API key management, prompt injection boundaries, and safety controls. |

---

## 🛠️ Quickstart & Setup

### 1. Installation
```bash
# Clone the repository
cd Simple_rag

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
```bash
cp .env.example .env
```
Edit `.env` to configure your LLM provider (`openai`, `ollama`, or offline mode):
```env
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
```
*(Note: If `LLM_API_KEY` is omitted, BookRAG automatically runs in deterministic offline fallback mode so all tests and retrieval workflows operate reliably without external API costs).*

### 3. Launching the Web Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser to `http://localhost:8501` to access the dark-mode developer dashboard.

### 4. Running the Evaluation Suite
```bash
# Run full automated test suite (25+ tests)
python -m pytest tests/ -v

# Run observability module tests specifically
python -m pytest tests/test_observability.py -v

# Run command-line Hit-Rate@5 benchmark
python eval.py
```

---

## 🧪 Benchmark Dataset & Reliability Verification

The platform ships with a pre-configured benchmark dataset (`quantum_bench`) containing 22 evaluation questions:
- **20 Domain-Specific QA Pairs:** Covering quantum mechanics definitions, Hadamard gate code, Grover search, and Shor's algorithm complexity.
- **2 Out-of-Domain Abstention Probes:** Verifying that the system appropriately abstains with `"I couldn't find that in this book"` when queries exceed document context.

Automated metrics recorded in the Evaluation Lab:
- **Retrieval Recall@k & MRR:** Deterministically calculated against ground-truth source pages.
- **Citation Accuracy:** Audited via `CitationVerifier`.
- **Groundedness Score:** Semantic entity overlap between generated text and retrieved evidence.
- **Abstention Quality:** Validated refusal on unanswerable inputs.
