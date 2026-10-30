# 📚 BookRAG Reliability Lab & Book Tutor

**BookRAG Reliability Lab** is a dual-portal platform built for production-grade Retrieval-Augmented Generation (RAG) over PDF textbooks and documents. It unites a reader-facing **Book Tutor** for conversational question answering with citation tracing, and a password-protected **Admin Reliability Lab** for deep RAG telemetry, retrieval diagnostics, citation auditing, and failure analysis.

$$\text{PDF Upload} \rightarrow \text{PyMuPDF Parsing} \rightarrow \text{Chunking \& Quality Check} \rightarrow \text{Dense + Sparse Indexing} \rightarrow \text{Hybrid RRF Retrieval} \rightarrow \text{CrossEncoder Rerank} \rightarrow \text{LLM Synthesis} \rightarrow \text{Citation Verification}$$

---

## 🌟 Key Architecture & Data Flow

```
                                [ PDF Book Upload ]
                                         │
                        (Ingestion Monitor & Diagnostics)
                ┌────────────────────────┼────────────────────────┐
                ▼                        ▼                        ▼
      [Heading Hierarchy]       [Block Classification]   [Tables & Figures]
      (Chapter > Sec > Sub)     (Text, Code, Lists)      (Markdown + Captions)
                                         │
                             (Chunk Quality Inspector)
                           [Sentence-Safe Text Splitter]
                          (Size / Content Heuristic Audit)
                                         │
                                [Indexing Pipeline]
                ┌────────────────────────┴────────────────────────┐
                ▼                                                 ▼
      [Dense Vector Store]                             [Sparse BM25 Index]
      - Local: ChromaDB (data/chroma)                  - Local: BM25 pickle (data/bm25)
      - Cloud: Supabase pgvector                       - Cloud: Supabase JSON store
      (SentenceTransformers / SIE)                     (Tokenized Okapi BM25)
                │                                                 │
                └────────────────────────┬────────────────────────┘
                                         ▼
                            [Dual-Portal Applications]
                ┌────────────────────────┴────────────────────────┐
                ▼                                                 ▼
        [ Reader Portal ]                                 [ Admin Lab ]
     - Book Tutor (QA + Citations)               - Observability & Metrics
     - User PDF Book Upload                      - Chunk & PDF Page Inspector
     - Reader Support & Feedback                 - Retrieval Playground (RRF)
                                                 - Execution Trace Waterfall
                                                 - Citation Auditor
                                                 - Evaluation Lab (Recall/MRR)
                                                 - Root-Cause Assistant
                                                 - Chunking Experiments
                                                 - Settings & Security
```

### 1. Ingestion (Phase 1)
- **Hierarchical Layout Parsing:** Uses `PyMuPDF` (`fitz`) and `pymupdf4llm` to preserve `chapter > section > subsection` heading structures across pages.
- **Content Type Classification:** Discriminates between `text`, `table`, `image`, `list`, and `code`.
- **Atomic Table Processing:** Detects tables, converts them into GitHub-flavored Markdown, keeps each table as a single intact chunk (never split), and attaches a 1-sentence summary.
- **Vision Image Processing:** Extracts images to `data/images/`, computes descriptive captions with a Vision LLM, and indexes the caption with the local image path in metadata.
- **Sentence-Safe Chunking:** Recursively splits text within sections using `RecursiveCharacterTextSplitter(chunk_size=800, overlap=150)` with `keep_separator=True` and prepends the heading hierarchy to every chunk.
- **Keyword & Term Extraction:** Identifies bold spans, defined terms (`is defined as`, `refers to`), and glossary patterns.
- **Parent Store:** Persists full parent sections separately in `data/parent_store/` for context expansion.
- **Diagnostic Validation:** Validates chunk counts, chunk types, empty chunks, and displays 5 random samples.

## 🚀 Application Modules & Portals

The application implements a secure role-separated interface with two dedicated workspaces:

### 1. 🎓 Reader Portal (Default Workspace)
Designed for end users and students learning from ingested literature:
- **Book Tutor:** Conversational knowledge assistant that answers queries with strict ground-truth citations in `(Chapter, Page)` format. Supports multi-book selection and query rewriting with conversation history.
- **Upload Book:** Reader-facing PDF uploader with automated parsing, table summarization, image captioning, and background vector/BM25 indexing.
- **Support & Reports:** Direct reader feedback channel allowing users to report hallucinations, missing pages, or indexing errors directly to administrators.

### 2. 🛡️ Admin Reliability Lab (Password-Protected)
Engineered for ML engineers and RAG operators to diagnose and trace pipeline behavior:
- **Overview:** Real-time production metrics: indexed chunks, total books, query latency percentiles, error rates, and active user tickets.
- **Ingestion Monitor:** Stage-by-stage tracking (parse, chunk, embed, index), duration timers, and heuristic detection of unreadable or low-quality scanned pages.
- **Chunk Inspector:** Two-panel visual inspector displaying raw PDF page rasters side-by-side with extracted markdown, chunk bounding metadata, and quality heuristic warnings.
- **Retrieval Playground:** Pre-generation query debugger: compare isolated Dense Vector Search, Sparse BM25, Reciprocal Rank Fusion (RRF), and CrossEncoder rerankers with score diffs.
- **Trace Viewer:** Detailed execution waterfall with Correlation IDs (`trace_id`), per-stage latency breakdowns, prompt inspection with automatic secret masking, and cost tracking.
- **Citation Verification:** Deterministic citation audit engine verifying whether cited `(Chapter, Page)` references actually exist in retrieved chunks.
- **Evaluation Lab:** Repeatable benchmark harness calculating Recall@k, Precision@k, MRR, groundedness, and out-of-domain abstention rates.
- **Root-Cause Assistant:** 10-stage failure classification assistant outputting observed evidence, root causes, and corrective action items.
- **Chunking Experiments:** Isolated sandbox comparing fixed-size, sentence-safe, paragraph, and hierarchical heading chunking strategies.
- **Settings & Security:** Masked API key management, admin credentials, threat boundaries, and live storage backend health monitoring.

### 4. Generation (Phase 4)
- **Expert Tutor Persona:** Answers strictly using provided context in conversational, natural language.
- **Grounded Citations:** Concludes every answer with `(Chapter, Page)` citations.
- **Strict Hallucination Prevention:** Returns `"I couldn't find that in this book."` when context is absent.

## 🏗️ Technology Stack & Default Configuration

| Layer | Component | Default (Out-of-the-Box) | Optional / Configurable |
|---|---|---|---|
| **Frontend** | Streamlit | `streamlit>=1.35.0` (Dark dev-console UI) | Fully custom CSS design tokens |
| **PDF Extraction** | PyMuPDF & PyMuPDF4LLM | Markdown + Block structural tagging | OCR / Image extraction |
| **Embeddings** | SentenceTransformers | `BAAI/bge-small-en-v1.5` (Local, 384-dim) | Superlinked SIE (`EMBEDDING_BACKEND=sie`) |
| **Dense Vector Store** | ChromaDB / pgvector | ChromaDB (`data/chroma`) | Supabase `pgvector` (`STORAGE_BACKEND=supabase`) |
| **Sparse Retrieval** | Okapi BM25 | rank_bm25 (`data/bm25`) | Supabase BM25 JSON store |
| **Neural Reranker** | CrossEncoder | `BAAI/bge-reranker-base` (Local) | Configurable via `RERANKER_MODEL` |
| **LLM Provider** | OpenAI API compatible | `ollama` (`hf.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB:Q4_K_M`) | OpenAI, Groq, or local deterministic fallback |
| **Telemetry & Observability** | File-backed JSON / Supabase | Local JSON (`data/observability/`) | Supabase `observability_records` |

---

## ⚙️ Storage & Database Modes

BookRAG provides seamless dual-backend support managed by `STORAGE_BACKEND`:

1. **Local Mode (`STORAGE_BACKEND=local`):**
   - No external database required.
   - Vector Store: Local ChromaDB files under `data/chroma/`.
   - Sparse Store: Pickled Okapi BM25 indices under `data/bm25/`.
   - Telemetry: JSON files under `data/observability/` (traces, evaluation runs, support tickets, ingestion jobs).
2. **Supabase Cloud Mode (`STORAGE_BACKEND=supabase` or `auto` with valid credentials):**
   - **Vector Store:** PostgreSQL `pgvector` table `book_chunks` (set up via `scripts/supabase_rag_schema.sql`).
   - **BM25 Store:** Supabase JSON table `bm25_indices`.
   - **Observability:** `observability_records` table (set up via `scripts/supabase_schema.sql`) for traces, evaluations, benchmarks, and support tickets.
   - **Graceful Fallback:** If the network or Supabase connection drops, operations fall back to local disk without crashing the app, flagging storage health in the sidebar.

---

## 🛠️ Quickstart & Setup Guide

### 1. Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python:** Python 3.10 or 3.11 recommended (tested on Python 3.11).
- **Git** installed and available in PATH.
- *(Optional)* **Ollama** installed if running local inference without cloud API keys.

---

### 2. Clone the Repository
```powershell
git clone https://github.com/vishnu94879872-svg/Simple_rag.git
git clone <repo-url>
cd Simple_rag
```

---

### 3. Create & Activate Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(If execution policy restricts scripts, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

### 5. Configure Environment (`.env`)

Copy the template file `.env.example` to `.env` in the project root:
```powershell
cp .env.example .env
```

Review and adjust settings in `.env`:

#### Default (Local Ollama / SentenceTransformers):
```env
LLM_PROVIDER=ollama
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=hf.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB:Q4_K_M
EMBEDDING_BACKEND=local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
STORAGE_BACKEND=local
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
```

#### Cloud LLM Alternative (Groq / OpenAI):
```env
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1   # or https://api.openai.com/v1
LLM_MODEL=llama-3.3-70b-versatile            # or gpt-4o-mini
EMBEDDING_BACKEND=local
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
STORAGE_BACKEND=local
```

> [!NOTE]
> If Ollama is not running and no API key is provided, the pipeline automatically falls back to deterministic local rule-based synthesis for table summaries and responses.

---

### 6. Optional: Supabase Setup

To persist vectors, BM25 indices, and telemetry in the cloud:
1. Open your Supabase SQL Editor.
2. Run [`scripts/supabase_schema.sql`](scripts/supabase_schema.sql) (for observability tables).
3. Run [`scripts/supabase_rag_schema.sql`](scripts/supabase_rag_schema.sql) (for pgvector & BM25 tables).
4. Update `.env`:
   ```env
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your_service_role_key_here
   STORAGE_BACKEND=supabase
   ```

---

### 7. Ingest Initial PDF Documents

You can ingest books either through the Web UI or via CLI:

**Option A — Ingestion via Streamlit Dashboard:**
1. Start the app.
2. In the **Reader Portal**, navigate to **Upload Book** (or in Admin Lab, go to **Ingestion Monitor**).
3. Upload any PDF book. The pipeline parses structure, computes embeddings, and updates indices.

**Option B — Ingestion via Command Line:**
```powershell
python ingest.py data/sample_quantum.pdf
```

---

### 8. Launch the Streamlit Web Application

Start the web application using:
```powershell
streamlit run app/streamlit_app.py
```

Open your browser to:
**`http://localhost:8501`**

- **Reader Workspace:** Direct access to **Book Tutor**, **Upload Book**, and **Support**.
- **Admin Lab:** Click **🔐 Admin Access** in the sidebar and enter your credentials (`admin` / `admin123` by default).

---

## 🧪 Testing & Evaluation

### 1. Run Automated Test Suite
```powershell
python -m pytest tests/ -v
```

To run individual module test suites:
```powershell
# Observability, secret masking & failure classifier tests
python -m pytest tests/test_observability.py -v

# Chunking boundary and ingestion robustness tests
python -m pytest tests/test_chunking_failure_fix.py -v

# Indexing & ChromaDB idempotency tests
python -m pytest tests/test_indexing.py -v
```

### 2. Run Retrieval Benchmark (Hit-Rate@5)
```powershell
python eval.py
```
Evaluates retrieval accuracy across 20 quantum computing QA pairs and verifies that Hit-Rate@5 meets the >80% threshold.

Output:
```text
===========================================================================
BookRAG Retrieval Evaluation Benchmark (Hit-Rate@5)
===========================================================================
+-----------------------------------------------------------------------------+
| # | Question                        | Expected | Retrieved Pages | Result   |
|---+---------------------------------+----------+-----------------+----------|
| 1 | What is a qubit?                | Page 1   | [1, 1, 1]       | HIT      |
| 2 | How is a qubit defined...       | Page 1   | [1, 2, 1]       | HIT      |
| ...                                                                         |
| 20| What performance comparison...  | Page 2   | [2, 2, 1]       | HIT      |
+-----------------------------------------------------------------------------+
Final Evaluation Metrics:
Total Benchmark Questions: 20
Successful Hits (in Top-5): 20
Hit-Rate@5 Metric: 100.00%
```

To output raw JSON for CI/CD pipelines:
```bash
python eval.py --json
```

---

## 🔒 Security & Safe Secret Handling

- **`.env` Protection:** The `.env` file is excluded from git tracking in `.gitignore`. Never commit API keys or service credentials.
- **Trace Redaction:** The `PipelineTracer` uses regex filters (`mask_secrets`) to sanitize all OpenAI, Groq, HuggingFace, Supabase keys, and JWT tokens before writing traces to storage.
- **Role Isolation:** Admin diagnostic views and system logs require authentication; reader users cannot inspect underlying prompt templates or database configurations.
- **HTML Escaping:** All user-supplied inputs and extracted PDF contents rendered dynamically in Streamlit views are sanitized using `html.escape` to prevent XSS.

---

## 📂 Project Structure

```
Simple_rag/
├── app/
│   ├── streamlit_app.py        # Application entrypoint & role-based router
│   ├── ui/                     # Design tokens, CSS injection, layout components
│   └── views/                  # Reader views (Chat, Upload, Support) & Admin views
├── config.py                   # Central Pydantic application settings & directory setup
├── data/                       # Local data storage (ChromaDB, BM25 indices, traces, images)
├── generation/                 # Prompt templates, QA chain, and AnswerGenerator
├── ingest/                     # PDF parser, block classifier, sentence-safe chunker
├── observability/              # Tracer, citation verifier, failure classifier, storage adapters
├── retrieval/                  # Embeddings manager, Chroma store, BM25 store, CrossEncoder reranker
├── scripts/                    # Supabase SQL schemas and benchmark scripts
├── tests/                      # Automated test suite (indexing, retrieval, observability)
├── .env.example                # Environment configuration template
├── eval.py                     # 20-pair QA benchmark harness (Hit-Rate@5)
├── ingest.py                   # CLI tool for PDF ingestion
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

Test coverage includes:
- `tests/test_ingest.py`: PDF layout parsing, table preservation, vision captions, text splitting, keywords, parent store, validation report.
- `tests/test_indexing.py`: SentenceTransformer embeddings, ChromaDB vector store, BM25 store, disk persistence, idempotency.
- `tests/test_retrieval.py`: Query rewriting, Reciprocal Rank Fusion, BGE reranker, context expansion, metadata filtering.
- `tests/test_generation.py`: Tutor system prompt adherence, context formatting, citation generation, missing context detection, end-to-end pipeline.
- `tests/test_eval.py`: Hit-Rate@5 benchmark validation.

---

## 📂 Project Structure

```
Simple_rag/
├── app/
│   ├── __init__.py
│   └── streamlit_app.py         # Streamlit Web Chat UI
├── config.py                     # Centralized settings & path configuration
├── data/
│   ├── bm25/                     # Persisted BM25 inverted index
│   ├── books/                    # Input PDF books
│   ├── chroma/                   # Persistent ChromaDB vector collection
│   ├── images/                   # Extracted book images & diagrams
│   └── parent_store/             # Stored parent sections for context expansion
├── generation/
│   ├── __init__.py
│   ├── generator.py              # LLM answer synthesis & citation extraction
│   ├── prompt_templates.py       # Expert tutor persona & context formatters
│   └── qa_chain.py               # Unified RAG question-answering pipeline
├── ingest/
│   ├── __init__.py
│   ├── keyword_extractor.py      # Bold font, definition & glossary extractor
│   ├── llm_helper.py             # Table summarizer & vision captioner
│   ├── models.py                 # Chunk, Block, ParentSection, ContentType models
│   ├── parent_store.py           # Hierarchical parent section store
│   ├── pdf_parser.py             # PyMuPDF page-by-page layout & block parser
│   ├── pipeline.py               # Ingestion pipeline coordinator
│   ├── text_splitter.py          # Sentence-preserving chunk builder
│   └── validator.py              # Chunk statistics & diagnostic report
├── retrieval/
│   ├── __init__.py
│   ├── bm25_store.py             # Disk-persistent BM25 store
│   ├── context_expansion.py      # Context expander for short chunks
│   ├── embeddings.py             # BAAI/bge-small-en-v1.5 manager
│   ├── engine.py                 # Hybrid retrieval engine
│   ├── fusion.py                 # Reciprocal Rank Fusion (RRF)
│   ├── indexer.py                # Idempotent Chroma & BM25 indexer
│   ├── query_rewriter.py         # LLM & heuristic query rewriter
│   ├── reranker.py               # BAAI/bge-reranker-base CrossEncoder
│   └── vector_store.py           # Persistent Chroma vector store
├── tests/
│   ├── __init__.py
│   ├── generate_sample_pdf.py    # Multi-page test PDF generator
│   ├── test_eval.py              # Retrieval benchmark tests
│   ├── test_generation.py        # Generation & persona tests
│   ├── test_indexing.py          # Chroma, BM25, & idempotency tests
│   ├── test_ingest.py            # PDF parsing & chunking tests
│   └── test_retrieval.py         # Hybrid search & reranking tests
├── .env.example                  # Environment configuration template
├── ask.py                        # CLI Question-Answering script
├── eval.py                       # 20-pair QA retrieval evaluation script
├── ingest.py                     # CLI PDF Ingestion script
├── README.md                     # Documentation
└── requirements.txt              # Python package dependencies
```
